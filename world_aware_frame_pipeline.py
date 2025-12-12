# world_aware_frame_pipeline.py
import torch
from torch import nn
from dataclasses import dataclass
from typing import List, Optional, Union, Callable

import numpy as np
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers.utils import BaseOutput, logging

logger = logging.get_logger(__name__)


@dataclass
class WorldAwarePipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


class WorldAwareFramePipeline(StableDiffusionPipeline):
    """
    Stable Diffusion 기반 단일 프레임 생성 파이프라인.
    world_emb (예: V-JEPA W-space 벡터)를 text embedding에 world token으로 concat.

    - 입력: prompt, (option) world_emb
    - 출력: 1장 또는 N장 이미지
    """

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

        # world_emb -> text_hidden_dim 으로 projection
        self.world_proj: Optional[nn.Linear] = None

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        world_emb: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[WorldAwarePipelineOutput, tuple]:
        """
        world_emb: (D_w,) or (B, D_w)
                   프레임 하나 생성할 때 world state 조건.
        """

        # -----------------------------
        # 0) 기본 체크 및 설정
        # -----------------------------
        self.check_inputs(prompt, height, width, callback_steps)

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # -----------------------------
        # 1) 텍스트 인코딩
        # -----------------------------
        text_embeddings = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        # text_embeddings: (N, L, D_text), N = batch * cfg_factor
        N, L, D_text = text_embeddings.shape

        # -----------------------------
        # 2) world_emb를 world token으로 concat
        # -----------------------------
        if world_emb is not None:
            if world_emb.dim() == 1:
                # (D_w,) -> (1, D_w)
                world_emb = world_emb.unsqueeze(0)

            B, D_world = world_emb.shape  # B: batch_size 또는 1

            # 배치 맞추기
            if B == 1 and batch_size > 1:
                world_emb = world_emb.repeat(batch_size, 1)

            cfg_factor = 2 if do_classifier_free_guidance else 1
            if world_emb.shape[0] != batch_size * cfg_factor:
                world_emb = world_emb.repeat(cfg_factor, 1)

            world_emb = world_emb.to(device=device, dtype=text_embeddings.dtype)

            # projection layer lazy init
            if (
                self.world_proj is None
                or self.world_proj.in_features != D_world
                or self.world_proj.out_features != D_text
            ):
                self.world_proj = nn.Linear(D_world, D_text).to(device)

            # (N, D_text) -> (N, 1, D_text)
            world_tokens = self.world_proj(world_emb).unsqueeze(1)

            # token 차원으로 concat: [world_token, text_tokens...]
            text_embeddings = torch.cat([world_tokens, text_embeddings], dim=1)
            # now: (N, L+1, D_text)

        # -----------------------------
        # 3) 타임스텝 준비
        # -----------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # -----------------------------
        # 4) 초기 latents
        # -----------------------------
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=text_embeddings.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # -----------------------------
        # 5) denoising loop
        # -----------------------------
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 모델 입력 스케일링
                latent_model_input = latents
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2, dim=0)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # noise 예측
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

                # CFG
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # scheduler step
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # callback & progress
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # -----------------------------
        # 6) VAE decode
        # -----------------------------
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        # safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, text_embeddings.dtype
        )

        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            images = [Image.fromarray((img * 255).astype("uint8")) for img in image]
        else:
            images = image

        if not return_dict:
            return images, has_nsfw_concept

        return WorldAwarePipelineOutput(
            images=images, nsfw_content_detected=has_nsfw_concept
        )
