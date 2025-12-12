# model.py
import torch
from typing import Optional
from world_aware_frame_pipeline import WorldAwareFramePipeline


class PT2VModel:
    """
    Physical-aware Text-to-Video (PT2V)를 위한 간단한 래퍼.
    - 내부: WorldAwareFramePipeline (단일 프레임 생성)
    - 외부: 여러 프레임 루프를 돌려 비디오 시퀀스를 만든다.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        disable_safety: bool = True,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = torch.Generator(device=self.device)

        safety_checker = None if disable_safety else None  # 필요하면 확장
        requires_safety_checker = False if disable_safety else True

        self.pipe = WorldAwareFramePipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=safety_checker,
            requires_safety_checker=requires_safety_checker,
        ).to(self.device)

    def set_seed(self, seed: int):
        self.generator.manual_seed(seed)

    def generate_frame(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        world_emb: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        프레임 1장을 생성.
        world_emb: (D_w,) 또는 (1, D_w) tensor (V-JEPA W-space 등)
        """
        if seed is not None:
            self.generator.manual_seed(seed)

        # autocast는 optional; GPU float16일 때만 사용
        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_autocast
            else torch.no_grad()
        )

        with ctx:
            out = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=self.generator,
                world_emb=world_emb,
            )

        return out.images[0]  # PIL.Image

    def generate_video(
        self,
        prompt: str,
        num_frames: int = 8,
        fps: int = 4,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        base_seed: int = 0,
        world_emb_seq: Optional[torch.Tensor] = None,
    ):
        """
        여러 프레임을 순차적으로 생성해서 시퀀스를 반환.

        world_emb_seq: (num_frames, D_w) or None
                       - None이면 world 없이 일반 T2I 반복
                       - 나중에 V-JEPA world state 시퀀스로 교체하면 됨
        """
        frames = []

        if world_emb_seq is not None:
            assert (
                world_emb_seq.shape[0] == num_frames
            ), "world_emb_seq 길이는 num_frames와 같아야 합니다."
            world_emb_seq = world_emb_seq.to(self.device, dtype=self.dtype)

        for t in range(num_frames):
            seed = base_seed + t  # 프레임마다 seed 다르게

            world_emb_t = None
            if world_emb_seq is not None:
                world_emb_t = world_emb_seq[t]

            frame = self.generate_frame(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                world_emb=world_emb_t,
                seed=seed,
            )
            frames.append(frame)

        return frames  # [PIL.Image, ...]
