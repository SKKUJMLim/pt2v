# test_pt2v.py
import os
import numpy as np
import torch
from model import PT2VModel

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def save_frames_as_gif(frames, fps: int, out_path: str):
    if not HAS_IMAGEIO:
        print("imageio가 설치되어 있지 않아 GIF를 저장할 수 없습니다.")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames_np = [np.array(f) for f in frames]
    imageio.mimsave(out_path, frames_np, duration=1.0 / fps)
    print(f"Saved GIF: {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "runwayml/stable-diffusion-v1-5"
    model = PT2VModel(model_id=model_id, device=device, dtype=dtype)

    prompt = "A basketball free falls in the air"
    num_frames = 8
    fps = 4

    # (옵션) world_emb_seq: 여기서는 일단 랜덤 텐서로 테스트
    # 나중에 V-JEPA W-space 시퀀스를 여기로 넣으면 됨.
    D_w = 768  # 예시 dim (CLIP hidden size에 맞춰두면 편함)
    world_emb_seq = torch.randn(num_frames, D_w)

    # world 없이 그냥 프레임 생성 테스트
    frames_no_world = model.generate_video(
        prompt=prompt,
        num_frames=num_frames,
        fps=fps,
        base_seed=0,
        world_emb_seq=None,   # ❌ world 사용 안 함
    )

    os.makedirs("outputs", exist_ok=True)
    for i, img in enumerate(frames_no_world):
        img.save(f"outputs/tiger_no_world_{i:02d}.png")

    # world_emb_seq를 넣어서 생성 테스트
    # frames_with_world = model.generate_video(
    #     prompt=prompt,
    #     num_frames=num_frames,
    #     fps=fps,
    #     base_seed=42,
    #     world_emb_seq=world_emb_seq,  # ✅ world 사용
    # )
    #
    # for i, img in enumerate(frames_with_world):
    #     img.save(f"outputs/tiger_with_world_{i:02d}.png")
    #
    # # (옵션) GIF로 저장
    # if HAS_IMAGEIO:
    #     save_frames_as_gif(
    #         frames_no_world, fps, "outputs/tiger_no_world.gif"
    #     )
    #     save_frames_as_gif(
    #         frames_with_world, fps, "outputs/tiger_with_world.gif"
    #     )


if __name__ == "__main__":
    main()
