import torch
from model import Model

import torch
import numpy as np
import os
import urllib.request

# (선택) decord로 비디오 읽기 – 안 쓰고 직접 frame 텐서를 만들고 싶으면 이 부분만 바꿔도 됨
from decord import VideoReader, cpu


# -----------------------------------
# 1. device 설정 & 모델 / preprocessor 로드
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# torch.hub에서 preprocessor / encoder 로드
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
loaded  = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')

if isinstance(loaded, tuple):
    vjepa2_encoder = loaded[0]    # encoder만 사용
else:
    vjepa2_encoder = loaded

vjepa2_encoder = vjepa2_encoder.to(device).eval()


# -------------------------
# 2) T2V로 비디오 생성
# -------------------------

model = Model(device = "cuda", dtype = torch.float16)

# prompt = "A horse galloping on a street"
prompt = "A basketball free falls in the air"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
model.process_text2video(prompt, fps = fps, path = out_path, **params)



# -------------------------
# 3) 비디오 → frame tensor로 변환 (decord)
# -------------------------
vr = VideoReader(out_path, ctx=cpu())
num_frames = len(vr)

indices = list(range(num_frames))
frames = vr.get_batch(indices).asnumpy()   # (T, H, W, 3), uint8

# -------------------------
# 4) V-JEPA2용 텐서로 변환
#    (T, H, W, 3) -> (1, T, 3, H, W)
# -------------------------
frames_t = torch.from_numpy(frames)              # (T, H, W, 3)
frames_t = frames_t.permute(0, 3, 1, 2)         # (T, 3, H, W)

# V-JEPA2는 통상 64프레임 기준 예시가 많으니, 맞춰주고 싶으면 샘플링/패딩:
T = frames_t.shape[0]
target_T = 64

if T >= target_T:
    frames_t = frames_t[:target_T]
else:
    pad = target_T - T
    frames_t = torch.cat([frames_t, frames_t[-1:].repeat(pad, 1, 1, 1)], dim=0)

# 배치 차원 추가: (1, T, 3, H, W)
video_tensor = frames_t.unsqueeze(0).float()      # (1, T, 3, H, W)

# -------------------------
# 5) processor + encoder
# -------------------------
# 5-1) processor 호출
inputs = processor(video_tensor)   # 공식 예시와 동일하게 사용

# 타입에 따라 정리
if isinstance(inputs, list):
    # 혹시라도 list로 감싸서 주면 첫 번째 요소만 사용
    inputs = inputs[0]

if isinstance(inputs, dict):
    inputs = {k: v.to(device) if torch.is_tensor(v) else v
              for k, v in inputs.items()}
elif torch.is_tensor(inputs):
    inputs = inputs.to(device)
else:
    raise TypeError(f"Unexpected processor output type: {type(inputs)}")

# 5-2) encoder로 피처 추출
with torch.no_grad():
    feats = vjepa2_encoder(**inputs) if isinstance(inputs, dict) else vjepa2_encoder(inputs)

if isinstance(feats, dict):
    for k, v in feats.items():
        if torch.is_tensor(v):
            print("feat:", k, v.shape)
else:
    print("feat tensor shape:", feats.shape)