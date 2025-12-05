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

# -----------------------------------
# 2. 비디오 로딩 & frame 샘플링
# -----------------------------------
# 테스트용 샘플 비디오 URL
video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
video_path = "sample_video.mp4"

# 없으면 다운로드
if not os.path.exists(video_path):
    print("Downloading sample video...")
    urllib.request.urlretrieve(video_url, video_path)
    print("Download complete:", video_path)
else:
    print("Sample video already exists:", video_path)
vr = VideoReader(video_path, ctx=cpu())

num_model_frames = 64
num_total_frames = len(vr)

if num_total_frames >= num_model_frames:
    indices = np.linspace(0, num_total_frames - 1, num_model_frames).astype(int)
else:
    indices = np.arange(num_total_frames)

frames = vr.get_batch(indices)        # (T, H, W, 3), uint8
frames_np = frames.asnumpy()

# 3. [T, H, W, 3] → 프레임 리스트 [T 개, 각 [3, H, W]]
clip = []
for i in range(frames_np.shape[0]):
    frame = torch.from_numpy(frames_np[i])      # (H, W, 3)
    frame = frame.permute(2, 0, 1)             # (3, H, W)
    clip.append(frame)

# 4. preprocessor 적용 (clip = list of [3, H, W])
with torch.no_grad():
    processed = processor(clip)    # processor는 내부에서 resize, crop 등 수행

# processor는 [transformed_clip] 형식의 리스트를 반환한다고 가정
if isinstance(processed, list):
    clip_tensor = processed[0]     # (T', 3, H', W')
else:
    clip_tensor = processed        # 이미 tensor인 경우

# 5. 배치 차원 추가 후 encoder 통과
clip_tensor = clip_tensor.unsqueeze(0).to(device)   # (1, T', 3, H', W')

with torch.no_grad():
    token_latent = vjepa2_encoder(clip_tensor)      # 보통 (B, N, D)

print("token_latent shape:", token_latent.shape)

# 6. 비디오 전체 latent (평균 풀링 예시)
video_latent = token_latent.mean(dim=1)             # (B, D)
print("video_latent shape:", video_latent.shape)
