import torch
from model import Model


## T2V
model = Model(device = "cuda", dtype = torch.float16)

# prompt = "A horse galloping on a street"
# prompt = "A stone falls into a still lake"
# prompt = "A basketball free falls in the air"
# prompt = "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement"
prompt = "A match is being lowered into a glass of water"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
model.process_text2video(prompt, fps = fps, path = out_path, **params)