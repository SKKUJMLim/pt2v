import torch
import numpy as np
import os
import urllib.request
from decord import VideoReader, cpu
from transformers import AutoVideoProcessor, AutoModel

hf_repo = "facebook/vjepa2-vitg-fpc64-256"

model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)