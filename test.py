import torch
import sys
from sparsemm import mistral_model
print(llava.__file__)
print("CUDA 版本:", torch.version.cuda)
print("Python 版本:", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("PyTorch 版本:", torch.__version__)
print("当前可见 CUDA 设备数量:", torch.cuda.device_count())
