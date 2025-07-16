import torch
print(torch.cuda.is_available())  # True 表示有可用的 CUDA 设备
print(torch.cuda.device_count())  # 可用 GPU 数量
print(torch.cuda.current_device())  # 当前默认 GPU 编号
print(torch.cuda.get_device_name(0)) 