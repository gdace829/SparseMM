export BUDGET=256
export RATIO=0.1
export METHOD="sparsemm"

CUDA_VISIBLE_DEVICES=0 python3 ./speed_and_memory.py
