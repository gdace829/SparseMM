# SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs

This repository contains PyTorch implementation for SparseMM.

[Project Page]() | [arXiv Paper]()

## Introduce SparseMM
We investigate how MLLMs process visual inputs by analyzing their attention mechanisms and reveal a surprising sparsity phenomenon: only a small subset (approximately less than 5\%) of attention heads in LLMs actively contribute to visual understanding, termed **Visual Heads**. To identify these heads efficiently, we design a training-free framework that quantifies head-level visual relevance through targeted response analysis. 

Building on this discovery, we introduce **SparseMM**, a KV-Cache optimization strategy that allocates asymmetric computation budgets to heads in LLMs based on their visual scores, leveraging **the sparity of visual heads** for accelerating the inference of MLLMs. Compared with prior KV-Cache acceleration methods that ignore the particularity of visual, SparseMM prioritizes stress and retaining visual semantics during decoding.

### Main idea of Visual Head
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/Visual_Head.png" alt="Visual_Head.png" width=80%>
</p>
<div>

### SparseMM for MLLM Acceleration
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/SparseMM.png" alt="SparseMM.png" width=50%>
</p>
<div>


## Main Results
### Results on Multi-modal Benchmarks
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/main_results.png" alt="main_results.png" width=90%>
</p>
<div>


### Efficiency Evaluation for SparseMM
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/efficiency.png" alt="efficiency.png" width=50%>
</p>
<div>


### Visualization of Visual Head
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/viz.png" alt="viz.png" width=80%>
</p>
<div>


## Get Started

### Install
```bash 
git clone xxx
cd SparseMM
conda create -n sparsemm python=3.10 -y
conda activate sparsemm


pip install packaging torch==2.5.1
pip uninstall ninja && pip cache purge && pip install ninja --no-cache-dir
cd csrc && make
cd ..
pip install -e .

pip install flash-attn==2.4.1 --no-build-isolation
pip install qwen-vl-utils

# lmms-eval
cd lmms-eval
pip install -e .
cd ..
```


### Chase Visual Head
```bash 
bash scripts/chase_visual_head/llava.sh
bash scripts/chase_visual_head/qwen.sh
```

### Eval
```bash
bash scripts/eval/llava.sh
bash scripts/eval/mistral.sh
bash scripts/eval/qwen.sh
```

### Viz
```bash
bash scripts/others/viz.sh
```

### Speed and Memory
```bash
bash scripts/others/speed_and_memory.sh
```


## Citation

If you found this repository useful, please consider citing:

``` 
```
