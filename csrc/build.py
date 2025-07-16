# 该脚本用于构建和安装一个基于 PyTorch 的 CUDA 扩展模块，主要实现了 CUDA/C++ 代码与 Python 的绑定，便于在 Python 中调用高性能的 CUDA 代码。

import subprocess
import os
from packaging.version import parse, Version
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

# 包名，安装后可用 pip uninstall tiny_pkg 卸载
PACKAGE_NAME = "tiny_pkg"

# 构建扩展模块的相关参数
ext_modules = []
generator_flag = []  # 这里可以添加生成器相关的编译参数
cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90,code=sm_90")  # 指定支持的 CUDA 架构（如 Ada, Hopper）

# 获取 CUDA 版本的辅助函数
def get_cuda_bare_metal_version(cuda_dir):
    # 通过调用 nvcc -V 获取 CUDA 版本信息
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version

# 获取当前脚本所在目录（绝对路径），用于 include_dirs
this_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 CUDA 扩展模块
ext_modules.append(
    CUDAExtension(
        name="tiny_api_cuda",  # Python 中 import 的模块名
        sources=[
            "csrc/cuda_api.cu",  # CUDA 源文件
        ],
        extra_compile_args={
            # C++ 编译参数
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # nvcc 编译参数
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "--use_fast_math",
                "-lineinfo",
                "--ptxas-options=-v",
                "--ptxas-options=-O2",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ] + generator_flag + cc_flag,
        },
        include_dirs=[
            Path(this_dir) / "csrc",    # 头文件目录
            Path(this_dir) / "include", # 头文件目录
            # 可以添加更多 include 路径
        ],
    )
)

# 调用 setuptools 的 setup 函数进行打包和安装
setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "tiny_pkg.egg-info",
        )
    ),
    description="Tiny cuda and c api binding for pytorch.",  # 包描述
    ext_modules=ext_modules,  # 扩展模块
    cmdclass={ "build_ext": BuildExtension},  # 构建扩展的命令
    python_requires=">=3.7",  # Python 版本要求
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],  # 依赖包
)

# 总结：
# 该脚本主要用于编译和安装一个自定义的 CUDA 扩展模块（tiny_api_cuda），
# 通过 setup.py 或 makefile 调用时，会自动编译 csrc/cuda_api.cu 并生成可供 Python 调用的扩展。
