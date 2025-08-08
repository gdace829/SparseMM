ratios=(0.1)
# ratios：只包含 0.1，表示稀疏率等参数
methods=("sparsemm" "fullkv" "adakv" "snapkv" "pyramidkv" "mask" "mask_random")
# 评测的不同方法/算法名。
budgets=(64)
# budgets：只包含 64，表示资源预算参数
mask_ratio=0.1 # only used for "mask" / "mask_random"
# mask_ratio：只在 "mask" 和 "mask_random" 方法下用到。

# 会把所有方法、预算、比例的组合都跑一遍。
for budget in ${budgets[@]}; do
    for ratio in ${ratios[@]}; do
        for method in ${methods[@]}; do
            # 把当前循环的参数写入环境变量，方便后续 Python 脚本读取。
            export METHOD=${method}
            export BUDGET=${budget}
            export RATIO=${ratio}
            export MASK_RATIO=${mask_ratio}
            # 创建保存日志的目录（如果不存在就新建）。
            mkdir -p ./ocrbench_results/llama_results/
            # 指定用哪些 GPU（0-7）。
            export CUDA_VISIBLE_DEVICES=0,1
            # 用 accelerate 启动 8 个进程，
            # 运行 lmms_eval 这个 Python 模块，评测 LLaVA 模型在 ocrbench 任务上的表现。

            # 评测参数（如模型路径、对话模板、batch size、日志等）都已指定。
            # 一开始调用这个
            # 指定了llava模型
            python3 -m accelerate.commands.launch \
                --num_processes=2 \
                --main_process_port 54323\
                -m lmms_eval \
                --model llava \
                --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b",conv_template=vicuna_v1 \
                --tasks ocrbench \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix llava_v1.6_mix \
                --output_path ./logs/ \
                --gen_kwargs temperature=0 \
                --verbosity=DEBUG 2>&1 | tee ./ocrbench_results/llama_results/ocrbench_${method}_${budget}_${ratio}.log
        done
    done
done
# tee：一个命令，作用是：把输入的内容同时输出到终端和文件