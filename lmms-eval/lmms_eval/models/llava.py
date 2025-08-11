import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm
# 本包模块自己索引
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

import sys
# sys.path.append("./visual_head/LLaVA-NeXT")
# python 在导入模块（比如 import llava）时，
# 会在 sys.path 里的所有目录下查找对应的包或模块。
#  pip install -e 安装模块后 可以变改边测试
try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model# 这里调用的应该是视觉头的llava初始化代码
    
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)
# 引入其他的包 这里是lmm评估的模型引入了替换的模块 sjs
try:
    from sparsemm.monkeypatch import replace_llama, replace_mistral
except Exception as e:
    eval_logger.debug("import sparsemm failed")

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"
# sjs
# 加载了这个模型
# 主要的多模态模型，LLaVA 主要的评估包

# @register_model("llava") 
# 的作用是把当前类注册为名为 "llava" 的模型类型，方便后续通过字符串名字动态调用和管理模型。
@register_model("llava")
class Llava(lmms):
    """
    Llava Model
    """
    # 初始化时就会加载对应的llava模型，只不过其中的底座语言模型被修改
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",  # 默认加载的pretrained模型
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # 是否在生成时截断上下文，LLaVA-1.6建议设为False
        customized_config=None,  # 以json结尾的自定义配置
        **kwargs,
    ) -> None:
        super().__init__()
        # 暂时不使用kwargs
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # 根据环境变量METHOD选择KV稀疏方法
        import os

        method = os.getenv('METHOD', None)# 从系统环境变量获取method
        # 根据不同底座选用不同方法
        # 不同底座采用不同的替换方法，替换的是底座的代码
        if 'mistral' not in pretrained:# 根据mistra选择不同的预训练模型
            # llama系列模型的KV稀疏替换
            if method == 'adakv':# 从llama看
                replace_llama("adakv")
            elif method == 'pyramidkv':
                replace_llama("pyramidkv")
            elif method == 'snapkv':
                replace_llama("snapkv")
            elif method == 'sparsemm':
                replace_llama("sparsemm")
            elif method == 'random':
                replace_llama("sparsemm")
            elif method == 'mask' or method == 'mask_random':
                replace_llama("mask")
            else:
                print("Use Full KV")
        else:
            # mistral系列模型的KV稀疏替换
            if method == 'adakv':
                replace_mistral("adakv")
            elif method == 'pyramidkv':
                replace_mistral("pyramidkv")
            elif method == 'snapkv':
                replace_mistral("snapkv")
            elif method == 'sparsemm':
                replace_mistral("sparsemm")
            elif method == 'random':
                replace_mistral("sparsemm")
            elif method == 'mask' or method == 'mask_random':
                replace_mistral("mask")
            else:
                print("Use Full KV")

        # 初始化accelerator，设置超长timeout防止分布式初始化超时
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        # 根据进程数和device_map设置设备
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")# 获得加速id
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # 构建LLaVA模型加载参数
        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:# 自定义配置
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:# 注意力机制实现
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:# 是否使用flash_attention
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        # 加载预训练模型llava模型
        try:
            # 尝试带multimodal参数加载
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
                # pretrained是预训练的模型参数
        except TypeError:
            # 兼容旧版LLaVA（无multimodal参数）
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        # 分布式/单卡/张量并行的设备和rank设置
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], \
                "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # deepspeed需要提前配置
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            # FSDP/DEEPSPEED直接prepare，其他用prepare_model
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # 返回transformers.AutoConfig对象
        return self._config

    @property
    def tokenizer(self):
        # 返回分词器
        return self._tokenizer

    @property
    def model(self):
        # 返回模型本体，如果用Accelerate则unwrap
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # 返回EOT（end of text）token id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # 返回最大长度
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        # 对输入序列进行pad，支持左pad和右pad
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        # 返回每卡batch size
        return self.batch_size_per_gpu

    @property
    def device(self):
        # 返回当前设备
        return self._device

    @property
    def rank(self):
        # 返回当前进程rank
        return self._rank

    @property
    def world_size(self):
        # 返回总进程数
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """
        文本编码为token id列表
        left_truncate_len: 若指定则左截断到该长度
        add_special_tokens: 是否添加特殊token
        """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # 左截断
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        """
        token id解码为文本
        """
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        计算loglikelihood（损失）和贪心生成是否与目标一致
        """
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # 处理目标文本
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            # 处理视觉输入
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            # 如果有图片但prompt中没有image token，则自动加上
            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                三种情况:
                1. 没有图片，不需要加image token
                2. prompt已包含image token，不需要再加
                3. 有图片但prompt没image token，需要加在开头
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # 构造对话模板
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # 把目标答案加到第二轮
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # 上下文部分不计入loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        """
        二维list展平成一维list
        """
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        批量生成文本，支持视觉输入
        """
        res = []

        def _collate(x):
            # 按context长度降序排序，方便batch pad和OOM提前暴露
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # 按generation参数分组，保证同一batch参数一致
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            # 获取视觉输入
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            # 取batch内统一的gen_kwargs
            gen_kwargs = all_gen_kwargs[0]

            # 设置默认until和max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # 如果gen_kwargs中有until则覆盖
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # 处理image_aspect_ratio参数
            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

            # 视觉输入转tensor
            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            question_input = []

            # 构造每个样本的prompt
            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    三种情况:
                    1. 没有图片，不需要加image token
                    2. prompt已包含image token，不需要再加
                    3. 有图片但prompt没image token，需要加在开头
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                # 构造对话模板
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # 配置生成参数默认值
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # 编码输入
            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # 生成
            try:
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            # 结果收集
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        # 恢复原始顺序
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        # 多轮对话生成暂未实现
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")
