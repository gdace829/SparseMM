from importlib.metadata import version
import transformers# 是不是用的一个transformers

from sparsemm.llama_model import llama_flash_attn2_forward_AdaKV, llama_flash_attn2_forward_PyramidKV, llama_flash_attn2_forward_SnapKV, \
                                 llama_flash_attn2_forward_SparseMM, llama_flash_attn2_forward_Mask
from sparsemm.llama_model import prepare_inputs_for_generation_llama_new, adaptive_LlamaModel_forward

from sparsemm.mistral_model import mistral_flash_attn2_forward_AdaKV,  mistral_flash_attn2_forward_PyramidKV, mistral_flash_attn2_forward_SnapKV, \
                                   mistral_flash_attn2_forward_SparseMM, mistral_flash_attn2_forward_Mask
from sparsemm.mistral_model import prepare_inputs_for_generation_mistral_new, adaptive_MistralModel_forward


from sparsemm.qwen_model import qwen_flash_attn2_forward_AdaKV, qwen_flash_attn2_forward_PyramidKV, qwen_flash_attn2_forward_SnapKV, \
                                qwen_flash_attn2_forward_SparseMM, qwen_flash_attn2_forward_Mask
from sparsemm.qwen_model import prepare_inputs_for_generation_qwen, adakv_qwen_forward
# 这段代码的作用是动态修改（猴子补丁/Monkey Patch）
# transformers库中Llama相关模型的forward方法，
# 以便用自定义的稀疏推理实现（SparseMM）。
def replace_llama(method):
# 这里主要改的是语言模型
    if method == "snapkv":
        # 有些不用改llamaforward
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
    
    elif method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        # 主要是改flash attention
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_AdaKV
    # 如果传入的 method 是 "sparsemm"，就执行下面的代码。
    elif method == "sparsemm":
        # 把 transformers 库里 LlamaModel 的 forward 方法替换成你自定义的
        # adaptive_LlamaModel_forward。
        # 这样以后所有 LlamaModel 的前向推理都会走你的自定义逻辑。
        print("Using SparseMM!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        # 同理，把 LlamaFlashAttention2 的 forward 方法替换成你自定义的 
        # llama_flash_attn2_forward_SparseMM。
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new



# 不同底座有不同的monkeypatch
def replace_mistral(method):

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new



def replace_qwen(method):
    if method == 'snapkv':
        print("Using SnapKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_SnapKV

    elif method == 'pyramidkv':
        print("Using PyramidKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_PyramidKV
    
    if method == "adakv":
        print("Using AdaKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen
