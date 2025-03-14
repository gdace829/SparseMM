from importlib.metadata import version
import transformers

from sparsemm.llama_model import llama_flash_attn2_forward_AdaKV, llama_flash_attn2_forward_PyramidKV, llama_flash_attn2_forward_SnapKV, \
                                 llama_flash_attn2_forward_SparseMM, llama_flash_attn2_forward_Mask
from sparsemm.llama_model import prepare_inputs_for_generation_llama_new, adaptive_LlamaModel_forward

from sparsemm.mistral_model import mistral_flash_attn2_forward_AdaKV,  mistral_flash_attn2_forward_PyramidKV, mistral_flash_attn2_forward_SnapKV, \
                                   mistral_flash_attn2_forward_SparseMM, mistral_flash_attn2_forward_Mask
from sparsemm.mistral_model import prepare_inputs_for_generation_mistral_new, adaptive_MistralModel_forward


from sparsemm.qwen_model import qwen_flash_attn2_forward_AdaKV, qwen_flash_attn2_forward_PyramidKV, qwen_flash_attn2_forward_SnapKV, \
                                qwen_flash_attn2_forward_SparseMM, qwen_flash_attn2_forward_Mask
from sparsemm.qwen_model import prepare_inputs_for_generation_qwen, adakv_qwen_forward

def replace_llama(method):

    if method == "snapkv":
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
    
    elif method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new




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
