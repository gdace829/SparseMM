import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    _flash_attention_forward
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
)
from sparsemm.sparsemm_utils import init_snapkv, init_pyramidkv, init_adakv, init_sparsemm, init_mask
import math
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from sparsemm.sparsemm_utils import DynamicCacheSplitHeadFlatten

logger = logging.get_logger(__name__)
# 使用不同的kv cache
def llama_flash_attn2_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    init_snapkv(self)

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    if past_key_value is not None:
        # NOTE: decoding update
        if q_len == 1:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_flash_attn2_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    init_pyramidkv(self)

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    if past_key_value is not None:
        # NOTE: decoding update
        if q_len == 1:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_flash_attn2_forward_AdaKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    init_adakv(self)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += cache_position[0]

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    dropout_rate = self.attention_dropout if self.training else 0.0

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
    


    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    
    is_prefill = q_len != 1

    if is_prefill:
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    else:
        cache_kwargs["head_lens"] = self.kv_cluster.head_lens
        cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        # NOTE: update meta data
        self.kv_cluster.klen_sum += self.num_heads
        self.kv_cluster.max_seqlen_k += 1
        self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
        self.kv_cluster.head_lens += 1

        query_states = query_states.view(-1, self.num_key_value_groups, self.head_dim)
        key_states = key_states.view(-1,1,self.head_dim)
        value_states = value_states.view(-1,1,self.head_dim)

        cu_seqlens_q = self.kv_cluster.cu_qlen
        cu_seqlens_k = self.kv_cluster.cu_klen
        max_seqlen_q = 1
        max_seqlen_k = self.kv_cluster.max_seqlen_k

        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                             cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        #  TODO: support batch size > 1
        assert bsz == 1
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

# 该函数 `llama_flash_attn2_forward_SparseMM` 
# 是 Llama 模型中用于稀疏注意力（SparseMM）前向传播的实现，
# 主要用于高效地处理大规模序列的自注意力计算，支持 KV cache 机制和滑动窗口等特性，
# 适配 FlashAttention2 的高性能实现。
# 初始化sparse mm0 sjs
def llama_flash_attn2_forward_SparseMM(
    self,
    hidden_states: torch.Tensor,                       # [B, Tq, D]：batch、query长度、隐层维度(D=H*Dh)
    attention_mask: Optional[torch.LongTensor] = None, # 广播到注意力核期望的形状；含因果/Pad遮罩
    position_ids: Optional[torch.LongTensor] = None,   # [B, Tq] 或 [1, Tq]：位置索引(旧路径)
    past_key_value: Optional[Cache] = None,            # 动态KV缓存(每层)：需支持 get_seq_length/update/索引
    output_attentions: bool = False,                   # 本实现会强制 False，节省显存/算力
    use_cache: bool = False,                           # 是否使用/更新缓存
    cache_position: Optional[torch.LongTensor] = None, # [Tq]：绝对位置(含历史偏移)，与 RoPE/Cache 对齐
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos, sin)，v4.46+ 推荐外部传入
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 1) Cache 类型检查：FlashAttention-2 不支持 StaticCache（它假设动态长度/布局）
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    # 2) 初始化稀疏注意力相关结构（只初始化一次）
    #    这里通常会创建 self.kv_cluster：按头/分组(GQA)管理KV缓存、滑窗与稀疏压缩参数等
    init_sparsemm(self)  # 备注：你的实现是“基于头/分组”来分配与压缩 K/V 的
    output_attentions = False  # 强制不输出注意力权重（减少显存/计算开销）

    # 3) 基本尺寸：batch 与 query长度
    bsz, q_len, _ = hidden_states.size()

    # 4) 线性变换：从隐层得到 Q/K/V（仍是 [B, Tq, D]）
    query_states = self.q_proj(hidden_states)   # [B, Tq, D]
    key_states   = self.k_proj(hidden_states)   # [B, Tq, D]
    value_states = self.v_proj(hidden_states)   # [B, Tq, D]

    # 5) 形状整理(预-RoPE)：为多头注意力/GQA 做准备
    #    Q: [B, Tq, H, Dh] -> [B, H, Tq, Dh]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    #    K/V: [B, Tq, HKV, Dh] -> [B, HKV, Tq, Dh]（HKV可能小于H，GQA下多个Q头共享一个KV头）
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 6) 当前K序列长度（考虑历史缓存）
    kv_seq_len = key_states.shape[-2]  # 这里是“本步”的 Tq
    if past_key_value is not None:
        kv_seq_len += cache_position[0]  # 加上历史长度偏移(绝对位置)，与Cache/Mask对齐

    # 7) RoPE（旋转位置编码）：4.46+ 推荐外部传 cos/sin；未传则用 position_ids 计算(将弃用)
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        # 旧路径：用 position_ids 计算 cos/sin
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    # 对 Q/K 施加 RoPE（V 通常不加），保留相对/绝对位置信息
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # 注意力 dropout：训练时用 config 指定，推理为 0
    dropout_rate = self.attention_dropout if self.training else 0.0

    # 8) KVCache 相关逻辑：滑窗、元信息等（仅当有 Cache）当有kvcache的时候
    if past_key_value is not None:
        # 下游 update() 需要的元信息（RoPE参数+绝对位置）
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # 判断缓存是否已有内容（本层）
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0

        # 若启用 sliding_window 且当前总长度超窗且已有缓存 → 裁剪历史 K/V 只保留最近窗
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window  # 负数，保留后 (W-1) 个历史
            # 取出对应层的 past K/V（形状常见： [B, HKV, L_hist, Dh]）
            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            # 仅保留滑窗内的历史（-W+1: 末尾）
            past_key   = past_key[:,   :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            # 断言裁剪后历史长度为 W-1（当前步再加1 则总可见为 W）
            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            # 同步裁剪 attention_mask（并确保当前 token 可见：末位补1）
            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]  # 对齐到窗
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

    # 9) dtype 统一（防止 FP32 上溢/慢；统一到投影权重或autocast dtype）
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()          # e.g., bf16/fp16
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype     # 预量化前的目标dtype
        else:
            target_dtype = self.q_proj.weight.dtype                # 回铸到权重dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        # 对 Q/K/V 统一回铸，避免核里混精度影响性能/数值
        query_states = query_states.to(target_dtype)
        key_states   = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # 10) 判定阶段：prefill(提示词/批量) vs decode(自回归/单步)
    is_prefill = q_len != 1  # True=预填充；False=解码(单token)
    if is_prefill:# 不同传播途径加入的kvcache不一样
        # ---------- prefill 路径：批量Tq>1 ----------
        # 稀疏压缩 K/V（SparseMM 核心）：按头/滑窗/打分/比例等筛选，得到压缩后的 K/V
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(
            key_states, query_states, value_states
        )
        # 将压缩后的kv并带有该层信息 写入缓存（本层），past_key_value记录了不同层的缓存，并携带 RoPE/位置等元信息
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

        # FlashAttention(定长核) 的输入布局期望：[B, T, H, Dh] / [B, T, HKV, Dh]，需要把轴转回去
        query_states = query_states.transpose(1, 2)  # [B, H, Tq, Dh] -> [B, Tq, H, Dh]
        key_states   = key_states.transpose(1, 2)    # [B, HKV, Tq, Dh] -> [B, Tq, HKV, Dh]
        value_states = value_states.transpose(1, 2)  # [B, Tq, HKV, Dh]

        # 调用 FA2 定长前向（可带 sliding_window / 顶左掩码 / 因果）
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,                              # 旧接口；新版本用外部 cos/sin
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,  # 不同实现的遮罩细节
            is_causal=self.is_causal,
        )
        # 输出还原到 [B, Tq, D]
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

    else:
        # ---------- decode 路径：单步 Tq=1 ----------
        # 写缓存：把新步的 K/V 追加到 Cache，并附带 varlen 需要的元数据
        cache_kwargs["head_lens"] = self.kv_cluster.head_lens  # 每头有效K长度
        cache_kwargs["cu_klen"]   = self.kv_cluster.cu_klen    # K长度前缀和(紧凑布局)
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

        # 更新 kv_cluster 的步进统计（供 varlen 边界与最大长度使用）
        self.kv_cluster.klen_sum    += self.num_heads           # 统计累计
        self.kv_cluster.max_seqlen_k += 1                       # 每步K最大长度+1
        self.kv_cluster.cu_klen     += self.kv_cluster.cu_offset# 前缀和游标推进
        self.kv_cluster.head_lens   += 1                        # 每头长度+1

        # 变形为 varlen 接口需要的紧凑布局
        # G = num_key_value_groups = H // HKV（GQA分组数）
        query_states = query_states.view(-1, self.num_key_value_groups, self.head_dim)  # [-1, G, Dh]
        key_states   = key_states.view(-1, 1, self.head_dim)                            # [-1, 1, Dh]
        value_states = value_states.view(-1, 1, self.head_dim)                          # [-1, 1, Dh]

        cu_seqlens_q = self.kv_cluster.cu_qlen        # Q 的前缀和（decode下每步max_seqlen_q=1）
        cu_seqlens_k = self.kv_cluster.cu_klen        # K 的前缀和（历史+本步）
        max_seqlen_q = 1
        max_seqlen_k = self.kv_cluster.max_seqlen_k   # 当前最大K长度（含历史）

        # 变长 FA2 内核：一次kernel处理不同头/不同有效长度的注意力
        attn_output = flash_attn_varlen_func(
            query_states, key_states, value_states,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            causal=True
        )
        # 目前 decode 路径仅支持单batch
        assert bsz == 1
        # 输出还原到 [B, Tq, D]
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
        # 也是把head和dim拼接到一块
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

    # 11) 输出投影：多头拼回隐维 D 对于head*dim ->dim
    attn_output = self.o_proj(attn_output)  # [B, Tq, D]

    # 12) 不返回注意力权重（强制）
    if not output_attentions:
        attn_weights = None

    # 13) 返回：注意力输出、权重(None)、以及(更新后的)缓存
    # 备注：你的注解中返回类型注解是 Optional[Tuple[torch.Tensor]]，
    #       实际这里返回的是 Cache 对象，严格说应改为 Optional[Cache] 更贴切。
    return attn_output, attn_weights, past_key_value


def llama_flash_attn2_forward_Mask(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    # get head list
    init_mask(self)


    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)



    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.

    # if self.head_list:
    for h in self.head_list:
        if self.layer_idx==h[0]:
            query_states[:,h[1], :, :] = 0

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value




def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

def prepare_inputs_for_generation_llama_new(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        if not isinstance(past_key_values, tuple):
            if len(past_key_values.key_cache) == 0:
                for layer in self.model.layers:
                    layer.self_attn.kv_seq_len = 0

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        # import pdb;pdb.set_trace()

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
# 这段代码是自定义的 adaptive_LlamaModel_forward 函数，
# 是对 Hugging Face transformers 中 Llama 模型 forward 方法的重写（猴子补丁用）。
# 它的作用是实现 Llama 模型的前向推理流程，并兼容稀疏推理、缓存、梯度检查点等高级特性。
def adaptive_LlamaModel_forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token ids
        attention_mask: Optional[torch.Tensor] = None,  # 自注意力机制的掩码，控制哪些位置应被关注
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码的 id
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,  # 过去的缓存，保存之前的计算结果以加速推理
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出每层的注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出每层的隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典类型的输出
        cache_position: Optional[torch.LongTensor] = None,  # 缓存位置
) -> Union[Tuple, BaseModelOutputWithPast]:
    # 根据配置，设置默认值
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 检查是否同时指定了 input_ids 和 inputs_embeds，二者只能指定其一
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    # 如果启用了梯度检查点（节省显存），且正在训练且使用缓存，则禁用缓存 
    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    # 如果没有提供 inputs_embeds，则根据 input_ids 生成嵌入向量
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    # 如果使用缓存且 past_key_values 不是 Cache 类型，则转换为新的缓存格式
    if (
        use_cache and not (type(past_key_values) == DynamicCacheSplitHeadFlatten) and not self.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        # 兼容旧版本的缓存格式
        past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    # 如果没有提供 cache_position，则根据 past_key_values 计算缓存位置
    # cache_position 计算当前输入 tokens 的位置，特别是在缓存（past_key_values）存在的情况下，
    # 通过 get_seq_length() 确定已经处理过的 tokens 数量，然后生成一个新的位置序列。
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    
    # 如果没有提供 position_ids，则使用 cache_position 作为 position_ids
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # 更新 causal_mask，用于防止信息泄漏的掩码
    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    
    hidden_states = inputs_embeds  # 初始化隐藏状态为输入的嵌入向量

    # 创建位置编码（旋转位置编码），将与每一层共享
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder 层的处理
    all_hidden_states = () if output_hidden_states else None  # 如果需要返回所有隐藏状态，初始化空元组
    all_self_attns = () if output_attentions else None  # 如果需要返回所有注意力，初始化空元组
    next_decoder_cache = None  # 初始化缓存
    # 不是每一层都生成token，最后一层的hidden_states是最终的输出
    # 遍历每一层的 decoder（通常是 LlamaBlock 层）
    for decoder_layer in self.layers:
        # 如果需要输出每层的 hidden states，将当前 hidden_states 存入 all_hidden_states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 如果启用了梯度检查点，则对每层的 forward 进行包裹，以节省显存
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            # 直接调用 decoder 层的 forward 方法
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        
        hidden_states = layer_outputs[0]  # 更新 hidden_states 为当前层的输出

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]  # 更新缓存

        if output_attentions:
            all_self_attns += (layer_outputs[1],)  # 如果需要，存储当前层的注意力

    # 对最后一层的 hidden_states 进行归一化
    hidden_states = self.norm(hidden_states)

    # 如果需要输出所有隐藏状态，则将最后一层的 hidden_state 添加到 all_hidden_states 中
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # 返回缓存（如果有）
    next_cache = next_decoder_cache if use_cache else None

    # 返回 legacy 缓存格式（如果需要）
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    # 如果不返回字典，按顺序返回各项内容
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    
    # 返回一个包含隐藏状态、缓存、隐藏状态历史、注意力的字典
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
