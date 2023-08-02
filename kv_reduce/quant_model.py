from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, LlamaConfig, apply_rotary_pos_emb, LlamaForCausalLM, LlamaDecoderLayer
import math
from typing import List, Optional, Tuple, Union
import numpy

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from torch import Tensor
from kv_reduce.kv_manage import replace_modules

from kv_reduce.model import KvLlamaAttentionForGenerate


class QuantKvAttnMixin():

    def __init__(self) -> None:
        k_scale = torch.tensor([1.0])
        v_scale = torch.tensor([1.0])

        self.register_buffer('k_scale', k_scale)
        self.register_buffer('v_scale', v_scale)

        self.k_scale: Tensor
        self.v_scale: Tensor

    def quant(self, past_key_value, dtype=torch.half):
        key_states, value_states = past_key_value
        key_states = torch.clamp(torch.round(key_states / self.k_scale),
                                 min=-128.0,
                                 max=127.0).to(dtype)
        value_states = torch.clamp(torch.round(value_states / self.v_scale),
                                   min=-128.0,
                                   max=127.0).to(dtype)
        past_key_value = (key_states, value_states)
        return past_key_value

    def dequant(self, past_key_value, dtype=torch.half):
        key_states, value_states = past_key_value
        key_states = (key_states * self.k_scale).to(dtype)
        value_states = value_states * self.v_scale.to(dtype)
        past_key_value = (key_states, value_states)
        return past_key_value


class QuantKvAttention(LlamaAttention, QuantKvAttnMixin):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        QuantKvAttnMixin.__init__(self)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids=None,
        past_key_value: Tuple[Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[Tensor, Tensor | None, Tuple[Tensor] | None]:
        if past_key_value is not None:
            past_key_value = self.dequant(past_key_value, hidden_states.dtype)

        attn_output, attn_weights, past_key_value = super().forward(
            hidden_states, attention_mask, position_ids, past_key_value,
            output_attentions, use_cache)

        if past_key_value is not None:
            past_key_value = self.quant(past_key_value, hidden_states.dtype)

        return attn_output, attn_weights, past_key_value

    @classmethod
    def convert_from(cls, module: LlamaAttention):
        new_module = cls(module.config)
        new_module.load_state_dict(module.state_dict(), strict=False)

        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        new_module = new_module.to(device).to(dtype)

        return new_module


class KvLlamaAttentionForGenerateQuant(KvLlamaAttentionForGenerate,
                                       QuantKvAttnMixin):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        QuantKvAttnMixin.__init__(self)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids=None,
        past_key_value: Tuple[Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[Tensor, Tensor | None, Tuple[Tensor] | None]:
        if past_key_value is not None:
            past_key_value = self.dequant(past_key_value, hidden_states.dtype)

        attn_output, attn_weights, past_key_value = super().forward(
            hidden_states, attention_mask, position_ids, past_key_value,
            output_attentions, use_cache)

        if past_key_value is not None:
            past_key_value = self.quant(past_key_value, hidden_states.dtype)

        return attn_output, attn_weights, past_key_value

# class KvLlamaAttentionForGenerateQuantNoHead(KvLlamaAttentionForGenerateQuant):
#     def 

def load_kv_scale(model, quant_kv_prefix='quant_kv/'):

    def quant_kv_path(layer_idx):
        return f"layers.{layer_idx}.past_kv_scale.0.weight"

    quant_kv_prefix = 'quant_kv/'

    for i, layer in enumerate(model.model.layers):
        path = quant_kv_prefix + quant_kv_path(i)
        scale = numpy.fromfile(path, dtype=numpy.float32)
        layer: LlamaDecoderLayer
        layer.self_attn.k_scale = layer.self_attn.k_scale.float()
        layer.self_attn.v_scale = layer.self_attn.v_scale.float()
        layer.self_attn.k_scale.data.fill_(scale[0])
        layer.self_attn.v_scale.data.fill_(scale[0])
    return model


if __name__ == '__main__':
    import numpy as np

    a = np.fromfile('quant_kv/layers.0.past_kv_scale.0.weight',
                    dtype=np.float32)

    from transformers.models.llama.modeling_llama import LlamaConfig

    B, N, C = 2, 10, 512
    h = 32
    He = C // h

    config = LlamaConfig(hidden_size=512, num_attention_heads=He)
    attention = QuantKvAttention(config)

    past_key_value = None
    for i in range(5):
        out = attention.forward(
            torch.randn(2, 1, 512),
            past_key_value=past_key_value,
            position_ids=torch.tensor([0 + i]),
            use_cache=True,
        )
        _, _, past_key_value = out
        print(
            past_key_value[0].shape,
            past_key_value[1].shape,
            past_key_value[0].dtype,
            past_key_value[0].view(-1)[:5],
        )
    llama = build_llama()
