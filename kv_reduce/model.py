from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, LlamaConfig, apply_rotary_pos_emb, LlamaForCausalLM, CausalLMOutputWithPast
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# from kv_manage import generate_mask
from kv_reduce.kv_manage import generate_mask, greedy_generate_mask, KvManager, SimpleKvManager, KvManagerQuick


class KvLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.num_kv = -1
        self.group = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        #####################################################################################################
        if self.num_kv != -1 and use_cache == False:
            attention_mask = greedy_generate_mask(attn_weights,
                                                  attention_mask,
                                                  remain_kv=self.num_kv,
                                                  recent=10,
                                                  group=self.group)
        #####################################################################################################

        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            #     )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @classmethod
    def convert_from(cls, module: LlamaAttention):
        new_module = cls(module.config)
        new_module.load_state_dict(module.state_dict())
        return new_module


class KvLlamaAttentionForGenerate(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_manager = KvManagerQuick()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            #####################################################################################
            kv_seq_len += self.kv_manager.passed_kv
            #####################################################################################
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        #####################################################################################
        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}")
        #####################################################################################
        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            #     )
            # attn_weights
            if attention_mask.shape[-2] != 1:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        #####################################################################################
        past_key_value = self.kv_manager.step(attn_weights, past_key_value,
                                              attention_mask)
        #####################################################################################
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @classmethod
    def convert_from(cls, module: LlamaAttention):
        new_module = cls(module.config)
        new_module.load_state_dict(module.state_dict(), strict=False)
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        new_module = new_module.to(device).to(dtype)
        return new_module


def llama_forward_wrapper(self, *args, **kwargs):
    module = self
    args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
    if module._hf_hook.no_grad:
        with torch.no_grad():
            output = llama_forward(self, *args, **kwargs)
    else:
        output = llama_forward(self, *args, **kwargs)
    return module._hf_hook.post_forward(module, output)


def llama_forward(
    self: LlamaForCausalLM,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    ####################################################################################################
    self.group: int
    self.num_kv: int
    #####################################################################################################
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states if output_hidden_states
                            is not None else self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

    #####################################################################################################
    hidden = []
    N = input_ids.shape[-1]

    max_num_kv = self.num_kv + self.group
    if N > max_num_kv:
        num_split = [max_num_kv] + [self.group] * (
            (N - max_num_kv) // self.group)
        if N != sum(num_split):
            num_split += [N - sum(num_split)]
        input_ids_list = input_ids.split(num_split, dim=-1)
        position_ids_list = position_ids.split(
            num_split,
            dim=-1) if position_ids is not None else [None] * len(num_split)

        for inputs, pos in zip(input_ids_list, position_ids_list):
            if past_key_values is not None:
                passed_num = inputs.shape[-1] + past_key_values[0][0].shape[-2]
            else:
                passed_num = inputs.shape[-1]
            outputs = self.model(
                input_ids=inputs,
                attention_mask=attention_mask[:, :passed_num]
                if attention_mask is not None else None,
                position_ids=pos,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            past_key_values = outputs.past_key_values
            hidden.append(outputs[0])
        hidden = torch.cat(hidden, dim=-2)
        outputs.last_hidden_state = hidden
    else:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    ####################################################################################################

    hidden_states = outputs[0]
    hidden = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = hidden[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (hidden, ) + outputs[1:]
        return (loss, ) + output if loss is not None else output
    ####################################################################################################
    assert outputs.past_key_values[0][0].shape[
        -2] <= self.num_kv + self.group + 2
    ####################################################################################################
    return CausalLMOutputWithPast(
        loss=loss,
        logits=hidden,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


if __name__ == '__main__':
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    B, N, C = 2, 10, 512
    h = 32
    He = C // h

    config = LlamaConfig(hidden_size=512, num_attention_heads=He)
    attention = KvLlamaAttentionForGenerate(config)
    attention.kv_manager.reset(5)
    attention.kv_manager.passed_kv = 10

    past_key_value = (
        torch.randn([B, He, N, h]),
        torch.randn(B, He, N, h),
    )
    for i in range(5):
        out = attention.forward(
            torch.randn(2, 1, 512),
            past_key_value=past_key_value,
            position_ids=torch.tensor([10 + i]),
            use_cache=True,
        )
        _, _, past_key_value = out
        print(past_key_value[0].shape, past_key_value[1].shape)