import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import _make_causal_mask


def replace_modules(model: nn.Module, module_map={}):

    def replace_op(model: nn.Module, name: str, module: nn.Module):
        names = name.split('.')
        for sub_name in names[:-1]:
            model = getattr(model, sub_name)

        setattr(model, names[-1], module)

    for name, module in model.named_modules():
        if name != '' and type(module) in module_map:
            new_module = module_map[type(module)].convert_from(module)
            replace_op(model, name, new_module)


def generate_mask(attention: torch.Tensor, attention_mask: torch.Tensor,
                  mask_k: int):
    attention = attention + attention_mask

    mask = torch.zeros_like(attention)
    topk = attention.topk(mask_k, -1, largest=False)[1]
    mask.scatter_(-1, topk, attention_mask.min().item())
    mask = torch.min(mask, attention_mask)
    return mask


def greedy_generate_mask(attention: torch.Tensor,
                         attention_mask: torch.Tensor,
                         remain_kv: int,
                         recent=10):
    inf = attention_mask.min().item() * -1

    mask = torch.zeros_like(attention)

    attention = attention + attention_mask * -1
    B, He, Q, K = attention.shape
    for i in range(Q):
        num_kv = i
        if num_kv > remain_kv:
            attn = attention[:, :, i, :]  # B He K
            attn[:, :, i - recent:i + 1] = torch.finfo(attn.dtype).min
            drop = attn.argmin(dim=-1)  # B He
            drop = drop.unsqueeze(-1).unsqueeze(-1)
            drop = drop.repeat([1, 1, Q, 1])
            drop[:, :, :i] = K - 1
            attention = attention.scatter(-1, drop,
                                          torch.finfo(attention.dtype).min)
            mask.scatter_(-1, drop, torch.finfo(mask.dtype).min)

    mask = torch.min(mask, attention_mask)
    return mask


if __name__ == '__main__':
    B, He, Q, K = 2, 2, 5, 5
    attention = torch.rand(B, He, Q, K)
    attention_mask = _make_causal_mask([B, Q], dtype=torch.float, device='cpu')

    mask = greedy_generate_mask(attention,
                                attention_mask,
                                attention.shape[-1] // 2,
                                recent=1)
    print(mask)