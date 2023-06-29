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
                         recent=10,
                         group=1):
    min_inf = torch.finfo(attention.dtype).min
    decay = 0.99
    B, He, Q, K = attention.shape

    mask = attention_mask.clone().repeat([1, He, 1, 1])  # B 1 Q K
    imp = torch.zeros([B, He, K],
                      device=attention.device,
                      dtype=attention.dtype)
    kv_mask = torch.zeros_like(imp)

    for i in range(Q):
        # update importance
        attn_i = attention[:, :, i, :]  # B He K
        mask_i = mask[:, :, i, :]
        attn_i = (attn_i + mask_i).softmax(dim=-1)
        imp = imp * decay + attn_i * (1 - decay)
        num_kv = int((mask_i[0][0] == 0).float().sum().item())

        # update kv_mask
        if num_kv > remain_kv and i % group == 0:
            mask_i2 = mask_i.clone()
            mask_i2[:, :, i - recent:i + 1] = min_inf
            attn_i = attn_i + (-mask_i2)
            index = attn_i.topk(num_kv - remain_kv, dim=-1, largest=False)[1]
            kv_mask = torch.zeros_like(mask_i2)
            kv_mask.scatter_(-1, index, min_inf)

        # write back mask
        mask_i = torch.min(kv_mask, mask_i)
        mask[:, :, i, :] = mask_i

    return mask


if __name__ == '__main__':
    B, He, Q, K = 2, 2, 5, 5
    attention = torch.rand(B, He, Q, K)
    attention_mask = _make_causal_mask([B, Q], dtype=torch.float, device='cpu')

    mask = greedy_generate_mask(attention,
                                attention_mask,
                                attention.shape[-1] // 2,
                                recent=1,
                                group=2)
    print(mask)