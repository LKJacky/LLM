import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import _make_causal_mask
from torch import Tensor

from typing import Tuple, List
import torch.nn.functional as F


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


# @torch.script()
def step(imp: Tensor,
         attn_weights: Tensor,
         past_key_value: Tuple[Tensor, Tensor],
         decay=0.99,
         recent=10,
         num_keep_kv=128,
         group=1,
         num_passed_kv=0):
    max_inf = torch.finfo(attn_weights.dtype).max
    B, He, N, _ = attn_weights.shape
    # # assert N == 1
    if imp is None:
        imp = attn_weights.mean(dim=2, keepdim=True) * (1 - decay)
    else:
        # imp: B He 1 KV-1
        imp = torch.cat([imp, torch.zeros(B, He, 1, 1).to(imp.device)], dim=-1)

        imp = imp * decay + attn_weights * (1 - decay)
    num_kv = past_key_value[0].shape[-2]
    if past_key_value is not None and num_kv > num_keep_kv and num_passed_kv % group == 0:
        imp = imp.clone()
        imp[-recent:] = max_inf
        index: torch.Tensor = torch.topk(imp,
                                         num_keep_kv,
                                         dim=-1,
                                         largest=True)[1]

        imp = imp.gather(-1, index)
        index = index.squeeze(-2).unsqueeze(-1)
        index = index.expand([*index.shape[:-1], past_key_value[0].shape[-1]])
        past_key_value = (
            past_key_value[0].gather(-2, index),
            past_key_value[1].gather(-2, index),
        )
    return imp, past_key_value


# @torch.jit.script
def attn_update(current: Tensor, past: Tensor, decay: float = 0.99):
    past = F.pad(past, (0, 1), value=0.)
    return past * decay + current * (1 - decay)


class KvManager:

    def __init__(self, num_kv=10, group=1, decay=0.99, recent=10) -> None:
        self.reset(num_kv, group, decay, recent)

    def reset(self, num_kv=10, group=1, decay=0.99, recent=10):
        self.num_kv = num_kv
        self.group = group
        self.decay = decay
        self.recent = recent
        self.imp: torch.Tensor = None

        self.passed_kv = 0

    @torch.no_grad()
    def step(self, attn_weights: Tensor, past_key_value: Tuple[Tensor,
                                                               Tensor]):
        max_inf = torch.finfo(attn_weights.dtype).max
        B, He, N, _ = attn_weights.shape
        # # assert N == 1
        if self.imp is None:
            self.imp = attn_weights.mean(dim=2,
                                         keepdim=True) * (1 - self.decay)
        else:
            self.imp = attn_update(attn_weights, self.imp, self.decay)
        num_kv = past_key_value[0].shape[-2]
        if past_key_value is not None and num_kv > self.num_kv and self.passed_kv % self.group == 0:
            imp = self.imp.clone()
            imp[-self.recent:] = max_inf
            index: torch.Tensor = torch.topk(imp,
                                             self.num_kv,
                                             dim=-1,
                                             largest=True)[1]

            self.imp = self.imp.gather(-1, index)
            index = index.squeeze(-2).unsqueeze(-1)
            index = index.expand(
                [*index.shape[:-1], past_key_value[0].shape[-1]])
            past_key_value = (
                past_key_value[0].gather(-2, index),
                past_key_value[1].gather(-2, index),
            )
        self.passed_kv += N
        return past_key_value


def compute_imp(past_imp: Tensor, new_imps: List[Tensor]):
    new_imps.insert(0, past_imp)
    B, He, N, _ = past_imp.shape
    new_imps = [imp.transpose(-1, 0) for imp in new_imps]  # KV He N B
    # KV = new_imps[-1].shape[-1]  # B He N KV
    # imps = [
    #     F.pad(imp, (0, KV - imp.shape[-1]), value=0).unsqueeze(-1)
    #     for imp in new_imps
    # ]
    imp = torch.nn.utils.rnn.pad_sequence(new_imps,
                                          batch_first=False,
                                          padding_value=0)  # KV L He N B
    imp = imp.mean(dim=1).transpose(0, -1)

    return imp


class KvManagerQuick(KvManager):

    def __init__(self, num_kv=10, group=1, decay=0.99, recent=10) -> None:
        super().__init__(num_kv, group, decay, recent)
        self.imp_list = []

    @torch.no_grad()
    def step(self, attn_weights: Tensor, past_key_value: Tuple[Tensor,
                                                               Tensor]):
        max_inf = torch.finfo(attn_weights.dtype).max
        B, He, N, _ = attn_weights.shape
        # # assert N == 1
        if self.imp is None:
            self.imp = attn_weights.mean(dim=2,
                                         keepdim=True) * (1 - self.decay)
        else:
            if self.passed_kv % self.group == 0:
                self.imp = compute_imp(self.imp, self.imp_list)
                self.imp_list = []
            else:
                self.imp_list.append(attn_weights)
            # self.imp = attn_update(attn_weights, self.imp, self.decay)
        num_kv = past_key_value[0].shape[-2]
        if past_key_value is not None and num_kv > self.num_kv and self.passed_kv % self.group == 0:
            imp = self.imp.clone()
            imp[-self.recent:] = max_inf
            index: torch.Tensor = torch.topk(imp,
                                             self.num_kv,
                                             dim=-1,
                                             largest=True)[1]

            self.imp = self.imp.gather(-1, index)
            index = index.squeeze(-2).unsqueeze(-1)
            index = index.expand(
                [*index.shape[:-1], past_key_value[0].shape[-1]])
            past_key_value = (
                past_key_value[0].gather(-2, index),
                past_key_value[1].gather(-2, index),
            )
        self.passed_kv += N
        return past_key_value


class SimpleKvManager(KvManager):

    def step(self, attn_weights: Tensor, past_key_value: Tuple[Tensor,
                                                               Tensor]):
        B, He, N, _ = attn_weights.shape
        if self.passed_kv >= self.num_kv and self.passed_kv % self.group == 0:
            current_kv = past_key_value[0].shape[-2]
            if current_kv > self.num_kv:

                past_key_value = (
                    past_key_value[0][:, :, -self.num_kv:],
                    past_key_value[1][:, :, -self.num_kv:],
                )
        self.passed_kv += N
        return past_key_value


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