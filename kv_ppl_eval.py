import argparse

import torch
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from ppl_data import get_loaders
from torch.amp import autocast
from kv_reduce.kv_manage import replace_modules
from kv_reduce.model import KvLlamaAttention, LlamaAttention


def PPLMetric(model,
              tokenizer,
              datasets,
              seq_len=128,
              batch_size=4,
              device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset,
                                     tokenizer,
                                     seq_len=seq_len,
                                     batch_size=batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric


@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        nlls.append(loss)
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def main(args):
    device = 'cuda:0'
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    replace_modules(model, {LlamaAttention: KvLlamaAttention})

    model = model.to(device)
    with autocast('cuda'):
        ppl = PPLMetric(model,
                        tokenizer, ['ptb', 'wikitext2'],
                        seq_len=2048,
                        device=device)
    print(ppl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--model',
                        type=str,
                        default="decapoda-research/llama-7b-hf",
                        help='base model name')

    args = parser.parse_args()

    main(args)

# {'wikitext2': 5.677364622731849, 'ptb': 8.804186998739304}
