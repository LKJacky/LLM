import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from kv_reduce.model import KvLlamaAttentionForGenerate
from transformers.models.llama.modeling_llama import LlamaAttention
from kv_reduce.kv_manage import replace_modules


def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('--text', type=str, help='input text')

    parser.add_argument(
        '--min_length',
        type=int,
        default=10,
        help='The minimum length of the sequence to be generated.')

    parser.add_argument(
        '--max_length',
        type=int,
        default=50,
        help='The maximum length of the sequence to be generated.')

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help=
        'If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='The value used to module the next token probabilities.')
    parser.add_argument('--kv', type=int, default=2048, help='base model name')
    parser.add_argument('--group', type=int, default=1, help='base model name')
    args = parser.parse_args()

    DEV = 'cuda:0'
    from transformers import LlamaForCausalLM
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    model.eval()


    replace_modules(model, {LlamaAttention: KvLlamaAttentionForGenerate})
    for module in model.modules():
        if isinstance(module, KvLlamaAttentionForGenerate):
            module.kv_manager.reset(10, 10)
    print(model)

    model.to(DEV)
    input_ids = tokenizer.encode(args.text, return_tensors="pt").to(DEV)


    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=50,
            max_length=100,
            top_p=args.top_p,
            temperature=args.temperature,
        )

    for module in model.modules():
        if isinstance(module, KvLlamaAttentionForGenerate):
            module.kv_manager.reset(args.kv, args.group)

    import time
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=args.min_length,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
        )
    torch.cuda.synchronize()
    print(f'use {time.time()-t0} with {len(generated_ids[0])} tokens')
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))