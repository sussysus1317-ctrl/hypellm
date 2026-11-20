"""Benchmark AirLLM against HypeLLM on the same model, prompt and greedy decoding.

    python bench.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode both
    python bench.py --model ... --mode both --vram-budget-gb 0.5 --ram-budget-gb 0.5

`--mode both` runs each engine in its own process and checks that they produce the same tokens.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def run_one(args):
    import torch

    if args.mode == 'air':
        from airllm import AutoModel
        model = AutoModel.from_pretrained(args.model, compression=args.compression)
    else:
        from hypellm import AutoModel
        model = AutoModel.from_pretrained(args.model, compression=args.compression,
                                          vram_budget_gb=args.vram_budget_gb,
                                          ram_budget_gb=args.ram_budget_gb,
                                          cleanup=args.cleanup,
                                          draft_model=args.draft,
                                          draft_tokens=args.draft_tokens,
                                          draft_confidence=args.draft_confidence)

    cuda = model.device.type == 'cuda'
    ids = model.tokenizer([args.prompt], return_tensors='pt',
                          return_attention_mask=False)['input_ids'].to(model.device)

    def generate(n):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
        if cuda:
            torch.cuda.synchronize()
        return out, time.perf_counter() - start

    # The first token pays one-off costs (CUDA start-up, cold file reads, filling the RAM tier).
    _, first = generate(1)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    stats = getattr(model, 'hype_stats', None)
    sweeps_before = stats['sweeps'] if stats else 0
    out, elapsed = generate(args.max_new_tokens)
    sweeps = stats['sweeps'] - sweeps_before if stats else args.max_new_tokens

    result = {
        'sweeps': sweeps,
        'tokens_per_sweep': round(args.max_new_tokens / sweeps, 2),
        'mode': args.mode,
        'first_token_s': round(first, 3),
        'new_tokens': args.max_new_tokens,
        'elapsed_s': round(elapsed, 3),
        's_per_token': round(elapsed / args.max_new_tokens, 4),
        'peak_vram_mb': round(torch.cuda.max_memory_allocated() / 1e6) if cuda else None,
        'tokens': out[0].tolist(),
        'text': model.tokenizer.decode(out[0], skip_special_tokens=True),
        'plan': getattr(model, 'hype_plan', None),
        'stats': getattr(model, 'hype_stats', None),
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f)


def run_both(args):
    results = {}
    for mode in ('air', 'hype'):
        out = os.path.join(tempfile.gettempdir(), f'hypellm_bench_{mode}.json')
        cmd = [sys.executable, os.path.abspath(__file__), '--model', args.model, '--mode', mode,
               '--prompt', args.prompt, '--max-new-tokens', str(args.max_new_tokens), '--out', out]
        if args.compression:
            cmd += ['--compression', args.compression]
        if args.vram_budget_gb is not None:
            cmd += ['--vram-budget-gb', str(args.vram_budget_gb)]
        if args.ram_budget_gb is not None:
            cmd += ['--ram-budget-gb', str(args.ram_budget_gb)]
        cmd += ['--cleanup', args.cleanup]
        print(f"--- running {mode} ---", flush=True)
        subprocess.run(cmd, check=True)
        with open(out, encoding='utf-8') as f:
            results[mode] = json.load(f)

    air, hype = results['air'], results['hype']
    print("\n=== result ===")
    print(f"{'':10}{'first token':>14}{'s/token':>12}{'peak VRAM':>12}")
    for r in (air, hype):
        print(f"{r['mode']:10}{r['first_token_s']:>13.2f}s{r['s_per_token']:>11.3f}s"
              f"{r['peak_vram_mb']:>10}MB")
    print(f"speedup per token: {air['s_per_token'] / hype['s_per_token']:.1f}x")
    print(f"plan : {hype['plan']}")
    print(f"stats: {hype['stats']}")
    print(f"text : {hype['text']!r}")
    match = air['tokens'] == hype['tokens']
    print(f"outputs: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print(f"air : {air['text']!r}")
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--mode', choices=['air', 'hype', 'both'], default='both')
    p.add_argument('--prompt', default='The three largest planets in the solar system are')
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--compression', choices=['4bit', '8bit'], default=None)
    p.add_argument('--vram-budget-gb', type=float, default=None)
    p.add_argument('--ram-budget-gb', type=float, default=None)
    p.add_argument('--cleanup', choices=['sweep', 'layer', 'never'], default='never')
    p.add_argument('--draft', default=None, help='small model with the same tokenizer (hype only)')
    p.add_argument('--draft-tokens', type=int, default=None)
    p.add_argument('--draft-confidence', type=float, default=None)
    p.add_argument('--out', default=os.path.join(tempfile.gettempdir(), 'hypellm_bench.json'))
    args = p.parse_args()
    if args.mode == 'both':
        run_both(args)
    else:
        run_one(args)


if __name__ == '__main__':
    main()
