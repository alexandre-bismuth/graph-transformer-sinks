"""Language-model reference, computed with exactly the paper's sink estimator.

For a causal LM, per layer l and head h, s_j = mean_u A_uj over all T query rows (the
same column mean as for the graph models), and

  sink rate = max_l  E_windows[ max_j (1/H) sum_h 1[s_j^(l,h) > eps] ],   eps = 0.3,

exactly as metrics_v2.compute_sink_metrics / aggregate_results.py compute it for graphs.
Context lengths default to 23 (median ZINC size) and 269. Text: WikiText-2 (raw) test
split, non-overlapping windows of T tokens, no BOS token prepended (the window's first
token is whatever the text gives), so the comparison is not inflated by a special token.

Because attention is causal, the "uniform" reference differs from 1/n: if every row
spreads uniformly over its prefix, s_1 = H_T / T (H_T the harmonic number), i.e. 0.163 at
T=23 and 0.023 at T=269. Both baselines are reported.

Usage:
  python scripts/lm_sink_reference.py [--models EleutherAI/pythia-70m EleutherAI/pythia-160m gpt2] \
      [--lengths 23 269] [--n-windows 200] -o outputs/lm_reference/lm_sink_reference.json
Needs: transformers, pyarrow, network access to the Hugging Face Hub (or a local cache).
"""
import argparse
import json
import os

import numpy as np
import torch

EPS = 0.3


def load_text():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet',
                        repo_type='dataset')
    return '\n'.join(pd.read_parquet(p)['text'].tolist())


@torch.no_grad()
def run_model(name, text, lengths, n_windows, eps, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation='eager').to(device).eval()
    ids = tok(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    res = {}
    for T in lengths:
        n_avail = len(ids) // T
        nw = min(n_windows, n_avail)
        per_layer_rate, per_layer_max_s, first_is_argmax = None, None, None
        for w in range(nw):
            x = ids[w * T:(w + 1) * T].unsqueeze(0).to(device)
            out = model(x, output_attentions=True)
            att = out.attentions                         # L x (1, H, T, T), rows sum to 1
            if per_layer_rate is None:
                L = len(att)
                per_layer_rate = np.zeros((nw, L)); per_layer_max_s = np.zeros((nw, L))
                first_is_argmax = np.zeros((nw, L))
            for l, A in enumerate(att):
                s = A[0].float().mean(dim=1)             # (H, T): column mean over queries
                per_layer_rate[w, l] = float((s > eps).float().mean(dim=0).max())
                sm = s.mean(dim=0)
                per_layer_max_s[w, l] = float(sm.max())
                first_is_argmax[w, l] = float(int(sm.argmax()) == 0)
        rate = per_layer_rate.mean(axis=0)
        ms = per_layer_max_s.mean(axis=0)
        harm = float(sum(1.0 / k for k in range(1, T + 1)))
        res[str(T)] = {
            'n_windows': nw,
            'sink_rate': float(rate.max()), 'peak_layer': int(rate.argmax()) + 1,
            'sink_rate_per_layer': rate.tolist(),
            'max_s_per_layer': ms.tolist(),
            'n_times_max_s_peak': float(T * ms.max()),
            'first_token_is_argmax_per_layer': first_is_argmax.mean(axis=0).tolist(),
            'causal_uniform_s1': harm / T,
            'frac_layers_with_rate_above_half': float((rate > 0.5).mean()),
        }
        print(f"{name:>24s} T={T:>3d}: sink rate {rate.max():.3f} (layer {rate.argmax() + 1}/{len(rate)}), "
              f"mean over layers {rate.mean():.3f}, n*max s {T * ms.max():.2f}, "
              f"causal-uniform s_1 {harm / T:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['EleutherAI/pythia-70m', 'EleutherAI/pythia-160m', 'gpt2'])
    ap.add_argument('--lengths', type=int, nargs='+', default=[23, 269])
    ap.add_argument('--n-windows', type=int, default=200)
    ap.add_argument('--eps', type=float, default=EPS)
    ap.add_argument('--device', default=None)
    ap.add_argument('-o', '--out', default='outputs/lm_reference/lm_sink_reference.json')
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text = load_text()
    allres = {'eps': args.eps, 'text': 'wikitext-2-raw-v1 test', 'models': {}}
    for m in args.models:
        allres['models'][m] = run_model(m, text, args.lengths, args.n_windows, args.eps, device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(allres, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
