#!/usr/bin/env python3
"""
AI Detection Score for LaTeX papers.
Extracts clean text (no formulas/tables/figures), computes per-paragraph
GPT-2 perplexity and sentence-length burstiness.

Usage:
  python3 scripts/ai_check.py paper/main.tex
  python3 scripts/ai_check.py paper/main.tex --copytext  # also save clean text for GPTZero
"""
import re, math, sys, argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np

# ── LaTeX → clean text ─────────────────────────────────────────────
def extract_text(tex_path):
    with open(tex_path) as f:
        tex = f.read()
    # Body only
    m = re.search(r'\\begin\{document\}(.+?)\\end\{document\}', tex, re.DOTALL)
    body = m.group(1) if m else tex
    # Strip environments
    for env in ['figure', 'table', 'enumerate', 'itemize', 'equation', 'align']:
        body = re.sub(rf'\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}', '', body, flags=re.DOTALL)
    # Split into paragraphs
    raw = re.split(r'\n\s*\n', body)
    paras = []
    for p in raw:
        t = p.strip()
        if t.startswith('%') or t.startswith('\\begin') or t.startswith('\\end'):
            continue
        # Clean LaTeX
        t = re.sub(r'\\textdegree\{?\}?', 'deg', t)
        t = re.sub(r'\\(sub)*section\*?\{([^}]+)\}', '', t)
        t = re.sub(r'\\cite[a-z]*\{[^}]+\}', '', t)
        t = re.sub(r'\\ref\{[^}]+\}', 'X', t)
        t = re.sub(r'\\label\{[^}]+\}', '', t)
        t = re.sub(r'\\url\{[^}]+\}', '', t)
        t = re.sub(r'\\text[a-z]+\{([^}]*)\}', r'\1', t)
        t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
        t = re.sub(r'\\[a-zA-Z]+', '', t)
        t = re.sub(r'[{}]', '', t)
        t = re.sub(r'\$[^$]+\$', 'X', t)
        t = re.sub(r'~', ' ', t)
        t = re.sub(r'---', '—', t)
        t = re.sub(r'--', '–', t)
        t = re.sub(r'  +', ' ', t).strip()
        if len(t) > 80:
            paras.append(t)
    return paras

# ── Perplexity ─────────────────────────────────────────────────────
def load_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model.eval()
    return model, tokenizer

def calc_ppl(model, tokenizer, text, max_len=512):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
    with torch.no_grad():
        out = model(enc.input_ids, labels=enc.input_ids)
    return math.exp(out.loss.item())

# ── Burstiness ─────────────────────────────────────────────────────
def calc_burstiness(text):
    """Sentence length variation. Higher = more human-like."""
    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 5]
    if len(sents) < 3:
        return 0.0
    lens = [len(s.split()) for s in sents]
    mean_len = np.mean(lens)
    if mean_len == 0:
        return 0.0
    return np.std(lens) / mean_len  # coefficient of variation

# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('texfile')
    parser.add_argument('--copytext', action='store_true',
                        help='Save clean text for manual GPTZero check')
    args = parser.parse_args()

    paras = extract_text(args.texfile)
    print(f'Extracted {len(paras)} paragraphs\n')

    if args.copytext:
        outpath = args.texfile.replace('.tex', '_cleantext.txt')
        with open(outpath, 'w') as f:
            for p in paras:
                f.write(p + '\n\n')
        print(f'Clean text saved to: {outpath}')
        print('(Copy this into GPTZero for accurate results)\n')

    print('Loading GPT-2...')
    model, tokenizer = load_model()

    results = []
    for i, p in enumerate(paras):
        ppl = calc_ppl(model, tokenizer, p[:512])
        burst = calc_burstiness(p)
        # Heuristic AI score: low PPL + low burstiness = AI-like
        # PPL < 40 → very AI, 40-80 → suspect, > 80 → likely human
        if ppl < 40:
            ai_score = 0.9
        elif ppl < 60:
            ai_score = 0.7
        elif ppl < 80:
            ai_score = 0.5
        elif ppl < 120:
            ai_score = 0.3
        else:
            ai_score = 0.1
        # Burstiness adjustment: high burstiness → less AI-like
        if burst > 0.5:
            ai_score *= 0.7
        elif burst < 0.2:
            ai_score *= 1.2
        ai_score = min(ai_score, 1.0)
        results.append((i, ppl, burst, ai_score, p[:120]))

    # Sort by AI score (highest first)
    results.sort(key=lambda x: -x[3])

    print(f'\n{"="*70}')
    print(f'  TOP 10 MOST AI-LIKE PARAGRAPHS')
    print(f'{"="*70}')
    for idx, ppl, burst, ai, preview in results[:10]:
        flag = '🔴' if ai >= 0.7 else '🟡' if ai >= 0.5 else '🟢'
        print(f'  {flag} [{idx:2d}] AI={ai:.0%}  PPL={ppl:6.1f}  Burst={burst:.2f}')
        print(f'       {preview}...\n')

    print(f'{"="*70}')
    print(f'  OVERALL STATISTICS')
    print(f'{"="*70}')
    ppls = [r[1] for r in results]
    ais = [r[3] for r in results]
    n_red = sum(1 for a in ais if a >= 0.7)
    n_yellow = sum(1 for a in ais if 0.5 <= a < 0.7)
    n_green = sum(1 for a in ais if a < 0.5)
    print(f'  Mean PPL:     {np.mean(ppls):.1f}')
    print(f'  Median PPL:   {np.median(ppls):.1f}')
    print(f'  Mean AI score: {np.mean(ais):.0%}')
    print(f'  🔴 High risk (AI ≥ 70%):  {n_red} paragraphs')
    print(f'  🟡 Medium risk (50-70%):   {n_yellow} paragraphs')
    print(f'  🟢 Low risk (< 50%):       {n_green} paragraphs')
    print()

if __name__ == '__main__':
    main()
