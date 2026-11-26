# -*- coding: utf-8 -*-
"""Summarise Survival Analysis runs into surv_results.json and print the key tables.
Usage: python3 -I surv_aggregate.py <out_dir> <results.json>"""
import glob, json, math, os, sys
import numpy as np
from scipy import stats
OUT, RES = sys.argv[1], sys.argv[2]
CK = os.path.join(OUT, 'checkpoints')
CARS = ['sonata', 'soul', 'spark']; KINDS = ['free', 'flood', 'malf', 'fuzzy']


def load(prefix):
    r = [json.load(open(os.path.join(CK, '%s__seed%d.json' % (prefix, s)))) for s in range(10)
         if os.path.exists(os.path.join(CK, '%s__seed%d.json' % (prefix, s)))]
    return r if len(r) == 10 else None


def ms(v):
    v = np.asarray(v, float)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'min': float(v.min()), 'max': float(v.max()),
            'values': v.tolist()}


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b
    half = stats.t.ppf(0.975, 9) * d.std(ddof=1) / math.sqrt(10)
    try:
        p = float(stats.wilcoxon(a, b, alternative='greater').pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {'mean_diff': float(d.mean()), 'ci95': [float(d.mean() - half), float(d.mean() + half)], 'p': p}


def summ(runs):
    out = {'within_f1': ms([r['within']['f1'] for r in runs]), 'attack_ids': ms([r['training_attack_ids'] for r in runs]),
           'test': {}}
    for t in runs[0]['test']:
        T = {}
        for k in KINDS:
            T[k] = {m: ms([r['test'][t][k][m] for r in runs]) for m in ('fpr', 'recall', 'precision', 'f1', 'fp')}
        # pooled over the target's four captures
        tp = np.array([sum(r['test'][t][k]['tp'] for k in KINDS) for r in runs], float)
        fp = np.array([sum(r['test'][t][k]['fp'] for k in KINDS) for r in runs], float)
        fn = np.array([sum(r['test'][t][k]['fn'] for k in KINDS) for r in runs], float)
        tn = np.array([sum(r['test'][t][k]['tn'] for k in KINDS) for r in runs], float)
        T['pooled'] = {'f1': ms(100 * 2 * tp / (2 * tp + fp + fn)), 'fpr': ms(100 * fp / (fp + tn)),
                       'recall': ms(100 * tp / (tp + fn)), 'precision': ms(100 * tp / np.maximum(tp + fp, 1))}
        out['test'][t] = T
    return out


R = {'E2': {}, 'E3': {}, 'E2_tests': {}}
for src in CARS:
    for rep in ('raw10', 'eng8'):
        c, f = load('SAE2__%s__clean__%s' % (src, rep)), load('SAE2__%s__fuzz__%s' % (src, rep))
        if not (c and f):
            continue
        R['E2'].setdefault(src, {})[rep] = {'clean': summ(c), 'fuzz': summ(f)}
        for t in c[0]['test']:
            R['E2_tests'].setdefault(src, {}).setdefault(rep, {})[t] = {
                'free_fpr_increase': paired([r['test'][t]['free']['fpr'] for r in f], [r['test'][t]['free']['fpr'] for r in c]),
                'pooled_f1_drop': paired(R['E2'][src][rep]['clean']['test'][t]['pooled']['f1']['values'],
                                         R['E2'][src][rep]['fuzz']['test'][t]['pooled']['f1']['values'])}
    for rep in ('raw10', 'eng8'):
        for d in (0, 500, 1000, 2500, 5000, 10000, 20000):
            r = load('SAE3__%s__dose%05d__%s' % (src, d, rep))
            if r:
                R['E3'].setdefault(src, {}).setdefault(rep, {})[str(d)] = summ(r)
json.dump(R, open(RES, 'w'), indent=1)

f = lambda s: '%6.2f±%5.2f' % (s['mean'], s['sd'])
print('SA-E2: target free-driving FPR (%), and recall on target flood / malf / fuzzy, pooled F1')
for src, reps in R['E2'].items():
    for rep, cc in reps.items():
        for comp in ('clean', 'fuzz'):
            S = cc[comp]
            for t, T in S['test'].items():
                print('%-6s->%-6s %-5s %-5s FPRfree %s | rec flood %s malf %s fuzz %s | FPR in flood %s malf %s fuzz %s | pooledF1 %s | within %s' % (
                    src, t, rep, comp, f(T['free']['fpr']), f(T['flood']['recall']), f(T['malf']['recall']), f(T['fuzzy']['recall']),
                    f(T['flood']['fpr']), f(T['malf']['fpr']), f(T['fuzzy']['fpr']), f(T['pooled']['f1']), f(S['within_f1'])))
print()
for src, reps in R['E3'].items():
    for rep, doses in reps.items():
        for t in [c for c in CARS if c != src]:
            print('E3 %-6s->%-6s %-5s FPRfree:' % (src, t, rep), ' '.join('%s:%.2f' % (d, v['test'][t]['free']['fpr']['mean']) for d, v in doses.items()),
                  '| pooledF1:', ' '.join('%.1f' % v['test'][t]['pooled']['f1']['mean'] for d, v in doses.items()))
