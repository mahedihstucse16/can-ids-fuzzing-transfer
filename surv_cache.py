# -*- coding: utf-8 -*-
"""18-feature arrays for every Survival Analysis capture, features computed over each complete
capture in time order (as for Car-Hacking). Spark fuzzy: frames flagged R whose identifier never
occurs in the Spark's free-driving capture are relabelled as injected (label repair, counted).
Usage: python3 -I surv_cache.py <parsed_dir>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surv_features import eng
D = sys.argv[1]
audit = {}
vocab = {c: set(np.load(os.path.join(D, 'raw_%s_free.npz' % c))['ids'].tolist()) for c in ('sonata', 'soul', 'spark')}
for c in ('sonata', 'soul', 'spark'):
    for k in ('free', 'flood', 'fuzzy', 'malf'):
        z = np.load(os.path.join(D, 'raw_%s_%s.npz' % (c, k)))
        ts, ids, dlc, data, y = z['ts'], z['ids'], z['dlc'], z['data'], z['y'].copy()
        fix = 0
        if k == 'fuzzy':
            m = (y == 0) & ~np.isin(ids, list(vocab[c]))
            fix = int(m.sum()); y[m] = 1
        X = np.column_stack([ids, dlc, data, eng(ts, ids, data)]).astype(np.float32)
        np.save(os.path.join(D, 'X_%s_%s.npy' % (c, k)), X); np.save(os.path.join(D, 'y_%s_%s.npy' % (c, k)), y)
        np.save(os.path.join(D, 'ts_%s_%s.npy' % (c, k)), ts)
        audit['%s_%s' % (c, k)] = {'frames': int(len(y)), 'injected': int(y.sum()), 'relabelled': fix}
        print(c, k, len(y), int(y.sum()), 'relabelled', fix, flush=True)
json.dump(audit, open(os.path.join(D, 'cache_audit.json'), 'w'), indent=1)
