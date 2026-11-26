# -*- coding: utf-8 -*-
"""Export the Song et al. windows (29 consecutive identifiers) and the exact training / held-out
window indices of every draw, so the PyTorch DCNN runs on the same data as stage_songcnn.
Usage: python3 -I song_export.py <cache_dir> <out.npz>"""
import sys
import numpy as np
C, OUT = sys.argv[1], sys.argv[2]
W = 29
CH = ['DoS', 'Fuzzy', 'gear', 'RPM']
SPEC = {'flooding_gear_rpm': {'DoS': 40000, 'gear': 40000, 'RPM': 40000},
        'all_four_with_fuzzing': {k: 30000 for k in CH}}


def win(key):
    X = np.load('%s/X_%s.npy' % (C, key), mmap_mode='r'); y = np.load('%s/y_%s.npy' % (C, key))
    n = (len(y) // W) * W
    return (np.asarray(X[:n, 0]).astype(np.int32).reshape(-1, W).astype(np.int16),
            y[:n].reshape(-1, W).max(1).astype(np.int8))


out = {}
for k in CH:
    out['ids_%s' % k], out['y_%s' % k] = win('ch_' + k)
a, b = win('ot_dos'), win('ot_attack_free')
out['ids_test'] = np.concatenate([a[0], b[0]]); out['y_test'] = np.concatenate([a[1], b[1]])
for comp, spec in SPEC.items():
    for s in range(10):
        rs = np.random.RandomState(30000 + s)
        for k in CH:
            perm = rs.permutation(len(out['y_%s' % k]))
            m = int(round(spec.get(k, 0) / W))
            if m:
                out['tr_%s_%d_%s' % (comp, s, k)] = perm[:m].astype(np.int32)
                out['ho_%s_%d_%s' % (comp, s, k)] = perm[m:m + 2000].astype(np.int32)
np.savez_compressed(OUT, **out)
print('test windows', len(out['y_test']), 'attack', int(out['y_test'].sum()))
