# -*- coding: utf-8 -*-
"""Identifier census of the OTIDS test set against the Sonata vocabulary.
Usage: python3 -I rev3_vocab.py <cache_dir> <out.json>"""
import json, sys
import numpy as np
C, OUT = sys.argv[1], sys.argv[2]
L = lambda k: (np.load('%s/X_%s.npy' % (C, k), mmap_mode='r'), np.load('%s/y_%s.npy' % (C, k)))
son = set()
for k in ('DoS', 'Fuzzy', 'gear', 'RPM'):
    X, y = L('ch_' + k)
    son |= set(np.unique(np.asarray(X[:, 0])[y == 0].astype(np.int64)).tolist())
Xd, yd = L('ot_dos'); Xa, ya = L('ot_attack_free')
need = 2573117 - len(yd)
sel = np.sort(np.random.RandomState(42).choice(len(ya), size=need, replace=False))
ids = np.concatenate([np.asarray(Xd[:, 0]), np.asarray(Xa[:, 0])[sel]]).astype(np.int64)
y = np.concatenate([yd, ya[sel]])
u, c = np.unique(ids[y == 0], return_counts=True)
fam = np.array([int(v) in son for v in u])
out = {'sonata_normal_ids': len(son), 'otids_normal_ids': int(len(u)),
       'otids_ids_in_sonata_vocab': int(fam.sum()),
       'normal_frames': int(c.sum()), 'normal_frames_on_familiar_ids': int(c[fam].sum()),
       'per_id': [{'can_id': int(a), 'normal_frames': int(b), 'in_sonata_vocabulary': bool(f)}
                  for a, b, f in zip(u, c, fam)]}
json.dump(out, open(OUT, 'w'), indent=1)
print({k: v for k, v in out.items() if k != 'per_id'})
