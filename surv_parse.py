# -*- coding: utf-8 -*-
"""Parse the HCRL Survival Analysis captures into arrays (same 18 features as Car-Hacking).

Attack files: ts,ID,DLC,b0..b{DLC-1},flag  (flag R normal / T injected)
Free-driving: ts,ID,DLC,"b0 b1 ..."        (no flag; all normal)
Usage: python3 -I surv_parse.py <dataset_dir> <out_dir> [--audit]
"""
import os
import sys
import json
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
FILES = {
    'sonata': {'free': 'Sonata/FreeDrivingData_20180323_SONATA.txt', 'flood': 'Sonata/Flooding_dataset_SONATA.txt',
               'fuzzy': 'Sonata/Fuzzy_dataset_SONATA.txt', 'malf': 'Sonata/Malfunction_dataset_SONATA.txt'},
    'soul': {'free': 'Soul/FreeDrivingData_20180112_KIA.txt', 'flood': 'Soul/Flooding_dataset_KIA.txt',
             'fuzzy': 'Soul/Fuzzy_dataset_KIA.txt', 'malf': 'Soul/Malfunction153_dataset_KIA.txt'},
    'spark': {'free': 'Spark/FreeDrivingData_20171231_Spark.txt', 'flood': 'Spark/Flooding_dataset_Spark.txt',
              'fuzzy': 'Spark/Fuzzy_dataset_Spark.txt', 'malf': 'Spark/Malfunction18E_dataset_Spark.txt'},
}


def parse(path):
    ts, ids, dlc, data, lab = [], [], [], [], []
    bad = 0
    for ln in open(path, encoding='ascii', errors='replace'):
        p = [x.strip() for x in ln.strip().split(',')]
        if len(p) < 3:
            bad += 1
            continue
        try:
            t = float(p[0]); i = int(p[1], 16); d = int(p[2])
        except ValueError:
            bad += 1
            continue
        rest = p[3:]
        flag = 'R'
        if rest and rest[-1] in ('R', 'T'):
            flag = rest[-1]; rest = rest[:-1]
        if len(rest) == 1 and ' ' in rest[0]:
            rest = rest[0].split()
        b = [int(x, 16) for x in rest if x][:8]
        b += [0] * (8 - len(b))
        ts.append(t); ids.append(i); dlc.append(d); data.append(b); lab.append(1 if flag == 'T' else 0)
    return (np.array(ts), np.array(ids, np.int64), np.array(dlc, np.int64), np.array(data, np.int64),
            np.array(lab, np.int8), bad)


A = {}
for car, fs in FILES.items():
    for kind, rel in fs.items():
        ts, ids, dlc, data, y, bad = parse(os.path.join(SRC, rel))
        np.savez_compressed(os.path.join(OUT, 'raw_%s_%s.npz' % (car, kind)), ts=ts, ids=ids, dlc=dlc, data=data, y=y)
        nid = sorted(set(ids[y == 0].tolist())); aid = sorted(set(ids[y == 1].tolist()))
        A['%s_%s' % (car, kind)] = {'frames': int(len(y)), 'injected': int(y.sum()), 'bad_lines': bad,
                                     'normal_ids': nid, 'n_normal_ids': len(nid), 'n_attack_ids': len(aid),
                                     'attack_ids_head': [hex(v) for v in aid[:5]],
                                     'duration_s': float(ts[-1] - ts[0]), 'monotonic': bool(np.all(np.diff(ts) >= 0))}
json.dump(A, open(os.path.join(OUT, 'audit.json'), 'w'), indent=1)
free = {c: set(A['%s_free' % c]['normal_ids']) for c in FILES}
print('%-14s %8s %7s %4s %5s %6s %8s  overlap of normal IDs with free-driving vocab: sonata/soul/spark' % (
    'file', 'frames', 'inj', 'bad', 'nIDs', 'attIDs', 'dur_s'))
for k, a in A.items():
    s = set(a['normal_ids'])
    ov = ' '.join('%d/%d' % (len(s & free[c]), len(s)) for c in FILES)
    print('%-14s %8d %7d %4d %5d %6d %8.1f  %s %s %s' % (k, a['frames'], a['injected'], a['bad_lines'], a['n_normal_ids'],
                                                     a['n_attack_ids'], a['duration_s'], ov, a['attack_ids_head'], a['monotonic']))
