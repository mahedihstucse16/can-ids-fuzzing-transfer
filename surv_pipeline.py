# -*- coding: utf-8 -*-
"""Second vehicle set: HCRL Survival Analysis (Sonata, Soul, Spark). Same protocol as rev3:
training-partition scaling, class weighting, random forest of 60 trees, ten draws (seeds 0-9).

SA-E2: per source vehicle, 60,000 training frames: flooding + malfunction (30,000 each) versus
       flooding + malfunction + fuzzing (20,000 each).
SA-E3: flooding + malfunction base (30,000 each) plus 0..20,000 frames of the fuzzing capture.
Each model is tested on both other vehicles: complete free-driving, flooding, malfunction and
fuzzing captures; detection (recall) and false alarms (FPR on legitimate frames) per capture.
Within-vehicle check: 50,000 held-out frames of the source's four attack captures.
Usage: python3 -I surv_pipeline.py <cache_dir> <out_dir> <stage>...   stages: e2 e3
"""
import gc, json, os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

D, OUT = sys.argv[1], sys.argv[2]
CK = os.path.join(OUT, 'checkpoints'); os.makedirs(CK, exist_ok=True)
CARS = ['sonata', 'soul', 'spark']
KINDS = ['free', 'flood', 'malf', 'fuzzy']
REPS = {'raw10': list(range(10)), 'eng8': list(range(10, 18))}
SEEDS = list(range(10))
DOSES = [0, 500, 1000, 2500, 5000, 10000, 20000]
DATA = {(c, k): (np.load('%s/X_%s_%s.npy' % (D, c, k)), np.load('%s/y_%s_%s.npy' % (D, c, k)))
        for c in CARS for k in KINDS}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def take(c, k, m, seed):
    X, y = DATA[(c, k)]
    ai, ni = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    mt = min(int(round(m * len(ai) / len(y))), len(ai)); mn = min(m - mt, len(ni))
    rs = np.random.RandomState(seed)
    return np.sort(np.concatenate([rs.choice(ai, mt, replace=False), rs.choice(ni, mn, replace=False)]))


def metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': 100 * pr, 'recall': 100 * rc,
            'f1': 100 * (2 * pr * rc / (pr + rc) if pr + rc else 0.0), 'fpr': 100 * fp / max(1, fp + tn)}


def unit(key, src, spec, rep, seed, extra):
    path = os.path.join(CK, key + '.json')
    if os.path.exists(path):
        return
    t0 = time.time()
    Xs, ys, used = [], [], {}
    for i, (k, m) in enumerate(sorted(spec.items())):
        if m <= 0:
            continue
        idx = take(src, k, m, seed * 1000 + 7 * i + 13)
        used[k] = idx
        Xs.append(DATA[(src, k)][0][idx]); ys.append(DATA[(src, k)][1][idx])
    X = np.concatenate(Xs)[:, REPS[rep]]; y = np.concatenate(ys)
    sc = StandardScaler().fit(X)
    mdl = RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=-1)
    mdl.fit(sc.transform(X), y, sample_weight=compute_sample_weight('balanced', y))
    P = lambda Z: mdl.predict(sc.transform(Z[:, REPS[rep]])).astype(np.int8)
    res = {'source': src, 'representation': rep, 'seed': seed, 'spec': spec, 'n_train': int(len(y)),
           'training_attack_ids': int(len(np.unique(np.concatenate(Xs)[np.concatenate(ys) == 1, 0]))), 'test': {}}
    res.update(extra)
    for t in CARS:
        if t == src:
            continue
        res['test'][t] = {k: metrics(DATA[(t, k)][1], P(DATA[(t, k)][0])) for k in KINDS}
    # within-vehicle: held-out frames of the source's attack captures not used in training
    rs = np.random.RandomState(50000 + seed)
    Xh, yh = [], []
    for k in ('flood', 'malf', 'fuzzy'):
        Xk, yk = DATA[(src, k)]
        free_idx = np.setdiff1d(np.arange(len(yk)), used.get(k, np.array([], int)))
        pick = rs.choice(free_idx, min(len(free_idx), 50000 // 3), replace=False)
        Xh.append(Xk[pick]); yh.append(yk[pick])
    res['within'] = metrics(np.concatenate(yh), P(np.concatenate(Xh)))
    res['seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
    s = ' | '.join('%s: fpr_free=%.2f flood=%.1f malf=%.1f fuzz=%.1f' % (
        t, r['free']['fpr'], r['flood']['recall'], r['malf']['recall'], r['fuzzy']['recall']) for t, r in res['test'].items())
    log('  %-44s within=%.2f  %s (%.0fs)' % (key, res['within']['f1'], s, res['seconds']))
    gc.collect()


def stage_e2():
    comps = {'clean': {'flood': 30000, 'malf': 30000}, 'fuzz': {'flood': 20000, 'malf': 20000, 'fuzzy': 20000}}
    for src in CARS:
        for comp, spec in comps.items():
            for rep in REPS:
                for s in SEEDS:
                    unit('SAE2__%s__%s__%s__seed%d' % (src, comp, rep, s), src, spec, rep, s, {'composition': comp})


def stage_e3():
    for src in CARS:
        for d in DOSES:
            spec = {'flood': 30000, 'malf': 30000, 'fuzzy': d}
            for rep in REPS:
                for s in SEEDS:
                    unit('SAE3__%s__dose%05d__%s__seed%d' % (src, d, rep, s), src, spec, rep, s, {'fuzzing_frames': d})


for st in sys.argv[3:]:
    {'e2': stage_e2, 'e3': stage_e3}[st]()
    log('STAGE', st, 'complete')
log('ALL DONE')
