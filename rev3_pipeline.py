# -*- coding: utf-8 -*-
"""Revision 3: every model-based result under one protocol.

Protocol (applied identically to every representation and every experiment)
  * features standardised with training-partition statistics (RF/trees are
    invariant to this; it matters for logistic regression and k-NN);
  * class weighting ('balanced') for every learner that accepts sample weights;
    k-NN cannot be weighted and is run unweighted for every representation;
  * no synthetic oversampling anywhere;
  * scikit-learn as installed (1.9.1) for every run;
  * repeated experiments use ten independent draws, seeds 0-9.

Sampling, OTIDS test-set construction and metrics are copied from the authors'
pipeline (validated bit-for-bit against the published seed-42 runs in
m4_matched_balancing.py), so only the preprocessing and the number of draws
differ from the submitted version.

Usage: python3 -I rev3_pipeline.py <cache_dir> <out_dir> <stage> [<stage> ...]
Stages: e2 e3 e4 e5 mech big hyper event e1cv e1split dedup explain
Every unit is check-pointed as JSON, so a stage can be re-run and resumes.
"""
import gc
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

CACHE, OUT = sys.argv[1], sys.argv[2]
STAGES = sys.argv[3:]
CK = os.path.join(OUT, 'checkpoints')
os.makedirs(CK, exist_ok=True)

# ---------------------------------------------------------------- constants
MASTER_SEED = 42
SEEDS = list(range(10))
E2_TOTAL = 120000
RF_TRANSFER_TREES = 60
RF_E1_TREES = 120
OTIDS_TEST_TOTAL = 2573117
CH_ORDER = ['DoS', 'Fuzzy', 'gear', 'RPM']
FEATS = ['CAN_ID', 'DLC'] + ['DATA%d' % i for i in range(8)] + [
    'IAT', 'IAT_ID', 'ID_Freq', 'Payload_Entropy', 'Payload_Hamming',
    'NonZero_Bytes', 'Data_Mean', 'Data_Std']
REPS = {'raw10': list(range(0, 10)), 'eng8': list(range(10, 18)),
        'all18': list(range(0, 18)), 'raw9': list(range(1, 10)), 'id_dlc': [0, 1],
        'eng_payload': [13, 14, 15, 16, 17], 'eng_timing': [10, 11, 12]}
E2_COMPOSITIONS = {
    'flooding_only': {'DoS': E2_TOTAL},
    'flooding_gear_rpm': {'DoS': E2_TOTAL // 3, 'gear': E2_TOTAL // 3, 'RPM': E2_TOTAL // 3},
    'all_four_with_fuzzing': {k: E2_TOTAL // 4 for k in CH_ORDER},
}
E3_BASE = {'DoS': 40000, 'gear': 40000, 'RPM': 40000}
E3_DOSES = [0, 2, 5, 10, 25, 50, 100]
E3_FUZZ_SUB = 40000
E1_ALLOC = {'DoS': 47408, 'Fuzzy': 49647, 'gear': 57462, 'RPM': 59771}
E1_CV_ALLOC = {'DoS': 11062, 'Fuzzy': 11584, 'gear': 13408, 'RPM': 13946}
E5_ALLOC = {'DoS': 33186, 'Fuzzy': 34753, 'gear': 40223, 'RPM': 41840}
SHARED_TAIL = 891068          # identical trailing normal frames in all four captures
DEDUP_KEEPER = 'RPM'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def ck_path(key):
    return os.path.join(CK, key + '.json')


def done(key):
    return os.path.exists(ck_path(key))


def save(key, obj):
    tmp = ck_path(key) + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, ck_path(key))


def load_ck(key):
    return json.load(open(ck_path(key)))


# ---------------------------------------------------------------- data
def load(key):
    return (np.load('%s/X_%s.npy' % (CACHE, key)),
            np.load('%s/y_%s.npy' % (CACHE, key)))


log('loading Car-Hacking captures')
CH = {k: load('ch_' + k) for k in CH_ORDER}
CH_FULL = dict(CH)            # untouched originals (dedup stage swaps CH)


def build_otids_test():
    Xd, yd = load('ot_dos')
    Xa, ya = load('ot_attack_free')
    nd, na = len(yd), len(ya)
    need = min(OTIDS_TEST_TOTAL - nd, na)
    rs = np.random.RandomState(MASTER_SEED)
    sel = np.sort(rs.choice(na, size=need, replace=False))
    X = np.concatenate([Xd, Xa[sel]], axis=0)
    y = np.concatenate([yd, ya[sel]], axis=0)
    calib = np.setdiff1d(np.arange(na), sel)      # attack-free frames never used in the test set
    return X, y, Xa[calib], nd


X_OT, y_OT, X_OT_CALIB, N_OT_DOS = build_otids_test()
log('OTIDS test: %d frames, %d attack' % (len(y_OT), int(y_OT.sum())))
SONATA_VOCAB = set(int(v) for k in CH_ORDER
                   for v in np.unique(CH[k][0][CH[k][1] == 0, 0].astype(np.int64)))


def stratified_take(key, m, seed):
    X, y = CH[key]
    ai = np.flatnonzero(y == 1)
    ni = np.flatnonzero(y == 0)
    n = len(y)
    mt = min(int(round(m * len(ai) / n)), len(ai))
    mn = min(m - mt, len(ni))
    rs = np.random.RandomState(seed)
    sel = np.concatenate([rs.choice(ai, mt, replace=False),
                          rs.choice(ni, mn, replace=False)])
    sel.sort()
    return X[sel], y[sel], sel


def build_training_set(spec, seed, with_source=False):
    Xs, ys, src = [], [], []
    for i, (k, m) in enumerate(sorted(spec.items())):
        if m <= 0:
            continue
        Xi, yi, _ = stratified_take(k, m, seed * 1000 + 7 * i + 13)
        Xs.append(Xi)
        ys.append(yi)
        src.append(np.full(len(yi), CH_ORDER.index(k), dtype=np.int8))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    s = np.concatenate(src, axis=0)
    rs = np.random.RandomState(seed)
    p = rs.permutation(len(y))
    if with_source:
        return X[p], y[p], s[p]
    return X[p], y[p]


def metrics(ytrue, ypred):
    ytrue = np.asarray(ytrue).astype(np.int8)
    ypred = np.asarray(ypred).astype(np.int8)
    tp = int(np.sum((ytrue == 1) & (ypred == 1)))
    fp = int(np.sum((ytrue == 0) & (ypred == 1)))
    fn = int(np.sum((ytrue == 1) & (ypred == 0)))
    tn = int(np.sum((ytrue == 0) & (ypred == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'precision': 100 * prec, 'recall': 100 * rec, 'f1': 100 * f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n_test': tp + fp + fn + tn}


# ---------------------------------------------------------------- protocol
def make_model(name, seed, trees=None, depth=None):
    if name == 'LogisticRegression':
        return LogisticRegression(C=1.0, max_iter=2000)
    if name == 'KNeighbours':
        return KNeighborsClassifier(n_neighbors=5, algorithm='brute', n_jobs=-1)
    if name == 'DecisionTree':
        return DecisionTreeClassifier(max_depth=12, random_state=seed)
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=trees or RF_E1_TREES, max_depth=depth,
                                      random_state=seed, n_jobs=-1)
    if name == 'AdaBoost':
        return AdaBoostClassifier(random_state=seed)
    if name == 'GradientBoosting':
        return GradientBoostingClassifier(random_state=seed)
    if name == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(random_state=seed)
    if name == 'Stacking':
        return StackingClassifier(
            estimators=[('dt', DecisionTreeClassifier(max_depth=12, random_state=seed)),
                        ('rf', RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=-1)),
                        ('hgb', HistGradientBoostingClassifier(random_state=seed))],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000), cv=5, n_jobs=1)
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, Xtes, seed, trees=None, depth=None, cols=None):
    """Scale with training statistics, weight classes, fit, predict each test set."""
    cols = cols if cols is not None else slice(None)
    Xtr = np.asarray(Xtr)[:, cols].astype(np.float32)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    mdl = make_model(name, seed, trees=trees, depth=depth)
    t0 = time.time()
    if name == 'KNeighbours':
        mdl.fit(Xtr, ytr)
    else:
        mdl.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
    tfit = time.time() - t0
    preds = []
    for Xte in Xtes:
        out = np.empty(len(Xte), dtype=np.int8)
        for a in range(0, len(Xte), 500000):
            blk = sc.transform(np.asarray(Xte[a:a + 500000])[:, cols].astype(np.float32))
            out[a:a + 500000] = mdl.predict(blk.astype(np.float32))
        preds.append(out)
    return preds, mdl, sc, tfit


def n_attack_ids(X, y):
    return int(len(np.unique(X[y.astype(bool), 0].astype(np.int64))))


def fp_by_id(pred, X, y):
    ids = X[:, 0].astype(np.int64)
    m = (y == 0) & (pred == 1)
    u, c = np.unique(ids[m], return_counts=True)
    un, cn = np.unique(ids[y == 0], return_counts=True)
    tot = dict(zip(un.tolist(), cn.tolist()))
    return [{'can_id': int(a), 'false_positives': int(b), 'normal_frames': int(tot[int(a)]),
             'in_sonata_vocabulary': int(a) in SONATA_VOCAB} for a, b in zip(u, c)]


def transfer_unit(key, spec, rep, seed, name='RandomForest', trees=RF_TRANSFER_TREES,
                  depth=None, extra=None, record_fp=False, Xtr_ytr=None):
    if done(key):
        return load_ck(key)
    t0 = time.time()
    Xtr, ytr = Xtr_ytr if Xtr_ytr is not None else build_training_set(spec, seed)
    nid = n_attack_ids(Xtr, ytr)
    (pred,), _, _, tfit = fit_predict(name, Xtr, ytr, [X_OT], seed, trees=trees,
                                     depth=depth, cols=REPS[rep])
    m = metrics(y_OT, pred)
    m.update({'model': name, 'representation': rep, 'seed': seed, 'trees': trees,
              'depth': depth, 'n_train': int(len(ytr)), 'training_attack_ids': nid,
              'spec': spec, 'fit_seconds': round(tfit, 1),
              'seconds': round(time.time() - t0, 1)})
    if extra:
        m.update(extra)
    if record_fp:
        m['fp_by_id'] = fp_by_id(pred, X_OT, y_OT)
    save(key, m)
    log('  %-62s F1=%6.2f prec=%6.2f fp=%8d (%.0fs)' % (key, m['f1'], m['precision'],
                                                     m['fp'], m['seconds']))
    gc.collect()
    return m


# ---------------------------------------------------------------- stages
def stage_e2():
    log('E2 composition ablation: 3 x {raw10, eng8, all18} x 10 draws')
    for comp, spec in E2_COMPOSITIONS.items():
        for rep in ('raw10', 'eng8', 'all18'):
            for s in SEEDS:
                transfer_unit('E2__%s__%s__seed%d' % (comp, rep, s), spec, rep, s,
                              record_fp=(rep == 'raw10'), extra={'composition': comp})


def stage_e3():
    log('E3 dose-response: 7 doses x {raw10, eng8} x 10 draws')
    for d in E3_DOSES:
        spec = dict(E3_BASE)
        if d:
            spec['Fuzzy'] = int(round(E3_FUZZ_SUB * d / 100))
        for rep in ('raw10', 'eng8'):
            for s in SEEDS:
                transfer_unit('E3__dose%03d__%s__seed%d' % (d, rep, s), spec, rep, s,
                              extra={'dose_pct': d, 'fuzzing_frames': spec.get('Fuzzy', 0)})


def stage_e4():
    log('E4 reverse transfer: OTIDS flooding -> Car-Hacking flooding, 10 draws')
    Xd, yd = load('ot_dos')
    Xt, yt = CH['DoS']
    ot_vocab = set(int(v) for v in np.unique(Xd[yd == 0, 0].astype(np.int64)))
    for rep in ('raw10', 'eng8'):
        for s in SEEDS:
            key = 'E4__%s__seed%d' % (rep, s)
            if done(key):
                continue
            t0 = time.time()
            ai, ni = np.flatnonzero(yd == 1), np.flatnonzero(yd == 0)
            mt = int(round(E2_TOTAL * len(ai) / len(yd)))
            rs = np.random.RandomState(s * 1000 + 13)
            sel = np.sort(np.concatenate([rs.choice(ai, mt, replace=False),
                                          rs.choice(ni, E2_TOTAL - mt, replace=False)]))
            (pred,), _, _, tfit = fit_predict('RandomForest', Xd[sel], yd[sel], [Xt], s,
                                             trees=RF_TRANSFER_TREES, cols=REPS[rep])
            m = metrics(yt, pred)
            ids = Xt[:, 0].astype(np.int64)
            fpm = (yt == 0) & (pred == 1)
            u, c = np.unique(ids[fpm], return_counts=True)
            m.update({'representation': rep, 'seed': s, 'n_train': E2_TOTAL,
                      'fp_by_id': [{'can_id': int(a), 'false_positives': int(b),
                                    'familiar_to_otids': int(a) in ot_vocab}
                                   for a, b in zip(u, c)],
                      'test_normal_ids': int(len(np.unique(ids[yt == 0]))),
                      'test_ids_unfamiliar': sorted(int(v) for v in np.unique(ids[yt == 0])
                                                    if int(v) not in ot_vocab),
                      'seconds': round(time.time() - t0, 1)})
            save(key, m)
            log('  %-40s F1=%6.2f prec=%6.2f fp=%8d' % (key, m['f1'], m['precision'], m['fp']))


def stage_e5():
    log('E5 classifier generality: 4 models x 3 reps x 10 draws, 150,002 frames')
    for name in ('LogisticRegression', 'DecisionTree', 'RandomForest', 'HistGradientBoosting'):
        for rep in ('raw10', 'all18', 'eng8'):
            for s in SEEDS:
                transfer_unit('E5__%s__%s__seed%d' % (name, rep, s), E5_ALLOC, rep, s,
                              name=name, trees=RF_E1_TREES if name == 'RandomForest' else None,
                              record_fp=(rep == 'raw10'))


def stage_mech():
    """Task 23: is it the identifier or the payload of the fuzzing frames?"""
    log('Mechanism ablation on the fuzzing-included composition, 10 draws')
    spec = E2_COMPOSITIONS['all_four_with_fuzzing']
    Xf, yf = CH['Fuzzy']
    nf = np.flatnonzero(yf == 0)
    fi = CH_ORDER.index('Fuzzy')
    for s in SEEDS:
        X, y, src = build_training_set(spec, s, with_source=True)
        fz = np.flatnonzero((src == fi) & (y == 1))
        rs = np.random.RandomState(10000 + s)
        donors = nf[rs.choice(len(nf), size=len(fz), replace=True)]
        # (a) fuzzing frames keep their random payloads but carry a legitimate
        #     Sonata identifier, drawn with the frequency of normal traffic
        Xa = X.copy()
        Xa[fz, 0] = Xf[donors, 0]
        # (b) fuzzing frames keep their random identifiers but carry the DLC and
        #     payload of real Sonata frames
        Xb = X.copy()
        Xb[fz, 1:10] = Xf[donors, 1:10]
        transfer_unit('MECH__baseline_raw10__seed%d' % s, spec, 'raw10', s,
                      Xtr_ytr=(X, y), record_fp=True, extra={'variant': 'baseline'})
        transfer_unit('MECH__legit_ids_random_payloads__seed%d' % s, spec, 'raw10', s,
                      Xtr_ytr=(Xa, y), record_fp=True,
                      extra={'variant': 'fuzzing frames with legitimate identifiers'})
        transfer_unit('MECH__random_ids_real_payloads__seed%d' % s, spec, 'raw10', s,
                      Xtr_ytr=(Xb, y), record_fp=True,
                      extra={'variant': 'fuzzing frames with real payloads'})
        transfer_unit('MECH__raw9_no_identifier__seed%d' % s, spec, 'raw9', s,
                      Xtr_ytr=(X, y), extra={'variant': 'raw features without CAN_ID'})
        del X, y, src, Xa, Xb
        gc.collect()


def stage_mech2():
    """Controls for the mechanism ablation: the reduced representations with and
    without fuzzing in training, so each attribution has its own baseline."""
    log('Mechanism controls: raw 9 and identifier+DLC, with and without fuzzing, 10 draws')
    for comp in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
        for rep in ('raw9', 'id_dlc'):
            for s in SEEDS:
                transfer_unit('MECH2__%s__%s__seed%d' % (comp, rep, s), E2_COMPOSITIONS[comp],
                              rep, s, record_fp=True, extra={'composition': comp})


def stage_big():
    """Task 29: does the collapse survive a much larger training set?"""
    log('Large training sets (about 1,000,000 frames), 3 draws')
    specs = {'flooding_gear_rpm': {'DoS': 333334, 'gear': 333333, 'RPM': 333333},
             'all_four_with_fuzzing': {k: 250000 for k in CH_ORDER}}
    for comp, spec in specs.items():
        for rep in ('raw10', 'eng8'):
            for s in (0, 1, 2):
                transfer_unit('BIG__%s__%s__seed%d' % (comp, rep, s), spec, rep, s,
                              extra={'composition': comp})


def stage_hyper():
    """Task 30: hyper-parameter sensitivity of the key contrast."""
    log('Hyper-parameter sensitivity, 3 draws')
    configs = [(60, None), (120, None), (300, None), (60, 8), (60, 16)]
    for comp in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
        for rep in ('raw10', 'eng8'):
            for trees, depth in configs:
                for s in (0, 1, 2):
                    transfer_unit('HYP__%s__%s__t%d_d%s__seed%d'
                                  % (comp, rep, trees, depth or 'none', s),
                                  E2_COMPOSITIONS[comp], rep, s, trees=trees, depth=depth,
                                  extra={'composition': comp})


def stage_event():
    """Task 31: event-level false alarms and detection delay on full OTIDS captures
    in time order, for the fuzzing-included composition."""
    log('Event-level alarms on the full OTIDS captures, 10 draws')
    Xa, _ = load('ot_attack_free')
    ta = np.load('%s/ts_ot_attack_free.npy' % CACHE)
    Xd, yd = load('ot_dos')
    td = np.load('%s/ts_ot_dos.npy' % CACHE)
    onset = td[np.argmax(yd == 1)]
    spec = E2_COMPOSITIONS['all_four_with_fuzzing']
    windows = (0.1, 1.0)
    taus = (0.01, 0.05, 0.10, 0.20, 0.50)

    def windowed(t, pred, w):
        b = np.floor((t - t[0]) / w).astype(np.int64)
        nb = b[-1] + 1
        cnt = np.bincount(b, minlength=nb)
        flg = np.bincount(b, weights=pred, minlength=nb)
        frac = np.where(cnt > 0, flg / np.maximum(cnt, 1), 0.0)
        return frac, t[0] + (np.arange(nb) + 1) * w

    for rep in ('eng8', 'raw10'):
        for s in SEEDS:
            key = 'EVENT__%s__seed%d' % (rep, s)
            if done(key):
                continue
            Xtr, ytr = build_training_set(spec, s)
            (pa, pd), _, _, _ = fit_predict('RandomForest', Xtr, ytr, [Xa, Xd], s,
                                            trees=RF_TRANSFER_TREES, cols=REPS[rep])
            hours = (ta[-1] - ta[0]) / 3600.0
            res = {'representation': rep, 'seed': s,
                   'frame_fpr_attack_free_pct': 100 * float(pa.mean()),
                   'attack_free_seconds': float(ta[-1] - ta[0]),
                   'frames_per_second_attack_free': float(len(ta) / (ta[-1] - ta[0])),
                   'dos_recall_pct': 100 * float(pd[yd == 1].mean()), 'rules': []}
            for w in windows:
                fa, _ = windowed(ta, pa.astype(float), w)
                fd, ed = windowed(td, pd.astype(float), w)
                for tau in taus:
                    al = fa > tau
                    # merge consecutive alarmed windows into one event
                    events = int(np.sum(al & ~np.r_[False, al[:-1]]))
                    ald = fd > tau
                    after = np.flatnonzero(ald & (ed > onset))
                    delay = float(ed[after[0]] - onset) if len(after) else None
                    res['rules'].append({'window_s': w, 'tau': tau,
                                         'alarmed_windows_pct': 100 * float(al.mean()),
                                         'events': events,
                                         'events_per_hour': events / hours,
                                         'detection_delay_s': delay})
            save(key, res)
            r1 = [r for r in res['rules'] if r['window_s'] == 1.0 and r['tau'] == 0.10][0]
            log('  %-24s frame FPR %.3f%%  1s/10%%: %d events (%.1f/h) delay %s'
                % (key, res['frame_fpr_attack_free_pct'], r1['events'],
                   r1['events_per_hour'], r1['detection_delay_s']))
            gc.collect()


def e1_pool(alloc, seed):
    """Proportional within-vehicle pool, keeping capture identity and time order."""
    Xs, ys, src, order = [], [], [], []
    for i, k in enumerate(CH_ORDER):
        Xi, yi, sel = stratified_take(k, alloc[k], seed * 1000 + 7 * i + 13)
        Xs.append(Xi)
        ys.append(yi)
        src.append(np.full(len(yi), i, dtype=np.int8))
        order.append(sel / float(len(CH[k][1])))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(src),
            np.concatenate(order))


E1_MODELS_CV = ['LogisticRegression', 'KNeighbours', 'DecisionTree', 'RandomForest',
                'AdaBoost', 'HistGradientBoosting']
E1_MODELS_SPLIT = E1_MODELS_CV[:5] + ['GradientBoosting', 'HistGradientBoosting', 'Stacking']


def stage_e1cv():
    log('E1 within-vehicle 5-fold CV: 6 models x {raw10, eng8, all18}')
    X, y, _, _ = e1_pool(E1_CV_ALLOC, MASTER_SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=MASTER_SEED)
    folds = list(skf.split(X, y))
    for name in E1_MODELS_CV:
        for rep in ('raw10', 'eng8', 'all18'):
            key = 'E1CV__%s__%s' % (rep, name)
            if done(key):
                continue
            res = []
            for f, (tr, te) in enumerate(folds):
                (p,), _, _, _ = fit_predict(name, X[tr], y[tr], [X[te]], MASTER_SEED,
                                            cols=REPS[rep])
                m = metrics(y[te], p)
                m['fold'] = f
                res.append(m)
            f1 = [r['f1'] for r in res]
            save(key, {'model': name, 'representation': rep, 'folds': res,
                       'f1_mean': float(np.mean(f1)), 'f1_std': float(np.std(f1, ddof=1))})
            log('  %-44s F1 %.2f +/- %.2f' % (key, np.mean(f1), np.std(f1, ddof=1)))


def e1_splits(X, y, src, order):
    idx = np.arange(len(y))
    tr_r, te_r = train_test_split(idx, test_size=0.3, stratify=y, random_state=MASTER_SEED)
    tr_c, te_c = [], []
    for i in range(len(CH_ORDER)):
        ii = idx[src == i]
        ii = ii[np.argsort(order[ii], kind='stable')]
        cut = int(round(0.7 * len(ii)))
        tr_c.append(ii[:cut])
        te_c.append(ii[cut:])
    tr_c, te_c = np.concatenate(tr_c), np.concatenate(te_c)
    target = y.mean()
    ta = te_c[y[te_c] == 1]
    tn = te_c[y[te_c] == 0]
    keep_n = int(round(len(ta) * (1 - target) / target))
    rs = np.random.RandomState(MASTER_SEED)
    te_p = np.sort(np.concatenate([ta, rs.choice(tn, min(keep_n, len(tn)), replace=False)]))
    return {'random': (tr_r, te_r), 'chronological': (tr_c, te_c),
            'prevalence_matched': (tr_c, te_p)}


def stage_e1split():
    log('E1 within-vehicle splits: 8 models x {raw10, eng8, all18} x 3 splits')
    X, y, src, order = e1_pool(E1_ALLOC, MASTER_SEED)
    sp = e1_splits(X, y, src, order)
    save('E1SPLIT__prevalence', {k: {'n_train': int(len(a)), 'n_test': int(len(b)),
                                     'train_attack_pct': 100 * float(y[a].mean()),
                                     'test_attack_pct': 100 * float(y[b].mean())}
                                 for k, (a, b) in sp.items()})
    for name in E1_MODELS_SPLIT:
        for rep in ('raw10', 'eng8', 'all18'):
            todo = [k for k in sp if not done('E1SPLIT__%s__%s__%s' % (k, rep, name))]
            if not todo:
                continue
            # random split trains on its own partition; chronological and
            # prevalence-matched share one training partition and one model
            if 'random' in todo:
                tr, te = sp['random']
                (p,), _, _, tf = fit_predict(name, X[tr], y[tr], [X[te]], MASTER_SEED,
                                             cols=REPS[rep])
                m = metrics(y[te], p)
                m.update({'model': name, 'representation': rep, 'split': 'random',
                          'fit_seconds': round(tf, 1)})
                save('E1SPLIT__random__%s__%s' % (rep, name), m)
                log('  %-56s F1 %.2f' % ('E1SPLIT__random__%s__%s' % (rep, name), m['f1']))
            if 'chronological' in todo or 'prevalence_matched' in todo:
                tr, te = sp['chronological']
                _, te_p = sp['prevalence_matched']
                (pc, pp), _, _, tf = fit_predict(name, X[tr], y[tr], [X[te], X[te_p]],
                                                 MASTER_SEED, cols=REPS[rep])
                for k, tt, p in (('chronological', te, pc), ('prevalence_matched', te_p, pp)):
                    m = metrics(y[tt], p)
                    m.update({'model': name, 'representation': rep, 'split': k,
                              'fit_seconds': round(tf, 1)})
                    save('E1SPLIT__%s__%s__%s' % (k, rep, name), m)
                    log('  %-56s F1 %.2f' % ('E1SPLIT__%s__%s__%s' % (k, rep, name), m['f1']))
            gc.collect()


def stage_dedup():
    """Task 15: the shared 891,068-frame tail counted once (kept in RPM only)."""
    global CH
    log('De-duplication sensitivity')
    CH = {k: ((v[0][:-SHARED_TAIL], v[1][:-SHARED_TAIL]) if k != DEDUP_KEEPER else v)
          for k, v in CH_FULL.items()}
    try:
        X, y, src, order = e1_pool(E1_ALLOC, MASTER_SEED)
        tr, te = e1_splits(X, y, src, order)['random']
        for name in E1_MODELS_CV:
            for rep in ('raw10', 'eng8', 'all18'):
                key = 'DEDUP__E1random__%s__%s' % (rep, name)
                if done(key):
                    continue
                (p,), _, _, _ = fit_predict(name, X[tr], y[tr], [X[te]], MASTER_SEED,
                                            cols=REPS[rep])
                m = metrics(y[te], p)
                m.update({'model': name, 'representation': rep})
                save(key, m)
                log('  %-50s F1 %.2f' % (key, m['f1']))
        for comp, spec in E2_COMPOSITIONS.items():
            for rep in ('raw10', 'eng8'):
                for s in SEEDS:
                    transfer_unit('DEDUP__E2__%s__%s__seed%d' % (comp, rep, s), spec, rep, s,
                                  extra={'composition': comp, 'dedup': True})
    finally:
        CH = CH_FULL


def stage_explain():
    """Within-vehicle all-18 random forest, Monte Carlo Shapley (permutation
    sampling with one background frame per permutation), impurity and
    permutation importance."""
    log('Explainability on the within-vehicle all-18 random forest')
    if done('EXPLAIN__shapley'):
        return
    from sklearn.inspection import permutation_importance
    X, y, src, order = e1_pool(E1_ALLOC, MASTER_SEED)
    tr, te = e1_splits(X, y, src, order)['random']
    Xtr = X[tr].astype(np.float64)
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr).astype(np.float32)
    mdl = RandomForestClassifier(n_estimators=RF_E1_TREES, random_state=MASTER_SEED, n_jobs=-1)
    mdl.fit(Ztr, y[tr], sample_weight=compute_sample_weight('balanced', y[tr]))
    Zte = sc.transform(X[te].astype(np.float64)).astype(np.float32)
    f = lambda Z: mdl.predict_proba(Z)[:, 1]
    rs = np.random.RandomState(MASTER_SEED)
    # 250 held-out frames, class balanced; 200 class-balanced real training frames
    ate, nte = te[y[te] == 1], te[y[te] == 0]
    expl = np.concatenate([rs.choice(ate, 125, replace=False), rs.choice(nte, 125, replace=False)])
    atr, ntr = np.flatnonzero(y[tr] == 1), np.flatnonzero(y[tr] == 0)
    bg = np.concatenate([rs.choice(atr, 100, replace=False), rs.choice(ntr, 100, replace=False)])
    B = Ztr[bg]
    Zx = sc.transform(X[expl].astype(np.float64)).astype(np.float32)
    d = Zx.shape[1]
    npm = len(B)
    phi = np.zeros((len(expl), d))
    perms = [rs.permutation(d) for _ in range(npm)]
    for j in range(len(expl)):
        x = Zx[j]
        rows = np.empty((npm * (d + 1), d), dtype=np.float32)
        for p in range(npm):
            z = B[p].copy()
            rows[p * (d + 1)] = z
            for q, feat in enumerate(perms[p]):
                z[feat] = x[feat]
                rows[p * (d + 1) + q + 1] = z
        pr = f(rows).reshape(npm, d + 1)
        for p in range(npm):
            phi[j, perms[p]] += np.diff(pr[p])
        phi[j] /= npm
    base = float(f(B).mean())
    fx = f(Zx)
    pred = (fx >= 0.5).astype(int)
    # medoid pick per group among correctly classified frames
    grp = np.where(y[expl] == 0, 'normal', np.array(CH_ORDER)[src[expl]])
    picks = {}
    for g in ['DoS', 'Fuzzy', 'gear', 'RPM', 'normal']:
        ii = np.flatnonzero((grp == g) & (pred == y[expl]))
        if len(ii):
            mu = phi[ii].mean(0)
            picks[g] = int(ii[np.argmin(((phi[ii] - mu) ** 2).sum(1))])
    pim = permutation_importance(mdl, Zte, y[te], scoring='f1', n_repeats=10,
                                 random_state=MASTER_SEED, n_jobs=1)
    save('EXPLAIN__shapley', {
        'features': FEATS, 'base_value': base, 'n_frames': int(len(expl)),
        'n_permutations': npm, 'phi': phi.tolist(), 'fx': fx.tolist(),
        'true': y[expl].astype(int).tolist(), 'group': grp.tolist(),
        'values': X[expl].astype(float).tolist(), 'picks': picks,
        'additivity_max_error': float(np.max(np.abs(phi.sum(1) - (fx - base)))),
        'impurity_importance': mdl.feature_importances_.tolist(),
        'permutation_importance_mean': pim.importances_mean.tolist(),
        'permutation_importance_std': pim.importances_std.tolist(),
        'within_vehicle_test_f1': metrics(y[te], mdl.predict(Zte))['f1']})
    log('  Shapley done; additivity error %.2e' % float(np.max(np.abs(phi.sum(1) - (fx - base)))))


def stage_engabl():
    """Review: is engineered-feature robustness only flood detectability?"""
    log('Engineered-feature ablation: payload-dynamics only and timing/load only, 10 draws')
    for comp, spec in E2_COMPOSITIONS.items():
        for rep in ('eng_payload', 'eng_timing'):
            for s in SEEDS:
                transfer_unit('ABL__%s__%s__seed%d' % (comp, rep, s), spec, rep, s,
                              extra={'composition': comp})


def stage_baselines():
    """Review: tuned, neural, benign-only and target-calibrated baselines (fuzzing-included
    composition unless stated, 10 draws)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPClassifier
    log('Baselines')
    spec_f = E2_COMPOSITIONS['all_four_with_fuzzing']
    spec_c = E2_COMPOSITIONS['flooding_gear_rpm']
    for s in SEEDS:
        # (1) random forest tuned on the source vehicle by 3-fold CV (F1)
        for comp, spec in (('flooding_gear_rpm', spec_c), ('all_four_with_fuzzing', spec_f)):
            key = 'BASE__tunedrf__%s__raw10__seed%d' % (comp, s)
            if not done(key):
                t0 = time.time()
                Xtr, ytr = build_training_set(spec, s)
                Z = Xtr[:, REPS['raw10']].astype(np.float32)
                sc = StandardScaler().fit(Z)
                gs = GridSearchCV(RandomForestClassifier(random_state=s, n_jobs=-1, class_weight='balanced'),
                                  {'n_estimators': [60, 200], 'max_depth': [None, 16, 8],
                                   'min_samples_leaf': [1, 5], 'max_features': ['sqrt', 0.5]},
                                  scoring='f1', cv=3, n_jobs=1)
                gs.fit(sc.transform(Z), ytr)
                pred = np.empty(len(y_OT), dtype=np.int8)
                for a in range(0, len(y_OT), 500000):
                    pred[a:a + 500000] = gs.predict(sc.transform(X_OT[a:a + 500000][:, REPS['raw10']].astype(np.float32)))
                m = metrics(y_OT, pred)
                m.update({'model': 'tuned RF', 'representation': 'raw10', 'composition': comp, 'seed': s,
                          'best_params': {k: (v if v is not None else 'None') for k, v in gs.best_params_.items()},
                          'cv_f1': float(gs.best_score_), 'seconds': round(time.time() - t0, 1)})
                save(key, m)
                log('  %-56s F1=%6.2f cvF1=%.4f %s' % (key, m['f1'], m['cv_f1'], m['best_params']))
        # (2) multilayer perceptron, both representations
        for rep in ('raw10', 'eng8'):
            for comp, spec in (('flooding_gear_rpm', spec_c), ('all_four_with_fuzzing', spec_f)):
                key = 'BASE__mlp__%s__%s__seed%d' % (comp, rep, s)
                if not done(key):
                    Xtr, ytr = build_training_set(spec, s)
                    Z = Xtr[:, REPS[rep]].astype(np.float32)
                    sc = StandardScaler().fit(Z)
                    mdl = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200,
                                        random_state=s)
                    mdl.fit(sc.transform(Z), ytr, sample_weight=compute_sample_weight('balanced', ytr))
                    pred = mdl.predict(sc.transform(X_OT[:, REPS[rep]].astype(np.float32))).astype(np.int8)
                    m = metrics(y_OT, pred)
                    m.update({'model': 'MLP', 'representation': rep, 'composition': comp, 'seed': s})
                    save(key, m)
                    log('  %-56s F1=%6.2f prec=%6.2f' % (key, m['f1'], m['precision']))
        # (3) benign-only isolation forest: trained on legitimate Sonata frames only,
        #     threshold set for 1% false alarms on held-out legitimate Sonata frames
        for rep in ('raw10', 'eng8'):
            key = 'BASE__iforest__benign__%s__seed%d' % (rep, s)
            if not done(key):
                Xtr, ytr = build_training_set(spec_f, s)
                Xn = Xtr[ytr == 0][:, REPS[rep]].astype(np.float32)
                rs = np.random.RandomState(s)
                p = rs.permutation(len(Xn))
                fit_i, val_i = p[:int(0.8 * len(p))], p[int(0.8 * len(p)):]
                sc = StandardScaler().fit(Xn[fit_i])
                mdl = IsolationForest(n_estimators=200, random_state=s, n_jobs=-1).fit(sc.transform(Xn[fit_i]))
                thr = np.quantile(-mdl.score_samples(sc.transform(Xn[val_i])), 0.99)
                score = -mdl.score_samples(sc.transform(X_OT[:, REPS[rep]].astype(np.float32)))
                m = metrics(y_OT, (score > thr).astype(np.int8))
                m.update({'model': 'Isolation forest (benign only)', 'representation': rep, 'seed': s})
                save(key, m)
                log('  %-56s F1=%6.2f prec=%6.2f rec=%6.2f' % (key, m['f1'], m['precision'], m['recall']))
        # (4) target-vehicle calibration: the fuzzing-included forest's decision threshold is
        #     set for at most 1% false alarms on KIA Soul attack-free frames outside the test set
        for rep in ('raw10', 'eng8'):
            key = 'BASE__calib__all_four_with_fuzzing__%s__seed%d' % (rep, s)
            if not done(key):
                Xtr, ytr = build_training_set(spec_f, s)
                Z = Xtr[:, REPS[rep]].astype(np.float32)
                sc = StandardScaler().fit(Z)
                mdl = RandomForestClassifier(n_estimators=RF_TRANSFER_TREES, random_state=s, n_jobs=-1)
                mdl.fit(sc.transform(Z), ytr, sample_weight=compute_sample_weight('balanced', ytr))
                pc = mdl.predict_proba(sc.transform(X_OT_CALIB[:, REPS[rep]].astype(np.float32)))[:, 1]
                thr = np.quantile(pc, 0.99)
                pt = mdl.predict_proba(sc.transform(X_OT[:, REPS[rep]].astype(np.float32)))[:, 1]
                m = metrics(y_OT, (pt > thr).astype(np.int8))
                m.update({'model': 'RF + target calibration', 'representation': rep, 'seed': s,
                          'threshold': float(thr), 'n_calibration_frames': int(len(X_OT_CALIB))})
                save(key, m)
                log('  %-56s F1=%6.2f prec=%6.2f rec=%6.2f thr=%.3f' % (key, m['f1'], m['precision'], m['recall'], thr))
        gc.collect()


def stage_boot():
    """Review: uncertainty over the test traffic, not only over draws. Block bootstrap of
    contiguous 10,000-frame blocks of the test set, resampled within the DoS and attack-free
    parts, applied to the paired clean-minus-fuzzing F1 difference averaged over draws."""
    key = 'BOOT__headline'
    if done(key):
        return
    log('Block bootstrap over test-set time blocks')
    preds = {}
    for comp in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
        for rep in ('raw10', 'eng8'):
            P = []
            for s in SEEDS:
                Xtr, ytr = build_training_set(E2_COMPOSITIONS[comp], s)
                (p,), _, _, _ = fit_predict('RandomForest', Xtr, ytr, [X_OT], s, trees=RF_TRANSFER_TREES,
                                            cols=REPS[rep])
                P.append(p)
            preds[(comp, rep)] = np.array(P)
    B = 10000
    n = len(y_OT)
    parts = [(0, N_OT_DOS), (N_OT_DOS, n)]
    blocks = [np.arange(a, min(a + B, hi)) for lo, hi in parts for a in range(lo, hi, B)]
    nb = [sum(1 for b in blocks if b[0] < N_OT_DOS), sum(1 for b in blocks if b[0] >= N_OT_DOS)]

    def counts(pred, idx):
        yt = y_OT[idx]
        pp = pred[:, idx]
        tp = ((pp == 1) & (yt == 1)).sum(1)
        fp = ((pp == 1) & (yt == 0)).sum(1)
        fn = ((pp == 0) & (yt == 1)).sum(1)
        return tp, fp, fn
    # per-block counts, so resampling is cheap
    cb = {k: np.array([counts(v, b) for b in blocks]) for k, v in preds.items()}   # blocks x 3 x seeds

    def f1(c):
        tp, fp, fn = c[0], c[1], c[2]
        return 100 * 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    rs = np.random.RandomState(MASTER_SEED)
    res = {k: [] for k in ('raw_drop', 'eng_change', 'raw_fuzz', 'eng_fuzz')}
    for _ in range(2000):
        pick = np.concatenate([rs.choice(nb[0], nb[0]), nb[0] + rs.choice(nb[1], nb[1])])
        tot = {k: v[pick].sum(0) for k, v in cb.items()}
        F = {k: f1(v).mean() for k, v in tot.items()}
        res['raw_drop'].append(F[('flooding_gear_rpm', 'raw10')] - F[('all_four_with_fuzzing', 'raw10')])
        res['eng_change'].append(F[('all_four_with_fuzzing', 'eng8')] - F[('flooding_gear_rpm', 'eng8')])
        res['raw_fuzz'].append(F[('all_four_with_fuzzing', 'raw10')])
        res['eng_fuzz'].append(F[('all_four_with_fuzzing', 'eng8')])
    out = {k: {'mean': float(np.mean(v)), 'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
           for k, v in res.items()}
    out.update({'block_frames': B, 'n_blocks': [int(x) for x in nb], 'replicates': 2000})
    save(key, out)
    log('  bootstrap: %s' % json.dumps({k: out[k] for k in res}))


def stage_calib2():
    """Review: per-vehicle calibration, interleaved and temporally separated, ten draws.
    Interleaved: threshold from the 341,187 attack-free frames not sampled into the test set.
    Temporal: threshold from the first 20% of the attack-free capture by timestamp; tested on
    the complete flooding capture plus the remaining 80% of the attack-free capture."""
    log('Calibration baselines: interleaved and temporally separated, 10 draws')
    Xa, _ = load('ot_attack_free')
    Xd, yd = load('ot_dos')
    cut = int(round(0.2 * len(Xa)))
    X_T = np.concatenate([Xd, Xa[cut:]])
    y_T = np.concatenate([yd, np.zeros(len(Xa) - cut, dtype=yd.dtype)])
    spec_f = E2_COMPOSITIONS['all_four_with_fuzzing']
    for s in SEEDS:
        for rep in ('raw10', 'eng8'):
            k1 = 'CAL__interleaved__%s__seed%d' % (rep, s)
            k2 = 'CAL__temporal__%s__seed%d' % (rep, s)
            if done(k1) and done(k2):
                continue
            Xtr, ytr = build_training_set(spec_f, s)
            Z = Xtr[:, REPS[rep]].astype(np.float32)
            sc = StandardScaler().fit(Z)
            mdl = RandomForestClassifier(n_estimators=RF_TRANSFER_TREES, random_state=s, n_jobs=-1)
            mdl.fit(sc.transform(Z), ytr, sample_weight=compute_sample_weight('balanced', ytr))
            P = lambda X: mdl.predict_proba(sc.transform(X[:, REPS[rep]].astype(np.float32)))[:, 1]
            for key, Xc, Xt, yt in ((k1, X_OT_CALIB, X_OT, y_OT), (k2, Xa[:cut], X_T, y_T)):
                thr = float(np.quantile(P(Xc), 0.99))
                pt = P(Xt)
                m = metrics(yt, (pt > thr).astype(np.int8))
                m.update({'representation': rep, 'seed': s, 'threshold': thr,
                          'n_calibration_frames': int(len(Xc)), 'n_test_frames': int(len(yt)),
                          'fpr_pct': 100.0 * m['fp'] / max(1, m['fp'] + m['tn'])})
                save(key, m)
                log('  %-40s F1=%6.2f prec=%6.2f rec=%6.2f fpr=%.3f thr=%.3f' % (
                    key, m['f1'], m['precision'], m['recall'], m['fpr_pct'], thr))
            gc.collect()


def stage_ifdiag():
    """Review: why does the benign-only isolation forest miss the flood? Score quantiles
    of the KIA flood frames, KIA legitimate frames and the Sonata validation frames."""
    from sklearn.ensemble import IsolationForest
    if done('IFDIAG'):
        return
    Xtr, ytr = build_training_set(E2_COMPOSITIONS['all_four_with_fuzzing'], 0)
    out = {}
    q = [0.5, 0.9, 0.99, 0.999]
    for rep in ('raw10', 'eng8'):
        Xn = Xtr[ytr == 0][:, REPS[rep]].astype(np.float32)
        rs = np.random.RandomState(0)
        p = rs.permutation(len(Xn))
        fit_i, val_i = p[:int(0.8 * len(p))], p[int(0.8 * len(p)):]
        sc = StandardScaler().fit(Xn[fit_i])
        mdl = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(sc.transform(Xn[fit_i]))
        sv = -mdl.score_samples(sc.transform(Xn[val_i]))
        st = -mdl.score_samples(sc.transform(X_OT[:, REPS[rep]].astype(np.float32)))
        thr = float(np.quantile(sv, 0.99))
        att, nor = st[y_OT == 1], st[y_OT == 0]
        r = {'threshold': thr,
             'sonata_val_q': np.quantile(sv, q).tolist(), 'kia_flood_q': np.quantile(att, q).tolist(),
             'kia_normal_q': np.quantile(nor, q).tolist(),
             'flood_above_thr_pct': 100 * float((att > thr).mean()),
             'normal_above_thr_pct': 100 * float((nor > thr).mean()),
             'flood_mean_feature_z': sc.transform(X_OT[y_OT == 1][:, REPS[rep]].astype(np.float32)).mean(0).tolist(),
             'sonata_flood_mean_feature_z': sc.transform(Xtr[ytr == 1][:, REPS[rep]].astype(np.float32)).mean(0).tolist()}
        # is a Sonata flood frame flagged? (flood frames are in training as attacks)
        sa = -mdl.score_samples(sc.transform(Xtr[ytr == 1][:, REPS[rep]].astype(np.float32)))
        r['sonata_attack_above_thr_pct'] = 100 * float((sa > thr).mean())
        Xnr = Xtr[ytr == 0]
        r['sonata_normal_id002_pct'] = 100 * float((Xnr[:, 0] == 2).mean())
        r['sonata_normal_zero_payload_pct'] = 100 * float((Xnr[:, 2:10].sum(1) == 0).mean())
        r['sonata_normal_idfreq_max'] = float(Xnr[:, 12].max())
        r['kia_flood_idfreq_median'] = float(np.median(X_OT[y_OT == 1][:, 12]))
        out[rep] = r
        log('  IFDIAG %s thr=%.3f flood>thr %.2f%% kia normal>thr %.2f%% sonata attacks>thr %.2f%%'
            % (rep, thr, r['flood_above_thr_pct'], r['normal_above_thr_pct'], r['sonata_attack_above_thr_pct']))
    save('IFDIAG', out)


def stage_fznorm():
    """Review control: the fuzzing capture's legitimate frames only (injected frames removed),
    30,000 of them, with 30,000 frames from each of the other three captures. Separates the
    effect of the fuzzing attack from that of the fuzzing capture's 12 extra legitimate IDs."""
    log('Control: fuzzing-capture normal frames only, 10 draws')
    Xf, yf = CH['Fuzzy']
    nf = np.flatnonzero(yf == 0)
    base = {'DoS': E2_TOTAL // 4, 'gear': E2_TOTAL // 4, 'RPM': E2_TOTAL // 4}
    other_vocab = set(int(v) for k in ('DoS', 'gear', 'RPM')
                      for v in np.unique(CH[k][0][CH[k][1] == 0, 0].astype(np.int64)))
    for s in SEEDS:
        Xb, yb = build_training_set(base, s)
        rs = np.random.RandomState(20000 + s)
        pick = np.sort(rs.choice(nf, E2_TOTAL // 4, replace=False))
        X = np.concatenate([Xb, Xf[pick]])
        y = np.concatenate([yb, yf[pick]])
        p = np.random.RandomState(s).permutation(len(y))
        X, y = X[p], y[p]
        ids = Xf[pick, 0].astype(np.int64)
        extra_ids = sorted(set(np.unique(ids).tolist()) - other_vocab)
        share = 100.0 * float(np.isin(ids, extra_ids).mean())
        for rep in ('raw10', 'eng8'):
            transfer_unit('FZNORM__%s__seed%d' % (rep, s), {'DoS+gear+RPM': 90000, 'Fuzzy normal only': 30000},
                          rep, s, Xtr_ytr=(X, y), record_fp=(rep == 'raw10'),
                          extra={'extra_ids_in_fuzzy_normal': extra_ids,
                                 'extra_id_frame_share_pct': share})
        gc.collect()


SONG_W = 29


def song_windows(X, y):
    """Song et al. (2020) input: 29 consecutive identifiers as a bit grid (11-bit standard
    identifiers here, so the 18 always-zero extended bits are dropped). Non-overlapping windows;
    a window is an attack if it contains at least one injected frame."""
    n = (len(y) // SONG_W) * SONG_W
    ids = X[:n, 0].astype(np.int64)
    bits = ((ids[:, None] >> np.arange(10, -1, -1)) & 1).astype(np.uint8)
    return bits.reshape(-1, SONG_W * 11), y[:n].reshape(-1, SONG_W).max(1).astype(np.int8)


def stage_song():
    """Review: a published CAN detector's input representation (Song et al., DCNN on 29-frame
    identifier grids), trained with and without the fuzzing capture. No deep-learning framework
    can be installed here, so the grid is classified by a multilayer perceptron and a random
    forest; the PyTorch DCNN script in the repository reproduces the original network."""
    from sklearn.neural_network import MLPClassifier
    log('Song et al. window representation, 2 compositions x {MLP, RF} x 10 draws')
    Wc = {k: song_windows(*CH[k]) for k in CH_ORDER}
    Xd, yd = load('ot_dos')
    Xa, ya = load('ot_attack_free')
    Wd, Wa = song_windows(Xd, yd), song_windows(Xa, ya)
    WX = np.concatenate([Wd[0], Wa[0]])
    Wy = np.concatenate([Wd[1], Wa[1]])
    log('  OTIDS windows %d (%d attack); Sonata windows %s' % (
        len(Wy), int(Wy.sum()), {k: (len(v[1]), int(v[1].sum())) for k, v in Wc.items()}))
    for comp in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
        spec = E2_COMPOSITIONS[comp]
        for s in SEEDS:
            rs = np.random.RandomState(30000 + s)
            Xs, ys, Xh, yh = [], [], [], []
            for k in CH_ORDER:
                nw = len(Wc[k][1])
                perm = rs.permutation(nw)
                m = int(round(spec.get(k, 0) / SONG_W))
                tr = perm[:m]
                ho = perm[m:m + 2000]          # held-out Sonata windows from the same captures
                if m:
                    Xs.append(Wc[k][0][tr]); ys.append(Wc[k][1][tr])
                    Xh.append(Wc[k][0][ho]); yh.append(Wc[k][1][ho])
            Xtr, ytr = np.concatenate(Xs).astype(np.float32), np.concatenate(ys)
            Xho, yho = np.concatenate(Xh).astype(np.float32), np.concatenate(yh)
            for name in ('MLP', 'RandomForest'):
                key = 'SONG__%s__%s__seed%d' % (comp, name, s)
                if done(key):
                    continue
                t0 = time.time()
                if name == 'MLP':
                    mdl = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                        max_iter=200, random_state=s)
                else:
                    mdl = RandomForestClassifier(n_estimators=RF_TRANSFER_TREES, random_state=s, n_jobs=-1)
                mdl.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
                pred = mdl.predict(WX.astype(np.float32)).astype(np.int8)
                m = metrics(Wy, pred)
                mh = metrics(yho, mdl.predict(Xho).astype(np.int8))
                m.update({'model': name, 'composition': comp, 'seed': s, 'unit': 'window of 29 frames',
                          'n_train_windows': int(len(ytr)), 'n_train_attack_windows': int(ytr.sum()),
                          'fpr_pct': 100.0 * m['fp'] / max(1, m['fp'] + m['tn']),
                          'within_vehicle_f1': mh['f1'], 'within_vehicle_fpr_pct':
                              100.0 * mh['fp'] / max(1, mh['fp'] + mh['tn']),
                          'seconds': round(time.time() - t0, 1)})
                save(key, m)
                log('  %-52s F1=%6.2f prec=%6.2f rec=%6.2f fpr=%.2f  within F1=%.2f (%.0fs)' % (
                    key, m['f1'], m['precision'], m['recall'], m['fpr_pct'], mh['f1'], m['seconds']))
            gc.collect()


class SmallCNN:
    """Compact convolutional network in NumPy (no deep-learning framework can be installed here):
    conv 3x3x16 - ReLU - conv 3x3x32 - ReLU - dense 64 - ReLU - dense 1 (sigmoid), class-weighted
    binary cross-entropy, Adam (lr 1e-3), mini-batches of 128, early stopping on a 10% split."""

    def __init__(self, H, W, seed):
        rs = np.random.RandomState(seed)
        he = lambda *s, fan: (rs.randn(*s) * np.sqrt(2.0 / fan)).astype(np.float32)
        self.H, self.W = H, W
        h1, w1, h2, w2 = H - 2, W - 2, H - 4, W - 4
        self.p = {'k1': he(9, 16, fan=9), 'b1': np.zeros(16, np.float32),
                  'k2': he(144, 32, fan=144), 'b2': np.zeros(32, np.float32),
                  'w3': he(h2 * w2 * 32, 64, fan=h2 * w2 * 32), 'b3': np.zeros(64, np.float32),
                  'w4': he(64, 1, fan=64), 'b4': np.zeros(1, np.float32)}
        self.dims = (h1, w1, h2, w2)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    @staticmethod
    def im2col(x, ho, wo):          # x: n,H,W,C -> n*ho*wo, 9*C
        cols = [x[:, i:i + ho, j:j + wo, :] for i in range(3) for j in range(3)]
        return np.concatenate(cols, axis=3).reshape(-1, 9 * x.shape[3])

    def forward(self, X):
        n = len(X)
        h1, w1, h2, w2 = self.dims
        x0 = X.reshape(n, self.H, self.W, 1)
        c1 = self.im2col(x0, h1, w1)
        a1 = np.maximum(c1 @ self.p['k1'] + self.p['b1'], 0).reshape(n, h1, w1, 16)
        c2 = self.im2col(a1, h2, w2)
        z2 = c2 @ self.p['k2'] + self.p['b2']
        a2 = np.maximum(z2, 0).reshape(n, -1)
        z3 = a2 @ self.p['w3'] + self.p['b3']
        a3 = np.maximum(z3, 0)
        z4 = (a3 @ self.p['w4'] + self.p['b4'])[:, 0]
        return z4, (n, c1, a1, c2, z2, a2, z3, a3)

    def backward(self, cache, dz4):
        n, c1, a1, c2, z2, a2, z3, a3 = cache
        h1, w1, h2, w2 = self.dims
        g = {}
        g['w4'] = a3.T @ dz4[:, None]; g['b4'] = np.array([dz4.sum()], np.float32)
        d3 = (dz4[:, None] @ self.p['w4'].T) * (z3 > 0)
        g['w3'] = a2.T @ d3; g['b3'] = d3.sum(0)
        d2 = (d3 @ self.p['w3'].T).reshape(-1, 32) * (z2.reshape(-1, 32) > 0)
        g['k2'] = c2.T @ d2; g['b2'] = d2.sum(0)
        dc2 = (d2 @ self.p['k2'].T).reshape(n, h2, w2, 9, 16)
        da1 = np.zeros((n, h1, w1, 16), np.float32)
        for q, (i, j) in enumerate((i, j) for i in range(3) for j in range(3)):
            da1[:, i:i + h2, j:j + w2, :] += dc2[:, :, :, q, :]
        d1 = da1.reshape(-1, 16) * (a1.reshape(-1, 16) > 0)
        g['k1'] = c1.T @ d1; g['b1'] = d1.sum(0)
        return g

    def step(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def loss(self, X, y, w):
        z, _ = self.forward(X)
        pr = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return float(-(w * (y * np.log(pr + 1e-7) + (1 - y) * np.log(1 - pr + 1e-7))).sum() / w.sum())

    def fit(self, X, y, seed, epochs=30, batch=128, patience=5):
        rs = np.random.RandomState(seed)
        X = X.astype(np.float32); y = y.astype(np.float32)
        w = compute_sample_weight('balanced', y).astype(np.float32)
        p = rs.permutation(len(y)); nv = len(y) // 10
        vi, ti = p[:nv], p[nv:]
        best, bad, bestp = np.inf, 0, None
        for ep in range(epochs):
            for a in range(0, len(ti), batch):
                b = ti[rs.permutation(len(ti))[:batch]] if False else ti[a:a + batch]
                z, cache = self.forward(X[b])
                pr = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
                dz = (w[b] * (pr - y[b]) / w[b].sum()).astype(np.float32)
                self.step(self.backward(cache, dz))
            ti = ti[rs.permutation(len(ti))]
            vl = self.loss(X[vi], y[vi], w[vi])
            if vl < best - 1e-4:
                best, bad, bestp = vl, 0, {k: v.copy() for k, v in self.p.items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.p = bestp
        self.epochs_run = ep + 1
        return self

    def predict(self, X, chunk=4096):
        out = np.empty(len(X), np.int8)
        for a in range(0, len(X), chunk):
            out[a:a + chunk] = (self.forward(X[a:a + chunk].astype(np.float32))[0] > 0)
        return out


def stage_songcnn():
    """Song et al. representation classified by a compact CNN (NumPy), with and without fuzzing."""
    log('Song et al. windows with a compact CNN, 2 compositions x 10 draws')
    Wc = {k: song_windows(*CH[k]) for k in CH_ORDER}
    Xd, yd = load('ot_dos')
    Xa, ya = load('ot_attack_free')
    Wd, Wa = song_windows(Xd, yd), song_windows(Xa, ya)
    WX = np.concatenate([Wd[0], Wa[0]])
    Wy = np.concatenate([Wd[1], Wa[1]])
    for comp in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
        spec = E2_COMPOSITIONS[comp]
        for s in SEEDS:
            key = 'SONG__%s__CNN__seed%d' % (comp, s)
            if done(key):
                continue
            rs = np.random.RandomState(30000 + s)          # same windows as stage_song
            Xs, ys, Xh, yh = [], [], [], []
            for k in CH_ORDER:
                nw = len(Wc[k][1])
                perm = rs.permutation(nw)
                m = int(round(spec.get(k, 0) / SONG_W))
                if m:
                    Xs.append(Wc[k][0][perm[:m]]); ys.append(Wc[k][1][perm[:m]])
                    Xh.append(Wc[k][0][perm[m:m + 2000]]); yh.append(Wc[k][1][perm[m:m + 2000]])
            Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
            Xho, yho = np.concatenate(Xh), np.concatenate(yh)
            t0 = time.time()
            # a run whose network fails to train (source-vehicle held-out F1 below 50%, e.g. all ReLUs
            # dead) is restarted from a new initialisation; only Sonata data enter this check
            restarts = 0
            while True:
                mdl = SmallCNN(SONG_W, 11, s + 1000 * restarts).fit(Xtr, ytr, s + 1000 * restarts)
                mh = metrics(yho, mdl.predict(Xho))
                if mh['f1'] >= 50 or restarts >= 3:
                    break
                restarts += 1
            m = metrics(Wy, mdl.predict(WX))
            m.update({'model': 'CNN', 'composition': comp, 'seed': s, 'unit': 'window of 29 frames',
                      'n_train_windows': int(len(ytr)), 'epochs': mdl.epochs_run, 'restarts': restarts,
                      'fpr_pct': 100.0 * m['fp'] / max(1, m['fp'] + m['tn']),
                      'within_vehicle_f1': mh['f1'],
                      'within_vehicle_fpr_pct': 100.0 * mh['fp'] / max(1, mh['fp'] + mh['tn']),
                      'seconds': round(time.time() - t0, 1)})
            save(key, m)
            log('  %-52s F1=%6.2f prec=%6.2f rec=%6.2f fpr=%.2f  within F1=%.2f ep=%d (%.0fs)' % (
                key, m['f1'], m['precision'], m['recall'], m['fpr_pct'], mh['f1'], mdl.epochs_run,
                m['seconds']))
            gc.collect()


STAGE_FN = {'songcnn': stage_songcnn, 'fznorm': stage_fznorm, 'song': stage_song, 'calib2': stage_calib2, 'ifdiag': stage_ifdiag, 'e2': stage_e2, 'e3': stage_e3, 'e4': stage_e4, 'e5': stage_e5,
            'mech': stage_mech, 'mech2': stage_mech2, 'big': stage_big, 'hyper': stage_hyper,
            'event': stage_event, 'e1cv': stage_e1cv, 'e1split': stage_e1split,
            'dedup': stage_dedup, 'explain': stage_explain, 'engabl': stage_engabl,
            'baselines': stage_baselines, 'boot': stage_boot}
for st in STAGES:
    t0 = time.time()
    STAGE_FN[st]()
    log('STAGE %s complete in %.0f s' % (st, time.time() - t0))
log('ALL REQUESTED STAGES COMPLETE')
