# -*- coding: utf-8 -*-
"""Aggregate every revision-3 checkpoint into one results file.

This file is the single source of truth for every number quoted in the
manuscript, highlights, cover letter, graphical abstract and supplement.

Usage: python3 -I rev3_aggregate.py <rev3_out_dir> <out.json>
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

OUT_DIR, OUT = sys.argv[1], sys.argv[2]
CK = os.path.join(OUT_DIR, 'checkpoints')


def ck(key):
    p = os.path.join(CK, key + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def seeds_of(prefix, n=10):
    out = []
    for s in range(n):
        d = ck('%s__seed%d' % (prefix, s))
        if d is None:
            return None
        out.append(d)
    return out


def ms(v):
    v = np.asarray(v, float)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            'min': float(v.min()), 'max': float(v.max()), 'n': int(len(v)),
            'values': [float(x) for x in v]}


def summarise(runs):
    if not runs:
        return None
    r = {k: ms([d[k] for d in runs]) for k in ('f1', 'precision', 'recall', 'fp')}
    if 'training_attack_ids' in runs[0]:
        r['attack_ids'] = ms([d['training_attack_ids'] for d in runs])
    r['n_normal_test'] = int(runs[0]['fp'] + runs[0]['tn'])
    return r


def paired(a, b, alternative='greater'):
    """a - b over paired draws: Wilcoxon (one-sided), mean diff, 95% t-CI, d_z."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float('nan')
    if np.all(d == 0):
        w, p = 0.0, 1.0
    else:
        res = stats.wilcoxon(a, b, alternative=alternative)
        w, p = float(res.statistic), float(res.pvalue)
    return {'n': n, 'mean_diff': md, 'ci95': [md - half, md + half],
            'd_z': (md / sd) if sd > 0 else float('inf'), 'W': w, 'p': p}


R = {}
COMPS = ['flooding_only', 'flooding_gear_rpm', 'all_four_with_fuzzing']

# ---------------------------------------------------------------- E2
R['E2'] = {}
for c in COMPS:
    R['E2'][c] = {}
    for rep in ('raw10', 'eng8', 'all18'):
        R['E2'][c][rep] = summarise(seeds_of('E2__%s__%s' % (c, rep)))
e2raw_clean = [d['f1'] for d in seeds_of('E2__flooding_gear_rpm__raw10')]
e2raw_fuzz = [d['f1'] for d in seeds_of('E2__all_four_with_fuzzing__raw10')]
e2eng_clean = [d['f1'] for d in seeds_of('E2__flooding_gear_rpm__eng8')]
e2eng_fuzz = [d['f1'] for d in seeds_of('E2__all_four_with_fuzzing__eng8')]
e2all_fuzz = [d['f1'] for d in seeds_of('E2__all_four_with_fuzzing__all18')]
R['E2_tests'] = {
    'raw10_clean_minus_fuzz': paired(e2raw_clean, e2raw_fuzz),
    'eng8_clean_minus_fuzz': paired(e2eng_clean, e2eng_fuzz, alternative='two-sided'),
    'eng8_minus_raw10_fuzz': paired(e2eng_fuzz, e2raw_fuzz),
    'raw10_minus_eng8_clean': paired(e2raw_clean, e2eng_clean),
    'eng8_minus_all18_fuzz': paired(e2eng_fuzz, e2all_fuzz, alternative='two-sided'),
}
# forward false positives by identifier (task 24), fuzzing-included raw 10
fps = seeds_of('E2__all_four_with_fuzzing__raw10')
tot, unf, per_id = [], [], {}
for d in fps:
    t = sum(x['false_positives'] for x in d['fp_by_id'])
    u = sum(x['false_positives'] for x in d['fp_by_id'] if not x['in_sonata_vocabulary'])
    tot.append(t)
    unf.append(u)
    for x in d['fp_by_id']:
        e = per_id.setdefault(x['can_id'], {'fp': [], 'normal_frames': x['normal_frames'],
                                            'in_sonata_vocabulary': x['in_sonata_vocabulary']})
        e['fp'].append(x['false_positives'])
ids_table = []
for cid, e in per_id.items():
    v = e['fp'] + [0] * (len(fps) - len(e['fp']))
    ids_table.append({'can_id': cid, 'hex': '0x%03X' % cid, 'in_sonata_vocabulary': e['in_sonata_vocabulary'],
                      'normal_frames': e['normal_frames'], 'fp_mean': float(np.mean(v)),
                      'fp_rate_pct': 100 * float(np.mean(v)) / e['normal_frames']})
ids_table.sort(key=lambda x: -x['fp_mean'])
R['E2_fp_by_id'] = {
    'total_fp': ms(tot), 'fp_on_unfamiliar_ids': ms(unf),
    'pct_on_unfamiliar': ms([100 * u / t for u, t in zip(unf, tot)]),
    'n_ids_flagged_any': len(per_id), 'per_id': ids_table}
# OTIDS normal-traffic vocabulary vs Sonata
ot_ids = sorted(set(x['can_id'] for x in ids_table))

# ---------------------------------------------------------------- mechanism
R['MECH'] = {}
for v in ('baseline_raw10', 'legit_ids_random_payloads', 'random_ids_real_payloads',
          'raw9_no_identifier'):
    R['MECH'][v] = summarise(seeds_of('MECH__%s' % v))
mb = [d['f1'] for d in seeds_of('MECH__baseline_raw10')]
R['MECH_tests'] = {v: paired([d['f1'] for d in seeds_of('MECH__%s' % v)], mb)
                   for v in ('legit_ids_random_payloads', 'random_ids_real_payloads',
                             'raw9_no_identifier')}

R['MECH2'] = {}
for c in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
    for rep in ('raw9', 'id_dlc'):
        runs = seeds_of('MECH2__%s__%s' % (c, rep))
        if runs:
            sm = summarise(runs)
            sm['pct_fp_unfamiliar'] = float(np.mean(
                [100 * sum(x['false_positives'] for x in d['fp_by_id'] if not x['in_sonata_vocabulary'])
                 / max(1, d['fp']) for d in runs]))
            R['MECH2'].setdefault(c, {})[rep] = sm
R['_vocab'] = {k: v for k, v in json.load(open(os.path.join(OUT_DIR, 'otids_vocab.json'))).items()
               if k != 'per_id'}

# ---------------------------------------------------------------- E3
R['E3'] = {}
doses = [0, 2, 5, 10, 25, 50, 100]
page_raw, page_eng = [], []
for dz in doses:
    R['E3'][dz] = {rep: summarise(seeds_of('E3__dose%03d__%s' % (dz, rep)))
                   for rep in ('raw10', 'eng8')}
    R['E3'][dz]['fuzzing_frames'] = seeds_of('E3__dose%03d__raw10' % dz)[0]['fuzzing_frames']
    page_raw.append(R['E3'][dz]['raw10']['f1']['values'])
    page_eng.append(R['E3'][dz]['eng8']['f1']['values'])
# Page's L: blocks = draws, treatments = doses ordered; test for DEcreasing raw F1
pr = np.array(page_raw).T   # draws x doses
pe = np.array(page_eng).T


def page_test(M, decreasing=True):
    M = -M if decreasing else M
    ranks = np.apply_along_axis(stats.rankdata, 1, M)
    res = stats.page_trend_test(ranks, ranked=True)
    return {'L': float(res.statistic), 'p': float(res.pvalue), 'method': res.method}


R['E3_trend'] = {'raw10_decreasing': page_test(pr, True),
                 'eng8_decreasing': page_test(pe, True),
                 'eng8_increasing': page_test(pe, False),
                 'monotonic_draws_raw10': int(sum(np.all(np.diff(row) < 0) for row in pr)),
                 'max_rise_between_neighbouring_doses_raw10': float(np.max(np.diff(pr, axis=1)))}
R['E3_tests'] = {'raw10_dose0_minus_dose2': paired(pr[:, 0], pr[:, 1])}

# ---------------------------------------------------------------- E4
R['E4'] = {}
for rep in ('raw10', 'eng8'):
    runs = seeds_of('E4__%s' % rep)
    R['E4'][rep] = summarise(runs)
    agg = {}
    for d in runs:
        for x in d['fp_by_id']:
            agg.setdefault(x['can_id'], []).append(x['false_positives'])
    R['E4'][rep]['fp_by_id'] = sorted(
        [{'can_id': k, 'hex': '0x%03X' % k, 'fp_mean': float(np.sum(v)) / len(runs),
          'n_draws_flagged': len(v)} for k, v in agg.items()], key=lambda x: -x['fp_mean'])
    R['E4'][rep]['test_normal_ids'] = runs[0]['test_normal_ids']
    R['E4'][rep]['test_ids_unfamiliar'] = runs[0]['test_ids_unfamiliar']

# ---------------------------------------------------------------- E5
R['E5'] = {}
for name in ('LogisticRegression', 'DecisionTree', 'RandomForest', 'HistGradientBoosting'):
    R['E5'][name] = {rep: summarise(seeds_of('E5__%s__%s' % (name, rep)))
                     for rep in ('raw10', 'all18', 'eng8')}
    a = [d['f1'] for d in seeds_of('E5__%s__eng8' % name)]
    b = [d['f1'] for d in seeds_of('E5__%s__all18' % name)]
    c = [d['f1'] for d in seeds_of('E5__%s__raw10' % name)]
    R['E5'][name]['eng8_minus_all18'] = paired(a, b, alternative='two-sided')
    R['E5'][name]['eng8_minus_raw10'] = paired(a, c)

# ---------------------------------------------------------------- E1
R['E1CV'] = {}
for name in ('LogisticRegression', 'KNeighbours', 'DecisionTree', 'RandomForest', 'AdaBoost',
             'HistGradientBoosting'):
    R['E1CV'][name] = {}
    for rep in ('raw10', 'eng8', 'all18'):
        d = ck('E1CV__%s__%s' % (rep, name))
        R['E1CV'][name][rep] = None if d is None else {
            'f1_mean': d['f1_mean'], 'f1_sd': d['f1_std'],
            'folds': [f['f1'] for f in d['folds']]}
have = [n for n in R['E1CV'] if all(R['E1CV'][n][r] for r in ('raw10', 'eng8', 'all18'))]
if have:
    raw = [R['E1CV'][n]['raw10']['f1_mean'] for n in have]
    eng = [R['E1CV'][n]['eng8']['f1_mean'] for n in have]
    al = [R['E1CV'][n]['all18']['f1_mean'] for n in have]
    R['E1CV_tests'] = {'models': have,
                       'all18_minus_raw10': paired(al, raw, alternative='two-sided'),
                       'eng8_minus_raw10': paired(eng, raw, alternative='two-sided'),
                       'raw10_minus_eng8_RF': R['E1CV']['RandomForest']['raw10']['f1_mean']
                       - R['E1CV']['RandomForest']['eng8']['f1_mean']}

R['E1SPLIT'] = {}
for p in glob.glob(os.path.join(CK, 'E1SPLIT__*__*__*.json')):
    _, split, rep, name = os.path.basename(p)[:-5].split('__')
    d = json.load(open(p))
    R['E1SPLIT'].setdefault(split, {}).setdefault(name, {})[rep] = {
        'f1': d['f1'], 'precision': d['precision'], 'recall': d['recall']}
R['E1SPLIT_prevalence'] = ck('E1SPLIT__prevalence')
R['_E1RANDOM_BASE'] = {m: {r: v['f1'] for r, v in reps.items()}
                       for m, reps in R['E1SPLIT'].get('random', {}).items()}

# ---------------------------------------------------------------- dedup
R['DEDUP'] = {'E1random': {}, 'E2': {}}
for p in glob.glob(os.path.join(CK, 'DEDUP__E1random__*.json')):
    _, _, rep, name = os.path.basename(p)[:-5].split('__')
    R['DEDUP']['E1random'].setdefault(name, {})[rep] = json.load(open(p))['f1']
for c in COMPS:
    for rep in ('raw10', 'eng8'):
        runs = seeds_of('DEDUP__E2__%s__%s' % (c, rep))
        if runs:
            R['DEDUP']['E2'].setdefault(c, {})[rep] = summarise(runs)

# ---------------------------------------------------------------- big / hyper
R['BIG'] = {}
for c in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
    for rep in ('raw10', 'eng8'):
        runs = seeds_of('BIG__%s__%s' % (c, rep), n=3)
        if runs:
            R['BIG'].setdefault(c, {})[rep] = summarise(runs)
R['HYP'] = {}
for c in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
    for rep in ('raw10', 'eng8'):
        for t, dp in [(60, None), (120, None), (300, None), (60, 8), (60, 16)]:
            runs = seeds_of('HYP__%s__%s__t%d_d%s' % (c, rep, t, dp or 'none'), n=3)
            if runs:
                R['HYP'].setdefault('t%d_d%s' % (t, dp or 'none'), {}).setdefault(c, {})[rep] = summarise(runs)

# ---------------------------------------------------------------- event-level
R['EVENT'] = {}
for rep in ('eng8', 'raw10'):
    runs = seeds_of('EVENT__%s' % rep)
    if not runs:
        continue
    out = {'frame_fpr_pct': ms([d['frame_fpr_attack_free_pct'] for d in runs]),
           'fps': runs[0]['frames_per_second_attack_free'],
           'attack_free_seconds': runs[0]['attack_free_seconds'],
           'dos_recall_pct': ms([d['dos_recall_pct'] for d in runs]), 'rules': []}
    for i, rule in enumerate(runs[0]['rules']):
        eph = [d['rules'][i]['events_per_hour'] for d in runs]
        aw = [d['rules'][i]['alarmed_windows_pct'] for d in runs]
        dl = [d['rules'][i]['detection_delay_s'] for d in runs]
        out['rules'].append({'window_s': rule['window_s'], 'tau': rule['tau'],
                             'events_per_hour': ms(eph), 'alarmed_windows_pct': ms(aw),
                             'detection_delay_s': ms([x for x in dl if x is not None]) if any(
                                 x is not None for x in dl) else None,
                             'draws_detected': int(sum(x is not None for x in dl))})
    R['EVENT'][rep] = out

# ---------------------------------------------------------------- review checks
def avail(prefix, nmax=10):
    out = []
    for s_ in range(nmax):
        d = ck('%s__seed%d' % (prefix, s_))
        if d is None:
            break
        out.append(d)
    return out


R['ABL'] = {}
for c in COMPS:
    for rep in ('eng_payload', 'eng_timing'):
        runs = seeds_of('ABL__%s__%s' % (c, rep))
        if runs:
            R['ABL'].setdefault(c, {})[rep] = summarise(runs)
R['BASE'] = {}
for key in ('tunedrf__flooding_gear_rpm__raw10', 'tunedrf__all_four_with_fuzzing__raw10',
            'mlp__flooding_gear_rpm__raw10', 'mlp__all_four_with_fuzzing__raw10',
            'mlp__flooding_gear_rpm__eng8', 'mlp__all_four_with_fuzzing__eng8',
            'iforest__benign__raw10', 'iforest__benign__eng8'):
    runs = avail('BASE__' + key)
    if runs:
        sm = summarise(runs)
        if 'cv_f1' in runs[0]:
            sm['cv_f1'] = ms([100 * d['cv_f1'] for d in runs])
            sm['best_params'] = [d['best_params'] for d in runs]
        R['BASE'][key] = sm
R['BOOT'] = ck('BOOT__headline')
R['CAL'] = {}
for mode in ('interleaved', 'temporal'):
    for rep in ('raw10', 'eng8'):
        runs = seeds_of('CAL__%s__%s' % (mode, rep))
        if runs:
            sm = summarise(runs)
            sm['fpr_pct'] = ms([d['fpr_pct'] for d in runs])
            sm['n_calibration_frames'] = runs[0]['n_calibration_frames']
            sm['n_test_frames'] = runs[0]['n_test_frames']
            R['CAL'].setdefault(mode, {})[rep] = sm
R['IFDIAG'] = ck('IFDIAG')

# fuzzing-capture legitimate frames only (injected frames removed)
R['FZNORM'] = {}
for rep in ('raw10', 'eng8'):
    runs = seeds_of('FZNORM__%s' % rep)
    if runs:
        sm = summarise(runs)
        sm['extra_id_share_pct'] = ms([d['extra_id_frame_share_pct'] for d in runs])
        sm['extra_ids_per_draw'] = ms([len(d['extra_ids_in_fuzzy_normal']) for d in runs])
        R['FZNORM'][rep] = sm
if 'raw10' in R['FZNORM']:
    R['FZNORM_tests'] = {
        'raw_control_minus_fuzz': paired(R['FZNORM']['raw10']['f1']['values'],
                                         R['E2']['all_four_with_fuzzing']['raw10']['f1']['values'])}

# second vehicle set (HCRL Survival Analysis), summarised by surv_aggregate.py
_sp = os.path.join(os.path.dirname(os.path.abspath(OUT_DIR)), 'surv_out', 'surv_results.json')
if os.path.exists(_sp):
    R['SURV'] = json.load(open(_sp))
    R['SURV']['cache_audit'] = json.load(open(os.path.join(os.path.dirname(os.path.abspath(OUT_DIR)), 'surv', 'cache_audit.json')))

# Song et al. (2020) 29-frame identifier-grid representation
R['SONG'] = {}
for c in ('flooding_gear_rpm', 'all_four_with_fuzzing'):
    for m in ('DCNN', 'CNN', 'MLP', 'RandomForest'):
        runs = seeds_of('SONG__%s__%s' % (c, m))
        if runs:
            sm = summarise(runs)
            sm['fpr_pct'] = ms([d['fpr_pct'] for d in runs])
            sm['within_f1'] = ms([d['within_vehicle_f1'] for d in runs])
            sm['n_test_windows'] = int(runs[0]['n_test'])
            sm['n_train_windows'] = int(runs[0]['n_train_windows'])
            R['SONG'].setdefault(m, {})[c] = sm
R['SONG_tests'] = {m: paired(v['flooding_gear_rpm']['f1']['values'], v['all_four_with_fuzzing']['f1']['values'])
                   for m, v in R['SONG'].items() if len(v) == 2}

# ---------------------------------------------------------------- explain
R['EXPLAIN'] = ck('EXPLAIN__shapley')

json.dump(R, open(OUT, 'w'), indent=1, default=float)
print('wrote', OUT)
