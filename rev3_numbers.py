# -*- coding: utf-8 -*-
"""Every number quoted in the revision-3 text, formatted from the results file.

N(results) returns a flat dict of strings; fill(text, N) replaces <<key>> and
raises on any key it cannot resolve, so no hand-typed number reaches the paper.
Keys whose experiment has not finished yet resolve to the string 'PENDING',
and the final build refuses to run while any PENDING remains.
"""
import json
import math
import re

import numpy as np

INJ_FRAC_FUZZY = 491847 / 3838860          # injected share of the fuzzing capture


def f2(x):
    return '%.2f' % x


def f1(x):
    return '%.1f' % x


def comma(x):
    return '{:,}'.format(int(round(x)))


def pm(s, k='f1'):
    return '%.2f $\\pm$ %.2f' % (s[k]['mean'], s[k]['sd'])


def sg(x, nd=2, plus=False):
    """Signed number with a typographic minus."""
    if x < 0:
        return '$-$%.*f' % (nd, -x)
    return ('+' if plus else '') + '%.*f' % (nd, x)


def ci(t, nd=2):
    return '%s to %s' % (sg(t['ci95'][0], nd), sg(t['ci95'][1], nd))


def pval(p):
    if p < 1e-30:
        return '$p < 10^{-30}$'
    if p < 1e-15:
        return '$p < 10^{-15}$'
    if p < 0.001:
        return 'p = %.1g' % p
    return 'p = %.3f' % p if p < 0.1 else 'p = %.2f' % p


def build(R):
    N = {}
    E2 = R['E2']
    t = R['E2_tests']
    g, f, fo = E2['flooding_gear_rpm'], E2['all_four_with_fuzzing'], E2['flooding_only']
    N['e2_raw_clean'] = pm(g['raw10'])
    N['e2_raw_clean_mean'] = f2(g['raw10']['f1']['mean'])
    N['e2_raw_fuzz'] = pm(f['raw10'])
    N['e2_raw_fuzz_mean'] = f2(f['raw10']['f1']['mean'])
    N['e2_raw_fuzz_1'] = f1(f['raw10']['f1']['mean'])
    N['e2_raw_fuzz_prec'] = pm(f['raw10'], 'precision')
    N['e2_raw_fuzz_prec_mean'] = f2(f['raw10']['precision']['mean'])
    N['e2_raw_fuzz_fp'] = comma(f['raw10']['fp']['mean'])
    N['e2_ids_mean'] = comma(f['raw10']['attack_ids']['mean'])
    N['e2_ids_min'] = comma(f['raw10']['attack_ids']['min'])
    N['e2_ids_max'] = comma(f['raw10']['attack_ids']['max'])
    d = t['raw10_clean_minus_fuzz']
    N['e2_drop'] = f2(d['mean_diff'])
    N['e2_drop_1'] = f1(d['mean_diff'])
    N['e2_drop_ci'] = ci(d)
    N['e2_drop_ci_1'] = ci(d, 1)
    N['e2_drop_W'] = '%.0f' % d['W']
    N['e2_drop_p'] = '%.3f' % d['p']
    N['e2_drop_dz'] = f1(d['d_z'])
    engs = [E2[c]['eng8']['f1']['mean'] for c in E2]
    N['e2_eng_min'], N['e2_eng_max'] = f2(min(engs)), f2(max(engs))
    N['e2_eng_clean'] = pm(g['eng8'])
    N['e2_eng_fuzz'] = pm(f['eng8'])
    d = t['eng8_clean_minus_fuzz']
    N['e2_eng_effect'] = sg(-d['mean_diff'], plus=True)   # fuzz minus clean
    N['e2_eng_effect_ci'] = '%s to %s' % (sg(-d['ci95'][1]), sg(-d['ci95'][0]))
    N['e2_eng_effect_p'] = '%.2f' % d['p']
    d = t['raw10_minus_eng8_clean']
    N['e2_clean_gap'] = f2(d['mean_diff'])
    N['e2_clean_gap_ci'] = ci(d)
    N['e2_fuzz_gap'] = f2(t['eng8_minus_raw10_fuzz']['mean_diff'])
    N['e2_all_fuzz'] = pm(f['all18'])
    N['e2_all_fuzz_min'] = f2(f['all18']['f1']['min'])
    N['e2_all_fuzz_max'] = f2(f['all18']['f1']['max'])
    N['e2_eng_fuzz_min'] = f2(f['eng8']['f1']['min'])
    N['e2_eng_fuzz_max'] = f2(f['eng8']['f1']['max'])
    N['e2_prec_drop'] = f2(100 - f['raw10']['precision']['mean'])
    N['n_normal_test'] = comma(f['raw10']['n_normal_test'])
    N['e2_raw_fo'] = pm(fo['raw10'])

    # forward false positives by identifier (task 24)
    fp = R['E2_fp_by_id']
    V = R['_vocab']
    nn = V['normal_frames']
    fam_frames = V['normal_frames_on_familiar_ids']
    unf_frames = nn - fam_frames
    tot = fp['total_fp']['mean']
    unf = fp['fp_on_unfamiliar_ids']['mean']
    N['ot_ids'] = str(V['otids_normal_ids'])
    N['ot_shared'] = str(V['otids_ids_in_sonata_vocab'])
    N['ot_unshared'] = str(V['otids_normal_ids'] - V['otids_ids_in_sonata_vocab'])
    N['ot_familiar_share'] = f1(100 * fam_frames / nn)
    N['ot_unfamiliar_share'] = f1(100 * unf_frames / nn)
    N['fp_unf_pct'] = f1(100 * unf / tot)
    N['fp_rate_unf'] = f1(100 * unf / unf_frames)
    N['fp_rate_fam'] = f1(100 * (tot - unf) / fam_frames)
    full = [x['hex'] for x in fp['per_id'] if x['fp_rate_pct'] > 99.5 and x['in_sonata_vocabulary']]
    N['fp_full_familiar'] = ' and '.join(full)
    N['fp_n_ids'] = str(fp['n_ids_flagged_any'])

    # mechanism
    M = R['MECH']
    MT = R['MECH_tests']
    N['m_base'] = pm(M['baseline_raw10'])
    N['m_legit'] = pm(M['legit_ids_random_payloads'])
    N['m_real'] = pm(M['random_ids_real_payloads'])
    N['m_raw9'] = pm(M['raw9_no_identifier'])
    for k, key in (('legit', 'legit_ids_random_payloads'), ('real', 'random_ids_real_payloads'),
                   ('raw9', 'raw9_no_identifier')):
        N['m_%s_d' % k] = sg(MT[key]['mean_diff'], plus=True)
        N['m_%s_ci' % k] = ci(MT[key])
    M2 = R['MECH2']
    for rep in ('raw9', 'id_dlc'):
        c, z = M2['flooding_gear_rpm'][rep], M2['all_four_with_fuzzing'][rep]
        tag = 'raw9' if rep == 'raw9' else 'iddlc'
        N['m2_%s_clean' % tag] = pm(c)
        N['m2_%s_fuzz' % tag] = pm(z)
        N['m2_%s_drop' % tag] = f2(c['f1']['mean'] - z['f1']['mean'])
        N['m2_%s_unf' % tag] = f1(z['pct_fp_unfamiliar'])

    # dose
    E3 = R['E3']
    k3 = sorted(E3, key=lambda x: int(x))
    raw = [E3[k]['raw10']['f1']['mean'] for k in k3]
    eng = [E3[k]['eng8']['f1']['mean'] for k in k3]
    tr = R['E3_trend']
    N['page_L'] = '%.0f' % tr['raw10_decreasing']['L']
    assert tr['raw10_decreasing']['method'] == 'exact' and tr['raw10_decreasing']['p'] < 0.001
    N['page_p'] = 'p $<$ 0.001'
    N['page_eng_p'] = ('$<$ 0.001' if tr['eng8_increasing']['p'] < 0.001 else '= %.3f' % tr['eng8_increasing']['p'])
    N['mono_draws'] = {7: 'seven', 8: 'eight', 9: 'nine', 10: 'all ten', 6: 'six', 5: 'five'}.get(
        tr['monotonic_draws_raw10'], str(tr['monotonic_draws_raw10']))
    N['max_rise'] = f1(tr['max_rise_between_neighbouring_doses_raw10'])
    d = R['E3_tests']['raw10_dose0_minus_dose2']
    N['d2_drop'] = f2(d['mean_diff'])
    N['d2_drop_1'] = f1(d['mean_diff'])
    N['d2_ci'] = ci(d)
    N['d2_prec_drop'] = f2(E3[k3[0]]['raw10']['precision']['mean'] - E3[k3[1]]['raw10']['precision']['mean'])
    ff = E3[k3[1]]['fuzzing_frames']
    inj = ff * INJ_FRAC_FUZZY
    N['ff800'] = comma(ff)
    N['inj800'] = comma(inj)
    N['inj800_share'] = '%.2f' % (100 * inj / (120000 + ff))
    N['e3_raw_end'] = pm(E3[k3[-1]]['raw10'])
    N['e3_raw_d2'] = pm(E3[k3[1]]['raw10'])
    N['e3_eng_min'], N['e3_eng_max'] = f2(min(eng)), f2(max(eng))
    N['e3_eng_range'] = f2(max(eng) - min(eng))
    N['e3_raw_range'] = f2(raw[0] - raw[-1])
    N['e3_raw_range_round'] = '%d' % int(round(raw[0] - raw[-1]))
    allg = engs + eng
    N['eng_all_min'], N['eng_all_max'] = f1(min(allg)), f1(max(allg))

    # reverse
    E4 = R['E4']
    N['e4_raw'] = pm(E4['raw10'])
    N['e4_raw_prec'] = pm(E4['raw10'], 'precision')
    N['e4_eng'] = pm(E4['eng8'])
    N['e4_eng_prec'] = pm(E4['eng8'], 'precision')
    vals = E4['raw10']['f1']['values']
    fps = E4['raw10']['fp']['values']
    mode = max(set(round(v, 2) for v in vals), key=lambda m: sum(round(v, 2) == m for v in vals))
    nmode = sum(round(v, 2) == mode for v in vals)
    N['e4_mode'] = f2(mode)
    N['e4_nmode'] = {8: 'eight', 9: 'nine', 7: 'seven', 6: 'six', 10: 'all ten'}.get(nmode, str(nmode))
    N['e4_mode_fp'] = comma(max(fps))
    others = sorted([(v, p) for v, p in zip(vals, fps) if round(v, 2) != mode], key=lambda x: -x[0])
    N['e4_others'] = ' and '.join('%s\\%% (%s false positives)' % (f2(v), comma(p)) for v, p in others)
    N['e4_unfam'] = str(len(E4['raw10']['test_ids_unfamiliar']))
    N['e4_test_ids'] = str(E4['raw10']['test_normal_ids'])
    ids0 = [x['hex'] for x in E4['raw10']['fp_by_id']]
    N['e4_fp_ids'] = ', '.join(ids0)

    # E5
    E5 = R['E5']
    trees = ('DecisionTree', 'RandomForest', 'HistGradientBoosting')
    rr = [E5[m]['raw10']['f1']['mean'] for m in trees]
    pr = [E5[m]['raw10']['precision']['mean'] for m in trees]
    N['e5_tree_raw_min'], N['e5_tree_raw_max'] = f2(min(rr)), f2(max(rr))
    N['e5_tree_prec_min'], N['e5_tree_prec_max'] = f2(min(pr)), f2(max(pr))
    ee = [E5[m]['eng8']['f1']['mean'] for m in E5]
    N['e5_eng_min'], N['e5_eng_max'] = f2(min(ee)), f2(max(ee))
    N['e5_lr_fp'] = comma(E5['LogisticRegression']['raw10']['fp']['mean'])
    N['e5_lr_fp_pct'] = f1(100 * E5['LogisticRegression']['raw10']['fp']['mean'] / f['raw10']['n_normal_test'])
    d = E5['LogisticRegression']['eng8_minus_all18']
    N['e5_lr_gain'] = f2(d['mean_diff'])
    N['e5_lr_gain_ci'] = ci(d)
    N['e5_lr_gain_p'] = '%.3f' % d['p']
    td = [E5[m]['eng8_minus_all18']['mean_diff'] for m in trees]
    tp = [E5[m]['eng8_minus_all18']['p'] for m in trees]
    N['e5_tree_diff_min'], N['e5_tree_diff_max'] = sg(min(td), plus=True), sg(max(td), plus=True)
    N['e5_tree_p_min'] = '%.2f' % min(tp)
    N['e5_dt_eng'] = pm(E5['DecisionTree']['eng8'])
    N['e5_dt_raw'] = pm(E5['DecisionTree']['raw10'])
    N['e5_rf_eng'] = pm(E5['RandomForest']['eng8'])
    N['e5_rf_eng_fp'] = comma(E5['RandomForest']['eng8']['fp']['mean'])

    # E1 CV
    C = R['E1CV']
    nonlin = ('KNeighbours', 'DecisionTree', 'RandomForest', 'AdaBoost', 'HistGradientBoosting')
    N['e1_raw_nonlin_min'] = f2(min(C[m]['raw10']['f1_mean'] for m in nonlin))
    N['e1_eng_nonlin_min'] = f2(min(C[m]['eng8']['f1_mean'] for m in nonlin))
    N['e1_eng_nonlin_max'] = f2(max(C[m]['eng8']['f1_mean'] for m in nonlin))
    for m, k in (('RandomForest', 'rf'), ('LogisticRegression', 'lr'), ('DecisionTree', 'dt'),
                 ('AdaBoost', 'ada')):
        for rep, rk in (('raw10', 'raw'), ('eng8', 'eng'), ('all18', 'all')):
            N['e1cv_%s_%s' % (k, rk)] = '%.2f $\\pm$ %.2f' % (C[m][rep]['f1_mean'], C[m][rep]['f1_sd'])
    N['e1cv_rf_cost'] = f2(C['RandomForest']['raw10']['f1_mean'] - C['RandomForest']['eng8']['f1_mean'])
    costs = [C[m]['raw10']['f1_mean'] - C[m]['eng8']['f1_mean'] for m in nonlin]
    N['e1cv_cost_max'] = f2(max(costs))

    # E1 splits
    S = R.get('E1SPLIT') or {}
    P = R.get('E1SPLIT_prevalence') or {}
    ok = all(s in S for s in ('random', 'chronological', 'prevalence_matched')) and \
        all(len(S[s]) == 8 for s in S) and all(len(v) == 3 for s in S for v in S[s].values())
    if ok:
        N['sp_chron_rf_raw'] = f2(S['chronological']['RandomForest']['raw10']['f1'])
        N['sp_prev_rf_raw'] = f2(S['prevalence_matched']['RandomForest']['raw10']['f1'])
        N['sp_rand_rf_raw'] = f2(S['random']['RandomForest']['raw10']['f1'])
        N['sp_chron_test_att'] = f2(P['chronological']['test_attack_pct'])
        N['sp_rand_test_att'] = f2(P['random']['test_attack_pct'])
        N['sp_prev_test_att'] = f2(P['prevalence_matched']['test_attack_pct'])
        nl = ('KNeighbours', 'DecisionTree', 'RandomForest', 'AdaBoost', 'GradientBoosting',
              'HistGradientBoosting', 'Stacking')
        worst = min(nl, key=lambda m: S['chronological'][m]['raw10']['f1'])
        N['sp_worst_name'] = {'KNeighbours': 'k-nearest neighbours', 'DecisionTree': 'decision tree',
                              'RandomForest': 'random forest', 'AdaBoost': 'AdaBoost',
                              'GradientBoosting': 'gradient boosting',
                              'HistGradientBoosting': 'histogram gradient boosting',
                              'Stacking': 'stacking ensemble'}[worst]
        N['sp_worst_chron'] = f2(S['chronological'][worst]['raw10']['f1'])
        N['sp_worst_prev'] = f2(S['prevalence_matched'][worst]['raw10']['f1'])
        N['sp_nl_prev_min'] = f2(min(S['prevalence_matched'][m]['raw10']['f1'] for m in nl))
        N['sp_nl_rand_min'] = f2(min(S['random'][m]['raw10']['f1'] for m in nl))
        N['sp_eng_rf_prev'] = f2(S['prevalence_matched']['RandomForest']['eng8']['f1'])
    else:
        for k in ('sp_chron_rf_raw', 'sp_prev_rf_raw', 'sp_rand_rf_raw', 'sp_chron_test_att',
                  'sp_rand_test_att', 'sp_prev_test_att', 'sp_worst_name', 'sp_worst_chron',
                  'sp_worst_prev', 'sp_nl_prev_min', 'sp_nl_rand_min', 'sp_eng_rf_prev'):
            N[k] = 'PENDING'

    # dedup
    D = R.get('DEDUP') or {}
    full_dedup = D.get('E1random') and len(D['E1random']) == 6 and 'all_four_with_fuzzing' in D.get('E2', {})
    if full_dedup:
        base = R['_E1RANDOM_BASE']
        diffs = {m: {r: D['E1random'][m][r] - base[m][r] for r in ('raw10', 'eng8', 'all18')}
                 for m in D['E1random']}
        nl = [m for m in diffs if m != 'LogisticRegression']
        N['dd_nl_max'] = f2(max(abs(v) for m in nl for v in diffs[m].values()))
        N['dd_lr_max'] = f2(max(abs(v) for v in diffs['LogisticRegression'].values()))
        N['dd_e2_raw_fuzz'] = pm(D['E2']['all_four_with_fuzzing']['raw10'])
        N['dd_e2_diff'] = sg(D['E2']['all_four_with_fuzzing']['raw10']['f1']['mean']
                             - f['raw10']['f1']['mean'], plus=True)
        N['dd_e2_eng'] = pm(D['E2']['all_four_with_fuzzing']['eng8'])
    else:
        for k in ('dd_nl_max', 'dd_lr_max', 'dd_e2_raw_fuzz', 'dd_e2_diff', 'dd_e2_eng'):
            N[k] = 'PENDING'

    # robustness
    H = R.get('HYP') or {}
    if len(H) == 5 and all(len(H[k]) == 2 and all(len(H[k][c]) == 2 for c in H[k]) for k in H):
        rf = [H[k]['all_four_with_fuzzing']['raw10']['f1']['mean'] for k in H]
        rc = [H[k]['flooding_gear_rpm']['raw10']['f1']['mean'] for k in H]
        ef = [H[k][c]['eng8']['f1']['mean'] for k in H for c in H[k]]
        N['hy_raw_fuzz_min'], N['hy_raw_fuzz_max'] = f2(min(rf)), f2(max(rf))
        N['hy_raw_clean_min'] = f2(min(rc))
        N['hy_raw_clean_phrase'] = ('stays at 100.00\\%' if min(rc) >= 99.995 else 'stays at or above %s\\%%' % f2(min(rc)))
        N['hy_eng_min'], N['hy_eng_max'] = f2(min(ef)), f2(max(ef))
    else:
        for k in ('hy_raw_fuzz_min', 'hy_raw_fuzz_max', 'hy_raw_clean_min', 'hy_raw_clean_phrase', 'hy_eng_min', 'hy_eng_max'):
            N[k] = 'PENDING'
    B = R.get('BIG') or {}
    if len(B) == 2 and all(len(B[c]) == 2 for c in B):
        N['big_raw_clean'] = pm(B['flooding_gear_rpm']['raw10'])
        N['big_raw_fuzz'] = pm(B['all_four_with_fuzzing']['raw10'])
        N['big_raw_fuzz_prec'] = pm(B['all_four_with_fuzzing']['raw10'], 'precision')
        N['big_eng_clean'] = pm(B['flooding_gear_rpm']['eng8'])
        N['big_eng_fuzz'] = pm(B['all_four_with_fuzzing']['eng8'])
        N['big_ids'] = comma(B['all_four_with_fuzzing']['raw10']['attack_ids']['mean'])
    else:
        for k in ('big_raw_clean', 'big_raw_fuzz', 'big_raw_fuzz_prec', 'big_eng_clean',
                  'big_eng_fuzz', 'big_ids'):
            N[k] = 'PENDING'

    # review checks: engineered-feature ablation, baselines, bootstrap
    A = R.get('ABL') or {}
    if all(c in A and len(A[c]) == 2 for c in ('flooding_only', 'flooding_gear_rpm', 'all_four_with_fuzzing')):
        pc, pf = A['flooding_gear_rpm']['eng_payload'], A['all_four_with_fuzzing']['eng_payload']
        tc, tf = A['flooding_gear_rpm']['eng_timing'], A['all_four_with_fuzzing']['eng_timing']
        N['abl_pay_clean'], N['abl_pay_fuzz'] = pm(pc), pm(pf)
        N['abl_pay_prec'] = f2(pf['precision']['mean'])
        N['abl_tim_fo'] = pm(A['flooding_only']['eng_timing'])
        N['abl_tim_clean'], N['abl_tim_fuzz'] = pm(tc), pm(tf)
        N['abl_tim_drop'] = f2(tc['f1']['mean'] - tf['f1']['mean'])
        N['abl_tim_prec'] = f2(tf['precision']['mean'])
    else:
        for k in ('abl_pay_clean', 'abl_pay_fuzz', 'abl_pay_prec', 'abl_tim_fo', 'abl_tim_clean',
                  'abl_tim_fuzz', 'abl_tim_drop', 'abl_tim_prec'):
            N[k] = 'PENDING'
    BT = R.get('BOOT')
    N['boot_drop_ci'] = ('%s to %s' % (f2(BT['raw_drop']['ci95'][0]), f2(BT['raw_drop']['ci95'][1]))) if BT else 'PENDING'
    N['boot_block'] = comma(BT['block_frames']) if BT else 'PENDING'
    N['boot_nblocks'] = str(sum(BT['n_blocks'])) if BT else 'PENDING'
    BA = R.get('BASE') or {}
    need = ('tunedrf__flooding_gear_rpm__raw10', 'tunedrf__all_four_with_fuzzing__raw10',
            'mlp__flooding_gear_rpm__raw10', 'mlp__all_four_with_fuzzing__raw10',
            'mlp__flooding_gear_rpm__eng8', 'mlp__all_four_with_fuzzing__eng8',
            'iforest__benign__raw10', 'iforest__benign__eng8')
    if all(k in BA for k in need):
        nmin = min(BA[k]['f1']['n'] for k in need)
        N['base_n'] = {3: 'three', 4: 'four', 5: 'five', 10: 'ten'}.get(nmin, str(nmin))
        N['base_trf_cv'] = f2(BA['tunedrf__all_four_with_fuzzing__raw10']['cv_f1']['mean'])
        N['base_trf_clean'] = pm(BA['tunedrf__flooding_gear_rpm__raw10'])
        N['base_trf_fuzz'] = pm(BA['tunedrf__all_four_with_fuzzing__raw10'])
        for rep in ('raw10', 'eng8'):
            t = 'raw' if rep == 'raw10' else 'eng'
            N['base_mlp_%s_clean' % t] = pm(BA['mlp__flooding_gear_rpm__%s' % rep])
            N['base_mlp_%s_fuzz' % t] = pm(BA['mlp__all_four_with_fuzzing__%s' % rep])
            for kk, lab in (('iforest__benign__%s' % rep, 'if'),):
                d = BA[kk]
                N['base_%s_%s' % (lab, t)] = pm(d)
                N['base_%s_%s_p' % (lab, t)] = f2(d['precision']['mean'])
                N['base_%s_%s_r' % (lab, t)] = f2(d['recall']['mean'])
    else:
        for k in ('base_n', 'base_trf_cv', 'base_trf_clean', 'base_trf_fuzz', 'base_mlp_raw_clean',
                  'base_mlp_raw_fuzz', 'base_mlp_eng_clean', 'base_mlp_eng_fuzz', 'base_if_raw', 'base_if_raw_p',
                  'base_if_raw_r', 'base_if_eng', 'base_if_eng_p', 'base_if_eng_r'):
            N[k] = 'PENDING'

    CL = R.get('CAL') or {}
    if len(CL) == 2 and all(len(CL[m]) == 2 for m in CL):
        for mode, mt in (('interleaved', ''), ('temporal', '_t')):
            for rep, t in (('raw10', 'raw'), ('eng8', 'eng')):
                d = CL[mode][rep]
                N['base_cal_%s%s' % (t, mt)] = pm(d)
                N['base_cal_%s%s_p' % (t, mt)] = f2(d['precision']['mean'])
                N['base_cal_%s%s_r' % (t, mt)] = f2(d['recall']['mean'])
                N['cal_%s%s_fpr' % (t, mt)] = f2(d['fpr_pct']['mean'])
                N['cal_%s%s_min' % (t, mt)] = f2(d['f1']['min'])
                N['cal_%s%s_maxfpr' % (t, mt)] = f2(d['fpr_pct']['max'])
                n100 = sum(1 for v in d['f1']['values'] if v >= 99.995)
                N['cal_%s%s_n100' % (t, mt)] = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                                               6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'all ten'}[n100]
        N['cal_n_inter'] = comma(CL['interleaved']['raw10']['n_calibration_frames'])
        N['cal_n_temp'] = comma(CL['temporal']['raw10']['n_calibration_frames'])
        N['cal_n_test_temp'] = comma(CL['temporal']['raw10']['n_test_frames'])
    else:
        for k in ('base_cal_raw', 'base_cal_raw_t', 'base_cal_eng', 'base_cal_eng_t', 'cal_n_inter'):
            N[k] = 'PENDING'
    IF = R.get('IFDIAG')
    if IF:
        N['if_raw_flood'] = '%.3f' % IF['raw10']['kia_flood_q'][0]
        N['if_raw_valmed'] = '%.3f' % IF['raw10']['sonata_val_q'][0]
        N['if_raw_thr'] = '%.3f' % IF['raw10']['threshold']
        N['if_id002'] = f1(IF['raw10']['sonata_normal_id002_pct'])
        N['if_zero'] = f1(IF['raw10']['sonata_normal_zero_payload_pct'])
        N['if_eng_flood'] = '%.3f' % IF['eng8']['kia_flood_q'][0]
        N['if_eng_thr'] = '%.3f' % IF['eng8']['threshold']
        N['if_eng_kia_norm'] = f2(IF['eng8']['normal_above_thr_pct'])
        N['if_idfreq_z'] = '%.0f' % IF['eng8']['flood_mean_feature_z'][2]
        N['if_idfreq_kia'] = '%.0f' % IF['eng8']['kia_flood_idfreq_median']
        N['if_idfreq_max'] = '%.0f' % IF['eng8']['sonata_normal_idfreq_max']
    else:
        N['if_raw_flood'] = 'PENDING'

    # event level
    EV = R['EVENT']
    e, r = EV['eng8'], EV['raw10']
    N['ev_fps'] = comma(e['fps'])
    N['ev_secs'] = '%.1f' % e['attack_free_seconds']
    N['ev_min'] = '%.1f' % (e['attack_free_seconds'] / 60)
    N['ev_eng_fpr'] = pm({'x': e['frame_fpr_pct']}, 'x')
    N['ev_eng_fpr_mean'] = f2(e['frame_fpr_pct']['mean'])
    N['ev_eng_fps_false'] = '%.0f' % (e['frame_fpr_pct']['mean'] / 100 * e['fps'])
    N['ev_raw_fpr'] = pm({'x': r['frame_fpr_pct']}, 'x')
    N['ev_raw_fpr_mean'] = f1(r['frame_fpr_pct']['mean'])

    def rule(rep, w, tau):
        return [x for x in EV[rep]['rules'] if abs(x['window_s'] - w) < 1e-9 and abs(x['tau'] - tau) < 1e-9][0]
    x = rule('eng8', 0.1, 0.05)
    N['ev_eng_01_5'] = '%.1f $\\pm$ %.1f' % (x['events_per_hour']['mean'], x['events_per_hour']['sd'])
    x = rule('eng8', 0.1, 0.01)
    N['ev_eng_01_1'] = comma(x['events_per_hour']['mean'])
    zero_eng = all(rule('eng8', w, tau)['events_per_hour']['max'] == 0
                   for w in (0.1, 1.0) for tau in (0.10, 0.20, 0.50))
    N['ev_eng_zero_ok'] = 'yes' if zero_eng else 'NO'
    dl = rule('eng8', 0.1, 0.10)
    N['ev_eng_delay'] = '%.1f' % dl['detection_delay_s']['mean']
    N['ev_eng_delay_draws'] = str(dl['draws_detected'])
    full_raw = all(rule('raw10', w, tau)['alarmed_windows_pct']['min'] > 99.9
                   for w in (0.1, 1.0) for tau in (0.01, 0.05, 0.10, 0.20))
    N['ev_raw_full_ok'] = 'yes' if full_raw else 'NO'
    x = rule('raw10', 0.1, 0.50)
    N['ev_raw_50_events'] = '%.1f' % x['events_per_hour']['max']
    N['ev_raw_50_delay'] = '%.1f' % x['detection_delay_s']['mean']

    # explainability
    EX = R['EXPLAIN']
    phi = np.array(EX['phi'])
    m = np.abs(phi).mean(0)
    feats = EX['features']
    order = list(np.argsort(-m))
    N['x_rank1'], N['x_rank2'], N['x_rank3'] = (feats[i].replace('_', '\\_') for i in order[:3])
    N['x_v1'], N['x_v2'], N['x_v3'] = ('%.4f' % m[i] for i in order[:3])
    pim = np.array(EX['permutation_importance_mean'])
    psd = np.array(EX['permutation_importance_std'])
    po = list(np.argsort(-pim))
    N['x_p1'], N['x_p2'] = (feats[i].replace('_', '\\_') for i in po[:2])
    N['x_pv1'] = '%.4f $\\pm$ %.4f' % (pim[po[0]], psd[po[0]])
    N['x_pv2'] = '%.4f $\\pm$ %.4f' % (pim[po[1]], psd[po[1]])
    imp = np.array(EX['impurity_importance'])
    io = int(np.argmax(imp))
    N['x_imp_feat'] = feats[io].replace('_', '\\_')
    N['x_imp_val'] = '%.3f' % imp[io]
    N['x_imp_shap_rank'] = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth',
                            7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'}[order.index(io) + 1]
    N['x_eng_top7'] = str(sum(1 for i in order[:7] if i >= 10))
    grp = np.array(EX['group'])
    N['x_id_dos'] = sg(phi[grp == 'DoS', 0].mean(), 3, True)
    N['x_id_fuzz'] = sg(phi[grp == 'Fuzzy', 0].mean(), 3, True)
    N['x_id_normal'] = sg(phi[grp == 'normal', 0].mean(), 3, True)
    N['x_base'] = '%.3f' % EX['base_value']
    N['x_f1'] = f2(EX['within_vehicle_test_f1'])
    N['x_nframes'] = str(EX['n_frames'])
    N['x_nperm'] = str(EX['n_permutations'])
    N['x_evals'] = comma(EX['n_frames'] * EX['n_permutations'] * (len(feats) + 1))

    # all engineered-8 random-forest and E5 means (composition, dose, classifier experiments)
    ee5 = [R['E5'][m]['eng8']['f1']['mean'] for m in R['E5']]
    N['eng_wide_min'], N['eng_wide_max'] = f1(min(allg + ee5)), f1(max(allg + ee5))

    # tuned forest: selected configuration, and its identity with the default forest
    BA = R['BASE']
    bp = [p for k in ('tunedrf__flooding_gear_rpm__raw10', 'tunedrf__all_four_with_fuzzing__raw10')
          for p in BA[k]['best_params']]
    default = {'max_depth': 'None', 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 60}
    assert all(p == default for p in bp), bp
    tv = BA['tunedrf__all_four_with_fuzzing__raw10']['f1']['values']
    dv = E2['all_four_with_fuzzing']['raw10']['f1']['values'][:len(tv)]
    assert max(abs(a - b) for a, b in zip(tv, dv)) < 1e-9
    cvv = [100 * x for x in BA['tunedrf__all_four_with_fuzzing__raw10']['cv_f1']['values']] \
        if max(BA['tunedrf__all_four_with_fuzzing__raw10']['cv_f1']['values']) <= 1.0 \
        else BA['tunedrf__all_four_with_fuzzing__raw10']['cv_f1']['values']
    N['base_trf_cv_min'] = f2(min(cvv))
    N['base_trf_default_fuzz'] = pm({'x': {'mean': float(np.mean(dv)), 'sd': float(np.std(dv, ddof=1))}}, 'x')

    # control: fuzzing-capture legitimate frames only
    FZ = R.get('FZNORM') or {}
    if len(FZ) == 2:
        N['fz_raw'] = pm(FZ['raw10'])
        N['fz_raw_fp'] = comma(FZ['raw10']['fp']['max'])
        N['fz_eng'] = pm(FZ['eng8'])
        N['fz_share'] = f1(FZ['raw10']['extra_id_share_pct']['mean'])
        N['fz_n_share'] = comma(FZ['raw10']['extra_id_share_pct']['mean'] / 100 * 30000)
        N['fz_ids_min'] = '%d' % FZ['raw10']['extra_ids_per_draw']['min']
        N['fz_ids_max'] = '%d' % FZ['raw10']['extra_ids_per_draw']['max']
        t = R['FZNORM_tests']['raw_control_minus_fuzz']
        N['fz_vs_fuzz'] = f2(t['mean_diff'])
        N['fz_vs_fuzz_ci'] = ci(t)
        N['fz_vs_fuzz_p'] = '%.3f' % t['p']
    else:
        N['fz_raw'] = 'PENDING'

    # Song et al. representation
    SO = R.get('SONG') or {}
    if all(m in SO and len(SO[m]) == 2 for m in ('CNN', 'MLP', 'RandomForest')):
        for m, t in (('CNN', 'cnn'), ('MLP', 'mlp'), ('RandomForest', 'rf')):
            c, f_ = SO[m]['flooding_gear_rpm'], SO[m]['all_four_with_fuzzing']
            N['so_%s_clean' % t] = pm(c)
            N['so_%s_fuzz' % t] = pm(f_)
            N['so_%s_fpr_clean' % t] = f2(c['fpr_pct']['mean'])
            N['so_%s_fpr_fuzz' % t] = f2(f_['fpr_pct']['mean'])
            N['so_%s_rec_fuzz' % t] = f2(f_['recall']['mean'])
            N['so_%s_within_clean' % t] = f2(c['within_f1']['mean'])
            N['so_%s_within_fuzz' % t] = f2(f_['within_f1']['mean'])
            d = R['SONG_tests'][m]
            N['so_%s_drop' % t] = f2(d['mean_diff'])
            N['so_%s_drop_ci' % t] = ci(d)
            N['so_%s_drop_p' % t] = '%.3f' % d['p']
        N['so_test_windows'] = comma(SO['CNN']['flooding_gear_rpm']['n_test_windows'])
        N['so_train_windows'] = comma(SO['CNN']['flooding_gear_rpm']['n_train_windows'])
        N['so_train_windows_f'] = comma(SO['CNN']['all_four_with_fuzzing']['n_train_windows'])
    else:
        N['so_cnn_clean'] = 'PENDING'

    # second vehicle set (Survival Analysis)
    SV = R.get('SURV')
    if SV and len(SV.get('E3', {})) == 3:
        E2s, T2 = SV['E2'], SV['E2_tests']
        pairs = [(a_, b_) for a_ in E2s for b_ in E2s[a_]['raw10']['clean']['test']]
        def v(src, t, rep, comp, kind, m):
            return E2s[src][rep][comp]['test'][t][kind][m]['mean']
        rc = [v(a_, b_, 'raw10', 'clean', 'free', 'fpr') for a_, b_ in pairs]
        rf = [v(a_, b_, 'raw10', 'fuzz', 'free', 'fpr') for a_, b_ in pairs]
        ec = [v(a_, b_, 'eng8', 'clean', 'free', 'fpr') for a_, b_ in pairs]
        ef = [v(a_, b_, 'eng8', 'fuzz', 'free', 'fpr') for a_, b_ in pairs]
        efns = [v(a_, b_, 'eng8', 'fuzz', 'free', 'fpr') for a_, b_ in pairs if b_ != 'spark']
        efs = [v(a_, b_, 'eng8', 'fuzz', 'free', 'fpr') for a_, b_ in pairs if b_ == 'spark']
        N['sa_raw_clean_min'], N['sa_raw_clean_max'] = f2(min(rc)), f2(max(rc))
        N['sa_raw_fuzz_min'], N['sa_raw_fuzz_max'] = f1(min(rf)), f1(max(rf))
        N['sa_eng_clean_max'] = f2(max(ec))
        N['sa_eng_fuzz_ns_max'] = f2(max(efns))
        N['sa_eng_fuzz_s_min'], N['sa_eng_fuzz_s_max'] = f1(min(efs)), f1(max(efs))
        rr = [v(a_, b_, 'raw10', 'fuzz', 'fuzzy', 'recall') for a_, b_ in pairs]
        er = [v(a_, b_, 'eng8', 'fuzz', 'fuzzy', 'recall') for a_, b_ in pairs]
        N['sa_raw_fzrec_min'] = f1(min(rr)); N['sa_eng_fzrec_min'] = f1(min(er))
        N['sa_eng_fzrec_max'] = f1(max(er))
        fl = [v(a_, b_, rep, comp, 'flood', 'recall') for a_, b_ in pairs for rep in ('raw10', 'eng8') for comp in ('clean', 'fuzz')]
        N['sa_flood_rec_min'] = f2(min(fl))
        ps = [T2[a_]['raw10'][b_]['free_fpr_increase']['p'] for a_, b_ in pairs]
        N['sa_raw_p_max'] = '%.3f' % max(ps)
        inc = [T2[a_]['raw10'][b_]['free_fpr_increase']['mean_diff'] for a_, b_ in pairs]
        N['sa_raw_inc_min'], N['sa_raw_inc_max'] = f1(min(inc)), f1(max(inc))
        wv = [E2s[a_][rep]['fuzz']['within_f1']['mean'] for a_ in E2s for rep in ('raw10',)]
        N['sa_within_raw_min'] = f2(min(wv))
        # dose: raw FPR at the smallest and largest dose, monotonic pairs
        E3s = SV['E3']
        lo = [E3s[a_]['raw10']['500']['test'][b_]['free']['fpr']['mean'] - E3s[a_]['raw10']['0']['test'][b_]['free']['fpr']['mean'] for a_, b_ in pairs]
        hi = [E3s[a_]['raw10']['20000']['test'][b_]['free']['fpr']['mean'] for a_, b_ in pairs]
        N['sa_d500_min'], N['sa_d500_max'] = f1(min(lo)), f1(max(lo))
        N['sa_d20k_min'], N['sa_d20k_max'] = f1(min(hi)), f1(max(hi))
        doses = ['0', '500', '1000', '2500', '5000', '10000', '20000']
        mono = sum(1 for a_, b_ in pairs if all(E3s[a_]['raw10'][doses[i + 1]]['test'][b_]['free']['fpr']['mean'] >=
                                                E3s[a_]['raw10'][doses[i]]['test'][b_]['free']['fpr']['mean'] - 0.5
                                                for i in range(len(doses) - 1)))
        N['sa_mono'] = {6: 'all six', 5: 'five', 4: 'four'}.get(mono, str(mono))
        ehi = [E3s[a_]['eng8'][d]['test'][b_]['free']['fpr']['mean'] for a_, b_ in pairs if b_ != 'spark' for d in doses]
        N['sa_e3_eng_ns_max'] = f2(max(ehi))
        es = [E3s[a_]['eng8'][d]['test'][b_]['free']['fpr']['mean'] for a_, b_ in pairs if b_ == 'spark' for d in doses[1:]]
        N['sa_e3_eng_s_min'], N['sa_e3_eng_s_max'] = f1(min(es)), f1(max(es))
        mal = [(v(a_, b_, rep, 'fuzz', 'malf', 'recall')) for a_, b_ in pairs for rep in ('raw10', 'eng8')]
        N['sa_malf_min'], N['sa_malf_max'] = f1(min(mal)), f1(max(mal))
        CA = SV['cache_audit']
        N['sa_relabel'] = str(CA['spark_fuzzy']['relabelled'])
        N['sa_frames'] = comma(sum(a_['frames'] for a_ in CA.values()))
    else:
        N['sa_raw_fuzz_min'] = 'PENDING'
    return N


KEY = re.compile(r'<<([a-zA-Z0-9_]+)>>')


def fill(text, N):
    missing = sorted(set(k for k in KEY.findall(text) if k not in N))
    if missing:
        raise SystemExit('unknown number keys: %s' % missing)
    return KEY.sub(lambda m_: N[m_.group(1)], text)


if __name__ == '__main__':
    import sys
    R = json.load(open(sys.argv[1]))
    for k, v in sorted(build(R).items()):
        print('%-22s %s' % (k, v))
