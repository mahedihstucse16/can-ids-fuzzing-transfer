# -*- coding: utf-8 -*-
"""Build every revision-3 table, figure and supplementary CSV from the results file.

Usage: python3 -I rev3_tables.py <rev3_results.json> <latex_dir> <supp_csv_dir>
Writes <latex_dir>/tables/*.tex, <latex_dir>/figs/f06..f10.png and CSVs.
"""
import csv
import json
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open(sys.argv[1]))
LAT, SUPP = sys.argv[2], sys.argv[3]
TDIR = os.path.join(LAT, 'tables')
FDIR = os.path.join(LAT, 'figs')
for d in (TDIR, FDIR, SUPP):
    os.makedirs(d, exist_ok=True)

GREY, TEAL, BLUE, EDGE, INK = '#8A8F94', '#1B7F79', '#6C7FA8', '#4A4F54', '#1C2024'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.edgecolor': '#B0B4B8', 'axes.linewidth': 0.8})
NICE = {'LogisticRegression': 'Logistic regression', 'KNeighbours': 'k-nearest neighbours',
        'DecisionTree': 'Decision tree', 'RandomForest': 'Random forest',
        'AdaBoost': 'AdaBoost', 'GradientBoosting': 'Gradient boosting',
        'HistGradientBoosting': 'Hist. gradient boosting', 'Stacking': 'Stacking ensemble'}
COMP = {'flooding_only': 'Flooding only', 'flooding_gear_rpm': 'Flooding + gear + RPM spoofing',
        'all_four_with_fuzzing': 'All four, fuzzing included'}
REPN = {'raw10': 'Raw 10', 'eng8': 'Engineered 8', 'all18': 'All 18', 'raw9': 'Raw 9'}


def pm(s, k='f1', nd=2):
    return '%.*f $\\pm$ %.*f' % (nd, s[k]['mean'], nd, s[k]['sd'])


def num(x, nd=0):
    return '{:,.{}f}'.format(x, nd)


def write(name, text):
    open(os.path.join(TDIR, name), 'w').write(text)


def style(ax):
    ax.yaxis.grid(True, color='#E4E6E8', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# ===================================================================== Table 6 + Fig 6
E1 = R['E1CV']
models = list(E1)
rows = []
for m in models:
    c = E1[m]
    rows.append('%s & %s & %s & %s \\\\' % (
        NICE[m], *['%.2f $\\pm$ %.2f' % (c[r]['f1_mean'], c[r]['f1_sd']) for r in ('raw10', 'eng8', 'all18')]))
write('tab06.tex', '\n'.join(rows))

fig, ax = plt.subplots(figsize=(12.9, 5.6), dpi=240)
x = np.arange(len(models))
w = 0.27
for i, (rep, col) in enumerate((('raw10', GREY), ('eng8', TEAL), ('all18', BLUE))):
    v = [E1[m][rep]['f1_mean'] for m in models]
    sd = [E1[m][rep]['f1_sd'] for m in models]
    bars = ax.bar(x + (i - 1) * w, v, w, yerr=sd, capsize=2.5, label=REPN[rep], color=col,
                  edgecolor=EDGE, linewidth=0.6, error_kw=dict(ecolor=INK, lw=0.9, capthick=0.9))
    for b, vv in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, min(vv, 100) + 0.6, '%.1f' % vv, ha='center',
                va='bottom', fontsize=8, color=INK, rotation=90)
ax.set_ylabel('F1 (%)')
ax.set_xticks(x)
ax.set_xticklabels([NICE[m] for m in models], rotation=15, ha='right')
lo = min(E1[m][r]['f1_mean'] for m in models for r in ('raw10', 'eng8', 'all18'))
ax.set_ylim(max(0, np.floor(lo / 5) * 5 - 5), 106)
style(ax)
ax.legend(loc='lower right', frameon=True, framealpha=1, edgecolor='#C8CCD0', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FDIR, 'f06.png'), dpi=240, facecolor='white')
plt.close(fig)

# ===================================================================== Table 7 + Fig 7
E2 = R['E2']
rows = []
for c in ('flooding_only', 'flooding_gear_rpm', 'all_four_with_fuzzing'):
    d = E2[c]
    rows.append('%s & %s & %s & %s & %s & %s & %s \\\\' % (
        COMP[c], num(d['raw10']['attack_ids']['mean']), pm(d['raw10']), pm(d['raw10'], 'precision'),
        pm(d['eng8']), pm(d['eng8'], 'precision'), pm(d['all18'])))
FZ = R.get('FZNORM') or {}
if FZ:
    rows.append('Flooding + gear + RPM + legitimate frames of the fuzzing capture (injected frames removed) & %s & %s & %s & %s & %s & -- \\\\' % (
        num(FZ['raw10']['attack_ids']['mean']), pm(FZ['raw10']), pm(FZ['raw10'], 'precision'),
        pm(FZ['eng8']), pm(FZ['eng8'], 'precision')))
e4 = R['E4']
rows.append('\\midrule')
rows.append('Reverse direction (E4): KIA Soul $\\rightarrow$ Sonata, flooding only & 1 & %s & %s & %s & %s & -- \\\\' % (
    pm(e4['raw10']), pm(e4['raw10'], 'precision'), pm(e4['eng8']), pm(e4['eng8'], 'precision')))
write('tab07.tex', '\n'.join(rows))

fig, axs = plt.subplots(1, 2, figsize=(13.4, 5.4), dpi=240, sharey=True)
comps = ['flooding_only', 'flooding_gear_rpm', 'all_four_with_fuzzing', 'fznorm']
labels = ['Flooding\nonly', 'Flooding +\ngear + RPM', 'All four,\nfuzzing included',
          'Flooding + gear + RPM\n+ fuzzing capture,\nlegitimate frames only']
GROUPS = dict(E2)
GROUPS['fznorm'] = dict(FZ)
x = np.arange(len(comps))
for ax, key, title in ((axs[0], 'f1', 'F1'), (axs[1], 'precision', 'Precision')):
    for i, (rep, col) in enumerate((('raw10', GREY), ('eng8', TEAL), ('all18', BLUE))):
        xs = [xx + (i - 1) * 0.27 for xx, c in zip(x, comps) if rep in GROUPS[c]]
        v = [GROUPS[c][rep][key]['mean'] for c in comps if rep in GROUPS[c]]
        sd = [GROUPS[c][rep][key]['sd'] for c in comps if rep in GROUPS[c]]
        bars = ax.bar(xs, v, 0.27, yerr=sd, capsize=3, label=REPN[rep], color=col,
                      edgecolor=EDGE, linewidth=0.6, error_kw=dict(ecolor=INK, lw=0.9, capthick=0.9))
        for b, vv, ss in zip(bars, v, sd):
            ax.text(b.get_x() + b.get_width() / 2, vv + ss + 1.2, '%.1f' % vv, ha='center', va='bottom',
                    fontsize=7.5, color=INK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Cross-vehicle %s' % title, fontsize=12)
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    style(ax)
axs[0].set_ylabel('%')
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='upper center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FDIR, 'f07.png'), dpi=240, facecolor='white')
plt.close(fig)

# ===================================================================== Table 8 + Fig 8 (mechanism)
M = R['MECH']
M2 = R['MECH2']
MLAB = [('baseline_raw10', 'Raw 10, fuzzing frames unchanged'),
        ('legit_ids_random_payloads', 'Raw 10, fuzzing frames given legitimate identifiers (random payloads kept)'),
        ('random_ids_real_payloads', 'Raw 10, fuzzing frames given real DLC and payloads (random identifiers kept)')]
rows = []
for k, lab in MLAB:
    d = M[k]
    rows.append('%s & %s & %s & %s \\\\' % (lab, pm(d), pm(d, 'precision'), num(d['fp']['mean'])))
rows.append('\\midrule')
for rep, lab in (('id_dlc', 'Identifier + DLC only'), ('raw9', 'Raw 9: DLC + payload only')):
    for c, cl in (('flooding_gear_rpm', 'without fuzzing'), ('all_four_with_fuzzing', 'with fuzzing')):
        d = M2[c][rep]
        rows.append('%s, %s & %s & %s & %s \\\\' % (lab, cl, pm(d), pm(d, 'precision'), num(d['fp']['mean'])))
write('tab08.tex', '\n'.join(rows))

fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.9), dpi=240, gridspec_kw={'width_ratios': [4, 3]})
ax = axs[0]
short = ['Unchanged\n(raw 10)', 'Legitimate IDs,\nrandom payloads', 'Random IDs,\nreal payloads']
v = [M[k]['f1']['mean'] for k, _ in MLAB]
sd = [M[k]['f1']['sd'] for k, _ in MLAB]
p = [M[k]['precision']['mean'] for k, _ in MLAB]
xx = np.arange(3)
bars = ax.bar(xx - 0.19, v, 0.38, yerr=sd, capsize=3, color=GREY, edgecolor=EDGE, linewidth=0.6,
              error_kw=dict(ecolor=INK, lw=0.9, capthick=0.9), label='F1')
bars2 = ax.bar(xx + 0.19, p, 0.38, color='white', edgecolor=EDGE, linewidth=0.9, hatch='///',
               label='Precision')
for bb, vv in list(zip(bars, v)) + list(zip(bars2, p)):
    ax.text(bb.get_x() + bb.get_width() / 2, vv + 2, '%.1f' % vv, ha='center', fontsize=9, color=INK)
ax.set_xticks(xx)
ax.set_xticklabels(short, fontsize=9.5)
ax.set_ylabel('Cross-vehicle score (%)')
ax.set_title('Fuzzing frames modified (all four attacks in training)', fontsize=11)
ax.set_ylim(0, 112)
style(ax)
ax.legend(loc='upper right', frameon=True, framealpha=1, edgecolor='#C8CCD0', fontsize=9.5)
ax = axs[1]
reps = [('raw10', 'Raw 10'), ('id_dlc', 'Identifier\n+ DLC'), ('raw9', 'Raw 9\n(DLC + payload)')]
clean = [R['E2']['flooding_gear_rpm']['raw10']] + [M2['flooding_gear_rpm'][r] for r, _ in reps[1:]]
fuzz = [R['E2']['all_four_with_fuzzing']['raw10']] + [M2['all_four_with_fuzzing'][r] for r, _ in reps[1:]]
xx = np.arange(3)
for off, data, col, lab in ((-0.19, clean, TEAL, 'Without fuzzing'), (0.19, fuzz, GREY, 'With fuzzing')):
    vv = [d['f1']['mean'] for d in data]
    ss = [d['f1']['sd'] for d in data]
    bb = ax.bar(xx + off, vv, 0.38, yerr=ss, capsize=3, color=col, edgecolor=EDGE, linewidth=0.6,
                error_kw=dict(ecolor=INK, lw=0.9, capthick=0.9), label=lab)
    for b_, v_ in zip(bb, vv):
        ax.text(b_.get_x() + b_.get_width() / 2, v_ + 2, '%.1f' % v_, ha='center', fontsize=9, color=INK)
ax.set_xticks(xx)
ax.set_xticklabels([lab for _, lab in reps], fontsize=9.5)
ax.set_title('Fields removed from every frame (F1)', fontsize=11)
ax.set_ylim(0, 112)
style(ax)
ax.legend(loc='upper right', frameon=True, framealpha=1, edgecolor='#C8CCD0', fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FDIR, 'f08.png'), dpi=240, facecolor='white')
plt.close(fig)

# ===================================================================== Table 9 + Fig 9 (dose)
E3 = R['E3']
doses = ['0', '2', '5', '10', '25', '50', '100']
rows = []
for dz in doses:
    d = E3[dz]
    ff = d['fuzzing_frames']
    inj = round(ff * 491847 / 3838860)
    lab = 'none' if ff == 0 else '%s (%s)' % (num(ff), num(inj))
    rows.append('%s & %s & %s & %s & %s & %s \\\\' % (
        lab, num(d['raw10']['attack_ids']['mean']), pm(d['raw10']), pm(d['raw10'], 'precision'),
        pm(d['eng8']), pm(d['eng8'], 'precision')))
write('tab09.tex', '\n'.join(rows))

fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=240)
ff = [E3[d]['fuzzing_frames'] for d in doses]
xpos = np.arange(len(doses))
for ax, key, title in ((axs[0], 'f1', 'F1'), (axs[1], 'precision', 'Precision')):
    for rep, col, mk in (('raw10', GREY, 'o'), ('eng8', TEAL, 's')):
        mean = np.array([E3[d][rep][key]['mean'] for d in doses])
        sd = np.array([E3[d][rep][key]['sd'] for d in doses])
        vals = np.array([E3[d][rep][key]['values'] for d in doses])
        for j in range(vals.shape[1]):
            ax.plot(xpos, vals[:, j], color=col, alpha=0.18, lw=0.8)
        ax.errorbar(xpos, mean, yerr=sd, color=col, marker=mk, ms=6, lw=2, capsize=3,
                    label=REPN[rep], mec=EDGE, mew=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(['0'] + ['{:,}'.format(f) for f in ff[1:]], rotation=0)
    ax.set_xlabel('Frames from the fuzzing capture added to the 120,000-frame base')
    ax.set_title('Cross-vehicle %s (thin lines: individual draws)' % title, fontsize=11.5)
    ax.set_ylim(0, 105)
    style(ax)
axs[0].set_ylabel('%')
axs[0].legend(loc='lower left', frameon=True, framealpha=1, edgecolor='#C8CCD0', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FDIR, 'f09.png'), dpi=240, facecolor='white')
plt.close(fig)

# ===================================================================== Table 10 (E5)
E5 = R['E5']
rows = []
for m in ('LogisticRegression', 'DecisionTree', 'RandomForest', 'HistGradientBoosting'):
    first = True
    for rep in ('raw10', 'all18', 'eng8'):
        d = E5[m][rep]
        rows.append('%s & %s & %s & %.2f & %s & %s \\\\' % (
            NICE[m] if first else '', REPN[rep], pm(d, 'precision'), d['recall']['mean'], pm(d),
            num(d['fp']['mean'])))
        first = False
write('tab10.tex', '\n'.join(rows))

# ===================================================================== Table 11 (robustness)
rows = []


def rob_row(label, cell):
    g, f = cell['flooding_gear_rpm'], cell['all_four_with_fuzzing']
    return '%s & %s & %s & %s & %s \\\\' % (label, pm(g['raw10']), pm(f['raw10']), pm(g['eng8']), pm(f['eng8']))


if R.get('HYP'):
    for key, lab in (('t60_dnone', '60 trees, unlimited depth (default)'), ('t120_dnone', '120 trees'),
                     ('t300_dnone', '300 trees'), ('t60_d16', '60 trees, depth 16'), ('t60_d8', '60 trees, depth 8')):
        if key in R['HYP']:
            rows.append(rob_row(lab, R['HYP'][key]))
if R.get('BIG'):
    rows.append(rob_row('About 1,000,000 training frames', R['BIG']))
write('tab11.tex', '\n'.join(rows))

# ===================================================================== Table 12 (event-level)
EV = R.get('EVENT', {})
rows = []
if EV:
    for i, rule in enumerate(EV['eng8']['rules']):
        cells = []
        for rep in ('eng8', 'raw10'):
            r = EV[rep]['rules'][i]
            dl = r['detection_delay_s']
            cells += ['%.1f' % r['events_per_hour']['mean'], '%.2f' % r['alarmed_windows_pct']['mean'],
                      ('%.1f' % dl['mean']) if dl else 'not detected']
        rows.append('%.1f & %d & %s \\\\' % (rule['window_s'], int(round(100 * rule['tau'])), ' & '.join(cells)))
write('tab12.tex', '\n'.join(rows))

# ===================================================================== Table 13 (ablation + baselines)
rows = []
A = R.get('ABL') or {}
BA = R.get('BASE') or {}


def cell(d, k='f1'):
    return pm(d, k) if d else '--'


def nrow(lab, clean, fuzz, n):
    rows.append('%s & %s & %s & %s & %s & %s \\\\' % (
        lab, cell(clean), cell(fuzz), cell(fuzz, 'precision') if fuzz else '--',
        ('%.2f' % fuzz['recall']['mean']) if fuzz else '--', n))


if A:
    nrow('Engineered 8 (reference, Table~\\ref{tbl:7})', R['E2']['flooding_gear_rpm']['eng8'],
         R['E2']['all_four_with_fuzzing']['eng8'], 10)
    nrow('Timing and load only (IAT, IAT\\_ID, ID\\_Freq)', A['flooding_gear_rpm']['eng_timing'],
         A['all_four_with_fuzzing']['eng_timing'], 10)
    nrow('Payload dynamics only (five features)', A['flooding_gear_rpm']['eng_payload'],
         A['all_four_with_fuzzing']['eng_payload'], 10)
    rows.append('\\midrule')
if BA:
    g = lambda k: BA.get(k)
    n = min(v['f1']['n'] for v in BA.values())
    nrow('Raw 10, random forest tuned by 3-fold CV on the Sonata', g('tunedrf__flooding_gear_rpm__raw10'),
         g('tunedrf__all_four_with_fuzzing__raw10'), n)
    nrow('Raw 10, multilayer perceptron', g('mlp__flooding_gear_rpm__raw10'), g('mlp__all_four_with_fuzzing__raw10'), n)
    nrow('Engineered 8, multilayer perceptron', g('mlp__flooding_gear_rpm__eng8'), g('mlp__all_four_with_fuzzing__eng8'), n)
    for rep, lab in (('raw10', 'Raw 10'), ('eng8', 'Engineered 8')):
        d = g('iforest__benign__%s' % rep)
        if d:
            rows.append('%s, isolation forest on legitimate Sonata frames only (no attack labels) & -- & %s & %s & %.2f & %d \\\\' % (
                lab, pm(d), pm(d, 'precision'), d['recall']['mean'], n))
CL = R.get('CAL') or {}
for mode, ml in (('interleaved', 'interleaved calibration frames'), ('temporal', 'first 20\\% of the capture, tested on the rest')):
    for rep, lab in (('raw10', 'Raw 10'), ('eng8', 'Engineered 8')):
        d = CL.get(mode, {}).get(rep)
        if d:
            rows.append('%s, threshold calibrated on KIA Soul legitimate frames (%s) & -- & %s & %s & %.2f & %d \\\\' % (
                lab, ml, pm(d), pm(d, 'precision'), d['recall']['mean'], d['f1']['n']))
SO = R.get('SONG') or {}
if SO:
    rows.append('\\midrule')
    for m, lab in (('CNN', 'convolutional network'), ('MLP', 'multilayer perceptron'), ('RandomForest', 'random forest')):
        if m in SO:
            nrow('Song et al. 29-frame identifier grid, %s (per window)' % lab, SO[m].get('flooding_gear_rpm'),
                 SO[m].get('all_four_with_fuzzing'), SO[m]['flooding_gear_rpm']['f1']['n'])
write('tab13.tex', '\n'.join(rows))

# ===================================================================== Fig 10 (global Shapley)
EX = R.get('EXPLAIN')
if EX:
    phi = np.abs(np.array(EX['phi'])).mean(0)
    feats = EX['features']
    order = np.argsort(phi)
    fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=240)
    cols = [TEAL if i >= 10 else GREY for i in order]
    ax.barh(np.arange(len(order)), phi[order], color=cols, edgecolor=EDGE, linewidth=0.5)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([feats[i] for i in order])
    ax.set_xlabel('Mean |Shapley contribution| to P(attack)')
    for j, i in enumerate(order):
        ax.text(phi[i] + phi.max() * 0.01, j, '%.4f' % phi[i], va='center', fontsize=8.5, color=INK)
    ax.xaxis.grid(True, color='#E4E6E8', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TEAL, label='Engineered'), Patch(color=GREY, label='Raw')],
              loc='lower right', frameon=True, edgecolor='#C8CCD0')
    fig.tight_layout()
    fig.savefig(os.path.join(FDIR, 'f10.png'), dpi=240, facecolor='white')
    plt.close(fig)

# ===================================================================== supplementary CSVs


def wcsv(name, header, rows_):
    with open(os.path.join(SUPP, name), 'w', newline='', encoding='utf-8') as fh:
        w_ = csv.writer(fh)
        w_.writerow(header)
        w_.writerows(rows_)


# S: E2 per draw
rr = []
for c in ('flooding_only', 'flooding_gear_rpm', 'all_four_with_fuzzing'):
    for rep in ('raw10', 'eng8', 'all18'):
        d = E2[c][rep]
        for s in range(10):
            rr.append([COMP[c], REPN[rep], s, int(d['attack_ids']['values'][s]),
                       '%.2f' % d['f1']['values'][s], '%.2f' % d['precision']['values'][s],
                       '%.2f' % d['recall']['values'][s], int(d['fp']['values'][s])])
wcsv('rev3_e2_per_draw.csv', ['composition', 'representation', 'seed', 'attack_ids', 'f1',
                              'precision', 'recall', 'fp'], rr)
# S: paired tests
rr = []
for k, t in R['E2_tests'].items():
    rr.append(['E2', k, t['n'], '%.2f' % t['mean_diff'], '%.2f' % t['ci95'][0], '%.2f' % t['ci95'][1],
               '%.2f' % t['d_z'], '%.1f' % t['W'], '%.4f' % t['p']])
for k, t in R['MECH_tests'].items():
    rr.append(['E2 mechanism', k + ' minus unchanged', t['n'], '%.2f' % t['mean_diff'],
               '%.2f' % t['ci95'][0], '%.2f' % t['ci95'][1], '%.2f' % t['d_z'], '%.1f' % t['W'], '%.4f' % t['p']])
for m in ('LogisticRegression', 'DecisionTree', 'RandomForest', 'HistGradientBoosting'):
    for k in ('eng8_minus_all18', 'eng8_minus_raw10'):
        t = E5[m][k]
        rr.append(['E5 ' + NICE[m], k, t['n'], '%.2f' % t['mean_diff'], '%.2f' % t['ci95'][0],
                   '%.2f' % t['ci95'][1], '%.2f' % t['d_z'], '%.1f' % t['W'], '%.4f' % t['p']])
wcsv('rev3_paired_tests.csv', ['experiment', 'comparison', 'n', 'mean_difference', 'ci95_low',
                               'ci95_high', 'd_z', 'W', 'p'], rr)
# S: forward FP by identifier
rr = [[x['hex'], 'yes' if x['in_sonata_vocabulary'] else 'no', x['normal_frames'],
       '%.1f' % x['fp_mean'], '%.2f' % x['fp_rate_pct']] for x in R['E2_fp_by_id']['per_id']]
wcsv('rev3_forward_fp_by_id.csv', ['identifier', 'in_sonata_vocabulary', 'normal_frames_in_test',
                                   'false_positives_mean', 'fp_rate_pct'], rr)
# S: E3 per draw
rr = []
for dz in doses:
    for rep in ('raw10', 'eng8'):
        d = E3[dz][rep]
        for s in range(10):
            rr.append([E3[dz]['fuzzing_frames'], REPN[rep], s, int(d['attack_ids']['values'][s]),
                       '%.2f' % d['f1']['values'][s], '%.2f' % d['precision']['values'][s]])
wcsv('rev3_e3_per_draw.csv', ['fuzzing_frames', 'representation', 'seed', 'attack_ids', 'f1', 'precision'], rr)
# S: E5 per draw
rr = []
for m in ('LogisticRegression', 'DecisionTree', 'RandomForest', 'HistGradientBoosting'):
    for rep in ('raw10', 'all18', 'eng8'):
        d = E5[m][rep]
        for s in range(10):
            rr.append([NICE[m], REPN[rep], s, '%.2f' % d['precision']['values'][s],
                       '%.2f' % d['recall']['values'][s], '%.2f' % d['f1']['values'][s], int(d['fp']['values'][s])])
wcsv('rev3_e5_per_draw.csv', ['model', 'representation', 'seed', 'precision', 'recall', 'f1', 'fp'], rr)
# S: E1 splits
rr = []
for split in ('random', 'chronological', 'prevalence_matched'):
    for m in ('LogisticRegression', 'KNeighbours', 'DecisionTree', 'RandomForest', 'AdaBoost',
              'GradientBoosting', 'HistGradientBoosting', 'Stacking'):
        c = R['E1SPLIT'].get(split, {}).get(m)
        if c:
            rr.append([split.replace('_', ' '), NICE[m]] + ['%.2f' % c[r]['f1'] if r in c else ''
                                                          for r in ('raw10', 'eng8', 'all18')])
wcsv('rev3_e1_splits.csv', ['split', 'model', 'raw10_f1', 'eng8_f1', 'all18_f1'], rr)
# S: dedup
rr = []
for m, c in R['DEDUP']['E1random'].items():
    rr.append(['E1 random split', NICE[m]] + ['%.2f' % c.get(r, float('nan')) for r in ('raw10', 'eng8', 'all18')])
for cc, c in R['DEDUP']['E2'].items():
    rr.append(['E2 (10 draws)', COMP[cc]] + ['%.2f' % c['raw10']['f1']['mean'], '%.2f' % c['eng8']['f1']['mean'], ''])
wcsv('rev3_dedup.csv', ['experiment', 'model_or_composition', 'raw10_f1', 'eng8_f1', 'all18_f1'], rr)
# S: Shapley ranking and local
if EX:
    phi_ = np.array(EX['phi'])
    mabs = np.abs(phi_).mean(0)
    order = np.argsort(-mabs)
    rr = [[j + 1, EX['features'][i], 'Engineered' if i >= 10 else 'Raw', '%.4f' % mabs[i],
           '%.4f' % EX['impurity_importance'][i], '%.4f' % EX['permutation_importance_mean'][i],
           '%.4f' % EX['permutation_importance_std'][i]] for j, i in enumerate(order)]
    wcsv('rev3_global_importance.csv', ['rank', 'feature', 'origin', 'mean_abs_shapley',
                                        'impurity_importance', 'permutation_importance_mean',
                                        'permutation_importance_sd'], rr)
    rr = []
    for g in ('DoS', 'Fuzzy', 'gear', 'RPM', 'normal'):
        if g in EX['picks']:
            j = EX['picks'][g]
            for i in range(len(EX['features'])):
                rr.append([g, '%.3f' % EX['fx'][j], EX['features'][i], '%.4g' % EX['values'][j][i],
                           '%.4f' % phi_[j, i]])
    wcsv('rev3_local_shapley.csv', ['frame', 'p_attack', 'feature', 'value', 'contribution'], rr)
# S: mechanism per draw
rr = []
for k, lab in MLAB:
    d = M[k]
    for s_ in range(10):
        rr.append([lab, s_, '%.2f' % d['f1']['values'][s_], '%.2f' % d['precision']['values'][s_],
                   int(d['fp']['values'][s_])])
for rep, lab in (('id_dlc', 'Identifier + DLC only'), ('raw9', 'Raw 9: DLC + payload only')):
    for c, cl in (('flooding_gear_rpm', 'without fuzzing'), ('all_four_with_fuzzing', 'with fuzzing')):
        d = M2[c][rep]
        for s_ in range(10):
            rr.append(['%s, %s' % (lab, cl), s_, '%.2f' % d['f1']['values'][s_],
                       '%.2f' % d['precision']['values'][s_], int(d['fp']['values'][s_])])
wcsv('rev3_mech_per_draw.csv', ['variant', 'seed', 'f1', 'precision', 'fp'], rr)
# S: reverse FP by identifier
rr = [[x['hex'], '%.1f' % x['fp_mean'], x['n_draws_flagged']] for x in R['E4']['raw10']['fp_by_id']]
wcsv('rev3_reverse_fp_by_id.csv', ['identifier', 'false_positives_mean_per_draw', 'draws_with_any'], rr)
# S: trend tests
rr = [[k, '%.1f' % v['L'], '%.3g' % v['p']] for k, v in R['E3_trend'].items() if isinstance(v, dict)]
wcsv('rev3_e3_trend.csv', ['test', 'page_L', 'p'], rr)
# S: E1 CV folds
rr = []
for m in E1:
    for rep in ('raw10', 'eng8', 'all18'):
        rr.append([NICE[m], REPN[rep]] + ['%.2f' % f for f in E1[m][rep]['folds']] +
                  ['%.2f' % E1[m][rep]['f1_mean'], '%.2f' % E1[m][rep]['f1_sd']])
wcsv('rev3_e1_cv_folds.csv', ['model', 'representation', 'fold_1', 'fold_2', 'fold_3', 'fold_4',
                              'fold_5', 'mean', 'sd'], rr)
# S: split prevalence
pv = R.get('E1SPLIT_prevalence') or {}
rr = [[k.replace('_', ' '), v['n_train'], '%.2f' % v['train_attack_pct'], v['n_test'],
       '%.2f' % v['test_attack_pct']] for k, v in pv.items()]
wcsv('rev3_e1_split_prevalence.csv', ['split', 'train_frames', 'train_attack_pct', 'test_frames',
                                      'test_attack_pct'], rr)
# S: robustness
rr = []
for key, cell in (R.get('HYP') or {}).items():
    for c, reps in cell.items():
        for rep, d in reps.items():
            rr.append([key.replace('t', '', 1).replace('_d', ' trees, depth '), COMP[c], REPN[rep],
                       '%.2f' % d['f1']['mean'], '%.2f' % d['f1']['sd'], '%.2f' % d['precision']['mean']])
for c, reps in (R.get('BIG') or {}).items():
    for rep, d in reps.items():
        rr.append(['about 1,000,000 frames', COMP[c], REPN[rep], '%.2f' % d['f1']['mean'],
                   '%.2f' % d['f1']['sd'], '%.2f' % d['precision']['mean']])
wcsv('rev3_robustness.csv', ['setting', 'composition', 'representation', 'f1_mean', 'f1_sd',
                             'precision_mean'], rr)
# S: event rules
rr = []
for rep in ('eng8', 'raw10'):
    if rep in EV:
        for r in EV[rep]['rules']:
            dl = r['detection_delay_s']
            rr.append([REPN[rep], r['window_s'], '%.0f' % (100 * r['tau']),
                       '%.2f' % r['alarmed_windows_pct']['mean'], '%.1f' % r['events_per_hour']['mean'],
                       '%.1f' % r['events_per_hour']['sd'], ('%.2f' % dl['mean']) if dl else 'not detected',
                       r['draws_detected']])
wcsv('rev3_event_rules.csv', ['representation', 'window_s', 'threshold_pct', 'alarmed_windows_pct',
                              'events_per_hour_mean', 'events_per_hour_sd', 'detection_delay_s',
                              'draws_detected'], rr)
rr = []
for c, reps in (R.get('ABL') or {}).items():
    for rep, d in reps.items():
        for i_, v in enumerate(d['f1']['values']):
            rr.append(['engineered ablation', COMP[c], {'eng_payload': 'payload dynamics only', 'eng_timing': 'timing and load only'}[rep],
                       i_, '%.2f' % v, '%.2f' % d['precision']['values'][i_], '%.2f' % d['recall']['values'][i_]])
for k, d in (R.get('BASE') or {}).items():
    for i_, v in enumerate(d['f1']['values']):
        rr.append(['baseline', k.replace('__', ' / ').replace('_', ' '), '', i_, '%.2f' % v,
                   '%.2f' % d['precision']['values'][i_], '%.2f' % d['recall']['values'][i_]])
for mode, reps in (R.get('CAL') or {}).items():
    for rep, d in reps.items():
        for i_, v in enumerate(d['f1']['values']):
            rr.append(['target calibration, %s' % mode, 'all four (fuzzing included)', REPN.get(rep, rep), i_, '%.2f' % v,
                       '%.2f' % d['precision']['values'][i_], '%.2f' % d['recall']['values'][i_]])
for rep_, d in (R.get('FZNORM') or {}).items():
    for i_, v in enumerate(d['f1']['values']):
        rr.append(['control: fuzzing-capture legitimate frames only', 'flooding + gear + RPM + 30,000 legitimate fuzzing-capture frames',
                   REPN.get(rep_, rep_), i_, '%.2f' % v, '%.2f' % d['precision']['values'][i_], '%.2f' % d['recall']['values'][i_]])
for m, comps_ in (R.get('SONG') or {}).items():
    for c, d in comps_.items():
        for i_, v in enumerate(d['f1']['values']):
            rr.append(['Song et al. identifier grid (per window), %s' % m, COMP[c], 'identifier bits', i_, '%.2f' % v,
                       '%.2f' % d['precision']['values'][i_], '%.2f' % d['recall']['values'][i_]])
BT = R.get('BOOT')
if BT:
    for k in ('raw_drop', 'eng_change', 'raw_fuzz', 'eng_fuzz'):
        rr.append(['block bootstrap (mean, 95% interval)', k.replace('_', ' '), '', '', '%.2f' % BT[k]['mean'],
                   '%.2f' % BT[k]['ci95'][0], '%.2f' % BT[k]['ci95'][1]])
wcsv('rev3_review_checks.csv', ['check', 'setting', 'features', 'seed', 'f1_or_mean', 'precision_or_ci_low',
                                'recall_or_ci_high'], rr)
print('tables, figures and CSVs written')

# ===================================================================== second vehicle set (Survival Analysis)
SV = R.get('SURV')
if SV:
    CN = {'sonata': 'Sonata', 'soul': 'Soul', 'spark': 'Spark'}
    E2s, E3s = SV['E2'], SV['E3']
    pairs = [(a, b) for a in ('sonata', 'soul', 'spark') for b in ('sonata', 'soul', 'spark') if a != b]
    q = lambda d: '%.2f $\\pm$ %.2f' % (d['mean'], d['sd'])
    rows = [r'\multicolumn{5}{@{}l}{\textit{(a) False-positive rate on the target vehicle\'s free-driving capture}} \\',
            r'\textbf{Source $\rightarrow$ target} & \textbf{Raw 10, clean} & \textbf{Raw 10, fuzzing} & \textbf{Engineered 8, clean} & \textbf{Engineered 8, fuzzing} \\',
            r'\midrule']
    for a, b in pairs:
        g = lambda rep, comp, kind, m: E2s[a][rep][comp]['test'][b][kind][m]
        rows.append('%s $\\rightarrow$ %s & %s & %s & %s & %s \\\\' % (
            CN[a], CN[b], q(g('raw10', 'clean', 'free', 'fpr')), q(g('raw10', 'fuzz', 'free', 'fpr')),
            q(g('eng8', 'clean', 'free', 'fpr')), q(g('eng8', 'fuzz', 'free', 'fpr'))))
    rows += [r'\midrule',
             r'\multicolumn{5}{@{}l}{\textit{(b) Recall on the target vehicle\'s own attack captures, fuzzing in training}} \\',
             r'\textbf{Source $\rightarrow$ target} & \textbf{Fuzzing, raw 10} & \textbf{Fuzzing, engineered 8} & \textbf{Malfunction, raw 10} & \textbf{Malfunction, engineered 8} \\',
             r'\midrule']
    for a, b in pairs:
        g = lambda rep, kind: E2s[a][rep]['fuzz']['test'][b][kind]['recall']
        rows.append('%s $\\rightarrow$ %s & %s & %s & %s & %s \\\\' % (
            CN[a], CN[b], q(g('raw10', 'fuzzy')), q(g('eng8', 'fuzzy')), q(g('raw10', 'malf')), q(g('eng8', 'malf'))))
    write('tabsa.tex', '\n'.join(rows))

    doses = ['0', '500', '1000', '2500', '5000', '10000', '20000']
    xd = [0, 500, 1000, 2500, 5000, 10000, 20000]
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=240, sharey=True)
    cols = {('sonata', 'soul'): '#1f4e9c', ('soul', 'sonata'): '#2a9fd6', ('sonata', 'spark'): '#b2182b',
            ('soul', 'spark'): '#e08214', ('spark', 'sonata'): '#1b7837', ('spark', 'soul'): '#6a3d9a'}
    mk = {('sonata', 'soul'): ('o', '-', False), ('soul', 'sonata'): ('s', '--', True),
          ('sonata', 'spark'): ('^', '-', False), ('soul', 'spark'): ('v', '--', True),
          ('spark', 'sonata'): ('D', '-', False), ('spark', 'soul'): ('P', ':', True)}
    for ax, rep, title in ((axs[0], 'raw10', 'Raw 10'), (axs[1], 'eng8', 'Engineered 8')):
        for a, b in pairs:
            m = [E3s[a][rep][d]['test'][b]['free']['fpr']['mean'] for d in doses]
            sd = [E3s[a][rep][d]['test'][b]['free']['fpr']['sd'] for d in doses]
            ax.errorbar(xd, m, yerr=sd, marker=mk[(a, b)][0], ls=mk[(a, b)][1], ms=5.5, lw=1.5, capsize=2,
                        color=cols[(a, b)], mfc='white' if mk[(a, b)][2] else cols[(a, b)], mew=1.3,
                        label='%s $\\rightarrow$ %s' % (CN[a], CN[b]))
        ax.set_xscale('symlog', linthresh=500, linscale=0.6)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(xd); ax.set_xticklabels(['0', '500', '1k', '2.5k', '5k', '10k', '20k'])
        ax.set_xlabel('Frames added from the source vehicle\'s fuzzing capture')
        ax.set_title('%s: false alarms on target free driving' % title, fontsize=11.5)
        ax.set_ylim(-2, 58)
        style(ax)
    axs[0].set_ylabel('False-positive rate (%)')
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FDIR, 'f11.png'), dpi=240, facecolor='white')
    plt.close(fig)

    rr = []
    for a, b in pairs:
        for rep in ('raw10', 'eng8'):
            for comp in ('clean', 'fuzz'):
                S = E2s[a][rep][comp]
                T = S['test'][b]
                rr.append([CN[a], CN[b], REPN[rep], {'clean': 'flooding + malfunction', 'fuzz': 'flooding + malfunction + fuzzing'}[comp],
                           '%.2f' % T['free']['fpr']['mean'], '%.2f' % T['flood']['recall']['mean'], '%.2f' % T['flood']['fpr']['mean'],
                           '%.2f' % T['malf']['recall']['mean'], '%.2f' % T['malf']['fpr']['mean'],
                           '%.2f' % T['fuzzy']['recall']['mean'], '%.2f' % T['fuzzy']['fpr']['mean'],
                           '%.2f' % T['pooled']['f1']['mean'], '%.2f' % S['within_f1']['mean']])
    wcsv('rev3_sa_e2_summary.csv', ['source', 'target', 'features', 'training attacks', 'FPR free driving',
                                    'flood recall', 'flood FPR', 'malf recall', 'malf FPR', 'fuzzy recall', 'fuzzy FPR',
                                    'pooled F1', 'within F1'], rr)
    rr = []
    for a, b in pairs:
        for rep in ('raw10', 'eng8'):
            rr.append([CN[a], CN[b], REPN[rep]] + ['%.2f' % E3s[a][rep][d]['test'][b]['free']['fpr']['mean'] for d in doses]
                      + ['%.2f' % E3s[a][rep][d]['test'][b]['fuzzy']['recall']['mean'] for d in ('500', '20000')])
    wcsv('rev3_sa_e3.csv', ['source', 'target', 'features'] + ['FPR at %s' % d for d in doses]
         + ['fuzzy recall at 500', 'fuzzy recall at 20000'], rr)
    rr = []
    for a in E2s:
        for rep in ('raw10', 'eng8'):
            for comp in ('clean', 'fuzz'):
                for b, T in E2s[a][rep][comp]['test'].items():
                    for i_ in range(10):
                        rr.append([CN[a], CN[b], REPN[rep], comp, i_, '%.2f' % T['free']['fpr']['values'][i_],
                                   '%.2f' % T['flood']['recall']['values'][i_], '%.2f' % T['malf']['recall']['values'][i_],
                                   '%.2f' % T['fuzzy']['recall']['values'][i_], '%.2f' % T['pooled']['f1']['values'][i_]])
    wcsv('rev3_sa_e2_per_draw.csv', ['source', 'target', 'features', 'training', 'seed', 'FPR free driving',
                                     'flood recall', 'malf recall', 'fuzzy recall', 'pooled F1'], rr)
    print('second-vehicle-set tables written')
