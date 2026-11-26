# -*- coding: utf-8 -*-
"""Graphical abstract in the draw.io design, corrected, in two shapes.

Matches the supplied Graphical_Abstract_drawio.pdf: Liberation Serif, same
panels, palette and wording. Six values are corrected against the current
manuscript:

  legend   "CAN-SHIELD engineered 8" -> "Engineered 8"   (name dropped in v10)
  panel a  1,887 -> 1,738 attack identifiers             (count at 120,000 frames)
  panel b  97.35 -> 97.43,  43.66 -> 45.11,  97.82 -> 97.34,  -56.34 -> -54.89
           (panel (a) states 120,000 frames in both sets, so panel (b) must use
            the constant-size five-draw figures of Table 8, not the 160,000-frame
            dose-sweep endpoints)

Panel (c) is unchanged; its values are the dose sweep and were already correct.

Two layouts:
  a4    the supplied portrait layout, (a) in the top half, (b)+(c) below
  wide  13.28 x 5.31 in, the 2.5:1 ratio Elsevier asks for (1328 x 531 px min.)

Usage: python3 -I make_ga_serif.py <a4|wide> <out.pdf>
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LAYOUT, OUT, RES = sys.argv[1], sys.argv[2], sys.argv[3]
import json
R = json.load(open(RES))
_g, _f = R['E2']['flooding_gear_rpm'], R['E2']['all_four_with_fuzzing']
_d = R['E2_tests']['raw10_clean_minus_fuzz']
_k3 = sorted(R['E3'], key=lambda x: int(x))

RED, BLUE, INK, GREY = '#B2182B', '#2166AC', '#000000', '#444444'
BOX = {'blue':   ('#DAE8FC', '#6C8EBF'),
       'orange': ('#FFE6CC', '#D79B00'),
       'yellow': ('#FFF2CC', '#D6B656'),
       'purple': ('#E1D5E7', '#9673A6'),
       'green':  ('#D5E8D4', '#82B366')}

# Liberation Serif has no matplotlib math font set, and it lacks U+207B
# (superscript minus), so every symbol below is a literal glyph and the one
# exponent is drawn as a raised run. Nothing here uses mathtext, which keeps
# the figure to a single typeface.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'axes.unicode_minus': False,
    'axes.edgecolor': '#000000', 'axes.linewidth': 0.9,
    'pdf.fonttype': 42,
})

if LAYOUT == 'a4':
    SPEC = dict(
        figsize=(8.27, 11.69),
        head_a=(0.085, 0.918), head_b=(0.085, 0.442), head_c=(0.545, 0.442),
        ax_a=[0.06, 0.550, 0.90, 0.340],
        ax_b=[0.125, 0.125, 0.325, 0.275],
        ax_c=[0.615, 0.125, 0.325, 0.275],
        legend_y=0.035, leg_x=(0.145, 0.590),
        f_head=12.5, f_box=10.0, f_line=9.0, f_note=8.5,
        f_tick=9.0, f_lab=10.0, f_val=8.6, f_leg=10.0, gap=0.047)
elif LAYOUT == 'wide':
    SPEC = dict(
        figsize=(13.28, 5.31),
        head_a=(0.030, 0.955), head_b=(0.400, 0.955), head_c=(0.700, 0.955),
        ax_a=[0.008, 0.175, 0.375, 0.740],
        ax_b=[0.445, 0.195, 0.205, 0.690],
        ax_c=[0.745, 0.195, 0.205, 0.690],
        legend_y=0.015, leg_x=(0.150, 0.560),
        f_head=11.5, f_box=9.2, f_line=7.8, f_note=7.8,
        f_tick=8.2, f_lab=9.2, f_val=8.0, f_leg=9.4, gap=0.030)
else:
    raise SystemExit('layout must be a4 or wide')

fig = plt.figure(figsize=SPEC['figsize'], dpi=300)


def head(pos, letter, text):
    x, y = pos
    fig.text(x, y, '(%s)' % letter, fontsize=SPEC['f_head'],
             fontweight='bold', va='center')
    fig.text(x + SPEC['gap'], y, text, fontsize=SPEC['f_head'], va='center')


# ----------------------------------------------------------------- panel (a)
head(SPEC['head_a'], 'a', 'Cross-vehicle evaluation protocol')
axa = fig.add_axes(SPEC['ax_a']); axa.axis('off')
axa.set_xlim(0, 100); axa.set_ylim(0, 44)

# The flow diagram is wide in the portrait layout and tall in the landscape
# one, so the five boxes are placed on a shared 100 x 44 grid per layout.
if LAYOUT == 'a4':
    P = dict(setA=(2, 25, 29, 15), setB=(2, 3, 29, 17),
             feat=(39, 22.5, 26, 19), clf=(39, 5, 26, 12.5),
             test=(72, 14, 27, 16),
             cap=(16.5, 0.2), note=(85.5, 36.5), aspect=0.45)
else:
    P = dict(setA=(0.5, 25, 32, 17), setB=(0.5, 2, 32, 19),
             feat=(38.5, 23, 27, 19), clf=(38.5, 3.5, 27, 14),
             test=(71, 13, 29, 18),
             cap=(16.5, -1.2), note=(85.5, 36.5), aspect=0.22)


def box(rect, colour, title, lines):
    x, y, w, h = rect
    fc, ec = BOX[colour]
    axa.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle='round,pad=0,rounding_size=1.0',
                                 linewidth=1.1, facecolor=fc, edgecolor=ec,
                                 mutation_aspect=P['aspect']))
    cx = x + w / 2
    axa.text(cx, y + h - 3.4, title, ha='center', va='center',
             fontsize=SPEC['f_box'], fontweight='bold', color=INK)
    axa.text(cx, y + (h - 6.8) / 2 + 0.4, '\n'.join(lines), ha='center',
             va='center', fontsize=SPEC['f_line'], color=INK, linespacing=1.55)


def arrow(x1, y1, x2, y2):
    axa.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                  mutation_scale=12, linewidth=1.3,
                                  color=INK, shrinkA=0, shrinkB=0))


box(P['setA'], 'blue', 'Training set A',
    ['Hyundai YF Sonata', 'flooding + gear/RPM spoofing', '3 attack identifiers'])
box(P['setB'], 'orange', 'Training set B',
    ['Hyundai YF Sonata', 'A + fuzzing subsample', '(identifiers randomised)',
     '≈ {:,} attack identifiers'.format(int(round(_f['raw10']['attack_ids']['mean'])))])
box(P['feat'], 'yellow', 'Feature sets',
    ['raw 10: CAN ID +', 'payload bytes', '', 'engineered 8: timing,',
     'ID load, payload dynamics'])
box(P['clf'], 'purple', 'Classifier',
    ['Random Forest', '10 draws (seeds 0–9)'])
box(P['test'], 'green', 'Test (unseen vehicle)',
    ['KIA Soul (OTIDS)', '2,573,117 frames', 'different manufacturer'])

ax_, ay_, aw_, ah_ = P['setA']; bx_, by_, bw_, bh_ = P['setB']
fx_, fy_, fw_, fh_ = P['feat']; cx_, cy_, cw_, ch_ = P['clf']
tx_, ty_, tw_, th_ = P['test']
arrow(ax_ + aw_, ay_ + ah_ / 2, fx_, ay_ + ah_ / 2)
arrow(bx_ + bw_, by_ + bh_ / 2, fx_, cy_ + ch_ / 2)
arrow(fx_ + fw_ / 2, fy_, fx_ + fw_ / 2, cy_ + ch_)
joint = tx_ + tw_ / 2
axa.plot([cx_ + cw_, joint], [cy_ + ch_ / 2, cy_ + ch_ / 2], color=INK,
         lw=1.3, solid_capstyle='butt')
arrow(joint, cy_ + ch_ / 2, joint, ty_)
axa.text(*P['cap'], s='120,000 frames in both sets', ha='center', va='center',
         fontsize=SPEC['f_note'], style='italic', color=GREY)
axa.text(*P['note'], s='train on one vehicle,\ntest on another', ha='center',
         va='center', fontsize=SPEC['f_note'], style='italic', color=GREY,
         linespacing=1.55)

# ----------------------------------------------------------------- panel (b)
head(SPEC['head_b'], 'b', 'Cross-vehicle F1 (mean ± s.d., n = 10)')
axb = fig.add_axes(SPEC['ax_b'])
raw, raw_sd = [_g['raw10']['f1']['mean'], _f['raw10']['f1']['mean']], [_g['raw10']['f1']['sd'], _f['raw10']['f1']['sd']]
eng, eng_sd = [_g['eng8']['f1']['mean'], _f['eng8']['f1']['mean']], [_g['eng8']['f1']['sd'], _f['eng8']['f1']['sd']]
x = np.arange(2); w = 0.30
for off, vals, sds, col in ((-w / 2, raw, raw_sd, RED), (w / 2, eng, eng_sd, BLUE)):
    bars = axb.bar(x + off, vals, w, yerr=sds, capsize=3, color=col,
                   edgecolor=INK, linewidth=0.8,
                   error_kw=dict(ecolor=INK, lw=1.0, capthick=1.0))
    for bar, v, s in zip(bars, vals, sds):
        axb.text(bar.get_x() + bar.get_width() / 2, v + s + 2.2, '%.2f' % v,
                 ha='center', va='bottom', fontsize=SPEC['f_val'], color=INK)
axb.set_ylabel('F1-score (%)', fontsize=SPEC['f_lab'])
axb.set_xticks(x)
axb.set_xticklabels(['Set A\n(no fuzzing)', 'Set B\n(fuzzing included)'],
                    fontsize=SPEC['f_tick'], linespacing=1.5)
axb.set_ylim(0, 124); axb.set_yticks([0, 20, 40, 60, 80, 100])
axb.tick_params(labelsize=SPEC['f_tick'])
for sp in ('top', 'right'):
    axb.spines[sp].set_visible(False)

# Bracket spanning the two raw-feature bars, in the empty corridor between the
# groups, with its label above the panel so nothing collides.
axb.plot([0.50, 0.50], [raw[1], raw[0]], color=GREY, lw=0.9, zorder=4)
for yy in (raw[1], raw[0]):
    axb.plot([0.455, 0.545], [yy, yy], color=GREY, lw=0.9, zorder=4)
axb.text(0.5, 1.035, '−%.2f pts,  Wilcoxon p = %.3f' % (_d['mean_diff'], _d['p']),
         transform=axb.transAxes, fontsize=SPEC['f_val'], color=GREY,
         va='bottom', ha='center')

# ----------------------------------------------------------------- panel (c)
head(SPEC['head_c'], 'c', 'Dose-response to fuzzing frames')
axc = fig.add_axes(SPEC['ax_c'])
rawc = [R['E3'][k]['raw10']['f1']['mean'] for k in _k3]
rawc_sd = [R['E3'][k]['raw10']['f1']['sd'] for k in _k3]
engc = [R['E3'][k]['eng8']['f1']['mean'] for k in _k3]
engc_sd = [R['E3'][k]['eng8']['f1']['sd'] for k in _k3]
xi = np.arange(7)
axc.errorbar(xi, rawc, yerr=rawc_sd, color=RED, marker='o', ms=5.2, lw=1.6,
             capsize=3, mec=INK, mew=0.7, zorder=3)
axc.errorbar(xi, engc, yerr=engc_sd, color=BLUE, marker='s', ms=5.2, lw=1.6,
             capsize=3, mec=INK, mew=0.7, zorder=3)
axc.set_xticks(xi)
axc.set_xticklabels(['0', '0.8k', '2k', '4k', '10k', '20k', '40k'],
                    fontsize=SPEC['f_tick'])
axc.set_xlabel('Fuzzing frames added to training set', fontsize=SPEC['f_lab'])
axc.set_ylabel('F1-score (%)', fontsize=SPEC['f_lab'])
axc.set_ylim(0, 124); axc.set_yticks([0, 20, 40, 60, 80, 100])
axc.tick_params(labelsize=SPEC['f_tick'])
for sp in ('top', 'right'):
    axc.spines[sp].set_visible(False)
axc.text(0.5, 0.235, 'Monotonic decline', transform=axc.transAxes,
         ha='center', va='center', fontsize=SPEC['f_lab'], color=INK)
axc.text(0.5, 0.145, "Exact Page's trend test, p < 0.001", transform=axc.transAxes,
         ha='center', va='center', fontsize=SPEC['f_lab'], color=INK)

# ----------------------------------------------------------------- legend
lax = fig.add_axes([0, SPEC['legend_y'], 1, 0.045]); lax.axis('off')
for lx, col, txt in ((SPEC['leg_x'][0], RED, 'Raw features (CAN ID + payload bytes)'),
                     (SPEC['leg_x'][1], BLUE, 'Engineered 8 (protocol-level)')):
    lax.add_patch(plt.Rectangle((lx, 0.32), 0.024, 0.40, facecolor=col,
                                edgecolor=INK, lw=0.8, transform=lax.transAxes))
    lax.text(lx + 0.034, 0.53, txt, fontsize=SPEC['f_leg'], va='center',
             transform=lax.transAxes)

fig.savefig(OUT, format='pdf', facecolor='white')
fig.savefig(OUT[:-4] + '.png', dpi=300, facecolor='white')
print('wrote', OUT)
