# -*- coding: utf-8 -*-
"""Assemble the revision-3 supplementary material.

The Data availability statement commits to "the per-seed dose-response and
classifier-generality tables, the five-draw transfer summary, the balancing
control, the duplication and identifier-overlap audits and the full five-frame
local Shapley decomposition", and several table captions say "per-seed values
are in the supplementary material". This builds that document from the v2 run's
CSV outputs plus the matched-balancing CSV produced for Table 8.

Table, figure and section numbers come from rev3_supp_ids (shared with the manuscript
build); references to the manuscript (@@REF:label@@) are resolved from its .aux file,
and numbers (<<key>>) from numbers.json.

Usage: python3 -I rev3_supplementary.py <results_v2_dir> <rev3_csv_dir> <out.tex> <title>
       <numbers.json> <manuscript.aux> <build_dir>
"""
import csv
import os
import re
import sys

V2, EXTRA, OUT, TITLE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
NUMS, AUX, BUILD = sys.argv[5], sys.argv[6], sys.argv[7]
import json  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev3_numbers  # noqa: E402
import rev3_supp_ids as SID  # noqa: E402
NUMBERS = json.load(open(NUMS))
LABELS = {}
for ln in open(AUX, encoding='utf-8', errors='replace'):
    m = re.match(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', ln)
    if m:
        LABELS[m.group(1)] = m.group(2)


def T(text):
    """Resolve manuscript references, supplementary numbers and result numbers."""
    def ref(m):
        if m.group(1) not in LABELS:
            raise SystemExit('unknown manuscript label %s' % m.group(1))
        return LABELS[m.group(1)]
    text = re.sub(r'@@REF:([^@]+)@@', ref, text)
    text = SID.resolve(text)
    text = rev3_numbers.fill(text, NUMBERS)
    if '@@' in text or 'PENDING' in text:
        raise SystemExit('unresolved marker in supplement: %r' % text[:200])
    return text

SPECIAL = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
           '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
           '^': r'\textasciicircum{}'}


def esc(s):
    s = str(s).replace('\\', r'\textbackslash{}')
    s = ''.join(SPECIAL.get(c, c) for c in s)
    return s.replace('->', r'$\rightarrow$').replace('+/-', r'$\pm$')


def pretty(h):
    h = h.replace('_', ' ').strip()
    fix = {'f1': 'F1', 'sd': 's.d.', 'std': 's.d.', 'pct': '%', 'ids': 'IDs',
           'id': 'ID', 'cv': 'CV', 'otids': 'OTIDS', 'ch': 'Car-Hacking',
           'eng8': 'Engineered 8', 'raw10': 'Raw 10', 'all18': 'All 18',
           'p': 'p', 'n': 'n', 'tp': 'TP', 'fp': 'FP', 'fn': 'FN', 'tn': 'TN'}
    out = []
    for w in h.split():
        out.append(fix.get(w.lower(), w))
    s = ' '.join(out)
    return s[0].upper() + s[1:] if s else s


NUM = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')


VALUE = {'no_fuzzing': 'No fuzzing (dose 0)', 'with_fuzzing': 'With fuzzing (dose 40,000)',
         'raw10': 'Raw 10', 'eng8': 'Engineered 8', 'all18': 'All 18',
         'prevalence_matched': 'Prevalence matched'}


def fmt(v):
    v = (v or '').strip()
    if not NUM.match(v):
        # the method name used in early runs was dropped from the paper
        v = re.sub(r'CAN-SHIELD (engineered )?', lambda m: 'Engineered ' if m.group(1) else 'All ', v)
        return esc(VALUE.get(v, v))
    f = float(v)
    if f == int(f) and abs(f) < 1e15 and '.' not in v and 'e' not in v.lower():
        return '{:,}'.format(int(f))
    a = abs(f)
    if a != 0 and (a < 1e-4 or a >= 1e7):
        m = '%.3e' % f
        mant, ex = m.split('e')
        return r'$%s \times 10^{%d}$' % (mant, int(ex))
    if a >= 1000:
        return '{:,.1f}'.format(f)
    if a >= 1:
        return '%.2f' % f
    return '%.4f' % f


def read(path):
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def table(tid, path, caption, landscape=False, longt=False, transpose=False,
          size=r'\footnotesize', drop=()):
    hdr, rows = read(path)
    keep = [i for i, h in enumerate(hdr) if h not in drop]
    hdr = [hdr[i] for i in keep]
    rows = [[r[i] if i < len(r) else '' for i in keep] for r in rows]
    if transpose:
        body = [[esc(pretty(hdr[i])), fmt(rows[0][i])] for i in range(len(hdr))]
        hdr, rows = ['Quantity', 'Value'], body
        cols = 'lr'
        cells = [[c[0], c[1]] for c in rows]
    else:
        cols = 'l' + 'r' * (len(hdr) - 1)
        cells = [[fmt(c) for c in r] for r in rows]
    head = ' & '.join(r'\textbf{%s}' % esc(pretty(h)) for h in hdr) + r' \\'
    L = []
    if landscape:
        L.append(r'\begin{landscape}')
    env = 'longtable' if longt else 'table'
    if longt:
        L += [r'\begingroup\setlength{\tabcolsep}{2.5pt}', size,
              r'\begin{longtable}{%s}' % cols,
              r'\caption{%s}\label{st:%s}\\' % (caption, tid), r'\toprule', head, r'\midrule',
              r'\endfirsthead', r'\toprule', head, r'\midrule', r'\endhead',
              r'\bottomrule', r'\endfoot']
        for r in cells:
            L.append(' & '.join(r) + r' \\')
        L += [r'\end{longtable}', r'\endgroup']
    else:
        L += [r'\begin{table}[htbp]', r'\centering', r'\caption{%s}\label{st:%s}' % (caption, tid), size,
              r'\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%',
              r'\begin{tabular}{%s}' % cols, r'\toprule', head, r'\midrule']
        for r in cells:
            L.append(' & '.join(r) + r' \\')
        L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    if landscape:
        L.append(r'\end{landscape}')
    L.append('')
    return '\n'.join(L)



S = os.path.join(V2, 'supplementary')   # unchanged data audits (no model fitted)
X = EXTRA                                # revision-3 model results

TAB = {
    'e1_folds': (os.path.join(X, 'rev3_e1_cv_folds.csv'),
                 'Within-vehicle five-fold cross-validation, every fold; the source of Table~@@SUP:T:e1_summary@@ '
                 'and Figure~@@SUP:F:within@@.', dict(longt=True, size=r'\scriptsize')),
    'e1_splits': (os.path.join(X, 'rev3_e1_splits.csv'),
                  'Within-vehicle F1 under the random, chronological and prevalence-matched splits '
                  '(Section~@@SUP:S:within@@).', dict(longt=True, size=r'\scriptsize')),
    'e1_prev': (os.path.join(X, 'rev3_e1_split_prevalence.csv'),
                'Attack prevalence of each within-vehicle split.', dict(size=r'\scriptsize')),
    'shap_global': (os.path.join(X, 'rev3_global_importance.csv'),
                    'Global feature importance of the within-vehicle all-18 random forest: mean absolute '
                    'Monte Carlo Shapley contribution (250 held-out frames, 200 permutations each), '
                    'impurity importance and permutation importance (Section~@@SUP:S:shap@@).',
                    dict(size=r'\scriptsize')),
    'shap_local': (os.path.join(X, 'rev3_local_shapley.csv'),
                   'Local Shapley decomposition of one representative frame per attack class and one '
                   'normal frame, each the medoid of its group (Section~@@SUP:S:shap@@).',
                   dict(longt=True, size=r'\scriptsize')),
    'e2_draws': (os.path.join(X, 'rev3_e2_per_draw.csv'),
                 'Composition ablation (experiment E2), every draw. Random forest, 120,000 training '
                 'frames, seeds 0--9, tested on the 2,573,117-frame OTIDS set; the source of Table~@@REF:tbl:7@@ '
                 'and Figure~@@REF:fig:7@@.', dict(longt=True, size=r'\scriptsize')),
    'paired': (os.path.join(X, 'rev3_paired_tests.csv'),
               'Paired comparisons over ten draws: one-sided Wilcoxon signed-rank test (two-sided '
               'where no direction was predicted), mean paired difference in F1 points with its 95\\% '
               'confidence interval, and the standardised effect size $d_z$.',
               dict(landscape=True, size=r'\scriptsize')),
    'mech_draws': (os.path.join(X, 'rev3_mech_per_draw.csv'),
                   'Mechanism variants of experiment E2 and the reduced-representation controls, every '
                   'draw; the source of Table~@@REF:tbl:8@@ and Figure~@@REF:fig:8@@ (Section~@@REF:sec:4-4@@).',
                   dict(longt=True, size=r'\scriptsize')),
    'fp_forward': (os.path.join(X, 'rev3_forward_fp_by_id.csv'),
                   'Forward-direction false positives by identifier: raw 10, fuzzing included, mean over '
                   'ten draws. "In Sonata vocabulary" marks identifiers that also occur in Car-Hacking '
                   'normal traffic (Section~@@REF:sec:4-3@@).', dict(longt=True, size=r'\scriptsize')),
    'fp_reverse': (os.path.join(X, 'rev3_reverse_fp_by_id.csv'),
                   'Reverse-direction false positives by identifier (experiment E4, raw 10), mean per draw '
                   'over ten draws, with the number of draws in which each identifier drew any '
                   '(Section~@@REF:sec:4-6@@).', dict(size=r'\scriptsize')),
    'e3_draws': (os.path.join(X, 'rev3_e3_per_draw.csv'),
                 'Dose-response (experiment E3), every draw; the source of Table~@@REF:tbl:9@@ and '
                 'Figure~@@REF:fig:9@@.', dict(longt=True, size=r'\scriptsize')),
    'e3_trend': (os.path.join(X, 'rev3_e3_trend.csv'),
                 "Dose-response trend tests: Page's L over seven doses with the ten draws as blocks; "
                 'p-values are from the exact null distribution.', dict(size=r'\scriptsize')),
    'sa_full': (os.path.join(X, 'rev3_sa_e2_summary.csv'),
                'Second vehicle set (HCRL Survival Analysis), experiment SA-E2, mean over ten draws: false-positive '
                'rate on the target vehicle\'s free-driving capture; recall and false-positive rate on its flooding, '
                'malfunction and fuzzing captures; F1 pooled over its four captures; and within-vehicle F1 on held-out '
                'source frames (Section~@@REF:sec:4-5b@@).', dict(longt=True, landscape=True, size=r'\scriptsize')),
    'sa_dose': (os.path.join(X, 'rev3_sa_e3.csv'),
                'Second vehicle set, experiment SA-E3: false-positive rate on the target\'s free-driving capture by '
                'number of fuzzing-capture frames added to a 60,000-frame flooding + malfunction base, mean over ten '
                'draws; the source of Figure~@@REF:fig:11@@.', dict(landscape=True, size=r'\scriptsize')),
    'sa_draws': (os.path.join(X, 'rev3_sa_e2_per_draw.csv'),
                 'Second vehicle set, experiment SA-E2, every draw; the source of Table~@@REF:tbl:sa@@.',
                 dict(longt=True, size=r'\scriptsize')),
    'e5_draws': (os.path.join(X, 'rev3_e5_per_draw.csv'),
                 'Classifier generality (experiment E5), every draw; the source of Table~@@REF:tbl:10@@.',
                 dict(longt=True, size=r'\scriptsize')),
    'robust': (os.path.join(X, 'rev3_robustness.csv'),
               'Robustness of the key contrast: forest size and depth (three draws each) and about '
               '1,000,000 training frames (three draws); the source of Table~@@REF:tbl:11@@.',
               dict(longt=True, size=r'\scriptsize')),
    'event': (os.path.join(X, 'rev3_event_rules.csv'),
              'Event-level alarms on the complete OTIDS captures in time order, every aggregation '
              'rule, mean over ten draws; the source of Table~@@REF:tbl:12@@.',
              dict(longt=True, landscape=True, size=r'\scriptsize')),
    'dedup': (os.path.join(X, 'rev3_dedup.csv'),
              'De-duplication sensitivity: the 891,068-frame tail shared by the four Car-Hacking '
              'captures counted once (Section~@@REF:sec:6@@).', dict(size=r'\scriptsize')),
    'corr': (os.path.join(S, 'table05b_feature_correlation.csv'),
             'Feature correlation matrix, the numerical form of Figure~@@REF:fig:5@@.',
             dict(landscape=True, size=r'\tiny')),
    'dup_audit': (os.path.join(S, 'tableA1_duplication_audit.csv'),
                  'Duplication audit. Identical trailing rows shared between the four Car-Hacking '
                  'captures (Section~@@REF:sec:6@@).', {}),
    'size_audit': (os.path.join(S, 'tableA2_corpus_size_audit.csv'),
                   'Corpus size audit. Exact duplicate frames as a fraction of the published corpus.',
                   dict(transpose=True)),
    'id_overlap': (os.path.join(S, 'tableA3_identifier_overlap.csv'),
                   'Identifier overlap between the two vehicles, by Car-Hacking capture (Sections~@@REF:sec:3-2@@ '
                   'and~@@REF:sec:4-3@@).', dict(size=r'\scriptsize')),
    'otids_audit': (os.path.join(X, 'table12_otids_label_audit.csv'),
                    'OTIDS label audit. Neither documented labelling rule for the fuzzy and '
                    'impersonation captures yields reproducible per-frame labels, which is why only '
                    'the denial-of-service capture is used (Section~@@REF:sec:3-2@@).', dict(size=r'\scriptsize')),
    'review': (os.path.join(X, 'rev3_review_checks.csv'),
               'Feature ablation, stronger baselines, the fuzzing-capture control, the Song et al. '
               'identifier-grid detectors and target calibration, every draw, and the block bootstrap of the '
               'headline drop over 10,000-frame blocks of the test set; the source of Table~@@REF:tbl:13@@ '
               'and of the control row of Table~@@REF:tbl:7@@ (Sections~@@REF:sec:4-3@@ and~@@REF:sec:4-8b@@).',
               dict(longt=True, landscape=True, size=r'\scriptsize')),
}


def summary_table():
    rows = open(os.path.join(BUILD, 'tables', 'tab06.tex'), encoding='utf-8').read().strip()
    return '\n'.join([
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Within-vehicle F1, five-fold stratified cross-validation (50,000 frames, seed 42), mean '
        r'$\pm$ standard deviation over folds. All values per cent.}\label{st:e1_summary}',
        r'\footnotesize', r'\begin{tabular}{llll}', r'\toprule',
        r'\textbf{Model} & \textbf{Raw 10} & \textbf{Engineered 8} & \textbf{All 18} \\', r'\midrule',
        rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def figure(fid, fname, width, caption):
    return '\n'.join([r'\begin{figure}[htbp]', r'\centering',
                      r'\includegraphics[width=%s\linewidth]{%s}' % (width, os.path.join(BUILD, 'figs', fname)),
                      r'\caption{%s}\label{sf:%s}' % (caption, fid), r'\end{figure}', ''])


SEC_WITHIN = r"""\section{Within-vehicle evaluation}\label{ss:within}

This section holds the within-vehicle results summarised in Section~@@REF:sec:4-2@@ of the manuscript.
Table~@@SUP:T:e1_summary@@ reports five-fold cross-validated F1 on Car-Hacking for the three representations,
and Fig.~@@SUP:F:within@@ plots the same values. Six of the eight classifiers were carried through
cross-validation, with gradient boosting and the stacking ensemble omitted for computational cost. Every
non-linear model reaches at least <<e1_raw_nonlin_min>>\% on the raw 10, and the random forest reaches
<<e1cv_rf_raw>>\%. The engineered 8 give the non-linear models between <<e1_eng_nonlin_min>>\% and
<<e1_eng_nonlin_max>>\%; for the random forest they cost <<e1cv_rf_cost>> points (<<e1cv_rf_eng>>\%), the
within-vehicle price of withholding the identifier and payload bytes, and for no non-linear model does
the cost exceed <<e1cv_cost_max>> points. Logistic regression is the exception, reaching
<<e1cv_lr_raw>>\% on the raw 10 against <<e1cv_lr_eng>>\% on the engineered 8: the raw fields encode
categories, not magnitudes, and a linear model cannot use them.

Two properties of the within-vehicle protocol make these figures optimistic rather than conservative.
The windowed features are computed over each complete capture before the data are split, so ID\_Freq
and IAT\_ID for a test frame draw on neighbouring training frames, and 16\% of the corpus consists of
exact duplicate frames. Both inflate the random split and the cross-validated results; the chronological
and prevalence-matched splits are the partial remedy. Neither affects cross-vehicle transfer, where
training and test traffic come from different vehicles.

The saturation is not an artefact of random splitting. Under a chronological split, in which the first
70\% of each capture by timestamp trains and the last 30\% tests, the random forest attains
<<sp_chron_rf_raw>>\% F1 on the raw 10. That split carries a confound, because attacks concentrate in the
earlier part of each capture: its test partition is <<sp_chron_test_att>>\% attack against
<<sp_rand_test_att>>\% for the random split, and F1 depends on prevalence. Rebalancing the chronological
test partition to <<sp_prev_test_att>>\% removes the confound. Under that prevalence-matched split every
non-linear model still scores at least <<sp_nl_prev_min>>\% on the raw 10 and the random forest
<<sp_prev_rf_raw>>\% (Tables~@@SUP:T:e1_splits@@ and~@@SUP:T:e1_prev@@).
"""

SEC_SHAP = r"""\section{What the within-vehicle model relies on}\label{ss:shap}

To check what a detector trained on all 18 features relies on, we estimate Shapley values for the
within-vehicle random forest (random split, test F1 <<x_f1>>\%), as defined for SHAP~[1], by Monte Carlo permutation sampling over
<<x_nframes>> held-out, class-balanced frames. Each frame receives <<x_nperm>> permutations, each paired
with a different frame from a 200-frame background of real, class-balanced training frames
(<<x_evals>> model evaluations), so contributions sum exactly to $f(x) - E[f(X)]$, with
$E[f(X)] = <<x_base>>$. Fig.~@@SUP:F:shap@@ gives the global ranking. The raw identifier ranks first
(<<x_v1>> mean absolute contribution), ahead of <<x_rank2>> (<<x_v2>>) and <<x_rank3>> (<<x_v3>>), and
engineered features hold <<x_eng_top7>> of the top seven positions. Permutation importance~[2] ranks
<<x_p1>> (<<x_pv1>>) and <<x_p2>> (<<x_pv2>>) first. <<x_imp_feat>> carries the largest impurity-based
importance (<<x_imp_val>>) but ranks only <<x_imp_shap_rank>> by Shapley value, the signature of
correlated-feature masking by ID\_Freq; impurity rankings alone would misattribute what the model relies
on. The identifier's role depends on the attack: its mean contribution is <<x_id_dos>> for flood frames
and <<x_id_fuzz>> for fuzzing frames, which the model recognises by their payload dynamics instead. Both
are properties of how one vehicle's traffic looks, and both change when the vehicle changes. The full
ranking under all three measures is in Table~@@SUP:T:shap_global@@ and per-frame decompositions in
Table~@@SUP:T:shap_local@@.
"""

SEC_IF = r"""\section{Why the benign-only isolation forest misses the flood}\label{ss:iforest}

The isolation forest of Section~@@REF:sec:4-8b@@ of the manuscript is trained on legitimate Sonata frames only, with its threshold set for 1\% false alarms on held-out Sonata frames, and detects almost none of the KIA Soul flood. The anomaly scores of the first draw explain why. On the raw 10, every KIA Soul flood frame (identifier 0x000, all-zero payload) scores <<if_raw_flood>>, below the median legitimate Sonata frame (<<if_raw_valmed>>; threshold <<if_raw_thr>>): <<if_id002>>\% of legitimate Sonata frames carry the neighbouring identifier 0x002 and <<if_zero>>\% have all-zero payloads, so the flood frame lies inside dense legitimate traffic. On the engineered 8, the flood's windowed identifier count (median <<if_idfreq_kia>>) lies <<if_idfreq_z>> standard deviations above the legitimate mean but within the range of the legitimate training frames (maximum <<if_idfreq_max>>), and its other features are unremarkable, so the median flood frame scores <<if_eng_flood>> against a threshold of <<if_eng_thr>>. The same threshold flags <<if_eng_kia_norm>>\% of KIA Soul legitimate frames, close to its 1\% target, which confirms that the score direction is as intended.
"""

PRE = r"""\documentclass[a4paper,11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=20mm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{array}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{placeins}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thesection}{S\arabic{section}}
\setlength{\tabcolsep}{4pt}

\title{Supplementary material\\[4pt]
\large @@TITLE@@}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-10mm}

\noindent
This document contains the within-vehicle evaluation and the Shapley analysis summarised in the
manuscript, and the per-draw results, statistical tests, robustness checks and audits it refers to.
Every model-based table was produced by \texttt{rev3\_pipeline.py} under one protocol
(training-partition scaling and class weighting for every representation, scikit-learn 1.9.1, ten
draws unless stated), and every table is generated by the released code from the public dataset
files; nothing is transcribed by hand. Table and figure numbers of the manuscript are quoted as they
appear there. Tables~@@SUP:T:dup_audit@@--@@SUP:T:otids_audit@@ are data audits that involve no
model, and Table~@@SUP:T:corr@@ is the numerical form of a manuscript figure.

\listoftables
\listoffigures
\newpage
"""


def tab(tid):
    path, cap, kw = TAB[tid]
    if not os.path.exists(path):
        raise SystemExit('missing supplementary source %s' % path)
    return table(tid, path, T(cap), **kw) + '\n\\FloatBarrier\n'


parts = [T(PRE.replace('@@TITLE@@', TITLE))]
order = []
parts.append(T(SEC_WITHIN))
parts.append(T(summary_table())); order.append('e1_summary')
parts.append(T(figure('within', 'f06.png', '0.95', 'Within-vehicle five-fold cross-validated F1 for the three '
                      'representations. Error bars are one standard deviation over folds.')))
for tid in ('e1_folds', 'e1_splits', 'e1_prev'):
    parts.append(tab(tid)); order.append(tid)
parts.append(r'\clearpage')
parts.append(T(SEC_SHAP))
parts.append(T(figure('shap', 'f10.png', '0.75', 'Global feature importance of the within-vehicle all-18 '
                      'random forest by Monte Carlo Shapley sampling (mean absolute contribution to '
                      'P(attack)). Engineered features in teal, raw in grey.')))
for tid in ('shap_global', 'shap_local'):
    parts.append(tab(tid)); order.append(tid)
parts.append(r'\clearpage')
parts.append(T(SEC_IF))
parts.append(r'\section{Per-draw results, tests and audits}\label{ss:draws}')
for tid in SID.TABLES:
    if tid not in order:
        parts.append(tab(tid)); order.append(tid)
if order != SID.TABLES:
    raise SystemExit('supplementary table order differs from rev3_supp_ids: %s' % order)
parts.append(r'''\section*{References cited in the supplementary material}
\begin{itemize}\setlength{\itemsep}{2pt}
\item[{[1]}] S.M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: Advances in Neural
Information Processing Systems 30 (NIPS 2017), 2017, pp. 4765--4774.
\item[{[2]}] A. Fisher, C. Rudin, F. Dominici, All models are wrong, but many are useful: learning a variable\textquoteright s
importance by studying an entire class of prediction models simultaneously, Journal of Machine Learning Research 20 (177)
(2019) 1--81.
\end{itemize}''')
parts.append(r'\end{document}')
open(OUT, 'w', encoding='utf-8').write('\n'.join(parts))
print('wrote %s with %d tables' % (OUT, len(order)))
