# -*- coding: utf-8 -*-
"""Order of the supplementary tables, figures and sections, shared by the supplement
builder and the manuscript build, so 'Table S7' in either document is always the
same table. Every id must appear exactly once."""

TABLES = ['e1_summary', 'e1_folds', 'e1_splits', 'e1_prev', 'shap_global', 'shap_local',
          'e2_draws', 'paired', 'mech_draws', 'fp_forward', 'fp_reverse', 'e3_draws', 'e3_trend', 'sa_full', 'sa_dose', 'sa_draws',
          'e5_draws', 'robust', 'event', 'dedup', 'corr', 'dup_audit', 'size_audit', 'id_overlap',
          'otids_audit', 'review']
FIGURES = ['within', 'shap']
SECTIONS = ['within', 'shap', 'iforest', 'draws']


def snum(kind, key):
    lst = {'T': TABLES, 'F': FIGURES, 'S': SECTIONS}[kind]
    if key not in lst:
        raise KeyError('%s:%s' % (kind, key))
    return 'S%d' % (lst.index(key) + 1)


def resolve(text):
    """@@SUP:T:id@@ -> S7 etc."""
    import re
    return re.sub(r'@@SUP:([TFS]):([a-z0-9_]+)@@', lambda m: snum(m.group(1), m.group(2)), text)
