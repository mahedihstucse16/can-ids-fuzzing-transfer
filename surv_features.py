# -*- coding: utf-8 -*-
"""The eight protocol-level features, as in the original pipeline (window 64), vectorised
where possible. eng(ts, ids, data) -> (n, 8) in the order
IAT, IAT_ID, ID_Freq, Payload_Entropy, Payload_Hamming, NonZero_Bytes, Data_Mean, Data_Std."""
from collections import defaultdict, deque

import numpy as np

WINDOW = 64


def eng(ts, ids, data):
    n = len(ts)
    iat = np.zeros(n); iat[1:] = np.diff(ts)
    iat_id = np.zeros(n); ham = np.zeros(n)
    last_t, last_p = {}, {}
    idf = np.zeros(n)
    win, cnt = deque(), defaultdict(int)
    dl = data.tolist(); il = ids.tolist(); tl = ts.tolist()
    for k in range(n):
        c = il[k]
        if c in last_t:
            iat_id[k] = tl[k] - last_t[c]
            ham[k] = sum(1 for a, b in zip(dl[k], last_p[c]) if a != b)
        last_t[c] = tl[k]; last_p[c] = dl[k]
        win.append(c); cnt[c] += 1
        if len(win) > WINDOW:
            o = win.popleft(); cnt[o] -= 1
            if cnt[o] == 0:
                del cnt[o]
        idf[k] = cnt[c]
    s = np.sort(data, axis=1)
    newv = np.ones_like(s, dtype=bool); newv[:, 1:] = s[:, 1:] != s[:, :-1]
    # entropy of the 8 byte values
    ent = np.zeros(n)
    for k in range(n):
        _, c = np.unique(s[k], return_counts=True)
        p = c / 8.0
        ent[k] = -(p * np.log2(p)).sum()
    nz = (data != 0).sum(1)
    return np.column_stack([iat, iat_id, idf, ent, ham, nz, data.mean(1), data.std(1)])
