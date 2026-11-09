"""Each live bell scored by one neutral grid, to sort ranging bells from the others: levels every 3.92 sigma / N from
P - 1.96 sigma to P + 1.96 sigma (sigma = vol / 1.2816, the curve's), one unit a level (1/N, so the grid's size is the
same for every N); long below P, short above, a fill each time the 1s close crosses the next level. From the bell's
birth to the next bell's (ranged), or to the first 1s close past +/-1.96 sigma (broke: all closed there). Score = PnL in
sigma (realised + open at the end). No fees: a category, not a strategy.
Which N? The one whose score says most about the next bell: a category that holds is a market state.

Usage: python bell_grid_n.py   (gridopt/bells_6m.json, the 1s cache of GRID-TRADING/vipoc.py)"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, r'C:\Users\sebas\Documents\FILEROOT\WORK\BINANCE\src\simplified\GRID-TRADING')
import vipoc as V
from scipy.stats import spearmanr

Z, NS = 1.96, (4, 6, 8, 10, 12, 14, 16, 20)
bells = json.load(open(Path(__file__).parent / 'bells_6m.json'))
day = {}
def secs(t0, t1):
    """1s [t, close] from t0 to t1."""
    out = []
    for d in range(t0 // V.DAY * V.DAY, t1, V.DAY):
        if d not in day:
            if len(day) > 3: day.pop(min(day))
            k = V.kl('BTCFDUSD', '1s', V.iso(d)); day[d] = k[:, [0, 4]]
        out.append(day[d])
    a = np.concatenate(out)
    return a[(a[:, 0] >= t0) & (a[:, 0] < t1)]

def grid(b, c, n):
    """(score in sigma, broke, fills) of one bell's grid on its closes c."""
    sg = b['sigma']; d = 2 * Z * sg / n; L = b['px'] - Z * sg + d * np.arange(n + 1)
    k = int(np.clip(np.round((c[0] - L[0]) / d), 0, n))  # the level the start is nearest: its position held
    pos, cash, fills = (n / 2 - k) / n, -(n / 2 - k) / n * c[0], 0  # bought / sold at the start price
    out = np.flatnonzero((c > L[-1]) | (c < L[0]))
    br = len(out) > 0
    c = c[:out[0] + 1] if br else c
    f = np.floor((c - L[0]) / d)  # only the closes where the level band changes can fill
    for x in c[np.r_[0, np.flatnonzero(np.diff(f)) + 1]]:
        while k > 0 and x <= L[k - 1]: k -= 1; pos += 1 / n; cash -= L[k] / n; fills += 1
        while k < n and x >= L[k + 1]: k += 1; pos -= 1 / n; cash += L[k] / n; fills += 1
    return (cash + pos * c[-1]) / sg, br, fills

bs = sorted(bells, key=lambda b: b['birth'])
bs = [b for i, b in enumerate(bs) if not (i + 1 < len(bs) and bs[i + 1]['birth'] == b['birth'])]  # born together: the last one
res = {n: [] for n in NS}
for i, b in enumerate(bs[:-1]):
    c = secs(b['birth'], bs[i + 1]['birth'])[:, 1]
    if len(c) < 2: continue
    for n in NS: res[n].append(grid(b, c, n))
    if i % 100 == 0: print(i, len(bs), b['day'], flush=True)
print(f'\n{len(res[NS[0]])} bells, {V.iso(bs[0]["birth"])} to {V.iso(bs[-1]["birth"])}; score = grid PnL in sigma, no fees')
print('   N  broke  score>0  mean   ranged mean  broke mean   next-bell corr  next ranged AUC')
for n in NS:
    s = np.array([r[0] for r in res[n]]); br = np.array([r[1] for r in res[n]])
    rho = spearmanr(s[:-1], s[1:]).correlation
    y, x = ~br[1:], s[:-1]  # does this score say the next bell ranges (no break)?
    pos, neg = x[y], x[~y]
    auc = (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    print(f'{n:4} {br.mean() * 100:5.0f}%  {(s > 0).mean() * 100:5.0f}%  {s.mean():+.3f}   {s[~br].mean():+.3f}      {s[br].mean():+.3f}'
          f'          {rho:+.3f}           {auc:.3f}')
