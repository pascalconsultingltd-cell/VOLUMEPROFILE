"""V95 study: follow each PF V95 box, the last 25 held, marked to market.

V95 as the chart's pfHot: a box = a BTCUSDT minute's row (row = the PF box, a tenth of the day's vol, on FDUSD prices:
each second's mid x the FDUSD / USDT ratio smoothed over ~60 s); a V95 = a box whose volume is in the top 8 (5% of 150)
of the boxes of the 750 minutes before, itself included; its side = its taker delta's sign.
Trade: at the box minute's close, enter its side; each position closes when the 25th V95 after it enters (the 26th
pushes the oldest out). Flat: 100 FDUSD each; real: sized by the box's volume, 100 FDUSD on average.
Equity by the minute (open positions at the minute's close); gross and net of the 0.10% taker fee both ways.

Usage: python v95_study.py [first day] [last day] [--keep 25]"""
import sys, bisect, math, json, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, r'C:\Users\sebas\Documents\FILEROOT\WORK\BINANCE\src\simplified\GRID-TRADING')
import vipoc as V

VH_MS, WIN, PERC, TAKER, UNIT = 750 * 60000, 150, 0.95, 0.001, 100.0
NTOP = math.ceil((1 - PERC) * WIN)  # 8
HERE = Path(__file__).parent

def usdt(day):
    """BTCUSDT 1s [t, high, low, close, volume, taker buy volume]."""
    return np.load(V.DATA / f'BTCUSDT-1s-tb-{day}.npy')

def boxes(day):
    """A day's boxes in the chart's order: [minute, row, volume, delta], rows by first second within the minute."""
    u, f, st = usdt(day), V.kl('BTCFDUSD', '1s', day), max(1, math.ceil(V.vol_of(day) / 10))
    j = np.searchsorted(f[:, 0], u[:, 0], 'right') - 1  # the last FDUSD trade up to this second (same day)
    ra, e, x = np.where(j >= 0, f[np.maximum(j, 0), 4] / u[:, 3], np.nan), np.full(len(u), np.nan), None
    for n, r in enumerate(ra):
        if r != r: continue
        x = r if x is None else x + (r - x) / 60
        e[n] = x
    ok = ~np.isnan(e)
    u, e = u[ok], e[ok]
    m0, row = (u[:, 0] // 60000 * 60000).astype(np.int64), np.floor((u[:, 1] + u[:, 2]) / 2 * e / st).astype(np.int64)
    out, cur = [], {}
    for k in range(len(u)):
        if cur and m0[k] != m0[k - 1]:
            out += [[m0[k - 1], r, q, d] for r, (q, d) in cur.items()]; cur = {}
        q, d = cur.get(row[k], (0.0, 0.0))
        cur[row[k]] = (q + u[k, 4], d + 2 * u[k, 5] - u[k, 4])
    out += [[m0[-1], r, q, d] for r, (q, d) in cur.items()]
    return out, st

def v95(days):
    """[minute, row, step, volume, delta] of each V95 box, the window sliding over all the days in order."""
    seq = []
    for d in days:
        b, st = boxes(d)
        seq += [x[:2] + [st] + x[2:] for x in b]
        print(d, len(b), 'boxes, step', st, flush=True)
    w, j, out, t0 = [], 0, [], seq[0][0]
    for k, (t, r, st, q, d) in enumerate(seq):
        bisect.insort(w, q)
        while seq[j][0] <= t - VH_MS:
            w.pop(bisect.bisect_left(w, seq[j][3])); j += 1
        if t - t0 >= VH_MS and len(w) > NTOP and q >= w[-NTOP]: out.append([t, r, st, q, d])
    return out

def run(ev, keep, kind):
    """Follow each V95 at its minute's close, the last `keep` held: minute equity gross / net, trades."""
    px = {}  # minute -> FDUSD close
    days = sorted({V.iso(e[0]) for e in ev})
    for d in days:
        f = V.kl('BTCFDUSD', '1s', d)
        m = f[:, 0] // 60000 * 60000
        last = np.r_[np.flatnonzero(np.diff(m)), len(m) - 1]
        px.update(zip(m[last].astype(np.int64).tolist(), f[last, 4].tolist()))
    mins = np.array(sorted(px)); cl = np.array([px[t] for t in mins])
    qs = np.array([e[3] for e in ev]); scale = UNIT / np.mean(qs)
    pos, trades, real, fee = [], [], 0.0, 0.0
    eq, ef, k = np.zeros(len(mins)), np.zeros(len(mins)), 0
    for i, t in enumerate(mins):
        while k < len(ev) and ev[k][0] == t:  # the box's minute closes now: enter
            s = 1 if ev[k][4] >= 0 else -1
            fd = UNIT if kind == 'flat' else ev[k][3] * scale
            pos.append(dict(t=t, px=cl[i], s=s, q=fd / cl[i]))
            fee += TAKER * fd
            if len(pos) > keep:
                p = pos.pop(0)
                g = p['s'] * p['q'] * (cl[i] - p['px'])
                real += g; fee += TAKER * p['q'] * cl[i]
                trades.append(dict(t0=int(p['t']), t1=int(t), s=p['s'], pnl=g))
            k += 1
        op = sum(p['s'] * p['q'] * (cl[i] - p['px']) for p in pos)
        eq[i], ef[i] = real + op, real + op - fee
    return mins, eq, ef, trades, len(pos)

def fade(ev, keep, kind, ttl=60):
    """Fade each V95 with a maker limit at its box's edge (green: sell at its top, red: buy at its bottom; at the minute's
    close if price is already past it), good for ttl minutes; filled when a second trades through it. The last `keep`
    fills held, the oldest closed at market (taker). Same output as run()."""
    days = sorted({V.iso(e[0]) for e in ev} | {V.iso(e[0] + ttl * 60000 + 60000) for e in ev})
    f = np.concatenate([V.kl('BTCFDUSD', '1s', d) for d in days])
    scale, fills = UNIT / np.mean([e[3] for e in ev]), []
    for t, r, st, q, d in ev:
        a = np.searchsorted(f[:, 0], t + 60000)
        b = np.searchsorted(f[:, 0], t + 60000 + ttl * 60000)
        c = f[a - 1, 4]
        s = -1 if d >= 0 else 1  # fade: green sells, red buys
        px = max((r + 1) * st, c) if s < 0 else min(r * st, c)
        w = f[a:b]
        hit = np.flatnonzero(w[:, 2] > px) if s < 0 else np.flatnonzero(w[:, 3] < px)
        if len(hit): fills.append((int(w[hit[0], 0]) // 60000 * 60000, px, s, (UNIT if kind == 'flat' else q * scale) / px))
    fills.sort()
    m = f[:, 0] // 60000 * 60000
    last = np.r_[np.flatnonzero(np.diff(m)), len(m) - 1]
    mins, cl = m[last].astype(np.int64), f[last, 4]
    pos, trades, real, fee, k = [], [], 0.0, 0.0, 0
    eq, ef = np.zeros(len(mins)), np.zeros(len(mins))
    for i, t in enumerate(mins):
        while k < len(fills) and fills[k][0] == t:
            tt, px, s, qq = fills[k]
            pos.append(dict(t=t, px=px, s=s, q=qq))
            if len(pos) > keep:
                p = pos.pop(0)
                g = p['s'] * p['q'] * (cl[i] - p['px'])
                real += g; fee += TAKER * p['q'] * cl[i]
                trades.append(dict(t0=int(p['t']), t1=int(t), s=p['s'], pnl=g))
            k += 1
        op = sum(p['s'] * p['q'] * (cl[i] - p['px']) for p in pos)
        eq[i], ef[i] = real + op, real + op - fee
    print(f"  fade {kind}: {len(fills)} of {len(ev)} limits filled")
    return mins, eq, ef, trades, len(pos)

def stats(mins, eq, trades, name):
    dd = float(np.max(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]))
    day = mins // V.DAY
    ends = np.r_[np.flatnonzero(np.diff(day)), len(day) - 1]
    de = np.diff(np.r_[0, eq[ends]])
    dr = []  # daily RoMaD: the day's PnL over its own DD
    for a, b in zip(np.r_[0, ends[:-1] + 1], ends + 1):
        x = eq[a:b] - (eq[a - 1] if a else 0)
        ddd = float(np.max(np.maximum.accumulate(np.r_[0, x]) - np.r_[0, x]))
        dr.append(x[-1] / max(ddd, 1))
    lo = sum(t['pnl'] for t in trades if t['s'] > 0); sh = sum(t['pnl'] for t in trades if t['s'] < 0)
    op = eq[-1] - lo - sh if name.endswith('gross') else float('nan')
    print(f"  {name:16} PnL {eq[-1]:9.1f} (long {lo:8.1f}, short {sh:8.1f}, open {op:7.1f})  DD {dd:7.1f}"
          f"  RoMaD {eq[-1] / max(dd, 1):6.2f}  daily RoMaD mean {np.mean(dr):5.2f} worst {min(dr):6.2f} best {max(dr):5.2f}"
          f"  days won {int((de > 0).sum())}/lost {int((de < 0).sum())}  best day {de.max():7.1f} worst {de.min():7.1f}")
    return dict(pnl=round(float(eq[-1]), 1), dd=round(dd, 1))

if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    keep = int(sys.argv[sys.argv.index('--keep') + 1]) if '--keep' in sys.argv else 25
    d0, d1 = (a + ['2026-03-20', '2026-11-07'])[:2]
    days = [V.iso(t) for t in range(V.day0(d0), V.day0(d1) + 1, V.DAY)]
    if '--cached' in sys.argv: ev = json.load(open(HERE / 'v95_events.json'))
    else:
        ev = v95(days)
        ev = [e for e in ev if e[0] >= V.day0(days[0]) + VH_MS]
        json.dump([[int(e[0]), int(e[1]), int(e[2]), float(e[3]), float(e[4])] for e in ev], open(HERE / 'v95_events.json', 'w'))
    if '--fade' in sys.argv:
        print(f"\n{len(ev)} V95 boxes; fade with maker limits at the box edge (60 min), last {keep} held, closed at market")
        for kind in ('flat', 'real'):
            mins, eq, ef, tr, n = fade(ev, keep, kind)
            print(f"{kind}: {len(tr)} closed, {n} open, won {sum(t['pnl'] > 0 for t in tr)}/{len(tr)}")
            stats(mins, eq, tr, kind + ' gross'); stats(mins, ef, tr, kind + ' net')
            mo = {}
            for t in tr: mo[V.iso(t['t1'])[:7]] = mo.get(V.iso(t['t1'])[:7], 0) + t['pnl']
            print('   by month (gross):', '  '.join(f'{m} {v:+.0f}' for m, v in sorted(mo.items())))
        sys.exit()
    up = sum(e[4] >= 0 for e in ev)
    print(f"\n{len(ev)} V95 boxes ({up} up, {len(ev) - up} down), {len(ev) / len(days):.1f} a day; follow, last {keep} held")
    for kind in ('flat', 'real'):
        mins, eq, ef, tr, n = run(ev, keep, kind)
        print(f"{kind}: {len(tr)} closed, {n} open, won {sum(t['pnl'] > 0 for t in tr)}/{len(tr)}")
        stats(mins, eq, tr, kind + ' gross'); stats(mins, ef, tr, kind + ' net')
        mo = {}
        for t in tr: mo[V.iso(t['t1'])[:7]] = mo.get(V.iso(t['t1'])[:7], 0) + t['pnl']
        print('   by month (gross):', '  '.join(f'{m} {v:+.0f}' for m, v in sorted(mo.items())))
