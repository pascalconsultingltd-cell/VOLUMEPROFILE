"""Short-only grid optimiser on the chart's bells (vipoc normal curves).

For each bell (bells.json, exported from the chart): from its birth to its exit
(price out of the curve, first point to last point, with no re-entry before the
next bell's birth), every short grid inside the curve is simulated on BTCFDUSD 1s
candles and scored by RoMaD.

Grid: sell levels from bottom B to top T every step s (sigma units), each sell
bought back one step lower and re-armed (maker both ways, fee 0; a fill needs
price to trade through the level, not just touch it); a sell only
rests while price is below it. Shorts open at the exit are bought back at market
(taker 0.10%). Capital 1500 FDUSD split over the levels: flat, heavier at the top
or heavier at the bottom. DD on equity marked at each minute's high (worst case
for shorts).

Constraints (against fitting each bell): by default a step of at most 0.5 sigma,
2 sells or more, flat size, the first sell no higher than +1 sigma. --free lifts
them all; --max-step, --min-sells, --max-first, --sizes flat,top,bottom set each.

Held out: for each bell, the shape (sells in sigma from the P) that did best on
the other bells, scored on that bell: what the rule would have made live.

Rolling: each bell from the R-th on, with the N best shapes of the R bells before
it (no look ahead): #1 alone, and the N averaged.

Usage: python grid_opt.py [bells.json] [--top 5] [--free] [--max-step 0.5]
       [--min-sells 2] [--max-first 1] [--sizes flat] [--roll 10] [--roll-top 10]
"""
import io, json, sys, zipfile, datetime as dt
from pathlib import Path
import numpy as np, requests

HERE = Path(__file__).parent
DATA = HERE / "data"
KL = Path(r"C:\Users\sebas\Documents\FILEROOT\WORK\BINANCE\src\simplified\data\kl")
CAP, TAKER = 1500.0, 0.001
DD_FLOOR = 5.0  # RoMaD = PnL / max(DD, this): a near-zero DD does not win alone
STEPS = [0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
ALLOCS = ["flat", "top", "bottom"]
DAY = 86_400_000
# constraints (see the docstring); set from the command line in main()
STOP = [None]  # --stop X: live exit, X sigma out of the curve ("next": none)
PAGE = ["results.html"]  # --free writes results_free.html
CON = dict(max_step=0.5, min_sells=2, max_first=1.0, sizes=["flat"])


def day_secs(day):
    """BTCFDUSD 1s candles of a UTC day: [t ms, high, low, close]; cached."""
    DATA.mkdir(exist_ok=True)
    f = DATA / f"BTCFDUSD-1s-{day}.npy"
    if f.exists():
        return np.load(f)
    if KL.joinpath(f.name).exists():  # the bot's cache (GRID-TRADING/vipoc.py): t, open, high, low, close, volume
        return np.load(KL / f.name)[:, [0, 2, 3, 4]]
    url = f"https://data.binance.vision/data/spot/daily/klines/BTCFDUSD/1s/BTCFDUSD-1s-{day}.zip"
    r = requests.get(url, timeout=60)
    if r.ok:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = np.loadtxt(z.open(z.namelist()[0]), delimiter=",", usecols=(0, 2, 3, 4))
        if raw[0, 0] > 1e14:  # microseconds since 2025
            raw[:, 0] //= 1000
        np.save(f, raw)
        return raw
    # not published yet (today): the REST API, not cached
    t0 = int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    out, t = [], t0
    while t < min(t0 + DAY, int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)):
        j = requests.get("https://data-api.binance.vision/api/v3/klines", timeout=30, params=dict(
            symbol="BTCFDUSD", interval="1s", limit=1000, startTime=t, endTime=t0 + DAY - 1)).json()
        if not j:
            break
        out += [[k[0], float(k[2]), float(k[3]), float(k[4])] for k in j]
        t = j[-1][0] + 1000
    return np.array(out)


def secs(t0, t1):
    """1s candles from t0 to t1 (ms)."""
    if t1 <= t0:
        return np.zeros((0, 4))
    d0 =dt.datetime.fromtimestamp(t0 / 1000, dt.timezone.utc).date()
    d1 = dt.datetime.fromtimestamp((t1 - 1) / 1000, dt.timezone.utc).date()
    days = [d0 + dt.timedelta(k) for k in range((d1 - d0).days + 1)]
    a = np.concatenate([day_secs(d.isoformat()) for d in days])
    return a[(a[:, 0] >= t0) & (a[:, 0] < t1)]


def window(b, nxt, a):
    """Exit index: the end of the last time inside the curve before the next
    bell's birth (no re-entry after it); None when still inside at the end.
    With STOP[0] (live): the first close X sigma out of the curve, else the next
    birth ("next": the next birth only)."""
    c = a[:, 3]
    if STOP[0] is not None:
        x = (STOP[0] if STOP[0] != "next" else 1e9) * b["sigma"]
        out = np.flatnonzero((c > b["hi"] + x) | (c < b["lo"] - x))
        return int(out[0]) + 1 if len(out) else (None if nxt is None else len(c))
    inside = (c >= b["lo"]) & (c <= b["hi"])
    if not inside.any():
        return 1
    last = np.flatnonzero(inside)[-1]
    if last == len(c) - 1 and nxt is None:
        return None  # still in the bell (not finished)
    return last + 1


class Walk:
    """The window's 1s highs / lows / closes with 256s block extremes: the next
    second crossing a price is found without scanning the whole window."""
    BS = 256

    def __init__(self, a):
        self.hi, self.lo, self.c = a[:, 1], a[:, 2], a[:, 3]
        nb = -(-len(a) // self.BS)
        pad = lambda x, v: np.r_[x, np.full(nb * self.BS - len(x), v)].reshape(nb, self.BS)
        self.bhi = pad(self.hi, -np.inf).max(1)
        self.blo = pad(self.lo, np.inf).min(1)
        self.bc = pad(self.c, np.inf).min(1)
        self.cache = {}

    def _next(self, x, bx, pos, ok):
        """First index >= pos where ok(x) (ok monotone on block extremes bx)."""
        bs, b = self.BS, pos // self.BS
        k = np.flatnonzero(ok(x[pos:(b + 1) * bs]))
        if len(k):
            return pos + k[0]
        nb = np.flatnonzero(ok(bx[b + 1:]))
        if not len(nb):
            return None
        b2 = b + 1 + nb[0]
        return b2 * bs + np.flatnonzero(ok(x[b2 * bs:(b2 + 1) * bs]))[0]

    def trips(self, L, s):
        """Round trips of one level (sell L, buy back L - s): [(i_open, i_close or None)].
        The sell rests only once price is below L."""
        key = (round(L, 2), round(s, 2))
        if key in self.cache:
            return self.cache[key]
        out, pos = [], self._next(self.c, self.bc, 0, lambda v: v < L)
        while pos is not None:
            i = self._next(self.hi, self.bhi, pos + 1, lambda v: v > L)
            if i is None:
                break
            j = self._next(self.lo, self.blo, i + 1, lambda v: v < L - s)
            out.append((i, j))
            pos = j
        self.cache[key] = out
        return out


def simulate(levels, w, s, pa, mins, mhi, curve=False):
    """PnL, max DD (minute highs), trips, open shorts at the end; curve: also the
    equity per minute (realized at the minute's end + open shorts at its high)."""
    close = pa.c
    n = len(close)
    real = np.zeros(n)
    unreal_hi = np.zeros(len(mins))  # worst mark per minute (at its high)
    trips = opn = 0
    for L, alloc in zip(levels, w):
        q = alloc / L
        for i, j in pa.trips(L, s):
            mi = np.searchsorted(mins, i, "right") - 1
            if j is None:
                # bought back at market at the exit (last second), taker fee
                opn += 1
                unreal_hi[mi:-1] += q * (L - mhi[mi:-1])
                real[-1] += q * (L - close[-1]) - TAKER * q * close[-1]
                continue
            trips += 1
            real[j] += q * s
            mj = np.searchsorted(mins, j, "right") - 1
            unreal_hi[mi:mj] += q * (L - mhi[mi:mj])
    cr = np.cumsum(real)
    eq = cr[np.r_[mins[1:] - 1, n - 1]] + unreal_hi
    dd = float(np.max(np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]))
    if curve:
        return float(cr[-1]), dd, trips, opn, eq
    return float(cr[-1]), dd, trips, opn


def weights(n, kind):
    k = np.arange(1, n + 1, dtype=float)
    w = {"flat": np.ones(n), "top": k, "bottom": k[::-1]}[kind]
    return CAP * w / w.sum()


def optimise(b, a, top=5):
    P, sg = b["px"], b["sigma"]
    q = 0.25 * sg
    bots = [b["lo"] + k * q for k in range(int((P - b["lo"]) / q) + 1)]
    tops = [b["hi"] - k * q for k in range(int((b["hi"] - P) / q) + 1)]
    mins = np.flatnonzero(np.r_[True, np.diff(a[:, 0] // 60000) > 0])
    pa, mhi = Walk(a), np.maximum.reduceat(a[:, 1], mins)
    res = []
    for B in bots:
        for T in tops:
            for st in STEPS:
                s = st * sg
                levels = np.arange(B + s, T + 1e-9, s)  # sells; their buy back is one step lower
                if (len(levels) < max(1, CON["min_sells"]) or st > CON["max_step"]
                        or (levels[0] - P) / sg > CON["max_first"] + 1e-9):
                    continue
                for al in CON["sizes"] if len(levels) > 1 else ["flat"]:  # one level: sizing is moot
                    pnl, dd, tr, op = simulate(levels, weights(len(levels), al), s, pa, mins, mhi)
                    res.append(dict(B=round((B - P) / sg, 2), T=round((T - P) / sg, 2), step=st, alloc=al,
                                    lv=len(levels), trips=tr, open=op, pnl=round(pnl, 2), dd=round(dd, 2),
                                    romad=round(pnl / max(dd, DD_FLOOR), 2), bp=B, tp=T))
    res.sort(key=lambda r: -r["romad"])
    # the window by the minute (high, low, close), for the page
    end = lambda k: mins[k + 1] if k + 1 < len(mins) else len(a)
    path = [[float(a[i, 0]), float(mhi[k]), float(np.min(a[i:end(k), 2])), float(a[end(k) - 1, 3])]
            for k, i in enumerate(mins)]
    return res[:top], res, path, (pa, mins, mhi)


def grid_of(pa, t, sg, r):
    """A grid's sells, their sizes (FDUSD) and fills [t sell, price, t buy back or
    None, price], for the page."""
    s = r["step"] * sg
    lv = np.arange(r["bp"] + s, r["tp"] + 1e-9, s)
    w = weights(len(lv), r["alloc"])
    return dict(levels=[float(L) for L in lv], sizes=[round(float(x), 1) for x in w], step=float(s),
                fills=[[float(t[i]), float(L), None if j is None else float(t[j]), float(L - s), round(float(x), 1)]
                       for L, x in zip(lv, w) for i, j in pa.trips(L, s)],
                **{k: r[k] for k in ("B", "T", "alloc", "pnl", "dd", "romad", "trips", "open")})


def main():
    args = sys.argv[1:]
    top = int(args[args.index("--top") + 1]) if "--top" in args else 5
    opt = lambda k, f, d: f(args[args.index(k) + 1]) if k in args else d
    if "--free" in args:
        PAGE[0] = "results_free.html"
        CON.update(max_step=99, min_sells=1, max_first=99, sizes=ALLOCS)
    CON.update(max_step=opt("--max-step", float, CON["max_step"]),
               min_sells=opt("--min-sells", int, CON["min_sells"]),
               max_first=opt("--max-first", float, CON["max_first"]),
               sizes=opt("--sizes", lambda x: x.split(","), CON["sizes"]))
    print("constraints:", CON)
    path = Path(next((x for x in args if x.endswith(".json")), HERE / "bells.json"))
    bells = sorted(json.loads(path.read_text()), key=lambda x: x["birth"])
    if "--stop" in args:
        v = args[args.index("--stop") + 1]
        STOP[0] = v if v == "next" else float(v)
        PAGE[0] = PAGE[0].replace(".html", f"_{path.stem}_stop{v}.html")
    # bells born together: the one nearest the price at their birth runs
    px0 = lambda t: secs(t, t + 60_000)[0, 3]
    bells = [b for b in bells if all(abs(b["px"] - px0(b["birth"])) <= abs(o["px"] - px0(b["birth"]))
                                     for o in bells if o["birth"] == b["birth"] and o is not b)]
    now = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    if "--at-exit" in args:  # each grid starts at its bell's first close out of the curve
        PAGE[0] = PAGE[0].replace(".html", "_atexit" + "_down" * ("--down-only" in args) + ".html")
        for b in bells:
            t, b["start"] = b["birth"], None
            while t < now and b["start"] is None:
                a = secs(t, min(t + DAY, now))
                o = np.flatnonzero((a[:, 3] > b["hi"]) | (a[:, 3] < b["lo"])) if len(a) else []
                if len(o): b["start"], b["up"] = float(a[o[0], 0]) + 1000, bool(a[o[0], 3] > b["hi"])
                t += DAY
        bells = sorted([dict(b, born=b["birth"], birth=b["start"]) for b in bells if b["start"]], key=lambda x: x["birth"])
    out = []
    for k, b in enumerate(bells):
        nxt = bells[k + 1]["birth"] if k + 1 < len(bells) else None
        a = secs(b["birth"], nxt or now)
        e = window(b, nxt, a)
        if "--down-only" in args and b.get("up"):
            continue  # an up exit ends the grid before it, no grid of its own
        tag = f'{b["day"]} P{b["n"]} {b["px"]:.0f} ({"green" if b["mv"] > 0 else "red" if b["mv"] < 0 else "black"})'
        if e is None:
            print(f"{tag}: still inside its bell, skipped")
            continue
        if e < 60:
            print(f"{tag}: window under a minute (next bell born with it), skipped")
            continue
        a = a[:e]
        hrs = (a[-1, 0] - a[0, 0]) / 3.6e6
        best, allr, path, (pa, mins, mhi) = optimise(b, a, top)
        print(f"\n{tag}  window {hrs:.1f}h  range {b['lo']}-{b['hi']}  grids {len(allr)}")
        for r in best:
            print("  ", {k: v for k, v in r.items() if k not in ("bp", "tp")})
        out.append(dict(bell=b, hours=round(hrs, 2), end=float(a[-1, 0]), best=best, all=allr, path=path,
                        grids={f"b{i}": grid_of(pa, a[:, 0], b["sigma"], r) for i, r in enumerate(best)},
                        pa=pa, t=a[:, 0], mins=mins, mhi=mhi))
    (HERE / "results.json").write_text(json.dumps(
        [{k: v for k, v in o.items() if k not in ("pa", "t", "grids", "path", "mins", "mhi", "seen")} for o in out]))
    # the same sell levels in sigma units (first one to 0.25 sigma, step, count)
    # over every bell
    agg = {}
    for o in out:
        seen = {}
        for r in o["all"]:
            k = (round((r["B"] + r["step"]) * 4) / 4, r["step"], r["lv"], r["alloc"])
            if k not in seen or r["romad"] > seen[k]["romad"]:
                seen[k] = r
        o["seen"] = seen
        for k, r in seen.items():
            agg.setdefault(k, []).append(r)
    rows = [(k, v) for k, v in agg.items() if len(v) == len(out)]
    rows.sort(key=lambda kv: -np.mean([r["romad"] for r in kv[1]]))
    print(f"\nBest grids on all {len(out)} bells (sigma from the P; mean RoMaD, total PnL, worst DD, won):")
    for (L1, st, nl, al), v in rows[:10]:
        print(f"  sells from {L1:+.2f} x{nl} every {st}, buy back 1 step lower, {al:6}"
              f"  RoMaD {np.mean([r['romad'] for r in v]):.2f}"
              f"  PnL {sum(r['pnl'] for r in v):.1f}  DD {max(r['dd'] for r in v):.1f}"
              f"  won {sum(r['pnl'] > 0 for r in v)}/{len(v)}")
    summary = [dict(first=L1, step=st, lv=nl, alloc=al, romad=round(float(np.mean([r["romad"] for r in v])), 2),
                    pnl=round(sum(r["pnl"] for r in v), 1), dd=round(max(r["dd"] for r in v), 1),
                    won=int(sum(r["pnl"] > 0 for r in v)), n=len(v)) for (L1, st, nl, al), v in rows[:15]]
    # each summary shape on each bell (agg lists follow the bells' order)
    for j, (_, v) in enumerate(rows[:15]):
        for o, r in zip(out, v):
            o["grids"][f"s{j}"] = grid_of(o["pa"], o["t"], o["bell"]["sigma"], r)
    # held out: each bell scored with the shape that did best on the others
    loo = []
    for i, o in enumerate(out):
        if len(out) < 2 or not rows:
            break
        (L1, st, nl, al), v = max(rows, key=lambda kv: np.mean([r["romad"] for j, r in enumerate(kv[1]) if j != i]))
        r = v[i]
        o["grids"]["h"] = grid_of(o["pa"], o["t"], o["bell"]["sigma"], r)
        loo.append(dict(k=i, first=L1, step=st, lv=nl, alloc=al, pnl=r["pnl"], dd=r["dd"], romad=r["romad"],
                        trips=r["trips"], open=r["open"]))
    if loo:
        print(f"\nHeld out (each bell with the best shape of the others): PnL {sum(x['pnl'] for x in loo):.1f}"
              f"  mean RoMaD {np.mean([x['romad'] for x in loo]):.2f}  worst DD {max(x['dd'] for x in loo):.1f}"
              f"  won {sum(x['pnl'] > 0 for x in loo)}/{len(loo)}")
        for x in loo:
            print(f"  {out[x['k']]['bell']['day']} P{out[x['k']]['bell']['n']}: {x['lv']} from {x['first']:+.2f}"
                  f" every {x['step']} {x['alloc']}  PnL {x['pnl']}  DD {x['dd']}  RoMaD {x['romad']}")
    # rolling: each bell from the R-th on, with the N best shapes of the R bells
    # before it (no look ahead); #1 alone and the N's average (about the 1500
    # split over them)
    R, N = opt("--roll", int, 10), opt("--roll-top", int, 10)
    roll = []
    for i in range(R, len(out)):
        past = [out[j]["seen"] for j in range(i - R, i)]
        keys = [k for k in past[0] if all(k in p for p in past) and k in out[i]["seen"]]
        ranked = sorted(keys, key=lambda k: -np.mean([p[k]["romad"] for p in past]))[:N]
        if not ranked:
            continue
        rs = [out[i]["seen"][k] for k in ranked]
        (L1, st, nl, al) = ranked[0]
        o = out[i]
        out[i]["grids"]["r"] = grid_of(o["pa"], o["t"], o["bell"]["sigma"], rs[0])
        # the N best together: each on 1500 / N, their minute equity added
        eq1 = curve_of(o, rs[0])
        eqN = sum(curve_of(o, r) for r in rs) / len(rs)
        ddN = float(np.max(np.maximum.accumulate(np.r_[0.0, eqN]) - np.r_[0.0, eqN]))
        pnlN = float(eqN[-1])
        # the N best merged into one grid: FDUSD added per (sell, buy back) pair
        gs, mg = [grid_of(o["pa"], o["t"], o["bell"]["sigma"], r) for r in rs], {}
        for gg in gs:
            for L, sz in zip(gg["levels"], gg["sizes"]):
                k = (round(L, 2), round(L - gg["step"], 2))
                mg[k] = mg.get(k, 0) + sz / len(gs)
        ks = sorted(mg, key=lambda k: (-k[0], -k[1]))
        o["grids"]["m"] = dict(levels=[k[0] for k in ks], buys=[k[1] for k in ks],
                               sizes=[round(mg[k], 1) for k in ks], step=None,
                               fills=[f[:4] + [round(f[4] / len(gs), 1)] for gg in gs for f in gg["fills"]],
                               pnl=round(pnlN, 2), dd=round(ddN, 2), romad=round(pnlN / max(ddN, DD_FLOOR), 2),
                               trips=sum(gg["trips"] for gg in gs), open=sum(gg["open"] for gg in gs),
                               B=None, T=None, alloc="merged")
        roll.append(dict(k=i, first=L1, step=st, lv=nl, alloc=al, pnl=rs[0]["pnl"], dd=rs[0]["dd"],
                         romad=rs[0]["romad"], trips=rs[0]["trips"], open=rs[0]["open"],
                         pnlN=round(pnlN, 2), ddN=round(ddN, 2), romadN=round(pnlN / max(ddN, DD_FLOOR), 2),
                         wonN=int(sum(r["pnl"] > 0 for r in rs)), n=len(rs),
                         shapes=[f"{k[2]} from {k[0]:+.2f} every {k[1]}" for k in ranked]))
        o["eq1"], o["eqN"] = eq1, eqN
    # both strategies chained over the forward bells (their windows do not
    # overlap): total PnL, overall DD, RoMaD, won, worst bell; equity every 5 min
    if "--split" in args:  # one fixed shape picked on the bells before the date, chained on the bells after
        cut = int(dt.datetime.fromisoformat(args[args.index("--split") + 1]).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        tr, te = [o for o in out if o["bell"]["birth"] < cut], [o for o in out if o["bell"]["birth"] >= cut]
        keys = [k for k in tr[0]["seen"] if all(k in o["seen"] for o in out)]
        print(f"\nSplit at {args[args.index('--split') + 1]}: picked on {len(tr)} bells, scored on {len(te)}")
        for nm, f in (("mean RoMaD", lambda k: np.mean([o["seen"][k]["romad"] for o in tr])),
                      ("total PnL", lambda k: sum(o["seen"][k]["pnl"] for o in tr))):
            k = max(keys, key=f)
            eq, off = [], 0.0
            for o in te:
                e = curve_of(o, o["seen"][k])
                eq.append(e + off); off += float(e[-1])
            eq = np.concatenate(eq)
            dd = float(np.max(np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]))
            pn = [o["seen"][k]["pnl"] for o in te]
            ptr = sum(o["seen"][k]["pnl"] for o in tr)
            print(f"  best {nm}: {k}  before: PnL {ptr:.1f} | after: PnL {off:.1f}  DD {dd:.1f}  RoMaD {off / max(dd, DD_FLOOR):.2f}"
                  f"  won {sum(p > 0 for p in pn)}/{len(pn)}")
    cmp = {}
    for nm, key, pk in (("#1", "eq1", "pnl"), (f"{N} best", "eqN", "pnlN")):
        if not roll:
            break
        parts, off, ts = [], 0.0, []
        for x in roll:
            o = out[x["k"]]
            parts.append(o[key] + off)
            ts.append(o["t"][o["mins"]])
            off += float(o[key][-1])
        eq, t = np.concatenate(parts), np.concatenate(ts)
        dd = float(np.max(np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]))
        pn = [x[pk] for x in roll]
        cmp[nm] = dict(pnl=round(off, 1), dd=round(dd, 1), romad=round(off / max(dd, DD_FLOOR), 2),
                       won=int(sum(p > 0 for p in pn)), n=len(pn), worst=round(min(pn), 2),
                       mean=round(float(np.mean(pn)), 2), curve=[[float(a), round(float(b), 2)]
                                                                 for a, b in zip(t[::5], eq[::5])])
    if roll:
        print(f"\nRolling ({R} bells before, {N} best shapes), forward over {len(roll)} bells:")
        for nm, c in cmp.items():
            print(f"  {nm:8} PnL {c['pnl']:7.1f}  overall DD {c['dd']:5.1f}  RoMaD {c['romad']:5.2f}"
                  f"  won {c['won']}/{c['n']}  worst bell {c['worst']}  mean {c['mean']}")
        for x in roll:
            b = out[x["k"]]["bell"]
            print(f"  {b['day']} P{b['n']}: #1 {x['shapes'][0]}  PnL {x['pnl']}  DD {x['dd']}"
                  f"  | {x['n']} best: PnL {x['pnlN']}  DD {x['ddN']}  {x['wonN']}/{x['n']} won")
    page(out, summary, loo, roll, R, N, cmp)


def curve_of(o, r):
    """A grid record's minute equity on its bell."""
    s = r["step"] * o["bell"]["sigma"]
    lv = np.arange(r["bp"] + s, r["tp"] + 1e-9, s)
    return simulate(lv, weights(len(lv), r["alloc"]), s, o["pa"], o["mins"], o["mhi"], curve=True)[4]


def page(out, summary, loo, roll, R, N, cmp):
    """results.html: settings, best grids over all bells, held out, rolling, a card per bell."""
    cards = [dict(bell=o["bell"], hours=o["hours"], end=o["end"], path=o["path"], grids=o["grids"],
                  best=[{k: v for k, v in r.items() if k not in ("bp", "tp")} for r in o["best"]]) for o in out]
    con = {k: v for k, v in CON.items()}
    data = json.dumps(dict(cards=cards, summary=summary, loo=loo, roll=roll, R=R, N=N, cmp=cmp, con=con, cap=CAP, page=PAGE[0], stop=STOP[0], atexit="_atexit" in PAGE[0],
                           made=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    (HERE / PAGE[0]).write_text((HERE / "page.html").read_text().replace("/*DATA*/null", data))
    print(f"\n{PAGE[0]} written ({len(cards)} bells)")


if __name__ == "__main__":
    main()
