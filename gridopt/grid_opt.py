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

Usage: python grid_opt.py [bells.json] [--top 5]
"""
import io, json, sys, zipfile, datetime as dt
from pathlib import Path
import numpy as np, requests

HERE = Path(__file__).parent
DATA = HERE / "data"
CAP, TAKER = 1500.0, 0.001
DD_FLOOR = 5.0  # RoMaD = PnL / max(DD, this): a near-zero DD does not win alone
STEPS = [0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
ALLOCS = ["flat", "top", "bottom"]
DAY = 86_400_000


def day_secs(day):
    """BTCFDUSD 1s candles of a UTC day: [t ms, high, low, close]; cached."""
    DATA.mkdir(exist_ok=True)
    f = DATA / f"BTCFDUSD-1s-{day}.npy"
    if f.exists():
        return np.load(f)
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
        j = requests.get("https://api.binance.com/api/v3/klines", timeout=30, params=dict(
            symbol="BTCFDUSD", interval="1s", limit=1000, startTime=t, endTime=t0 + DAY - 1)).json()
        if not j:
            break
        out += [[k[0], float(k[2]), float(k[3]), float(k[4])] for k in j]
        t = j[-1][0] + 1000
    return np.array(out)


def secs(t0, t1):
    """1s candles from t0 to t1 (ms)."""
    d0 = dt.datetime.fromtimestamp(t0 / 1000, dt.timezone.utc).date()
    d1 = dt.datetime.fromtimestamp((t1 - 1) / 1000, dt.timezone.utc).date()
    days = [d0 + dt.timedelta(k) for k in range((d1 - d0).days + 1)]
    a = np.concatenate([day_secs(d.isoformat()) for d in days])
    return a[(a[:, 0] >= t0) & (a[:, 0] < t1)]


def window(b, nxt, a):
    """Exit index: the end of the last time inside the curve before the next
    bell's birth (no re-entry after it); None when still inside at the end."""
    c = a[:, 3]
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


def simulate(levels, w, s, pa, mins, mhi):
    """PnL, max DD (minute highs), trips, open shorts at the end."""
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
                opn += 1
                unreal_hi[mi:] += q * (L - mhi[mi:])
                real[-1] += q * (L - close[-1]) - TAKER * q * close[-1]
                continue
            trips += 1
            real[j] += q * s
            mj = np.searchsorted(mins, j, "right") - 1
            unreal_hi[mi:mj] += q * (L - mhi[mi:mj])
    eq_real = np.cumsum(real)[mins]
    eq = eq_real + unreal_hi
    eq[-1] = eq_real[-1] if opn else eq[-1]
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    return float(np.cumsum(real)[-1]), dd, trips, opn


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
                if len(levels) < 1:
                    continue
                for al in ALLOCS if len(levels) > 1 else ["flat"]:  # one level: sizing is moot
                    pnl, dd, tr, op = simulate(levels, weights(len(levels), al), s, pa, mins, mhi)
                    res.append(dict(B=round((B - P) / sg, 2), T=round((T - P) / sg, 2), step=st, alloc=al,
                                    lv=len(levels), trips=tr, open=op, pnl=round(pnl, 2), dd=round(dd, 2),
                                    romad=round(pnl / max(dd, DD_FLOOR), 2), bp=B, tp=T))
    res.sort(key=lambda r: -r["romad"])
    # the best grid's levels and fills, and the window by the minute, for the page
    r, s = res[0], res[0]["step"] * sg
    lv = np.arange(r["bp"] + s, r["tp"] + 1e-9, s)
    t = a[:, 0]
    fills = [[float(t[i]), float(L), None if j is None else float(t[j]), float(L - s)]
             for L in lv for i, j in pa.trips(L, s)]
    path = [[float(t[i]), float(mhi[k]), float(np.min(a[i:(mins[k + 1] if k + 1 < len(mins) else len(a)), 2])),
             float(a[(mins[k + 1] if k + 1 < len(mins) else len(a)) - 1, 3])] for k, i in enumerate(mins)]
    return res[:top], res, dict(levels=[float(L) for L in lv], step=float(s), fills=fills, path=path)


def main():
    args = sys.argv[1:]
    top = int(args[args.index("--top") + 1]) if "--top" in args else 5
    path = Path(next((x for x in args if x.endswith(".json")), HERE / "bells.json"))
    bells = sorted(json.loads(path.read_text()), key=lambda x: x["birth"])
    now = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    out = []
    for k, b in enumerate(bells):
        nxt = bells[k + 1]["birth"] if k + 1 < len(bells) else None
        a = secs(b["birth"], nxt or now)
        e = window(b, nxt, a)
        tag = f'{b["day"]} P{b["n"]} {b["px"]:.0f} ({"green" if b["mv"] > 0 else "red" if b["mv"] < 0 else "black"})'
        if e is None:
            print(f"{tag}: still inside its bell, skipped")
            continue
        a = a[:e]
        hrs = (a[-1, 0] - a[0, 0]) / 3.6e6
        best, allr, show = optimise(b, a, top)
        print(f"\n{tag}  window {hrs:.1f}h  range {b['lo']}-{b['hi']}  grids {len(allr)}")
        for r in best:
            print("  ", {k: v for k, v in r.items() if k not in ("bp", "tp")})
        out.append(dict(bell=b, hours=round(hrs, 2), end=float(a[-1, 0]), best=best, all=allr, show=show))
    (HERE / "results.json").write_text(json.dumps(out))
    # the same sell levels in sigma units (first one to 0.25 sigma, step, count)
    # over every bell
    agg = {}
    for o in out:
        seen = {}
        for r in o["all"]:
            k = (round((r["B"] + r["step"]) * 4) / 4, r["step"], r["lv"], r["alloc"])
            if k not in seen or r["romad"] > seen[k]["romad"]:
                seen[k] = r
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
    page(out, summary)


def page(out, summary):
    """results.html: settings, best grids over all bells, a card per bell."""
    cards = [dict(bell=o["bell"], hours=o["hours"], end=o["end"], show=o["show"],
                  best=[{k: v for k, v in r.items() if k not in ("bp", "tp")} for r in o["best"]]) for o in out]
    data = json.dumps(dict(cards=cards, summary=summary, cap=CAP,
                           made=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    (HERE / "results.html").write_text((HERE / "page.html").read_text().replace("/*DATA*/null", data))
    print(f"\nresults.html written ({len(cards)} bells)")


if __name__ == "__main__":
    main()
