"""Pick the ~6AM AEST forecast snapshots for today and yesterday from NEMweb
(FORECAST_HH), then draw demand charts and a day-by-day forecast-change chart.

No file rotation is needed: both snapshots are chosen by name from the archive.
"""
import csv, io, json, re, sys, tempfile, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA, CHARTS = ROOT / "data", ROOT / "charts"
META = DATA / "meta.json"
HIST = DATA / "revision_history.csv"
CFG = json.loads((ROOT / "config.json").read_text())
TH = CFG["thresholds"]
FLAG_Z = CFG.get("flag_z", 2.0)          # |robust z| above this is flagged
MIN_N = CFG.get("min_history_days", 15)  # baseline days needed per state and lead
FLAG_TARGET = CFG.get("flag_target_pct", 5)  # asterisk = top X% most unusual of each state's OWN past revisions (0/null: use flag_z for all)
FLAG_MIN_Z = CFG.get("flag_min_z", 1.5)     # floor for a state's auto cut-off
WINDOW = CFG.get("history_window_days", 90)  # rolling baseline length in days
_W = CFG.get("watchlist", {})                    # all optional, defaults shown
# Every size test is relative to the state, so big-demand states aren't over-represented:
W_BAND_PCT = _W.get("band_change_pct", 25)               # POE10-POE90 spread change that counts, % (trial value)
W_BAND_FLOOR_PCT = _W.get("band_change_min_pct_of_peak", 0.3)  # ...and at least this % of the peak demand
W_PEAK_MIN = _W.get("peak_shift_min_minutes", 60)        # daily-peak time move that counts
W_PEAK_PCT = _W.get("peak_shift_min_pct", 0.5)           # POE50 must differ this % of peak between the two peak times
W_STREAK_N = _W.get("streak_min_days", 3)                # consecutive same-direction revisions
W_STREAK_Z = _W.get("streak_min_z", 1.0)                 # each must be this many typical spreads from that state's normal
W_ROWS_PER_STATE = _W.get("max_rows_per_state", 3)  # rows kept per state; the rest are counted as "not shown"
W_MIN_STATES = _W.get("min_states", ["SA1"])    # states flagged ONLY on negative-minimum days (no peak flags)
W_MIN_WINDOW = _W.get("min_window", ["10:00", "16:00"])  # hours (NEM time) used for those states' uncertainty flag
W_D0_FROM = _W.get("d0_earliest_time", "10:00")  # on today (D0) skip flags whose key time (peak / minimum) is earlier than this, AEST
W_P10_GATE = _W.get("poe10_gate_pct", 95) / 100          # NSW/QLD/VIC chips (except Revised): highest POE10 of the day >= this share of the threshold
W_EVE_WINDOW = _W.get("evening_window", ["16:00", "21:00"])  # hours (NEM time) for the evening-spread chip
W_EVE_TOP = _W.get("evening_top_pct", 20)                 # chip if the evening spread is in the top X% of that state's own past values
AEST = timezone(timedelta(hours=10))  # NEM time: fixed UTC+10, no DST
# Snapshot cutoff is a wall-clock time in Sydney (follows daylight saving). File names and
# issue times from NEMweb are NEM time (AEST), so cutoffs are converted to AEST.
if "snapshot_time_local" in CFG:
    SNAP_TIME, SNAP_TZ = CFG["snapshot_time_local"], CFG.get("snapshot_tz", "Australia/Sydney")
else:  # older config: fixed NEM time, same behaviour as before
    SNAP_TIME, SNAP_TZ = CFG["snapshot_time_aest"], "Australia/Brisbane"
SYD = ZoneInfo(SNAP_TZ)
SNAP_KEY = f"{SNAP_TIME} {SNAP_TZ}"


def cutoff_for(day):
    hh, mm = map(int, SNAP_TIME.split(":"))
    local = datetime(day.year, day.month, day.day, hh, mm, tzinfo=SYD)
    return local.astimezone(AEST).replace(tzinfo=None)


def local_today():
    return datetime.now(SYD).date()

HDR = {"User-Agent": "Mozilla/5.0 (demand-dashboard)"}
PAT = re.compile(r"PUBLIC_FORECAST_OPERATIONAL_DEMAND_HH_\d{12}_(\d{14})\.zip")


# ---------- selecting and downloading the two snapshots ----------
def list_files():
    r = requests.get(CFG["base_url"], headers=HDR, timeout=60)
    r.raise_for_status()
    out = {}
    for m in PAT.finditer(r.text):
        out[m.group(0)] = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    if not out:
        sys.exit("No forecast files found in directory listing")
    return out


def pick(files, day):
    cutoff = cutoff_for(day)
    ok = {n: t for n, t in files.items() if t <= cutoff}
    if not ok:
        sys.exit(f"No snapshot issued on/before {cutoff} (archive too short?)")
    name = max(ok, key=ok.get)
    if cutoff - ok[name] > timedelta(hours=3):
        sys.exit(f"Closest snapshot to {cutoff} is stale: {name}")
    return name


def fetch(name, dest):
    r = requests.get(CFG["base_url"] + name, headers=HDR, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        dest.write_bytes(z.read(z.namelist()[0]))


def sync():
    DATA.mkdir(exist_ok=True)
    today = local_today()
    files = list_files()
    want = {"today": pick(files, today), "yest": pick(files, today - timedelta(days=1))}
    have = json.loads(META.read_text()) if META.exists() else {}
    for key, name in want.items():
        dest = DATA / f"{key}.csv"
        if have.get(key) != name or not dest.exists():
            fetch(name, dest)
            print("Downloaded", key, name)
    META.write_text(json.dumps(want, indent=2))
    return files


# ---------- parsing ----------
def read_aemo(path):
    """Handles AEMO's C/I/D multi-record CSV layout or a plain CSV."""
    text = path.read_text(errors="replace")
    first = text.lstrip()[:2]
    if first[:1] in ("C", "I", "D") and first[1:2] == ",":
        hdr, rows = None, []
        for p in csv.reader(io.StringIO(text)):
            if not p:
                continue
            if p[0] == "I":
                if hdr is not None:
                    break  # only the first table
                hdr = p[4:]
            elif p[0] == "D" and hdr is not None:
                rows.append(p[4:4 + len(hdr)])
        return pd.DataFrame(rows, columns=hdr)
    return pd.read_csv(path)


def find(cols, override, *keys):
    if override:
        return override
    for k in keys:
        for c in cols:
            if k in c.upper().replace("_", "").replace(" ", ""):
                return c
    sys.exit(f"Could not detect column {keys}; set it in config.json. Found: {list(cols)}")


def load(path):
    df = read_aemo(path)
    c = CFG["columns"]
    reg = find(df.columns, c["region"], "REGIONID", "REGION")
    tim = find(df.columns, c["time"], "PERIODID", "INTERVALDATETIME", "DATETIME", "PERIOD")
    cols = {"region": reg, "time": tim}
    for p in ("poe10", "poe50", "poe90"):
        cols[p] = find(df.columns, c[p], p.upper())
    out = df[list(cols.values())].copy()
    out.columns = list(cols.keys())
    out["time"] = pd.to_datetime(out["time"])
    for p in ("poe10", "poe50", "poe90"):
        out[p] = pd.to_numeric(out[p], errors="coerce")
    out = out[out.region.isin(TH)].dropna()
    return out.sort_values(["region", "time"]).drop_duplicates(["region", "time"], keep="last")


def aest_day(ts):
    # NEM stamps are interval-ending: 00:00 belongs to the previous day
    return (ts - pd.Timedelta(minutes=30)).dt.date if CFG["interval_ending"] else ts.dt.date


# ---------- charts ----------
def band(ax, g, color="grey", alpha=.25):
    lo, hi = g[["poe10", "poe90"]].min(axis=1), g[["poe10", "poe90"]].max(axis=1)
    ax.fill_between(g.time, lo, hi, color=color, alpha=alpha, lw=0)


def ordered(t):
    """Yield (region, rows) in the order set by config.json thresholds."""
    for st in TH:
        g = t[t.region == st]
        if not g.empty:
            yield st, g


def label_neg_min(ax, g):
    """Mark each AEST day's minimum POE50 with time + value, only if negative."""
    gd = g.assign(day=aest_day(g["time"]))
    gd = gd[gd.groupby("day")["poe50"].transform("size") >= 24]  # skip sliver days
    lows = gd.loc[gd.groupby("day")["poe50"].idxmin()]
    lows = lows[lows.poe50 < 0]
    ax.scatter(lows.time, lows.poe50, s=18, color="black", zorder=5)
    for _, r in lows.iterrows():
        ax.annotate(f"{r.time:%H:%M}\n{r.poe50:,.0f} MW", (r.time, r.poe50), xytext=(0, -6),
                    textcoords="offset points", ha="center", va="top", fontsize=8)


def fmt_days(ax, fig):
    """One tick per day at midnight, labelled like 'Fri 09 Oct'."""
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.grid(axis="x", ls=":", alpha=.4)
    fig.autofmt_xdate(rotation=30, ha="right")


def disp(code):
    """Display name for a region code: NSW1 -> NSW, SA1 -> SA (data and config keep the AEMO codes)."""
    return code[:-1] if code.endswith("1") else code


def plot_states(t):
    CHARTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for st, g in ordered(t):
        (line,) = ax.plot(g.time, g.poe50, label=f"{disp(st)} POE50")
        band(ax, g, color=line.get_color(), alpha=.12)
        if st == "SA1":
            label_neg_min(ax, g)
        if TH[st] is not None:
            ax.axhline(TH[st], ls="--", lw=1, color=line.get_color(), alpha=.8)
    ax.set(title="Operational demand forecast, POE50 (band = POE10-POE90, dashed = threshold)",
           ylabel="MW")
    ax.legend(); fmt_days(ax, fig); fig.tight_layout()
    fig.savefig(CHARTS / "all_states.png", dpi=130); plt.close(fig)

    for st, g in ordered(t):
        fig, ax = plt.subplots(figsize=(11, 4.2))
        band(ax, g)
        ax.plot(g.time, g.poe10, ls="--", lw=1, color="dimgrey", label="POE10")
        ax.plot(g.time, g.poe90, ls="--", lw=1, color="darkgrey", label="POE90")
        ax.plot(g.time, g.poe50, lw=1.8, label="POE50")
        # Mark each AEST day's peak POE50 and label the time it occurs
        gd = g.assign(day=aest_day(g["time"]))
        gd = gd[gd.groupby("day")["poe50"].transform("size") >= 24]  # skip sliver days
        peaks = gd.loc[gd.groupby("day")["poe50"].idxmax()]
        ax.scatter(peaks.time, peaks.poe50, s=18, color="black", zorder=5, label="Daily peak")
        for _, r in peaks.iterrows():
            over = TH[st] is not None and r.poe50 > TH[st]  # peak above threshold: add value, in red
            txt = f"{r.time:%H:%M}\n{r.poe50:,.0f} MW" if over else r.time.strftime("%H:%M")
            ax.annotate(txt, (r.time, r.poe50), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                        color="red" if over else "black", fontweight="bold" if over else "normal")
        if st == "SA1":
            label_neg_min(ax, g)
        ax.margins(y=.15)
        if TH[st] is not None:
            ax.axhline(TH[st], color="red", ls="--", label=f"Threshold {TH[st]:,}")
            lo, hi = g[["poe10", "poe90"]].min(axis=1), g[["poe10", "poe90"]].max(axis=1)
            floor = np.maximum(lo, TH[st])  # shade only inside the 80% band: never below the POE90 line
            over = hi > TH[st]
            # light red: the band wherever the high-demand case (POE10) is above the threshold
            ax.fill_between(g.time, floor, hi, where=over, interpolate=True, color="red", alpha=.18, lw=0)
            # darker red on top, all the way up to POE10, wherever POE50 is above the threshold
            ax.fill_between(g.time, floor, hi, where=over & (g.poe50 > TH[st]), color="red", alpha=.3, lw=0)
        ax.set_title(f"{disp(st)} operational demand forecast", pad=30)
        ax.set_ylabel("MW")
        ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False); fmt_days(ax, fig); fig.tight_layout()
        fig.savefig(CHARTS / f"{st}.png", dpi=130); plt.close(fig)


# ---------- is today's revision unusual? (baseline built from the archive) ----------
_frames = {}


def snap_frame(name):
    if name not in _frames:
        r = requests.get(CFG["base_url"] + name, headers=HDR, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z, tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "snap.csv"
            tmp.write_bytes(z.read(z.namelist()[0]))
            _frames[name] = load(tmp)
    return _frames[name]


def try_pick(files, day):
    """Like pick(), but returns None instead of exiting when there is no suitable file."""
    cutoff = cutoff_for(day)
    ok = {n: t for n, t in files.items() if t <= cutoff and cutoff - t <= timedelta(hours=3)}
    return max(ok, key=ok.get) if ok else None


def daily_changes(t, y):
    """Daily max and min POE50, like-for-like, for two snapshots
    (columns: region, day, today_max, yest_max, delta, today_min, yest_min, delta_min)."""
    m = t.merge(y, on=["region", "time"], suffixes=("_t", "_y"))
    if m.empty:
        return None
    m["day"] = aest_day(m["time"])
    d = m.groupby(["region", "day"]).agg(today_max=("poe50_t", "max"), yest_max=("poe50_y", "max"),
                                         today_min=("poe50_t", "min"), yest_min=("poe50_y", "min")).reset_index()
    d["delta"] = d.today_max - d.yest_max
    d["delta_min"] = d.today_min - d.yest_min
    # evening upside spread: average of (POE10 - POE50) / POE50 over the evening window, in % (needs >= 8 of 10 half-hours)
    tod = interval_start(m["time"]).dt.hour * 60 + interval_start(m["time"]).dt.minute
    e0, e1 = (int(x[:2]) * 60 + int(x[3:5]) for x in W_EVE_WINDOW)
    ev = m[(tod >= e0) & (tod < e1) & (m.poe50_t > 0)].copy()
    if ev.empty:
        d["ev_spread"] = np.nan
    else:
        ev["sp"] = (ev.poe10_t - ev.poe50_t) / ev.poe50_t * 100
        g = ev.groupby(["region", "day"]).sp.agg(ev_spread="mean", n="count").reset_index()
        g.loc[g.n < 8, "ev_spread"] = np.nan
        d = d.merge(g[["region", "day", "ev_spread"]], on=["region", "day"], how="left")
    return d


def update_history(files, today):
    """One row per (snapshot date, state, target day); only missing dates are downloaded."""
    snap = SNAP_KEY
    cols = ["snap", "snap_date", "region", "day", "lead", "today_max", "yest_max", "delta",
            "today_min", "yest_min", "delta_min", "ev_spread"]
    if HIST.exists():
        h = pd.read_csv(HIST, dtype={"snap": str})
        if set(cols) <= set(h.columns):
            h = h[(h.snap == snap) & (h.snap_date != today.isoformat())]  # today is always refreshed
        else:  # older file without the minimum columns: rebuild once
            h = pd.DataFrame(columns=cols)
    else:
        h = pd.DataFrame(columns=cols)
    have = set(h.snap_date)
    start = max(min(files.values()).date() + timedelta(days=1), today - timedelta(days=WINDOW))
    new, d = [], start
    while d <= today:
        if d.isoformat() not in have:
            a, b = try_pick(files, d), try_pick(files, d - timedelta(days=1))
            if a and b:
                ch = daily_changes(snap_frame(a), snap_frame(b))
                if ch is not None:
                    ch["snap"], ch["snap_date"] = snap, d.isoformat()
                    ch["lead"] = (pd.to_datetime(ch.day) - pd.Timestamp(d)).dt.days
                    ch["day"] = ch.day.astype(str)
                    new.append(ch[cols])
        d += timedelta(days=1)
    h = pd.concat([h] + new, ignore_index=True)[cols]
    h = h[h.snap_date >= (today - timedelta(days=WINDOW)).isoformat()]  # rolling window
    h.to_csv(HIST, index=False)
    return h


def state_cutoffs(hist):
    """Per-state z cut-off: the |z| exceeded by only FLAG_TARGET % of that state's own past revisions,
    so every state flags at about the same rate. Needs enough history; otherwise flag_z is used."""
    cuts = {}
    if not FLAG_TARGET:
        return cuts
    for reg, g in hist.groupby("region"):
        zs = []
        for _, gl in g.groupby("lead"):
            x = gl.delta.dropna()
            if len(x) < MIN_N:
                continue
            med = x.median()
            sc = 1.4826 * (x - med).abs().median() or x.std()
            if sc and sc > 0:
                zs.append(((x - med) / sc).abs())
        if zs and sum(len(z) for z in zs) >= 100:
            cuts[reg] = max(float(pd.concat(zs).quantile(1 - FLAG_TARGET / 100)), FLAG_MIN_Z)
    return cuts


def add_flags(d, hist, today):
    """Compare each of today's changes with past changes for the SAME state and lead (days ahead)."""
    hist = hist[hist.snap_date < today.isoformat()]
    cuts = state_cutoffs(hist)
    rows = []
    for r in d.itertuples():
        lead = (pd.Timestamp(r.day) - pd.Timestamp(today)).days
        h = hist[(hist.region == r.region) & (hist.lead == lead)].delta.dropna()
        n, z, flag, med, p5, p95 = len(h), np.nan, False, np.nan, np.nan, np.nan
        scale, lo, hi = np.nan, np.nan, np.nan
        cut = cuts.get(r.region, FLAG_Z)  # this state's cut-off (flag_z if there isn't enough history yet)
        if n >= MIN_N:
            med = h.median()
            scale = 1.4826 * (h - med).abs().median() or h.std()   # robust spread, falls back to std
            p5, p95 = h.quantile(.05), h.quantile(.95)
            if scale and scale > 0:
                z = (r.delta - med) / scale
                flag = bool(abs(z) > cut)
                lo, hi = med - cut * scale, med + cut * scale  # a change outside these MW limits gets the *
        rows.append((lead, n, med, p5, p95, scale, cut, lo, hi, z, flag))
    d = d.copy()
    d[["lead", "base_n", "base_median", "base_p5", "base_p95", "base_scale", "flag_cut", "flag_low", "flag_high", "z", "flag"]] = rows
    return d


# ---------- watchlist: spread, peak timing and revision streaks ----------
def interval_start(ts):
    return ts - pd.Timedelta(minutes=30) if CFG["interval_ending"] else ts


def peak_info(t, y):
    """Per state and AEST day (like-for-like intervals only), for today's and yesterday's snapshots:
    the daily POE50 peak (time, POE10-POE90 spread there), today's highest POE10 and lowest POE50,
    and the average POE10-POE90 spread inside the W_MIN_WINDOW hours."""
    m = t.merge(y, on=["region", "time"], suffixes=("_t", "_y"))
    if m.empty:
        return pd.DataFrame()
    m["day"] = aest_day(m["time"])
    st = interval_start(m["time"])
    m["tod"] = st.dt.hour * 60 + st.dt.minute
    w0, w1 = (int(x[:2]) * 60 + int(x[3:5]) for x in W_MIN_WINDOW)
    e0, e1 = (int(x[:2]) * 60 + int(x[3:5]) for x in W_EVE_WINDOW)
    out = []
    for (reg, day), g in m.groupby(["region", "day"]):
        if len(g) < 24:  # skip sliver days
            continue
        it, iy = g.poe50_t.idxmax(), g.poe50_y.idxmax()
        gw = g[(g.tod >= w0) & (g.tod < w1)]
        ok = len(gw) >= 6
        ge = g[(g.tod >= e0) & (g.tod < e1)]
        out.append(dict(
            region=reg, day=day, peak_t=g.loc[it, "time"], peak_y=g.loc[iy, "time"],
            width_t=abs(g.loc[it, "poe10_t"] - g.loc[it, "poe90_t"]),
            width_y=abs(g.loc[iy, "poe10_y"] - g.loc[iy, "poe90_y"]),
            drop_t=g.loc[it, "poe50_t"] - g.loc[iy, "poe50_t"],  # today's peak vs today's value at yesterday's peak time
            level=g.loc[it, "poe50_t"],
            p10max_t=g.poe10_t.max(),                            # highest POE10 of the day (today's forecast)
            min_t=g.poe50_t.min(),                               # lowest POE50 of the day (today's forecast)
            min_time_t=g.loc[g.poe50_t.idxmin(), "time"],       # when that lowest value occurs
            ev_p10max_t=ge.poe10_t.max() if len(ge) >= 8 else np.nan,  # highest POE10 in the evening window
            win_t=(gw.poe10_t - gw.poe90_t).abs().mean() if ok else np.nan,
            win_y=(gw.poe10_y - gw.poe90_y).abs().mean() if ok else np.nan))
    return pd.DataFrame(out)


def revision_streaks(hist, today, col="delta"):
    """Consecutive daily revisions in the same direction for each (state, target day), ending today.
    A revision counts only if it is at least W_STREAK_Z typical spreads away from that state's normal
    change at that many days ahead, so known biases and naturally bigger swings don't trigger it.
    col: 'delta' (daily maximum) or 'delta_min' (daily minimum)."""
    hist = hist.dropna(subset=[col])

    def spread(x):
        med = x.median()
        return med, (1.4826 * (x - med).abs().median() or x.std())
    stats = {k: spread(g[col]) for k, g in hist.groupby(["region", "lead"]) if len(g) >= MIN_N}
    res = {}
    for (reg, day), g in hist.groupby(["region", "day"]):
        g = g.sort_values("snap_date", ascending=False)
        if g.snap_date.iloc[0] != today.isoformat():
            continue
        n, tot, sign, prev = 0, 0.0, 0, None
        for r in g.itertuples():
            val = getattr(r, col)
            dt, (med, sc) = pd.Timestamp(r.snap_date), stats.get((reg, r.lead), (0, 0))
            if not sc or sc != sc or (prev is not None and (prev - dt).days != 1):
                break
            z = (val - med) / sc
            if abs(z) < W_STREAK_Z:
                break
            sg = 1 if z > 0 else -1
            if n == 0:
                sign = sg
            elif sg != sign:
                break
            n, tot, prev = n + 1, tot + val, dt
        res[(reg, str(day))] = (n, tot, sign)
    return res


def build_watchlist(t, y, hist, today):
    pk = peak_info(t, y)
    st, stm = revision_streaks(hist, today), revision_streaks(hist, today, "delta_min")
    hrs = lambda x: x[:2].lstrip("0") or "0"
    win_label = f"{hrs(W_MIN_WINDOW[0])}\u2013{hrs(W_MIN_WINDOW[1])}h"
    d0_from = int(W_D0_FROM[:2]) * 60 + int(W_D0_FROM[3:5])
    tod = lambda ts: ts.hour * 60 + ts.minute
    # evening spread: today's value, and each state's own past distribution at the same days-ahead
    eve = hist.dropna(subset=["ev_spread"])
    base = eve[eve.snap_date < today.isoformat()]
    q = {k: g.quantile(1 - W_EVE_TOP / 100) for k, g in base.groupby(["region", "lead"]).ev_spread if len(g) >= MIN_N}
    cur = eve[eve.snap_date == today.isoformat()].set_index(["region", "day"]).ev_spread.to_dict()
    rows = []
    for r in pk.itertuples():
        chips = []
        is_d0 = pd.Timestamp(r.day) == pd.Timestamp(today)
        if r.region in W_MIN_STATES:
            # e.g. SA1: no peak flags at all; only on days the minimum is forecast negative
            if r.min_t < 0:
                dw = r.win_t - r.win_y
                pct = dw / r.win_y * 100 if r.win_y and r.win_y > 0 else 0
                if abs(pct) >= W_BAND_PCT:
                    chips.append({"kind": "band", "dir": "up" if dw > 0 else "down",
                                  "text": f"Uncertainty {win_label} {'▲' if dw > 0 else '▼'} {abs(pct):.0f}%"})
                nm, totm, sgm = stm.get((r.region, str(r.day)), (0, 0.0, 0))
                if nm >= W_STREAK_N and not (is_d0 and tod(r.min_time_t) < d0_from):
                    chips.append({"kind": "streak", "dir": "up" if sgm > 0 else "down",
                                  "text": f"Min revised {'↑' if sgm > 0 else '↓'} {nm} days, {totm:+,.0f} MW"})
        else:
            th = TH.get(r.region)
            early = is_d0 and tod(r.peak_t) < d0_from  # today: nothing about a peak that is over before the cut-off time
            gate = th is not None and r.p10max_t >= W_P10_GATE * th  # high-demand case (POE10) close to / over the threshold
            if not early and gate:
                dw = r.width_t - r.width_y
                pct = dw / r.width_y * 100 if r.width_y > 0 else 0
                if abs(pct) >= W_BAND_PCT and abs(dw) >= W_BAND_FLOOR_PCT / 100 * r.level:
                    chips.append({"kind": "band", "dir": "up" if dw > 0 else "down",
                                  "text": f"Spread changed {'▲' if dw > 0 else '▼'} {abs(pct):.0f}%"})
            lead = (pd.Timestamp(r.day) - pd.Timestamp(today)).days
            sp, lim = cur.get((r.region, str(r.day))), q.get((r.region, lead))
            if (th is not None and sp is not None and lim is not None and sp > lim
                    and r.ev_p10max_t >= W_P10_GATE * th):  # evening spread unusually high AND POE10 near the threshold
                chips.append({"kind": "band", "dir": "up", "text": "Evening spread high"})
            if not early and gate:
                shift = (r.peak_t - r.peak_y).total_seconds() / 60
                if abs(shift) >= W_PEAK_MIN and r.drop_t >= W_PEAK_PCT / 100 * r.level:
                    chips.append({"kind": "peak", "dir": "later" if shift > 0 else "earlier",
                                  "text": f"Peak {r.peak_y:%H:%M} → {r.peak_t:%H:%M}"})
            n, tot, sg = st.get((r.region, str(r.day)), (0, 0.0, 0))
            if not early and n >= W_STREAK_N:  # Revised is not tied to the threshold
                chips.append({"kind": "streak", "dir": "up" if sg > 0 else "down",
                              "text": f"Revised {'↑' if sg > 0 else '↓'} {n} days, {tot:+,.0f} MW"})
        if chips:
            rows.append({"region": r.region, "day": str(r.day), "day_label": pd.Timestamp(r.day).strftime("%a %d %b"),
                         "lead": (pd.Timestamp(r.day) - pd.Timestamp(today)).days, "chips": chips,
                         # which days to keep if a state has too many: highest peak demand, or lowest minimum for min states
                         "_rank": r.min_t if r.region in W_MIN_STATES else -r.level})
    omitted, kept = 0, []
    for reg in TH:
        rs = sorted((x for x in rows if x["region"] == reg), key=lambda x: (x["_rank"], x["day"]))
        kept += rs[:W_ROWS_PER_STATE]
        omitted += max(0, len(rs) - W_ROWS_PER_STATE)
    rows = kept  # grouped by state in config order; within a state: highest peak first (SA1: lowest minimum first)
    for x in rows:
        x.pop("_rank")
    (DATA / "watchlist.json").write_text(json.dumps({"rows": rows, "omitted": omitted}, indent=1, ensure_ascii=False))
    return rows


def plot_delta(t, y, files):
    # Like-for-like: only intervals present in BOTH snapshots, so the partial
    # first day and the extra final day of 'today' cannot distort the maxima.
    d = daily_changes(t, y)
    if d is None:
        print("No overlap between today and yest - skipping delta chart")
        return
    today = local_today()
    hist = update_history(files, today)
    d = add_flags(d, hist, today)
    build_watchlist(t, y, hist, today)
    d.round(2).to_csv(DATA / "daily_max_poe50.csv", index=False)
    fl = {(r.region, r.day): bool(r.flag) for r in d.itertuples()}
    p = d.pivot(index="day", columns="region", values="delta")
    p = p[[st for st in TH if st in p.columns]]  # configured state order
    days = list(p.index)
    p.index = [x.strftime("%a %d %b") for x in p.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    p.rename(columns=disp).plot.bar(ax=ax, width=.8)
    for c, st in zip(ax.containers, p.columns):
        labs = [f"{v:+.0f}" + ("*" if fl.get((st, dy)) else "") for v, dy in zip(p[st], days)]
        for tx, lb in zip(ax.bar_label(c, labels=labs, fontsize=7, padding=2), labs):
            if lb.endswith("*"):
                tx.set_color("red"); tx.set_fontweight("bold")
    ax.axhline(0, color="k", lw=.8)
    names = json.loads(META.read_text())
    lab = lambda n: datetime.strptime(PAT.search(n).group(1), "%Y%m%d%H%M%S").strftime("%a %d %b %H:%M")
    ax.set(title=("Change in daily max POE50 forecast (MW)\n"
                  f"Forecast issued {lab(names['today'])} AEST (today) "
                  f"minus forecast issued {lab(names['yest'])} AEST (yesterday)"),
           ylabel="MW (positive = forecast raised)", xlabel="")
    ax.tick_params(axis="x", rotation=30)
    n = int(d.base_n.max()) if len(d) else 0
    rule = (f"among the {FLAG_TARGET:g}% most unusual of that state's own past revisions at that days-ahead"
            if FLAG_TARGET else f"|robust z| > {FLAG_Z:g}")
    fig.text(0.01, 0.01, f"* unusually large: {rule} (vs up to {n} past days)", fontsize=7, color="dimgrey")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(CHARTS / "delta_demand.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    files = sync()
    today, yest = load(DATA / "today.csv"), load(DATA / "yest.csv")
    plot_states(today)
    plot_delta(today, yest, files)
