"""Pick the ~7AM AEST forecast snapshots for today and yesterday from NEMweb
(FORECAST_HH), then draw demand charts and a day-by-day forecast-change chart.

No file rotation is needed: both snapshots are chosen by name from the archive.
"""
import csv, io, json, re, sys, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA, CHARTS = ROOT / "data", ROOT / "charts"
META = DATA / "meta.json"
CFG = json.loads((ROOT / "config.json").read_text())
TH = CFG["thresholds"]
AEST = timezone(timedelta(hours=10))  # NEM time: fixed UTC+10, no DST
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
    hh, mm = map(int, CFG["snapshot_time_aest"].split(":"))
    cutoff = datetime(day.year, day.month, day.day, hh, mm)
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
    today = datetime.now(AEST).date()
    files = list_files()
    want = {"today": pick(files, today), "yest": pick(files, today - timedelta(days=1))}
    have = json.loads(META.read_text()) if META.exists() else {}
    for key, name in want.items():
        dest = DATA / f"{key}.csv"
        if have.get(key) != name or not dest.exists():
            fetch(name, dest)
            print("Downloaded", key, name)
    META.write_text(json.dumps(want, indent=2))


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


def plot_states(t):
    CHARTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for st, g in ordered(t):
        (line,) = ax.plot(g.time, g.poe50, label=f"{st} POE50")
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
            ax.annotate(r.time.strftime("%H:%M"), (r.time, r.poe50), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8)
        if st == "SA1":
            label_neg_min(ax, g)
        ax.margins(y=.12)
        if TH[st] is not None:
            ax.axhline(TH[st], color="red", ls="--", label=f"Threshold {TH[st]:,}")
            ax.fill_between(g.time, TH[st], g.poe50, where=g.poe50 > TH[st], color="red", alpha=.3)
        ax.set_title(f"{st} operational demand forecast", pad=30)
        ax.set_ylabel("MW")
        ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False); fmt_days(ax, fig); fig.tight_layout()
        fig.savefig(CHARTS / f"{st}.png", dpi=130); plt.close(fig)


def plot_delta(t, y):
    # Like-for-like: only intervals present in BOTH snapshots, so the partial
    # first day and the extra final day of 'today' cannot distort the maxima.
    m = t.merge(y, on=["region", "time"], suffixes=("_t", "_y"))
    if m.empty:
        print("No overlap between today and yest - skipping delta chart")
        return
    m["day"] = aest_day(m["time"])
    d = m.groupby(["region", "day"]).agg(today_max=("poe50_t", "max"),
                                         yest_max=("poe50_y", "max")).reset_index()
    d["delta"] = d.today_max - d.yest_max
    d.round(0).to_csv(DATA / "daily_max_poe50.csv", index=False)
    p = d.pivot(index="day", columns="region", values="delta")
    p = p[[st for st in TH if st in p.columns]]  # configured state order
    p.index = [x.strftime("%a %d %b") for x in p.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    p.plot.bar(ax=ax, width=.8)
    for c in ax.containers:
        ax.bar_label(c, fmt="%+.0f", fontsize=7, padding=2)
    ax.axhline(0, color="k", lw=.8)
    names = json.loads(META.read_text())
    lab = lambda n: datetime.strptime(PAT.search(n).group(1), "%Y%m%d%H%M%S").strftime("%a %d %b %H:%M")
    ax.set(title=("Change in daily max POE50 forecast (MW)\n"
                  f"Forecast issued {lab(names['today'])} AEST (today) "
                  f"minus forecast issued {lab(names['yest'])} AEST (yesterday)"),
           ylabel="MW (positive = forecast raised)", xlabel="")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(CHARTS / "delta_demand.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    sync()
    today, yest = load(DATA / "today.csv"), load(DATA / "yest.csv")
    plot_states(today)
    plot_delta(today, yest)
