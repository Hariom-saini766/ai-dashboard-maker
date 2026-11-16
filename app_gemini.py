"""
Dashboard Maker with FREE Gemini AI
-----------------------------------
Write ANYTHING about your sheet ("delhi store ki sales last month", "top 5 products",
"monthly trend for electronics", "full dashboard"). Gemini (free key from
https://aistudio.google.com) works out what you mean; the app then calculates the exact numbers
itself with pandas. Without a key, a built-in smart parser (fuzzy columns, values, dates) is used.

Run:
    pip install streamlit pandas plotly openpyxl requests
    streamlit run app_gemini.py
    API - AQ.Ab8RN6JF8ZYTyWMOGdWkmVZ32i2mFLgKo9NecIpf1KTWEuyMRA
"""

import difflib
import io
import json
import os
import re
import warnings

import requests

import pandas as pd
import plotly.express as px
import streamlit as st

AGGS = ["sum", "mean", "count", "nunique", "max", "min", "median"]
MEASURE_HINTS = ["sales", "amount", "revenue", "qty", "quantity", "value", "price",
                 "profit", "total", "cost", "units", "mrp", "discount"]
ID_WORDS = {"id", "code", "sr", "no", "index", "sku", "number", "pin", "pincode", "phone"}

st.set_page_config(page_title="Dashboard Maker", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------
# Loading and cleaning
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, name: str, sheet):
    buf = io.BytesIO(file_bytes)
    if name.lower().endswith(".csv"):
        try:
            return pd.read_csv(buf)
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf, encoding="latin-1")
    return pd.read_excel(buf, sheet_name=sheet or 0)


def prepare_df(df: pd.DataFrame, dayfirst: bool) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")

    for col in df.columns:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue
        sample = df[col].dropna().astype(str).head(300)
        if sample.empty:
            continue

        # numbers stored as text, e.g. "1,250" or "Rs 500"
        cleaned = sample.str.replace(r"[,\s₹$]|rs\.?", "", regex=True, case=False)
        if pd.to_numeric(cleaned, errors="coerce").notna().mean() > 0.95:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[,\s₹$]|rs\.?", "", regex=True, case=False),
                errors="coerce",
            )
            continue

        # dates stored as text
        has_date_chars = sample.str.contains(
            r"[-/:]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", case=False, regex=True
        ).mean() > 0.8
        name_hint = any(w in col.lower() for w in ["date", "time", "month", "day"])
        if has_date_chars or name_hint:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst)
            if parsed.notna().mean() > 0.9:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)
    return df


def norm(s: str) -> str:
    return re.sub(r"[_\-\s]+", " ", str(s).lower()).strip()


def is_date(s): return pd.api.types.is_datetime64_any_dtype(s)


def is_measure(df, col) -> bool:
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s) or is_date(s):
        return False
    if set(norm(col).split()) & ID_WORDS:
        return False
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and len(s) > 20:
        return False
    return True


def is_dim(df, col) -> bool:
    s = df[col]
    return (not is_date(s)) and (not pd.api.types.is_numeric_dtype(s)) and 2 <= s.nunique() <= 60


def measures(df): 
    cols = [c for c in df.columns if is_measure(df, c)]
    return sorted(cols, key=lambda c: 0 if any(h in norm(c) for h in MEASURE_HINTS) else 1)


def dims(df): return [c for c in df.columns if is_dim(df, c)]
def dates(df): return [c for c in df.columns if is_date(df[c])]


def fmt_num(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(v)
    if a >= 1e7: return f"{v / 1e7:,.2f} Cr"
    if a >= 1e5: return f"{v / 1e5:,.2f} L"
    if a >= 1e3: return f"{v / 1e3:,.1f} K"
    return f"{int(v):,}" if v == int(v) else f"{v:,.2f}"


# ----------------------------------------------------------------------------
# Planning (rule based, replaces the AI)
# ----------------------------------------------------------------------------
def auto_plan(df: pd.DataFrame) -> dict:
    ms, ds, dt = measures(df), dims(df), dates(df)
    kpis = [{"title": "Total Rows", "column": df.columns[0], "agg": "count"}]
    for m in ms[:3]:
        kpis.append({"title": f"Total {m}", "column": m, "agg": "sum"})
    if ms:
        kpis.append({"title": f"Avg {ms[0]}", "column": ms[0], "agg": "mean"})

    charts = []
    main = ms[0] if ms else None
    if dt and main:
        charts.append({"type": "line", "title": f"{main} trend (monthly)", "x": dt[0], "y": main,
                       "agg": "sum", "time_grain": "month"})
    for d in ds[:3]:
        charts.append({"type": "bar", "title": f"{main or 'Rows'} by {d}", "x": d, "y": main,
                       "agg": "sum" if main else "count", "top_n": 10, "sort": "desc"})
    pie_dim = next((d for d in ds if df[d].nunique() <= 8), None)
    if pie_dim:
        charts.append({"type": "pie", "title": f"{main or 'Rows'} share by {pie_dim}", "x": pie_dim,
                       "y": main, "agg": "sum" if main else "count"})
    if main:
        charts.append({"type": "histogram", "title": f"Distribution of {main}", "x": main})

    filters = dt[:1] + [d for d in ds if df[d].nunique() <= 50][:3]
    return {"title": "Auto Dashboard", "kpis": kpis, "charts": charts, "filters": filters}


def find_mentioned(df: pd.DataFrame, p: str):
    found = []
    for c in df.columns:
        n = norm(c)
        if n and (re.search(rf"\b{re.escape(n)}s?\b", p) or re.search(rf"\b{re.escape(n.rstrip('s'))}s?\b", p)):
            found.append(c)
    return found


def plan_from_prompt(df: pd.DataFrame, prompt: str) -> dict:
    p = norm(prompt)
    ments = find_mentioned(df, p)
    ms_all, ds_all, dt_all = measures(df), dims(df), dates(df)

    m_sel = [c for c in ments if c in ms_all or (pd.api.types.is_numeric_dtype(df[c]) and not is_date(df[c]))]
    d_sel = [c for c in ments if c in ds_all or (c not in m_sel and not is_date(df[c]) and df[c].nunique() <= 60)]
    t_sel = [c for c in ments if is_date(df[c])]

    # aggregation
    if re.search(r"\b(average|avg|mean)\b", p): agg = "mean"
    elif re.search(r"\b(count|number of|how many)\b", p): agg = "count"
    elif re.search(r"\b(max|maximum|highest)\b", p): agg = "max"
    elif re.search(r"\b(min|minimum|lowest)\b", p): agg = "min"
    else: agg = "sum"

    grain = "day" if re.search(r"\b(daily|day wise|per day)\b", p) else \
            "year" if re.search(r"\b(yearly|annual|year wise|per year)\b", p) else "month"
    m = re.search(r"\b(?:top|best)\s*(\d+)", p)
    top_n = int(m.group(1)) if m else (10 if "top" in p or "best" in p else None)
    ascending = bool(re.search(r"\b(bottom|worst|lowest|least)\b", p))

    want_pie = bool(re.search(r"\b(pie|share|percentage|percent|contribution|split)\b", p))
    want_trend = bool(re.search(r"\b(trend|over time|monthly|daily|yearly|growth|timeline)\b", p)) or bool(t_sel)
    want_scatter = bool(re.search(r"\b(scatter|relationship|correlation|vs|versus)\b", p)) and len(m_sel) >= 2
    want_hist = bool(re.search(r"\b(histogram|distribution)\b", p)) and not want_pie

    if not ments and not want_trend:
        plan = auto_plan(df)
        plan["title"] = "Dashboard (no specific columns found in your prompt, showing auto view)"
        return plan

    if not m_sel and ms_all:
        m_sel = ms_all[:1]
    if want_trend and not t_sel and dt_all:
        t_sel = dt_all[:1]
    if not d_sel and not t_sel and not want_scatter:
        d_sel = ds_all[:2]

    main = m_sel[0] if m_sel else None
    kpis = [{"title": "Total Rows", "column": df.columns[0], "agg": "count"}]
    for mm in m_sel[:4]:
        kpi_agg = agg if agg != "count" else "sum"
        kpis.append({"title": f"{'Total' if kpi_agg == 'sum' else kpi_agg.title()} {mm}",
                     "column": mm, "agg": kpi_agg})

    charts = []
    if t_sel:
        for mm in (m_sel[:2] or [None]):
            charts.append({"type": "line", "title": f"{mm or 'Rows'} trend ({grain}ly)", "x": t_sel[0],
                           "y": mm, "agg": agg if mm else "count", "time_grain": grain})
    for d in d_sel[:4]:
        n_ = re.escape(norm(d))
        pie_words = r"(?:pie|share|percentage|percent|contribution|split)"
        near = re.search(rf"{n_}s?\W+(?:\w+\W+){{0,1}}{pie_words}|{pie_words}\W+(?:\w+\W+){{0,2}}{n_}", p)
        ctype = "pie" if (near or (want_pie and len(d_sel) == 1)) else "bar"
        charts.append({"type": ctype, "title": f"{main or 'Rows'} by {d}", "x": d, "y": main,
                       "agg": agg if main else "count", "top_n": top_n,
                       "sort": "asc" if ascending else "desc"})
    if want_scatter:
        charts.append({"type": "scatter", "title": f"{m_sel[0]} vs {m_sel[1]}", "x": m_sel[0], "y": m_sel[1]})
    if want_hist and main:
        charts.append({"type": "histogram", "title": f"Distribution of {main}", "x": main})

    filters = t_sel[:1] + dt_all[:1] if not t_sel else t_sel[:1]
    filters += [d for d in ds_all if df[d].nunique() <= 50][:3]
    filters = list(dict.fromkeys(filters))[:4]
    return {"title": "Custom Dashboard", "kpis": kpis, "charts": charts, "filters": filters}


# ----------------------------------------------------------------------------
# Computing and charts
# ----------------------------------------------------------------------------
def to_num(s): return pd.to_numeric(s, errors="coerce")


def compute_kpi(df, k):
    col, agg = k["column"], k.get("agg", "sum")
    if col not in df.columns:
        raise ValueError(f"column '{col}' not found")
    s = df[col]
    if agg == "count": return s.count()
    if agg == "nunique": return s.nunique()
    return getattr(to_num(s), agg)()


def build_chart(df, ch):
    ctype, x, y = ch.get("type", "bar"), ch.get("x"), ch.get("y")
    agg, grain, title = ch.get("agg", "sum"), ch.get("time_grain", "none"), ch.get("title", "")
    top_n, sort = ch.get("top_n"), ch.get("sort", "desc")
    if x not in df.columns:
        raise ValueError(f"column '{x}' not found")
    if y in (None, "", "None"): y = None
    if y is not None and y not in df.columns:
        raise ValueError(f"column '{y}' not found")
    if agg not in AGGS: agg = "sum"
    d = df.copy()

    if ctype == "histogram":
        d[x] = to_num(d[x])
        return px.histogram(d.dropna(subset=[x]), x=x, title=title), None
    if ctype == "scatter":
        d[x], d[y] = to_num(d[x]), to_num(d[y])
        d = d.dropna(subset=[x, y])
        if len(d) > 5000: d = d.sample(5000, random_state=1)
        return px.scatter(d, x=x, y=y, title=title), None

    if is_date(d[x]):
        if grain in (None, "none") and ctype == "line": grain = "month"
        freq = {"day": "D", "month": "M", "year": "Y"}.get(grain)
        if freq: d[x] = d[x].dt.to_period(freq).dt.to_timestamp()

    if y is None or agg == "count":
        g, ycol = d.groupby(x).size().reset_index(name="count"), "count"
    else:
        if agg not in ("nunique",): d[y] = to_num(d[y])
        g, ycol = d.groupby(x)[y].agg(agg).reset_index(), y

    if ctype == "line" or is_date(g[x]):
        g = g.sort_values(x)
    else:
        g = g.sort_values(ycol, ascending=(sort == "asc")) if sort != "none" else g
        if ctype == "pie" and not top_n: top_n = 8
        if top_n: g = g.head(int(top_n))

    if ctype == "line": fig = px.line(g, x=x, y=ycol, title=title, markers=True)
    elif ctype == "pie": fig = px.pie(g, names=x, values=ycol, title=title)
    else:
        fig = px.bar(g, x=x, y=ycol, title=title)
        if not is_date(g[x]):
            fig.update_layout(xaxis={"categoryorder": "total ascending" if sort == "asc" else "total descending"})
    return fig, (g, x, ycol, ctype)


def insight_from(info, title):
    """Simple rule-based insight from an aggregated table."""
    if not info: return None
    g, x, ycol, ctype = info
    if g.empty: return None
    try:
        if ctype == "line" and len(g) >= 2:
            first, last = g[ycol].iloc[0], g[ycol].iloc[-1]
            peak = g.loc[g[ycol].idxmax()]
            chg = (last - first) / first * 100 if first else 0
            xs = lambda v: v.strftime("%b %Y") if hasattr(v, "strftime") else str(v)
            return (f"**{title}:** peak was {fmt_num(peak[ycol])} in {xs(peak[x])}; "
                    f"latest period is {fmt_num(last)} ({chg:+.1f}% vs first period).")
        if ctype in ("bar", "pie"):
            tot = g[ycol].sum()
            top = g.loc[g[ycol].idxmax()]
            low = g.loc[g[ycol].idxmin()]
            share = top[ycol] / tot * 100 if tot else 0
            return (f"**{title}:** '{top[x]}' is highest at {fmt_num(top[ycol])} "
                    f"({share:.1f}% of shown total); '{low[x]}' is lowest at {fmt_num(low[ycol])}.")
    except Exception:
        return None
    return None


def sidebar_filters(df, cols):
    st.sidebar.header("🔍 Filters")
    shown = False
    for col in cols:
        if col not in df.columns: continue
        s = df[col]
        if is_date(s):
            if s.notna().any():
                lo, hi = s.min().date(), s.max().date()
                rng = st.sidebar.date_input(col, (lo, hi), min_value=lo, max_value=hi, key=f"f_{col}")
                if isinstance(rng, tuple) and len(rng) == 2:
                    df = df[(s >= pd.Timestamp(rng[0])) & (s < pd.Timestamp(rng[1]) + pd.Timedelta(days=1))]
                shown = True
        elif s.nunique() <= 200:
            opts = sorted(s.dropna().astype(str).unique())
            chosen = st.sidebar.multiselect(col, opts, default=[], key=f"f_{col}", placeholder="All")
            if chosen: df = df[s.astype(str).isin(chosen)]
            shown = True
    if not shown: st.sidebar.caption("No filters for this dashboard.")
    return df


def render_dashboard(df, spec):
    st.header(spec.get("title", "Dashboard"))
    if df.empty:
        st.warning("No data after filters.")
        return
    insights = []

    kpis = spec.get("kpis", [])
    if kpis:
        for c, k in zip(st.columns(len(kpis)), kpis):
            try: c.metric(k["title"], fmt_num(compute_kpi(df, k)))
            except Exception as e: c.error(f"KPI error: {e}")

    charts = spec.get("charts", [])
    for i in range(0, len(charts), 2):
        for col_ui, ch in zip(st.columns(2), charts[i:i + 2]):
            with col_ui:
                try:
                    fig, info = build_chart(df, ch)
                    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    ins = insight_from(info, ch.get("title", ""))
                    if ins: insights.append(ins)
                except Exception as e:
                    st.warning(f"Could not build '{ch.get('title', '')}': {e}")

    if insights:
        st.subheader("💡 Quick insights")
        for t in insights: st.markdown(f"- {t}")


# ----------------------------------------------------------------------------
# Understanding free text: fuzzy matching, validation and the query engine
# ----------------------------------------------------------------------------
AGG_WORD = {"sum": "Total", "mean": "Average", "count": "Count of", "nunique": "Unique",
            "max": "Max", "min": "Min", "median": "Median"}
MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
          "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
          "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
STOPWORDS = set("""a an the of in on at for to by from with and or is are was were be me my show give tell
what which who whom how much many total sum average avg mean count number list all each every per wise
top best worst bottom highest lowest most least please kindly data sheet file chart graph dashboard report
sales sale amount value this last next month year week day days weeks months years""".split())


def clean(s: str) -> str:
    """lowercase, punctuation removed, single spaces"""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%.]", " ", str(s).lower())).strip()


def match_column(df, name):
    """Map a (possibly misspelled / different case) column name to a real column, or None."""
    if name is None:
        return None
    name = str(name).strip()
    if not name or name.lower() in ("none", "null", "nan"):
        return None
    if name in df.columns:
        return name
    n = norm(name)
    by_norm = {norm(c): c for c in df.columns}
    if n in by_norm:
        return by_norm[n]
    cands = [c for k, c in by_norm.items() if len(k) >= 3 and n and (n in k or k in n)]
    if len(cands) == 1:
        return cands[0]
    close = difflib.get_close_matches(n, list(by_norm), n=1, cutoff=0.7)
    return by_norm[close[0]] if close else None


def match_value(series, val):
    """Map a (possibly misspelled) value to a real value in a text column, or None."""
    v = str(val).strip()
    uniq = series.dropna().astype(str).unique()
    if v in uniq:
        return v
    low = {u.strip().casefold(): u for u in uniq}
    if v.casefold() in low:
        return low[v.casefold()]
    subs = [u for k, u in low.items() if len(v) >= 3 and (v.casefold() in k)]
    if len(subs) == 1:
        return subs[0]
    close = difflib.get_close_matches(v.casefold(), list(low), n=1, cutoff=0.75)
    return low[close[0]] if close else None


def num(v):
    try:
        return float(str(v).replace(",", "").replace("₹", "").replace("$", "").strip())
    except ValueError:
        return float("nan")


def sanitize_spec(df, spec):
    """Fix spelling of column names in a dashboard spec and drop anything that can't work."""
    kpis, charts, notes = [], [], []
    for k in spec.get("kpis") or []:
        col = match_column(df, k.get("column"))
        if col is None:
            notes.append(f"KPI '{k.get('title', '')}' skipped (column '{k.get('column')}' not found)")
            continue
        kpis.append({**k, "column": col, "agg": k.get("agg") if k.get("agg") in AGGS else "sum"})
    for ch in spec.get("charts") or []:
        x = match_column(df, ch.get("x"))
        y = match_column(df, ch.get("y"))
        if x is None:
            notes.append(f"Chart '{ch.get('title', '')}' skipped (column '{ch.get('x')}' not found)")
            continue
        if ch.get("type") not in ("bar", "line", "pie", "scatter", "histogram"):
            ch = {**ch, "type": "bar"}
        charts.append({**ch, "x": x, "y": y})
    filters = []
    for f in spec.get("filters") or []:
        col = match_column(df, f)
        if col and col not in filters:
            filters.append(col)
    return {**spec, "kpis": kpis, "charts": charts, "filters": filters[:4]}, notes


# ---- date phrases ------------------------------------------------------------
def month_range(y, m):
    lo = pd.Timestamp(year=y, month=m, day=1)
    return lo, lo + pd.offsets.MonthEnd(0)


def parse_time_phrase(p, dmin, dmax):
    """Find a date range in the text. Returns (lo, hi, label) or None. 'Today' = latest date in data."""
    ref = dmax.normalize()
    m = re.search(r"\blast (\d+) (day|week|month|year)s?\b", p)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "day": lo = ref - pd.Timedelta(days=n - 1)
        elif unit == "week": lo = ref - pd.Timedelta(days=7 * n - 1)
        elif unit == "month": lo = ref - pd.DateOffset(months=n) + pd.Timedelta(days=1)
        else: lo = ref - pd.DateOffset(years=n) + pd.Timedelta(days=1)
        return lo, ref, f"last {n} {unit}s"
    if re.search(r"\byesterday\b", p):
        d = ref - pd.Timedelta(days=1); return d, d, "yesterday"
    if re.search(r"\btoday\b", p):
        return ref, ref, "today"
    if re.search(r"\bthis month\b", p):
        lo, hi = month_range(ref.year, ref.month); return lo, hi, "this month"
    if re.search(r"\blast month\b", p):
        prev = ref - pd.DateOffset(months=1)
        lo, hi = month_range(prev.year, prev.month); return lo, hi, "last month"
    if re.search(r"\bthis quarter\b|\blast quarter\b", p):
        q = (ref.month - 1) // 3
        if "last quarter" in p:
            q -= 1
            y = ref.year
            if q < 0: q, y = 3, y - 1
        else:
            y = ref.year
        lo = pd.Timestamp(year=y, month=q * 3 + 1, day=1)
        return lo, (lo + pd.DateOffset(months=3) - pd.Timedelta(days=1)), ("last quarter" if "last quarter" in p else "this quarter")
    if re.search(r"\bthis year\b", p):
        return pd.Timestamp(ref.year, 1, 1), pd.Timestamp(ref.year, 12, 31), "this year"
    if re.search(r"\blast year\b", p):
        return pd.Timestamp(ref.year - 1, 1, 1), pd.Timestamp(ref.year - 1, 12, 31), "last year"
    m = re.search(r"\bq([1-4])\b(?:\s+(?:fy\s*)?(\d{4}))?", p)
    if m:
        q, y = int(m.group(1)), int(m.group(2) or ref.year)
        lo = pd.Timestamp(year=y, month=(q - 1) * 3 + 1, day=1)
        return lo, (lo + pd.DateOffset(months=3) - pd.Timedelta(days=1)), f"Q{q} {y}"
    m = re.search(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b(?:\s+(\d{4}))?", p)
    if m:
        mon = MONTHS[m.group(1)]
        if m.group(2): y = int(m.group(2))
        else:
            y = ref.year if mon <= ref.month else ref.year - 1
            y = max(y, dmin.year)
        lo, hi = month_range(y, mon)
        return lo, hi, lo.strftime("%b %Y")
    m = re.search(r"\b((?:19|20)\d{2})\b", p)
    if m:
        y = int(m.group(1))
        return pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31), str(y)
    return None


# ---- the query engine (all numbers are computed here, never by the AI) --------
OP_ALIASES = {"=": "==", "eq": "==", "is": "==", "equals": "==", "equal": "==", "ne": "!=", "not": "!=",
              "<>": "!=", "gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "nin": "not in",
              "like": "contains", "includes": "contains"}


def filter_mask(s, col, f, notes):
    """Boolean mask for one filter, plus a readable description. Returns (mask, text) or (None, None)."""
    op = str(f.get("op") or "==").lower().strip()
    op = OP_ALIASES.get(op, op)
    val = f.get("value")

    if is_date(s):
        one = pd.Timedelta(days=1)
        if op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
            lo, hi = pd.to_datetime(val[0], errors="coerce"), pd.to_datetime(val[1], errors="coerce")
            if pd.isna(lo) or pd.isna(hi):
                notes.append(f"Couldn't read the dates for {col}; ignored."); return None, None
            return (s >= lo.normalize()) & (s < hi.normalize() + one), \
                f"{col} from {lo:%d %b %Y} to {hi:%d %b %Y}"
        v = pd.to_datetime(val if not isinstance(val, (list, tuple)) else val[0], errors="coerce")
        if pd.isna(v):
            notes.append(f"Couldn't read the date '{val}' for {col}; ignored."); return None, None
        d0 = v.normalize()
        masks = {"==": (s >= d0) & (s < d0 + one), ">=": s >= d0, ">": s >= d0 + one,
                 "<": s < d0, "<=": s < d0 + one}
        if op not in masks:
            notes.append(f"Unknown date condition '{op}'; ignored."); return None, None
        return masks[op], f"{col} {op} {d0:%d %b %Y}"

    if pd.api.types.is_numeric_dtype(s):
        sn = to_num(s)
        if op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
            lo, hi = num(val[0]), num(val[1])
            return sn.between(lo, hi), f"{col} between {lo:g} and {hi:g}"
        if op in ("in", "not in"):
            vals = [num(x) for x in (val if isinstance(val, (list, tuple)) else [val])]
            m = sn.isin(vals)
            return (m if op == "in" else ~m), f"{col} {op} {vals}"
        v = num(val if not isinstance(val, (list, tuple)) else val[0])
        if pd.isna(v):
            notes.append(f"'{val}' is not a number for {col}; ignored."); return None, None
        ops = {"==": sn == v, "!=": sn != v, ">": sn > v, ">=": sn >= v, "<": sn < v, "<=": sn <= v}
        if op not in ops:
            notes.append(f"Unknown condition '{op}' on {col}; ignored."); return None, None
        return ops[op], f"{col} {op} {v:g}"

    # text column
    st_ = s.astype(str)
    vals = val if isinstance(val, (list, tuple)) else [val]
    if op == "contains":
        pat = str(vals[0])
        return st_.str.contains(pat, case=False, na=False, regex=False), f"{col} contains '{pat}'"
    real, bad = [], []
    for v in vals:
        mv = match_value(s, v)
        (real if mv is not None else bad).append(mv if mv is not None else v)
    if bad:
        sample = ", ".join(map(str, s.dropna().astype(str).unique()[:8]))
        notes.append(f"I couldn't find '{', '.join(map(str, bad))}' in column {col}. Values there look like: {sample}")
    negative = op in ("!=", "not in")
    if not real:
        return (pd.Series(True, index=s.index) if negative else pd.Series(False, index=s.index)), f"{col} = {vals}"
    m = st_.isin(real)
    return (~m if negative else m), f"{col} {'is not' if negative else 'is'} {', '.join(real)}"


def run_query(df, q):
    """Run a structured query on the real data. Returns a dict with the table and a plain-English summary."""
    notes, ftexts = [], []
    d = df
    for f in q.get("filters") or []:
        col = match_column(df, f.get("column"))
        if col is None:
            notes.append(f"I couldn't find a column like '{f.get('column')}', so I ignored that condition.")
            continue
        mask, txt = filter_mask(d[col], col, f, notes)
        if mask is not None:
            d = d[mask]
            ftexts.append(txt)
    n_rows = len(d)
    d = d.copy()

    gb = []
    for g in q.get("group_by") or []:
        col = match_column(df, g)
        if col is None:
            notes.append(f"I couldn't find a column like '{g}' to group by; ignored.")
        elif col not in gb:
            gb.append(col)
    grain = str(q.get("time_grain") or "").lower()
    for g in gb:
        if is_date(d[g]):
            freq = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}.get(grain or "month")
            if freq:
                d[g] = d[g].dt.to_period(freq).dt.to_timestamp()

    specs = []      # (label, column, agg)
    for m in q.get("metrics") or []:
        agg = str(m.get("agg") or "sum").lower()
        agg = agg if agg in AGGS else "sum"
        col = match_column(df, m.get("column"))
        if col is None:
            if m.get("column") not in (None, "", "null", "None"):
                notes.append(f"I couldn't find a column like '{m.get('column')}'; ignored.")
                continue
            specs.append(("Rows", None, "count"))
            continue
        if agg not in ("count", "nunique") and to_num(d[col]).notna().sum() == 0 and d[col].notna().any():
            notes.append(f"Column {col} is not numeric, so I counted it instead of using '{agg}'.")
            agg = "count"
        specs.append((f"{AGG_WORD[agg]} {col}", col, agg))
    if not specs:
        specs = [("Rows", None, "count")]
    seen = {}
    for i, (lab, col, agg) in enumerate(specs):
        seen[lab] = seen.get(lab, 0) + 1
        if seen[lab] > 1:
            specs[i] = (f"{lab} ({seen[lab]})", col, agg)

    for _, col, agg in specs:
        if col and agg not in ("count", "nunique"):
            d[col] = to_num(d[col])

    if gb:
        grp = d.groupby(gb)
        cols = {}
        for lab, col, agg in specs:
            if col is None: cols[lab] = grp.size()
            elif agg == "count": cols[lab] = grp[col].count()
            elif agg == "nunique": cols[lab] = grp[col].nunique()
            else: cols[lab] = grp[col].agg(agg)
        table = pd.concat(cols, axis=1).reset_index()
    else:
        row = {}
        for lab, col, agg in specs:
            if col is None: row[lab] = len(d)
            elif agg == "count": row[lab] = d[col].count()
            elif agg == "nunique": row[lab] = d[col].nunique()
            else: row[lab] = getattr(d[col], agg)() if len(d) else float("nan")
        table = pd.DataFrame([row])

    mcols = [s[0] for s in specs]
    order = str(q.get("sort") or q.get("order") or "desc").lower()
    asc = order.startswith("asc")
    sort_by = match_column(table, q.get("sort_by")) if q.get("sort_by") else None
    if gb and order != "none" or sort_by:
        if sort_by:
            table = table.sort_values(sort_by, ascending=asc)
        elif any(is_date(table[g]) for g in gb):
            table = table.sort_values(gb)
        else:
            table = table.sort_values(mcols[0], ascending=asc)
    limit = q.get("limit") or q.get("top_n")
    try:
        if limit: table = table.head(int(limit))
    except (TypeError, ValueError):
        pass
    table = table.reset_index(drop=True)

    what = ", ".join(mcols)
    understood = what
    if gb: understood += " by " + ", ".join(gb)
    if ftexts: understood += " where " + "; ".join(ftexts)
    if limit: understood += f" (top {limit})"
    return {"table": table, "notes": notes, "n_rows": n_rows, "group_cols": gb,
            "metric_cols": mcols, "understood": understood}


def summarize_result(res):
    t, gb, mc = res["table"], res["group_cols"], res["metric_cols"]
    if t.empty:
        return None
    if not gb:
        return "  |  ".join(f"**{c}:** {fmt_num(t[c].iloc[0])}" for c in mc) + f"   _(from {res['n_rows']:,} matching rows)_"
    m = mc[0]
    lab = lambda r: " / ".join(r[g].strftime("%b %Y") if hasattr(r[g], "strftime") else str(r[g]) for g in gb)
    top, low = t.loc[t[m].idxmax()], t.loc[t[m].idxmin()]
    if len(t) == 1:
        return f"**{lab(top)}** → {m}: **{fmt_num(top[m])}**   _(from {res['n_rows']:,} matching rows)_"
    out = f"Highest {m}: **{lab(top)}** ({fmt_num(top[m])}).  Lowest: **{lab(low)}** ({fmt_num(low[m])})."
    return out + f"   _(from {res['n_rows']:,} matching rows)_"


# ---- local (no-AI) understanding ----------------------------------------------
def find_values(df, p_clean):
    """Category values that the text mentions, e.g. 'delhi' -> (Store, 'Delhi')."""
    padded = f" {p_clean} "
    hits = []
    for c in df.columns:
        s = df[c]
        if is_date(s) or pd.api.types.is_numeric_dtype(s) or s.nunique() > 3000:
            continue
        for v in s.dropna().astype(str).unique():
            nv = clean(v)
            if len(nv) >= 2 and (f" {nv} " in padded or f" {nv}s " in padded):
                hits.append((c, v, nv))
    hits.sort(key=lambda h: -len(h[2]))
    kept = []
    for h in hits:
        if not any(h[2] != k[2] and h[2] in k[2] for k in kept):
            kept.append(h)
    return kept


def fuzzy_values(df, p_clean, used_text):
    """Typo-tolerant value matching, e.g. 'dehli' -> Delhi."""
    col_tokens = set()
    for c in df.columns:
        col_tokens |= set(clean(c).split())
    hits = []
    cat_cols = [c for c in df.columns if not is_date(df[c]) and not pd.api.types.is_numeric_dtype(df[c])
                and df[c].nunique() <= 500]
    for tok in p_clean.split():
        if len(tok) < 5 or tok in STOPWORDS or tok in col_tokens or tok in used_text or tok.rstrip("s") in col_tokens:
            continue
        if any(difflib.SequenceMatcher(None, tok, ct).ratio() >= 0.8 for ct in col_tokens):
            continue
        for c in cat_cols:
            keys = {clean(v): v for v in df[c].dropna().astype(str).unique()}
            close = difflib.get_close_matches(tok, list(keys), n=1, cutoff=0.78)
            if close:
                hits.append((c, keys[close[0]], close[0]))
                break
    return hits


def plan_local(df, prompt):
    """No-key understanding: turns free text into either a query or a dashboard."""
    p = norm(prompt)
    pc = clean(prompt)
    ms_all, ds_all, dt_all = measures(df), dims(df), dates(df)
    ments = find_mentioned(df, p)

    # column named by one distinctive word, e.g. "revenue" -> "Revenue Amount"
    tokmap = {}
    for c in df.columns:
        for t in set(norm(c).split()):
            if len(t) >= 4 and t not in ID_WORDS:
                tokmap.setdefault(t, []).append(c)
    for t, cs in tokmap.items():
        if len(cs) == 1 and re.search(rf"\b{re.escape(t)}s?\b", p) and cs[0] not in ments:
            ments.append(cs[0])

    if re.search(r"\b(dashboard|overview|report|summary of everything)\b", p):
        return {"intent": "dashboard", "dashboard": plan_from_prompt(df, prompt)}

    vals = find_values(df, pc)
    used = " ".join(h[2] for h in vals)
    vals += fuzzy_values(df, pc, used)
    filters, assumptions = [], []
    by_col = {}
    for c, v, _ in vals:
        by_col.setdefault(c, []).append(v)
    for c, vs in by_col.items():
        filters.append({"column": c, "op": "in" if len(vs) > 1 else "==", "value": vs if len(vs) > 1 else vs[0]})

    time_txt = None
    if dt_all:
        dcol = next((c for c in ments if is_date(df[c])), dt_all[0])
        tr = parse_time_phrase(p, df[dcol].min(), df[dcol].max())
        if tr:
            lo, hi, label = tr
            filters.append({"column": dcol, "op": "between", "value": [str(lo.date()), str(hi.date())]})
            time_txt = label
            if re.match(r"(last|this|yesterday|today)", label):
                assumptions.append(f"'{label}' is counted from the latest date in your data ({df[dcol].max():%d %b %Y}).")

    asked = bool(re.search(r"\b(what|which|who|how much|how many|kitna|kitne|total|average|avg|highest|lowest|top|bottom|"
                           r"best|worst|most|least|compare|list|show|give|count|trend|growth)\b", p))
    if not (filters or (asked and ments)):
        return {"intent": "dashboard", "dashboard": plan_from_prompt(df, prompt)}

    # aggregation
    if re.search(r"\b(average|avg|mean)\b", p): agg = "mean"
    elif re.search(r"\b(how many|number of|count|kitne)\b", p): agg = "count"
    elif re.search(r"\b(unique|distinct)\b", p): agg = "nunique"
    elif re.search(r"\b(maximum|max)\b", p): agg = "max"
    elif re.search(r"\b(minimum|min)\b", p): agg = "min"
    else: agg = "sum"
    asc = bool(re.search(r"\b(bottom|worst|lowest|least|minimum)\b", p)) and agg not in ("min",)
    m = re.search(r"\b(?:top|best|bottom|worst|first)\s*(\d+)\b", p)
    limit = int(m.group(1)) if m else None

    m_sel = [c for c in ments if c in ms_all or (pd.api.types.is_numeric_dtype(df[c]) and not is_date(df[c]))]
    d_sel = [c for c in ments if c in ds_all and c not in m_sel]
    filtered_cols = set(by_col)
    group = []
    for c in d_sel:
        n_ = re.escape(norm(c))
        explicit = re.search(rf"\b(?:by|per|each|every|wise)\s+{n_}s?\b|\b{n_}s?\s+wise\b|\b(?:which|top|best|worst)\s+(?:\d+\s+)?{n_}", p)
        if c not in filtered_cols or explicit:
            group.append(c)

    grain = "day" if re.search(r"\b(daily|day wise|per day)\b", p) else \
            "week" if re.search(r"\b(weekly|week wise|per week)\b", p) else \
            "quarter" if re.search(r"\b(quarterly|quarter wise)\b", p) else \
            "year" if re.search(r"\b(yearly|annual|year wise|per year)\b", p) else "month"
    trend = bool(re.search(r"\b(trend|over time|monthly|daily|weekly|quarterly|yearly|growth|timeline|month wise)\b", p))
    if trend and dt_all:
        group = [next((c for c in ments if is_date(df[c])), dt_all[0])] + group

    if agg == "count" or (not m_sel and not ms_all):
        metrics = [{"column": None, "agg": "count"}]
    else:
        use = m_sel[:3] or ms_all[:1]
        if not m_sel:
            assumptions.append(f"No value column was named, so I used '{use[0]}'.")
        metrics = [{"column": c, "agg": agg} for c in use]

    title = prompt.strip().capitalize()[:80]
    q = {"title": title, "filters": filters, "group_by": group, "time_grain": grain if trend else "month",
         "metrics": metrics, "sort": "asc" if asc else "desc", "limit": limit,
         "chart": "line" if trend else "bar", "assumptions": " ".join(assumptions)}
    return {"intent": "query", "query": q}


# ----------------------------------------------------------------------------
# Gemini AI planner (understands free text, but never calculates numbers itself)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior data analyst inside a spreadsheet app. The user types ANYTHING in plain
language (English, Hindi, Hinglish, short words and spelling mistakes are all fine) about their uploaded data.
Work out what they really mean and map it to the REAL columns and REAL values of the dataset.
You only write a plan. The app calculates the numbers itself, so never invent or calculate numbers.

Decide the intent:
- "query": a question, lookup, list, ranking or comparison with a specific answer
  (e.g. "delhi store ki sales last month", "top 5 products", "which category is lowest", "how many orders in march").
- "dashboard": they want an overview / dashboard / report / many charts.

Reply with ONLY valid JSON in exactly this shape (leave out the part you do not use):
{
  "intent": "query | dashboard",
  "query": {
    "title": "short title",
    "filters": [{"column": "exact column", "op": "== | != | in | not in | > | >= | < | <= | between | contains",
                 "value": "text or number or [list] (for between: [low, high])"}],
    "group_by": ["exact column"],
    "time_grain": "day | week | month | quarter | year | none",
    "metrics": [{"column": "exact column or null to count rows", "agg": "sum | mean | count | nunique | max | min | median"}],
    "sort": "desc | asc | none", "sort_by": null, "limit": null,
    "chart": "bar | line | pie | none",
    "assumptions": "one short sentence about anything you had to guess, or empty string"
  },
  "dashboard": {
    "title": "Dashboard title",
    "kpis": [{"title": "Total Sales", "column": "exact column", "agg": "sum"}],
    "charts": [{"type": "bar | line | pie | scatter | histogram", "title": "Chart title",
                "x": "exact column", "y": "exact column or null",
                "agg": "sum | mean | count | nunique | max | min | median",
                "time_grain": "none | day | month | year", "top_n": null, "sort": "asc | desc | none"}],
    "filters": ["exact column"]
  }
}
Rules:
- Use ONLY column names that exist, spelled exactly. For category filters use the EXACT values listed in the
  dataset description (fix spelling, case and language, e.g. "dehli" -> "Delhi").
- Dates must be ISO YYYY-MM-DD. Resolve relative words (last month, this year, last 7 days, Q1, march...)
  using the LATEST DATE IN THE DATA as "today", and say so in assumptions.
- "total / sales of / kitna" -> sum. "average" -> mean. "how many / count of rows" -> count with column null.
  "unique / distinct" -> nunique.
- highest / best / top -> sort desc. lowest / worst / bottom -> sort asc. "top N" -> limit N.
- "by X", "X wise", "per X", "which X", "each X" -> group_by X. A trend over time -> group_by the date column,
  set time_grain, chart "line".
- If the question cannot be answered from these columns, return intent "query" with empty metrics and explain
  why in assumptions. Never make up columns or values.
- If a PREVIOUS QUERY is given and the request is clearly a follow-up ("now only Delhi", "same for Mumbai",
  "show as pie"), modify that query. Otherwise ignore it.
- For a dashboard: 3-5 KPIs and 3-6 charts unless asked otherwise. If updating a CURRENT DASHBOARD,
  return the FULL updated dashboard."""


def describe_dataset(df: pd.DataFrame) -> str:
    lines = [f"Rows: {len(df)}", "Columns:"]
    for col in df.columns:
        s = df[col]
        info = f"- {col} ({s.dtype}), unique={s.nunique()}"
        if is_date(s):
            info += f", from {s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}"
        elif pd.api.types.is_numeric_dtype(s):
            info += f", min={s.min()}, max={s.max()}"
        else:
            vc = s.dropna().astype(str).value_counts()
            if len(vc) <= 40:
                info += ", values=" + json.dumps(list(vc.index), ensure_ascii=False)
            else:
                info += ", most common values=" + json.dumps(list(vc.index[:15]), ensure_ascii=False)
        lines.append(info)
    dts = [c for c in df.columns if is_date(df[c])]
    if dts:
        lines.append(f"LATEST DATE IN DATA: {max(df[c].max() for c in dts):%Y-%m-%d}")
    sample = df.head(5).copy()
    for c in sample.columns:
        if is_date(sample[c]):
            sample[c] = sample[c].astype(str)
    lines.append("Sample rows (JSON): " + sample.to_json(orient="records", force_ascii=False))
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("AI did not return JSON.")
    return json.loads(text[a:b + 1])


def ask_gemini(api_key, model, schema, request, previous_spec=None, previous_query=None, force_dashboard=False):
    """Try the chosen model first, then a few fallbacks. Returns (result, model_used)."""
    content = f"DATASET:\n{schema}\n\nUSER REQUEST: {request}"
    if previous_query and not force_dashboard:
        content += f"\n\nPREVIOUS QUERY JSON (only use if the request is a follow-up):\n{json.dumps(previous_query)}"
    if force_dashboard:
        content += "\n\nThe user pressed UPDATE DASHBOARD: intent MUST be \"dashboard\"."
        if previous_spec:
            content += f"\nCURRENT DASHBOARD JSON (update this):\n{json.dumps(previous_spec)}"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": content}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    models = [model] + [m for m in ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"] if m != model]
    last_err = "unknown error"
    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        try:
            r = requests.post(url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                              json=body, timeout=60)
        except requests.RequestException as e:
            last_err = f"network problem: {e}"
            continue
        if r.status_code == 200:
            try:
                parts = r.json()["candidates"][0]["content"]["parts"]
                raw = extract_json("".join(p.get("text", "") for p in parts))
                intent = "dashboard" if force_dashboard else raw.get("intent")
                if intent == "dashboard" or ("charts" in raw and "query" not in raw):
                    dash = raw.get("dashboard") or raw
                    dash.setdefault("kpis", []); dash.setdefault("charts", []); dash.setdefault("filters", [])
                    return {"intent": "dashboard", "dashboard": dash}, m
                q = raw.get("query")
                if not isinstance(q, dict):
                    raise ValueError("no query in reply")
                return {"intent": "query", "query": q}, m
            except Exception as e:
                last_err = f"{m}: unreadable reply ({e})"
                continue
        if r.status_code in (400, 401, 403) and ("API key" in r.text or "API_KEY" in r.text):
            raise PermissionError("Invalid API key. Create a new one at aistudio.google.com")
        last_err = f"{m}: HTTP {r.status_code} - {r.text[:150]}"
    raise RuntimeError(last_err)


# ----------------------------------------------------------------------------
# Answer card (for questions)
# ----------------------------------------------------------------------------
def answer_chart(res, q):
    t, gb, mc = res["table"], res["group_cols"], res["metric_cols"]
    if not gb or len(t) < 2:
        return None
    ctype = str(q.get("chart") or ("line" if is_date(t[gb[0]]) else "bar")).lower()
    if ctype == "none":
        return None
    try:
        if len(gb) == 1:
            x = gb[0]
            if len(mc) == 1:
                y = mc[0]
                if ctype == "pie": fig = px.pie(t.head(12), names=x, values=y)
                elif ctype == "line": fig = px.line(t, x=x, y=y, markers=True)
                else: fig = px.bar(t.head(30), x=x, y=y)
            else:
                long = t.melt(id_vars=x, value_vars=mc, var_name="metric", value_name="value")
                fig = (px.line(long, x=x, y="value", color="metric", markers=True) if ctype == "line"
                       else px.bar(long, x=x, y="value", color="metric", barmode="group"))
        elif len(gb) == 2 and len(mc) == 1:
            fig = (px.line(t, x=gb[0], y=mc[0], color=gb[1], markers=True) if ctype == "line"
                   else px.bar(t, x=gb[0], y=mc[0], color=gb[1], barmode="group"))
        else:
            return None
        if not is_date(t[gb[0]]) and ctype != "pie":
            fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": list(t[gb[0]])})
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        return fig
    except Exception:
        return None


def render_answer(df, ans):
    q = ans["query"]
    res = run_query(df, q)
    st.header("💬 " + (q.get("title") or "Answer"))
    st.markdown("**How I understood it:** " + res["understood"])
    if q.get("assumptions"):
        st.caption("ℹ️ " + str(q["assumptions"]))
    for n in res["notes"]:
        st.warning(n)
    t = res["table"]
    if res["n_rows"] == 0:
        st.error("No rows match these conditions, so there is nothing to calculate. "
                 "Check the spelling/dates, or rephrase your question.")
        return
    summary = summarize_result(res)
    if summary:
        st.success(summary)
    gb, mc = res["group_cols"], res["metric_cols"]
    if not gb:
        for c, col_ui in zip(mc, st.columns(len(mc))):
            col_ui.metric(c, fmt_num(t[c].iloc[0]))
        return
    fig = answer_chart(res, q)
    left, right = st.columns([1, 1.2]) if fig is not None else (st.container(), None)
    with left:
        st.dataframe(t.head(200), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download this table (CSV)", t.to_csv(index=False).encode("utf-8"),
                           file_name="answer.csv", mime="text/csv")
    if fig is not None:
        right.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def manual_builder(df, spec):
    with st.expander("🛠️ Build manually (add your own chart or KPI)"):
        cols = list(df.columns)
        c1, c2, c3, c4 = st.columns(4)
        ctype = c1.selectbox("Chart type", ["bar", "line", "pie", "scatter", "histogram"])
        x = c2.selectbox("X (category / date)", cols, key="mx")
        y = c3.selectbox("Y (value)", ["(none - count rows)"] + cols, key="my")
        agg = c4.selectbox("Aggregation", AGGS)
        c5, c6 = st.columns(2)
        grain = c5.selectbox("Date grouping", ["month", "day", "year", "none"])
        top_n = c6.number_input("Top N (0 = all)", 0, 100, 10)
        if st.button("➕ Add chart"):
            spec["charts"].append({
                "type": ctype, "title": f"{y if y != '(none - count rows)' else 'Rows'} by {x}",
                "x": x, "y": None if y.startswith("(none") else y, "agg": agg,
                "time_grain": grain, "top_n": int(top_n) or None, "sort": "desc"})
            st.rerun()
        k1, k2 = st.columns(2)
        kcol = k1.selectbox("KPI column", cols, key="kc")
        kagg = k2.selectbox("KPI aggregation", AGGS, key="ka")
        if st.button("➕ Add KPI"):
            spec["kpis"].append({"title": f"{kagg.title()} {kcol}", "column": kcol, "agg": kagg})
            st.rerun()
        if st.button("🗑️ Remove last chart") and spec["charts"]:
            spec["charts"].pop()
            st.rerun()


def main():
    st.title("📊 AI Dashboard Maker (Free Gemini)")
    st.caption("Upload a sheet and just write what you want, in any words. Ask a question or ask for a dashboard. "
               "The AI understands, the app calculates the exact numbers.")
    st.session_state.setdefault("spec", None)
    st.session_state.setdefault("answer", None)

    with st.sidebar:
        st.header("⚙️ Settings")
        dayfirst = st.checkbox("Dates are DD/MM/YYYY", value=True)
        api_key = st.text_input("Gemini API key (free)", type="password",
                                value=os.getenv("GEMINI_API_KEY", ""),
                                help="Get a free key at aistudio.google.com")
        model = st.text_input("Gemini model", value="gemini-3.8-flash",
                              help="If you get a model error, change this to another model name from aistudio.google.com")
        st.caption("Sent to Google: column names, category values, 5 sample rows and your text. "
                   "Never send sensitive data on the free tier.")

    file = st.file_uploader("Upload your sheet", type=["csv", "xlsx", "xls"])
    if not file:
        st.info("👆 Upload a CSV or Excel file to begin.")
        return

    sheet = None
    if file.name.lower().endswith((".xlsx", ".xls")):
        try:
            names = pd.ExcelFile(io.BytesIO(file.getvalue())).sheet_names
            if len(names) > 1: sheet = st.selectbox("Sheet", names)
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            return

    try:
        df = prepare_df(load_file(file.getvalue(), file.name, sheet), dayfirst)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    key = f"{file.name}|{sheet}|{dayfirst}"
    if st.session_state.get("file_key") != key:
        st.session_state.file_key = key
        st.session_state.spec = auto_plan(df)   # show something immediately
        st.session_state.answer = None

    with st.expander(f"👀 Data preview ({len(df):,} rows × {len(df.columns)} columns)"):
        st.dataframe(df.head(50), use_container_width=True)
        st.caption("Detected measures: " + (", ".join(measures(df)) or "none") +
                   "  |  categories: " + (", ".join(dims(df)) or "none") +
                   "  |  dates: " + (", ".join(dates(df)) or "none"))

    prompt = st.text_area(
        "Write anything about your data",
        placeholder="e.g. delhi store ki total sales last month  |  top 5 products by revenue  |  "
                    "monthly trend of sales for electronics  |  how many orders in march  |  full dashboard",
        height=90)
    c1, c2, c3, _ = st.columns([1.3, 1.5, 1, 2])
    go = c1.button("✨ Ask / Create", type="primary")
    update_ai = c2.button("🔄 Update dashboard", disabled=not st.session_state.spec)
    auto = c3.button("🪄 Auto")

    if auto:
        st.session_state.spec = auto_plan(df)
        st.session_state.answer = None
    elif go or update_ai:
        if not prompt.strip():
            st.warning("Write something first.")
        else:
            result = None
            use_local = not api_key
            if api_key:
                with st.spinner("Understanding your request..."):
                    try:
                        prev_q = (st.session_state.answer or {}).get("query")
                        result, used = ask_gemini(api_key, model.strip(), describe_dataset(df), prompt,
                                                  st.session_state.spec if update_ai else None,
                                                  prev_q, force_dashboard=update_ai)
                        st.success(f"Understood by AI ({used}).")
                    except PermissionError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.warning(f"AI was not available ({e}). Used the built-in understanding instead.")
                        use_local = True
            else:
                st.info("No Gemini key entered, so the built-in understanding was used. "
                        "Add a free key in the sidebar for much smarter understanding.")
            if use_local:
                result = plan_local(df, prompt)
                if update_ai and result["intent"] == "query":
                    result = {"intent": "dashboard", "dashboard": plan_from_prompt(df, prompt)}
            if result:
                if result["intent"] == "dashboard":
                    spec, notes = sanitize_spec(df, result["dashboard"])
                    st.session_state.spec = spec
                    st.session_state.answer = None
                    for n in notes: st.caption("⚠️ " + n)
                else:
                    st.session_state.answer = {"query": result["query"], "asked": prompt.strip()}

    if st.session_state.answer:
        render_answer(df, st.session_state.answer)
        st.caption("Answers use all rows of your file. The sidebar filters only change the dashboard below.")

    spec = st.session_state.spec
    if not spec: return
    st.divider()
    manual_builder(df, spec)
    filtered = sidebar_filters(df, spec.get("filters", []))
    render_dashboard(filtered, spec)


if __name__ == "__main__":
    main()
