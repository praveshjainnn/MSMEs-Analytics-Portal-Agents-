"""
live_data.py  —  Resilient real-time data layer for MSME Analytics Dashboard
=============================================================================
Strategy (tries each in order, first success wins):
  1. World Bank Open Data API     (api.worldbank.org)
  2. IMF DataMapper API           (imf.org)  ← alternative if WB is blocked
  3. Hardcoded fallback           ← latest known values; never fails

Caching: all successful fetches are stored in-process for 1 hour.
SSL:     tries verified → unverified automatically.
"""

import urllib.request
import ssl
import json
import time
import re
from datetime import datetime
from typing import Optional, Dict, List

# ─────────────────────────────────────────────────────────────
#  CACHE
# ─────────────────────────────────────────────────────────────
_cache: Dict = {}
CACHE_TTL = 3600          # 1 hour

def _get(key):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return val
    return None

def _set(key, val):
    _cache[key] = (val, time.time())

def clear_cache():
    """Force-expire all cached entries so the next call re-fetches."""
    _cache.clear()

# ─────────────────────────────────────────────────────────────
#  FALLBACK DATA  (latest known values — never raises)
#  Update these numbers when you have internet access.
# ─────────────────────────────────────────────────────────────
_FALLBACK = {
    "gdp_growth":     {"value": 8.15,  "year": "2023", "live": False},
    "inflation":      {"value": 5.65,  "year": "2023", "live": False},
    "unemployment":   {"value": 3.49,  "year": "2023", "live": False},
    "credit_private": {"value": 57.4,  "year": "2022", "live": False},
    "self_employed":  {"value": 53.9,  "year": "2022", "live": False},
    "manufacturing":  {"value": 14.08, "year": "2022", "live": False},
    "gni_per_capita": {"value": 2270,  "year": "2022", "live": False},
}

# ─────────────────────────────────────────────────────────────
#  WORLD BANK INDICATOR CODES
# ─────────────────────────────────────────────────────────────
WB_CODES = {
    "gdp_growth":     "NY.GDP.MKTP.KD.ZG",
    "inflation":      "FP.CPI.TOTL.ZG",
    "unemployment":   "SL.UEM.TOTL.ZS",
    "credit_private": "FS.AST.PRVT.GD.ZS",
    "self_employed":  "SL.EMP.SELF.ZS",
    "gni_per_capita": "NY.GNP.PCAP.CD",
    "manufacturing":  "NV.IND.MANF.ZS",
}

# IMF indicator codes (alternative source)
IMF_CODES = {
    "gdp_growth":   "NGDP_RPCH",   # Real GDP growth
    "inflation":    "PCPIPCH",     # Inflation (CPI)
    "unemployment": "LUR",         # Unemployment rate
}

_HEADERS = {
    "User-Agent": "MSME-Analytics-Dashboard/2.0",
    "Accept":     "application/json",
}
TIMEOUT = 8  # seconds per request

# ─────────────────────────────────────────────────────────────
#  HELPER: resilient HTTP GET
# ─────────────────────────────────────────────────────────────
def _http_get(url: str) -> Optional[bytes]:
    """
    Try HTTPS with verified cert, then with unverified cert.
    Returns raw bytes or None on complete failure.
    """
    for ctx in [ssl.create_default_context(), ssl._create_unverified_context()]:
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            resp = urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx)
            return resp.read()
        except ssl.SSLError:
            continue           # try next context
        except Exception:
            return None        # non-SSL error — don't retry SSL
    return None

# ─────────────────────────────────────────────────────────────
#  SOURCE 1: WORLD BANK
# ─────────────────────────────────────────────────────────────
def _fetch_wb_indicator(key: str) -> Optional[Dict]:
    code = WB_CODES.get(key)
    if not code:
        return None

    ck = f"wb_{key}"
    cached = _get(ck)
    if cached is not None:
        return cached

    url = (
        f"https://api.worldbank.org/v2/country/IN"
        f"/indicator/{code}?format=json&mrv=3&per_page=3"
    )
    raw = _http_get(url)
    if raw is None:
        return None

    try:
        data = json.loads(raw)
        records = data[1] if (isinstance(data, list) and len(data) > 1) else []
        for rec in records:
            if rec.get("value") is not None:
                result = {
                    "value": round(float(rec["value"]), 2),
                    "year":  rec.get("date", "N/A"),
                    "live":  True,
                    "src":   "World Bank",
                }
                _set(ck, result)
                return result
    except Exception as ex:
        print(f"[LiveData] WB parse error ({key}): {ex}")
    return None

# ─────────────────────────────────────────────────────────────
#  SOURCE 2: IMF DataMapper
# ─────────────────────────────────────────────────────────────
def _fetch_imf_indicator(key: str) -> Optional[Dict]:
    code = IMF_CODES.get(key)
    if not code:
        return None

    ck = f"imf_{key}"
    cached = _get(ck)
    if cached is not None:
        return cached

    url = f"https://www.imf.org/external/datamapper/api/v1/{code}/IND"
    raw = _http_get(url)
    if raw is None:
        return None

    try:
        data = json.loads(raw)
        values = data.get("values", {}).get(code, {}).get("IND", {})
        if not values:
            return None
        # get the most recent year with a non-null value
        for year in sorted(values.keys(), reverse=True):
            v = values[year]
            if v is not None:
                result = {
                    "value": round(float(v), 2),
                    "year":  year,
                    "live":  True,
                    "src":   "IMF",
                }
                _set(ck, result)
                return result
    except Exception as ex:
        print(f"[LiveData] IMF parse error ({key}): {ex}")
    return None

# ─────────────────────────────────────────────────────────────
#  SOURCE 3: FALLBACK (hardcoded latest-known values)
# ─────────────────────────────────────────────────────────────
def _get_fallback(key: str) -> Optional[Dict]:
    fb = _FALLBACK.get(key)
    if fb:
        return dict(fb)   # copy so callers can't mutate
    return None

# ─────────────────────────────────────────────────────────────
#  MAIN: get one indicator with full fallback chain
# ─────────────────────────────────────────────────────────────
def get_indicator(key: str) -> Optional[Dict]:
    """
    Returns {'value', 'year', 'live', 'src'} for key.
    Tries WB → IMF → hardcoded fallback in that order.
    'live' is True only if the value came from a live API.
    """
    return (
        _fetch_wb_indicator(key)
        or _fetch_imf_indicator(key)
        or _get_fallback(key)
    )

# ─────────────────────────────────────────────────────────────
#  MAIN: full India macro snapshot
# ─────────────────────────────────────────────────────────────
def get_india_macro() -> Dict:
    """
    Collect all indicators. Always returns a populated dict
    (at minimum with fallback values so the UI never breaks).
    """
    ck = "india_macro_snapshot"
    cached = _get(ck)
    if cached is not None:
        return cached

    indicators = {}
    any_live = False
    for key in WB_CODES:
        v = get_indicator(key)
        if v:
            indicators[key] = v
            if v.get("live"):
                any_live = True

    result = {
        "fetched_at":    datetime.now().strftime("%d %b %Y  %H:%M IST"),
        "source":        "World Bank / IMF (live)" if any_live else "Cached reference values",
        "any_live":      any_live,
        "indicators":    indicators,
    }
    # only cache if we got at least one live value — otherwise retry sooner
    if any_live:
        _set(ck, result)
    return result

# ─────────────────────────────────────────────────────────────
#  UDYAM PORTAL  —  scrape national total (optional)
# ─────────────────────────────────────────────────────────────
def get_udyam_total() -> Optional[Dict]:
    """
    Attempt to scrape the Udyam Registration portal for the current
    total registered enterprise count. Returns None on failure.
    """
    ck = "udyam_total"
    cached = _get(ck)
    if cached is not None:
        return cached

    for url in [
        "https://udyamregistration.gov.in/",
        "https://udyamregistration.gov.in/Government-India/"
        "Central-Government-Portal-Udyam-Registration.aspx",
    ]:
        raw = _http_get(url)
        if not raw:
            continue
        try:
            html = raw.decode("utf-8", errors="ignore")
            # Look for large Indian-format numbers (e.g. 2,40,73,456)
            for pat in [
                r'(\d{1,2}(?:,\d{2}){3,})',   # crore-style  e.g. 2,40,73,456
                r'(\d{8,})',                    # plain 8+ digit
            ]:
                for m in re.findall(pat, html):
                    n = int(m.replace(",", ""))
                    if n > 1_00_00_000:         # > 1 crore as sanity check
                        result = {
                            "count":  n,
                            "label":  f"{n:,}",
                            "source": "Udyam Portal (live)",
                        }
                        _set(ck, result)
                        return result
        except Exception:
            continue
    return None

# ─────────────────────────────────────────────────────────────
#  FORMAT: text block for Ollama prompts
# ─────────────────────────────────────────────────────────────
_LABELS = {
    "gdp_growth":     "India GDP Growth Rate (%)",
    "inflation":      "India CPI Inflation (%)",
    "unemployment":   "India Unemployment Rate (%)",
    "credit_private": "Credit to Private Sector (% of GDP)",
    "self_employed":  "Self-Employed Workers (% of employed)",
    "gni_per_capita": "GNI per Capita (USD)",
    "manufacturing":  "Manufacturing Value-Added (% of GDP)",
}

def macro_prompt_block(macro: Optional[Dict] = None) -> str:
    """
    Return a text block ready for injection into Ollama prompts.
    Always produces output (uses fallback values if APIs are down).
    """
    if macro is None:
        macro = get_india_macro()

    inds = macro.get("indicators", {})
    if not inds:
        return ""

    is_live = macro.get("any_live", False)
    tag = "LIVE API DATA" if is_live else "REFERENCE DATA (latest known)"
    lines = [
        "",
        f"[{tag} — {macro.get('fetched_at', 'N/A')} | Source: {macro.get('source', '')}]",
    ]
    for k, v in inds.items():
        lbl = _LABELS.get(k, k.replace("_", " ").title())
        src = f"  [{v.get('src', 'ref')}]" if is_live else ""
        lines.append(f"  • {lbl}: {v['value']}  ({v['year']}){src}")

    udyam = get_udyam_total()
    if udyam:
        lines.append(f"  • Total Udyam Registered Enterprises (live): {udyam['label']}")

    return "\n".join(lines) + "\n"

# ─────────────────────────────────────────────────────────────
#  CONNECTIVITY STATUS
# ─────────────────────────────────────────────────────────────
def check_connectivity() -> Dict:
    """
    Quick probe of each live data source.
    Returns a dict with status strings and whether live data is available.
    """
    results = {}

    # World Bank
    raw = _http_get(
        "https://api.worldbank.org/v2/country/IN"
        "/indicator/NY.GDP.MKTP.KD.ZG?format=json&mrv=1&per_page=1"
    )
    results["world_bank"] = "🟢 World Bank — Connected" if raw else "🔴 World Bank — Blocked/Offline"

    # IMF
    raw2 = _http_get("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/IND")
    results["imf"] = "🟢 IMF DataMapper — Connected" if raw2 else "🔴 IMF — Blocked/Offline"

    # Udyam
    raw3 = _http_get("https://udyamregistration.gov.in/")
    results["udyam"] = "🟢 Udyam Portal — Reachable" if raw3 else "🔴 Udyam Portal — Unreachable"

    results["data_available"] = bool(raw or raw2)
    results["using_fallback"] = not bool(raw or raw2)
    return results
