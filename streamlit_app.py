# paper_trader_app.py
"""
Merged Paper-only Options/Derivatives Trading Simulator
- Instrument selection & expiry parsing UI from expiry_underlying_app.py (Code 1)
- Login + ledger + trade logic from paper_trader_app.py (Code 2)
- Harmonized helpers for expiry parsing, strike parsing and instrument lookup
- Simple ledger -> dataframe and FIFO-based position/realized calculation included
"""

import os
import re
import json
from datetime import datetime, date
from functools import lru_cache

import requests
import pandas as pd
import streamlit as st

# Optional SmartAPI (read-only LTP)
try:
    from SmartApi.smartConnect import SmartConnect
except Exception:
    SmartConnect = None

# ---------------- Page config ----------------
APP_TITLE = "Shashidharan_Paper_Trade_API"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"📉 {APP_TITLE} (Paper-only)")

# ---------------- Constants / files ----------------
MASTER_FILENAME = "OpenAPIScripMaster.json"
LEDGER_DIR = "ledgers"
os.makedirs(LEDGER_DIR, exist_ok=True)

# ---------------- Utilities ----------------
def sanitize_name(name):
    if not name:
        return "anon"
    s = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))
    return s or "anon"

def current_user_id():
    """
    Prefer SmartAPI clientcode from st.session_state['profile'] if present.
    Otherwise fallback to a stable sidebar text input 'paper_user'.
    """
    prof = st.session_state.get("profile", None)
    clientcode = None
    if isinstance(prof, dict):
        clientcode = (prof.get("data") or {}).get("clientcode")
    else:
        try:
            if hasattr(prof, "get"):
                clientcode = (prof.get("data") or {}).get("clientcode")
        except Exception:
            clientcode = None
        if clientcode is None:
            try:
                data_attr = getattr(prof, "data", None)
                if isinstance(data_attr, dict):
                    clientcode = data_attr.get("clientcode")
            except Exception:
                pass
    if clientcode:
        return sanitize_name(clientcode)

    if "paper_user" not in st.session_state:
        with st.sidebar:
            st.session_state["paper_user"] = st.text_input("Paper ledger user id", value="user", key="paper_user_input")
    return sanitize_name(st.session_state.get("paper_user", "user"))

# Build per-app + per-user ledger filename
safe_title = re.sub(r'[^A-Za-z0-9_]+', '', APP_TITLE).lower()
uid = current_user_id()
PAPER_TRADE_FILE = os.path.join(LEDGER_DIR, f"{safe_title}_{uid}_trades.json")
if not os.path.exists(PAPER_TRADE_FILE):
    with open(PAPER_TRADE_FILE, "w") as f:
        json.dump([], f)

def ledger_filename_for_user():
    return PAPER_TRADE_FILE

# ----------------- Networking / master load -----------------
@st.cache_data(ttl=600)
def fetch_json_data(url: str):
    headers = {"User-Agent": "streamlit-paper-trader/1.0"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=3600)
def load_master():
    """Load instrument master: prefer local MASTER_FILENAME else fallback to known remote URL."""
    try:
        if os.path.exists(MASTER_FILENAME):
            with open(MASTER_FILENAME, "r") as f:
                return json.load(f)
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # return empty list to allow UI to handle
        return []

# ----------------- Expiry / strike parsing helpers -----------------
MONTH_ALIASES = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'SEPT': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

def try_parse_expiry_string(s: str):
    """Robust parser returning datetime.date or None."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None

    # ISO-like: YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD
    for pat in (r'^(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})$',
                r'^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$'):
        m = re.match(pat, s)
        if m:
            try:
                return datetime(int(m.group('y')), int(m.group('m')), int(m.group('d'))).date()
            except Exception:
                pass

    # DD-MM-YYYY / DD/MM/YYYY
    m = re.match(r'^(?P<d>\d{1,2})[./-](?P<m>\d{1,2})[./-](?P<y>\d{4})$', s)
    if m:
        try:
            return datetime(int(m.group('y')), int(m.group('m')), int(m.group('d'))).date()
        except Exception:
            pass

    # 18NOV2025, 18-NOV-2025, 18 Nov 2025, 18NOV25
    m = re.match(r'^(?P<d>\d{1,2})\s*[-]?\s*(?P<mon>[A-Za-z]{3,5})\s*[-]?\s*(?P<y>\d{2,4})$', s)
    if m:
        dd = int(m.group('d'))
        mon = m.group('mon').upper()[:3]
        yy = m.group('y')
        if len(yy) == 2:
            yy = '20' + yy
        mm = MONTH_ALIASES.get(mon)
        if mm:
            try:
                return datetime(int(yy), mm, dd).date()
            except Exception:
                pass

    # try strptime fallbacks
    for fmt in ("%d%b%Y", "%d%B%Y", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass

    return None

# compatibility wrapper
def parse_expiry(x):
    return try_parse_expiry_string(x)

def parse_strike(raw):
    """Normalize strike to integer. Heuristic for stored *100 values."""
    try:
        s = float(raw)
    except Exception:
        return None
    if s >= 10000:
        s = s / 100.0
    return int(round(s))

# ---------------- instrument master helpers ----------------
def available_underlyings(master):
    names = set()
    for r in master:
        name = (r.get("name") or "").strip().upper()
        itype = (r.get("instrumenttype") or "").upper()
        exch_seg = (r.get("exch_seg") or "").upper()
        if not name:
            continue
        if "OPT" in itype or "FUT" in itype or "MCX" in exch_seg or "MCX" in name or itype in ("OPTIDX","OPTFO","FUTIDX","FUTCOM"):
            names.add(name)
    return sorted(names)

def get_all_expiries(master):
    exps = set()
    for r in master:
        raw = r.get("expiry") or r.get("expirydate") or r.get("expiryDate")
        d = try_parse_expiry_string(raw)
        if not d:
            sym = r.get("symbol") or ""
            if isinstance(sym, str):
                m = re.search(r'(\d{1,2}[A-Za-z]{3,5}\d{2,4})', sym)
                if m:
                    d = try_parse_expiry_string(m.group(1))
        if d:
            exps.add(d)
    return sorted(exps)

def get_underlyings_for_expiry(master, expiry_date):
    if not isinstance(expiry_date, date):
        expiry_date = try_parse_expiry_string(expiry_date)
    if expiry_date is None:
        return []
    names = set()
    for r in master:
        raw = r.get("expiry") or r.get("expirydate") or r.get("expiryDate")
        d = try_parse_expiry_string(raw)
        if d != expiry_date:
            if d is None:
                sym = r.get("symbol") or ""
                m = re.search(r'(\d{1,2}[A-Za-z]{3,5}\d{2,4})', str(sym))
                if m:
                    d = try_parse_expiry_string(m.group(1))
            if d != expiry_date:
                continue
        name = (r.get("name") or "").strip().upper()
        if name:
            names.add(name)
    return sorted(names)

def get_strikes_for_expiry_and_underlying(master, expiry_date, underlying):
    if not isinstance(expiry_date, date):
        expiry_date = try_parse_expiry_string(expiry_date)
    if expiry_date is None:
        return []
    target = (underlying or "").upper()
    strikes = set()
    for r in master:
        raw = r.get("expiry") or r.get("expirydate") or r.get("expiryDate")
        d = try_parse_expiry_string(raw)
        if d != expiry_date:
            if d is None:
                sym = r.get("symbol") or ""
                m = re.search(r'(\d{1,2}[A-Za-z]{3,5}\d{2,4})', str(sym))
                if m:
                    d = try_parse_expiry_string(m.group(1))
            if d != expiry_date:
                continue
        name = (r.get("name") or "").upper()
        if name != target:
            continue
        s = parse_strike(r.get("strike", 0))
        if s is not None:
            strikes.add(s)
    return sorted(strikes)

def find_instrument_row_by_expiry(master, expiry_date, underlying, strike, option_type):
    if not isinstance(expiry_date, date):
        expiry_date = try_parse_expiry_string(expiry_date)
    if expiry_date is None:
        return None
    target = (underlying or "").upper()
    for r in master:
        raw = r.get("expiry") or r.get("expirydate") or r.get("expiryDate")
        d = try_parse_expiry_string(raw)
        if d != expiry_date:
            if d is None:
                sym = r.get("symbol") or ""
                m = re.search(r'(\d{1,2}[A-Za-z]{3,5}\d{2,4})', str(sym))
                if m:
                    d = try_parse_expiry_string(m.group(1))
            if d != expiry_date:
                continue
        if (r.get("name") or "").upper() != target:
            continue
        s = parse_strike(r.get("strike", 0))
        if s is None or s != strike:
            continue
        sym = (r.get("symbol") or "").upper()
        if option_type:
            if sym.endswith(option_type) or f"-{option_type}" in sym or sym.split()[-1].endswith(option_type):
                return r
            else:
                continue
        return r
    return None

# ----------------- Ledger helpers -----------------
def load_ledger():
    fname = ledger_filename_for_user()
    if not os.path.exists(fname):
        return []
    try:
        with open(fname, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_ledger(trades):
    fname = ledger_filename_for_user()
    with open(fname, "w") as f:
        json.dump(trades, f, indent=2, default=str)

def append_trade(trade):
    fname = ledger_filename_for_user()
    trades = load_ledger()
    trades.append(trade)
    save_ledger(trades)

# ----------------- lightweight ledger -> df and P&L helpers -----------------
def ledger_to_df(trades):
    """Convert list of trades (dicts) into a normalized pandas DataFrame."""
    if not trades:
        return pd.DataFrame()
    df = pd.json_normalize(trades)
    # Ensure expected columns exist
    expected_cols = ['timestamp','symbol','token','exch_seg','underlying','expiry','strike','option_type',
                     'lotsize','contracts','qty','side','price','note']
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    # normalize types
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0).astype(int)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    df['side'] = df['side'].astype(str).str.upper().fillna('BUY')
    return df

def compute_positions_and_realized(df):
    """
    Compute per-symbol position summary and a basic FIFO realized P&L.
    Returns (positions_df, realized_trades_list)
    positions_df includes: symbol, netqty, avgcost (per unit), realized_for_symbol
    """
    if df is None or df.empty:
        return pd.DataFrame(), []

    # Sort by timestamp ascending for FIFO
    df_sorted = df.sort_values('timestamp').copy().reset_index(drop=True)
    realized_per_symbol = {}
    # maintain per-symbol list of open lots: list of dicts {'qty', 'price'}
    open_lots = {}

    for _, row in df_sorted.iterrows():
        sym = row['symbol']
        side = row['side']
        qty = int(row['qty'])
        price = float(row['price'])
        if sym not in open_lots:
            open_lots[sym] = []
        if sym not in realized_per_symbol:
            realized_per_symbol[sym] = 0.0

        # BUY adds lots, SELL consumes lots
        if side == 'BUY':
            open_lots[sym].append({'qty': qty, 'price': price})
        else:
            # SELL: consume lots FIFO, compute realized P&L = (sell_price - buy_price) * units
            remaining = qty
            while remaining > 0 and open_lots[sym]:
                lot = open_lots[sym][0]
                take = min(remaining, lot['qty'])
                pnl = (price - lot['price']) * take
                realized_per_symbol[sym] += pnl
                lot['qty'] -= take
                remaining -= take
                if lot['qty'] == 0:
                    open_lots[sym].pop(0)
            # if remaining > 0 (short-selling or no prior buys), treat as negative open lot (short)
            if remaining > 0:
                # represent short as negative lot with the sell price as 'price'
                open_lots[sym].insert(0, {'qty': -remaining, 'price': price})
                remaining = 0

    # Build positions summary
    pos_rows = []
    for sym, lots in open_lots.items():
        netqty = sum([lot['qty'] for lot in lots])
        # compute avgcost for open lots: weighted avg by positive qty only (ignore short negative lots for avgcost)
        positive_lots = [lot for lot in lots if lot['qty'] > 0]
        if positive_lots:
            denom = sum([lot['qty'] for lot in positive_lots])
            avgcost = sum([lot['qty'] * lot['price'] for lot in positive_lots]) / denom
        else:
            avgcost = None
        pos_rows.append({
            'symbol': sym,
            'netqty': int(netqty),
            'avgcost': float(avgcost) if avgcost is not None else None,
            'realized_for_symbol': round(realized_per_symbol.get(sym, 0.0), 2)
        })

    pos_df = pd.DataFrame(pos_rows)
    if pos_df.empty:
        pos_df = pd.DataFrame(columns=['symbol','netqty','avgcost','realized_for_symbol'])
    return pos_df, realized_per_symbol

# ----------------- SmartAPI LTP helpers -----------------
def fetch_ltp_for_ledger_rows(sc, ledger_df):
    if sc is None or ledger_df.empty:
        return [None] * len(ledger_df)
    ltps = []
    for i, row in ledger_df.iterrows():
        try:
            token = row.get("token")
            exch_seg = row.get("exch_seg") or "NFO"
            sym = row.get("symbol")
            resp = sc.ltpData(exch_seg, sym, token)
            ltp_val = None
            if isinstance(resp, dict) and resp.get("status") and resp.get("data"):
                d = resp.get("data")
                if isinstance(d, dict):
                    ltp_val = d.get("ltp")
                elif isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
                    ltp_val = d[0].get("ltp")
            ltps.append(ltp_val)
        except Exception:
            ltps.append(None)
    return ltps

def fetch_ltp_map_for_symbols(sc, df, symbols):
    ltp_map = {}
    if sc is None:
        for s in symbols:
            ltp_map[s] = None
        return ltp_map
    for sym in symbols:
        try:
            rows = df[df["symbol"] == sym]
            if rows.empty:
                ltp_map[sym] = None
                continue
            token = rows["token"].iloc[-1]
            exch_seg = rows.get("exch_seg").iloc[-1] if "exch_seg" in rows.columns else "NFO"
            try:
                resp = sc.ltpData(exch_seg or "NFO", rows["symbol"].iloc[-1], token)
            except Exception:
                resp = None
            ltp_val = None
            if isinstance(resp, dict) and resp.get("status") and resp.get("data"):
                d = resp.get("data")
                if isinstance(d, dict):
                    ltp_val = d.get("ltp")
                elif isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
                    ltp_val = d[0].get("ltp")
            ltp_map[sym] = ltp_val
        except Exception:
            ltp_map[sym] = None
    return ltp_map

# ----------------- Rerun helper -----------------
def rerun_app():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
            return
    except Exception:
        pass
    try:
        if hasattr(st, "experimental_rerun"):
            try:
                st.experimental_rerun()
                return
            except Exception:
                pass
    except Exception:
        pass
    st.info("Please refresh the page manually (your Streamlit version doesn't support programmatic rerun).")

# ---------------- UI: JSON loader (from Code 1) ----------------
st.markdown("Paste the `OpenAPIScripMaster.json` URL (or use the default) and click **Load JSON**. After loading, choose an expiry (parsed) to see matching instruments.")

default_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
url_input = st.text_input("Scrip master JSON URL (optional)", value=default_url, key="master_url_input")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Load JSON (override master)", key="load_json_btn"):
        try:
            with st.spinner("Fetching JSON..."):
                raw = fetch_json_data(url_input)
                # write to local MASTER_FILENAME to use everywhere
                with open(MASTER_FILENAME, "w") as f:
                    json.dump(raw, f)
                # clear cached master
                load_master.clear()
                st.success("Master JSON saved locally and loaded.")
                rerun_app()
        except requests.RequestException as e:
            st.error(f"Network / HTTP error: {e}")
        except Exception as ex:
            st.exception(f"Unexpected error while loading JSON: {ex}")

with col2:
    st.info(f"Local master: {MASTER_FILENAME} (if exists). You may override via URL.")

st.markdown("---")

# Load instrument master (from file or fallback)
master = load_master()
if not master:
    st.error("Instrument master not available. Place OpenAPIScripMaster.json locally or use 'Load JSON (override master)'.")
    st.stop()

# prepare available underlyings (for quick pick)
underlyings_all = available_underlyings(master)
if not underlyings_all:
    st.warning("No underlyings found in master (check scrip master).")

# ---------------- Sidebar: SmartAPI login (Code 2) ----------------
st.sidebar.header("Optional SmartAPI (read-only LTP)")
with st.sidebar.form("login_form", clear_on_submit=False):
    api_key = st.text_input("API Key", type="password", key="login_api_key")
    client_code = st.text_input("Client Code", key="login_client")
    password = st.text_input("Password / PIN", type="password", key="login_pin")
    totp = st.text_input("TOTP", key="login_totp")
    login_submit = st.form_submit_button("Login (read LTP)")

if login_submit:
    if SmartConnect is None:
        st.sidebar.error("smartapi-python not installed. Install to use SmartAPI LTP features.")
    else:
        try:
            sc = SmartConnect(api_key=api_key)
            profile = sc.generateSession(client_code, password, totp)
            st.session_state["sc"] = sc
            st.session_state["profile"] = profile
            st.sidebar.success("Logged in for LTP (read-only).")
            rerun_app()
        except Exception as e:
            st.sidebar.error(f"Login failed: {e}")

if st.sidebar.button("Logout (if logged)", key="sidebar_logout"):
    sc = st.session_state.get("sc")
    prof = st.session_state.get("profile")
    if sc and prof and isinstance(prof, dict):
        try:
            clientcode = prof.get("data", {}).get("clientcode")
            if clientcode:
                sc.terminateSession(clientcode)
        except:
            pass
    st.session_state.pop("sc", None)
    st.session_state.pop("profile", None)
    st.sidebar.success("Logged out.")
    rerun_app()

sc = st.session_state.get("sc", None)

# ---------------- Tabs: Live Market / Place Trade / Ledger ----------------
tabs = st.tabs(["Live Market", "Place Paper Trade", "Paper Positions & Ledger"])

# ---------- Live Market ----------
with tabs[0]:
    st.subheader("Live Market (LTP display - read-only)")
    cL, cR = st.columns([7,1])
    with cR:
        if st.button("Refresh LTP", key="lm_refresh"):
            rerun_app()

    all_exps = get_all_expiries(master)
    if not all_exps:
        st.info("No expiries found in master.")
    else:
        expiry = st.selectbox("Expiry (all)", all_exps, format_func=lambda d: d.strftime("%d-%b-%Y"), key="lm_expiry")
        underlyings_for_exp = get_underlyings_for_expiry(master, expiry)
        if not underlyings_for_exp:
            st.info("No underlyings found for selected expiry.")
        else:
            underlying = st.selectbox("Underlying (for selected expiry)", underlyings_for_exp, index=0, key="lm_underlying")
            strikes = get_strikes_for_expiry_and_underlying(master, expiry, underlying)
            if not strikes:
                st.info("No strikes found.")
            else:
                strike = st.selectbox("Strike", strikes, key="lm_strike")
                opt_type = st.radio("Option Type", ["CE", "PE"], horizontal=True, key="lm_opt")
                row = find_instrument_row_by_expiry(master, expiry, underlying, strike, opt_type)
                if not row:
                    st.warning("Instrument row not found for this selection.")
                else:
                    st.json({"symbol": row.get("symbol"), "token": row.get("token"), "lotsize": row.get("lotsize")})
                    # show LTP if logged
                    suggested_price = None
                    if sc:
                        try:
                            resp = sc.ltpData(row.get("exch_seg") or "NFO", row.get("symbol"), row.get("token"))
                            if isinstance(resp, dict) and resp.get("status") and resp.get("data"):
                                d = resp.get("data")
                                if isinstance(d, dict):
                                    suggested_price = d.get("ltp")
                                elif isinstance(d, list) and len(d) > 0:
                                    suggested_price = d[0].get("ltp")
                        except:
                            suggested_price = None
                    st.write("Suggested LTP (if available):", suggested_price if suggested_price is not None else "N/A")

# ---------- Place Paper Trade ----------
with tabs[1]:
    st.subheader("Place Paper Trade (simulate & save locally)")
    pcol1, pcol2 = st.columns([3,1])
    with pcol2:
        if st.button("Refresh (Place Trade tab)", key="pt_refresh"):
            rerun_app()

    all_exps = get_all_expiries(master)
    if not all_exps:
        st.info("No expiries found in master.")
    else:
        u_exp = st.selectbox("Expiry (all)", all_exps, format_func=lambda d: d.strftime("%d-%b-%Y"), key="pt_expiry")
        underlyings_for_exp = get_underlyings_for_expiry(master, u_exp)
        if not underlyings_for_exp:
            st.info("No underlyings found for this expiry.")
        else:
            u = st.selectbox("Underlying (for selected expiry)", underlyings_for_exp, index=0, key="pt_underlying")
            u_strikes = get_strikes_for_expiry_and_underlying(master, u_exp, u)
            if not u_strikes:
                st.info("No strikes found for this expiry + underlying.")
            else:
                u_strike = st.selectbox("Strike", u_strikes, key="pt_strike")
                u_opt = st.radio("Option Type", ["CE", "PE"], horizontal=True, key="pt_opt")

                selected_row = find_instrument_row_by_expiry(master, u_exp, u, u_strike, u_opt)
                if not selected_row:
                    st.error("Instrument not found in master — cannot place.")
                else:
                    lotsize = int(selected_row.get("lotsize", 1))
                    st.markdown(f"**Lot size:** 1 contract = **{lotsize}** units")

                    contracts = st.number_input(f"Qty (contracts) — 1 contract = {lotsize} units", min_value=1, value=1, step=1, key="pt_qty")
                    side = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="pt_side")

                    # Suggested price if logged
                    suggested_price = None
                    if sc:
                        try:
                            resp = sc.ltpData(selected_row.get("exch_seg") or "NFO", selected_row.get("symbol"), selected_row.get("token"))
                            if isinstance(resp, dict) and resp.get("status") and resp.get("data"):
                                d = resp.get("data")
                                if isinstance(d, dict):
                                    suggested_price = d.get("ltp")
                                elif isinstance(d, list) and len(d) > 0:
                                    suggested_price = d[0].get("ltp")
                        except:
                            suggested_price = None

                                        # Price and SL / Target inputs
                    price_source = st.radio("Price source", ["Market (LTP)", "Manual"], horizontal=True, key="pt_price_source")
                    default_manual = float(suggested_price) if suggested_price is not None else 100.0
                    manual_disabled = (price_source != "Manual")
                    manual_price = st.number_input("Manual Price (fallback)", value=default_manual, key="pt_manual_price", format="%.2f", disabled=manual_disabled)

                    # NEW: Stop Loss and Target (per unit)
                    st.markdown("**Optional exit conditions**")
                    stop_loss = st.number_input("Stop Loss (per unit) — leave 0 for none", value=0.0, key="pt_stop_loss", format="%.2f")
                    target = st.number_input("Target (per unit) — leave 0 for none", value=0.0, key="pt_target", format="%.2f")

                    if st.button("Place Paper Trade (Simulate)", key="pt_place"):
                        if price_source == "Market (LTP)":
                            if suggested_price is None:
                                st.error("Market price not available (not logged in). Choose Manual or login.")
                                st.stop()
                            exec_price = float(suggested_price)
                        else:
                            exec_price = float(manual_price)

                        contracts_int = int(contracts)
                        executed_qty = contracts_int * lotsize

                        # include stop_loss and target in saved trade metadata if provided (0 -> None)
                        trade = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "symbol": selected_row.get("symbol"),
                            "token": selected_row.get("token"),
                            "exch_seg": selected_row.get("exch_seg"),
                            "underlying": u,
                            "expiry": str(u_exp),
                            "strike": u_strike,
                            "option_type": u_opt,
                            "lotsize": lotsize,
                            "contracts": contracts_int,
                            "qty": int(executed_qty),
                            "side": side,
                            "price": float(exec_price),
                            # NEW: store SL/TGT as numeric (or None)
                            "stop_loss": (float(stop_loss) if stop_loss and float(stop_loss) > 0 else None),
                            "target": (float(target) if target and float(target) > 0 else None),
                            "note": "simulated"
                        }
                        append_trade(trade)
                        st.success(f"Simulated trade saved: {side} {trade['symbol']} contracts {contracts_int} ({trade['qty']} units) @ {exec_price}")


# ---------- Paper Positions & Ledger ----------
with tabs[2]:
    st.subheader("Paper Positions & Ledger")

    # Backup & clear helper
    def backup_and_clear_ledger(make_backup=True):
        try:
            ledger_filename = ledger_filename_for_user()
            if not os.path.exists(ledger_filename):
                return (None, True, "Ledger file does not exist — nothing to clear.")
            backup_path = None
            if make_backup:
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                backup_path = f"{ledger_filename}.bak.{ts}"
                try:
                    with open(ledger_filename, "r") as src, open(backup_path, "w") as dst:
                        dst.write(src.read())
                except Exception as e:
                    return (None, False, f"Backup failed: {e}")
            with open(ledger_filename, "w") as f:
                json.dump([], f)
            return (backup_path, True, "Ledger cleared successfully.")
        except Exception as e:
            return (None, False, f"Failed to clear ledger: {e}")

    colA, colB = st.columns([8,1])
    with colB:
        if st.button("Refresh Ledger", key="ledger_refresh_btn"):
            rerun_app()
        if st.button("Clear Ledger", key="clear_ledger_btn"):
            st.session_state["confirm_clear_ledger"] = True
            rerun_app()

    if st.session_state.get("confirm_clear_ledger"):
        st.warning("You are about to permanently clear ALL simulated trades from the ledger. A timestamped backup will be created before clearing.")
        confirm_col, cancel_col = st.columns([1,1])
        with confirm_col:
            if st.button("Confirm Clear Ledger", key="confirm_clear_btn"):
                backup_path, ok, msg = backup_and_clear_ledger()
                if ok:
                    if backup_path:
                        st.success(f"Ledger cleared. Backup created: {backup_path}")
                    else:
                        st.success("Ledger cleared.")
                    for k in ["last_trade_saved", "confirm_clear_ledger"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    rerun_app()
                else:
                    st.error(f"Could not clear ledger: {msg}")
        with cancel_col:
            if st.button("Cancel Clear", key="cancel_clear_btn"):
                if "confirm_clear_ledger" in st.session_state:
                    del st.session_state["confirm_clear_ledger"]
                st.info("Clear ledger cancelled.")
                rerun_app()

    trades = load_ledger()
    if not trades:
        st.info("No simulated trades. Place a paper trade first.")
    else:
        # normalized ledger dataframe
        df = ledger_to_df(trades)
        if df.empty:
            st.error("Unable to parse ledger into table.")
        else:
            # *** NEW: run auto-exit checks (uses SmartAPI sc and ledger dataframe) ***
            # Call this BEFORE computing positions/display so that if a SL/TGT was hit,
            # an exit trade will be appended and the app will rerun to show updated ledger.
            try:
                auto_exit_by_sl_target(sc, df)
            except Exception:
                # swallow errors from auto-exit to avoid breaking the ledger view
                pass

            # compute positions and unrealized using existing helpers
            pos_df, _ = compute_positions_and_realized(df)
            unreal_total = 0.0

            if sc is not None and not pos_df.empty:
                symbols = pos_df['symbol'].tolist()
                ltp_map = fetch_ltp_map_for_symbols(sc, df, symbols)
                unreal_list = []
                ltp_col = []
                for _, r in pos_df.iterrows():
                    sym = r['symbol']
                    net = r['netqty']
                    avg = r['avgcost']
                    ltp_val = ltp_map.get(sym)
                    ltp_col.append(ltp_val)
                    if ltp_val is not None and avg is not None and net != 0:
                        pnl = (ltp_val - avg) * net
                    else:
                        pnl = 0.0
                    unreal_list.append(round(pnl, 2))
                pos_df['current_ltp'] = ltp_col
                pos_df['unrealized_pnl'] = unreal_list
                unreal_total = sum(unreal_list)
            else:
                pos_df['current_ltp'] = None
                pos_df['unrealized_pnl'] = 0.0
                unreal_total = 0.0

            # formatted helpers
            def fmt(x):
                try:
                    return f"{float(x):,.2f}"
                except Exception:
                    return "0.00"
            def colored_metric(col, label, amount):
                s = fmt(amount)
                color = "#138000" if float(amount) > 0 else ("#b00000" if float(amount) < 0 else "#333333")
                col.markdown(f"**{label}**<br><span style='color:{color};font-size:22px;font-weight:600'>{s}</span>", unsafe_allow_html=True)

            open_count = 0 if pos_df.empty else int((pos_df['netqty'] != 0).sum())

            st.markdown("#### Trade Ledger (most recent first)")
            ledger_df = pd.DataFrame(trades)[::-1].reset_index(drop=True)
            if ledger_df.empty:
                st.info("Ledger is empty.")
            else:
                ledger_df = ledger_df.copy()
                ledger_df['timestamp'] = pd.to_datetime(ledger_df['timestamp'], errors='coerce')
                ledger_df['trade_date'] = ledger_df['timestamp'].dt.date
                current_prices = fetch_ltp_for_ledger_rows(sc, ledger_df) if sc else [None] * len(ledger_df)
                ledger_df['current_ltp'] = [ (float(x) if (x is not None and pd.notna(x)) else None) for x in current_prices ]
                ledger_df['side'] = ledger_df['side'].astype(str).str.upper().fillna('BUY')
                ledger_df['qty'] = pd.to_numeric(ledger_df.get('qty', 0), errors='coerce').fillna(0).astype(int)
                ledger_df['contracts'] = pd.to_numeric(ledger_df.get('contracts', 0), errors='coerce').fillna(0).astype(int)
                ledger_df['price'] = pd.to_numeric(ledger_df.get('price', 0.0), errors='coerce').fillna(0.0)

                view = st.radio("Ledger view", ["Detailed", "Aggregated (per trade date & symbol)"], horizontal=True, index=1, key="ledger_view")
                if view == "Detailed":
                    detailed = ledger_df.copy()
                    detailed['current_ltp'] = detailed['current_ltp'].map(lambda x: f"{x:,.2f}" if (x is not None and pd.notna(x)) else "N/A")
                    if 'price' in detailed.columns:
                        detailed['price'] = detailed['price'].map(lambda x: f"{x:,.2f}")
                    display_cols = ['timestamp','trade_date','symbol','token','underlying','expiry','strike','option_type','lotsize','contracts','qty','side','price','current_ltp','note']
                    available_cols = [c for c in display_cols if c in detailed.columns]
                    st.dataframe(detailed[available_cols], use_container_width=True)
                else:
                    df_ag = ledger_df.copy()
                    def weighted_avg_price(sub):
                        denom = sub['qty'].sum()
                        if denom == 0:
                            return None
                        return (sub['qty'] * sub['price']).sum() / denom

                    agg_rows = []
                    for (td, sym), g in df_ag.groupby(['trade_date', 'symbol'], sort=True):
                        total_bought_qty = int(g[g['side'] == 'BUY']['qty'].sum())
                        total_bought_contracts = int(g[g['side'] == 'BUY']['contracts'].sum())
                        total_sold_qty = int(g[g['side'] == 'SELL']['qty'].sum())
                        total_sold_contracts = int(g[g['side'] == 'SELL']['contracts'].sum())
                        net_units = int(total_bought_qty - total_sold_qty)
                        net_contracts = int(total_bought_contracts - total_sold_contracts)
                        avg_buy = weighted_avg_price(g[g['side'] == 'BUY'])
                        avg_sell = weighted_avg_price(g[g['side'] == 'SELL'])
                        last = g.iloc[-1]
                        last_trade_price = float(last['price']) if 'price' in last and pd.notna(last['price']) else None
                        last_ltp_num = None
                        try:
                            if last.get('current_ltp') is not None and str(last.get('current_ltp')).strip() != "":
                                last_ltp_num = float(last.get('current_ltp'))
                        except Exception:
                            last_ltp_num = None
                        if avg_buy is None or avg_sell is None or total_sold_qty == 0:
                            net_pnl_val = 0.0
                        else:
                            net_pnl_val = (avg_sell - avg_buy) * total_sold_qty
                        realized_pnl = round(net_pnl_val, 2)
                        unrealized_pnl = 0.0
                        total_net_pnl = round(realized_pnl + unrealized_pnl, 2)
                        agg_rows.append({
                            'trade_date': td,
                            'symbol': sym,
                            'underlying': last.get('underlying'),
                            'expiry': last.get('expiry'),
                            'strike': last.get('strike'),
                            'realized_pnl': realized_pnl,
                            'unrealized_pnl': round(unrealized_pnl, 2),
                            'net_pnl': total_net_pnl,
                            'option_type': last.get('option_type'),
                            'lotsize': int(last.get('lotsize', 1)) if pd.notna(last.get('lotsize')) else 1,
                            'total_bought_contracts': int(total_bought_contracts),
                            'total_bought_units': int(total_bought_qty),
                            'total_sold_contracts': int(total_sold_contracts),
                            'total_sold_units': int(total_sold_qty),
                            'net_contracts': int(net_contracts),
                            'net_units': int(net_units),
                            'avg_buy_price': (avg_buy if avg_buy is not None else None),
                            'avg_sell_price': (avg_sell if avg_sell is not None else None),
                            'last_ltp': (f"{last_ltp_num:,.2f}" if last_ltp_num is not None else ("N/A" if last_trade_price is None else f"{last_trade_price:,.2f}"))
                        })

                    agg_df = pd.DataFrame(agg_rows)
                    if 'realized_pnl' in agg_df.columns:
                        agg_df['realized_pnl'] = agg_df['realized_pnl'].astype(float).round(2)
                    else:
                        agg_df['realized_pnl'] = 0.0
                    if 'unrealized_pnl' in agg_df.columns:
                        agg_df['unrealized_pnl'] = agg_df['unrealized_pnl'].astype(float).round(2)
                    else:
                        agg_df['unrealized_pnl'] = 0.0
                    agg_df['net_pnl'] = (agg_df['realized_pnl'] + agg_df['unrealized_pnl']).astype(float).round(2)
                    agg_df['net_pnl_display'] = agg_df['net_pnl'].map(lambda x: f"{x:,.2f}")
                    agg_df['avg_buy_price_display']  = agg_df.get('avg_buy_price').map(lambda x: f"{x:,.2f}" if pd.notna(x) and x is not None else "N/A")
                    agg_df['avg_sell_price_display'] = agg_df.get('avg_sell_price').map(lambda x: f"{x:,.2f}" if pd.notna(x) and x is not None else "N/A")
                    realized_total = round(float(agg_df['net_pnl'].sum() if not agg_df.empty else 0.0), 2)

                    if agg_df.empty:
                        st.info("No data to aggregate.")
                    else:
                        for col in ['total_bought_units','total_sold_units','net_units','lotsize','total_bought_contracts','total_sold_contracts','net_contracts']:
                            if col in agg_df.columns:
                                agg_df[col] = agg_df[col].astype(int)
                        display_cols_agg = ['trade_date','symbol','underlying','expiry','strike','avg_buy_price_display','avg_sell_price_display','net_pnl_display','option_type','lotsize','total_bought_contracts','total_bought_units','total_sold_contracts','total_sold_units','net_contracts','net_units','last_ltp']
                        available_cols = [c for c in display_cols_agg if c in agg_df.columns]
                        display_df = agg_df[available_cols].copy()

                        def row_style_by_pnl(row):
                            try:
                                pnl = float(agg_df.loc[row.name, 'net_pnl'])
                            except Exception:
                                pnl = 0.0
                            if pnl > 0:
                                return ['background-color: #e6ffed'] * len(row)
                            elif pnl < 0:
                                return ['background-color: #ffecec'] * len(row)
                            else:
                                return [''] * len(row)

                        def emphasize_net_pnl_using_index(row):
                            try:
                                pnl = float(agg_df.loc[row.name, 'net_pnl'])
                            except Exception:
                                pnl = 0.0
                            style = []
                            for col in display_df.columns:
                                if col == 'net_pnl_display':
                                    color = "#138000" if pnl > 0 else ("#b00000" if pnl < 0 else "#333333")
                                    bg = "#e6ffed" if pnl > 0 else ("#ffecec" if pnl < 0 else "white")
                                    style.append(f"background-color:{bg}; color:{color}; font-weight:900; font-size:18px;")
                                else:
                                    style.append("")
                            return style

                        styler = display_df.style.apply(row_style_by_pnl, axis=1)
                        styler = styler.apply(emphasize_net_pnl_using_index, axis=1)
                        styler = styler.set_table_styles([
                            {'selector': 'th', 'props': [('text-align', 'left'),('padding', '10px'),('font-size', '16px'),('background-color', '#f5f7fa'),('font-weight', '600')]},
                            {'selector': 'td', 'props': [('padding', '10px'),('white-space', 'nowrap'),('overflow', 'hidden'),('text-overflow', 'ellipsis'),('font-size', '16px')]}

                        ])
                        import streamlit.components.v1 as components
                        try:
                            html = styler.to_html()
                        except Exception:
                            html = display_df.to_html(classes="dataframe", index=False, escape=False)
                        height_px = min(900, 220 + max(1, len(display_df)) * 30)
                        components.html(html, height=height_px, scrolling=True)

                mcol1, mcol2, mcol3 = st.columns([1,1,1])
                mcol1.metric("Open Positions", value=open_count)
                colored_metric(mcol2, "Realized P&L (paper)", realized_total)
                colored_metric(mcol3, "Unrealized P&L (paper)", unreal_total)

                open_pos_df = pos_df[pos_df['netqty'] != 0].copy()
                def _get_lotsize_for_symbol(sym):
                    try:
                        rows = df[df['symbol'] == sym]
                        if not rows.empty and 'lotsize' in rows.columns:
                            return int(rows['lotsize'].iloc[-1])
                    except Exception:
                        pass
                    return 1

                if not open_pos_df.empty:
                    open_pos_df['lotsize'] = open_pos_df['symbol'].map(lambda s: _get_lotsize_for_symbol(s))
                    open_pos_df['open_contracts'] = open_pos_df.apply(lambda r: int(abs(r['netqty']) // r['lotsize']) if r['lotsize'] and r['netqty']!=0 else 0, axis=1)

                st.markdown("#### Open Positions")
                if open_pos_df.empty:
                    st.info("No open positions.")
                else:
                    pos_df_display = open_pos_df.copy()
                    for c in ['avgcost', 'unrealized_pnl', 'realized_for_symbol', 'current_ltp']:
                        if c in pos_df_display.columns:
                            if c == 'current_ltp':
                                pos_df_display[c] = pos_df_display[c].map(lambda x: f"{float(x):,.2f}" if (x is not None and pd.notna(x)) else "N/A")
                            else:
                                pos_df_display[c] = pos_df_display[c].map(lambda x: f"{x:,.2f}")
                    display_cols_pos = [c for c in ['symbol','netqty','open_contracts','lotsize','avgcost','realized_for_symbol','current_ltp','unrealized_pnl'] if c in pos_df_display.columns]
                    st.dataframe(pos_df_display[display_cols_pos], use_container_width=True)

                    # exit flow
                    open_syms = open_pos_df['symbol'].tolist()
                    if open_syms:
                        pick = st.selectbox("Select open position to exit (paper)", open_syms, key="exit_pick")
                        # show small info about selected position
                        if pick:
                            # derive token, lotsize and exch_seg from most recent ledger row for this symbol
                            rows_for_sym = df[df['symbol'] == pick]
                            if rows_for_sym.empty:
                                st.error("No ledger rows found for selected symbol.")
                            else:
                                token = rows_for_sym['token'].iloc[-1]
                                symbol_lotsize = int(rows_for_sym['lotsize'].iloc[-1]) if 'lotsize' in rows_for_sym.columns and pd.notna(rows_for_sym['lotsize'].iloc[-1]) else 1
                                exch_seg_for_sym = rows_for_sym['exch_seg'].iloc[-1] if 'exch_seg' in rows_for_sym.columns else "NFO"

                                # Try fetching LTP if logged in
                                current_ltp = None
                                if sc:
                                    try:
                                        resp = sc.ltpData(exch_seg_for_sym, pick, token)
                                        if isinstance(resp, dict) and resp.get("status") and resp.get("data"):
                                            d = resp.get("data")
                                            if isinstance(d, dict):
                                                current_ltp = d.get("ltp")
                                            elif isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
                                                current_ltp = d[0].get("ltp")
                                    except Exception:
                                        current_ltp = None

                                # show current LTP (if available)
                                st.markdown(f"**Current LTP:** {current_ltp if current_ltp is not None else 'N/A'}")
                                # Price source selection
                                price_src = st.radio("Exit price source", ["Market (LTP)", "Manual"], horizontal=True, key="exit_price_source")

                                # manual price input (defaults to LTP if available)
                                default_exit_manual = float(current_ltp) if current_ltp is not None else 0.0
                                manual_exit_price = st.number_input(
                                    "Manual Exit Price (per unit)",
                                    min_value=0.0,
                                    value=default_exit_manual,
                                    format="%.2f",
                                    key=f"manual_exit_price_{pick}"
                                )

                                # Confirm exit button
                                if st.button("Exit Selected (simulate)", key=f"exit_button_{pick}"):
                                    # choose exec price depending on selection
                                    if price_src == "Market (LTP)":
                                        if current_ltp is None:
                                            st.error("Market LTP not available — either login for LTP or use Manual price.")
                                            st.stop()
                                        exec_price = float(current_ltp)
                                    else:
                                        exec_price = float(manual_exit_price)
                                        if exec_price <= 0:
                                            st.error("Enter a valid manual exit price (> 0).")
                                            st.stop()

                                    # compute net qty and units to exit
                                    netqty = int(pos_df[pos_df['symbol'] == pick]['netqty'].iloc[0])
                                    abs_netqty = abs(netqty)

                                    # compute contracts to exit and remainder handling (same logic as before)
                                    if symbol_lotsize > 1:
                                        contracts_to_exit = abs_netqty // symbol_lotsize
                                        remainder = abs_netqty % symbol_lotsize
                                        if contracts_to_exit == 0:
                                            qty_exit = abs_netqty
                                            contracts_exit = 0
                                            st.info(f"Position {abs_netqty} units is less than one contract (lotsize {symbol_lotsize}). Exiting {qty_exit} units.")
                                        else:
                                            qty_exit = contracts_to_exit * symbol_lotsize
                                            contracts_exit = int(contracts_to_exit)
                                            if remainder != 0:
                                                st.warning(f"Open position has remainder {remainder} units after exiting {contracts_exit} full contracts ({qty_exit} units). Remainder will remain open.")
                                    else:
                                        # unit-based lotsize
                                        qty_exit = abs_netqty
                                        contracts_exit = int(qty_exit)

                                    side_exit = "SELL" if netqty > 0 else "BUY"

                                    # construct exit trade and append
                                    exit_trade = {
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "symbol": pick,
                                        "token": token,
                                        "exch_seg": exch_seg_for_sym if exch_seg_for_sym is not None else None,
                                        "underlying": rows_for_sym['underlying'].iloc[-1] if 'underlying' in rows_for_sym.columns else "",
                                        "expiry": rows_for_sym['expiry'].iloc[-1] if 'expiry' in rows_for_sym.columns else "",
                                        "strike": rows_for_sym['strike'].iloc[-1] if 'strike' in rows_for_sym.columns else 0,
                                        "option_type": rows_for_sym['option_type'].iloc[-1] if 'option_type' in rows_for_sym.columns else "",
                                        "lotsize": int(symbol_lotsize),
                                        "contracts": int(contracts_exit),
                                        "qty": int(qty_exit),
                                        "side": side_exit,
                                        "price": float(exec_price),
                                        "note": "simulated_exit"
                                    }
                                    append_trade(exit_trade)
                                    st.success(f"Simulated exit {side_exit} {exit_trade['qty']} units ({exit_trade['contracts']} contracts) @ {exec_price} for {pick}")
                                    rerun_app()

    st.caption("Paper-only: no real orders are sent. All trades are simulated and saved locally per user ledger file.")


# End of merged app
