# app.py
# Cap Table & Dilution Simulator (Streamlit, single-file full stack)
# Run: streamlit run app.py

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# ---------- Engine -----------
# -----------------------------

@dataclass
class Stake:
    """Represents a stakeholder's current shares."""
    name: str
    shares: float = 0.0

@dataclass
class RoundConfig:
    name: str
    pre_money_valuation: float        # dollars
    investment_amount: float          # dollars
    target_esop_pct: float            # e.g., 0.10 for 10%
    pool_topup_timing: str            # "Pre-money" or "Post-money"
    new_investor_name: str = "Investor"

def _issue_option_topup_pre_money(stakes: Dict[str, Stake], target_esop_pct: float) -> Tuple[float, float]:
    """
    Top up ESOP BEFORE the priced investment so that:
      ESOP / (Total) = target_esop_pct  (pre-money, pre-investor)
    Δ = (X*T - ESOP) / (1 - X)
    """
    total = sum(s.shares for s in stakes.values())
    current_esop = stakes.get("ESOP", Stake("ESOP", 0.0)).shares
    X = target_esop_pct
    delta = max(0.0, (X * total - current_esop) / (1 - X)) if X < 0.999999 else 0.0
    if delta > 0:
        stakes["ESOP"] = Stake("ESOP", current_esop + delta)
    return delta, total + delta

def _issue_option_topup_post_money(stakes: Dict[str, Stake], target_esop_pct: float) -> Tuple[float, float]:
    """
    Top up ESOP AFTER the investor shares are issued so that:
      ESOP / (Total) = target_esop_pct  (post-money)
    Δ = (X*T - ESOP) / (1 - X)
    """
    total = sum(s.shares for s in stakes.values())
    current_esop = stakes.get("ESOP", Stake("ESOP", 0.0)).shares
    X = target_esop_pct
    delta = max(0.0, (X * total - current_esop) / (1 - X)) if X < 0.999999 else 0.0
    if delta > 0:
        stakes["ESOP"] = Stake("ESOP", current_esop + delta)
    return delta, total + delta

def _price_per_share(pre_money_valuation: float, total_pre_round_shares: float) -> float:
    if total_pre_round_shares <= 0:
        return 0.0
    return pre_money_valuation / total_pre_round_shares

def _issue_investor_shares(stakes: Dict[str, Stake],
                           investor_name: str,
                           pre_money_valuation: float,
                           investment_amount: float) -> Tuple[float, float]:
    """
    Issues new investor shares at price derived from pre-money valuation.
    """
    total_pre = sum(s.shares for s in stakes.values())
    pps = _price_per_share(pre_money_valuation, total_pre)
    new_shares = 0.0 if pps <= 0 else investment_amount / pps
    if investor_name not in stakes:
        stakes[investor_name] = Stake(investor_name, 0.0)
    stakes[investor_name].shares += new_shares
    return new_shares, total_pre + new_shares

def run_priced_round(stakes: Dict[str, Stake], rc: RoundConfig) -> Dict[str, float]:
    """
    Mutates stakes in-place to reflect a priced round with optional ESOP top-up
    either pre- or post-money. Returns a snapshot dict {name: percent}.
    """
    # 1) Optional pre-money option top-up
    if rc.pool_topup_timing == "Pre-money" and rc.target_esop_pct > 0:
        _issue_option_topup_pre_money(stakes, rc.target_esop_pct)

    # 2) Investor shares
    _issue_investor_shares(stakes, rc.new_investor_name, rc.pre_money_valuation, rc.investment_amount)

    # 3) Optional post-money option top-up
    if rc.pool_topup_timing == "Post-money" and rc.target_esop_pct > 0:
        _issue_option_topup_post_money(stakes, rc.target_esop_pct)

    # Snapshot
    total = sum(s.shares for s in stakes.values())
    snapshot = {k: (v.shares / total if total > 0 else 0.0) for k, v in stakes.items()}
    return snapshot

def initialize_cap_table(founders: List[Tuple[str, float]], initial_esop_pct: float, base_shares: float = 10_000_000.0) -> Dict[str, Stake]:
    """
    Create initial stakeholders with a base number of fully diluted shares.
    Founders' input is % split (sums to 100% minus ESOP%). ESOP is a % of total base.
    """
    stakes: Dict[str, Stake] = {}
    esop_shares = base_shares * initial_esop_pct
    non_esop_shares = base_shares - esop_shares

    # Normalize founder pct inputs to sum to 1.0 of non-ESOP bucket
    total_pct = sum(p for _, p in founders) / 100.0 if founders else 1.0
    if total_pct <= 0:
        # single founder default
        stakes["Founder 1"] = Stake("Founder 1", non_esop_shares)
    else:
        for name, pct in founders:
            shares = non_esop_shares * (pct / 100.0) / total_pct
            stakes[name] = Stake(name, shares)

    if esop_shares > 0:
        stakes["ESOP"] = Stake("ESOP", esop_shares)

    return stakes

def stakes_to_dataframe(stakes: Dict[str, Stake]) -> pd.DataFrame:
    total = sum(s.shares for s in stakes.values())
    rows = []
    for s in stakes.values():
        rows.append({"Stakeholder": s.name, "Shares": s.shares, "Ownership %": (s.shares / total * 100.0) if total > 0 else 0.0})
    df = pd.DataFrame(rows).sort_values("Ownership %", ascending=False).reset_index(drop=True)
    return df

def history_to_long_df(history: List[Tuple[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Convert snapshots like [(label, {owner: pct,...}), ...] → long DF for plotting.
    """
    rows = []
    for label, snap in history:
        for owner, frac in snap.items():
            rows.append({"Round": label, "Stakeholder": owner, "Ownership %": frac * 100.0})
    return pd.DataFrame(rows)

# -----------------------------
# --------- Streamlit ---------
# -----------------------------

st.set_page_config(page_title="Cap Table & Dilution Simulator", page_icon="📈", layout="wide")

st.title("📈 Cap Table & Dilution Simulator")
st.caption("Model dilution across multiple priced rounds, with pre- or post-money option pool top-ups. 100% client-side logic.")

with st.sidebar:
    st.header("⚙️ Setup")

    # Founders
    st.subheader("Founders")
    n_founders = st.number_input("Number of founders", min_value=1, max_value=10, value=2, step=1)
    founders: List[Tuple[str, float]] = []
    starting_pct_left = 100.0
    default_pct = round(100.0 / n_founders, 2)
    for i in range(int(n_founders)):
        col1, col2 = st.columns([2, 1])
        with col1:
            name = st.text_input(f"Founder {i+1} name", value=f"Founder {i+1}", key=f"f_name_{i}")
        with col2:
            pct = st.number_input(f"% for {name}", min_value=0.0, max_value=100.0, value=default_pct, step=0.1, key=f"f_pct_{i}")
        founders.append((name, pct))

    initial_esop_pct = st.number_input("Initial ESOP % of fully diluted (pre-seed)", min_value=0.0, max_value=50.0, value=10.0, step=0.5) / 100.0
    base_shares = st.number_input("Base fully diluted shares (for precision)", min_value=100_000.0, max_value=1_000_000_000.0, value=10_000_000.0, step=100_000.0)

    st.divider()
    st.subheader("Rounds")
    n_rounds = st.number_input("How many priced rounds to simulate?", min_value=0, max_value=10, value=2, step=1)

    default_names = ["Seed", "Series A", "Series B", "Series C"]
    round_configs: List[RoundConfig] = []

    for i in range(int(n_rounds)):
        with st.expander(f"Round {i+1} config", expanded=(i == 0)):
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                rname = st.text_input("Round name", value=default_names[i] if i < len(default_names) else f"Round {i+1}", key=f"r_name_{i}")
                pre_money = st.number_input("Pre-money valuation ($)", min_value=0.0, value=10_000_000.0, step=500_000.0, key=f"r_pre_{i}")
                invest = st.number_input("New investment amount ($)", min_value=0.0, value=2_000_000.0, step=250_000.0, key=f"r_invest_{i}")
            with rcol2:
                target_pool_pct = st.number_input("Target ESOP % (for this round)", min_value=0.0, max_value=50.0, value=10.0, step=0.5, key=f"r_pool_{i}") / 100.0
                pool_timing = st.radio("Option pool top-up timing", ["Pre-money", "Post-money", "No top-up"], horizontal=True, key=f"r_timing_{i}")
                inv_name = st.text_input("New investor label", value=f"Investor {i+1}", key=f"r_invname_{i}")

            if pool_timing == "No top-up":
                target_pool_pct = 0.0

            round_configs.append(
                RoundConfig(
                    name=rname,
                    pre_money_valuation=pre_money,
                    investment_amount=invest,
                    target_esop_pct=target_pool_pct,
                    pool_topup_timing=pool_timing,
                    new_investor_name=inv_name
                )
            )

    st.divider()
    st.subheader("📦 Export / Import")
    if "saved_payload" not in st.session_state:
        st.session_state["saved_payload"] = None

# Build initial cap table
stakes = initialize_cap_table(founders, initial_esop_pct, base_shares)

# Record history
history: List[Tuple[str, Dict[str, float]]] = []
initial_snapshot = {k: v.shares / sum(s.shares for s in stakes.values()) for k, v in stakes.items()}
history.append(("Initial", initial_snapshot))

# Simulate rounds
for rc in round_configs:
    snap = run_priced_round(stakes, rc)
    history.append((rc.name, snap))

# DataFrames
df_current = stakes_to_dataframe(stakes)
df_hist = history_to_long_df(history)

# -----------------------------
# --------- Display -----------
# -----------------------------
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Current Cap Table")
    st.dataframe(df_current, use_container_width=True)

    st.subheader("Ownership Over Rounds")
    if not df_hist.empty:
        fig = px.area(
            df_hist,
            x="Round",
            y="Ownership %",
            color="Stakeholder",
            groupnorm=None,
            line_group="Stakeholder",
            markers=False
        )
        fig.update_layout(yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Current Ownership (Pie)")
    if not df_current.empty:
        fig2 = px.pie(df_current, names="Stakeholder", values="Ownership %", hole=0.45)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Key Metrics")
    total_shares = df_current["Shares"].sum()
    founder_pct = df_current[df_current["Stakeholder"].str.startswith("Founder")]["Ownership %"].sum()
    esop_pct = df_current.loc[df_current["Stakeholder"] == "ESOP", "Ownership %"].sum() if "ESOP" in df_current["Stakeholder"].values else 0.0
    last_round = round_configs[-1].name if round_configs else "Initial"
    st.metric("Total Fully Diluted Shares", f"{total_shares:,.0f}")
    st.metric("Founders (aggregate)", f"{founder_pct:.2f}%")
    st.metric("ESOP", f"{esop_pct:.2f}%")
    st.caption(f"Latest round: **{last_round}**")

st.divider()

# -----------------------------
# -------- Export / Import ----
# -----------------------------
st.subheader("Save / Share Scenarios")

# Wide history table (rounds x stakeholders)
def history_to_wide_table(history: List[Tuple[str, Dict[str, float]]]) -> pd.DataFrame:
    rounds = [h[0] for h in history]
    owners = sorted({o for _, snap in history for o in snap.keys()})
    data = []
    for label, snap in history:
        row = {"Round": label}
        for o in owners:
            row[o] = snap.get(o, 0.0) * 100.0
        data.append(row)
    return pd.DataFrame(data)

df_wide = history_to_wide_table(history)
st.dataframe(df_wide.style.format(precision=2), use_container_width=True)

# Download buttons
csv_bytes = df_wide.to_csv(index=False).encode("utf-8")
st.download_button("Download ownership by round (CSV)", data=csv_bytes, file_name="cap_table_ownership_by_round.csv", mime="text/csv")

# Raw JSON scenario payload (so users can re-load)
scenario_payload = {
    "founders": founders,
    "initial_esop_pct": initial_esop_pct,
    "base_shares": base_shares,
    "rounds": [asdict(rc) for rc in round_configs]
}
payload_bytes = json.dumps(scenario_payload, indent=2).encode("utf-8")
st.download_button("Download scenario (JSON)", data=payload_bytes, file_name="cap_table_scenario.json", mime="application/json")

st.markdown("#### Load scenario (JSON)")
uploaded = st.file_uploader("Upload a scenario.json exported from this app", type=["json"])
if uploaded is not None:
    try:
        d = json.load(uploaded)
        st.session_state["saved_payload"] = d
        st.success("Scenario loaded into memory. Click below to apply.")
        if st.button("Apply loaded scenario"):
            # Rebuild UI state from uploaded scenario
            try:
                # Founders & ESOP/base
                fnds = d.get("founders", [])
                st.session_state["f_name_0"] = fnds[0][0] if len(fnds) > 0 else "Founder 1"
                # We can't programmatically change number inputs directly in Streamlit reliably;
                # provide instructions and echo the loaded payload so the user can match fields.
                st.info("Loaded. Please match the sidebar inputs to the payload echoed below (Streamlit restricts programmatic widget updates).")
            except Exception:
                pass
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

st.markdown("##### Loaded payload (read-only)")
st.code(json.dumps(st.session_state.get("saved_payload", scenario_payload), indent=2), language="json")

st.caption("Note: To keep hosting costs near zero, this app uses local/ephemeral state only. For cloud saves, add Supabase/Firebase later.")

# -----------------------------
# --------- Footer ------------
# -----------------------------
with st.expander("ℹ️ How the simulator works"):
    st.markdown("""
- **Priced rounds** issue new investor shares at: `price_per_share = pre_money / pre_round_total_shares`.
- **Pre-money option pool top-up** issues new options so that `ESOP / (Total pre-investor) = target%`.
- **Post-money top-up** happens after issuing investor shares so that `ESOP / (Total post-investor) = target%`.
- Shares are kept abstract (e.g., start at 10,000,000) to avoid floating precision issues while keeping ratios exact.
- SAFE/convertible modeling can be added later by converting notes into shares at a cap/discount before the priced round.
""")
