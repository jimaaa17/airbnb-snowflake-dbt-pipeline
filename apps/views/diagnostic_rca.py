"""Diagnostic Root Cause Analysis (RCA) & A/B Experimentation Studio.

Empowers SMEs and Growth Leads to isolate metric variance across dimensions
and validate product hypotheses with rigorous statistical inference.
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
from apps.components.data_store import load_gold_obt_dataset


def render():
    st.markdown("### Diagnostic Root Cause Analysis (RCA) & Experimentation")
    st.caption("Investigate underlying variance drivers and quantify causal uplift from growth experiments.")

    df_obt = load_gold_obt_dataset()

    rca_tab, ab_tab = st.tabs([
        "🔬 Dimensional Root Cause Analysis",
        "🧪 A/B Experimentation & Hypothesis Engine"
    ])

    with rca_tab:
        st.markdown('<p class="section-header">Variance Attribution Across Market Dimensions</p>', unsafe_allow_html=True)
        st.caption("Decomposes platform-level shifts into granular dimensional drivers to answer *'Why did this metric change?'*")

        col1, col2 = st.columns(2)
        with col1:
            target_metric = st.selectbox("Select Target Metric to Investigate:", [
                "Booking Confirmation Rate (%)",
                "Total Revenue ($)",
                "Average Booking Value ($)"
            ])
        with col2:
            drill_dim = st.selectbox("Decompose Across Dimension:", [
                "CITY",
                "PROPERTY_TYPE",
                "ROOM_TYPE",
                "PRICE_PER_NIGHT_TAG",
                "IS_SUPERHOST"
            ])

        if "Confirmation" in target_metric or "Conversion" in target_metric:
            decomp_df = df_obt.groupby(drill_dim).apply(
                lambda x: pd.Series({
                    "Total Attempts": len(x),
                    "Confirmed Reservations": (x["BOOKING_STATUS"] == "confirmed").sum(),
                    "Cancelled Reservations": (x["BOOKING_STATUS"] == "cancelled").sum(),
                    "Confirmation Rate (%)": round((x["BOOKING_STATUS"] == "confirmed").mean() * 100, 2),
                    "Cancellation Rate (%)": round((x["BOOKING_STATUS"] == "cancelled").mean() * 100, 2)
                })
            ).reset_index().sort_values(by="Confirmation Rate (%)", ascending=False)
            st.dataframe(decomp_df, use_container_width=True)
        else:
            decomp_df = df_obt.groupby(drill_dim).agg(
                Total_Gross_Revenue=("TOTAL_AMOUNT", "sum"),
                Mean_Booking_Value=("TOTAL_AMOUNT", "mean"),
                Booking_Volume=("BOOKING_ID", "count")
            ).reset_index().sort_values(by="Total_Gross_Revenue", ascending=False)
            st.dataframe(decomp_df, use_container_width=True)

        st.markdown("##### 💡 SME Diagnostic Summary")
        top_driver = decomp_df.iloc[0][drill_dim]
        bottom_driver = decomp_df.iloc[-1][drill_dim]
        st.info(
            f"**Key Finding:** Across dimension **`{drill_dim}`**, the top performing cohort is **`{top_driver}`**, "
            f"while **`{bottom_driver}`** represents the largest opportunity for optimization."
        )

    with ab_tab:
        st.markdown('<p class="section-header">Two-Sample Hypothesis Testing (Conversion Uplift)</p>', unsafe_allow_html=True)
        st.caption("Standardized two-proportion Z-test evaluating product changes, pricing incentives, or algorithm adjustments.")

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("##### 🅰️ Control Variant (Baseline)")
            ctrl_visitors = st.number_input("Control Impressions / Attempts:", min_value=10, value=1200, step=50, key="ctrl_n")
            ctrl_converted = st.number_input("Control Confirmed Conversions:", min_value=0, value=840, step=10, key="ctrl_c")
            ctrl_rate = (ctrl_converted / ctrl_visitors) if ctrl_visitors > 0 else 0
            st.metric("Control Conversion Rate", f"{ctrl_rate * 100:.2f}%")

        with c_b:
            st.markdown("##### 🅱️ Test Treatment (New Feature)")
            test_visitors = st.number_input("Test Impressions / Attempts:", min_value=10, value=1250, step=50, key="test_n")
            test_converted = st.number_input("Test Confirmed Conversions:", min_value=0, value=960, step=10, key="test_c")
            test_rate = (test_converted / test_visitors) if test_visitors > 0 else 0
            st.metric("Test Conversion Rate", f"{test_rate * 100:.2f}%")

        # Two-proportion Z-test computation
        p_pool = (ctrl_converted + test_converted) / (ctrl_visitors + test_visitors)
        se = np.sqrt(p_pool * (1 - p_pool) * ((1 / ctrl_visitors) + (1 / test_visitors)))
        z_score = (test_rate - ctrl_rate) / se if se > 0 else 0.0
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / np.sqrt(2))))
        uplift_pct = ((test_rate - ctrl_rate) / ctrl_rate * 100) if ctrl_rate > 0 else 0.0

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Relative Uplift", f"{uplift_pct:+.2f}%", delta=f"{(test_rate - ctrl_rate) * 100:+.2f}% pts")
        with r2:
            st.metric("Z-Score", f"{z_score:.3f}")
        with r3:
            st.metric("p-Value", f"{p_value:.4f}")

        if p_value < 0.05:
            st.success(
                f"🎉 **Statistically Significant Uplift (p = {p_value:.4f} < 0.05):** "
                f"We reject the null hypothesis with >95% confidence. The treatment variant demonstrates genuine performance improvement."
            )
        else:
            st.warning(
                f"⏳ **Inconclusive Result (p = {p_value:.4f} ≥ 0.05):** "
                f"The observed difference could be attributed to natural variance. Recommend continuing the test until target sample size is reached."
            )
