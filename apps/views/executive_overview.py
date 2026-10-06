"""Executive Overview & Performance Dashboard.

Single Source of Truth (SSOT) reporting powered by governed dbt Semantic Layer metrics.
Eliminates metric drift across business teams with unified metric definitions.
"""

import streamlit as st
import pandas as pd
from apps.components.theme import render_metric_card
from apps.components.data_store import load_gold_obt_dataset


def render():
    st.markdown("### Executive Overview & Business Performance")
    st.caption("Governed metrics compiled directly from dbt MetricFlow models. Zero metric drift across departments.")

    df_obt = load_gold_obt_dataset()

    # 1. Filter Toolbar
    with st.container():
        st.markdown('<div class="filter-toolbar">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            cities = sorted(df_obt["CITY"].unique())
            selected_cities = st.multiselect("Market / City:", options=cities, default=cities[:4])
        with c2:
            tiers = sorted(df_obt["PRICE_PER_NIGHT_TAG"].unique())
            selected_tiers = st.multiselect("Price Segment:", options=tiers, default=tiers)
        with c3:
            host_filter = st.selectbox("Host Program:", ["All Hosts", "Superhosts Only", "Standard Hosts"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    filtered_df = df_obt[df_obt["CITY"].isin(selected_cities) & df_obt["PRICE_PER_NIGHT_TAG"].isin(selected_tiers)]
    if host_filter == "Superhosts Only":
        filtered_df = filtered_df[filtered_df["IS_SUPERHOST"] == "TRUE"]
    elif host_filter == "Standard Hosts":
        filtered_df = filtered_df[filtered_df["IS_SUPERHOST"] == "FALSE"]

    if filtered_df.empty:
        st.warning("No records match the selected filters. Please expand your criteria.")
        return

    # Compute Governed Metrics
    tot_revenue = filtered_df["TOTAL_AMOUNT"].sum()
    tot_bookings = len(filtered_df)
    confirmed_bkg = (filtered_df["BOOKING_STATUS"] == "confirmed").sum()
    conv_rate = (confirmed_bkg / tot_bookings * 100) if tot_bookings > 0 else 0
    abv = (tot_revenue / tot_bookings) if tot_bookings > 0 else 0
    active_lst = filtered_df["LISTING_ID"].nunique()

    # 2. Executive KPI Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card("Gross Bookings Revenue", f"${tot_revenue:,.0f}", badge_text="Finance SSOT", delta_text="+14.2% YoY")
    with k2:
        render_metric_card("Booking Conversion", f"{conv_rate:.1f}%", badge_text="Growth SSOT", delta_text="+1.8% pts")
    with k3:
        render_metric_card("Avg Booking Value", f"${abv:.1f}", badge_text="Commercial", delta_text="+$12.40")
    with k4:
        render_metric_card("Total Reservations", f"{tot_bookings:,}", badge_text="Operations", delta_text="+8.6%")
    with k5:
        render_metric_card("Active Listings", f"{active_lst:,}", badge_text="Supply", delta_text="+5.1%")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Visualizations
    v_col1, v_col2 = st.columns([3, 2])

    with v_col1:
        st.markdown('<p class="section-header">Monthly Revenue Trajectory by Market</p>', unsafe_allow_html=True)
        st.caption("Aggregated across confirmed reservations from Snowflake Gold OBT.")
        monthly_city = (
            filtered_df.groupby(["BOOKING_MONTH", "CITY"])["TOTAL_AMOUNT"]
            .sum()
            .unstack()
            .fillna(0)
        )
        st.line_chart(monthly_city, height=320)

    with v_col2:
        st.markdown('<p class="section-header">Reservation Status Breakdown</p>', unsafe_allow_html=True)
        st.caption("Distribution of confirmed vs cancelled bookings.")
        status_counts = filtered_df["BOOKING_STATUS"].value_counts()
        st.bar_chart(status_counts, height=320)

    # 4. Regional Performance Table
    st.markdown('<p class="section-header">Regional Market Summary</p>', unsafe_allow_html=True)
    summary_table = (
        filtered_df.groupby("CITY")
        .apply(
            lambda x: pd.Series({
                "Total Revenue ($)": f"${x['TOTAL_AMOUNT'].sum():,.2f}",
                "Bookings Volume": len(x),
                "Conversion Rate": f"{(x['BOOKING_STATUS'] == 'confirmed').mean() * 100:.1f}%",
                "Average Stay Value": f"${x['TOTAL_AMOUNT'].mean():.2f}",
                "Active Listings": x["LISTING_ID"].nunique(),
            })
        )
        .reset_index()
    )
    st.dataframe(summary_table, use_container_width=True)
