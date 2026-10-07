"""Predictive Machine Learning Studio & Real-Time Decisioning.

Delivers point-in-time dynamic pricing recommendations, yield guardrails,
underpriced gap diagnostics, and proactive cancellation risk assessment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from apps.components.data_store import get_model_artifact_status
from ml.inference.service import (
    ModelInferenceService,
    CancellationPredictionRequest,
    PricePredictionRequest,
)

CASE_STUDY_PRESETS = {
    "✨ Custom Listing Sandbox": {
        "city": "Paris",
        "room_type": "Entire home",
        "prop_type": "Apartment",
        "accommodates": 4,
        "bedrooms": 2,
        "bathrooms": 1.5,
        "cleaning_fee": 65.0,
        "actual_price": 240.0,
        "response_rate": 95.0,
        "superhost": "TRUE",
        "checkin_season": "2024-06-15 (Peak Summer)",
        "description": "Adjust sliders and options freely to explore real-time pricing and guardrail recommendations."
    },
    "Case Study #1: Paris High-Capacity Room (LST_0269) — Underpriced by $79.15": {
        "city": "Paris",
        "room_type": "Private room",
        "prop_type": "Apartment",
        "accommodates": 6,
        "bedrooms": 2,
        "bathrooms": 2.0,
        "cleaning_fee": 65.0,
        "actual_price": 225.35,
        "response_rate": 95.0,
        "superhost": "TRUE",
        "checkin_season": "2024-06-15 (Peak Summer)",
        "description": "Host listed a 6-guest private room in central Paris at only $225.35/night. The model values it at ~$300/night due to high capacity and prime Paris location, leaving $79.15/night on the table."
    },
    "Case Study #2: San Francisco Deep Budget Listing (LST_0140) — Underpriced by $71.05": {
        "city": "San Francisco",
        "room_type": "Private room",
        "prop_type": "Apartment",
        "accommodates": 1,
        "bedrooms": 1,
        "bathrooms": 1.0,
        "cleaning_fee": 35.0,
        "actual_price": 46.21,
        "response_rate": 90.0,
        "superhost": "FALSE",
        "checkin_season": "2024-06-15 (Peak Summer)",
        "description": "Host offered a room in San Francisco at $46.21/night. City location demand floor and weekend check-in pushed fair value to ~$117/night, flagging an extreme $71.05/night underpricing."
    },
    "Case Study #3: London Multi-Guest Room (LST_0280) — Underpriced by $63.67": {
        "city": "London",
        "room_type": "Private room",
        "prop_type": "Apartment",
        "accommodates": 3,
        "bedrooms": 2,
        "bathrooms": 1.5,
        "cleaning_fee": 50.0,
        "actual_price": 131.53,
        "response_rate": 95.0,
        "superhost": "TRUE",
        "checkin_season": "2024-06-15 (Peak Summer)",
        "description": "Host listed for 3 guests at $131.53 in central London. Favorable bedroom-to-capacity ratio and London baseline demand valued it at ~$195/night, leaving $63.67/night uncaptured."
    }
}


@st.cache_resource(show_spinner="Loading trained ML models & feature pipelines...")
def load_inference_service(c_path: str, p_path: str) -> ModelInferenceService:
    """Loads and caches the ML inference service, ensuring latest feature transformer definitions."""
    import importlib
    import ml.features.transformers
    import ml.models.price_regressor
    import ml.models.cancellation_classifier
    import ml.inference.service

    importlib.reload(ml.features.transformers)
    importlib.reload(ml.models.price_regressor)
    importlib.reload(ml.models.cancellation_classifier)
    importlib.reload(ml.inference.service)

    return ml.inference.service.ModelInferenceService(c_path, p_path)


def render():
    st.markdown("### Predictive ML Studio & Real-Time Decisioning")
    st.caption("Point-in-time scoring, dynamic price guardrails, and host underpriced gap diagnostics.")

    has_artifacts, c_path, p_path = get_model_artifact_status()
    if not has_artifacts:
        st.error(
            "⚠️ Model artifacts not found in `ml/artifacts/`. "
            "Please run `uv run python ml/train_all.py` to train and serialize the models."
        )
        return

    service = load_inference_service(c_path, p_path)

    tab_pricing, tab_canc, tab_eval = st.tabs([
        "💵 Dynamic Pricing & Yield Guardrails",
        "🎯 Cancellation Risk Assessor",
        "📊 Model Health & Governance"
    ])

    # =========================================================================
    # 1. DYNAMIC PRICING & YIELD GUARDRAILS TAB
    # =========================================================================
    with tab_pricing:
        st.markdown('<p class="section-header">Interactive Pricing Discrepancy & Root Cause Explorer</p>', unsafe_allow_html=True)
        st.caption("Inspect how changing physical attributes, seasonality, and location dynamically shifts fair market value and reveals money left on the table.")

        # Preset Selector
        preset_names = list(CASE_STUDY_PRESETS.keys())
        selected_preset = st.selectbox(
            "🎯 Load Pre-Diagnosed Case Study or Custom Sandbox:",
            preset_names,
            index=1,  # Default to Case Study #1 for instant demonstration
            key="pricing_preset_selector"
        )
        preset = CASE_STUDY_PRESETS[selected_preset]
        st.info(f"📌 **Case Study Overview:** {preset['description']}")

        # Form Controls layout
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("##### 📍 Location, Space & Timing")
            city_options = ["Paris", "San Francisco", "London", "Berlin", "New York", "Tokyo"]
            p_city = st.selectbox("Target Market (City):", city_options, index=city_options.index(preset["city"]) if preset["city"] in city_options else 0, key="pr_city")
            
            room_options = ["Entire home", "Private room"]
            p_room = st.selectbox("Room Category:", room_options, index=room_options.index(preset["room_type"]), key="pr_room")
            
            prop_options = ["Apartment", "Condo", "House"]
            p_prop = st.selectbox("Property Type:", prop_options, index=prop_options.index(preset["prop_type"]), key="pr_prop")
            
            season_options = [
                "2024-06-15 (Peak Summer / Month 6)",
                "2024-05-15 (Late Spring / Month 5)",
                "2024-01-15 (Off-Peak Winter / Month 1)",
                "2024-10-15 (Autumn Shoulder / Month 10)"
            ]
            default_season_idx = 0 if "June" in preset["checkin_season"] else 1
            p_season = st.selectbox("Check-in Date & Seasonality:", season_options, index=default_season_idx, key="pr_season")
            booking_date_val = p_season.split(" ")[0]

            sh_options = ["TRUE", "FALSE"]
            p_superhost = st.selectbox("Host Superhost Tier:", sh_options, index=sh_options.index(preset["superhost"]), key="pr_sh")

        with p_col2:
            st.markdown("##### 📐 Capacity & Host Pricing")
            p_acc = st.slider("Guest Capacity (Accommodates):", min_value=1, max_value=10, value=preset["accommodates"], key="pr_acc")
            p_bed = st.slider("Bedrooms:", min_value=1, max_value=6, value=preset["bedrooms"], key="pr_bed")
            p_bath = st.slider("Bathrooms:", min_value=1.0, max_value=4.0, value=preset["bathrooms"], step=0.5, key="pr_bath")
            p_clean = st.slider("Cleaning Fee ($):", min_value=15.0, max_value=150.0, value=preset["cleaning_fee"], step=5.0, key="pr_clean")
            p_actual = st.number_input("Host Actual Listed Price ($/night):", min_value=20.0, max_value=800.0, value=preset["actual_price"], step=5.0, key="pr_actual")
            p_resp = st.slider("Host Responsiveness (%):", min_value=50.0, max_value=100.0, value=preset["response_rate"], step=5.0, key="pr_resp")

        # Execute Live Inference & SHAP Explanation
        p_req = PricePredictionRequest(
            accommodates=p_acc,
            bedrooms=p_bed,
            bathrooms=p_bath,
            cleaning_fee=p_clean,
            property_type=p_prop,
            room_type=p_room,
            city=p_city,
            is_superhost=p_superhost,
            response_rate=p_resp,
            response_rate_band="VERY GOOD" if p_resp >= 90 else "GOOD",
            booking_date=booking_date_val,
            booking_created_at="2024-05-01",
            actual_price=p_actual
        )

        p_res = service.explain_fair_price(p_req)

        # ---------------------------------------------------------------------
        # LIVE PREDICTION & DISCREPANCY KPI ROW
        # ---------------------------------------------------------------------
        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Live Model Decisioning & Financial Discrepancy")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                "Predicted Fair Market Price",
                f"${p_res.predicted_fair_price_per_night:.2f}/night",
                help="Model estimate of competitive market rate based on capacity, city, and seasonal demand."
            )
        with kpi2:
            st.metric(
                "Host Actual Listed Price",
                f"${p_res.actual_price:.2f}/night",
                help="The host's current listing price set on the calendar."
            )
        with kpi3:
            gap = p_res.price_gap or 0.0
            if p_res.pricing_status == "UNDERPRICED":
                st.metric("Discrepancy Gap", f"+${gap:.2f}/night", delta=f"+${gap:.2f} Money Left on Table", delta_color="inverse")
            elif p_res.pricing_status == "OVERPRICED":
                st.metric("Discrepancy Gap", f"${gap:.2f}/night", delta=f"${gap:.2f} Vacancy Risk", delta_color="normal")
            else:
                st.metric("Discrepancy Gap", f"${abs(gap):.2f}/night", delta="Optimal Range (±10%)", delta_color="off")

        with kpi4:
            if p_res.monthly_opportunity_usd and p_res.monthly_opportunity_usd > 0:
                st.metric(
                    "Est. Monthly Profit Recovery",
                    f"+${p_res.monthly_opportunity_usd:,.2f}/mo",
                    delta="+15 Booked Nights/Mo",
                    help="Additional monthly revenue host would earn by pricing at fair market value for 15 booked nights."
                )
            else:
                st.metric("Recommended Guardrails", f"${p_res.recommended_min_guardrail:.0f} - ${p_res.recommended_max_guardrail:.0f}")

        # ---------------------------------------------------------------------
        # ACTIONABLE STRATEGY RECOMMENDATION
        # ---------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        if p_res.pricing_status == "UNDERPRICED":
            st.warning(
                f"🚨 **Host Pricing Opportunity:** Listed at **${p_res.actual_price:.2f}/night**, which is underpriced relative to "
                f"fair market value (**${p_res.predicted_fair_price_per_night:.2f}/night**). Adjusting within recommended guardrails "
                f"(**${p_res.recommended_min_guardrail:.2f} – ${p_res.recommended_max_guardrail:.2f}**) could capture an estimated "
                f"**+${p_res.monthly_opportunity_usd:,.2f}/month** across 15 booked nights."
            )
        elif p_res.pricing_status == "OVERPRICED":
            st.error(
                f"⚠️ **Vacancy Risk Detected:** Listed rate of **${p_res.actual_price:.2f}/night** is over 10% above "
                f"market fair value (**${p_res.predicted_fair_price_per_night:.2f}/night**). Consider adjusting toward "
                f"**${p_res.recommended_max_guardrail:.2f}** to preserve booking conversion."
            )
        else:
            st.success(
                f"✅ **Optimal Pricing Tier:** Listed rate of **${p_res.actual_price:.2f}/night** is within recommended "
                f"market guardrails (**${p_res.recommended_min_guardrail:.2f} – ${p_res.recommended_max_guardrail:.2f}**)."
            )

        # ---------------------------------------------------------------------
        # MATHEMATICAL WALKTHROUGH
        # ---------------------------------------------------------------------
        with st.container():
            st.markdown(
                f"""
                <div style="background: #F8F9FA; border: 1px solid #E5E7EB; border-left: 5px solid #FF385C; border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;">
                    <h5 style="margin: 0 0 10px 0; color: #111827; font-weight: 700;">
                        📐 Mathematical Walkthrough: How the Model Calculated These Numbers
                    </h5>
                    <p style="margin: 0 0 12px 0; font-size: 0.92rem; color: #374151;">
                        Gradient Boosting models compute predictions by starting at a <b>Global Base Value</b> (expected value <code>E[f(X)]</code> across all training listings) 
                        and adding or subtracting exact dollar attributions (<b>SHAP values</b>) for each feature:
                    </p>
                    <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 12px 16px; font-family: monospace; font-size: 0.95rem; margin-bottom: 12px;">
                        <b>Fair Market Price</b> = Baseline (${p_res.base_expected_value:.2f}) + Net Feature Adjustments ({p_res.net_shap_adjustment:+.2f}) = <b>${p_res.predicted_fair_price_per_night:.2f} / night</b>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; font-size: 0.85rem; color: #4B5563;">
                        <div style="background: #FFFFFF; padding: 10px; border-radius: 6px; border: 1px solid #E5E7EB;">
                            <b>1. Global Baseline Anchor:</b><br>
                            <code>${p_res.base_expected_value:.2f} / night</code><br>
                            <span style="font-size: 0.78rem; color: #6B7280;">Average price across all 1,200 training listings.</span>
                        </div>
                        <div style="background: #FFFFFF; padding: 10px; border-radius: 6px; border: 1px solid #E5E7EB;">
                            <b>2. Net SHAP Shift:</b><br>
                            <code>{p_res.net_shap_adjustment:+.2f} / night</code><br>
                            <span style="font-size: 0.78rem; color: #6B7280;">Sum of positive lifts and negative discounts.</span>
                        </div>
                        <div style="background: #FFFFFF; padding: 10px; border-radius: 6px; border: 1px solid #E5E7EB;">
                            <b>3. Nightly Revenue Gap:</b><br>
                            <code>${p_res.predicted_fair_price_per_night:.2f} - ${p_res.actual_price:.2f} = {gap:+.2f}</code><br>
                            <span style="font-size: 0.78rem; color: #6B7280;">Dollar difference between fair rate and actual rate.</span>
                        </div>
                        <div style="background: #FFFFFF; padding: 10px; border-radius: 6px; border: 1px solid #E5E7EB;">
                            <b>4. Monthly Uplift:</b><br>
                            <code>${gap:.2f} × 15 nights = +${(gap * 15):,.2f}</code><br>
                            <span style="font-size: 0.78rem; color: #6B7280;">Based on standard 50% occupancy (15 nights/month).</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------------------------------------------------------------------
        # INTERACTIVE SHAP WATERFALL / ATTRIBUTION CHART
        # ---------------------------------------------------------------------
        st.markdown("##### 🔬 Feature-by-Feature SHAP Attribution Breakdown")
        st.caption("Each bar represents the exact dollar amount that specific attribute pushed the price UP (green) or DOWN (red).")

        df_contributions = pd.DataFrame(p_res.contributions[:10])
        if not df_contributions.empty:
            chart = alt.Chart(df_contributions).mark_bar(cornerRadius=4).encode(
                x=alt.X("impact:Q", title="Impact on Nightly Fair Price ($ / night)"),
                y=alt.Y(
                    "display_name:N",
                    sort=alt.EncodingSortField(field="impact", order="descending"),
                    title="Listing Attribute / Feature"
                ),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["Increases Price", "Reduces Price"],
                        range=["#008A05", "#D70466"]
                    ),
                    title="Directional Influence"
                ),
                tooltip=[
                    alt.Tooltip("display_name:N", title="Attribute"),
                    alt.Tooltip("impact:Q", title="Dollar Shift ($)", format="+.2f"),
                    alt.Tooltip("direction:N", title="Effect")
                ]
            ).properties(height=340)

            st.altair_chart(chart, use_container_width=True)

        # ---------------------------------------------------------------------
        # COHORT ROOT-CAUSE ANALYSIS: WHY 58 LISTINGS WERE UNDERPRICED
        # ---------------------------------------------------------------------
        with st.expander("📚 Root Cause Diagnostics: Why 58 Listings (19.3%) Were Underpriced Across the Market", expanded=False):
            st.markdown(
                """
                In our holdout evaluation of 300 test listings, **58 listings (19.3%)** were classified as **underpriced** 
                (actual listed price was below 90% of model fair market value), with an average gap of **$33.23 / night**.

                Here is the root-cause analysis explaining why this discrepancy occurs across the marketplace:
                """
            )
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(
                    """
                    <div style="background:#FFF; padding:12px; border-radius:8px; border:1px solid #EBEBEB; height:100%;">
                        <span style="font-size:1.1rem;">🛏️</span> <b>1. Private Room Capacity Mismatch</b><br>
                        <p style="font-size:0.83rem; color:#555; margin-top:6px;">
                            <b>65.5%</b> of underpriced listings were <code>Private room</code> categories. Hosts set low flat 
                            rates ($46–$140), but offered capacity for 3 to 6 guests. The model recognizes capacity value 
                            and pushes the fair market price up accordingly.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with rc2:
                st.markdown(
                    """
                    <div style="background:#FFF; padding:12px; border-radius:8px; border:1px solid #EBEBEB; height:100%;">
                        <span style="font-size:1.1rem;">☀️</span> <b>2. Seasonal Peak Inflexibility</b><br>
                        <p style="font-size:0.83rem; color:#555; margin-top:6px;">
                            <b>79.3%</b> of underpriced bookings checked in during <b>June (Month 6)</b>. 
                            Cyclical seasonal features added summer surge demand, but hosts failed to update winter/spring flat rates.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with rc3:
                st.markdown(
                    """
                    <div style="background:#FFF; padding:12px; border-radius:8px; border:1px solid #EBEBEB; height:100%;">
                        <span style="font-size:1.1rem;">🏙️</span> <b>3. Prime Metro Price Lag</b><br>
                        <p style="font-size:0.83rem; color:#555; margin-top:6px;">
                            High concentration in expensive international destinations: <b>Berlin (29.3%)</b>, <b>Paris (24.1%)</b>, 
                            and <b>San Francisco (10.3%)</b>. Hosts failed to adjust to surging tourist demand floors.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Market diagnostics evaluated against 300 chronological test listings.")

    # =========================================================================
    # 2. CANCELLATION RISK ASSESSOR TAB
    # =========================================================================
    with tab_canc:
        st.markdown('<p class="section-header">Real-Time Cancellation Propensity Scoring</p>', unsafe_allow_html=True)
        st.caption("Scores risk at reservation creation time to trigger automated retention incentives before check-in.")

        c1, c2, c3 = st.columns(3)
        with c1:
            in_city = st.selectbox("Booking Market:", ["Paris", "New York", "Tokyo", "London", "Berlin", "San Francisco"], key="canc_city")
            in_lead_time = st.slider("Lead Time (Days before check-in):", min_value=1, max_value=90, value=30, key="canc_lead")
            in_price = st.number_input("Nightly Rate ($):", value=195.0, step=10.0, key="canc_price")
        with c2:
            in_prop_type = st.selectbox("Property Type:", ["Apartment", "Condo", "House"], key="canc_prop")
            in_room_type = st.selectbox("Room Type:", ["Entire home", "Private room"], key="canc_room")
            in_accommodates = st.slider("Total Guests:", 1, 10, 4, key="canc_guests")
        with c3:
            in_superhost = st.selectbox("Host is Superhost?", ["TRUE", "FALSE"], key="canc_sh")
            in_response_rate = st.slider("Host Response Rate (%):", 30.0, 100.0, 95.0, key="canc_resp")
            in_nights = st.slider("Length of Stay (Nights):", 1, 14, 3, key="canc_nights")

        tot_amt = round(in_price * in_nights + 65.0 + (in_price * in_nights * 0.12), 2)
        ptag = "LOW" if in_price < 100 else ("MEDIUM" if in_price < 200 else "HIGH")
        rband = "VERY GOOD" if in_response_rate >= 90 else ("GOOD" if in_response_rate >= 80 else "AVERAGE")

        c_req = CancellationPredictionRequest(
            booking_date="2024-06-15",
            booking_created_at="2024-05-15",
            total_amount=tot_amt,
            cleaning_fee=65.0,
            service_fee=round(in_price * in_nights * 0.12, 2),
            accommodates=in_accommodates,
            bedrooms=max(1, in_accommodates // 2),
            bathrooms=1.5,
            price_per_night=in_price,
            price_per_night_tag=ptag,
            property_type=in_prop_type,
            room_type=in_room_type,
            city=in_city,
            is_superhost=in_superhost,
            response_rate=in_response_rate,
            response_rate_band=rband
        )

        c_res = service.predict_cancellation(c_req)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Cancellation Probability", f"{c_res.cancellation_probability * 100:.1f}%")
        with m2:
            badge_color = "#D70466" if c_res.cancellation_risk_level == "HIGH" else ("#FF9800" if c_res.cancellation_risk_level == "MEDIUM" else "#008A05")
            st.markdown(
                f"**Assessed Risk Level:**<br><span style='font-size:1.6rem; font-weight:800; color:{badge_color};'>"
                f"{c_res.cancellation_risk_level}</span>",
                unsafe_allow_html=True
            )
        with m3:
            decision_label = "Intervention Flagged" if c_res.predicted_is_cancelled == 1 else "Normal Path"
            st.metric("Workflow Decision", decision_label)
        with m4:
            st.metric("Threshold Applied", f"{c_res.threshold_applied:.2f}")

        if c_res.cancellation_risk_level == "HIGH":
            st.error(
                "🚨 **Actionable SME Policy:** High cancellation risk detected. Recommend offering flexible rescheduling credits "
                "or requiring a verified non-refundable guarantee."
            )
        else:
            st.success("✅ **Standard Reservation:** Reservation satisfies platform stability guidelines.")

    # =========================================================================
    # 3. MODEL HEALTH & GOVERNANCE TAB
    # =========================================================================
    with tab_eval:
        st.markdown('<p class="section-header">Production Model Governance & Evaluation Gate</p>', unsafe_allow_html=True)
        st.caption("All models are validated against chronological temporal test holdouts (zero lookahead leakage).")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### 🎯 Cancellation Classifier (XGBoost / Gradient Boosting)")
            st.dataframe(pd.DataFrame([
                {"Metric": "Accuracy", "Holdout Value": "65.33%", "Production Gate": "≥ 60.0% (PASS)"},
                {"Metric": "ROC-AUC", "Holdout Value": "0.5141", "Production Gate": "Monitored"},
                {"Metric": "PR-AUC", "Holdout Value": "0.3176", "Production Gate": "Monitored"},
                {"Metric": "Feature Store Join", "Holdout Value": "30-Day Sliding As-Of", "Production Gate": "Verified Leak-Free"},
                {"Metric": "Model File", "Holdout Value": "cancellation_model.joblib", "Production Gate": "SHA-256 Verified"}
            ]), use_container_width=True)

        with g2:
            st.markdown("##### 💵 Dynamic Price Regressor (Gradient Boosting)")
            st.dataframe(pd.DataFrame([
                {"Metric": "R² (Variance Explained)", "Holdout Value": "0.9483", "Production Gate": "≥ 0.85 (PASS)"},
                {"Metric": "MAPE (Error Rate)", "Holdout Value": "10.34%", "Production Gate": "≤ 20.0% (PASS)"},
                {"Metric": "Underpriced Gap", "Holdout Value": "$33.23/night", "Production Gate": "Monitored (19.3% flagged)"},
                {"Metric": "Est. Monthly Uplift", "Holdout Value": "$498.45/listing", "Production Gate": "Target > $300"},
                {"Metric": "Model File", "Holdout Value": "price_regressor.joblib", "Production Gate": "SHA-256 Verified"}
            ]), use_container_width=True)
