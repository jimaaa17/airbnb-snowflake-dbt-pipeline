"""Airbnb Analytics Studio — Enterprise Semantic Intelligence Platform.

Scalable multi-page data application powered by Snowflake Medallion Marts,
dbt Semantic Layer / MetricFlow, and Scikit-Learn/Gradient Boosting ML Pipelines.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from apps.components.theme import apply_airbnb_theme
from apps.components.header import render_app_header
from apps.views import (
    executive_overview,
    diagnostic_rca,
    predictive_studio,
    metric_explorer,
    catalog_hub,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Airbnb Analytics Studio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Airbnb Corporate Design System
apply_airbnb_theme()

# -----------------------------------------------------------------------------
# 2. APPLICATION ROUTING & NAVIGATION
# -----------------------------------------------------------------------------
pages = {
    "Executive Analytics": [
        st.Page(executive_overview.render, title="Executive Overview", icon="📈", url_path="executive_overview", default=True),
        st.Page(diagnostic_rca.render, title="Diagnostic RCA & A/B", icon="🔬", url_path="diagnostic_rca"),
        st.Page(metric_explorer.render, title="Self-Service Metric Explorer", icon="⚡", url_path="metric_explorer"),
    ],
    "Predictive Intelligence": [
        st.Page(predictive_studio.render, title="Predictive ML Studio", icon="🎯", url_path="predictive_studio"),
    ],
    "Data Governance": [
        st.Page(catalog_hub.render, title="Catalog & Lineage Hub", icon="📚", url_path="catalog_hub"),
    ],
}

# Modern Streamlit Navigation (Native browser routing, deep-linking, collapsible sections)
current_page = st.navigation(pages, position="sidebar")

# -----------------------------------------------------------------------------
# 3. SIDEBAR FOOTER & ENTERPRISE CONTEXT (CLEAN PRODUCT UI)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        <div style="padding: 10px 0;">
            <p style="font-size: 0.8rem; font-weight: 700; color: #717171; text-transform: uppercase; margin-bottom: 6px;">Enterprise Governance</p>
            <div style="background: #F7F7F7; border: 1px solid #EBEBEB; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                <span style="font-size: 0.82rem; font-weight: 600; color: #222222;">🛡️ MetricFlow SSOT</span><br>
                <span style="font-size: 0.74rem; color: #717171;">Governed calculations enforced across all teams.</span>
            </div>
            <div style="background: #F7F7F7; border: 1px solid #EBEBEB; border-radius: 8px; padding: 10px 12px;">
                <span style="font-size: 0.82rem; font-weight: 600; color: #008A05;">● Live Data Mart</span><br>
                <span style="font-size: 0.74rem; color: #717171;">Snowflake Gold Mart • Synced</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("📖 SME Architecture Guide", expanded=False):
        st.caption(
            "Detailed architectural documentation, metric definitions, and ML gate reports are available in "
            "`docs/SEMANTIC_ARCHITECTURE_AND_ML_REPORT.md`."
        )

# -----------------------------------------------------------------------------
# 4. RENDER TOP HEADER & ACTIVE PAGE
# -----------------------------------------------------------------------------
render_app_header()
current_page.run()
