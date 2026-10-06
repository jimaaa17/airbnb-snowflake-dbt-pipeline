"""Airbnb Design System & UI Theme Components.

Provides custom CSS, typography, color palettes, and container components
faithfully inspired by Airbnb's design language (Rausch #FF385C, Circular font stack,
subtle card elevations, and clean whitespace).
"""

import streamlit as st


def apply_airbnb_theme():
    """Injects high-end Airbnb corporate CSS styling into Streamlit."""
    st.markdown(
        """
        <style>
            /* 1. Global Typography & Canvas */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                color: #222222;
            }

            /* Main Container Spacing */
            .main .block-container {
                padding-top: 1.8rem;
                padding-bottom: 3.5rem;
                padding-left: 2.5rem;
                padding-right: 2.5rem;
                max-width: 1400px;
            }

            /* 2. Top Header & Hero Branding */
            .app-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding-bottom: 1.2rem;
                margin-bottom: 1.5rem;
                border-bottom: 1px solid #EBEBEB;
            }
            .brand-lockup {
                display: flex;
                align-items: center;
                gap: 14px;
            }
            .brand-title {
                font-size: 1.6rem;
                font-weight: 800;
                color: #222222;
                letter-spacing: -0.02em;
                margin: 0;
                line-height: 1.2;
            }
            .brand-subtitle {
                font-size: 0.9rem;
                color: #717171;
                margin: 0;
            }
            .header-status-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background-color: #F7F7F7;
                border: 1px solid #EBEBEB;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                color: #484848;
            }
            .status-dot-green {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #008A05;
                display: inline-block;
            }

            /* 3. Surface & Metric Cards */
            .metric-card-container {
                background: #FFFFFF;
                border: 1px solid #EBEBEB;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .metric-card-container:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.07);
            }
            .metric-card-label {
                font-size: 0.82rem;
                font-weight: 600;
                color: #717171;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 6px;
            }
            .metric-card-value {
                font-size: 2.1rem;
                font-weight: 800;
                color: #222222;
                letter-spacing: -0.03em;
                line-height: 1.1;
                margin-bottom: 8px;
            }
            .metric-card-footer {
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-size: 0.8rem;
            }
            .badge-ssot {
                background-color: #FFF0F2;
                color: #FF385C;
                font-weight: 700;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.72rem;
                letter-spacing: 0.02em;
            }
            .badge-neutral {
                background-color: #F7F7F7;
                color: #717171;
                font-weight: 600;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.72rem;
            }

            /* 4. Filter Toolbar */
            .filter-toolbar {
                background-color: #F7F7F7;
                border: 1px solid #EBEBEB;
                border-radius: 12px;
                padding: 16px 20px;
                margin-bottom: 24px;
            }

            /* 5. Section Headers */
            .section-header {
                font-size: 1.25rem;
                font-weight: 700;
                color: #222222;
                letter-spacing: -0.01em;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
            }
            .section-caption {
                font-size: 0.88rem;
                color: #717171;
                margin-bottom: 1.2rem;
            }

            /* 6. Formula Code Callout */
            .governed-code-callout {
                background-color: #F8F9FA;
                border-left: 3px solid #FF385C;
                border-radius: 0 8px 8px 0;
                padding: 10px 14px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 0.82rem;
                color: #222222;
                margin-top: 6px;
                margin-bottom: 6px;
            }

            /* 7. Streamlit Native Elements Customization */
            div[data-testid="stSidebarHeader"] {
                padding-bottom: 0.5rem;
            }
            div[data-testid="stSidebarNav"] {
                padding-top: 0.5rem;
            }
            
            /* Clean tabs */
            button[data-baseweb="tab"] {
                font-weight: 600 !important;
                font-size: 0.95rem !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #FF385C !important;
                border-bottom-color: #FF385C !important;
            }

            /* Primary buttons */
            button[kind="primary"] {
                background-color: #FF385C !important;
                border-color: #FF385C !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
                border-radius: 8px !important;
            }
            button[kind="primary"]:hover {
                background-color: #E00B41 !important;
                border-color: #E00B41 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, badge_text: str = "Verified SSOT", delta_text: str = ""):
    """Renders a sleek Airbnb Cereal style metric card."""
    delta_markup = f"<span style='color: #008A05; font-weight: 600;'>{delta_text}</span>" if delta_text else ""
    card_html = f"""
    <div class="metric-card-container">
        <div class="metric-card-label">{title}</div>
        <div class="metric-card-value">{value}</div>
        <div class="metric-card-footer">
            <span class="badge-ssot">{badge_text}</span>
            {delta_markup}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
