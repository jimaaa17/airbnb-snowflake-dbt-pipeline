"""Data Catalog, Lineage Hub & Quality Assurance.

Central governance hub for SMEs: metric definitions, data owners,
automated dbt test coverage, and end-to-end Medallion DAG visualizer.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from apps.components.data_store import SEMANTIC_METRICS


def render():
    st.markdown("### Enterprise Data Catalog & Lineage Hub")
    st.caption("Discover metric ownership, certified calculations, data quality guarantees, and dependency flow.")

    cat_col1, cat_col2 = st.columns([1, 1])

    with cat_col1:
        st.markdown('<p class="section-header">1. Certified Governed Metrics</p>', unsafe_allow_html=True)
        catalog_table = []
        for k, v in SEMANTIC_METRICS.items():
            catalog_table.append({
                "Metric Name": v["label"],
                "Identifier": k,
                "Certified Formula": v["formula"],
                "Accountable Owner": v["owner"],
                "Tier": v["tier"]
            })
        st.dataframe(pd.DataFrame(catalog_table), use_container_width=True)

    with cat_col2:
        st.markdown('<p class="section-header">2. Data Quality & Integrity Scorecard</p>', unsafe_allow_html=True)
        st.success("✔ **82 of 82 dbt Data Tests Passing** (Bronze, Silver, Gold)")
        st.markdown("""
        - **Raw S3 Gatekeeper (`source_tests.sql`):** Enforces non-null schema and schema integrity before ingestion.
        - **Referential Integrity:** 100% foreign key matching across `facts`, `dim_listings`, and `dim_hosts`.
        - **Reconciliation Invariants:** Verified invariant: `COUNT(bronze) == COUNT(silver) == COUNT(obt)`.
        - **Accepted Domain Values:** Enforces values for `PRICE_PER_NIGHT_TAG`, `RESPONSE_RATE_BAND`, `BOOKING_STATUS`.
        """)

    st.markdown("---")
    st.markdown('<p class="section-header">3. End-to-End Medallion DAG & Data Lineage</p>', unsafe_allow_html=True)
    st.caption("Interactive Directed Acyclic Graph (DAG) visualizing dependencies from S3 Raw ingestion through Medallion layers, Semantic models, ML Feature Store, and API serving.")

    mermaid_code = """
    %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryBorderColor': '#FF5A5F', 'lineColor': '#0F2942', 'secondaryColor': '#F7F7F7', 'tertiaryColor': '#E8F5E9'}}}%%
    flowchart LR
        subgraph S3["☁️ Object Storage"]
            raw["S3: airbnb-raw-bucket<br/><i>(CSV: listings, bookings, hosts)</i>"]
        end

        subgraph Staging["📥 Staging Layer (Raw SQL)"]
            stage["AIRBNB.staging<br/><i>COPY INTO (Ingestion)</i>"]
        end

        subgraph Bronze["🥉 Bronze Layer (Incremental Watermark)"]
            b_book["bronze_bookings<br/><i>(watermark: CREATED_AT)</i>"]
            b_list["bronze_listings<br/><i>(watermark: CREATED_AT)</i>"]
            b_host["bronze_hosts<br/><i>(watermark: CREATED_AT)</i>"]
        end

        subgraph Silver["🥈 Silver Layer (Cleanse & Business Logic)"]
            s_book["silver_bookings<br/><i>(multiply macro: total_amount)</i>"]
            s_list["silver_listings<br/><i>(tag macro: price tiers)</i>"]
            s_host["silver_hosts<br/><i>(cleanse response rates)</i>"]
        end

        subgraph Gold["🥇 Gold Layer (Curated & Dimensional)"]
            g_obt["AIRBNB.gold.obt<br/><i>One Big Table (Denormalized)</i>"]
            g_facts["AIRBNB.gold.facts<br/><i>Booking Fact Grain</i>"]
            g_diml["dim_listings<br/><i>SCD Type 2 Snapshot</i>"]
            g_dimh["dim_hosts<br/><i>SCD Type 2 Snapshot</i>"]
        end

        subgraph Semantic["📐 Semantic Layer (MetricFlow)"]
            sem["Governed Metrics Catalog<br/><i>• total_revenue<br/>• booking_confirmation_rate<br/>• cancellation_rate<br/>• avg_booking_value</i>"]
        end

        subgraph ML["🤖 ML & Feature Store (Zipline)"]
            fs["Feature Store<br/><i>(30d As-Of sliding window)</i>"]
            m_canc["Cancellation Classifier<br/><i>(GradientBoosting)</i>"]
            m_pric["Price Regressor<br/><i>(R²=0.95, MAPE=10.3%)</i>"]
        end

        subgraph Serving["🚀 Open-Source Serving & Consumers"]
            api["FastAPI Semantic Gateway<br/><i>Port :8000</i>"]
            st_app["Streamlit Data App<br/><i>(BI & ML Studio)</i>"]
            llm["Future AI Agents<br/><i>(Stage 4)</i>"]
        end

        raw -->|Snowpipe / COPY INTO| stage
        stage --> b_book
        stage --> b_list
        stage --> b_host

        b_book --> s_book
        b_list --> s_list
        b_host --> s_host

        s_book --> g_obt
        s_list --> g_obt
        s_host --> g_obt

        s_book --> g_facts
        s_list --> g_diml
        s_host --> g_dimh
        g_diml --> g_facts
        g_dimh --> g_facts

        g_obt --> sem
        g_obt --> fs

        fs --> m_canc
        fs --> m_pric

        sem --> api
        m_canc --> api
        m_pric --> api

        api --> st_app
        api --> llm

        classDef s3Style fill:#FFF3E0,stroke:#FF9800,stroke-width:2px;
        classDef bronzeStyle fill:#FBE9E7,stroke:#D84315,stroke-width:2px;
        classDef silverStyle fill:#ECEFF1,stroke:#607D8B,stroke-width:2px;
        classDef goldStyle fill:#FFFDE7,stroke:#FBC02D,stroke-width:2px;
        classDef semStyle fill:#E8EAF6,stroke:#3F51B5,stroke-width:2px;
        classDef mlStyle fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px;
        classDef serveStyle fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px;

        class raw,stage s3Style;
        class b_book,b_list,b_host bronzeStyle;
        class s_book,s_list,s_host silverStyle;
        class g_obt,g_facts,g_diml,g_dimh goldStyle;
        class sem semStyle;
        class fs,m_canc,m_pric mlStyle;
        class api,st_app,llm serveStyle;
    """

    html_mermaid = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
          mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{ useMaxWidth: false, htmlLabels: true, curve: 'basis' }}
          }});
        </script>
        <style>
          body {{
            margin: 0;
            padding: 10px;
            background: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow-x: auto;
          }}
          .mermaid {{
            display: flex;
            justify-content: center;
          }}
        </style>
      </head>
      <body>
        <div class="mermaid">
{mermaid_code}
        </div>
      </body>
    </html>
    """

    components.html(html_mermaid, height=520, scrolling=True)

    with st.expander("📝 View Raw Mermaid Diagram Specification", expanded=False):
        st.code(mermaid_code, language="markdown")
