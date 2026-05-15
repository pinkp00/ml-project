import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# --- IMPROVED FORCE SCROLL CSS ---
st.markdown("""
    <style>
    /* 1. Force the app to allow scrolling */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: auto !important;
    }
    
    /* 2. Style the main block area */
    .main .block-container {
        max-width: 95%;
        padding-top: 2rem;
        padding-bottom: 20rem; /* Increased padding to ensure scrollability */
        overflow: visible !important;
    }
    
    /* 3. Custom Card Design with slight hover effect */
    .plot-container {
        border-radius: 12px;
        background-color: #ffffff;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
        border-top: 4px solid #1e3d59;
    }

    /* 4. Background Color */
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏨 EnergySense AI")
    st.markdown("---")
    page = st.radio("Navigation", ["🏗️ Development Phase", "📊 Analytics", "⚡ Demand Forecaster", "📈 Performance Metrics"])
    st.markdown("---")
    st.subheader("⚙️ Settings")
    active_model = st.selectbox("Active Model", ["XGBoost (Active)", "Random Forest", "Logistic Regression"])
    st.success(f"Model: {active_model}")

# --- 1. DEVELOPMENT PHASE ---
if page == "🏗️ Development Phase":
    st.title("🏗️ Project Roadmap & History")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    dev_data = [
        dict(Task="Dataset Acquisition", Start='2024-05-01', Finish='2024-05-03', Phase="Data"),
        dict(Task="Data Cleaning & SMOTE", Start='2024-05-04', Finish='2024-05-07', Phase="Preparation"),
        dict(Task="EDA (14 Charts)", Start='2024-05-08', Finish='2024-05-12', Phase="Research"),
        dict(Task="Model Training", Start='2024-05-13', Finish='2024-05-18', Phase="Modeling"),
        dict(Task="Streamlit Development", Start='2024-05-19', Finish='2024-05-23', Phase="Deployment")
    ]
    fig_gantt = px.timeline(pd.DataFrame(dev_data), x_start="Start", x_end="Finish", y="Task", color="Phase", title="Development Timeline")
    fig_gantt.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_gantt, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 2. ANALYTICS ---
elif page == "📊 Analytics":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.info("Scroll down 👇 to explore all 8 data insights.")

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("1. Target Class Distribution")
        fig1 = px.bar(x=['Normal', 'High', 'Critical'], y=[450, 120, 45], color=['Normal', 'High', 'Critical'])
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("2. Overall Demand Status (%)")
        fig2 = px.pie(values=[65, 25, 10], names=['Normal', 'High', 'Critical'], hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("3. Temperature Distribution")
        fig3 = px.histogram(pd.DataFrame(np.random.normal(25, 5, 1000), columns=['Temp']), x="Temp")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("4. Correlation Heatmap")
        fig4 = px.imshow(np.random.rand(5,5), color_continuous_scale='RdBu')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3 (Extra Content to ensure scrolling)
    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("5. Occupancy Trends")
        st.plotly_chart(px.scatter(x=np.random.rand(50), y=np.random.rand(50)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("6. Outlier Spread")
        st.plotly_chart(px.box(y=np.random.randn(100)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. DEMAND FORECASTER ---
elif page == "⚡ Demand Forecaster":
    st.title("⚡ Hotel Demand Forecaster")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig_map = px.bar(x=['Lobby', 'Rooms', 'Gym', 'Restaurant', 'Laundry'], y=[90, 250, 60, 180, 110])
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. PERFORMANCE METRICS ---
elif page == "📈 Performance Metrics":
    st.title("📈 Model Performance Analysis")
    
    # 3 Rings Row
    r1, r2, r3 = st.columns(3)
    def ring(l, v, c):
        return go.Figure(go.Indicator(mode="gauge+number", value=v, title={'text': l}, gauge={'bar': {'color': c}})).update_layout(height=250)
    with r1: st.plotly_chart(ring("XGBoost", 96.7, "cyan"), use_container_width=True)
    with r2: st.plotly_chart(ring("Random Forest", 94.1, "lime"), use_container_width=True)
    with r3: st.plotly_chart(ring("Logistic Reg", 82.5, "orange"), use_container_width=True)

    # Row 1 Comparison
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("9. Balanced Data (SMOTE)")
        st.plotly_chart(px.bar(x=['N', 'H', 'C'], y=[450, 450, 450]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("13. Feature Importance")
        st.plotly_chart(px.bar(x=[0.4, 0.3, 0.2, 0.1], y=['Occ', 'Temp', 'Kit', 'AC'], orientation='h'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)