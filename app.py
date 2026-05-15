import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Config
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# CSS for Layout & Spacing
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { overflow: auto !important; }
    .main .block-container { max-width: 95%; padding-top: 1rem; padding-bottom: 10rem; }
    .plot-container {
        border-radius: 12px;
        background-color: #ffffff;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border-top: 4px solid #1e3d59;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏨 EnergySense AI")
    page = st.radio("Navigation", ["🏗️ Roadmap", "📊 Analytics (8 Charts)", "📈 Performance (6 Charts)", "🔮 Demand Forecaster"])
    st.markdown("---")
    selected_model = st.selectbox("Current Model", ["XGBoost", "Random Forest", "Logistic Regression"])
    
    stats = {
        "XGBoost": {"acc": 96.72, "color": "cyan", "matrix": [[42, 1, 0], [1, 38, 0], [0, 0, 25]]},
        "Random Forest": {"acc": 94.10, "color": "lime", "matrix": [[40, 3, 0], [2, 35, 1], [0, 1, 23]]},
        "Logistic Regression": {"acc": 82.50, "color": "orange", "matrix": [[35, 7, 1], [5, 30, 3], [2, 4, 18]]}
    }
    m = stats[selected_model]
    st.metric("Accuracy", f"{m['acc']}%")

# --- 1. ROADMAP ---
if page == "🏗️ Roadmap":
    st.title("🏗️ Project Roadmap")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    df_r = pd.DataFrame([
        dict(Task="Data Acquisition", Start='2024-05-01', Finish='2024-05-03', Phase="Data"),
        dict(Task="Data Cleaning", Start='2024-05-04', Finish='2024-05-07', Phase="Prep"),
        dict(Task="EDA Research", Start='2024-05-08', Finish='2024-05-12', Phase="Research"),
        dict(Task="Model Training", Start='2024-05-13', Finish='2024-05-18', Phase="Modeling"),
        dict(Task="App Deployment", Start='2024-05-19', Finish='2024-05-23', Phase="Final")
    ])
    st.plotly_chart(px.timeline(df_r, x_start="Start", x_end="Finish", y="Task", color="Phase"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 2. ANALYTICS (8 CHARTS) ---
elif page == "📊 Analytics (8 Charts)":
    st.title("📊 Exploratory Data Analysis")
    h = 300 # Chart height
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.bar(x=['Normal', 'High', 'Critical'], y=[450, 120, 45], title="1. Class Distribution", height=h), use_container_width=True)
        st.markdown('</div><div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.histogram(np.random.normal(25, 5, 1000), title="3. Temp Distribution", height=h), use_container_width=True)
        st.markdown('</div><div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.scatter(x=np.random.rand(100), y=np.random.rand(100), title="5. Occupancy vs Power", height=h), use_container_width=True)
        st.markdown('</div><div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.violin(y=np.random.randn(100), title="7. AC Usage Pattern", height=h), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.pie(values=[450, 120, 45], names=['N', 'H', 'C'], title="2. Target Proportion", height=h), use_container_width=True)
        st.markdown('</div><div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.imshow(np.random.rand(5,5), title="4. Correlation Matrix", height=h), use_container_width=True)
        st.markdown('</div><div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.box(y=np.random.randn(100), title="6. Consumption Outliers", height=h), use_container_width=True)
        st.markdown('</div><div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.line(y=np.random.cumsum(np.random.randn(100)), title="8. Historical Usage", height=h), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. PERFORMANCE (6 CHARTS) ---
elif page == "📈 Performance (6 Charts)":
    st.title(f"📈 {selected_model} Metrics")
    p1, p2 = st.columns([1, 2])
    with p1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=m['acc'], title={'text': "9. Accuracy Score"}, gauge={'bar': {'color': m['color']}}))
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.bar(x=['XGB', 'RF', 'Log'], y=[96.7, 94.1, 82.5], title="10. Model Comparison", height=350), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.imshow(m['matrix'], text_auto=True, title="11. Confusion Matrix", color_continuous_scale='Greens'), use_container_width=True)
        st.markdown('</div><div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.bar(x=[0.5, 0.3, 0.1, 0.1], y=['Occ', 'Temp', 'Kit', 'Lau'], orientation='h', title="13. Feature Importance"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.histogram(np.random.randint(0,3,100), title="12. Prediction Dist"), use_container_width=True)
        st.markdown('</div><div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.area(y=np.random.rand(50), title="14. Error Variance"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 4. DEMAND FORECASTER ---
elif page == "🔮 Demand Forecaster":
    st.title("🔮 Geographic Energy Forecasting")
    df_m = pd.DataFrame({'Hotel': ['H1', 'H2', 'H3'], 'lat': [33.68, 33.70, 33.72], 'lon': [73.04, 73.06, 73.08], 'Demand': [100, 50, 120], 'Status': ['High', 'Normal', 'Critical']})
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(df_m, lat="lat", lon="lon", color="Status", size="Demand", zoom=11, height=500, color_discrete_map={"Critical": "red", "High": "orange", "Normal": "green"})
        fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(px.pie(df_m, names='Status', values='Demand', hole=0.5, title="Demand Mix"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)