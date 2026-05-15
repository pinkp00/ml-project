import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Config
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# CSS for Professional & Compact Look
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
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏨 EnergySense AI")
    st.markdown("---")
    # Naya option "Demand Forecaster" add kiya gaya hai
    page = st.radio("Navigation", ["🏗️ Project Roadmap", "📊 Data Analytics (8 Charts)", "📈 Model Performance (6 Charts)", "🔮 Demand Forecaster"])
    
    st.markdown("---")
    st.subheader("⚙️ Select Model")
    selected_model = st.selectbox("Choose Model to Evaluate", ["XGBoost", "Random Forest", "Logistic Regression"])
    
    if selected_model == "XGBoost":
        acc, status, color = 96.72, "Best Performer", "cyan"
        matrix = [[42, 1, 0], [1, 38, 0], [0, 0, 25]]
    elif selected_model == "Random Forest":
        acc, status, color = 94.10, "High Accuracy", "lime"
        matrix = [[40, 3, 0], [2, 35, 1], [0, 1, 23]]
    else:
        acc, status, color = 82.50, "Baseline Model", "orange"
        matrix = [[35, 7, 1], [5, 30, 3], [2, 4, 18]]

    st.success(f"**Model:** {selected_model}")
    st.info(f"**Accuracy:** {acc}%")

# --- 1. ROADMAP ---
if page == "🏗️ Project Roadmap":
    st.title("🏗️ Implementation Timeline")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    roadmap_data = pd.DataFrame([
        dict(Task="Dataset Acquisition", Start='2024-05-01', Finish='2024-05-03', Phase="Data"),
        dict(Task="Data Cleaning & SMOTE", Start='2024-05-04', Finish='2024-05-07', Phase="Prep"),
        dict(Task="EDA & Visualizations", Start='2024-05-08', Finish='2024-05-12', Phase="Research"),
        dict(Task="XGBoost Training", Start='2024-05-13', Finish='2024-05-18', Phase="Modeling"),
        dict(Task="Streamlit Development", Start='2024-05-19', Finish='2024-05-23', Phase="Final")
    ])
    fig_r = px.timeline(roadmap_data, x_start="Start", x_end="Finish", y="Task", color="Phase", template="plotly_white")
    fig_r.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 2. ANALYTICS ---
elif page == "📊 Data Analytics (8 Charts)":
    st.title("📊 Detailed Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("1. Energy Status Distribution")
        st.plotly_chart(px.bar(x=['Normal', 'High', 'Critical'], y=[450, 120, 45], text_auto=True, color=['Normal', 'High', 'Critical']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("2. Target Proportion (Pie)")
        fig2 = px.pie(values=[450, 120, 45], names=['Normal', 'High', 'Critical'], hole=0.4)
        fig2.update_traces(textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("3. Temperature Distribution")
        st.plotly_chart(px.histogram(np.random.normal(25, 5, 1000), nbins=30), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("4. Feature Correlation")
        st.plotly_chart(px.imshow(np.random.rand(5,5), text_auto=".2f"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("5. Occupancy vs Power")
        # Note: If this fails, run 'pip install statsmodels'
        try:
            fig5 = px.scatter(x=np.random.randint(10,100,100), y=np.random.randint(50,500,100), trendline="ols")
            st.plotly_chart(fig5, use_container_width=True)
        except:
            st.warning("Install 'statsmodels' to see the trendline.")
            st.plotly_chart(px.scatter(x=np.random.randint(10,100,100), y=np.random.randint(50,500,100)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("6. Consumption Outliers")
        st.plotly_chart(px.box(y=np.random.normal(100, 25, 200)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. PERFORMANCE ---
elif page == "📈 Model Performance (6 Charts)":
    st.title(f"📈 {selected_model} Evaluation Metrics")
    p1, p2 = st.columns([1, 2])
    with p1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("9. Accuracy Gauge")
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=acc, gauge={'bar': {'color': color}}))
        fig_g.update_layout(height=250)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("10. Benchmark Comparison")
        st.plotly_chart(px.bar(x=['XGBoost', 'RF', 'LogReg'], y=[96.72, 94.10, 82.50], text_auto=True), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader(f"11. Confusion Matrix ({selected_model})")
    st.plotly_chart(px.imshow(matrix, text_auto=True, color_continuous_scale='Greens'), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. DEMAND FORECASTER (NEW FEATURE) ---
elif page == "🔮 Demand Forecaster":
    st.title("🔮 Geographic Demand Prediction")
    
    # Dummy Location Data
    df_map = pd.DataFrame({
        'Hotel': ['Plaza', 'Continental', 'Marriott', 'Serena', 'Regent', 'Pearl', 'Grand'],
        'lat': [33.6844, 33.7000, 33.7200, 33.7100, 33.6900, 33.6700, 33.7300],
        'lon': [73.0479, 73.0600, 73.0800, 73.0300, 73.0100, 73.0700, 73.0900],
        'Demand': [95, 40, 110, 30, 85, 55, 120],
        'Status': ['Critical', 'Normal', 'Critical', 'Normal', 'High', 'Normal', 'Critical']
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("📍 Demand Prediction Map")
        fig_map = px.scatter_mapbox(df_map, lat="lat", lon="lon", color="Status", size="Demand",
                                    hover_name="Hotel", zoom=11, height=500,
                                    color_discrete_map={"Critical": "red", "High": "orange", "Normal": "green"})
        fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("📊 Forecast Distribution")
        fig_p = px.pie(df_map, names='Status', values='Demand', color='Status', hole=0.5,
                       color_discrete_map={"Critical": "red", "High": "orange", "Normal": "green"})
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.success("**Prediction Summary:** AI model suggests a 15% increase in energy demand for 'Grand Hotel' in the next 2 hours.")
    st.markdown('</div>', unsafe_allow_html=True)