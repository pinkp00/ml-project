import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# Custom CSS for Professional Card UI
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .plot-container {
        border-radius: 15px;
        background-color: #ffffff;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    h1, h2, h3 { color: #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏨 EnergySense AI")
    st.markdown("---")
    page = st.radio("Navigation", ["🏗️ Development Phase", "📊 Analytics", "⚡ Demand Forecaster", "📈 Performance Metrics"])
    st.markdown("---")
    active_model = st.selectbox("Active Model", ["XGBoost", "Random Forest", "Logistic Regression"])
    st.success(f"Running: {active_model}")

# --- HELPER FUNCTION FOR CARDS ---
def create_card(title):
    container = st.container()
    container.markdown(f"### {title}")
    return container

# --- 1. DEVELOPMENT PHASE ---
if page == "🏗️ Development Phase":
    st.title("🏗️ Project Roadmap")
    
    with st.container():
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        dev_data = [
            dict(Task="Dataset Acquisition", Start='2024-05-01', Finish='2024-05-03', Phase="Data"),
            dict(Task="Data Cleaning", Start='2024-05-04', Finish='2024-05-07', Phase="Preparation"),
            dict(Task="EDA (14 Charts)", Start='2024-05-08', Finish='2024-05-12', Phase="Research"),
            dict(Task="Model Training", Start='2024-05-13', Finish='2024-05-18', Phase="Modeling"),
            dict(Task="Streamlit Development", Start='2024-05-19', Finish='2024-05-23', Phase="Deployment")
        ]
        fig_gantt = px.timeline(pd.DataFrame(dev_data), x_start="Start", x_end="Finish", y="Task", color="Phase")
        fig_gantt.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_gantt, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 2. ANALYTICS (Multiple Scrollable Cards) ---
elif page == "📊 Analytics":
    st.title("📊 Detailed Hotel Analytics")
    
    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Feature Correlation")
        st.plotly_chart(px.imshow(np.random.rand(5,5), color_continuous_scale='Viridis'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Daily Energy Trend")
        st.plotly_chart(px.line(x=range(24), y=np.random.randint(40,100,24)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2 (More Charts for Scrolling)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Occupancy vs Energy")
        st.plotly_chart(px.scatter(x=np.random.rand(50), y=np.random.rand(50)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Temperature Impact")
        st.plotly_chart(px.histogram(np.random.randn(100)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. DEMAND FORECASTER ---
elif page == "⚡ Demand Forecaster":
    st.title("⚡ Hotel Demand Forecaster")
    
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("Zone-wise Market Demand")
    st.plotly_chart(px.bar(x=['Lobby', 'Rooms', 'Gym', 'Kitchen'], y=[80, 120, 60, 200], color=['Lobby', 'Rooms', 'Gym', 'Kitchen']), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Usage Breakdown")
        st.plotly_chart(px.pie(values=[40, 30, 20, 10], names=['AC', 'Lights', 'Kitchen', 'Laundry'], hole=0.4), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_p2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Peak Hour Analysis")
        st.plotly_chart(px.bar(x=range(12), y=np.random.randint(10,50,12)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 4. PERFORMANCE METRICS ---
elif page == "📈 Performance Metrics":
    st.title("📈 Model Comparison")
    
    # Large Doughnut Card
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig_d = go.Figure(data=[go.Pie(labels=['Accuracy', 'Error'], values=[96.72, 3.28], hole=.7)])
    fig_d.update_layout(annotations=[dict(text='96.72%', x=0.5, y=0.5, font_size=30, showarrow=False)])
    st.plotly_chart(fig_d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Comparison Rings
    r1, r2, r3 = st.columns(3)
    models = [("XGBoost", 96.7, "cyan"), ("Random Forest", 94.1, "lime"), ("Logistic Reg", 82.5, "orange")]
    
    for i, (name, acc, col) in enumerate(models):
        with [r1, r2, r3][i]:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(mode="gauge+number", value=acc, title={'text': name}, gauge={'bar': {'color': col}}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)