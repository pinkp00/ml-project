import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Config
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# Custom CSS for a clean look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏨 EnergySense AI")
    st.markdown("---")
    page = st.radio("Navigation", ["🏗️ Development Phase", "📊 Analytics", "⚡ Demand Forecaster", "📈 Performance Metrics"])
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    active_model = st.selectbox("Active Model", ["XGBoost", "Random Forest", "Logistic Regression"])
    st.success(f"Running: {active_model}")

# --- 1. DEVELOPMENT PHASE ---
if page == "🏗️ Development Phase":
    st.title("🏗️ Project Roadmap & Development Phase")
    st.info("Tracking progress from Data Collection to Deployment")

    dev_data = [
        dict(Task="Dataset Acquisition", Start='2024-05-01', Finish='2024-05-03', Phase="Data"),
        dict(Task="Data Cleaning", Start='2024-05-04', Finish='2024-05-07', Phase="Preparation"),
        dict(Task="EDA (14 Charts)", Start='2024-05-08', Finish='2024-05-12', Phase="Research"),
        dict(Task="Model Training", Start='2024-05-13', Finish='2024-05-18', Phase="Modeling"),
        dict(Task="Streamlit Development", Start='2024-05-19', Finish='2024-05-23', Phase="Deployment"),
        dict(Task="Final Testing", Start='2024-05-24', Finish='2024-05-25', Phase="Final")
    ]
    df_gantt = pd.DataFrame(dev_data)
    
    fig_gantt = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Phase",
                           title="Project Progress Timeline",
                           color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_gantt.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_gantt, use_container_width=True)

# --- 2. ANALYTICS ---
elif page == "📊 Analytics":
    st.title("📊 Hotel Energy Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        corr_data = np.random.rand(5,5)
        fig_heat = px.imshow(corr_data, 
                             x=['Temp', 'Occupancy', 'AC', 'Kitchen', 'Laundry'],
                             y=['Temp', 'Occupancy', 'AC', 'Kitchen', 'Laundry'],
                             title="Feature Correlation Heatmap", color_continuous_scale='Viridis')
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col2:
        df_line = pd.DataFrame({'Hour': range(24), 'Consumption': np.random.randint(40, 100, 24)})
        fig_line = px.area(df_line, x='Hour', y='Consumption', title="Hourly Energy Usage Trend")
        st.plotly_chart(fig_line, use_container_width=True)

# --- 3. DEMAND FORECASTER ---
elif page == "⚡ Demand Forecaster":
    st.title("⚡ Hotel Demand Forecaster")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        zones = ['North Wing', 'South Wing', 'Restaurant', 'Gym', 'Lobby']
        demand = [85, 92, 78, 65, 88]
        fig_market = px.bar(x=zones, y=demand, color=zones, title="Demand by Hotel Zone")
        st.plotly_chart(fig_market, use_container_width=True)
        
    with col_b:
        fig_pie = px.pie(values=[45, 25, 20, 10], names=['Cooling', 'Heating', 'Lighting', 'Appliances'], hole=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- 4. PERFORMANCE METRICS ---
elif page == "📈 Performance Metrics":
    st.title("📈 Model Performance Analysis")
    
    fig_doughnut = go.Figure(data=[go.Pie(labels=['Accuracy', 'Error Rate'], values=[96.72, 3.28], hole=.7)])
    fig_doughnut.update_layout(annotations=[dict(text='96.72%', x=0.5, y=0.5, font_size=30, showarrow=False)])
    st.plotly_chart(fig_doughnut)

    st.markdown("---")
    ring_col1, ring_col2, ring_col3 = st.columns(3)
    
    def create_ring(label, value, color):
        return go.Figure(go.Indicator(
            mode = "gauge+number", value = value,
            title = {'text': label},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}}
        )).update_layout(height=300)

    with ring_col1:
        st.plotly_chart(create_ring("XGBoost", 96.7, "cyan"), use_container_width=True)
    with ring_col2:
        st.plotly_chart(create_ring("Random Forest", 94.1, "lime"), use_container_width=True)
    with ring_col3:
        st.plotly_chart(create_ring("Log. Regression", 82.5, "orange"), use_container_width=True)