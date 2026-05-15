import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="EnergySense AI - Hotel Dashboard", layout="wide")

# Custom CSS for UI
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #0e1117; color: white; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🏨 EnergySense AI")
    st.markdown("---")
    page = st.radio("Navigation", ["📊 Analytics", "⚡ Demand Forecaster", "🏗️ Development Phase", "📈 Model Performance"])
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    active_model = st.selectbox("Active Model", ["XGBoost (Recommended)", "Random Forest", "Logistic Regression"])
    st.info(f"Current Model: {active_model}")

# --- 1. ANALYTICS PAGE ---
if page == "📊 Analytics":
    st.title("📊 Hotel Energy Analytics")
    st.write("Exploratory Data Analysis (EDA) Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        # Example Heatmap logic
        df_corr = pd.DataFrame(pd.np.random.rand(5, 5), columns=['Temp', 'Occupancy', 'Humidity', 'AC_Load', 'Kitchen'])
        fig_heat = px.imshow(df_corr, text_auto=True, title="Feature Correlation Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with col2:
        # Energy Trend
        df_trend = pd.DataFrame({'Day': range(1, 31), 'Demand': pd.np.random.randint(50, 200, 30)})
        fig_trend = px.line(df_trend, x='Day', y='Demand', title="Monthly Energy Demand Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

# --- 2. DEMAND FORECASTER PAGE ---
elif page == "⚡ Demand Forecaster":
    st.title("⚡ Market Demand Forecaster")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        # Map or Distribution
        st.subheader("Demand Market Map")
        df_map = pd.DataFrame({'Location': ['North Wing', 'South Wing', 'Lobby', 'Kitchen', 'Pool'],
                              'Usage': [450, 300, 150, 600, 200]})
        fig_bar = px.bar(df_map, x='Location', y='Usage', color='Usage', title="Area-wise Energy Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_b:
        st.subheader("Demand Split")
        fig_pie = px.pie(values=[40, 30, 20, 10], names=['AC', 'Lighting', 'Heating', 'Others'], hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- 3. DEVELOPMENT PHASE (Gantt Chart) ---
elif page == "🏗️ Development Phase":
    st.title("🏗️ Project Development Roadmap")
    
    # Development Data
    tasks = [
        dict(Task="Dataset Collection", Start='2024-05-01', Finish='2024-05-05', Resource="ChatGPT/Kaggle"),
        dict(Task="Data Cleaning", Start='2024-05-06', Finish='2024-05-10', Resource="Python/Pandas"),
        dict(Task="EDA & Visuals", Start='2024-05-11', Finish='2024-05-15', Resource="Plotly"),
        dict(Task="Model Training", Start='2024-05-16', Finish='2024-05-20', Resource="XGBoost/RF"),
        dict(Task="Deployment", Start='2024-05-21', Finish='2024-05-25', Resource="Streamlit")
    ]
    df_gantt = pd.DataFrame(tasks)
    
    fig_gantt = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Resource", 
                           title="Project Timeline (dd/mm/yyyy format)",
                           hover_data={'Start':True, 'Finish':True})
    fig_gantt.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_gantt, use_container_width=True)
    
    st.table(df_gantt) # Clear table for dates

# --- 4. MODEL PERFORMANCE (Colourful Rings) ---
elif page == "📈 Model Performance":
    st.title("📈 Model Comparison & Accuracy")
    
    # Performance Doughnut
    st.subheader("Overall Model Stability")
    fig_doughnut = go.Figure(data=[go.Pie(labels=['Accuracy', 'Error', 'Uncertainty'], values=[96.72, 2.28, 1], hole=.6)])
    fig_doughnut.update_layout(annotations=[dict(text='96.72%', x=0.5, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig_doughnut)

    st.markdown("---")
    st.subheader("Model Accuracy Rings")
    
    # Colourful Rings for 3 models
    c1, c2, c3 = st.columns(3)
    
    def draw_ring(label, value, color):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            title = {'text': label},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': color},
                     'bgcolor': "white",
                     'borderwidth': 2,
                     'bordercolor': "gray"}
        ))
        fig.update_layout(height=250)
        return fig

    with c1:
        st.plotly_chart(draw_ring("XGBoost", 96.7, "cyan"), use_container_width=True)
    with c2:
        st.plotly_chart(draw_ring("Random Forest", 94.1, "lime"), use_container_width=True)
    with c3:
        st.plotly_chart(draw_ring("Logistic Regression", 82.5, "orange"), use_container_width=True)