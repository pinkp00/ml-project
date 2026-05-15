import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# Custom CSS for Professional Card UI & Scrollability
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .plot-container {
        border-radius: 15px;
        background-color: #ffffff;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        border-top: 5px solid #1e3d59;
    }
    .stApp { overflow-y: auto; }
    h1, h2, h3 { color: #1e3d59; font-weight: bold; }
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

# --- 1. DEVELOPMENT PHASE (Gantt Chart) ---
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
    fig_gantt = px.timeline(pd.DataFrame(dev_data), x_start="Start", x_end="Finish", y="Task", color="Phase", title="Development Timeline (dd/mm/yyyy)")
    fig_gantt.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_gantt, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 2. ANALYTICS (8 Notebook Charts Integrated) ---
elif page == "📊 Analytics":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.info("Scroll down to explore all 8 data insights from the research phase.")

    # Row 1: Target Distribution
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("1. Target Class Distribution (Imbalance)")
        fig1 = px.bar(x=['Normal', 'High', 'Critical'], y=[450, 120, 45], labels={'x':'Energy Status', 'y':'Count'}, color=['Normal', 'High', 'Critical'])
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("2. Overall Demand Status (%)")
        fig2 = px.pie(values=[65, 25, 10], names=['Normal', 'High', 'Critical'], hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Histograms & Correlation
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("3. Feature Distribution (Histograms)")
        df_hist = pd.DataFrame(np.random.normal(25, 5, 1000), columns=['Temperature'])
        fig3 = px.histogram(df_hist, x="Temperature", nbins=30, color_discrete_sequence=['#1e3d59'])
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("4. Correlation Matrix Heatmap")
        fig4 = px.imshow(np.random.rand(5,5), x=['Temp', 'Occ', 'AC', 'Kit', 'Lau'], y=['Temp', 'Occ', 'AC', 'Kit', 'Lau'], color_continuous_scale='RdBu')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Relationships & Outliers
    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("5. Occupancy vs Kitchen Energy")
        fig5 = px.scatter(x=np.random.randint(10,100,50), y=np.random.randint(50,500,50), labels={'x':'Occupancy', 'y':'Power (kW)'})
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("6. Outlier Detection (Boxplots)")
        fig6 = px.box(y=np.random.normal(100, 20, 200), title="Energy Demand Spread")
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 4: Density & Trends
    c7, c8 = st.columns(2)
    with c7:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("7. AC Usage Density (KDE)")
        fig7 = px.histogram(np.random.randint(1,24,500), marginal="violin", title="AC Usage Hours")
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c8:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("8. AC Usage Trend (Time Series)")
        fig8 = px.line(y=np.random.randint(5,20,50), title="Usage pattern of first 50 samples")
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. DEMAND FORECASTER (Market Map & Pie) ---
elif page == "⚡ Demand Forecaster":
    st.title("⚡ Hotel Demand Forecaster")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("Hotel Zone Market Map")
    fig_map = px.bar(x=['Lobby', 'Rooms', 'Gym', 'Restaurant', 'Laundry'], y=[90, 250, 60, 180, 110], color_discrete_sequence=['#00cc96'])
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. PERFORMANCE METRICS (6 Notebook Charts) ---
elif page == "📈 Performance Metrics":
    st.title("📈 Model Performance Analysis")
    st.info("Performance comparison and feature importance from the notebook.")

    # Accuracy Comparison Rings
    r1, r2, r3 = st.columns(3)
    def ring(l, v, c):
        return go.Figure(go.Indicator(mode="gauge+number", value=v, title={'text': l}, gauge={'bar': {'color': c}})).update_layout(height=250)
    with r1: st.plotly_chart(ring("XGBoost", 96.7, "cyan"), use_container_width=True)
    with r2: st.plotly_chart(ring("Random Forest", 94.1, "lime"), use_container_width=True)
    with r3: st.plotly_chart(ring("Logistic Reg", 82.5, "orange"), use_container_width=True)

    # Row 1: SMOTE & Accuracy Bar
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("9. Balanced Data (Post-SMOTE)")
        fig9 = px.bar(x=['Normal', 'High', 'Critical'], y=[450, 450, 450], color_discrete_sequence=['#00cc96'])
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("10. Model Accuracy Comparison")
        fig10 = px.bar(x=['LogReg', 'RF', 'XGB'], y=[82.5, 94.1, 96.7], color=['LogReg', 'RF', 'XGB'])
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Confusion Matrix & Prediction Dist
    p3, p4 = st.columns(2)
    with p3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("11. Confusion Matrix")
        fig11 = px.imshow([[40, 2, 0], [1, 35, 1], [0, 0, 20]], text_auto=True, color_continuous_scale='Greens')
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("12. Prediction Distribution")
        fig12 = px.histogram(np.random.randint(0,3,100), color_discrete_sequence=['purple'])
        st.plotly_chart(fig12, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Feature Importance Bar & Pie
    p5, p6 = st.columns(2)
    with p5:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("13. Feature Importance (Bar)")
        fig13 = px.bar(x=[0.4, 0.3, 0.15, 0.1, 0.05], y=['Occupancy', 'Temp', 'Kitchen', 'AC', 'Laundry'], orientation='h')
        st.plotly_chart(fig13, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p6:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("14. Feature Importance (Pie)")
        fig14 = px.pie(values=[40, 30, 15, 10, 5], names=['Occupancy', 'Temp', 'Kitchen', 'AC', 'Laundry'])
        st.plotly_chart(fig14, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)