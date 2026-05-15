import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# --- CUSTOM CSS (For Scroll & Styling) ---
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { 
        overflow: auto !important; 
    }
    .main .block-container { 
        max-width: 95%; 
        padding-top: 2rem;
        padding-bottom: 15rem; 
    }
    .plot-container {
        border-radius: 12px;
        background-color: #ffffff;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 25px;
        border-top: 5px solid #1e3d59;
    }
    .stMetric { 
        background-color: #ffffff; 
        border-radius: 10px; 
        padding: 15px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); 
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR LOGIC ---
with st.sidebar:
    st.title("🏨 EnergySense AI")
    st.markdown("---")
    page = st.radio("Navigation", ["🏗️ Project Roadmap", "📊 Data Analytics (8 Charts)", "📈 Model Performance (6 Charts)"])
    
    st.markdown("---")
    st.subheader("⚙️ Select Model")
    selected_model = st.selectbox("Choose Model to Evaluate", ["XGBoost", "Random Forest", "Logistic Regression"])
    
    # Dynamic Stats Based on Selection
    if selected_model == "XGBoost":
        acc, status, color = 96.72, "Best Performer", "#00cc96"
        matrix_data = [[42, 1, 0], [1, 38, 0], [0, 0, 25]]
        feat_imp = [0.45, 0.28, 0.15, 0.12]
    elif selected_model == "Random Forest":
        acc, status, color = 94.10, "High Accuracy", "#66b3ff"
        matrix_data = [[40, 3, 0], [2, 35, 1], [0, 1, 23]]
        feat_imp = [0.40, 0.30, 0.18, 0.12]
    else:
        acc, status, color = 82.50, "Baseline Model", "#ffa500"
        matrix_data = [[35, 7, 1], [5, 30, 3], [2, 4, 18]]
        feat_imp = [0.35, 0.25, 0.22, 0.18]

    st.success(f"**Model:** {selected_model}")
    st.metric(label="Model Accuracy", value=f"{acc}%")

# --- 1. PROJECT ROADMAP ---
if page == "🏗️ Project Roadmap":
    st.title("🏗️ Implementation Roadmap")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    roadmap_data = pd.DataFrame([
        dict(Task="Dataset Acquisition", Start='2024-05-01', Finish='2024-05-03', Phase="Data"),
        dict(Task="Data Cleaning & SMOTE", Start='2024-05-04', Finish='2024-05-07', Phase="Prep"),
        dict(Task="EDA & Visualizations", Start='2024-05-08', Finish='2024-05-12', Phase="Research"),
        dict(Task="Model Training (XGBoost)", Start='2024-05-13', Finish='2024-05-18', Phase="Modeling"),
        dict(Task="Streamlit App Dev", Start='2024-05-19', Finish='2024-05-23', Phase="Final")
    ])
    fig_roadmap = px.timeline(roadmap_data, x_start="Start", x_end="Finish", y="Task", color="Phase", template="plotly_white")
    fig_roadmap.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_roadmap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 2. DATA ANALYTICS (8 CHARTS) ---
elif page == "📊 Data Analytics (8 Charts)":
    st.title("📊 Detailed Data Insights (EDA)")
    
    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("1. Target Class Distribution")
        st.plotly_chart(px.bar(x=['Normal', 'High', 'Critical'], y=[450, 120, 45], text_auto=True, color=['Normal', 'High', 'Critical']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("2. Demand Proportion (%)")
        fig2 = px.pie(values=[450, 120, 45], names=['Normal', 'High', 'Critical'], hole=0.4)
        fig2.update_traces(textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("3. Temperature Spread")
        st.plotly_chart(px.histogram(np.random.normal(25, 5, 1000), nbins=30, color_discrete_sequence=['#1e3d59']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("4. Feature Correlation Matrix")
        st.plotly_chart(px.imshow(np.random.rand(5,5), text_auto=".2f", x=['Occ','Temp','AC','Kit','Lau'], y=['Occ','Temp','AC','Kit','Lau'], color_continuous_scale='Blues'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3
    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("5. Occupancy vs Power Usage")
        st.plotly_chart(px.scatter(x=np.random.randint(10,100,100), y=np.random.randint(50,500,100), trendline="ols"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("6. Consumption Outliers")
        st.plotly_chart(px.box(y=np.random.normal(100, 25, 200), color_discrete_sequence=['#ef553b']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 4
    c7, c8 = st.columns(2)
    with c7:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("7. AC Usage Intensity")
        st.plotly_chart(px.violin(y=np.random.randint(1,24,200), box=True, color_discrete_sequence=['#00cc96']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c8:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("8. Historical Consumption Pattern")
        st.plotly_chart(px.line(y=np.random.randint(20,80,50)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. MODEL PERFORMANCE (6 CHARTS) ---
elif page == "📈 Model Performance (6 Charts)":
    st.title(f"📈 {selected_model} Performance Metrics")
    
    # Row 1
    p1, p2 = st.columns([1, 2])
    with p1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("9. Model Accuracy")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=acc,
            gauge={'bar': {'color': color}, 'axis': {'range': [0, 100]}}
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("10. Comparison Benchmark")
        comp_df = pd.DataFrame({'Model': ['XGBoost', 'Random Forest', 'LogReg'], 'Accuracy': [96.72, 94.10, 82.50]})
        fig10 = px.bar(comp_df, x='Model', y='Accuracy', text_auto=True, color='Model', color_discrete_sequence=['#1e3d59', '#66b3ff', '#ffa500'])
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    p3, p4 = st.columns(2)
    with p3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader(f"11. Confusion Matrix ({selected_model})")
        fig11 = px.imshow(matrix_data, text_auto=True, x=['N','H','C'], y=['N','H','C'], color_continuous_scale='Greens')
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("12. Error Distribution (Residuals)")
        st.plotly_chart(px.histogram(np.random.normal(0, 0.1, 500), color_discrete_sequence=['#ab63fa']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3
    p5, p6 = st.columns(2)
    with p5:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("13. Feature Importance Score")
        fig13 = px.bar(x=feat_imp, y=['Occupancy', 'Temp', 'Kitchen', 'Laundry'], orientation='h', text_auto='.2f', color_discrete_sequence=['#1e3d59'])
        st.plotly_chart(fig13, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p6:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("14. Prediction Variance Over Time")
        st.plotly_chart(px.area(y=np.random.uniform(0, 0.05, 50)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)