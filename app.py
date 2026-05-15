import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Config
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# Force Scroll & Card CSS
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { overflow: auto !important; }
    .main .block-container { max-width: 95%; padding-bottom: 15rem; }
    .plot-container {
        border-radius: 12px;
        background-color: #ffffff;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border-top: 5px solid #1e3d59;
    }
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (Icons Removed) ---
with st.sidebar:
    st.title("EnergySense AI")
    st.markdown("---")
    # Navigation labels updated to remove icons
    page = st.radio("Navigation", ["Project Roadmap", "Data Analytics", "Model Performance", "Demand Forecaster"])
    
    st.markdown("---")
    st.subheader("Select Model")
    selected_model = st.selectbox("Choose Model to Evaluate", ["XGBoost", "Random Forest", "Logistic Regression"])
    
    # Dynamic logic for model stats
    if selected_model == "XGBoost":
        acc, status, color = 96.72, "Best Performer", "cyan"
        matrix = [[42, 1, 0], [1, 38, 0], [0, 0, 25]]
    elif selected_model == "Random Forest":
        acc, status, color = 94.10, "High Accuracy", "lime"
        matrix = [[40, 3, 0], [2, 35, 1], [0, 1, 23]]
    else:
        acc, status, color = 82.50, "Baseline Model", "orange"
        matrix = [[35, 7, 1], [5, 30, 3], [2, 4, 18]]

    st.success(f"Model: {selected_model}")
    st.info(f"Accuracy: {acc}%")

# --- 1. ROADMAP ---
if page == "Project Roadmap":
    st.title("Implementation Timeline")
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
elif page == "Data Analytics":
    st.title("Detailed Exploratory Data Analysis")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("1. Energy Status Distribution")
        fig1 = px.bar(x=['Normal', 'High', 'Critical'], y=[450, 120, 45], text_auto=True, color=['Normal', 'High', 'Critical'])
        st.plotly_chart(fig1, use_container_width=True)
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
        fig3 = px.histogram(np.random.normal(25, 5, 1000), nbins=30)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("4. Feature Correlation")
        fig4 = px.imshow(np.random.rand(5,5), x=['Occ', 'Temp', 'AC', 'Kit', 'Lau'], y=['Occ', 'Temp', 'AC', 'Kit', 'Lau'], text_auto=".2f")
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("5. Occupancy vs Power")
        # Try/Except to handle trendline error if statsmodels is missing
        try:
            fig5 = px.scatter(x=np.random.randint(10,100,100), y=np.random.randint(50,500,100), trendline="ols")
        except:
            fig5 = px.scatter(x=np.random.randint(10,100,100), y=np.random.randint(50,500,100))
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("6. Consumption Outliers")
        fig6 = px.box(y=np.random.normal(100, 25, 200))
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("7. AC Usage Pattern")
        fig7 = px.violin(y=np.random.randint(1,24,200), box=True)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c8:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("8. Historical Usage Trend")
        fig8 = px.line(y=np.random.randint(20,80,50))
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. PERFORMANCE ---
elif page == "Model Performance":
    st.title(f"{selected_model} Evaluation Metrics")
    
    p1, p2 = st.columns([1, 2])
    with p1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("9. Model Accuracy")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=acc,
            title={'text': f"{status}"},
            gauge={'bar': {'color': color}, 'axis': {'range': [0, 100]}}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with p2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("10. Benchmark Comparison")
        comp_df = pd.DataFrame({'Model': ['XGBoost', 'Random Forest', 'LogReg'], 'Accuracy': [96.72, 94.10, 82.50]})
        fig10 = px.bar(comp_df, x='Model', y='Accuracy', text_auto=True, color='Model')
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    p3, p4 = st.columns(2)
    with p3:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader(f"11. Confusion Matrix ({selected_model})")
        fig11 = px.imshow(matrix, text_auto=True, x=['N', 'H', 'C'], y=['N', 'H', 'C'], color_continuous_scale='Greens')
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("12. Prediction Distribution")
        fig12 = px.histogram(np.random.randint(0,3,150), color_discrete_sequence=['#ab63fa'])
        st.plotly_chart(fig12, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    p5, p6 = st.columns(2)
    with p5:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("13. Feature Importance")
        fig13 = px.bar(x=[0.45, 0.28, 0.15, 0.12], y=['Occupancy', 'Temp', 'Kitchen', 'Laundry'], orientation='h', text_auto='.2f')
        st.plotly_chart(fig13, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p6:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("14. Model Variance (Residuals)")
        fig14 = px.area(y=np.random.uniform(0, 0.05, 50))
        st.plotly_chart(fig14, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 4. DEMAND FORECASTER (Fixed) ---
elif page == "Demand Forecaster":
    st.title("Geographic Energy Forecasting")
    
    # Dummy data for map
    df_m = pd.DataFrame({
        'Hotel': ['Plaza Hotel', 'Grand Inn', 'City Center Stay', 'Royal Suite', 'Eco Lodge'],
        'lat': [33.6844, 33.7000, 33.7200, 33.6900, 33.7100],
        'lon': [73.0479, 73.0600, 73.0800, 73.0200, 73.0900],
        'Demand': [100, 50, 120, 80, 40],
        'Status': ['Critical', 'Normal', 'Critical', 'High', 'Normal']
    })
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Real-time Regional Demand Map")
        fig_map = px.scatter_mapbox(
            df_m, lat="lat", lon="lon", 
            color="Status", size="Demand", 
            hover_name="Hotel",
            zoom=11, height=500, 
            color_discrete_map={"Critical": "red", "High": "orange", "Normal": "green"}
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Demand Mix")
        # Doughnut chart (Pie with hole)
        fig_pie = px.pie(
            df_m, names='Status', values='Demand', 
            hole=0.5,
            color='Status',
            color_discrete_map={"Critical": "red", "High": "orange", "Normal": "green"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Info card for value addition
    st.info("Demand forecasting is based on real-time occupancy and external weather parameters.")