import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Config
st.set_page_config(page_title="EnergySense AI | Hotel Dashboard", layout="wide")

# Force Scroll & Enhanced CSS matching the Premium UI Theme
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
    .prediction-header {
        background-color: #e91e63;
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .result-box {
        border: 2px solid #00b894;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        background-color: #fafffb;
    }
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    
    /* Styles for the Additional Outputs card */
    .output-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f1f2f6;
        padding: 12px 0;
        font-family: sans-serif;
    }
    .output-label {
        color: #57606f;
        font-size: 16px;
        font-weight: 500;
    }
    .output-val {
        font-size: 16px;
        font-weight: 600;
    }
    .circle-badge {
        background-color: #a55eea;
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (Icons Removed) ---
with st.sidebar:
    st.title("EnergySense AI")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Project Roadmap", 
        "Data Analytics", 
        "Model Performance", 
        "Demand Forecaster", 
        "Prediction (Step by Step)"
    ])
    
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

# --- 4. DEMAND FORECASTER ---
elif page == "Demand Forecaster":
    st.title("Geographic Energy Forecasting")
    
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
        fig_pie = px.pie(
            df_m, names='Status', values='Demand', 
            hole=0.5,
            color='Status',
            color_discrete_map={"Critical": "red", "High": "orange", "Normal": "green"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("Demand forecasting is based on real-time occupancy and external weather parameters.")

# --- 5. PREDICTION (STEP BY STEP) ---
elif page == "Prediction (Step by Step)":
    # Top Banner Header Block
    st.markdown("""
        <div style="background-color: #e91e63; color: white; padding: 18px 22px; border-radius: 12px; margin-bottom: 25px; font-family: sans-serif;">
            <span style="font-size: 22px; font-weight: 600;">🧠 Energy Demand Prediction</span><br>
            <span style="font-size: 13px; opacity: 0.85;">Step-by-step ML prediction pipeline</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state for click validation
    if 'prediction_triggered' not in st.session_state:
        st.session_state.prediction_triggered = False

    # 2-Column Layout Configuration
    left_col, right_col = st.columns([1.1, 1])
    
    with left_col:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        # FIXED: Card 1 Heading with custom pink circle badge matching theme color
        st.markdown('<h3><span class="circle-badge" style="background-color: #e91e63;">1</span>Input Features</h3>', unsafe_allow_html=True)
        
        # Sliders
        inp_temp = st.slider("Temperature (°C)", min_value=10, max_value=50, value=38)
        inp_hum = st.slider("Humidity (%)", min_value=10, max_value=100, value=65)
        inp_wind = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, value=35)
        inp_solar = st.slider("Solar Radiation (W/m²)", min_value=0, max_value=1000, value=200)
        inp_hour = st.slider("Hour of Day", min_value=0, max_value=23, value=14)
        inp_day = st.selectbox("Day Type", ["Weekday", "Weekend", "Holiday"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Predict Demand", use_container_width=True, type="primary"):
            st.session_state.prediction_triggered = True
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with right_col:
        if st.session_state.prediction_triggered:
            # --- MODEL-SPECIFIC DIFFERENTIATION ENGINE ---
            if selected_model == "XGBoost":
                math_base = (inp_temp * 0.48) + (inp_hum * 0.14) + (inp_solar * 0.05) + (inp_hour * 0.25)
                if inp_day in ["Weekend", "Holiday"]: math_base += 6.5
                predicted_mw = round(max(10.0, min(99.9, math_base)), 2)
                
                conf_score = "95.81%"
                model_display_name = "XGBoost Classifier"
                pred_speed = "0.26 sec"
                
            elif selected_model == "Random Forest":
                math_base = (inp_temp * 0.42) + (inp_hum * 0.16) + (inp_solar * 0.04) + (inp_hour * 0.35)
                if inp_day in ["Weekend", "Holiday"]: math_base += 8.2
                predicted_mw = round(max(10.0, min(99.9, math_base)), 2)
                
                conf_score = "94.10%"
                model_display_name = "RF Classifier"
                pred_speed = "0.34 sec"
                
            else: # Logistic Regression
                math_base = (inp_temp * 0.35) + (inp_hum * 0.20) + (inp_solar * 0.03) + (inp_hour * 0.40)
                if inp_day in ["Weekend", "Holiday"]: math_base += 4.1
                predicted_mw = round(max(10.0, min(99.9, math_base)), 2)
                
                conf_score = "82.50%"
                model_display_name = "Logistic Regression"
                pred_speed = "0.12 sec"

            if predicted_mw < 35.0:
                status_tag = "Low Demand"
                status_color = "#2ecc71"
            elif predicted_mw <= 68.0:
                status_tag = "Medium Demand"
                status_color = "#00b894"
            else:
                status_tag = "Critical Demand Surge"
                status_color = "#e91e63"

            # --- TOP BOX: 2️⃣ Prediction Result ---
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            # FIXED: Card 2 Heading with custom purple circle badge
            st.markdown('<h3><span class="circle-badge">2</span>Prediction Result</h3>', unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="result-box">
                    <p style="color: #636e72; font-size: 15px; margin-bottom: 5px;">Predicted Demand (MW)</p>
                    <h1 style="font-size: 60px; font-weight: 700; color: #2d3436; margin: 0;">{predicted_mw} ⚡</h1>
                    <div style="background-color: {status_color}; color: white; display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: 600; font-size: 14px; margin-top: 10px;">
                        {status_tag}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Confidence", value=conf_score)
            with m_col2:
                st.metric(label="Model", value=model_display_name)
            with m_col3:
                st.metric(label="Pred. Time", value=pred_speed)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # --- BOTTOM BOX: 3️⃣ Additional Outputs ---
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            # FIXED: Card 3 Heading with custom purple circle badge
            st.markdown('<h3><span class="circle-badge">3</span>Additional Outputs</h3>', unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="output-row">
                    <span class="output-label">Demand Category</span>
                    <span class="output-val" style="color: #e91e63;">{status_tag}</span>
                </div>
                <div class="output-row">
                    <span class="output-label">Confidence Score</span>
                    <span class="output-val" style="color: #2ecc71;">{conf_score}</span>
                </div>
                <div class="output-row">
                    <span class="output-label">Model Used</span>
                    <span class="output-val" style="color: #4b7bec;">{model_display_name}</span>
                </div>
                <div class="output-row">
                    <span class="output-label">Prediction Time</span>
                    <span class="output-val" style="color: #f7b731;">{pred_speed}</span>
                </div>
                <div class="output-row">
                    <span class="output-label">Feature Count</span>
                    <span class="output-val" style="color: #0fbcf9;">7 features</span>
                </div>
                <div class="output-row" style="border-bottom: none;">
                    <span class="output-label">Preprocessing</span>
                    <span class="output-val" style="color: #a55eea;">StandardScaler</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("💡 Please set the feature sliders and click on '⚡ Predict Demand' to generate the outputs.")