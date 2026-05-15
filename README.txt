==========================================================
PROJECT NAME: EnergySense AI - Hotel Energy Demand Prediction
==========================================================

1. PROJECT OBJECTIVE:
--------------------
The primary objective of EnergySense AI is to provide a smart predictive 
analytics tool for the hotel industry. It forecasts the daily and hourly 
electrical energy consumption based on variables like room occupancy, 
outdoor temperature, and operational hours. This helps hotel management 
to optimize energy resources, reduce utility costs, and support 
sustainable operations.

2. TOOLS AND TECHNOLOGIES USED:
------------------------------
- Programming Language: Python 3.9+
- Web Framework: Streamlit
- Machine Learning Models: XGBoost, Random Forest, Logistic Regression
- Data Libraries: Pandas, NumPy, Scikit-learn
- Visualization: Plotly Express, Plotly Graph Objects
- Development Environment: Google Colab & VS Code

3. INSTALLATION STEPS:
----------------------
To run this project locally, follow these steps:

Step 1: Extract the project ZIP file.
Step 2: Open your terminal or command prompt in the project folder.
Step 3: Install the required libraries using the following command:
        pip install -r requirements.txt

4. EXECUTION COMMAND:
---------------------
To launch the interactive dashboard, run the following command in your terminal:

        streamlit run app.py

5. DESCRIPTION OF INPUTS AND OUTPUTS:
-------------------------------------
INPUTS:
- Outdoor Temperature: Current weather temperature in Celsius.
- Room Occupancy: Number of rooms currently occupied by guests.
- AC Usage Hours: Total hours the air conditioning system is active.
- Kitchen/Laundry Activity: Operational status of heavy appliances.

OUTPUTS:
- Energy Forecast: Predicted energy demand in Kilowatts (kW).
- Analytics Dashboard: Interactive Heatmaps and Trend lines for data insights.
- Performance Metrics: Accuracy rings and doughnut charts comparing 
  different ML models (XGBoost, RF, Logistic Regression).
- Development Roadmap: A Gantt chart showing the project's timeline 
  and implementation phases.

==========================================================
Developed as part of the Machine Learning Project Submission.