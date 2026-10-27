# Retail Intelligence — Sales Forecasting & Analytics

A retail analytics project combining data engineering, time-series feature engineering, machine learning, and interactive visualization to support sales forecasting and business decision-making.

## Project Overview

Retail Intelligence analyzes historical retail sales and provides an interactive dashboard for exploring sales performance and evaluating machine learning forecasts.

## Key Features

- **Data processing:** Retail data queried and processed with DuckDB.
- **Feature engineering:** Lagged sales, rolling statistics, calendar variables, promotions, and store/product attributes.
- **Machine learning:** Random Forest regression with preprocessing.
- **Model evaluation:** Time-based validation using Root Mean Squared Logarithmic Error (RMSLE).
- **Interactive dashboard:** Streamlit filters, KPIs, charts, forecast comparisons, and CSV export.

## Technologies

Python, Pandas, NumPy, DuckDB, Scikit-learn, Streamlit, Plotly.

## Model Evaluation

Previous evaluation results reported an RMSLE of approximately **0.42814** for Random Forest and **0.55787** for the `lag_14` baseline, corresponding to an improvement of about **23.16%**. These results should be reproduced using the validation script before being presented as a final benchmark.

## Run Locally

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Prepare the local dataset and DuckDB database using the project's ingestion and feature-engineering scripts.
5. Launch the dashboard:

   ```bash
   python -m streamlit run src/Tableau.py
   ```

The dashboard requires the local database and trained model at the paths expected by the application.

## Data and Limitations

Raw data and local DuckDB databases are excluded from the repository. Results depend on the available historical dataset. Historical evaluation is not a live forecast for current dates.

## Future Improvements

- Add automated tests and reproducible training instructions.
- Compare additional forecasting algorithms.
- Improve inventory planning and demand monitoring.
- Deploy the dashboard for public access.

## Author

**Ba Amadou**  
Master's student in Mathematics, Data Analysis and Machine Learning