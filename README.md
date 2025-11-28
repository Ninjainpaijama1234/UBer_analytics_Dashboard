🚖 NCR Ride Analytics Pro

A comprehensive Streamlit application for analyzing ride-hailing data, predicting cancellations, estimating booking values, and forecasting demand using Machine Learning.

📋 Project Overview

This dashboard allows operations managers and data analysts to:

Analyze historical ride data (Revenue, VTAT, CTAT, Ratings).

Clean & Process raw data with an automated pipeline (Imputation, Outlier Clipping).

Predict ride cancellations using classification models (RandomForest, XGBoost, CatBoost).

Estimate booking revenue using regression models.

Segment rides/customers using KMeans clustering and PCA.

Forecast future daily demand using Facebook Prophet.

🛠️ Installation & Setup

1. Prerequisites

Ensure you have Python 3.10+ installed.

2. Install Dependencies

Run the following command in your terminal to install the required libraries:

pip install streamlit pandas numpy plotly scikit-learn xgboost catboost prophet


Note: If you encounter issues installing prophet on Windows, you may need to install C++ compilers or use conda install -c conda-forge prophet.

3. Run the Application

Navigate to the project folder and run:

streamlit run app.py


📂 Application Workflow

The app is structured into a logical workflow accessible via the sidebar:

Data Overview: Upload your CSV and view basic profiling (Nulls, Types, Duplicates).

Preprocessing: Handle missing values (Median/Mean/Zero) and clip outliers (IQR method).

Feature Engineering: Generate temporal features (Hour, Day, Weekend), cluster locations, and encode categorical reasons.

ML Classification: Train models to predict if a ride will be cancelled.

ML Regression: Train models to predict the Booking Value of completed rides.

Clustering: Segment rides into profiles (e.g., "High Value/Long Distance" vs "Short Commute").

Forecasting: Predict future ride demand (Daily Volume) for the next 7-90 days.

📊 Dataset Schema

The app expects a CSV file named ncr_ride_bookings.csv (or uploaded via UI) with the following key columns:

Time/Date: Date, Time

IDs: Booking ID, Customer ID

Status: Booking Status

Metrics: Avg VTAT, Avg CTAT, Booking Value, Ride Distance

Ratings: Driver Ratings, Customer Rating

Locations: Pickup Location, Drop Location

Reasons (Optional): Reason for cancelling by Customer, Driver Cancellation Reason, Incomplete Rides Reason

🤖 Modeling Details

Classification: Uses RandomForestClassifier, XGBClassifier, or CatBoostClassifier.

Target: Binary (1 if Cancelled, 0 otherwise).

Metrics: ROC-AUC, Precision, Recall, F1-Score.

Regression: Uses RandomForestRegressor, XGBRegressor, or CatBoostRegressor.

Target: Booking Value.

Metrics: MAE (Mean Absolute Error), RMSE, R².

Clustering: Uses KMeans on standardized features (Booking Value, Ride Distance, Ratings). Visualized with PCA (2D) and Radar Charts.

Forecasting: Uses Prophet to model daily seasonality and trend.

❓ Troubleshooting

"File not found": Ensure you upload the CSV on the "Data Overview" page first.

"Input contains NaN": Go to the Preprocessing page and run the cleaning pipeline before attempting ML tasks.

"KeyError": Ensure your CSV has the exact column names listed in the Dataset Schema section.
