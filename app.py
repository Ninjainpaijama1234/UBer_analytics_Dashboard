import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ML & Stats Libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from prophet import Prophet
from prophet.plot import plot_plotly

# ==========================================
# 1. UTILITY FUNCTIONS (Merged from Modules)
# ==========================================

# --- Preprocessing ---
def clean_data(df, impute_strat='Median', clip_outliers=True):
    """Performs comprehensive cleaning on the NCR rides dataset."""
    
    # Date & Time Handling
    try:
        # Combine Date and Time if strictly necessary, but Date is usually enough for daily aggregation
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        df = df.sort_values('DateTime')
    except Exception:
        pass 

    # Numerical Imputation
    numeric_cols = [
        'Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance', 
        'Driver Ratings', 'Customer Rating', 'Incomplete Rides',
        'Cancelled Rides by Customer', 'Cancelled Rides by Driver'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if impute_strat == 'Median':
                val = df[col].median()
            elif impute_strat == 'Mean':
                val = df[col].mean()
            else:
                val = 0
            df[col] = df[col].fillna(val)

    # Categorical Imputation
    cat_cols = [
        'Reason for cancelling by Customer', 'Driver Cancellation Reason', 
        'Incomplete Rides Reason', 'Payment Method', 'Vehicle Type', 'Booking Status'
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Outlier Clipping (IQR)
    if clip_outliers:
        for col in ['Booking Value', 'Ride Distance', 'Avg VTAT']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    return df

# --- Feature Engineering ---
def engineer_features(df, loc_clusters=5):
    """Generates new features for ML models."""
    df_eng = df.copy()
    
    # Temporal Features
    if 'DateTime' in df_eng.columns:
        df_eng['month'] = df_eng['DateTime'].dt.month
        df_eng['day_of_week'] = df_eng['DateTime'].dt.dayofweek
        df_eng['is_weekend'] = df_eng['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df_eng['hour'] = df_eng['DateTime'].dt.hour
        df_eng['minute'] = df_eng['DateTime'].dt.minute
        # Cyclical Time Encoding
        df_eng['sin_hour'] = np.sin(2 * np.pi * df_eng['hour'] / 24)
        df_eng['cos_hour'] = np.cos(2 * np.pi * df_eng['hour'] / 24)
    
    # Location Clustering (Text-based approximation)
    le = LabelEncoder()
    if 'Pickup Location' in df_eng.columns:
        loc_numeric = le.fit_transform(df_eng['Pickup Location'].astype(str)).reshape(-1, 1)
        kmeans = KMeans(n_clusters=loc_clusters, random_state=42, n_init=10)
        df_eng['pickup_cluster'] = kmeans.fit_predict(loc_numeric)
        
    if 'Drop Location' in df_eng.columns:
        loc_numeric_drop = le.fit_transform(df_eng['Drop Location'].astype(str)).reshape(-1, 1)
        kmeans_drop = KMeans(n_clusters=loc_clusters, random_state=42, n_init=10)
        df_eng['drop_cluster'] = kmeans_drop.fit_predict(loc_numeric_drop)

    # Rating Features
    if 'Driver Ratings' in df_eng.columns and 'Customer Rating' in df_eng.columns:
        df_eng['rating_gap'] = df_eng['Driver Ratings'] - df_eng['Customer Rating']
    
    # One-Hot Encoding for Reasons (keep original for CatBoost if needed later)
    reasons = ['Reason for cancelling by Customer', 'Driver Cancellation Reason', 'Incomplete Rides Reason']
    for r in reasons:
        if r in df_eng.columns:
            dummies = pd.get_dummies(df_eng[r], prefix=r.split()[0][:3])
            df_eng = pd.concat([df_eng, dummies], axis=1)
            
    return df_eng

# --- Modeling Helpers ---
def get_feature_importance(model, feature_names, model_type):
    if model_type == 'CatBoost':
        importance = model.get_feature_importance()
    else:
        importance = model.feature_importances_
    
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15)
    
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title=f'{model_type} Feature Importance')
    return fig

def train_classification(df, target, model_type='RandomForest'):
    # Drop non-numeric/high-cardinality columns
    exclude_cols = [target, 'Date', 'Time', 'DateTime', 'Booking ID', 'Customer ID', 'Booking Status',
                    'Pickup Location', 'Drop Location', 'Reason for cancelling by Customer', 
                    'Driver Cancellation Reason', 'Incomplete Rides Reason']
    
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]) 
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_type == 'CatBoost':
        model = CatBoostClassifier(verbose=0)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    results = {
        'roc_auc': roc_auc_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'feat_fig': get_feature_importance(model, X.columns, model_type)
    }
    return results

def train_regression(df, target, model_type='RandomForest'):
    exclude_cols = [target, 'Date', 'Time', 'DateTime', 'Booking ID', 'Customer ID', 'Booking Status',
                    'Pickup Location', 'Drop Location', 'Reason for cancelling by Customer', 
                    'Driver Cancellation Reason', 'Incomplete Rides Reason']
    
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'XGBoost':
        model = XGBRegressor()
    elif model_type == 'CatBoost':
        model = CatBoostRegressor(verbose=0)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'feat_fig': get_feature_importance(model, X.columns, model_type)
    }
    return results

# --- Clustering ---
def run_clustering(df, k=4):
    features = ['Booking Value', 'Ride Distance', 'Avg VTAT', 'Driver Ratings', 'Customer Rating']
    X = df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_result = df.copy()
    df_result['Cluster_Label'] = clusters
    
    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    fig_pca = px.scatter(
        x=components[:,0], y=components[:,1], 
        color=clusters.astype(str),
        title=f"Customer Segments (PCA Projection) - K={k}",
        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'}
    )
    
    # Radar Chart
    cluster_means = pd.DataFrame(X_scaled, columns=features).groupby(clusters).mean()
    fig_radar = go.Figure()
    for i in range(k):
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_means.iloc[i].values,
            theta=features,
            fill='toself',
            name=f'Cluster {i}'
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 3])), showlegend=True, title="Cluster Profiles")
    
    return df_result, fig_pca, fig_radar

# --- Forecasting ---
def forecast_demand(df, periods=30):
    if 'DateTime' not in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        daily_counts = df.groupby('Date').size().reset_index(name='y')
        daily_counts.rename(columns={'Date': 'ds'}, inplace=True)
    else:
        df['DateOnly'] = df['DateTime'].dt.date
        daily_counts = df.groupby('DateOnly').size().reset_index(name='y')
        daily_counts.rename(columns={'DateOnly': 'ds'}, inplace=True)
        
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(daily_counts)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title=f"Ride Demand Forecast ({periods} Days Ahead)")
    
    fig_components = go.Figure()
    fig_components.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name="Trend"))
    fig_components.update_layout(title="Underlying Trend Component")
    
    return fig_forecast, fig_components, forecast

# ==========================================
# 2. MAIN STREAMLIT APP LOGIC
# ==========================================

st.set_page_config(page_title="NCR Ride Analytics Pro", page_icon="🚖", layout="wide")

st.sidebar.title("🚖 NCR Analytics Suite")
page = st.sidebar.radio("Navigate", [
    "1. Data Overview",
    "2. Preprocessing", 
    "3. Feature Engineering",
    "4. ML Classification (Cancellation)",
    "5. ML Regression (Booking Value)",
    "6. Clustering & Segmentation",
    "7. Time-Series Forecasting"
])

# Initialize Session State
if 'raw_df' not in st.session_state:
    st.session_state['raw_df'] = None
if 'clean_df' not in st.session_state:
    st.session_state['clean_df'] = None
if 'engineered_df' not in st.session_state:
    st.session_state['engineered_df'] = None

# --- PAGE 1: DATA OVERVIEW ---
if page == "1. Data Overview":
    st.title("📂 Data Overview & Profiling")
    
    uploaded_file = st.file_uploader("Upload ncr_ride_bookings.csv", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['raw_df'] = df
            st.success("File uploaded successfully!")
            st.markdown("### 🔍 Dataset Preview")
            st.dataframe(df.head())
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col1.metric("Columns", df.shape[1])
            col2.metric("Duplicates", df.duplicated().sum())
            col3.metric("Missing Values", df.isnull().sum().sum())
            
            st.markdown("### 🧬 Column Types")
            st.write(df.dtypes.astype(str))
            
            st.markdown("### 🔥 Missing Value Heatmap")
            fig = px.imshow(df.isnull(), color_continuous_scale='RdBu_r', title="Null Value Distribution")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    elif st.session_state['raw_df'] is not None:
        st.info("Using previously uploaded data.")
        st.dataframe(st.session_state['raw_df'].head())
    else:
        st.warning("Please upload a dataset to begin.")

# --- PAGE 2: PREPROCESSING ---
elif page == "2. Preprocessing":
    st.title("🧹 Automated Preprocessing Pipeline")
    
    if st.session_state['raw_df'] is None:
        st.warning("Please upload data in Page 1 first.")
    else:
        st.markdown("### Configuration")
        col1, col2 = st.columns(2)
        with col1:
            impute_strategy = st.selectbox("Numerical Imputation", ["Median", "Mean", "Zero"])
        with col2:
            handle_outliers = st.checkbox("Clip Outliers (IQR Method)", value=True)
            
        if st.button("Run Preprocessing"):
            with st.spinner("Cleaning data..."):
                df_clean = clean_data(
                    st.session_state['raw_df'].copy(), 
                    impute_strat=impute_strategy,
                    clip_outliers=handle_outliers
                )
                st.session_state['clean_df'] = df_clean
                st.success("Preprocessing Complete!")
                st.markdown("### 📊 Post-Processing Stats")
                st.dataframe(df_clean.describe())
                st.markdown("#### Null Check (Should be 0)")
                st.write(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# --- PAGE 3: FEATURE ENGINEERING ---
elif page == "3. Feature Engineering":
    st.title("⚙️ Feature Engineering Factory")
    
    if st.session_state['clean_df'] is None:
        st.warning("Please run Preprocessing in Page 2 first.")
    else:
        st.write("Generates temporal features, location clusters, and one-hot encoded reasons.")
        n_clusters = st.slider("Number of Location Clusters (KMeans)", 3, 10, 5)
        
        if st.button("Generate Features"):
            with st.spinner("Engineering features..."):
                df_eng = engineer_features(
                    st.session_state['clean_df'].copy(), 
                    loc_clusters=n_clusters
                )
                st.session_state['engineered_df'] = df_eng
                st.success("Feature Engineering Complete!")
                st.markdown("### 🆕 New Features Generated")
                new_cols = [c for c in df_eng.columns if c not in st.session_state['clean_df'].columns]
                st.write(new_cols)
                if 'pickup_cluster' in df_eng.columns:
                    st.write(df_eng[['Pickup Location', 'pickup_cluster']].drop_duplicates().head(10))

# --- PAGE 4: ML CLASSIFICATION ---
elif page == "4. ML Classification (Cancellation)":
    st.title("🚫 Cancellation Prediction (Classification)")
    
    if st.session_state['engineered_df'] is None:
        st.warning("Please run Feature Engineering in Page 3.")
    else:
        df = st.session_state['engineered_df'].copy()
        df['Target_Cancel'] = df['Booking Status'].apply(lambda x: 1 if 'Cancel' in str(x) else 0)
        
        model_type = st.selectbox("Select Model", ["RandomForest", "XGBoost", "CatBoost"])
        
        if st.button("Train Classification Model"):
            with st.spinner(f"Training {model_type}..."):
                results = train_classification(df, target='Target_Cancel', model_type=model_type)
                col1, col2, col3 = st.columns(3)
                col1.metric("ROC AUC", f"{results['roc_auc']:.3f}")
                col2.metric("Precision", f"{results['precision']:.3f}")
                col3.metric("Recall", f"{results['recall']:.3f}")
                st.markdown("### Feature Importance")
                st.plotly_chart(results['feat_fig'], use_container_width=True)
                st.markdown("### Confusion Matrix")
                st.write(results['conf_matrix'])

# --- PAGE 5: ML REGRESSION ---
elif page == "5. ML Regression (Booking Value)":
    st.title("💰 Booking Value Prediction (Regression)")
    
    if st.session_state['engineered_df'] is None:
        st.warning("Please run Feature Engineering in Page 3.")
    else:
        df = st.session_state['engineered_df'].copy()
        df_reg = df[df['Booking Status'] == 'Completed']
        
        model_type = st.selectbox("Select Regressor", ["RandomForest", "XGBoost", "CatBoost"])
        
        if st.button("Train Regression Model"):
            with st.spinner(f"Training {model_type}..."):
                results = train_regression(df_reg, target='Booking Value', model_type=model_type)
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"₹{results['mae']:.2f}")
                col2.metric("RMSE", f"₹{results['rmse']:.2f}")
                col3.metric("R² Score", f"{results['r2']:.3f}")
                st.markdown("### Feature Importance")
                st.plotly_chart(results['feat_fig'], use_container_width=True)

# --- PAGE 6: CLUSTERING ---
elif page == "6. Clustering & Segmentation":
    st.title("🎯 Customer/Ride Segmentation")
    
    if st.session_state['engineered_df'] is None:
        st.warning("Please run Feature Engineering in Page 3.")
    else:
        k = st.slider("Number of Segments (K)", 2, 8, 4)
        if st.button("Run Clustering"):
            df_clustered, fig_pca, fig_radar = run_clustering(st.session_state['engineered_df'], k)
            st.plotly_chart(fig_pca, use_container_width=True)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("### Segment Profiles")
            st.dataframe(df_clustered.groupby('Cluster_Label').mean(numeric_only=True).T)
            csv = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button("Download Segmented Data", csv, "segmented_rides.csv", "text/csv")

# --- PAGE 7: FORECASTING ---
elif page == "7. Time-Series Forecasting":
    st.title("📈 Demand Forecasting")
    
    if st.session_state['clean_df'] is None:
        st.warning("Need clean data with Dates.")
    else:
        df = st.session_state['clean_df'].copy()
        days_forecast = st.slider("Forecast Horizon (Days)", 7, 90, 30)
        
        if st.button("Generate Forecast"):
            with st.spinner("Forecasting with Prophet..."):
                forecast_fig, components_fig, forecast_df = forecast_demand(df, days_forecast)
                st.plotly_chart(forecast_fig, use_container_width=True)
                st.markdown("### Trend & Seasonality")
                st.plotly_chart(components_fig, use_container_width=True)
                st.dataframe(forecast_df.tail(10))
