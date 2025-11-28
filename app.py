import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings

# ML & Stats Libraries
# --- FIXED IMPORTS BELOW ---
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest 
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from prophet import Prophet
from prophet.plot import plot_plotly

warnings.filterwarnings('ignore')

# ==========================================
# 0. UI CONFIGURATION & CSS STYLING
# ==========================================
st.set_page_config(
    page_title="NCR Ride Analytics Pro V2", 
    page_icon="🚖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for High Contrast & Visibility
st.markdown("""
<style>
    /* 1. Global Reset - Force Light Mode */
    html, body, [class*="css"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* 2. Main App Background */
    .stApp {
        background-color: #ffffff !important;
    }

    /* 3. Text Visibility - Force Black */
    p, span, div, li, label, .stMarkdown {
        color: #000000 !important;
    }
    
    /* 4. Headers - Bold Black */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* 5. Metric Cards - distinct borders */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #ced4da !important;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricLabel"] {
        color: #333333 !important; /* Dark Gray for label */
        font-weight: bold !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Pure Black for value */
    }
    
    /* 6. Sidebar - Light Gray Background, Black Text */
    section[data-testid="stSidebar"] {
        background-color: #f1f3f5 !important;
        border-right: 1px solid #dee2e6;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] p {
        color: #000000 !important;
    }

    /* 7. Input Widgets (Selectbox, Inputs) */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ced4da !important;
    }
    
    /* 8. Custom Info Box */
    .info-box {
        background-color: #e3f2fd !important;
        color: #000000 !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin-bottom: 20px;
    }
    
    /* 9. Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #dee2e6 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #007bff !important;
        color: #ffffff !important; /* Keep active text white */
        border: 1px solid #007bff !important;
    }
    
    /* 10. Buttons */
    button[kind="primary"] {
        background-color: #007bff !important;
        color: #ffffff !important;
        border: none !important;
    }
    button[kind="secondary"] {
        background-color: #e2e6ea !important;
        color: #000000 !important;
        border: 1px solid #ced4da !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================

# --- Preprocessing ---
def clean_data(df, impute_strat='Median', clip_outliers=True):
    """Performs comprehensive cleaning."""
    try:
        # Combine Date and Time
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        df = df.sort_values('DateTime')
    except Exception:
        pass 

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

    cat_cols = ['Reason for cancelling by Customer', 'Driver Cancellation Reason', 
                'Incomplete Rides Reason', 'Payment Method', 'Vehicle Type', 'Booking Status']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

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
    """Generates temporal, cluster, and interaction features."""
    df_eng = df.copy()
    
    if 'DateTime' in df_eng.columns:
        df_eng['month'] = df_eng['DateTime'].dt.month
        df_eng['day_of_week'] = df_eng['DateTime'].dt.day_name()
        df_eng['hour'] = df_eng['DateTime'].dt.hour
        df_eng['is_weekend'] = df_eng['DateTime'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        # Cyclical Features
        df_eng['sin_hour'] = np.sin(2 * np.pi * df_eng['hour'] / 24)
        df_eng['cos_hour'] = np.cos(2 * np.pi * df_eng['hour'] / 24)
    
    le = LabelEncoder()
    if 'Pickup Location' in df_eng.columns:
        loc_numeric = le.fit_transform(df_eng['Pickup Location'].astype(str)).reshape(-1, 1)
        kmeans = KMeans(n_clusters=loc_clusters, random_state=42, n_init=10)
        df_eng['pickup_cluster'] = kmeans.fit_predict(loc_numeric)
    
    if 'Drop Location' in df_eng.columns:
        loc_numeric_drop = le.fit_transform(df_eng['Drop Location'].astype(str)).reshape(-1, 1)
        kmeans_drop = KMeans(n_clusters=loc_clusters, random_state=42, n_init=10)
        df_eng['drop_cluster'] = kmeans_drop.fit_predict(loc_numeric_drop)
    
    # Interaction Feature: Value per km
    if 'Booking Value' in df_eng.columns and 'Ride Distance' in df_eng.columns:
        df_eng['value_per_km'] = df_eng['Booking Value'] / (df_eng['Ride Distance'] + 1e-5)

    return df_eng

# --- Modeling Wrappers ---
def get_feature_importance(model, feature_names, model_type):
    if model_type == 'CatBoost':
        importance = model.get_feature_importance()
    else:
        importance = model.feature_importances_
    
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15)
    
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                 title=f'✨ {model_type} Key Drivers', color='Importance', color_continuous_scale='Viridis')
    return fig

# ==========================================
# 2. MAIN APP LAYOUT
# ==========================================

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/235/235861.png", width=80)
st.sidebar.title("🚖 NCR Analytics V2")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate Module", [
    "1. 🏠 Data Cockpit",
    "2. 🧹 Smart Preprocessing", 
    "3. 🎨 Visual Deep Dive",
    "4. ⚙️ Feature Lab",
    "5. 🤖 Predictive ML (Classif.)",
    "6. 💰 Revenue AI (Regress.)",
    "7. 🕵️ Anomaly Detection (New)",
    "8. 🎯 Customer Segmentation",
    "9. 📈 Demand Forecasting"
])

# Session State Init
for key in ['raw_df', 'clean_df', 'engineered_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# --- PAGE 1: DATA COCKPIT (Enhanced) ---
if page == "1. 🏠 Data Cockpit":
    st.title("🏠 Data Cockpit & Profiling")
    st.markdown("Upload your raw data to get a comprehensive 360-degree view.")
    
    uploaded_file = st.file_uploader("Upload ncr_ride_bookings.csv", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['raw_df'] = df
        
        # Top Level Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rides", f"{df.shape[0]:,}", delta="Raw Data")
        col2.metric("Total Revenue", f"₹{df['Booking Value'].sum():,.0f}", help="Sum of Booking Value (pre-cleaning)")
        col3.metric("Avg Ride Dist", f"{df['Ride Distance'].mean():.1f} km")
        col4.metric("Missing Values", f"{df.isnull().sum().sum():,}", delta_color="inverse")
        
        st.markdown("---")
        
        # Tabs for detailed view
        tab1, tab2, tab3 = st.tabs(["📊 Dataset Preview", "📈 Advanced Statistics", "🔥 Missing Heatmap"])
        
        with tab1:
            st.dataframe(df.head(), use_container_width=True)
            st.caption("First 5 rows of the uploaded dataset.")
            
        with tab2:
            st.subheader("Descriptive Statistics")
            st.markdown("""
            <div class='info-box'>
            <b>💡 Stats Explained:</b><br>
            - <b>Skewness:</b> Measures asymmetry. >1 means long tail on right (outliers).<br>
            - <b>Kurtosis:</b> Measures 'tailedness'. High value = heavy outliers.
            </div>
            """, unsafe_allow_html=True)
            
            # Select only numeric columns for describe
            numeric_df = df.select_dtypes(include=np.number)
            desc = numeric_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
            
            # Calculate Skew/Kurt only for numeric
            desc['skew'] = numeric_df.skew()
            desc['kurtosis'] = numeric_df.kurt()
            
            st.dataframe(desc.style.background_gradient(cmap='Blues'), use_container_width=True)
            
        with tab3:
            fig_null = px.imshow(df.isnull(), color_continuous_scale='RdBu_r', aspect='auto')
            st.plotly_chart(fig_null, use_container_width=True)
    
    elif st.session_state['raw_df'] is not None:
         st.info("Using cached data. Go to other tabs to analyze.")

# --- PAGE 2: PREPROCESSING ---
elif page == "2. 🧹 Smart Preprocessing":
    st.title("🧹 Smart Preprocessing Pipeline")
    
    if st.session_state['raw_df'] is None:
        st.warning("⚠️ Please upload data in the Cockpit first.")
    else:
        st.markdown("<div class='info-box'><b>Why this matters:</b> Real-world data is messy. This pipeline fills missing values using statistical medians and caps extreme outliers (like a ₹100,000 ride) using the IQR method.</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            impute_strategy = st.selectbox("Numerical Imputation Strategy", ["Median (Robust)", "Mean", "Zero"])
        with col2:
            handle_outliers = st.checkbox("Clip Outliers (IQR Method)", value=True, help="Caps values at 1.5*IQR to remove extreme anomalies.")
            
        if st.button("🚀 Run Cleaning Pipeline", type="primary"):
            with st.spinner("Scrubbing data..."):
                df_clean = clean_data(st.session_state['raw_df'].copy(), impute_strat=impute_strategy.split()[0], clip_outliers=handle_outliers)
                st.session_state['clean_df'] = df_clean
                st.success("✅ Data Cleaned Successfully!")
                
                # Validation
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows Retained", len(df_clean))
                with col2:
                    remaining_nulls = df_clean.isnull().sum().sum()
                    st.metric("Remaining Nulls", remaining_nulls, delta="Ideal: 0", delta_color="normal" if remaining_nulls==0 else "inverse")

# --- PAGE 3: VISUAL DEEP DIVE (NEW) ---
elif page == "3. 🎨 Visual Deep Dive":
    st.title("🎨 Visual Analytics Explorer")
    
    if st.session_state['clean_df'] is None:
        st.warning("⚠️ Please run Preprocessing first.")
    else:
        df = st.session_state['clean_df']
        
        # 1. Sunburst
        st.subheader("1. Hierarchical Status Analysis")
        st.markdown("Drill down into *Booking Status* → *Reasons*. Click on segments to expand.")
        
        # Prep data for sunburst
        df_sun = df[df['Booking Status'] != 'Completed'].fillna("Unknown")
        # We need a hierarchy: Status -> Reason
        # Consolidate reason columns
        df_sun['Combined_Reason'] = df_sun['Reason for cancelling by Customer'] + df_sun['Driver Cancellation Reason'] + df_sun['Incomplete Rides Reason']
        df_sun['Combined_Reason'] = df_sun['Combined_Reason'].str.replace("UnknownUnknown", "").str.replace("Unknown", "")
        df_sun.loc[df_sun['Combined_Reason'] == "", 'Combined_Reason'] = "Unspecified"
        
        # Handle case where filtered df might be empty
        if not df_sun.empty:
            fig_sun = px.sunburst(
                df_sun, path=['Booking Status', 'Combined_Reason'], 
                title="Lost Demand Breakdown (Interactive)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("No cancellations or incomplete rides found to visualize.")
        
        col1, col2 = st.columns(2)
        
        # 2. Violin Plot
        with col1:
            st.subheader("2. Price Distribution by Vehicle")
            st.markdown("Violin plots show the density of data at different values.")
            fig_vio = px.violin(df, x="Vehicle Type", y="Booking Value", box=True, points="all", color="Vehicle Type")
            fig_vio.update_layout(showlegend=False)
            st.plotly_chart(fig_vio, use_container_width=True)
            
        # 3. Heatmap
        with col2:
            st.subheader("3. Peak Demand Heatmap")
            if 'Date' in df.columns and 'Time' in df.columns:
                try:
                    # Ensure DateTime exists
                    if 'DateTime' not in df.columns:
                         df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
                    
                    df['Day'] = df['DateTime'].dt.day_name()
                    df['Hour'] = df['DateTime'].dt.hour
                    
                    heatmap_data = df.groupby(['Day', 'Hour']).size().unstack(fill_value=0)
                    # Reorder days
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    # Only reindex if the days exist in the data
                    existing_days = [d for d in days_order if d in heatmap_data.index]
                    heatmap_data = heatmap_data.reindex(existing_days)
                    
                    fig_heat = px.imshow(heatmap_data, title="Rides by Day & Hour", color_continuous_scale='Viridis')
                    st.plotly_chart(fig_heat, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate heatmap: {e}")

# --- PAGE 4: FEATURE ENGINEERING ---
elif page == "4. ⚙️ Feature Lab":
    st.title("⚙️ Feature Engineering Lab")
    
    if st.session_state['clean_df'] is None:
        st.warning("⚠️ Clean data required.")
    else:
        st.info("Transforms raw timestamps into cyclical features (sin_hour, cos_hour) and uses K-Means to cluster location names into zones.")
        
        if st.button("⚡ Generate Advanced Features"):
            df_eng = engineer_features(st.session_state['clean_df'].copy())
            st.session_state['engineered_df'] = df_eng
            st.success("Features Generated!")
            st.write(df_eng[['DateTime', 'sin_hour', 'pickup_cluster', 'is_weekend']].head())

# --- PAGE 5: ML CLASSIFICATION ---
elif page == "5. 🤖 Predictive ML (Classif.)":
    st.title("🤖 Cancellation Prediction AI")
    
    if st.session_state['engineered_df'] is None:
        st.error("⚠️ Please run Feature Engineering first.")
    else:
        df = st.session_state['engineered_df'].copy()
        # Ensure we have a target
        df['Target'] = df['Booking Status'].apply(lambda x: 1 if 'Cancel' in str(x) else 0)
        
        # Feature Selection
        features = ['Booking Value', 'Ride Distance', 'hour', 'is_weekend', 'sin_hour', 'pickup_cluster']
        # Check if cols exist
        available_feats = [f for f in features if f in df.columns]
        
        X = df[available_feats].fillna(0)
        y = df['Target']
        
        model_name = st.selectbox("Choose Model Architecture", ["Gradient Boosting (Best)", "Random Forest", "XGBoost"])
        
        if st.button("Train Classifier"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            if "Random" in model_name: model = RandomForestClassifier()
            elif "Gradient" in model_name: model = GradientBoostingClassifier()
            else: model = XGBClassifier()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.3f}", help="Area Under Curve. 1.0 is perfect, 0.5 is random.")
            c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}", help="% of predicted cancellations that were actual cancellations.")
            c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}", help="% of actual cancellations detected.")
            c4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")
            
            st.plotly_chart(get_feature_importance(model, available_feats, model_name), use_container_width=True)

# --- PAGE 6: ML REGRESSION ---
elif page == "6. 💰 Revenue AI (Regress.)":
    st.title("💰 Fare/Revenue Prediction")
    st.markdown("Predict the `Booking Value` for completed rides.")
    
    if st.session_state['engineered_df'] is not None:
        df = st.session_state['engineered_df'].copy()
        df = df[df['Booking Status'] == 'Completed']
        
        features = ['Ride Distance', 'hour', 'pickup_cluster', 'drop_cluster']
        available_feats = [f for f in features if f in df.columns]
        
        X = df[available_feats].fillna(0)
        y = df['Booking Value']
        
        model_name = st.selectbox("Regressor", ["Gradient Boosting", "Random Forest", "CatBoost"])
        
        if st.button("Train Regressor"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            if "Gradient" in model_name: model = GradientBoostingRegressor()
            elif "Random" in model_name: model = RandomForestRegressor()
            else: model = CatBoostRegressor(verbose=0)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"₹{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}", help="Root Mean Squared Error. Typical prediction error in Rupees.")
            col2.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}", help="Variance explained by the model (Max 1.0).")
            
            # Residual Plot
            st.subheader("Residual Analysis")
            st.markdown("Ideal residuals should be randomly scattered around 0.")
            residuals = y_test - y_pred
            fig_res = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Value', 'y': 'Residuals (Error)'}, title="Residual Plot")
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)

# --- PAGE 7: ANOMALY DETECTION (NEW) ---
elif page == "7. 🕵️ Anomaly Detection (New)":
    st.title("🕵️ Anomaly & Fraud Detection")
    st.markdown("""
    <div class='info-box'>
    Uses <b>Isolation Forest</b> to detect 'strange' rides. 
    Examples: High price for short distance, weird times, or unusual locations.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['engineered_df'] is not None:
        df = st.session_state['engineered_df'].copy()
        cols = ['Booking Value', 'Ride Distance', 'Avg VTAT']
        # Filter numeric only
        numeric_cols = [c for c in cols if c in df.columns]
        
        if numeric_cols:
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            
            contamination = st.slider("Contamination (Expected % of outliers)", 0.01, 0.1, 0.02)
            
            if st.button("🔍 Scan for Anomalies"):
                iso = IsolationForest(contamination=contamination, random_state=42)
                df['anomaly'] = iso.fit_predict(X)
                # -1 is anomaly, 1 is normal
                anomalies = df[df['anomaly'] == -1]
                
                st.error(f"⚠️ Found {len(anomalies)} anomalies out of {len(df)} rides!")
                
                st.subheader("Anomaly Visualization")
                fig_anom = px.scatter(df, x="Ride Distance", y="Booking Value", color=df['anomaly'].astype(str),
                                      color_discrete_map={'-1':'red', '1':'blue'},
                                      title="Red points are anomalies")
                st.plotly_chart(fig_anom, use_container_width=True)
                
                st.subheader("Inspect Suspicious Rides")
                st.dataframe(anomalies[numeric_cols + ['Booking Status']].head(20))
        else:
            st.error("Required columns for anomaly detection not found.")

# --- PAGE 8: CLUSTERING ---
elif page == "8. 🎯 Customer Segmentation":
    st.title("🎯 Customer/Ride Segmentation")
    st.info("Groups rides into 'personas' based on behavior.")
    
    if st.session_state['engineered_df'] is not None:
        k = st.slider("Number of Clusters", 2, 6, 4)
        if st.button("Run K-Means"):
            df = st.session_state['engineered_df'].copy()
            feat = ['Booking Value', 'Ride Distance', 'Driver Ratings']
            available_feats = [f for f in feat if f in df.columns]
            
            X = StandardScaler().fit_transform(df[available_feats].fillna(0))
            
            clusters = KMeans(n_clusters=k).fit_predict(X)
            df['Cluster'] = clusters
            
            # Radar Chart
            st.subheader("Cluster Profiles")
            means = df.groupby('Cluster')[available_feats].mean()
            scaler = MinMaxScaler()
            means_scaled = pd.DataFrame(scaler.fit_transform(means), columns=means.columns)
            
            fig_radar = go.Figure()
            for i in range(k):
                fig_radar.add_trace(go.Scatterpolar(r=means_scaled.iloc[i], theta=available_feats, fill='toself', name=f'Cluster {i}'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig_radar, use_container_width=True)

# --- PAGE 9: FORECASTING ---
elif page == "9. 📈 Demand Forecasting":
    st.title("📈 Future Demand Forecasting")
    
    if st.session_state['clean_df'] is not None:
        df = st.session_state['clean_df'].copy()
        
        # Ensure Date is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            daily = df.groupby('Date').size().reset_index(name='y').rename(columns={'Date':'ds'})
            
            horizon = st.slider("Forecast Days", 7, 60, 30)
            
            if st.button("Generate Forecast"):
                m = Prophet()
                m.fit(daily)
                future = m.make_future_dataframe(periods=horizon)
                forecast = m.predict(future)
                
                st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
