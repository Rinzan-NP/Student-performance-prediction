import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    """Prepare features and target for student performance prediction"""
    # Create target variable (average of all scores)
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
    
    # Create performance categories for binary classification
    df['performance_category'] = (df['average_score'] >= 70).astype(int)  # 0 = Low Performance, 1 = High Performance
    
    # Encode categorical variables
    categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare features - ONLY demographic features to avoid data leakage
    # We do NOT include the actual test scores as features since we're trying to predict them
    feature_columns = ['gender_encoded', 'race/ethnicity_encoded', 'parental level of education_encoded', 
                      'lunch_encoded', 'test preparation course_encoded']
    
    X = df[feature_columns].values
    y_regression = df['average_score'].values
    y_classification = df['performance_category'].values
    
    return X, y_regression, y_classification, feature_columns, label_encoders


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute classification metrics for binary classification"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    if cm.shape == (2, 2):  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        class_metrics = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1
        }
    else:
        class_metrics = {}
    
    return accuracy, cm, class_metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: list = None):
    """Plot confusion matrix for binary classification"""
    if class_names is None:
        class_names = ['Low Performance', 'High Performance']
    
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title("Confusion Matrix - Performance Categories")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, alpha=0.5, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve - Performance Classification", fontsize=11, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Plot actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Scores")
    ax.set_ylabel("Predicted Scores")
    ax.set_title(f"Actual vs Predicted - {model_name}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Plot residuals"""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted Scores")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals Plot - {model_name}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def build_model(model_name: str, **kwargs):
    """Build the specified model"""
    if model_name == "Linear Regression":
        return LinearRegression()
    elif model_name == "Decision Tree":
        return DecisionTreeRegressor(random_state=42, **kwargs)
    elif model_name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=42, **kwargs)
    else:
        raise ValueError("Unknown model")


def main():
    st.set_page_config(page_title="Student Performance Prediction", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main > div {padding-top: 2rem;}
        .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .stMetric label {font-size: 14px !important; font-weight: 600 !important;}
        .stMetric [data-testid="stMetricValue"] {font-size: 24px !important;}
        h1 {color: #1f77b4; padding-bottom: 10px; border-bottom: 3px solid #1f77b4;}
        h2 {color: #2c3e50; margin-top: 25px;}
        h3 {color: #34495e;}
        .stButton>button {width: 100%; background-color: #1f77b4; color: white; font-weight: 600;}
        .stButton>button:hover {background-color: #155a8a; border-color: #155a8a;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Student Performance Prediction System")
    st.markdown("**Predict student academic performance based on demographic and educational factors**")
    st.caption("📊 Models: Linear Regression · Decision Tree · Gradient Boosting | 📈 Metrics: MAE, RMSE, R², AUC")

    with st.sidebar:
        st.header("📁 Data & Preprocessing")
        data_source = st.radio("Dataset source", ["Included StudentsPerformance.csv", "Upload CSV"], index=0)
        uploaded = None
        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload student performance CSV", type=["csv"])

        st.markdown("---")
        st.header("🤖 Model Configuration")
        model_name = st.selectbox("Choose model", ["Linear Regression", "Decision Tree", "Gradient Boosting"], index=2)

        # Model-specific parameters
        if model_name == "Decision Tree":
            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=10, step=1)
            min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
            model_params = {"max_depth": max_depth, "min_samples_split": min_samples_split}
        elif model_name == "Gradient Boosting":
            n_estimators = st.slider("N Estimators", min_value=10, max_value=500, value=100, step=10)
            learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3, step=1)
            model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}
        else:
            model_params = {}

        st.markdown("---")
        st.header("⚙️ Training Configuration")
        test_size = st.slider("Test size (%)", min_value=10, max_value=50, value=20, step=5)
        random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)
        scale_features = st.checkbox("Standardize features", value=True)
        st.markdown("---")
        run_button = st.button("🚀 Train & Evaluate", type="primary", use_container_width=True)

    # Load data
    if data_source == "Included StudentsPerformance.csv":
        df = load_data("StudentsPerformance.csv")
    else:
        if uploaded is None:
            st.info("Upload a CSV to proceed or switch to the included dataset.")
            return
        df = pd.read_csv(uploaded)

    # Check for required columns
    expected_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", 
                       "test preparation course", "math score", "reading score", "writing score"]
    if not set(expected_columns).issubset(set(df.columns)):
        st.error("CSV must contain standard student performance columns.")
        st.stop()

    st.subheader("📊 Data Overview")
    
    # Display dataset info in a more compact way
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", df.shape[0])
    with col2:
        st.metric("Avg Math Score", f"{df['math score'].mean():.1f}")
    with col3:
        st.metric("Avg Reading Score", f"{df['reading score'].mean():.1f}")
    with col4:
        st.metric("Avg Writing Score", f"{df['writing score'].mean():.1f}")
    
    with st.expander("📋 View Dataset Sample & Statistics", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("**Score Summary Statistics**")
        score_cols = ["math score", "reading score", "writing score"]
        st.dataframe(df[score_cols].describe().T, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Score Correlations")
    corr = df[["math score", "reading score", "writing score"]].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt='.2f', linewidths=1, linecolor='white')
    ax.set_title("Correlation Matrix - Test Scores", fontsize=12, fontweight='bold')
    st.pyplot(fig, clear_figure=True)

    # Prepare features
    X, y_regression, y_classification, feature_cols, label_encoders = prepare_features(df)

    # Split data
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_regression, y_classification, test_size=test_size/100.0, random_state=random_state
    )

    # Scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Build model
    model = build_model(model_name, **model_params)

    if run_button:
        with st.spinner("Training model..."):
            # Train regression model
            model.fit(X_train, y_train_reg)
            y_pred_reg = model.predict(X_test)
            
            # Compute regression metrics
            mae, rmse, r2 = compute_regression_metrics(y_test_reg, y_pred_reg)

            # For classification, we'll use the regression predictions to classify
            y_pred_cls = (y_pred_reg >= 70).astype(int)  # 0=Low Performance, 1=High Performance
            accuracy, cm, class_metrics = compute_classification_metrics(y_test_cls, y_pred_cls)

            st.markdown("---")
            st.subheader("🎯 Performance Metrics")
            
            st.markdown("#### 📐 Regression Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE (Mean Absolute Error)", f"{mae:.4f}", help="Average absolute difference between predicted and actual scores")
            col2.metric("RMSE (Root Mean Squared Error)", f"{rmse:.4f}", help="Square root of average squared differences")
            col3.metric("R² Score", f"{r2:.4f}", help="Proportion of variance explained by the model")
            
            st.markdown("#### 🎲 Classification Metrics (Performance Category)")
            col4, col5, col6 = st.columns(3)
            col4.metric("Accuracy", f"{accuracy:.4f}", help="Overall classification accuracy")
            col5.metric("Sensitivity (Recall)", f"{class_metrics.get('sensitivity', 0):.4f}", help="True Positive Rate")
            col6.metric("Specificity", f"{class_metrics.get('specificity', 0):.4f}", help="True Negative Rate")

            # Visualizations
            st.markdown("---")
            st.subheader("📊 Model Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Actual vs Predicted Scores")
                plot_actual_vs_predicted(y_test_reg, y_pred_reg, model_name)
            
            with col2:
                st.markdown("##### Residuals Distribution")
                plot_residuals(y_test_reg, y_pred_reg, model_name)

            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("##### Confusion Matrix")
                plot_confusion_matrix(cm, ['Low Performance', 'High Performance'])
            
            with col4:
                st.markdown("##### ROC Curve")
                # For ROC curve, use regression scores normalized to [0, 1] as probabilities
                # Map scores to probability of being high performance (>=70)
                y_score = (y_pred_reg - 0) / 100.0  # Assuming scores are 0-100
                y_score = np.clip(y_score, 0, 1)  # Ensure values are in [0, 1]
                plot_roc(y_test_cls, y_score)

            # Model comparison
            st.markdown("---")
            st.subheader("⚖️ Model Comparison")
            
            with st.spinner("Comparing all models..."):
                comparison_results = []
                
                models_to_compare = [
                    ("Linear Regression", LinearRegression()),
                    ("Decision Tree", DecisionTreeRegressor(random_state=42, max_depth=10)),
                    ("Gradient Boosting", GradientBoostingRegressor(random_state=42, n_estimators=100))
                ]

                for name, clf in models_to_compare:
                    clf.fit(X_train, y_train_reg)
                    y_pred_cmp = clf.predict(X_test)
                    mae_cmp, rmse_cmp, r2_cmp = compute_regression_metrics(y_test_reg, y_pred_cmp)
                    
                    # Binary classification metrics
                    y_pred_cls_cmp = (y_pred_cmp >= 70).astype(int)
                    acc_cmp, _, class_metrics_cmp = compute_classification_metrics(y_test_cls, y_pred_cls_cmp)
                    
                    # Calculate AUC
                    y_score_cmp = (y_pred_cmp - 0) / 100.0
                    y_score_cmp = np.clip(y_score_cmp, 0, 1)
                    fpr, tpr, _ = roc_curve(y_test_cls, y_score_cmp)
                    auc_score = auc(fpr, tpr)
                    
                    comparison_results.append({
                        "Model": name,
                        "MAE": f"{mae_cmp:.4f}",
                        "RMSE": f"{rmse_cmp:.4f}",
                        "R²": f"{r2_cmp:.4f}",
                        "Accuracy": f"{acc_cmp:.4f}",
                        "AUC": f"{auc_score:.4f}"
                    })

                cmp_df = pd.DataFrame(comparison_results)
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                # Plot comparison charts side by side
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Plot regression metrics
                    cmp_df_plot = cmp_df.copy()
                    cmp_df_plot['MAE'] = cmp_df_plot['MAE'].astype(float)
                    cmp_df_plot['RMSE'] = cmp_df_plot['RMSE'].astype(float)
                    cmp_df_plot['R²'] = cmp_df_plot['R²'].astype(float)
                    cmp_df_plot = cmp_df_plot.set_index("Model")
                    
                    fig_cmp1, ax_cmp1 = plt.subplots(figsize=(6, 4.5))
                    cmp_df_plot[["MAE", "RMSE"]].plot(kind="bar", ax=ax_cmp1, rot=0, color=['#FF6B6B', '#4ECDC4'])
                    ax_cmp1.set_ylabel("Error Value", fontsize=11, fontweight='bold')
                    ax_cmp1.set_xlabel("Model", fontsize=11, fontweight='bold')
                    ax_cmp1.set_title("Regression Metrics: MAE & RMSE", fontsize=12, fontweight='bold')
                    ax_cmp1.legend(loc="upper right")
                    ax_cmp1.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_cmp1, clear_figure=True)
                
                with col_chart2:
                    # Plot R² and AUC
                    cmp_df_plot2 = cmp_df.copy()
                    cmp_df_plot2['R²'] = cmp_df_plot2['R²'].astype(float)
                    cmp_df_plot2['Accuracy'] = cmp_df_plot2['Accuracy'].astype(float)
                    cmp_df_plot2['AUC'] = cmp_df_plot2['AUC'].astype(float)
                    cmp_df_plot2 = cmp_df_plot2.set_index("Model")
                    
                    fig_cmp2, ax_cmp2 = plt.subplots(figsize=(6, 4.5))
                    cmp_df_plot2[["R²", "Accuracy", "AUC"]].plot(kind="bar", ax=ax_cmp2, rot=0, ylim=(0, 1), color=['#95E1D3', '#F38181', '#AA96DA'])
                    ax_cmp2.set_ylabel("Score", fontsize=11, fontweight='bold')
                    ax_cmp2.set_xlabel("Model", fontsize=11, fontweight='bold')
                    ax_cmp2.set_title("Performance Metrics: R², Accuracy & AUC", fontsize=12, fontweight='bold')
                    ax_cmp2.legend(loc="lower right")
                    ax_cmp2.grid(True, alpha=0.3, axis='y')
                    ax_cmp2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random Baseline')
                    st.pyplot(fig_cmp2, clear_figure=True)

            # Save trained artifacts
            st.session_state["trained_model"] = model
            st.session_state["trained_scaler"] = scaler
            st.session_state["trained_features"] = feature_cols
            st.session_state["label_encoders"] = label_encoders
            st.session_state["feature_bounds"] = df[feature_cols].describe()

    st.markdown("---")
    st.subheader("🔮 Make a Prediction")
    if "trained_model" not in st.session_state:
        st.info("⚠️ Train a model first, then use this form to predict student performance.")
    else:
        trained_model = st.session_state["trained_model"]
        trained_scaler = st.session_state["trained_scaler"]
        trained_features = st.session_state["trained_features"]
        label_encoders = st.session_state["label_encoders"]

        st.write("Enter student demographic information to predict their expected average performance:")

        # Categorical inputs - only demographic features
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("👤 Gender", ["female", "male"], key="pred_gender")
            race_ethnicity = st.selectbox("🌍 Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"], key="pred_race")
        
        with col2:
            parental_education = st.selectbox("🎓 Parental Education", 
                                            ["some high school", "high school", "some college", 
                                             "associate's degree", "bachelor's degree", "master's degree"], key="pred_edu")
            lunch = st.selectbox("🍽️ Lunch Type", ["standard", "free/reduced"], key="pred_lunch")
        
        with col3:
            test_prep = st.selectbox("📚 Test Preparation Course", ["none", "completed"], key="pred_prep")

        predict_btn = st.button("🎯 Predict Performance", type="primary", use_container_width=True)
        if predict_btn:
            # Prepare input data - only demographic features
            input_data = {
                'gender': gender,
                'race/ethnicity': race_ethnicity,
                'parental level of education': parental_education,
                'lunch': lunch,
                'test preparation course': test_prep
            }
            
            # Encode categorical variables
            encoded_data = []
            for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
                encoded_data.append(label_encoders[col].transform([input_data[col]])[0])
            
            # Scale the input
            x = np.array(encoded_data, dtype=float).reshape(1, -1)
            if trained_scaler is not None:
                x = trained_scaler.transform(x)
            
            # Make prediction
            pred_score = trained_model.predict(x)[0]
            
            # Classify performance (binary)
            if pred_score >= 70:
                category = "🟢 High Performance"
                perf_emoji = "🌟"
            else:
                category = "🔴 Low Performance"
                perf_emoji = "⚠️"

            st.markdown("### Prediction Results")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("📊 Predicted Average Score", f"{pred_score:.2f}", help="Predicted average across math, reading, and writing")
            with col_res2:
                st.metric("🎯 Performance Category", category, help="High Performance: ≥70, Low Performance: <70")
            
            st.info(f"{perf_emoji} Based on the demographic factors, the model predicts this student will score approximately **{pred_score:.1f}** on average.")


if __name__ == "__main__":
    main()
