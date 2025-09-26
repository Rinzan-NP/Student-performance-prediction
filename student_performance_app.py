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
    
    # Prepare features
    feature_columns = ['gender_encoded', 'race/ethnicity_encoded', 'parental level of education_encoded', 
                      'lunch_encoded', 'test preparation course_encoded', 'math score', 'reading score', 'writing score']
    
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
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
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
    st.set_page_config(page_title="Student Performance Prediction", layout="wide")
    st.title("🎓 Student Performance Prediction System")
    st.caption("Linear Regression · Decision Tree · Gradient Boosting | Metrics: MAE, RMSE, R²")

    with st.sidebar:
        st.header("Data & Preprocess")
        data_source = st.radio("Dataset source", ["Included StudentsPerformance.csv", "Upload CSV"], index=0)
        uploaded = None
        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload student performance CSV", type=["csv"])

        st.markdown("---")
        st.header("Model")
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
        test_size = st.slider("Test size (%)", min_value=10, max_value=50, value=20, step=5)
        random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)
        scale_features = st.checkbox("Standardize features (recommended)", value=True)
        run_button = st.button("Train & Evaluate", type="primary")

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

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Shape**")
        st.write(df.shape)
        st.markdown("**Score Statistics**")
        math_mean = df["math score"].mean()
        reading_mean = df["reading score"].mean()
        writing_mean = df["writing score"].mean()
        st.write(f"Math: {math_mean:.1f}, Reading: {reading_mean:.1f}, Writing: {writing_mean:.1f}")
    
    with right:
        st.markdown("**Summary Stats**")
        score_cols = ["math score", "reading score", "writing score"]
        st.dataframe(df[score_cols].describe().T, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Correlations")
    corr = df[["math score", "reading score", "writing score"]].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt='.2f')
    ax.set_title("Score Correlations")
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
            st.subheader("Performance Metrics")
            
            # Regression metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.3f}")
            col2.metric("RMSE", f"{rmse:.3f}")
            col3.metric("R² Score", f"{r2:.3f}")
            
            # Classification metrics
            col4, col5, col6 = st.columns(3)
            col4.metric("Classification Accuracy", f"{accuracy:.3f}")
            col5.metric("Sensitivity (TPR)", f"{class_metrics.get('sensitivity', 0):.3f}")
            col6.metric("Specificity (TNR)", f"{class_metrics.get('specificity', 0):.3f}")

            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Actual vs Predicted")
                plot_actual_vs_predicted(y_test_reg, y_pred_reg, model_name)
            
            with col2:
                st.subheader("Residuals Plot")
                plot_residuals(y_test_reg, y_pred_reg, model_name)

            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(cm, ['Low Performance', 'High Performance'])
            
            with col4:
                st.subheader("ROC Curve")
                # For ROC curve, we need probability scores
                # We'll use the regression scores as probability proxies
                y_score = (y_pred_reg - y_pred_reg.min()) / (y_pred_reg.max() - y_pred_reg.min())
                plot_roc(y_test_cls, y_score)

            # Model comparison
            st.subheader("Model Comparison")
            comparison_results = []
            
            models_to_compare = [
                ("Linear Regression", LinearRegression()),
                ("Decision Tree", DecisionTreeRegressor(random_state=42)),
                ("Gradient Boosting", GradientBoostingRegressor(random_state=42))
            ]

            for name, clf in models_to_compare:
                clf.fit(X_train, y_train_reg)
                y_pred_cmp = clf.predict(X_test)
                mae_cmp, rmse_cmp, r2_cmp = compute_regression_metrics(y_test_reg, y_pred_cmp)
                
                # Binary classification metrics
                y_pred_cls_cmp = (y_pred_cmp >= 70).astype(int)
                acc_cmp, _, class_metrics_cmp = compute_classification_metrics(y_test_cls, y_pred_cls_cmp)
                
                comparison_results.append({
                    "Model": name,
                    "MAE": mae_cmp,
                    "RMSE": rmse_cmp,
                    "R²": r2_cmp,
                    "Accuracy": acc_cmp,
                    "Sensitivity": class_metrics_cmp.get('sensitivity', 0),
                    "Specificity": class_metrics_cmp.get('specificity', 0),
                })

            cmp_df = pd.DataFrame(comparison_results)
            cmp_df = cmp_df.set_index("Model")

            # Plot regression metrics
            fig_cmp1, ax_cmp1 = plt.subplots(figsize=(7.5, 4.2))
            cmp_df[["MAE", "RMSE", "R²"]].plot(kind="bar", ax=ax_cmp1, rot=0)
            ax_cmp1.set_ylabel("Score")
            ax_cmp1.set_title("Model Comparison - Regression Metrics")
            ax_cmp1.legend(loc="upper right")
            st.pyplot(fig_cmp1, clear_figure=True)
            
            # Plot classification metrics
            fig_cmp2, ax_cmp2 = plt.subplots(figsize=(7.5, 4.2))
            cmp_df[["Accuracy", "Sensitivity", "Specificity"]].plot(kind="bar", ax=ax_cmp2, rot=0, ylim=(0, 1))
            ax_cmp2.set_ylabel("Score")
            ax_cmp2.set_title("Model Comparison - Classification Metrics")
            ax_cmp2.legend(loc="lower right")
            st.pyplot(fig_cmp2, clear_figure=True)

            # Save trained artifacts
            st.session_state["trained_model"] = model
            st.session_state["trained_scaler"] = scaler
            st.session_state["trained_features"] = feature_cols
            st.session_state["label_encoders"] = label_encoders
            st.session_state["feature_bounds"] = df[feature_cols].describe()

    st.markdown("---")
    st.subheader("Manual Prediction")
    if "trained_model" not in st.session_state:
        st.info("Train a model first, then use this form to predict student performance.")
    else:
        trained_model = st.session_state["trained_model"]
        trained_scaler = st.session_state["trained_scaler"]
        trained_features = st.session_state["trained_features"]
        label_encoders = st.session_state["label_encoders"]
        bounds = st.session_state["feature_bounds"]

        st.write("Enter student information to predict performance:")

        # Categorical inputs
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["female", "male"])
            race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
            parental_education = st.selectbox("Parental Education", 
                                            ["some high school", "high school", "some college", 
                                             "associate's degree", "bachelor's degree", "master's degree"])
        
        with col2:
            lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
            test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
            math_score = st.slider("Math Score", 0, 100, 70, step=1)
            reading_score = st.slider("Reading Score", 0, 100, 70, step=1)
            writing_score = st.slider("Writing Score", 0, 100, 70, step=1)

        predict_btn = st.button("Predict Performance", type="primary")
        if predict_btn:
            # Prepare input data
            input_data = {
                'gender': gender,
                'race/ethnicity': race_ethnicity,
                'parental level of education': parental_education,
                'lunch': lunch,
                'test preparation course': test_prep,
                'math score': math_score,
                'reading score': reading_score,
                'writing score': writing_score
            }
            
            # Encode categorical variables
            encoded_data = []
            for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
                encoded_data.append(label_encoders[col].transform([input_data[col]])[0])
            
            # Add numerical scores
            encoded_data.extend([math_score, reading_score, writing_score])
            
            # Scale the input
            x = np.array(encoded_data, dtype=float).reshape(1, -1)
            if trained_scaler is not None:
                x = trained_scaler.transform(x)
            
            # Make prediction
            pred_score = trained_model.predict(x)[0]
            
            # Classify performance (binary)
            if pred_score >= 70:
                category = "High Performance"
                color = "green"
            else:
                category = "Low Performance"
                color = "red"

            st.success(f"Predicted Average Score: **{pred_score:.2f}**")
            st.markdown(f"**Performance Category:** <span style='color: {color}'>{category}</span>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
