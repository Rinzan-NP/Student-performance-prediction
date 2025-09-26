# 🎓 Student Performance Prediction System

A comprehensive machine learning application for predicting student examination performance using internal assessment data. This system implements multiple regression algorithms, binary classification, and provides detailed visualizations for educational data analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Graphs and Visualizations Explained](#graphs-and-visualizations-explained)
- [Performance Metrics](#performance-metrics)
- [Models Implemented](#models-implemented)
- [Dataset Information](#dataset-information)
- [How to Use](#how-to-use)
- [Technical Details](#technical-details)

## 🎯 Overview

This application predicts student performance using machine learning algorithms based on demographic information and test scores. It provides both regression analysis (predicting exact scores) and binary classification (High/Low performance categories).

## ✨ Features

- **Multiple ML Models**: Linear Regression, Decision Tree, Gradient Boosting
- **Binary Classification**: High Performance (≥70%) vs Low Performance (<70%)
- **Comprehensive Visualizations**: 6 different charts and graphs
- **Performance Metrics**: MAE, RMSE, R², Accuracy, Sensitivity, Specificity
- **Interactive Predictions**: Real-time student performance prediction
- **Model Comparison**: Side-by-side performance analysis

## 🚀 Installation

1. **Install required packages:**
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Ensure you have the dataset:**
   - Place `StudentsPerformance.csv` in the project directory

3. **Run the application:**
   ```bash
   streamlit run student_performance_app.py
   ```

## 💻 Usage

1. **Load Data**: Choose between included dataset or upload your own CSV
2. **Select Model**: Choose from Linear Regression, Decision Tree, or Gradient Boosting
3. **Configure Parameters**: Adjust model-specific hyperparameters
4. **Train & Evaluate**: Click "Train & Evaluate" to run the analysis
5. **View Results**: Examine metrics, visualizations, and model comparisons
6. **Make Predictions**: Use the manual prediction form for new students

## 📊 Graphs and Visualizations Explained

### 1. **Score Correlations Heatmap**
- **Purpose**: Shows the relationship between Math, Reading, and Writing scores
- **What it shows**: 
  - Values range from -1 to +1
  - +1 = Perfect positive correlation
  - 0 = No correlation
  - -1 = Perfect negative correlation
- **Interpretation**: Higher values indicate that students who perform well in one subject tend to perform well in others
- **Color coding**: Red = positive correlation, Blue = negative correlation

### 2. **Actual vs Predicted Scatter Plot**
- **Purpose**: Evaluates how well the model predicts actual student scores
- **What it shows**:
  - X-axis: Actual average scores from the dataset
  - Y-axis: Predicted scores from the model
  - Red dashed line: Perfect prediction line (y=x)
- **Interpretation**: 
  - Points close to the red line = Good predictions
  - Points scattered far from line = Poor predictions
  - Tight clustering = Consistent model performance

### 3. **Residuals Plot**
- **Purpose**: Analyzes prediction errors to check model assumptions
- **What it shows**:
  - X-axis: Predicted scores
  - Y-axis: Residuals (Actual - Predicted)
  - Red dashed line: Zero error line
- **Interpretation**:
  - Random scatter around zero = Good model
  - Patterns or trends = Model bias
  - Increasing/decreasing spread = Heteroscedasticity

### 4. **Confusion Matrix (2x2)**
- **Purpose**: Shows binary classification performance
- **What it shows**:
  - **True Negatives (TN)**: Correctly predicted Low Performance
  - **False Positives (FP)**: Incorrectly predicted High Performance
  - **False Negatives (FN)**: Incorrectly predicted Low Performance
  - **True Positives (TP)**: Correctly predicted High Performance
- **Interpretation**:
  - Higher diagonal values = Better classification
  - Off-diagonal values = Classification errors

### 5. **ROC Curve (Receiver Operating Characteristic)**
- **Purpose**: Evaluates binary classification performance across all thresholds
- **What it shows**:
  - X-axis: False Positive Rate (1 - Specificity)
  - Y-axis: True Positive Rate (Sensitivity)
  - Diagonal line: Random classifier performance
  - AUC Score: Area Under the Curve
- **Interpretation**:
  - Curve closer to top-left = Better performance
  - AUC = 1.0 = Perfect classifier
  - AUC = 0.5 = Random classifier
  - AUC > 0.7 = Good performance

### 6. **Model Comparison Charts**
- **Regression Metrics Chart**:
  - Shows MAE, RMSE, and R² for all three models
  - Lower MAE/RMSE = Better regression performance
  - Higher R² = Better model fit
- **Classification Metrics Chart**:
  - Shows Accuracy, Sensitivity, and Specificity
  - All values range from 0 to 1
  - Higher values = Better classification performance

## 📈 Performance Metrics

### **Regression Metrics**
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
  - Range: 0 to ∞
  - Lower is better
  - Units: Same as target variable (score points)

- **RMSE (Root Mean Square Error)**: Square root of average squared differences
  - Range: 0 to ∞
  - Lower is better
  - More sensitive to large errors than MAE

- **R² Score (Coefficient of Determination)**: Proportion of variance explained by the model
  - Range: 0 to 1
  - Higher is better
  - 1.0 = Perfect prediction, 0.0 = No better than mean

### **Classification Metrics**
- **Accuracy**: Overall correct predictions
  - Range: 0 to 1
  - Higher is better
  - Formula: (TP + TN) / (TP + TN + FP + FN)

- **Sensitivity (True Positive Rate)**: Correctly identified high performers
  - Range: 0 to 1
  - Higher is better
  - Formula: TP / (TP + FN)

- **Specificity (True Negative Rate)**: Correctly identified low performers
  - Range: 0 to 1
  - Higher is better
  - Formula: TN / (TN + FP)

## 🤖 Models Implemented

### 1. **Linear Regression**
- **Type**: Linear relationship modeling
- **Strengths**: Fast, interpretable, good baseline
- **Weaknesses**: Assumes linear relationships
- **Best for**: When features have linear relationships with target

### 2. **Decision Tree**
- **Type**: Non-linear pattern recognition
- **Strengths**: Handles non-linear relationships, feature importance
- **Weaknesses**: Prone to overfitting
- **Best for**: Complex feature interactions, interpretable rules

### 3. **Gradient Boosting**
- **Type**: Ensemble learning
- **Strengths**: High accuracy, handles complex patterns
- **Weaknesses**: More complex, longer training time
- **Best for**: Best overall performance, complex datasets

## 📊 Dataset Information

### **Features Used:**
- **Demographics**: Gender, Race/Ethnicity, Parental Education Level
- **Academic Background**: Lunch Type, Test Preparation Course
- **Test Scores**: Math Score, Reading Score, Writing Score

### **Target Variables:**
- **Regression Target**: Average Score (Math + Reading + Writing) / 3
- **Classification Target**: Binary (High ≥70%, Low <70%)

### **Data Preprocessing:**
- Categorical variables encoded using Label Encoding
- Features standardized using StandardScaler
- Train-test split: 80% training, 20% testing

## 🎯 How to Use

### **Step 1: Data Loading**
- Select "Included StudentsPerformance.csv" or upload your own CSV
- Ensure your CSV has the required columns

### **Step 2: Model Configuration**
- Choose your preferred model
- Adjust hyperparameters (if applicable)
- Set test size and random state

### **Step 3: Training & Evaluation**
- Click "Train & Evaluate"
- Wait for model training to complete
- Review performance metrics and visualizations

### **Step 4: Model Comparison**
- Compare all three models side-by-side
- Identify the best performing model
- Analyze different metric perspectives

### **Step 5: Manual Prediction**
- Fill in student information
- Click "Predict Performance"
- View predicted score and performance category

## 🔧 Technical Details

### **Dependencies:**
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
```

### **File Structure:**
```
student-performance-prediction/
├── student_performance_app.py    # Main application
├── StudentsPerformance.csv       # Dataset
├── requirements.txt              # Dependencies
└── README.md                    # Documentation
```

### **Key Functions:**
- `load_data()`: Loads and caches dataset
- `prepare_features()`: Encodes categorical variables and prepares features
- `compute_regression_metrics()`: Calculates MAE, RMSE, R²
- `compute_classification_metrics()`: Calculates accuracy, sensitivity, specificity
- `plot_*()`: Various visualization functions
- `build_model()`: Creates specified ML model

## 📝 Example Results

### **Typical Performance (Example):**
- **Linear Regression**: MAE: 4.2, RMSE: 5.7, R²: 0.89
- **Decision Tree**: MAE: 3.5, RMSE: 4.9, R²: 0.92
- **Gradient Boosting**: MAE: 2.9, RMSE: 4.1, R²: 0.94

### **Classification Performance:**
- **Accuracy**: 0.85-0.92
- **Sensitivity**: 0.80-0.90
- **Specificity**: 0.85-0.95

## 🎓 Educational Use Cases

1. **Early Intervention**: Identify students at risk of poor performance
2. **Resource Allocation**: Target support programs effectively
3. **Curriculum Planning**: Understand factors affecting performance
4. **Policy Making**: Data-driven educational decisions
5. **Research**: Academic performance analysis

## 🔍 Troubleshooting

### **Common Issues:**
1. **Missing Dependencies**: Install all required packages
2. **Dataset Format**: Ensure CSV has correct column names
3. **Memory Issues**: Reduce dataset size or model complexity
4. **Slow Performance**: Use smaller test size or fewer estimators

### **Performance Tips:**
- Use feature scaling for better model performance
- Experiment with different hyperparameters
- Consider feature engineering for better results
- Use cross-validation for robust evaluation

## 📞 Support

For questions or issues:
1. Check this README for common solutions
2. Review the code comments for technical details
3. Ensure all dependencies are properly installed
4. Verify dataset format and content

## 🎯 Future Enhancements

- [ ] Additional ML models (Random Forest, SVM, Neural Networks)
- [ ] Feature importance analysis
- [ ] Cross-validation implementation
- [ ] Export results functionality
- [ ] Advanced visualizations
- [ ] Real-time data integration

---

**Keywords**: Student Performance Prediction, Machine Learning, Educational Data Mining, Binary Classification, Regression Analysis, ROC Curve, Confusion Matrix, MAE, RMSE, R² Score

**Version**: 1.0.0  
**Last Updated**: 2024
