# Task 2: End-to-End ML Pipeline for Customer Churn Prediction

## Objective
Build a production-ready, reusable machine learning pipeline using scikit-learn's Pipeline API to predict customer churn in a telecommunications company.



## Dataset
**Source**: Telco Customer Churn Dataset  
**URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Size**: 7,043 customers × 21 features

**Features**:
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, InternetService, OnlineSecurity, etc.
- **Account**: Contract, PaymentMethod, MonthlyCharges, TotalCharges
- **Target**: Churn (Yes/No)

**Class Distribution**:
- No Churn: 73.5%
- Churn: 26.5% (imbalanced)

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Programming language |
| **pandas** | Data manipulation |
| **numpy** | Numerical computing |
| **scikit-learn** | ML pipeline, models, metrics |
| **joblib** | Model serialization |
| **matplotlib/seaborn** | Visualization |

---


## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```


## Output Files

1. **churn_prediction_pipeline.pkl** (42 MB)
   - Complete pipeline ready for production
   - Includes preprocessing + trained model
   
2. **preprocessing_info.pkl** (1 KB)
   - Feature names
   - Model type
   - Performance metrics

3. **Console Output**
   - Step-by-step progress
   - Evaluation metrics
   - Classification report

---
