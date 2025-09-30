# Avocado Price Prediction using Machine Learning

This project applies machine learning techniques to predict avocado prices across U.S. regions and time periods. It focuses on analyzing sales patterns, seasonal trends, and regional differences, and demonstrates the application of regression models with preprocessing and hyperparameter tuning.

> Developed as a final project for CS550 – Machine Learning.

---

## Objectives
- Analyze avocado price fluctuations across **regions and years**  
- Preprocess sales and packaging features for modeling  
- Train regression models to predict **AveragePrice**  
- Compare model performance with and without hyperparameter tuning  
- Visualize results and interpret business implications  

---

## Dataset
- **Source:** [Kaggle – Avocado Prices Dataset](https://www.kaggle.com/neuromusic/avocado-prices)  
- **Size:** ~18,000 records (2015–2018)  
- **Shape:** (18,249 rows, 13 columns)  
- **Target variable:** `AveragePrice`  
- **Features:**  
  - Sales volumes (Small, Large, XLarge, Bags)  
  - Product type (Organic vs Conventional)  
  - Region  
  - Year  

> No missing values were present in the dataset.

---

## Workflow
1. **Data Collection** – Imported Kaggle dataset  
2. **Data Preprocessing** –  
   - Applied **StandardScaler** to numeric features  
   - Used **OneHotEncoder** for categorical features via `ColumnTransformer`  
3. **Exploratory Data Analysis (EDA)** – Visualized distributions, regional trends, and time-series patterns  
4. **Model Training** – Trained baseline models: Ridge, Lasso, Decision Tree, Random Forest, XGBoost  
5. **Hyperparameter Tuning** – Applied **GridSearchCV (5-fold)** for DT, RF, and XGBoost  
6. **Final Evaluation** – Compared models using **MAE, RMSE, and R²**  
7. **Result Interpretation** – Scatter plots (Actual vs Predicted), performance comparison table  

---

## Methods & Tools
- **Python Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost  
- **Models:** Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor  
- **Evaluation Metrics:** MAE, RMSE, R²  
- **Hyperparameter Optimization:** GridSearchCV  

---

## Key Findings
- Most avocado prices fall between **$1.0 and $1.5**; high prices are less common  
- Seasonal consumption spikes appear in **January–February** (likely tied to events like the Super Bowl)  
- **Top regions by price:** Hartford-Springfield, San Francisco  
- **Top regions by consumption:** California and Western U.S.  

---

## Model Performance
| Model            | R² Score | RMSE   |
|------------------|----------|--------|
| **Random Forest** | **0.8537** | 0.281  |
| XGBoost          | 0.8145   | 0.2579 |
| Decision Tree    | 0.7036   | -      |
| Ridge            | Moderate | -      |
| Lasso            | ~0.0 (underfit) | -  |

- **Random Forest** achieved the best predictive accuracy (highest R², low RMSE).  
- **XGBoost** also showed strong predictive capability, slightly lower R² but best RMSE.  
- **Lasso** underfit the data, proving unsuitable for this task.  

---

## Repository Contents
- `Final_project_tasmita_Notebook_IPYNB.ipynb` → Full analysis, preprocessing, model training, and evaluation  
- `avocado.csv` → Dataset used for training and testing  
- `CS550_FINAL_tanha_PPT.pdf` → Final project presentation slides  

---


