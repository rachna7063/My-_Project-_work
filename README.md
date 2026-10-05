# My-_Project-_work
Python project
# Socioeconomic Classification & Optimization Pipeline

## 📊 Project Overview
This repository features an end-to-end data analytics and predictive modeling pipeline developed on the **Adult Income (Census) Dataset**. As a Data Analyst, the primary objective was to clean, preprocess, and evaluate demographic and financial factors to accurately classify whether an individual's annual income (`SalStat`) is **less than or equal to \$50,000** (`0`) or **greater than \$50,000** (`1`).

---

## 🛠️ Data Cleaning & Analytics Architecture

### 1. Missing Value Imputation & Structural Auditing
* **Implicit Null Parsing:** Identified and handled hidden structural missing values (` ?`, `NaN`, `?`) natively during the data ingestion lifecycle via `pd.read_csv()`.
* **Schema Standardization:** Audited and corrected tracking discrepancies by renaming structural attributes (e.g., converting the index-level `zage` back to standard `age`).
* **Memory Management:** Enforced structural array copying via `.dropna(axis=0).copy()`. Utilizing an independent dataset copy successfully isolated data slices and permanently suppressed pandas `SettingWithCopyWarning` execution alerts during label formatting.

### 2. Feature Selection & Categorical Transformations
* **Feature Selection:** Conducted dimensional trimming by dropping secondary demographic constraints (`gender`, `nativecountry`, `race`). This optimized the matrix to prioritize core actionable socioeconomic indicators.
* **One-Hot Encoding:** Binarized non-numeric categorical attributes using `pd.get_dummies(drop_first=True)` to prevent mathematical multicollinearity.
* **Target Mapping:** Normalized target classification strings into pure mathematical binary vectors (`0` and `1`).

### 3. Feature Scaling & Statistical Convergence
* **The Convergence Challenge:** Unscaled numeric values with varying spatial magnitudes (such as `capitalgain` and `hoursperweek`) caused severe optimization bottlenecks. The initial Logistic Regression model failed to find global minimums, throwing a `ConvergenceWarning: lbfgs failed to converge` even when capped at `max_iter=2000`.
* **The Solution:** Applied a `StandardScaler` to bring continuous numeric attributes down to standard normal distributions (mean = 0, variance = 1).
* **Impact:** Feature scaling completely neutralized the optimization lag, allowing the linear model to achieve instant mathematical convergence while radically stabilizing the distance metrics of the KNN model.

---

## 📈 Hyperparameter Tuning & Model Performance

To identify the absolute optimal baseline for the distance-based algorithm, a parameter grid loop was executed evaluating neighborhood thresholds (\(K\)) ranging from 1 to 10 on the scaled feature space.

### **Analytical Insights from K-Value Optimization:**
* **The most optimal (best) neighborhood structure was determined at K = 9.**
* At this intersection, total classification error reaches its absolute minimum, yielding **approximately 1,524 misclassified samples** on the test split.
* The variance curve showed high distortion at \(K = 1\) (~1,838 misclassifications). The error rate continuously decreased as neighbor parameters widened, indicating stable performance gains without overfitting.

### **Final Model Evaluation Matrix**

| Model Identifier | Feature Space Dimension | Feature Data State | Pipeline Status / Accuracy Metric |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | All System Features | Unscaled Raw Data | ❌ Optimization Failed (`ConvergenceWarning`) |
| **Logistic Regression** | Selected Socioeconomic Features | Scaled Normalized Data | ✅ Successfully Converged / Trained |
| **KNN Classifier** | Selected Socioeconomic Features | Scaled Normalized Data (\(K=9\)) | 🏆 **83.16% Final Accuracy (Optimal Model)** |

---

##  Technical Infrastructure
* **Execution Environment:** Google Colab / Jupyter Notebook
* **Core Analytics & Math Stack:** Pandas, NumPy, Scikit-Learn
* **Data Visualization Suites:** Matplotlib, Seaborn
* **Model Serialization:** Joblib (Serialized `.pkl` objects for rapid deployment pipeline)
