# House Sales in King County, USA

A Python data science project that analyzes and predicts residential house sale prices in King County, Washington (including Seattle) using exploratory data analysis, linear regression, Ridge regression, and polynomial feature engineering.

---

## Overview

This is a capstone-style assignment from IBM's Data Analysis course on Coursera. As a simulated Data Analyst for a Real Estate Investment Trust, the goal is to determine the market price of a house given a set of features. The notebook walks through the full data science workflow — from raw data ingestion to a tuned polynomial regression model.

**Dataset period:** May 2014 – May 2015  
**Source:** [Kaggle – House Sales Prediction](https://www.kaggle.com/harlfoxem/housesalesprediction) (slightly modified for course use)

---

## Project Structure

```
House_Sales_in_King_Count_USA.ipynb   # Main notebook
README.md
```

---

## Modules & Questions

### Module 1 — Importing Data
- Load the King County housing CSV into a pandas DataFrame
- **Q1:** Inspect column data types using `df.info()`

### Module 2 — Data Wrangling
- **Q2:** Drop irrelevant columns (`id`, `Unnamed: 0`) and replace missing values in `bedrooms` and `bathrooms` with column means

### Module 3 — Exploratory Data Analysis
- **Q3:** Count houses by number of floors using `value_counts()`
- **Q4:** Boxplot comparing price distributions for waterfront vs. non-waterfront properties
- **Q5:** Regression plot (`seaborn.regplot`) of `sqft_above` vs. `price` to assess correlation

### Module 4 — Model Development
- **Q6:** Simple linear regression — `sqft_living` → `price`, report R²
- **Q7:** Multiple linear regression using 11 features, report R²
- **Q8:** Sklearn `Pipeline` with `StandardScaler` → `PolynomialFeatures` → `LinearRegression`, report R²

### Module 5 — Model Evaluation and Refinement
- Train/test split (85% / 15%, `random_state=1`)
- **Q9:** Ridge regression (`alpha=0.1`) on the 11-feature set, report test R²
- **Q10:** 2nd-degree polynomial transform + Ridge regression (`alpha=0.1`), report test R²

---

## Dataset Features

| Variable | Description |
|----------|-------------|
| `price` | Sale price (prediction target) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Interior living space (sq ft) |
| `sqft_lot` | Lot size (sq ft) |
| `floors` | Number of floors |
| `waterfront` | Waterfront view (0 = No, 1 = Yes) |
| `view` | Number of times viewed |
| `condition` | Overall condition rating |
| `grade` | King County grading score |
| `sqft_above` | Above-ground sq ft (excludes basement) |
| `sqft_basement` | Basement sq ft |
| `yr_built` | Year built |
| `yr_renovated` | Year renovated |
| `zipcode` | ZIP code |
| `lat` / `long` | Coordinates |
| `sqft_living15` | Living area in 2015 (post-renovation) |
| `sqft_lot15` | Lot size in 2015 (post-renovation) |

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and wrangling |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, pipelines, and evaluation |

---

## Usage

### JupyterLite (browser-based)
The notebook uses `pyodide` / `pyfetch` to download the dataset automatically. Run all cells in order.

### Local (Jupyter / Anaconda)
Skip the `pyfetch` download cells and load the dataset directly:

```python
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(filepath)
```

Then run the remaining cells normally.

---

## Key Results (expected outputs)

| Model | Features | R² (approx.) |
|-------|----------|--------------|
| Simple linear regression | `sqft_living` | ~0.49 |
| Multiple linear regression | 11 features | ~0.66 |
| Pipeline (Polynomial + Scaling) | 11 features | ~0.75 |
| Ridge regression | 11 features | ~0.65 |
| Polynomial Ridge regression | 11 features (degree=2) | ~0.70+ |

---

## Authors

- [Joseph Santarcangelo](https://www.linkedin.com/in/joseph-s-50398b136/) — IBM
- [Michelle Carey](https://www.linkedin.com/in/michelleccarey/)
- [Mavis Zhou](https://www.linkedin.com/in/jiahui-mavis-zhou-a4537814a)

---

## License

© IBM Corporation 2020. All rights reserved.
