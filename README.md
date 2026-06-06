\# HR Employee Attrition Prediction



A supervised machine learning project that predicts whether an employee will leave the company, using HR data and a Random Forest classifier optimized with GridSearchCV.



\---



\## Problem Statement



Employee turnover is costly for organizations. This project builds a binary classification model to predict employee attrition (`left = 1` means the employee left, `left = 0` means they stayed), enabling HR teams to take proactive retention measures.



\---



\## Dataset



\*\*File:\*\* `HR\_data.csv`  

\*\*Records:\*\* 14,999 employees  

\*\*Target variable:\*\* `left` (binary: 0 or 1)



| Column | Type | Description |

|---|---|---|

| `satisfaction\_level` | float | Employee satisfaction score (0–1) |

| `last\_evaluation` | float | Last performance evaluation score |

| `number\_project` | int | Number of projects assigned |

| `average\_montly\_hours` | int | Average monthly working hours |

| `time\_spend\_company` | int | Years spent at the company |

| `Work\_accident` | int | Whether the employee had a work accident (0/1) |

| `left` | int | \*\*Target\*\* — whether the employee left (0/1) |

| `promotion\_last\_5years` | int | Promoted in the last 5 years (0/1) |

| `Department` | object | Employee's department |

| `salary` | object | Salary level (low / medium / high) |



\---



\## Project Pipeline



\### 1. Import Libraries

```python

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model\_selection import train\_test\_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear\_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy\_score, classification\_report, confusion\_matrix

```



\### 2. Data Loading

```python

df = pd.read\_csv('HR\_data.csv')

```



\### 3. Exploratory Data Analysis (EDA)



Correlation of each feature with the target `left`:



| Feature | Correlation |

|---|---|

| `satisfaction\_level` | \*\*−0.39\*\* (strongest predictor) |

| `time\_spend\_company` | +0.14 |

| `average\_montly\_hours` | +0.07 |

| `Work\_accident` | −0.15 |

| `promotion\_last\_5years` | −0.06 |



Key findings:

\- Employees with low satisfaction are far more likely to leave.

\- Employees with low salaries show higher attrition rates.



\### 4. Feature Engineering



Selected features based on EDA results:



```python

data = df\[\['satisfaction\_level', 'average\_montly\_hours', 'promotion\_last\_5years', 'salary']]

```



The categorical `salary` column (low/medium/high) was encoded using \*\*One-Hot Encoding\*\*:



```python

dummies = pd.get\_dummies(data\['salary'])

data\_dummies = pd.concat(\[data, dummies], axis='columns').drop('salary', axis='columns')

```



Final feature set `X`: `satisfaction\_level`, `average\_montly\_hours`, `promotion\_last\_5years`, `high`, `low`, `medium`  

Target `y`: `df\['left']`



\### 5. Preprocessing \& Model Pipeline



A `sklearn Pipeline` was used to chain preprocessing and modeling steps cleanly:



```python

preprocessor = ColumnTransformer(transformers=\[

&#x20;   ('num', StandardScaler(), numerical\_features),

&#x20;   ('cat', OneHotEncoder(handle\_unknown='ignore'), categorical\_features)

])

```



Train/test split: \*\*30% train / 70% test\*\*



```python

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, train\_size=0.3)

```



\### 6. Model 1 — Logistic Regression



A simple baseline model was trained first to establish a performance reference.



```python

model = LogisticRegression()

model.fit(X\_train, y\_train)

```



\### 7. Model 2 — Random Forest



A `RandomForestClassifier` was added to the pipeline for a stronger model:



```python

rf\_pipeline = Pipeline(steps=\[

&#x20;   ('preprocessor', preprocessor),

&#x20;   ('classifier', RandomForestClassifier(random\_state=42))

])

```



\*\*Baseline Random Forest Results (before tuning):\*\*



| Metric | Score |

|---|---|

| Accuracy | 78.05% |

| Precision (class 1) | 0.58 |

| Recall (class 1) | 0.25 |

| F1-score (class 1) | 0.35 |



\### 8. Hyperparameter Tuning — GridSearchCV



`GridSearchCV` was used to find the best hyperparameters via 5-fold cross-validation:



```python

param\_grid = {

&#x20;   'classifier\_\_n\_estimators': \[50, 100, 200],

&#x20;   'classifier\_\_max\_depth': \[None, 10, 20]

}



grid\_search = GridSearchCV(

&#x20;   estimator=rf\_pipeline,

&#x20;   param\_grid=param\_grid,

&#x20;   cv=5,

&#x20;   scoring='accuracy',

&#x20;   n\_jobs=-1

)

grid\_search.fit(X\_train, y\_train)

```



\*\*Best parameters found:\*\* `max\_depth=10`, `n\_estimators=200`  

\*\*Best cross-validation score:\*\* 91.11%



\---



\## Final Results



| Metric | Score |

|---|---|

| \*\*Final Accuracy\*\* | \*\*91.67%\*\* |

| Precision (stayed) | 0.91 |

| Recall (stayed) | 0.99 |

| Precision (left) | 0.94 |

| Recall (left) | 0.69 |

| F1-score (left) | 0.80 |



The tuned Random Forest model improved accuracy from \*\*78%\*\* to \*\*91.6%\*\* — a significant gain from hyperparameter optimization alone.



\---



\## Project Structure



```

├── HR\_data.csv          # Dataset

├── notebook.ipynb       # Main Jupyter notebook

├── rf\_confusion\_matrix.png

└── README.md

```



\---



\## How to Run



1\. Clone the repository and install dependencies:

```bash

pip install pandas numpy matplotlib seaborn scikit-learn

```



2\. Place `HR\_data.csv` in the project directory.



3\. Open and run the Jupyter notebook:

```bash

jupyter notebook notebook.ipynb

```



\---



\## Key Concepts Used



\- \*\*Supervised Learning\*\* — Binary classification

\- \*\*Pipeline\*\* — Chaining preprocessing and model in one object

\- \*\*StandardScaler\*\* — Normalizing numerical features

\- \*\*One-Hot Encoding\*\* — Converting categorical salary to numeric

\- \*\*Random Forest\*\* — Ensemble of decision trees

\- \*\*GridSearchCV\*\* — Exhaustive hyperparameter search with cross-validation

\- \*\*Classification Report\*\* — Precision, Recall, F1-score per class

\- \*\*Confusion Matrix\*\* — Visualizing true/false positives and negatives

