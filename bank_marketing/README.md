# Bank Marketing - Term Deposit Classification

This project experiments with classification models using data from the UCI Machine Learning Repository - Bank Marketing dataset. 

The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution.

**Goal:** The classification target is to predict if the client will **subscribe to a term deposit** (variable `y`).

### Dataset Citation
Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## Dataset Overview

The dataset includes various attributes related to client profiles and campaign interaction features:

- **Demographic & Socioeconomic Features:**  
  `age`, `job status`, `marital status`, `education level`, `prior defaults`, `bank balance`, `house ownership`, `prior loans`

- **Campaign-Related Features:**  
  `contact method`, `month and day of call`, `duration of call`, `number of prior campaigns`, `days since last call`, `number of previous calls`, `outcome of previous calls`

- **Target Variable:**  
  `y` – Indicates if the client subscribed to a term deposit (`yes`/`no`)

## Folder Structure
├── **data/processed** # Processed data<br>
├── **handling_imbalanced_labels** # Experimenting dealing with imbalanced data<br>
├── README.md # Project overview<br>
├── bank_marketing_preprocessing.py # Python script to preprocess data and store them in csv files<br>
└── Jupyter notebooks for each classification model

## Tasks Performed

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)  
- Feature Engineering: One Hot Encoding, Standard Scaling, 
- Model Training (More details below)  
- Model Evaluation (Accuracy, Precision, Recall, ROC-AUC)  
- Future Trials

## ML Models Tested
- SVC
- XGB
- Decision Tree
- Logistic Regression

## Results

### SVC

Recall: **0.18**<br>
Precision: **0.64**<br>
F1 Score: **0.28**<br>
Accuracy: **0.89**<br>

### XGB

Recall: **0.37**<br>
Precision: **0.65**<br>
F1 Score: **0.47**<br>
Accuracy: **0.90**<br>
AUC: **0.90**<br>

### Decision Tree

Recall: **0.39**<br>
Precision: **0.64**<br>
F1 Score: **0.48**<br>
Accuracy: **0.90**<br>
AUC: **0.89**<br>

### Logistic Regression

Recall: **0.80**<br>
Precision: **0.39**<br>
F1 Score: **0.53**<br>
Accuracy: **0.86**<br>
AUC: **0.89**<br>

## Insights

## Future Works