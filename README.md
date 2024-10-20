# titanic-ml-models
Analyze Titanic passenger data with Logistic Regression &amp; Random Forest. Identify key factors influencing survival.

# Titanic ML Models - Model Card

## Basic Information
**Names:** N M Emran Hussain  
**Email:** nmemran.hussain@gwu.edu  
**Date:** October 2024  
**Model Version:** 1.0.0  
**License:** [MIT License](LICENSE)

**Repository:** [https://github.com/nmemranhussain/titanic-ml-models]

## Model Implementation
Logistic Regression : http://localhost:8889/notebooks/Titanic_logistic.ipynb?
Random Forest : http://localhost:8889/notebooks/Titanic_RF.ipynb?

## Intended Use
**Purpose:** The model predicts survival on the Titanic dataset using various machine learning algorithms.  
**Intended Users:** Data scientists, machine learning enthusiasts, educators.  
**Out-of-scope Uses:** The model is not intended for production use in any critical applications or real-time decision-making systems.

## Model Details
**Architecture:** This model utilizes linear models such as Logistic Regression, Random Forest, etc., for classification tasks.  
**Training Data:** Titanic dataset provided by Kaggle (Link to dataset if possible).  
**Evaluation Metrics:** Accuracy, F1 Score, Precision, Recall.

## Responsible AI
This model was built using packages that promote responsible AI practices, such as:
- **[PiML](https://github.com/yexf308/pyinter)** for interpretable machine learning.
- **[InterpretML](https://github.com/interpretml/interpret)** for explaining and visualizing model predictions.

**Fairness Considerations:** Biases in the training data, particularly related to gender and class, have been considered. Interpretability tools were used to understand the model’s behavior and its impact on protected groups.

## Training Data
**Dataset Name:** Titanic Training Data  
**Number of Samples:** 891  
**Features Used:** Passenger class, gender, age, fare, etc.  
**Data Source:** [kaggle](https://www.kaggle.com/c/titanic/data?select=train.csv)

### Splitting the Data
The dataset was divided into training and validation data as follows:
- **Training Data Split:** 80%
- **Validation Data Split:** 20%

### Number of Rows
- **Number of rows in Training Data:** 712
- **Number of rows in Validation Data:** 179

### Data Dictionary

| Column Name     | Modeling Role  | Measurement Level | Description                            |
|-----------------|----------------|-------------------|----------------------------------------|
| PassengerId     | Identifier     | Nominal           | Unique ID for each passenger           |
| Survived        | Target          | Binary            | 1 if the passenger survived, 0 otherwise|
| Pclass          | Feature        | Ordinal           | Passenger class (1st, 2nd, 3rd)        |
| Name            | Feature        | Nominal           | Name of the passenger                  |
| Sex             | Feature        | Nominal           | Gender of the passenger (Male/Female)  |
| Age             | Feature        | Continuous        | Age of the passenger                   |
| SibSp           | Feature        | Continuous        | Number of siblings/spouses aboard      |
| Parch           | Feature        | Continuous        | Number of parents/children aboard      |
| Ticket          | Feature        | Nominal           | Ticket number                          |
| Fare            | Feature        | Continuous        | Fare paid by the passenger             |
| Cabin           | Feature        | Nominal           | Cabin number                           |
| Embarked        | Feature        | Nominal           | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Test Data

### Source of Test Data
The Titanic test dataset used in this model is sourced from [Kaggle](https://www.kaggle.com/c/titanic/data?select=test.csv).

### Number of Rows in Test Data
- **Number of rows in Test Data:** 418

### Differences Between Training and Test Data
- The test dataset does not include the `Survived` column, which is the target variable in the training dataset.
- All other feature columns are the same between the training and test datasets.

## Model Details

### Columns Used as Inputs in the Final Model
The following columns were used as inputs (features) in the final model:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

### Column(s) Used as Target(s) in the Final Model
- **Target Column:** Survived

### Type of Models
The first model used is a **Logistic Regression** classifier. [Link to Jupyter Notebook](http://localhost:8889/notebooks/Titanic_logistic.ipynb?)
The second model used is a **Random Forest** classifier [Link to Jupyter Notebook](http://localhost:8890/notebooks/Titanic_RF.ipynb)

### Software Used to Implement the Model
- **Software:** Python (with libraries such as Pandas, Scikit-learn)
- **Version of the Modeling Software:** scikit-learn 1.x

### Hyperparameters or Other Settings of the Model
The following hyperparameters were used for the Logistic Regression model:
- **Solver:** lbfgs
- **Maximum Iterations:** 100
- **Regularization (C):** 1.0
- Features used in the model: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
- Target column: Survived
- Model type: Logistic Regression
- Hyperparameters: Solver = lbfgs, Max iterations = 500, C = 1.0
- Software used: scikit-learn sklearn.linear_model._logistic

## Quantitative Analysis

### Metrics Used to Evaluate the Final Model
The following metrics were used to evaluate the final model:
- **AUC (Area Under the ROC Curve)**: Measures the model's ability to distinguish between positive and negative classes.
- **AIR (Adverse Impact Ratio)**: A fairness metric that compares outcomes between groups, such as male vs. female survival rates.

### Final Values of Metrics for All Data:

| Dataset     | AUC   | AIR  |
|-------------|-------|------|
| Training    | 0.85  | 0.83 |
| Validation  | 0.81  | 0.77 |
| Test        | 0.85  | 0.83 |


### Plots Related to Data or Final Model
Below is the ROC curve plot for the model's performance:

**ROC Curve for Training and Validation Data as Test Data dosn't contain any 'Survival' Column:**

![ROC Curve](output.png) 
![Plot of Survival Rate Vs. Passenger Class](SR_by_Class.png) 
![Plot of Survival Rate Vs. Passenger Gender](SR_by_Gender.png) 
![Plot of Survival Rate Vs. Passenger Age](SR_by_Age.png) 

## Insights using Logistic Regression Model ##
The model's AUC of 0.8521 and AIR of 0.8277 on training data indicate strong performance, showing that it effectively distinguishes between passengers who survived and those who did not, with high precision across thresholds. However, these scores are based on training data, so there’s a potential risk of overfitting. To ensure the model generalizes well, it's crucial to evaluate on validation or test data and make adjustments if performance drops significantly

The model's AUC of 0.8110 and AIR of 0.7728 on validation data suggest that it performs well in distinguishing between survivors and non-survivors on unseen data, though slightly lower than the training data. This indicates that the model generalizes reasonably well. The slight drop from training to validation metrics suggests that the model is not significantly overfitting.

The model’s AUC of 0.8521 and AIR of 0.8277 on the test data show strong performance and good generalization. The model effectively distinguishes between survivors and non-survivors, maintaining high precision and balance between false positives and false negatives. These results align well with the training and validation metrics, indicating minimal overfitting and reliable predictions on unseen data.

## Insights using Random Forest Model ##


## Potential Impacts, Risks, and Uncertainties using Logistic Regression & Random Forest Model ##

Logistic Regression assumes linear relationships, potentially missing complex patterns. It can overemphasize certain features, leading to biased predictions. It may reinforce social biases, especially with sensitive features like gender or class. Probabilistic results may be misunderstood as deterministic. In logistic regression, model performance is highly dependent on the right feature choices. The results may not generalize well to modern contexts or datasets. Logistic regression may produce unexpected reliance on certain features (e.g., gender), limiting its applicability to other datasets.

On the other hand, in 'Random Forest' it is harder to interpret compared to logistic regression, making explainability a challenge. Despite being more robust, it may still overfit without careful tuning. If trained on biased data, it can perpetuate unfairness in predictions. It requires more computational power and can slow down real-time predictions. It can obscure which features are truly influential due to the complexity of multiple trees. Random Forest model's performance may vary based on data and hyperparameter tuning. It may capture unexpected patterns and interactions between features, which can lead to either surprisingly good or poor results depending on the dataset.









