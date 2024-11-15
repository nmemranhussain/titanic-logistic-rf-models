# Titanic ML Models - Model Card

Analyze Titanic passenger data with Logistic Regression &amp; Random Forest. Identify key factors influencing survival.

## Basic Information
**Names:** N M Emran Hussain  
**Email:** nmemranhussain2023@gmail.com  
**Date:** October 2024  
**Model Version:** 1.0.0  
**License:** [MIT License](LICENSE)

## Intended Use
**Purpose:** The model predicts survival on the Titanic dataset using various machine learning algorithms.  
**Intended Users:** Data Analysts, Data scientists, machine learning enthusiasts, educators.  
**Out-of-scope Uses:** The model is not intended for production use in any critical applications or real-time decision-making systems.

## Training Data
**Dataset Name:** Titanic Training Data  
**Number of Samples:** 891  
**Features Used:** Passenger class, gender, age, fare, etc.  
**Data Source:** [kaggle](https://www.kaggle.com/c/titanic/data?select=train.csv)

### Splitting the Data for logistic regression model
The dataset was divided into training and validation data as follows:
- **Training Data Split:** 80%
- **Validation Data Split:** 20%

### Data Dictionary

| Column Name     | Modeling Role  | Measurement Level | Description                            |
|-----------------|----------------|-------------------|----------------------------------------|
| PassengerId     | Identifier     | Nominal           | Unique ID for each passenger           |
| Survived        | Target         | Binary            | 1 if the passenger survived, 0 otherwise|
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
- The Titanic test dataset used in this model is sourced from [Kaggle](https://www.kaggle.com/c/titanic/data?select=gender_submission.csv).

### Number of Rows in Test Data
- **Number of rows in Test Data:** 418

### Differences Between Training and Test Data
- The training data includes the target variable (Survived), allowing us to train and evaluate the model, while the test data lacks this target, so it’s used solely for generating predictions to assess model performance on unseen data.
- All other feature columns are the same between the training and test datasets.

## Model Details
### Architecture  
- This model card utilizes linear model such as **Logistic Regression**. As an alternative model **Random Forest** is used.  

### Evaluation Metrics  
- AUC (Area Under the ROC Curve): Measures the model's ability to distinguish between positive and negative classes.

### Final Values of Metrics for All Data using 'logistic regression' model:

| Dataset     | AUC   | 
|-------------|-------|
| Training    | 0.78  | 
| Validation  | 0.80  |
| Test        | 0.76  | 

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
* **[Logistic Regression Classifier](https://github.com/nmemranhussain/titanic-ml-models/blob/main/Titanic_logistic%20(1).ipynb)**
* **[Random Forest Classifier](https://github.com/nmemranhussain/titanic-ml-models/blob/main/Titanic_RF.ipynb)**

### Software Used to Implement the Model
- **Software:** Python (with libraries such as Pandas, Scikit-learn, seaborn & matplotlib)

### Version of the Modeling Software: 
- **'pandas'**: '2.2.2',
- **'scikit-learn'**: '1.4.2',
- **'seaborn'**: '0.13.2',
- **'matplotlib'**: '3.8.4**

### Hyperparameters or Other Settings of the Model
The following hyperparameters were used for the 'logistic regression' model:
- **Solver:** lbfgs
- **Maximum Iterations:** 100
- **Regularization (C):** 1.0
- **Features used in the model**: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
- **Target column**: Survived
- **Model type**: Logistic Regression
- **Hyperparameters**: Solver = lbfgs, Max iterations = 500, C = 1.0
- **Software used**: scikit-learn sklearn.linear_model._logistic

The following hyperparameters were used for the 'random forest' as an alternative model:
- **Columns used as inputs**: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 
- **Target column**: 'Survived',
- **Type of model**: 'Random Forest Classifier',
- **Software used**: 'scikit-learn',

## Quantitative Analysis

### Plots Related to Data or Final Model
 
![Plot of Survival Rate Vs. Passenger Class](SR_by_Class.png) 

**Description**: Passengers in 1st class had the highest survival rate, followed by those in 2nd class. 3rd class passengers had the lowest survival rate.

![Plot of Survival Rate Vs. Passenger Gender](SR_by_Gender.png) 

**Description**: Females had a significantly higher survival rate than males, aligning with the negative coefficient for the "Sex" feature in the logistic regression model.

![Plot of Survival Rate Vs. Passenger Age](SR_by_Age.png) 

**Description**: Children (ages 0-12) had the highest survival rate, while seniors (ages 50-80) had the lowest. Young adults and adults had relatively similar survival rates, though slightly lower than children.

## Potential Impacts, Risks, and Uncertainties using Logistic Regression & Random Forest Model ##
Logistic regression, while a powerful tool, presents several limitations. Its assumption of linear relationships can overlook intricate patterns within data. This can lead to overemphasis on certain features, biasing predictions and potentially reinforcing societal biases, especially when dealing with sensitive attributes like gender or class. The probabilistic nature of its output can be misinterpreted as deterministic, leading to misinformed decisions. Additionally, model performance is highly contingent on the selection of relevant features, and its results may not generalize well to diverse or evolving datasets. Moreover, the model's reliance on specific features, such as gender, can limit its applicability to different contexts. To mitigate potential biases, the training data was carefully examined for disparities related to gender and class. Interpretability tools were employed to analyze the model's decision-making process and its impact on protected groups.

## Responsible AI
This model was built using packages that promote responsible AI practices, such as:
* **[PiML]()** for interpretable machine learning.
* **[InterpretML](https://github.com/interpretml/interpret)** for explaining and visualizing model predictions.

On the other hand, While random forest comes with its own set of challenges. Its complex structure makes it difficult to interpret, hindering explainability. Despite its resilience, it can still be susceptible to overfitting if not carefully tuned. Furthermore, if trained on biased data, it can perpetuate unfairness in predictions. Additionally, it demands significant computational resources, which can slow down real-time applications. The model's reliance on multiple decision trees can obscure the true influence of individual features. The performance of a random forest model is sensitive to both data quality and hyperparameter tuning, and it may uncover unexpected patterns or interactions that can lead to either surprisingly good or poor results. To ensure fairness, the training data was scrutinized for biases related to gender and class. Interpretability tools were utilized to understand the model's behavior and its potential impact on protected groups.









