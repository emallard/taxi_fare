# Kaggle : New York City Taxi Fare Prediction

- https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction
- https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python

# Development notes

## Virtual environment + dependencies
```
python3 -m venv env

.\env\Scripts\activate
deactivate

python -m pip freeze > requirements.txt
python -m pip install -r requirements.txt
```
## VSCode
```
F1 : "Python: Select Interpreter"
```
## Interpreter
```
exec(open("script.py").read())
```
## Data

https://www.kaggle.com/datasets?search=taxi+new+york+prediction
https://www.kaggle.com/datasets/raviiloveyou/predict-taxi-fare-with-a-bigquery-ml-forecasting

Store Item Demand Forecasting Challenge

RandomForestRegressor
xgboost
ggplot 

understand the problem
EDA (exploratory data analysis)
local validation
modelling

df['col'].describe()
df['col'].value_counts()

k-fold cross validation
The general rule is to prefer Stratified K-Fold over usual K-Fold in any classification problem

TimeSeriesSplit
print('Mean validation MSE: {:.5f}'.format(np.mean(mse_scores)))
print('Overall validation MSE: {:.5f}'.format(np.mean(mse_scores) + np.std(mse_scores)))

Feature engineering : create new features
pd.to_datetime(df['date'])
df['date'].dt.year

Lable Data : 'a', 'b', 'c'
from sklearn.preprocessing import LabelEncoder
houses['RoofStyle_enc'] = le.fit_transform(houses['RoofStyle'])
pd.get_dummies : One-Hot encoding

Other encoding:
Backward Difference Coding
BaseN
Binary
CatBoost Encoder
Helmert Coding
James-Stein Encoder
Leave One Out 
...
Target Encoder ?????

Missing Data:
df.isnull().head() / .sum()
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(strategy="mean")
