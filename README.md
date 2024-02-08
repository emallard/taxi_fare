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