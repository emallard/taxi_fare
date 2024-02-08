
import xgboost as xgb
import data

# Create DMatrix on train data
dtrain = xgb.DMatrix(data=data.train[['store', 'item']],
                     label=data.train['sales'])

# Define xgboost parameters
params = {'objective': 'reg:linear',
          'max_depth': 15,
          'verbosity': 0}

# Train xgboost model
xg_depth_15 = xgb.train(params=params, dtrain=dtrain)

from sklearn.metrics import mean_squared_error

dtrain = xgb.DMatrix(data=data.train[['store', 'item']])
dtest = xgb.DMatrix(data=data.test[['store', 'item']])

# For each of 3 trained models
#for model in [xg_depth_2, xg_depth_8, xg_depth_15]:
for model in [xg_depth_15]:

    # Make predictions
    train_pred = model.predict(dtrain)     
    test_pred = model.predict(dtest)          
    
    # Calculate metrics
    mse_train = mean_squared_error(data.train['sales'], train_pred)                  
    mse_test = mean_squared_error(data.test['sales'], test_pred)
    print('MSE Train: {:.3f}. MSE Test: {:.3f}'.format(mse_train, mse_test))