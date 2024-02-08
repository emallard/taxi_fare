from sklearn.linear_model import LinearRegression
import data
from datetime import datetime

data.load()

X = data.train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
y = data.train['fare_amount']

linreg = LinearRegression()
linreg.fit(X, y)

features = ['pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 
            'passenger_count']

data.test['fare_amount'] = linreg.predict(data.test[features])

taxi_submission = data.test[['pickup_datetime', 'fare_amount']]
taxi_submission = taxi_submission.rename(columns={'pickup_datetime':'key'})
def toKey(s):
    return datetime.strptime(s,"%Y-%m-%d %H:%M:%S UTC").strftime("%Y-%m-%d %H:%M:%S.%f")

taxi_submission['key'] = taxi_submission['key'].apply(toKey)
taxi_submission.to_csv('submission.csv', index=False)