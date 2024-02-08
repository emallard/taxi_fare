import pandas as pd
import matplotlib.pyplot as plt
import data

data.load()
data.train.fare_amount.hist(bins=30, alpha=0.5)
plt.show()

# => This is a regression problem