import numpy as np
import matplotlib.pyplot as plt
import pandas
import gc
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn import neural_network

def is_number(s):
    try:
        float(s)
        return s
    except ValueError:
        return 0


x_train = pandas.read_csv("TrainingSet.csv", converters = {'Work Experience in Current Job [years]' : is_number})
y_train = pandas.read_csv("TrainingResults.csv")
x_test =  pandas.read_csv("TestSet.csv")
RentalIncome = pandas.read_csv("RentalIncome.csv")
x_train['Work Experience in Current Job [years]'] = pandas.to_numeric(x_train['Work Experience in Current Job [years]'], errors='coerce').fillna(x_train['Work Experience in Current Job [years]'].mean())
x_test['Work Experience in Current Job [years]'] = pandas.to_numeric(x_test['Work Experience in Current Job [years]'], errors='coerce').fillna(x_test['Work Experience in Current Job [years]'].mean())
# print(x_train.dtypes)
# gc.collect()
# x_data.to_csv("Sanitized.csv")

# print(x_train)

# Create linear regression object
# regr = linear_model.MLPRegression()
regr = neural_network.MLPRegressor()

# Train the model using the training sets
regr.fit(x_train, y_train)
print("Trained models")
# Make predictions using the testing set
x_test['Income'] = regr.predict(x_test)
print("Made predictions")
# y_test = tdf['Income']
# results = pd.DataFrame()
results = x_test['Income'].copy() + RentalIncome
# results.columns = ['Income']

# print(results)
results.to_csv("groupIncomePredSubmission.csv", header = "Instance, Income")
