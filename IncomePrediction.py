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

def main():
    #, converters = {'Work Experience in Current Job [years]' : is_number}
    x_train = pandas.read_csv("TrainingSet.csv")
    y_train = pandas.read_csv("TrainingResults.csv")
    x_test =  pandas.read_csv("TestSet.csv")

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
    results = x_test['Income'].copy()
    # results.columns = ['Income']

    # print(results)
    results.to_csv("groupIncomePredSubmission.csv", header = "Instance, Income")



if __name__ == '__main__':
    main()
