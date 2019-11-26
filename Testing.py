import numpy as np
import matplotlib.pyplot as plt
import pandas
import gc
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn import neural_network


def main():
    x_train = pandas.read_csv("TrainingSet.csv")
    y_train = pandas.read_csv("TrainingResults.csv")
    x_test =  pandas.read_csv("TestSet.csv")
    RentalIncome = pandas.read_csv("RentalIncome.csv")
    # fulldf = pandas.concat([df, tdf], sort = True)
    # fulldf.to_csv("CombinedParams.csv")
    print(x_train.shape)
    print(y_train.shape)
    gc.collect()
    #clear some memory used for df and tdf as they're no longer needed
    print("Read data")

    jobs = fulldf['Profession'].unique()
    np.savetxt("Jobs.txt", jobs, fmt = "%s")

if __name__ == '__main__':
    main()
