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
    df = pandas.read_csv("data/tcd-ml-1920-group-income-train.csv", index_col='Instance')
    trainingDataLength = len(df.index)
    # y_train = df['Total Yearly Income [EUR]']
    # y_train.to_csv("TrainingResults.csv")
    # y_train = y_train[:trainingDataLength]
    # print(trainingDataLength)
    tdf = pandas.read_csv("data/tcd-ml-1920-group-income-test.csv", index_col='Instance')
    fulldf = pandas.concat([df, tdf], sort = True)
    # fulldf.to_csv("CombinedParams.csv")

    gc.collect()
    #clear some memory used for df and tdf as they're no longer needed
    print("Read data")

    jobs = fulldf['Profession'].unique()
    np.savetxt("Jobs.txt", jobs, fmt = "%s")

if __name__ == '__main__':
    main()
