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
    y_train = df['Total Yearly Income [EUR]']
    y_train.to_csv("TrainingResults.csv")
    # y_train = y_train[:trainingDataLength]
    # print(trainingDataLength)
    tdf = pandas.read_csv("data/tcd-ml-1920-group-income-test.csv", index_col='Instance')
    fulldf = pandas.concat([df, tdf], sort = True)
    # fulldf.to_csv("CombinedParams.csv")

    gc.collect()
    #clear some memory used for df and tdf as they're no longer needed
    print("Read data")
    fulldf['Year of Record'] = pandas.to_numeric(fulldf['Year of Record'], errors='coerce').fillna(fulldf['Year of Record'].mean())
    # fulldf['Housing Situation'] = pandas.to_numeric(fulldf['Housing Situation'], errors='coerce').fillna(0)
    fulldf['Crime Level in the City of Employement'] = pandas.to_numeric(fulldf['Crime Level in the City of Employement'], errors='coerce').fillna(fulldf['Crime Level in the City of Employement'].mean())
    fulldf['Age'] = pandas.to_numeric(fulldf['Age'], errors='coerce').fillna(fulldf['Age'].mean())
    # fulldf['Work Experience in Current Job [years]'] = pandas.to_numeric(fulldf['Work Experience in Current Job [years]'], errors='coerce').fillna(fulldf['Work Experience in Current Job [years]'].mean())
    fulldf['Size of City'] = pandas.to_numeric(fulldf['Size of City'], errors='coerce').fillna(fulldf['Size of City'].mean())
    fulldf['Body Height [cm]'] = pandas.to_numeric(fulldf['Body Height [cm]'], errors='coerce').fillna(fulldf['Body Height [cm]'].mean())
    fulldf['Wears Glasses'] = pandas.to_numeric(fulldf['Wears Glasses'], errors='coerce').fillna(0)
    print("Coerced Numeric data")

    # TODO Sanitized Work Experience
    # params = fulldf[['Year of Record', 'Crime Level in the City of Employement',  'Age', 'Body Height [cm]', 'Size of City', 'Wears Glasses']].copy()
    gender_df = pandas.get_dummies(fulldf['Gender'])
    fulldf['Male'] = gender_df['male'].copy()
    fulldf['Female'] = gender_df['female'].copy()
    fulldf.drop('Gender', axis = 1, inplace = True)

    fulldf["Profession"].fillna("Unknown", inplace = True)
    fulldf['Profession'] = replaceWithMeans(fulldf[['Profession', 'Total Yearly Income [EUR]']].copy())
    fulldf['Country'].fillna("Unknown", inplace = True)
    fulldf['Country'] = replaceWithMeans(fulldf[['Country', 'Total Yearly Income [EUR]']].copy())
    allowed_vals = ['Black', 'Blond', 'Brown', 'Red', 'Unknown']
    fulldf.loc[~fulldf["Hair Color"].isin(allowed_vals), "Hair Color"] = "Unknown"
    fulldf['Hair Color'] = replaceWithMeans(fulldf[['Hair Color', 'Total Yearly Income [EUR]']].copy())

    fulldf['Satisfation with employer'].replace({'Unhappy': -1, 'Average': 0, 'Happy': 2, 'Somewhat Happy': 1}, inplace = True)
    fulldf['Satisfation with employer'] = pandas.to_numeric(fulldf['Satisfation with employer'], errors='coerce').fillna(0)

    fulldf['Work Experience in Current Job [years]'] = pandas.to_numeric(fulldf['Work Experience in Current Job [years]'], errors='coerce').fillna(0)

    fulldf.drop('Housing Situation', axis = 1, inplace = True)
    fulldf.drop('Yearly Income in addition to Salary (e.g. Rental Income)', axis = 1, inplace = True)

    #iterate through all professions and replace them with the average for that profession
    #get all rows with profession X
    #get mean income for that professions
    #replace that profession with the mean in all instances in original DataFrame
    #reset, next profession



    # jobs_df = pandas.get_dummies(fulldf['Profession'])
    # fulldf = pandas.merge(fulldf, jobs_df, on = 'Instance', copy = False)
    # fulldf.drop('Profession', axis = 1, inplace = True)
    #
    # Country_df = pandas.get_dummies(fulldf['Country'])
    # fulldf = pandas.merge(fulldf, Country_df, on = 'Instance', copy = False)
    # fulldf.drop('Country', axis = 1, inplace = True)
    #
    # HairColor_df = pandas.get_dummies(fulldf['Hair Color'])
    # fulldf = pandas.merge(fulldf, HairColor_df, on = 'Instance', copy = False)
    # fulldf.drop('Hair Color', axis = 1, inplace = True)
    #
    Degree_df = pandas.get_dummies(fulldf['University Degree'])
    fulldf['Bachelor'] = Degree_df['Bachelor'].copy()
    fulldf['Master'] = Degree_df['Master'].copy()
    fulldf['PhD'] = Degree_df['PhD'].copy()
    fulldf.drop('University Degree', axis = 1, inplace = True)

    print("Replaced Categorical Values")

    gc.collect()

    print(fulldf.head())
    print("Merged into params")

    # Normalizing is not providing enough of an improvement to justify the added run-time
    # min_max_scaler = preprocessing.MinMaxScaler()
    # scaled_values = min_max_scaler.fit_transform(params)
    # params.loc[:,:] = scaled_values
    # print("normalized")
    # stan_scaler = preprocessing.StandardScaler()
    # scaled_values = stan_scaler.fit_transform(params)
    # params.loc[:,:] = scaled_values
    # print("standardized")

    x_train = fulldf[:trainingDataLength]
    x_train.to_csv("TrainingSet.csv")
    x_test = fulldf[trainingDataLength:]
    x_test.to_csv("TestSet.csv")

    gc.collect()
    # x_data.to_csv("Sanitized.csv")

    # # print(x_train)
    #
    # # Create linear regression object
    # # regr = linear_model.MLPRegression()
    # regr = neural_network.MLPRegressor()
    #
    # # Train the model using the training sets
    # regr.fit(x_train, y_train)
    # print("Trained models")
    # # Make predictions using the testing set
    # x_test['Income'] = regr.predict(x_test)
    # print("Made predictions")
    # # y_test = tdf['Income']
    # # results = pd.DataFrame()
    # results = x_test['Income'].copy()
    # # results.columns = ['Income']
    #
    # # print(results)
    # results.to_csv("groupIncomePredSubmission.csv", header = "Instance, Income")


def replaceWithMeans(cols):
    vals = cols.iloc[:, 0].unique()
    for x in vals:
    # x = vals[0]
    # print(x)
        allOfProfX = cols.loc[cols.iloc[:, 0] == x]
        meanIncome = allOfProfX['Total Yearly Income [EUR]'].mean()
        cols.iloc[:, 0].replace(x, meanIncome, inplace = True)
    return cols.iloc[:,0]


if __name__ == '__main__':
    main()
