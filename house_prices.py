# Predict the SalePrice
__author__ = 'mizio'
import csv as csv
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import pylab as plt
from fancyimpute import MICE
import random
from sklearn.model_selection import cross_val_score

class HousePrices(object):
    def __init__(self):
        self.df = HousePrices.df
        self.df_test = HousePrices.df_test


    ''' Pandas Data Frame '''
    df = pd.read_csv('/home/user/Documents/Kaggle/HousePrices/train.csv', header=0)
    df_test = pd.read_csv('/home/user/Documents/Kaggle/HousePrices/test.csv', header=0)

    def square_feet_to_meters(self, area):
        square_meter_per_square_feet = 0.3048**2
        return area*square_meter_per_square_feet


    def clean_data(self, df):
        # Drop all nulls and non-value type features
        if any('SalePrice' == df.columns.values):
            # Additionally for training data, drop all rows with nulls
            df_cleaned = df.select_dtypes(include=[np.number]).dropna()
        else:
            df_cleaned = df.select_dtypes(include=[np.number])

        return df_cleaned
        # return df

    def feature_engineering(self, df):
        # df['LotAreaSquareMeters'] = df['LotArea']
        #
        # df['LotAreaSquareMeters'] = self.square_feet_to_meters(df.LotAreaSquareMeters.values)
        pass


    def drop_variable(self, df):
        if any('SalePrice' == df.columns.values):
            return df
        else:
            # Drop the set of features in test data that has null in column
            df = df.dropna(axis=1)
        return df

    def prepare_data_random_forest(self, df):
        df = self.clean_data(df)
        self.feature_engineering(df)
        df = self.drop_variable(df)
        # self.feature_scaling(df)
        return df



def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from collections import OrderedDict
    from sklearn.ensemble import IsolationForest
    import seaborn as sns
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV

    ''' Prepare data '''

    house_prices = HousePrices()
    df_publ = house_prices.df[:][:]
    df_test_publ = house_prices.df_test[:][:]

    df = house_prices.prepare_data_random_forest(df_publ)
    print '\n TRAINING DATA:----------------------------------------------- \n'
    print df.head(3)
    print '\n'
    print df.info()
    print '\n'
    print df.describe()

    # Test data
    Id_df_test = house_prices.df_test['Id']  # Submission column
    df_test = house_prices.prepare_data_random_forest(df_test_publ)
    print '\n TEST DATA:----------------------------------------------- \n'
    print df_test.info()
    print '\n'
    print df_test.describe()
    print '\n'


    # Todo: return list of non-null columns in test data
    non_null_feature_columns = df_test.columns.values
    df = df[np.append(non_null_feature_columns, 'SalePrice')]
    print df.info()
    train_data = df.values
    test_data = df_test.values


    ''' Explore data '''
    explore_data = 1
    if explore_data:
        print('Total Records for missing values: {}\n'.format(house_prices.df['Alley'].count() + house_prices.df_test['Alley'].count()))

        print('Training set missing values')
        print(house_prices.df.isnull().sum())
        print('\n')

        print("\n=== AFTER IMPUTERS ===\n")
        print("=== Check for missing values in set ===")
        print("Total Records for missing values: {}\n".format(house_prices.df["Alley"].count() + house_prices.df_test['Alley'].count()))

        print("Training set missing values")
        print(df.isnull().sum())
        print("\n")

        print("Testing set missing values")
        print(df_test.isnull().sum())

        # SalePrice square meter plot
        # Overview of data with histograms
        # df[df.LotAreaSquareMeters <= 2500.0][['YearBuilt', 'SalePrice', 'LotAreaSquareMeters', 'OverallCond', 'TotalBsmtSF']].dropna().hist(bins='auto', alpha=.5)

        # We expect more houses to be sold in the summer. Which is also the case month MM, year YYYY.
        # Sale tops in juli
        # df[['MoSold', 'YrSold']].dropna().hist(bins='auto', alpha=.5)

        # plt.show()



    ''' Random Forest '''
    # Fit the training data to the survived labels and create the decision trees
    x_train = train_data[0::, :-1]
    y_train = train_data[0::, -1]

    # Random forest classifier based on cross validation parameter dictionary
    # Create the random forest object which will include all the parameters for the fit
    # forest = RandomForestClassifier(max_features='sqrt')#n_estimators=100)#, n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))
    # parameter_grid = {'max_depth': [4,5,6,7,8], 'n_estimators': [200,210,240,250],'criterion': ['gini', 'entropy']}
    # cross_validation = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)  #, n_folds=10)
    # grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation)
    # grid_search = grid_search.fit(x_train, y_train)
    # output = grid_search.predict(test_data)
    # print('Best score: {}'.format(grid_search.best_score_))
    # print('Best parameters: {}'.format(grid_search.best_params_))

    # Random forest (rf) classifier for feature selection
    forest_feature_selection = RandomForestClassifier(max_features='sqrt')#n_estimators=100)#, n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))
    forest_feature_selection = forest_feature_selection.fit(x_train, y_train)
    output = forest_feature_selection.predict(test_data)
    print np.shape(output)
    score = forest_feature_selection.score(x_train, y_train)
    print '\nSCORE random forest train data:---------------------------------------------------'
    print score
    # print titanic_panda_inst.compute_score_crossval(forest_feature_selection, x_train, y_train)
    # Take the same decision trees and run it on the test data
    # output = forest_feature_selection.predict(test_data)



    ''' Submission '''
    save_path = '/home/user/Documents/Kaggle/HousePrices/submission/'
    # dfBestScore = pd.read_csv(''.join(save_path, 'submissionTitanic.csv'), header=0)

    # We do not expect all to be equal since the learned model differs from time to time.
    # print (dfBestScore.values[0::, 1::].ravel() == output.astype(int))
    # print np.array_equal(dfBestScore.values[0::, 1::].ravel(), output.astype(int))  # But they are almost never all equal
    submission = pd.DataFrame({'Id': Id_df_test, 'SalePrice': output})
    submission.to_csv(''.join([save_path, 'submission_house_prices.csv']), index=False)

    # Using numpy instead of Pandas for submit
    # submitArray = np.array([passengerId_df_test.values, output]).T
    # np.savetxt(''.join([savePath, 'submissionTitanic.csv']), submitArray, delimiter=',')
    # plt.plot(np.array(passengerId_df_test), np.array(output), 'o')
    # plt.show()




if __name__ == '__main__':
    main()