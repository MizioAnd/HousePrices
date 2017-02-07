# Predict the SalePrice
__author__ = 'mizio'
import csv as csv
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import pylab as plt
from fancyimpute import MICE
# import sys
# sys.path.append('/custom/path/to/modules')
import random
from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import RandomForestClassifier

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

    def extract_numerical_features(self, df):
        return df.select_dtypes(include=[np.number])

    def clean_data(self, df):
        # Drop all non numeric type features
        if any('SalePrice' == df.columns.values):
            drop_non_or_nan_rows = 0
            if drop_non_or_nan_rows:
                # Additionally for training data, drop all rows with nulls
                df_cleaned = self.extract_numerical_features(df).dropna()
            else:
                df_cleaned = self.extract_numerical_features(df)
        else:
            df_cleaned = self.extract_numerical_features(df)

        # Imputation using MICE
        df_cleaned = self.estimate_by_mice(df_cleaned)
        return df_cleaned

    def feature_engineering(self, df):
        df['LotAreaSquareMeters'] = self.square_feet_to_meters(df.LotArea.values)



    def drop_variable(self, df):
        if any('SalePrice' == df.columns.values):
            return df
        else:
            # Drop the set of features in test data that has null in column
            df = df.dropna(axis=1)
        return df

    def prepare_data_random_forest(self, df):
        # df = self.clean_data(df)
        self.feature_engineering(df)
        # df = self.drop_variable(df)
        # self.feature_scaling(df)
        return df

    def   indices_with_none_or_nan_values_logical(self, df, axis=1):
        # Logical extracting only columns with non-NA/null values.
        # False is when NA/null values occur. I.e. when mask[mask == 0].
        agg_axis = 1 - axis
        agg_obj = df[:][:]
        # Return Series with number of non-NA/null observations over requested axis
        count = agg_obj.count(axis=agg_axis)
        mask = count == len(agg_obj._get_axis(agg_axis))
        return mask


    def estimate_by_mice(self, df):
        df_estimated_var = df[:][:]
        random.seed(129)
        mice = MICE()  #model=RandomForestClassifier(n_estimators=100))
        res = mice.complete(df.values)
        df_estimated_var[df.columns] = res[:][:]
        return df_estimated_var


    def feature_scaling(self, df):
        # Scales all features to be values in [0,1]
        features = list(df.columns)
        df[features] = df[features].apply(lambda x: x/x.max(), axis=0)


def main():
    # from graphviz import plot_tree
    # from graphviz import Digraph
    # dot = Digraph(comment='The Round Table')
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from collections import OrderedDict
    from sklearn.ensemble import IsolationForest
    import seaborn as sns
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import KFold, train_test_split

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

    train_data = df.values
    test_data = df_test.values


    # Imputation for the 11 columns with none or nan values in the test data.
    # Using only numerical feature columns as first approach.
    # Print numeric feature columns with none or nan in test data
    print '\nColumns in train data with none/nan values:\n'
    df_publ_numerical_features = house_prices.extract_numerical_features(df_publ)
    mask = house_prices.  indices_with_none_or_nan_values_logical(df_publ_numerical_features)
    print('\nTraining set numerical features\' missing values')
    print(df_publ_numerical_features[mask[mask == 0].index.values].isnull().sum())
    print('\n')

    # Print numeric feature columns with none/nan in test data
    print '\nColumns in test data with none/nan values:\n'
    df_test_publ_numerical_features = house_prices.extract_numerical_features(df_test_publ)
    mask = house_prices.  indices_with_none_or_nan_values_logical(df_test_publ_numerical_features)
    print('\nTest set numerical features\' missing values')
    print(df_test_publ_numerical_features[mask[mask == 0].index.values].isnull().sum())
    print('\n')

    # Imputation method applied to numeric columns in test data with non/nan values
    df_imputed = house_prices.estimate_by_mice(df_publ_numerical_features)
    mask = house_prices.  indices_with_none_or_nan_values_logical(df_imputed)
    print("Training set missing values after imputation")
    print(df_imputed[mask[mask == 0].index.values].isnull().sum())
    df_test_imputed = house_prices.estimate_by_mice(df_test_publ_numerical_features)
    mask = house_prices.  indices_with_none_or_nan_values_logical(df_test_imputed)
    print("Testing set missing values after imputation")
    print(df_test_imputed[mask[mask == 0].index.values].isnull().sum())


    ''' Explore data '''
    explore_data = 1
    if explore_data:
        print('\nTotal Records for values: {}\n'.format(house_prices.df.count().sum() + house_prices.df_test.count().sum()))
        print('Total Records for missing values: {}\n'.format(house_prices.df.isnull().sum().sum() + house_prices.df_test.isnull().sum().sum()))

        print('Training set missing values')
        mask = house_prices.  indices_with_none_or_nan_values_logical(house_prices.df)
        print(house_prices.df[mask[mask == 0].index.values].isnull().sum())
        print('\n')

        print('Test set missing values')
        mask = house_prices.  indices_with_none_or_nan_values_logical(house_prices.df_test)
        print(house_prices.df_test[mask[mask == 0].index.values].isnull().sum())
        print('\n')

        print("\n=== AFTER IMPUTERS ===\n")
        print("=== Check for missing values in set ===")
        # Todo: fix the bug that "Total Records for missing values" stays unchanged while "Total Records for values" changes
        print('\nTotal Records for values: {}\n'.format(df.count().sum() + df_test.count().sum()))
        print('Total Records for missing values: {}\n'.format(df.isnull().sum().sum() + df_test.isnull().sum().sum()))

        print("Training set missing values")
        mask = house_prices.  indices_with_none_or_nan_values_logical(df)
        print(df[mask[mask == 0].index.values].isnull().sum())
        print("\n")

        print("Testing set missing values")
        mask = house_prices.  indices_with_none_or_nan_values_logical(df_test)
        print(df_test[mask[mask == 0].index.values].isnull().sum())

        # SalePrice square meter plot
        # Overview of data with histograms
        feature_to_plot = ['LotAreaSquareMeters', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']
        # feature_to_plot = ['YearBuilt', 'SalePrice', 'LotAreaSquareMeters', 'OverallCond', 'TotalBsmtSF']
        df_imputed_prepared = house_prices.prepare_data_random_forest(df_imputed)
        bin_number = 25
        df[df.LotAreaSquareMeters <= 2500.0][feature_to_plot].hist(bins=bin_number, alpha=.5)
        df_imputed_prepared[df_imputed_prepared.LotAreaSquareMeters <= 2500.0][feature_to_plot].hist(bins=bin_number, alpha=.5)


        # We expect more houses to be sold in the summer. Which is also the case month MM, year YYYY.
        # Sale tops in juli
        df[['MoSold', 'YrSold']].dropna().hist(bins='auto', alpha=.5)

        plt.show()


    is_make_a_prediction = 0
    if is_make_a_prediction:
        ''' Random Forest '''
        # Fit the training data to the survived labels and create the decision trees
        x_train = train_data[0::, :-1]
        y_train = train_data[0::, -1]

        # Todo: OBS. the below strategy with RandomForestClassifier produces best prediction 07.02.17
        # Random forest classifier based on cross validation parameter dictionary
        # Create the random forest object which will include all the parameters for the fit
        # forest = RandomForestClassifier()  # max_features='sqrt')#n_estimators=100)#, n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))
        # parameter_grid = {'max_depth': [4,5,6,7,8], 'n_estimators': [200,210,240,250],'criterion': ['gini', 'entropy']}
        # cross_validation = StratifiedKFold(random_state=None, shuffle=False)  #, n_folds=10)
        # grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation)
        # grid_search = grid_search.fit(x_train, y_train)
        # output = grid_search.predict(test_data)
        # print('Best score: {}'.format(grid_search.best_score_))
        # print('Best parameters: {}'.format(grid_search.best_params_))

        # Random forest (rf) classifier for feature selection
        forest_feature_selection = RandomForestClassifier()  #max_features='sqrt')#n_estimators=100)#, n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))
        # forest_feature_selection = RandomForestRegressor(n_estimators=100)
        forest_feature_selection = forest_feature_selection.fit(x_train, y_train)
        output = forest_feature_selection.predict(test_data)
        print np.shape(output)
        score = forest_feature_selection.score(x_train, y_train)
        print '\nSCORE random forest train data:---------------------------------------------------'
        print score
        # print titanic_panda_inst.compute_score_crossval(forest_feature_selection, x_train, y_train)
        # Take the same decision trees and run it on the test data
        # output = forest_feature_selection.predict(test_data)


        ''' xgboost '''
        # Grid search xgb
        # Is a parallel job
        xgb_model = xgb.XGBRegressor()
        # XGBClassifier gives the best prediction
        # xgb_model = xgb.XGBClassifier()
        cross_validation = StratifiedKFold(shuffle=False, random_state=None)  # , n_folds=10)
        # parameter_grid = {'max_depth': [4, 5, 6, 7, 8], 'n_estimators': [200, 210, 240, 250]}
        parameter_grid = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}  #, 'criterion': ['gini', 'entropy']}
        clf = GridSearchCV(xgb_model, param_grid=parameter_grid, cv=cross_validation)  #verbose=1)
        clf.fit(x_train, y_train)
        output = clf.predict(test_data)
        print '\nSCORE XGBRegressor train data:---------------------------------------------------'
        print(clf.best_score_)
        print(clf.best_params_)





        ''' Submission '''
        save_path = '/home/user/Documents/Kaggle/HousePrices/submission/'
        # Submission requires a csv file with Id and SalePrice columns.
        dfBestScore = pd.read_csv(''.join([save_path, 'submission_house_prices.csv']), header=0)

        # We do not expect all to be equal since the learned model differs from time to time.
        # print (dfBestScore.values[0::, 1::].ravel() == output.astype(int))
        # print np.array_equal(dfBestScore.values[0::, 1::].ravel(), output.astype(int))  # But they are almost never all equal
        submission = pd.DataFrame({'Id': Id_df_test, 'SalePrice': output})
        submission.to_csv(''.join([save_path, 'submission_house_prices.csv']), index=False)

if __name__ == '__main__':
    main()