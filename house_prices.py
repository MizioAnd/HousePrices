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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math

class HousePrices(object):
    def __init__(self):
        self.df = HousePrices.df
        self.df_test = HousePrices.df_test
        self.df_all_feature_var_names = []


    ''' Pandas Data Frame '''
    df = pd.read_csv('/home/user/Documents/Kaggle/HousePrices/train.csv', header=0)
    df_test = pd.read_csv('/home/user/Documents/Kaggle/HousePrices/test.csv', header=0)

    def square_feet_to_meters(self, area):
        square_meter_per_square_feet = 0.3048**2
        return area*square_meter_per_square_feet

    def extract_numerical_features(self, df):
        return df.select_dtypes(include=[np.number])

    def extract_non_numerical_features(self, df):
        return df.select_dtypes(exclude=[np.number])


    def clean_data(self, df):
        df = df.copy()
        # Imputation using MICE
        numerical_features_names = self.extract_numerical_features(df)._get_axis(1)
        df[numerical_features_names] = self.estimate_by_mice(df[numerical_features_names])
        return df

    def encode_labels_in_numeric_format(self, df, estimated_var):
        # Transform non-numeric labels into numerical values
        # Cons.: gives additional unwanted structure to data, since some values are high and others low, despite labels where no such comparing measure exists.
        # Alternative: use one-hot-encoding giving all labels their own column represented with only binary values.
        le = LabelEncoder()
        le.fit(df[estimated_var].values)
        # Check that all values are represented
        list(le.classes_)
        df[''.join([estimated_var, 'Num'])] = le.transform(df[estimated_var].values)

    def label_classes(self, df, estimated_var):
        le = LabelEncoder()
        le.fit(df[estimated_var].values)
        return le.classes_


    def one_hot_encoder(self, df, estimated_var):
        ohe = OneHotEncoder()
        # Get every feature_var_name and exclude nan in label_classes
        label_classes = self.label_classes(df, estimated_var)
        label_classes = np.asarray(map(lambda x: str(x), label_classes))
        # if (estimated_var == 'SaleType') & (not any(df.columns == 'SalePrice')):
        #     print 'hello'
        # if any(label_classes == 'nan'):
        #     print 'debug'
        label_classes_is_not_nan = label_classes != 'nan'
        label_classes = label_classes[label_classes_is_not_nan]
        new_one_hot_encoded_features = map(lambda x: ''.join([estimated_var, '_', str(x)]), label_classes)

        # Create new feature_var columns with one-hot encoded values
        feature_var_values = ohe.fit_transform(np.reshape(np.array(df[''.join([estimated_var, 'Num'])].values), (df.shape[0], 1))).toarray().astype(int)
        column_index = 0
        for ite in new_one_hot_encoded_features:
            df[ite] = feature_var_values[0::, column_index]
            column_index += 1


    def add_feature_var_name_with_zeros(self, df, feature_var_name):
        df[feature_var_name] = np.zeros((df.shape[0], 1), dtype=int)
        pass


    def feature_var_names_in_training_set_not_in_test_set(self, feature_var_names_training, feature_var_names_test):
        feature_var_name_addition_list = []
        for feature_var_name in feature_var_names_training:
            if not any(feature_var_name == feature_var_names_test):
                feature_var_name_addition_list.append(feature_var_name)
        return np.array(feature_var_name_addition_list)


    def feature_mapping_to_numerical_values(self, df):
        is_one_hot_encoder = 1
        if is_one_hot_encoder:
            non_numerical_feature_names = self.extract_non_numerical_features(df)._get_axis(1)
            for feature_name in non_numerical_feature_names:
                self.encode_labels_in_numeric_format(df, feature_name)
                self.one_hot_encoder(df, feature_name)

            # Assume that training set has all possible feature_var_names
            # Although it may occur in real life that a training set may hold a feature_var_name. But it is probably avoided since such features cannot
            # be part of the trained learning algo.
            # Add missing feature_var_names of traning set not occuring in test set. Add these with zeros in columns.
            if not any(df.columns == 'SalePrice'):
                feature_var_names_traning_set = self.df_all_feature_var_names
                feature_var_name_addition_list = self.feature_var_names_in_training_set_not_in_test_set(feature_var_names_traning_set, df.columns)
                for ite in feature_var_name_addition_list:
                    self.add_feature_var_name_with_zeros(df, ite)


    def feature_engineering(self, df):
        df['LotAreaSquareMeters'] = self.square_feet_to_meters(df.LotArea.values)


    def drop_variable(self, df):
        # Drop all categorical feature columns
        non_numerical_feature_names = self.extract_non_numerical_features(df)._get_axis(1)
        for feature_name in non_numerical_feature_names:
            df = df.drop([''.join([feature_name, 'Num'])], axis=1)
            df = df.drop([feature_name], axis=1)
        return df


    def prepare_data_random_forest(self, df):
        df = df.copy()
        self.feature_mapping_to_numerical_values(df)
        df = self.clean_data(df)
        self.feature_engineering(df)
        df = self.drop_variable(df)
        # df = self.feature_scaling(df)
        return df


    def features_with_null_logical(self, df, axis=1):
        row_length = len(df._get_axis(0))
        # Axis to count non null values in. aggregate_axis=0 implies counting for every feature
        aggregate_axis = 1 - axis
        features_non_null_series = df.count(axis=aggregate_axis)
        # Whenever count() differs from row_length it implies a null value exists in feature column and a False in mask
        mask = row_length == features_non_null_series
        return mask


    def estimate_by_mice(self, df):
        df_estimated_var = df.copy()
        random.seed(129)
        mice = MICE()  #model=RandomForestClassifier(n_estimators=100))
        res = mice.complete(df.values)
        df_estimated_var[df.columns] = res[:][:]
        return df_estimated_var


    def feature_scaling(self, df):
        df = df.copy()
        # Scales all features to be values in [0,1]
        numerical_features_names = self.extract_numerical_features(df)._get_axis(1).values
        df[numerical_features_names] = df[numerical_features_names].apply(lambda x: x/x.max(), axis=0)
        return df

    def missing_values_in_DataFrame(self, df):
        mask = self.features_with_null_logical(df)
        print(df[mask[mask == 0].index.values].isnull().sum())
        print('\n')


def main():
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
    df_publ = house_prices.df.copy()
    df_test_publ = house_prices.df_test.copy()


    df = house_prices.prepare_data_random_forest(df_publ)
    house_prices.df_all_feature_var_names = df[df.columns[df.columns != 'SalePrice']].columns
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

    # Check if feature_var_names of test exist that do not appear in training set
    feature_var_names_addition_to_training_set = house_prices.feature_var_names_in_training_set_not_in_test_set(df_test.columns, df.columns)

    train_data = df.values
    test_data = df_test.values
    # train_data = house_prices.extract_numerical_features(df).values
    # test_data = house_prices.extract_numerical_features(df_test).values




    ''' Explore data '''
    explore_data = 0
    if explore_data:

        # Imputation for the 11 columns with none or nan values in the test data.
        # Using only numerical feature columns as first approach.
        # Print numeric feature columns with none or nan in test data
        print '\nColumns in train data with none/nan values:\n'
        print('\nTraining set numerical features\' missing values')
        df_publ_numerical_features = house_prices.extract_numerical_features(df_publ)
        house_prices.missing_values_in_DataFrame(df_publ_numerical_features)

        # Print numeric feature columns with none/nan in test data
        print '\nColumns in test data with none/nan values:\n'
        print('\nTest set numerical features\' missing values')
        df_test_publ_numerical_features = house_prices.extract_numerical_features(df_test_publ)
        house_prices.missing_values_in_DataFrame(df_test_publ_numerical_features)

        # Imputation method applied to numeric columns in test data with none/nan values
        print("Training set missing values after imputation")
        df_imputed = house_prices.estimate_by_mice(df_publ_numerical_features)
        house_prices.missing_values_in_DataFrame(df_imputed)
        print("Testing set missing values after imputation")
        df_test_imputed = house_prices.estimate_by_mice(df_test_publ_numerical_features)
        house_prices.missing_values_in_DataFrame(df_test_imputed)

        print('\nTotal Records for values: {}\n'.format(house_prices.df.count().sum() + house_prices.df_test.count().sum()))
        print('Total Records for missing values: {}\n'.format(house_prices.df.isnull().sum().sum() + house_prices.df_test.isnull().sum().sum()))

        print('Training set missing values')
        house_prices.missing_values_in_DataFrame(house_prices.df)

        print('Test set missing values')
        house_prices.missing_values_in_DataFrame(house_prices.df_test)

        print("\n=== AFTER IMPUTERS ===\n")
        print("=== Check for missing values in set ===")
        # Todo: fix the bug that "Total Records for missing values" stays unchanged while "Total Records for values" changes
        print('\nTotal Records for values: {}\n'.format(df.count().sum() + df_test.count().sum()))
        print('Total Records for missing values: {}\n'.format(df.isnull().sum().sum() + df_test.isnull().sum().sum()))

        # Overview of missing values in non numerical features
        print("Training set missing values")
        house_prices.missing_values_in_DataFrame(df)
        print("Testing set missing values")
        house_prices.missing_values_in_DataFrame(df_test)

        print("Training set with all non numerical features without missing values\n")
        df_all_non_numerical_features = house_prices.extract_non_numerical_features(df_publ)
        print df_all_non_numerical_features.count()
        # house_prices.missing_values_in_DataFrame(df)
        print("\nTesting set with all non numerical features without missing values\n")
        df_test_all_non_numerical_features = house_prices.extract_non_numerical_features(df_test_publ)
        print df_test_all_non_numerical_features.count()
        # house_prices.missing_values_in_DataFrame(df_test)




        # SalePrice square meter plot
        # Overview of data with histograms
        feature_to_plot = ['LotAreaSquareMeters', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']
        # feature_to_plot = ['YearBuilt', 'SalePrice', 'LotAreaSquareMeters', 'OverallCond', 'TotalBsmtSF']
        df_imputed_prepared = df_imputed.copy()
        house_prices.feature_engineering(df_imputed_prepared)
        bin_number = 25
        # df[df.LotAreaSquareMeters <= 2500.0][feature_to_plot].hist(bins=bin_number, alpha=.5)
        # df_imputed_prepared[df_imputed_prepared.LotAreaSquareMeters <= 2500.0][feature_to_plot].hist(bins=bin_number, alpha=.5)

        # We expect more houses to be sold in the summer. Which is also the case month MM, year YYYY.
        # Sale tops in juli
        # df[['MoSold', 'YrSold']].dropna().hist(bins='auto', alpha=.5)
        # plt.show()
        # plt.close()

        # Categorical plot with seaborn
        # sns.countplot(y='MSZoning', hue='MSSubClass', data=df, palette='Greens_d')
        # plt.show()
        # sns.stripplot(x='SalePrice', y='MSZoning', data=df, jitter=True, hue='LandContour')
        # plt.show()
        # sns.boxplot(x='SalePrice', y='MSZoning', data=df, hue='MSSubClass')
        # plt.show()
        # sns.boxplot(x='SalePrice', y='MSZoning', data=df)
        # plt.show()
        sns.boxplot(x='SalePrice', y='Neighborhood', data=df)
        plt.show()
        # sns.violinplot(x='SalePrice', y='MSZoning', data=df)
        # plt.show()
        sns.violinplot(x='SalePrice', y='Neighborhood', data=df)
        plt.show()

        # Arbitrary estimate, using the mean by default.
        # It also uses bootstrapping to compute a confidence interval around the estimate and plots that using error bars
        sns.barplot(x='SalePrice', y='MSZoning', hue='LotShape', data=df)
        plt.show()
        sns.barplot(x='SalePrice', y='Neighborhood', data=df)#, hue='LotShape')
        plt.show()
        sns.pointplot(x='SalePrice', y='MSZoning', hue='LotShape', data=df,
                      palette={"Reg": "g", "IR1": "m", "IR2": "b", "IR3": "r"}, markers=["^", "o", 'x', '<'], linestyles=["-", "--", '-.', ':'])
        plt.show()

        g = sns.PairGrid(df, x_vars=['SalePrice', 'LotArea'], y_vars=['MSZoning', 'Utilities', 'LotShape'], aspect=.75, size=3.5)
        g.map(sns.violinplot, palette='pastel')
        plt.show()

        # Quite slow
        # sns.swarmplot(x='MSZoning', y='MSSubClass', data=df, hue='LandContour')
        # plt.show()





    is_make_a_prediction = 0
    if is_make_a_prediction:
        ''' Random Forest '''
        # Fit the training data to the survived labels and create the decision trees
        x_train = train_data[0::, :-1]
        y_train = train_data[0::, -1]
        x_train = np.asarray(x_train, dtype=long)
        y_train = np.asarray(y_train, dtype=long)
        # test_data = np.asarray(test_data, dtype=long)

        # Todo: OBS. the below strategy with RandomForestClassifier produces best prediction 07.02.17
        # Random forest classifier based on cross validation parameter dictionary
        # Create the random forest object which will include all the parameters for the fit
        forest = RandomForestClassifier(max_features='sqrt')  #n_estimators=100)#, n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))
        parameter_grid = {'max_depth': [4,5,6,7,8], 'n_estimators': [200,210,240,250],'criterion': ['gini', 'entropy']}
        cross_validation = StratifiedKFold(random_state=None, shuffle=False)  #, n_folds=10)
        grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation, n_jobs=24)
        output = grid_search.predict(test_data)
        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))


        # Random forest (rf) classifier for feature selection
        # forest_feature_selection = RandomForestClassifier()  #max_features='sqrt')#n_estimators=100)#, n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))
        forest_feature_selection = RandomForestRegressor(n_estimators=100)
        forest_feature_selection = forest_feature_selection.fit(x_train, y_train)
        # forest_feature_selection = forest_feature_selection.fit(np.asarray(x_train, dtype=long), np.asarray(y_train, dtype=long))
        output = forest_feature_selection.predict(test_data)
        # output = forest_feature_selection.predict(np.asarray(test_data, dtype=long))
        # print np.shape(output)
        score = forest_feature_selection.score(x_train, y_train)
        # score = forest_feature_selection.score(np.asarray(x_train, dtype=long), np.asarray(y_train, dtype=long))
        print '\nSCORE random forest train data:---------------------------------------------------'
        print score

        # print titanic_panda_inst.compute_score_crossval(forest_feature_selection, x_train, y_train)
        # Take the same decision trees and run it on the test data
        # output = forest_feature_selection.predict(test_data)


        ''' xgboost '''
        # Grid search xgb
        use_xgbRegressor = 0
        if use_xgbRegressor:
            # Is a parallel job
            # xgb_model = xgb.XGBRegressor()
            # XGBClassifier gives the best prediction
            xgb_model = xgb.XGBClassifier()
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