# %% [markdown]
# # CS541 Applied Machine Learning Fall 2024 - Class Challenge
# 
# In this class challenge assignment, you will be building a machine learning model to predict the price of an Airbnb rental, given the dataset we have provided. Total points: **100 pts**

# %% [markdown]
# To submit your solution, you need to submit a python (.py) file named challenge.py on Gradescope.
# Final Submission due Dec 10, 2024 (Initial submission due Nov 26)
# 
# There will be a Leaderboard for the challenge that can be seen by all students. You can give yourself a nickname on the leaderboard if you'd like to keep your score anonymous.
# 
# *If you choose a nickname, you are not allowed to use FULL CREDIT PERFORMANCE, 60 POINT SCORE BASELINE, or RANDOM BASELINE as they all are used by course staff (more on that below)*

# %% [markdown]
# To encourage you to get started early on the challenge, you are required to submit an initial submission due on **Nov 26, 11:59 pm**. For this submission, your model needs to be better than the linear model with random weights that we provided. The final submission will be due on **Dec 10, 11:59 pm**.
# 

# %% [markdown]
# ## Problem and dataset description
# Pricing a rental property such as an apartment or house on Airbnb is a difficult challenge. A model that accurately predicts the price can potentially help renters and hosts on the platform make better decisions. In this assignment, your task is to train a model that takes features of a listing as input and predicts the price.
# 
# We have provided you with a dataset collected from the Airbnb website for New York, which has a total of 29,985 entries, each with 764 features. You may use the provided data as you wish in development. We will train your submitted code on the same provided dataset, and will evaluate it on 2 other test sets (one public, and one hidden during the challenge).
# 
# We have already done some minimal data cleaning for you, such as converting text fields into categorical values and getting rid of the NaN values. To convert text fields into categorical values, we used different strategies depending on the field. For example, sentiment analysis was applied to convert user reviews to numerical values ('comments' column). We added different columns for state names, '1' indicating the location of the property. Column names are included in the data files and are mostly descriptive.
# 
# Also in this data cleaning step, the price value that we are trying to predict is calculated by taking the log of original price. Hence, the minimum value for our output price is around 2.302 and maximum value is around 9.21 on the training set.
# 

# %% [markdown]
# ## Datasets and Codebase
# 
# Please download the zip file from the link posted on Piazza/Resources.
# In this notebook, we implemented a linear regression model with random weights (**attached in the end**). For datasets, there’re 2 CSV files for features and labels:
# 
#     challenge.ipynb (This file: you need to add your code in here, convert it to .py to submit)
#     data_cleaned_train_comments_X.csv
#     data_cleaned_train_y.csv
# 

# %% [markdown]
# ## Instructions to build your model
# 1.  Implement your model in **challenge.ipynb**. You need to modify the *train()* and *predict()* methods of **Model** class (*attached at the end of this notebook*). You can also add other methods/attributes  to the class, or even add new classes in the same file if needed, but do NOT change the signatures of the *train()* and *predict()* as we will call these 2 methods for evaluating your model.
# 
# 2. To submit, you need to convert your notebook (.ipynb) to a python **(.py)** file. Make sure in the python file, it has a class named **Model**, and in the class, there are two methods: *train* and *predict*. Other experimental code should be removed if needed to avoid time limit exceeded on gradescope.
# 
# 3.  You can submit your code on gradescope to test your model. You can submit as many times you like. The last submission will count as the final model.
# 
# An example linear regression model with random weights is provided to you in this notebook. Please take a look and replace the code with your own.
# 

# %% [markdown]
# ## Evaluation
# 
# We will evaluate your model as follows
# 
#     model = Model() # Model class imported from your submission
#     X_train = pd.read_csv("data_cleaned_train_comments_X.csv")  # pandas Dataframe
#     y_train = pd.read_csv("data_cleaned_train_y.csv")  # pandas Dataframe
#     model.train(X_train, y_train) # train your model on the dataset provided to you
#     y_pred = model.predict(X_test) # test your model on the hidden test set (pandas Dataframe)
#     mse = mean_squared_error(y_test, y_pred) # compute mean squared error
# 
# 
# **There will be 2 test sets, one is public which means you can see MSE on this test set on the Leaderboard (denoted as *MSE (PUBLIC TESTSET)*), and the other one is hidden during the challenge (denoted as *MSE (HIDDEN TESTSET)*)**.
# Your score on the hidden test set will be your performance measure. So, don’t try to overfit your model on the public test set. Your final grade will depend on the following criteria:
# 

# %% [markdown]
# 1.  	Is it original code (implemented by you)?
# 2.  	Does it take a reasonable time to complete?
#     Your model needs to finish running in under 40 minutes on our machine. We run the code on a machine with 4 CPUs, 6.0GB RAM.
# 3.  	Does it achieve a reasonable MSE?
#     - **Initial submission (10 pts)**: Your model has to be better than the random weights linear model (denoted as RANDOM BASELINE on Leaderboard) provided in the file. Note this will due on **Nov 26, 11:59pm**.
#     - **Final submission (90 pts):** Your last submission will count as the final submission. There are four MSE checkpoints and you will be graded accordingly.
#         - Random Chance MSE ~40 and above: Grade=0
#         - MSE 0.5: Grade = 30
#         - MSE 0.157: Grade = 60 (denoated as 60 POINT SCORE BASELINE on the Leaderboard)
#         - MSE 0.143: Grade = 76.5
#         - MSE 0.1358 and below: Grade = 90 (denoated as FULL CREDIT PERFORMANCE on the Leaderboard)
#     
#     The grade will be linearly interpolated for the submissions that lie in between the checkpoints above. We will use MSE on the hidden test set to evaluate your model (lower is better).
# 
#     **Bonus**: **Top 3** with the best MSE on the hidden test set will get a 5 point bonus.

# %% [markdown]
# **Note 1: This is a regression problem** in which we want to predict the price for an AirBnB property. You should try different models and finetune their hyper parameters.  A little feature engineering can also help to boost the performance.
# 
# **Note 2**: You may NOT use additional datasets. This assignment is meant to challenge you to build a better model, not collect more training data, so please only use the data we provided. We tested the code on Python 3.10 and 3.9, thus it’s highly recommended to use these Python versions for the challenge.
# 

# %% [markdown]
# In this challenge, you can only use built-in python modules, and these following:
# - Numpy
# - pandas
# - scikit_learn
# - matplotlib
# - scipy
# - torchsummary
# - xgboost
# - torchmetrics
# - lightgbm
# - catboost
# - torch
# 
# 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchmetrics
import torchsummary
import xgboost as xg
import random
import scipy.stats as stats

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold

from sklearn.linear_model import Ridge 
from sklearn.linear_model import Perceptron


from numpy import sort

from xgboost import plot_importance

from catboost import CatBoostRegressor


from typing import Tuple

# %%

# def hist_box(x):
    
#     s,p = stats.shapiro(x)
#     skew = stats.skew(x)
#     print(f'The statistic value is {s} with p-value is {p}')
    
    
#     if p>0.05:
#         print(f'{x.name} is normally distributed')
#     else:
#         print(f'{x.name} is not normally distributed.')
    
    
#     if skew < 0:
#         print(f'{x.name} is left skewed with skew = {skew}')
#     elif skew > 0 :
#         print(f'{x.name} is right skewed with skew = {skew}')
#     else:
#         print(f'{x.name} has no skew.')
    
    
#     fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = [15,6])
#     fig.suptitle(f'Distribution of {x.name}')
    
#     sns.histplot(x = x, ax = ax1)
#     ax1.set_title('Histplot', fontsize = 16)
    
#     sns.boxplot(x = x, ax = ax2)
#     ax2.set_title('Boxplot', fontsize = 16)

#     plt.show()



# # %%
# def split_data(x, y, seed=random.randint(0, 10000)):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, test_size=0.2, shuffle=True)
#     # # print(train.iloc[:2, :])
#     # # print("dkajnkdaj")
#     # # print(x_train.iloc[:2, :])   
#     # # print("askjfnkajdnk")
#     # y_train = train['price']
#     # # print(y_train.iloc[:2])

#     # x_test = test.drop('price', axis=1)
#     # y_test = test['price']
#     # print(x_train.head(10))
#     # print(y_train.head(10))
#     return x_train, y_train, x_test, y_test

# # %%
# ### Sample code for the challenge


class Model:
    # Modify your model, default is a linear regression model with random weights

    def __init__(self):
        # self.theta = None
        self.xgb_regressor = xg.XGBRegressor(objective="reg:squarederror", 
                                             eval_metric=mean_squared_error)#,
                                            
                                            #  #early_stopping_rounds=20,
                                            #  tree_method='gpu_hist', 
                                            #  gpu_id=0)
    
        self.xgb_regressor2 = xg.XGBRegressor(objective="reg:squarederror", 
                                             eval_metric=mean_squared_error,
                                            
                                             #early_stopping_rounds=20,
                                             tree_method='gpu_hist', 
                                             gpu_id=0)

# booster.set_param({"device": "cuda:0"})

        self.cb_regressor = CatBoostRegressor(iterations=1000,
                        depth=7,
                        learning_rate=0.05,
                        loss_function='RMSE',
                        
                        verbose=0)

        self.dt = DecisionTreeRegressor()

        self.rr = Ridge()

        self.perc = Perceptron(shuffle=True, early_stopping=True)

        dtParams = {'splitter': ["best", "random"],
                    'max_depth': [30, 50, 100, 200],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [10, 30, 50, 90]}


        # Best Hyperparams:  {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'max_leaves': 30, 'n_estimators': 300}
        bestparams = {'n_estimators': [750],
                  'learning_rate': [0.1],
                  'max_depth': [3],
                  'max_leaves': [15],
                  'gamma': [0],
                  'colsample_bytree': [0.7, 0.8, 0.9],
                  'subsample': [0.7, 0.8, 0.9]
        }

        params = {'n_estimators': [100],
                  'colsample_bytree': [0.5],
                  'learning_rate': [0.1],
                  'max_depth': [5],
                  'max_leaves': [30]#,
                  #'gamma': [0, 1, 3, 5]
        }
        params2 = {'n_estimators': [700, 900, 1500],
                  'colsample_bytree': [0.5, 0.7, 1.0],
                  'subsample': [0.5, 0.7, 1.0],
                  'learning_rate': [0.05, 0.01],
                  'max_depth': [3, 5],
                  'max_leaves': [15, 30]#,
                  #'gamma': [0, 1, 3, 5]
        }
        # Best Hyperparams:  {'learning_rate': 0.1, 'max_depth': 9, 'max_leaves': 45, 'n_estimators': 120}
        # Best Hyperparams:  {'learning_rate': 0.1, 'max_depth': 11, 'max_leaves': 60, 'n_estimators': 120}

        # 'learning_rate': 0.1, 'max_depth': 9, 'max_leaves': 45, 'n_estimators': 90
        # params = {'n_estimators': [120],
        #           'learning_rate': [0.1],
        #           'max_depth': [11],
        #           'max_leaves': [60]
        # }
        # params = {'n_estimators': [40],
        #           'learning_rate': [0.15],
        #           'max_depth': [8],
        #           'max_leaves': [15, 25]
        # }

        paramsrr = {'solver': ['auto', 'svd', 'lsqr', 'cholesky'],
                  'alpha': [0.5, 0.7],
                  'max_iter': [None, 500, 1000, 2000]
        }

        paramspc = {'penalty': ['l2', 'l1', 'elasticnet'],
                #   'alpha': [0.5, 0.7],
                  'max_iter': [None, 500, 1000, 2000]
        }
        self.hyperparameter_grid_search_best = GridSearchCV(self.xgb_regressor, bestparams, cv=5, verbose=3, scoring='neg_mean_squared_error')
        self.hyperparameter_grid_search = GridSearchCV(self.xgb_regressor, params, cv=5, verbose=3, scoring='neg_mean_squared_error')
        self.hyperparameter_grid_search2 = GridSearchCV(self.xgb_regressor2, params2, cv=5, verbose=3, scoring='neg_mean_squared_error')

        self.hyperparameter_grid_searchdt = GridSearchCV(self.dt, dtParams, cv=5, verbose=3, scoring='neg_mean_squared_error')

        self.hyperparameter_grid_searchrr = GridSearchCV(self.rr, paramsrr, cv=5, verbose=3, scoring='neg_mean_squared_error')

        self.hyperparameter_grid_searchpc = GridSearchCV(self.perc, paramspc, cv=5, verbose=3, scoring='neg_mean_squared_error')

        # Scale data before applying PCA
        self.ss = StandardScaler()
        self.rs = RobustScaler()
        self.ms = MinMaxScaler()
        self.pca = PCA(n_components=0.99)
        self.var_filter = VarianceThreshold(threshold=0)
        self.selector = SelectKBest(f_classif, k=256)

        self.cv = RepeatedKFold(n_splits=5, n_repeats=2)

        
        # self.dropped_features = ['host_has_profile_pic', 'has_availability', 'require_guest_phone_verification', 'require_guest_profile_picture'
        #                          'zhima_selfie_verification', 'weibo_verification', 'photographer_verification', 'sesame_verification', 
        #                          'manual_online_verification', 'sesame_offline_verification', 'manual_offline_verification', 'sent_id_verification',
        #                          'Firm_mattress', 'Mobile_hoist', 'Beach_essentials', 'Roll-in_shower', 'Ground_floor_access', 'Ski-in/Ski-out', 
        #                          'Shower_chair', 'Ceiling_hoist', 'Waterfront', 'Fireplace_guards', 'Private_bathroom', 'Lake_access', 'Other_pet(s)',
        #                          'Pool_with_pool_hoist'
        #                          ]

                                        
        
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Train model with training data.
        Currently, we use a linear regression with random weights
        You need to modify this function.
        :param X_train: shape (N,d)
        :param y_train: shape (N,1)
            where N is the number of observations, d is feature dimension
        :return: None
        
        """

        # Use fit and transform method 

        X_train_mean = X_train.mean()

        print(X_train.shape)
        print(X_train_mean)
        self.dropped_columns = X_train.columns[X_train_mean > 0.015]
        X_train = X_train[self.dropped_columns]
        print(X_train.shape)

        # X_train = X_train.drop(self.dropped_features, axis=1)
        # print(X_train.dtypes.to_string())
        # print(X_train["maximum_nights"].dtype)

        
        scaled_data = self.ss.fit_transform(X_train)
        # scaled_data = self.rs.fit_transform(X_train)

        # X_train.info()

        # self.var_filter.fit(X_train)
      
        # X_train = self.var_filter.transform(X_train)

        # scaled_data = self.ms.fit_transform(X_train)

        # scaled_data = self.rfecv.fit_transform(scaled_data, y_train)

        # print(f"Optimal number of features: {self.rfecv.n_features_}")
        
        # X_train = self.selector.fit_transform(X_train, y_train)

        
        
        



        # scaled_mean = scaled_data.mean()

        # print(scaled_data.shape)
        # print(scaled_data[:10, X_train.columns.get_loc("maximum_nights")])
        # # print(scaled_mean)
        # self.dropped_columns = scaled_data.columns[scaled_mean > 0.05]
        # scaled_data = scaled_data[self.dropped_columns]
        # print(scaled_data.shape)
        # scaled_data = np.nan_to_num(scaled_data)
        # # print(np.mean(scaled_data, axis=0).shape)
        # mask = (np.mean(scaled_data, axis=0) == 0.0)
        # # print(mask)
        # self.idx = np.any([mask], axis=0)
        # # print(idx.shape)
        # # print(idx)
        # scaled_data = scaled_data[:, ~self.idx]

        # print(scaled_data.shape)


        # Set the n_components=3
        # var_ratio = []
        # nums = np.arange(1, X_train.shape[1], 10)
        # for num in nums:
        #     pca = PCA(n_components=num)
        #     pca.fit(scaled_data)
        #     var_ratio.append(np.sum(pca.explained_variance_ratio_))

        # print(var_ratio)
        # plt.figure(figsize=(4,2),dpi=150)
        # plt.grid()
        # plt.plot(nums,var_ratio,marker='o')
        # plt.xlabel('n_components')
        # plt.ylabel('Explained variance ratio')
        # plt.title('n_components vs. Explained Variance Ratio')
        # principal = PCA(n_components=3)
        # principal.fit(Scaled_data)
        # x=principal.transform(Scaled_data)

        # self.pca.fit(scaled_data)
        # X_train_PCA = self.pca.transform(scaled_data)

        # print(X_train.shape)
        # # print(sum(self.pca.explained_variance_))
        # print(self.pca.n_components_)
        # N, d = X_train.shape
        # self.theta = np.random.randn(d, 1)

        
        self.hyperparameter_grid_search_best.fit(scaled_data, y_train)

        print("Best Hyperparams: ", self.hyperparameter_grid_search_best.best_params_)
        print("Best MSE: ", self.hyperparameter_grid_search_best.best_score_)
        self.bestimator = self.hyperparameter_grid_search_best.best_estimator_


        # self.hyperparameter_grid_search.fit(X_train, y_train)

        # print("Best Hyperparams: ", self.hyperparameter_grid_search.best_params_)
        # print("Best MSE: ", self.hyperparameter_grid_search.best_score_)
        # self.bestimator = self.hyperparameter_grid_search.best_estimator_

        # # print(X_train.head(1).to_string())
        # self.cols_to_drop = []
        # for name, val in zip(self.bestimator.feature_names_in_, self.bestimator.feature_importances_):
        #     if val <= 0.000:
        #         # print("'" + name + "'")
        #         self.cols_to_drop.append(name)


        # print("Dropping: ", len(self.cols_to_drop))
        # dropped_X_train = X_train.drop(self.cols_to_drop, axis = 1)
        # print(X_train.shape)
        # print(dropped_X_train.shape)
        # scaled_data = self.ss.fit_transform(dropped_X_train)

        # self.hyperparameter_grid_search2.fit(scaled_data, y_train)

        # print("Best Hyperparams: ", self.hyperparameter_grid_search2.best_params_)
        # print("Best MSE: ", self.hyperparameter_grid_search2.best_score_)

        


        # self.cb_regressor.fit(scaled_data, y_train)
        # self.xgb_regressor.fit(scaled_data, y_train)


        # plot_importance(bestimator, max_num_features=20)
        # plt.show()  

        # thresholds = sort(bestimator.feature_importances_)
        # for thresh in thresholds:
        #     # select features using threshold
        #     selection = SelectFromModel(bestimator, threshold=thresh, prefit=True)
        #     select_X_train = selection.transform(X_train)
        #     # train model
        #     selection_model = xg.XGBRegressor(objective="reg:squarederror", 
        #                                             eval_metric=mean_squared_error, 
        #                                             tree_method='gpu_hist', 
        #                                             gpu_id=0)
        #     selection_model.fit(select_X_train, y_train)
        #     # eval model
        #     select_X_test = selection.transform(X_test)
        #     y_pred = selection_model.predict(select_X_test)
        #     predictions = [round(value) for value in y_pred]
        #     accuracy = accuracy_score(y_test, predictions)
        #     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

        # return None

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Use the trained model to predict on un-seen dataset
        You need to modify this function
        :param X_test: shape (N, d), where N is the number of observations, d is feature dimension
        return: prediction, shape (N,1)
        """
        # y_pred = X_test @ self.theta
        # self.dropped_columns = X_train.columns[X_train_mean > 0.01]
        # X_test = self.var_filter.transform(X_test)
        
        # X_test = self.selector.transform(X_test)

        # scaled_data = self.ms.transform(X_test)


        X_test = X_test[self.dropped_columns]
        # scaled_data = scaled_data[:, ~self.idx]

        # scaled_data = self.ss.transform(X_test)
        # scaled_data = self.rs.transform(X_test)
        # dropped_X_test = X_test.drop(self.cols_to_drop, axis = 1)
        # print(X_test.shape)
        # print(dropped_X_test.shape)
        scaled_data = self.ss.fit_transform(X_test)

        # X_test_PCA = self.pca.transform(scaled_data)
        y_pred = self.hyperparameter_grid_search_best.best_estimator_.predict(scaled_data)
        # y_pred = self.cb_regressor.predict(scaled_data)
        # y_pred = self.xgb_regressor.predict(scaled_data)



        return y_pred


# %%

# model = Model() # Model class imported from your su|bmission

# x = pd.read_csv("data_cleaned_train_comments_X.csv")  # pandas Dataframe
# y = pd.read_csv("data_cleaned_train_y.csv")  # pandas Dataframe

# # explore = pd.concat([x, y], axis=1)
# # print(x.shape, explore.shape)
# # hist_box(explore["maximum_nights"])


# X_train, y_train, X_test, y_test = split_data(x, y)
# model.train(X_train, y_train) # train your model on the dataset provided to you
# y_pred = model.predict(X_test) # test your model on the hidden test set (pandas Dataframe)




# # %%

# mse = mean_squared_error(y_test, y_pred) # compute mean squared error
# print(mse)

# # plot_importance(model.hyperparameter_grid_search.best_estimator_, max_num_features=100)
# # plt.show()
# # print(sort(model.bestimator.feature_importances_))
# # thresholds = sort( model.bestimator.feature_importances_)
# # for thresh in thresholds:
# #     # select features using threshold
# #     # if thresh != 0.0:
# #     selection = SelectFromModel(model.bestimator, threshold=thresh, prefit=True)
# #     select_X_train = selection.transform(X_train)
# #     # train model
# #     selection_model =  xg.XGBRegressor()
# #     selection_model.fit(select_X_train, y_train)
# #     # eval model
# #     select_X_test = selection.transform(X_test)
# #     predictions = selection_model.predict(select_X_test)
# #     mse = mean_squared_error(y_test, predictions)
# #     print("Thresh=%.3f, n=%d, MSE: %.5f" % (thresh, select_X_train.shape[1], mse))

# # %%



# # %% [markdown]
# # **GOOD LUCK!**
# # 

# # %% [markdown]
# # 

# # %% [markdown]
# # 

# # %% [markdown]
# # 

# # %% [markdown]
# # 


