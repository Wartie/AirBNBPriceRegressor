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
import matplotlib.pyplot as plt
import torchmetrics
import torchsummary
import xgboost as xg


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from numpy import sort

from xgboost import plot_importance



from typing import Tuple

# %%
### Sample code for the challenge




class Model:
    # Modify your model, default is a linear regression model with random weights

    def __init__(self):
        # self.theta = None
        self.xgb_regressor = xg.XGBRegressor(objective="reg:squarederror", 
                                             eval_metric=mean_squared_error)

                                            #  tree_method='gpu_hist', 
                                            #  gpu_id=0)

# booster.set_param({"device": "cuda:0"})

        
        params = {'n_estimators': [700, 800, 900],
                  'learning_rate': [0.1],
                  'max_depth': [3, 4],
                  'max_leaves': [10, 15],
                  #'gamma': [0],
                  #'colsample_bytree': [0.7, 0.8, 0.9],
                  #'subsample': [0.7, 0.8, 0.9]
        }
     
        #Best Hyperparams:  {'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'max_leaves': 15, 'n_estimators': 700, 'subsample': 0.8}

        # 'learning_rate': 0.1, 'max_depth': 9, 'max_leaves': 45, 'n_estimators': 90
        # params = {'n_estimators': [90],
        #           'learning_rate': [0.1],
        #           'max_depth': [9],
        #           'max_leaves': [45]
        # }
        # params = {'n_estimators': [40],
        #           'learning_rate': [0.15],
        #           'max_depth': [8],
        #           'max_leaves': [15, 25]
        # }

        self.hyperparameter_grid_search = GridSearchCV(self.xgb_regressor, params, cv=5, verbose=3, scoring='neg_mean_squared_error')
        # Scale data before applying PCA
        self.ss = StandardScaler()
        self.pca = PCA(n_components=300)

        self.dropped_features = ['host_has_profile_pic', 'has_availability', 'require_guest_phone_verification', 'require_guest_profile_picture'
                                 'zhima_selfie_verification', 'weibo_verification', 'photographer_verification', 'sesame_verification', 
                                 'manual_online_verification', 'sesame_offline_verification', 'manual_offline_verification', 'sent_id_verification',
                                 'Firm_mattress', 'Mobile_hoist', 'Beach_essentials', 'Roll-in_shower', 'Ground_floor_access', 'Ski-in/Ski-out', 
                                 'Shower_chair', 'Ceiling_hoist', 'Waterfront', 'Fireplace_guards', 'Private_bathroom', 'Lake_access', 'Other_pet(s)', 
                                 ]
        
        
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
        self.dropped_columns = X_train.columns[X_train_mean > 0.015]
        X_train = X_train[self.dropped_columns]
        print(X_train.shape)

        # X_train = x_train.drop(self.dropped_features, axis=1)

        
        scaled_data = self.ss.fit_transform(X_train)
        
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
        self.hyperparameter_grid_search.fit(scaled_data, y_train)

        print("Best Hyperparams: ", self.hyperparameter_grid_search.best_params_)
        print("Best MSE: ", self.hyperparameter_grid_search.best_score_)

        # bestimator = self.hyperparameter_grid_search.best_estimator_

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
        X_test = X_test[self.dropped_columns]
        scaled_data = self.ss.transform(X_test)

        # X_test_PCA = self.pca.transform(scaled_data)
        y_pred = self.hyperparameter_grid_search.best_estimator_.predict(scaled_data)
        


        return y_pred


# %%
# def split_data(x, y, seed):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, test_size=0.2, shuffle=True)
#     # # print(train.iloc[:2, :])
#     # # print("dkajnkdaj")
#     # # print(x_train.iloc[:2, :])   
#     # # print("askjfnkajdnk")
#     # y_train = train['price']
#     # # print(y_train.iloc[:2])

#     # x_test = test.drop('price', axis=1)
#     # y_test = test['price']
#     print(x_train.head(10))
#     print(y_train.head(10))
#     return x_train, y_train, x_test, y_test

# %%

# model = Model() # Model class imported from your submission

# x = pd.read_csv("data_cleaned_train_comments_X.csv")  # pandas Dataframe
# y = pd.read_csv("data_cleaned_train_y.csv")  # pandas Dataframe
# X_train, y_train, X_test, y_test = split_data(x, y, 84)
# model.train(X_train, y_train) # train your model on the dataset provided to you
# y_pred = model.predict(X_test) # test your model on the hidden test set (pandas Dataframe)


# %%

# mse = mean_squared_error(y_test, y_pred) # compute mean squared error
# print(mse)

# plot_importance(model.hyperparameter_grid_search.best_estimator_, max_num_features=300)
# plt.show()

# %%



# %% [markdown]
# **GOOD LUCK!**
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 


