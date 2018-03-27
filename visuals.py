###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, log_loss
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier

def distribution(df, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    fig = plt.figure(figsize = (11,5));

    for i, feature in enumerate(df):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(df[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()
    
def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize = (40,25))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"], fontsize = 30)
                ax[j//3, j%3].set_xlabel("Training Set Size", fontsize = 40)
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)", fontsize = 40)
    ax[0, 1].set_ylabel("Accuracy Score", fontsize = 40)
    ax[0, 2].set_ylabel("F-score", fontsize = 40)
    ax[1, 0].set_ylabel("Time (in seconds)", fontsize = 40)
    ax[1, 1].set_ylabel("Accuracy Score", fontsize = 40)
    ax[1, 2].set_ylabel("F-score", fontsize = 40)
    
    # Add titles
    ax[0, 0].set_title("Model Training", fontsize = 40)
    ax[0, 1].set_title("Accuracy Score on Training Subset", fontsize = 40)
    ax[0, 2].set_title("F-score on Training Subset", fontsize = 40)
    ax[1, 0].set_title("Model Predicting", fontsize = 40)
    ax[1, 1].set_title("Accuracy Score on Testing Set", fontsize = 40)
    ax[1, 2].set_title("F-score on Testing Set", fontsize = 40)
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper right', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    #plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    #plt.tight_layout()
    plt.show()
    

def feature_plot(importances, X_train, y_train):
    
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    fig = plt.figure(figsize = (14,5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()  
    
def ModelLearning(X, y, clf, cv, scoring, train_sizes):
    
    sizes, train_scores, test_scores = learning_curve(clf, X, y, \
        cv = cv, train_sizes = train_sizes, scoring = scoring)

    train_std, train_mean = np.std(train_scores, axis = 1), np.mean(train_scores, axis = 1)
    test_std, test_mean = np.std(test_scores, axis = 1), np.mean(test_scores, axis = 1)

    plt.figure(figsize=(14,5))
    plt.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')
    
    plt.title(clf.__class__.__name__)
    plt.xlabel('Number of Training Points')
    plt.ylabel('Score')
    plt.xlim([0, X.shape[0]*0.8])
    plt.ylim(np.min(test_mean) - 0.05, np.max(train_mean) + 0.05)
    plt.legend(loc='lower right')
    plt.show()

def ModelComplexity(X, y, clf, cv, scoring, param_name, param_range):
    
    train_scores, test_scores = validation_curve(clf, X, y, \
        param_name = param_name, param_range = param_range, cv = 3, scoring = scoring)

    train_std, train_mean = np.std(train_scores, axis = 1), np.mean(train_scores, axis = 1)
    test_std, test_mean = np.std(test_scores, axis = 1), np.mean(test_scores, axis = 1)
    
    plt.figure(figsize=(14, 5))
    plt.plot(param_range, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(param_range, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')
    
    plt.title(clf.__class__.__name__)
    plt.xlabel(param_name)
    plt.ylabel('Score')    
    #plt.xlim([0, X.shape[0]*0.8])
    plt.ylim(np.min(test_mean) - 0.05, np.max(train_mean) + 0.05)
    plt.legend(loc='upper right')
    plt.show()
    
