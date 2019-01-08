from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence 
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
import matplotlib.pyplot as plt
import numpy as np
import textwrap


def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    #Refactor to take ax
    '''
        Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: A plot of the number of iterations vs the MSE for the model for
        both the training set and test set.
    '''
    estimator.fit(X_train, y_train)
    name = estimator.__class__.__name__.replace('Regressor', '')
    learn_rate = estimator.learning_rate
    # initialize 
    train_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    # Get train score from each boost
    for i, y_train_pred in enumerate(estimator.staged_predict(X_train)):
        train_scores[i] = mean_squared_error(y_train, y_train_pred)
    # Get test score from each boost
    for i, y_test_pred in enumerate(estimator.staged_predict(X_test)):
        test_scores[i] = mean_squared_error(y_test, y_test_pred)
    plt.plot(train_scores, alpha=.5, label="{0} Train - learning rate {1}".format(
                                                                name, learn_rate))
    plt.plot(test_scores, alpha=.5, label="{0} Test  - learning rate {1}".format(
                                                      name, learn_rate), ls='--')
    plt.title(name, fontsize=16, fontweight='bold')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)

def rf_score_plot(randforest, X_train, y_train, X_test, y_test):
    #Refactor to take estimator and ax
    '''
        Parameters: randforest: RandomForestRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: The prediction of a random forest regressor on the test set
    '''
    randforest.fit(X_train, y_train)
    y_test_pred = randforest.predict(X_test)
    test_score = mean_squared_error(y_test, y_test_pred)
    plt.axhline(test_score, alpha = 0.7, c = 'y', lw=3, ls='-.', label = 
                                                        'Random Forest Test')


def plot_feature_importances():
    #Refactor to take estimator and ax
    importances = rf.feature_importances_[:n]
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    features = list(df.columns[indices])

    # Print the feature ranking
    print("\n13. Feature ranking:")

    for f in range(n):
        print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.savefig('13_Feature_ranking.png')
    plt.close()
    print('\nPlotted 13) feature importances')


def plot_partial_dependence_skater(estimator, X_train, feature_names):
    # Initialize names and interpreter class (which serves as a 'data manager')
    interpreter = Interpretation()
    interpreter.load_data(X_train, feature_names=feature_names)
    model = InMemoryModel(estimator.predict_proba, examples = X_train)
    # Plot partial dependence plots
    pdplots =  interpreter.partial_dependence.plot_partial_dependence(feature_names, model, n_samples=100, 
                                                                  n_jobs=3, grid_resolution = 50, figsize = (10,15))


def plot_skboost_partial_dependence(model, ax, X_train):
    plot_partial_dependence(model, X_train, [0,1],
                                           feature_names=X_train.feature_names[0:3],
                                           n_jobs=-1, grid_resolution=50)
    fig.suptitle('Partial Dependence Plot')
    fig.set_figwidth(15)


def plot_mse_vs_numestimators(num_estimator_list):
    #Need to refactor to take estimator and ax
    train_errors_rf = []
    test_errors_rf = []

    for num_est in num_estimator_list:
        rf = RandomForestRegressor(n_estimators = num_est, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_test =  rf.predict(X_test)
        y_pred_train =  rf.predict(X_train)
        
        train_errors_rf.append(mean_squared_error(y_pred_train, y_train)) 
        test_errors_rf.append(mean_squared_error(y_pred_test, y_test))
    plt.figure(figsize=(15,10))
    plt.plot(num_estimator_list, train_errors_rf, label='Training MSE')
    plt.plot(num_estimator_list, test_errors_rf, label='Test MSE')
    plt.xlabel('Number of Estimators')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.title('Random Forest MSE vs. Num Estimators')
    plt.legend()
