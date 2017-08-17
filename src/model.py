import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot
from pathlib import Path
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib
from definitions import *


def cv(org_train, your_cpu_num=-1):
    def fit_with_parameter_grid(X_train, X_test, y_train, y_test):
        fixed_params = {
            'max_depth': 7,
            'learning_rate': 0.03,
            # 'n_estimators': 100,  # 1600
            'silent': True,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            # 'tree_method': 'gpu_hist',
            'n_jobs': your_cpu_num,
            'gamma': 0.5,
            'min_child_weight': 5,
            'max_delta_step': 0,
            'subsample': 0.76,
            'colsample_bytree': 0.95,
            'colsample_bylevel': 1,
            'reg_alpha': 2e-05,
            'reg_lambda': 10,
            'scale_pos_weight': 1,
            'base_score': 0.5,
            'random_state': 0
        }
        grid_params = {
            'n_estimators': [10]
        }
        print('- Start ParameterGrid')
        score = []
        feature_importance = []
        for params in ParameterGrid(grid_params):
            print('-- Apply params: {}'.format(params))
            params.update(fixed_params)
            model = xgb.XGBClassifier(**params)
            print('-- Fit...')
            model = fit(model, X_train, y_train,
                        fit_forcely=True, do_not_save=True, fit_verbose=False, show_importance=False)
            feature_importance.append(model.feature_importances_)
            print('-- Predict...')
            preds = apply(model, X_test)
            print('-- Evaluate...')
            score.append(evaluate(np.array(y_test), preds, show_roc_curve=False))

        return score, feature_importance

    # Define X and y for train
    X = org_train.drop(['USER_ID_1', 'USER_ID_2', 'USER_ID_3', 'PRODUCT_ID', 'ORDERED'], axis=1)
    y = org_train.ORDERED

    # Cross-validation with StratifiedKFold
    cv_score = []
    cv_f_importance = []
    kf = StratifiedKFold(n_splits=3)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print('- Cross validation for fold #{}: train:{}, test{}'.format(i + 1, train_index, test_index))
        s, fi = fit_with_parameter_grid(X[X.index.isin(train_index)], X[X.index.isin(test_index)],
                                        y[y.index.isin(train_index)], y[y.index.isin(test_index)])
        cv_score.append(s)
        cv_f_importance.append(fi)

    print('- Final score of ROC AUC score')
    print('  {} averaged from\n  {}'.format(np.mean(cv_score), cv_score))
    # print('- Final feature importance')
    # print('  {} averaged from our model'.format(np.average(cv_f_importance, axis=0)))


def create(your_cpu_num=-1):
    return XGBClassifier(max_depth=7,
                         learning_rate=0.03,
                         n_estimators=1600,
                         silent=True,
                         objective='binary:logistic',
                         booster='gbtree',
                         # tree_method='gpu_hist',
                         n_jobs=your_cpu_num,
                         gamma=0.5,
                         min_child_weight=5,
                         max_delta_step=0,
                         subsample=0.76,
                         colsample_bytree=0.95,
                         colsample_bylevel=1,
                         reg_alpha=2e-05,
                         reg_lambda=10,
                         scale_pos_weight=1,
                         base_score=0.5,
                         random_state=0)


def fit(alg, X, y, fit_forcely=False, do_not_save=False, fit_verbose=True, show_importance=False):
    if fit_forcely is False and Path(name__model).exists():
        print('{} is existed already. Let\'s load it'.format(name__model))
        return joblib.load(name__model)

    alg.fit(X, y,
            early_stopping_rounds=3,
            eval_metric='auc',
            eval_set=[[X, y]],
            verbose=fit_verbose)

    if show_importance:
        plot_importance(alg, importance_type='gain')
        pyplot.show()

    if do_not_save is False:
        print('Saving...')
        joblib.dump(alg, name__model)
        print('Saved {}'.format(name__model))

    return alg


def apply(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate(ground_truth, score, show_roc_curve=False):
    auc = roc_auc_score(ground_truth, score)
    print('-- AUC score: {}'.format(auc))

    if show_roc_curve:
        print('-- Calculate ROC curve...')
        fpr, tpr, thresholds = roc_curve(ground_truth, score, pos_label=0)

        pyplot.plot(tpr, fpr)
        pyplot.show()

        # auc = np.trapz(fpr, tpr)
        # print('AUC:', auc)

    return auc
