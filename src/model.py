import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot
from pathlib import Path
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle
from sklearn.externals import joblib
from definitions import *


def cv(org_train, fixed_params, grid_params=None):
    def __cv_pipeline(model, X_train, y_train, X_test, y_test, fi, s):
        print('-- Fit...')
        model = fit(model, X_train, y_train, fit_forcely=True, do_not_save=True, fit_verbose=False, show_importance=False)
        fi.append(model.feature_importances_)
        print('-- Predict...')
        preds = apply(model, X_test)
        print('-- Evaluate...')
        s.append(evaluate(np.array(y_test), preds, show_roc_curve=False))

    # Cross-validation with GroupKFold
    cv_score = []
    cv_f_importance = []
    kf = GroupKFold(n_splits=4)  # train 75% / test 25%
    for i in range(0, 25):
        # Shuffle!
        print('- Shuffle data!')
        X_shuffled, y_shuffled, groups_shuffled = shuffle(
            org_train.drop(['ORDERED'], axis=1), org_train.ORDERED, org_train['USER_ID'].values, random_state=0)

        for j, (train_index, test_index) in enumerate(kf.split(X_shuffled, y=y_shuffled, groups=groups_shuffled)):
            print('- Cross validation for iter #{} - fold #{}:\n-- train[{}]:{}\n-- test[{}]:{}'.format(
                i + 1, j + 1, len(train_index), train_index, len(test_index), test_index))

            # Define X_train, y_train, X_test, y_test
            train = org_train[org_train.index.isin(train_index)]
            test = org_train[org_train.index.isin(test_index)]
            X_train = train.drop(['USER_ID', 'PRODUCT_ID', 'ORDERED'], axis=1)
            y_train = train.ORDERED
            X_test = test.drop(['USER_ID', 'PRODUCT_ID', 'ORDERED'], axis=1)
            y_test = test.ORDERED
            del train, test

            score = []
            feature_importance = []
            if grid_params is None:
                print('- Start a fixed single pipeline')
                model = xgb.XGBClassifier(**fixed_params)
                __cv_pipeline(model, X_train, y_train, X_test, y_test, feature_importance, score)

            else:
                print('- Start some ParameterGrid pipeline')
                for params in ParameterGrid(grid_params):
                    print('-- Apply params: {}'.format(params))
                    params.update(fixed_params)
                    model = xgb.XGBClassifier(**params)
                    __cv_pipeline(model, X_train, y_train, X_test, y_test, feature_importance, score)
            cv_score.append(score)
            cv_f_importance.append(feature_importance)

    cv_score_mean = np.mean(cv_score)
    cv_score_std = np.std(cv_score)
    cv_f_importance_average = np.average(cv_f_importance, axis=0)
    print('- Final score of ROC AUC score')
    print('  {} average, {} standard deviation from\n  {}'.format(cv_score_mean, cv_score_std, cv_score))
    print('- Final feature importance')
    print('  {} average from our model\n'.format(cv_f_importance_average))

    return cv_score_mean, cv_score_std, cv_f_importance_average


def select_features(org_train, fixed_params, prev_cv_mean=None, prev_cv_std=None, prev_f_importance_average=None):
    # Generate previous feature importance DataFrame and sort them
    prev = org_train.drop(['USER_ID', 'PRODUCT_ID', 'ORDERED'], axis=1)
    prev = pd.DataFrame(prev_f_importance_average.reshape(-1, 1), index=prev.columns, columns=['FI'])
    prev = prev.apply(lambda x: x.sort_values(ascending=True), axis=0)

    filtered = prev[prev['FI'] == 0.0]
    if len(filtered) == 0:
        print('- No item does not have feature importance \'0.0\'. Stop to select features')

        final = org_train.drop(['USER_ID', 'PRODUCT_ID', 'ORDERED'], axis=1)
        print('- Final features {}: [{}]'.format(final.shape, ', '.join(final.columns)))

    else:
        print('- All columns ({}) having feature importance \'0.0\' are dropped'.format(filtered.index))
        org_train.drop(filtered.index, axis=1, inplace=True)

        print('- Try cross-validation for new train set')
        cv_mean, cv_std, cv_fi_avg = cv(org_train, fixed_params)

        if cv_mean < prev_cv_mean:
            print('-- Current cv_mean({}) < prev_cv_mean({}). Stop to select features'.format(cv_mean, prev_cv_mean))
            print('-- Current cv_std({}), prev_cv_std({})'.format(cv_std, prev_cv_std))

            final = org_train.drop(['USER_ID', 'PRODUCT_ID', 'ORDERED'], axis=1)
            print('-- Final features {}: [{}]'.format(final.shape, ', '.join(final.columns)))

        else:
            print('-- Current cv_mean({}) > prev_cv_mean({}). Keep to select features'.format(cv_mean, prev_cv_mean))
            print('-- Current cv_std({}), prev_cv_std({})'.format(cv_std, prev_cv_std))

            # Try one more!
            select_features(org_train, fixed_params, cv_mean, cv_std, cv_fi_avg)


def create_for_cv(your_cpu_num=-1):
    fixed_params = {
        'max_depth': 7,
        'learning_rate': 0.03,
        # 'n_estimators': 500,
        'silent': True,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        # 'tree_method': 'gpu_exact',
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
        'n_estimators': [500]
    }

    return fixed_params, grid_params


def create_for_feature_selection(your_cpu_num=-1):
    fixed_params = {
        'max_depth': 7,
        'learning_rate': 0.03,
        'n_estimators': 1600,
        'silent': True,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        # 'tree_method': 'gpu_exact',
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

    return fixed_params


def create(your_cpu_num=-1):
    return XGBClassifier(max_depth=7,
                         learning_rate=0.03,
                         n_estimators=1600,
                         silent=True,
                         objective='binary:logistic',
                         booster='gbtree',
                         # tree_method='gpu_exact',
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
        print('--- {} is existed already. Let\'s load it'.format(name__model))
        return joblib.load(name__model)

    print('--- Input features {}: [{}]'.format(X.shape, ', '.join(X.columns)))
    alg.fit(X, y,
            early_stopping_rounds=5,
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
    print('--- AUC score: {}\n'.format(auc))

    if show_roc_curve:
        print('--- Calculate ROC curve...')
        fpr, tpr, thresholds = roc_curve(ground_truth, score, pos_label=0)

        pyplot.plot(tpr, fpr)
        pyplot.show()

        # auc = np.trapz(fpr, tpr)
        # print('AUC:', auc)

    return auc
