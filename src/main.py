
import features
import model
import pandas as pd
from definitions import *


if __name__ == '__main__':
    cpus = 24
    do_cv = False
    do_feature_selection = False

    # Load features
    print('======================================================')
    f_train, f_test = features.get_features()

    if do_cv:
        print('======================================================')
        print('Try cross-validation for our model')
        fixed_params, grid_params = model.create_for_cv(your_cpu_num=cpus)
        model.cv(f_train, fixed_params, grid_params)

    elif do_feature_selection:
        print('======================================================')
        print('Select features for our model')
        fixed_params = model.create_for_feature_selection(your_cpu_num=cpus)
        cv_mean, cv_std, f_importance_avg = model.cv(f_train, fixed_params)
        model.select_features(f_train, fixed_params, cv_mean, cv_std, f_importance_avg)

    else:
        print('======================================================')
        print('Train our model using train set')
        X = f_train.drop(['USER_ID', 'PRODUCT_ID', 'ORDERED'], axis=1)
        y = f_train.ORDERED
        trained_model = model.fit(model.create(your_cpu_num=cpus), X, y, show_importance=False)

        print('======================================================')
        print('Apply our model to test set')
        X = f_test.drop(['USER_ID', 'PRODUCT_ID'], axis=1)
        predicted = model.apply(trained_model, X)

        print('======================================================')
        print('Save predicted probabilities list')
        pd.DataFrame(predicted, columns=['predicted']).to_csv(name__submission, index=False)

    print ('Finish!')

