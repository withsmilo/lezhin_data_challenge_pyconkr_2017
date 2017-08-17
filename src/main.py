
import features
import model
import pandas as pd
from definitions import *


if __name__ == '__main__':
    cpus = 8
    do_cv = False

    # Load features
    print('======================================================')
    f_train, f_test = features.get_features()

    if do_cv:
        print('======================================================')
        print('Try cross-validation for our model')
        model.cv(f_train, your_cpu_num=cpus)

    else:
        print('======================================================')
        print('Train our model using train set')
        X = f_train.drop(['USER_ID_1', 'USER_ID_2', 'USER_ID_3', 'PRODUCT_ID', 'ORDERED'], axis=1)
        y = f_train.ORDERED
        trained_model = model.fit(model.create(your_cpu_num=cpus), X, y, show_importance=False)

        print('======================================================')
        print('Apply our model to test set')
        X = f_test.drop(['USER_ID_1', 'USER_ID_2', 'USER_ID_3', 'PRODUCT_ID'], axis=1)
        predicted = model.apply(trained_model, X)

        print('======================================================')
        print('Save predicted probabilities list')
        pd.DataFrame(predicted, columns=['predicted']).to_csv(name__submission, index=False)

    print ('Finish!')

