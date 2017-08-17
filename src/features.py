import loader
import numpy as np
import pandas as pd
from pathlib import Path
from definitions import *


def get_features():
    def generate_user_features(df):
        print('Generate user features')

        if Path(name__features_user).exists():
            print('- {} is existed already. Let\'s load it'.format(name__features_user))
            return pd.read_pickle(name__features_user)

        print('- {} is not existed. Let\'s generate it'.format(name__features_user))

        # Step #1. Grouped by USER_ID_1, USER_ID_2, USER_ID_3
        usr1 = df.groupby([df.USER_ID_1, df.USER_ID_2, df.USER_ID_3]).agg({'SESSION_CNT': ['sum', 'mean']})
        usr1.columns = usr1.columns.droplevel(0)
        usr1.columns = ['USR_SESSION_CNT_SUM', 'USR_SESSION_CNT_MEAN']
        usr1.reset_index(inplace=True)

        # Step #2. Grouped by USER_ID_1, USER_ID_2
        usr2 = df.groupby([df.USER_ID_1, df.USER_ID_2]).agg({'TENDENCY_1': ['sum', 'mean'],
                                                             'TENDENCY_2': ['sum', 'mean'],
                                                             'TENDENCY_3': ['sum', 'mean'],
                                                             'TENDENCY_4': ['sum', 'mean'],
                                                             'TENDENCY_5': ['sum', 'mean'],
                                                             'TENDENCY_6': ['sum', 'mean'],
                                                             'TENDENCY_7': ['sum', 'mean'],
                                                             'TENDENCY_8': ['sum', 'mean'],
                                                             'TENDENCY_9': ['sum', 'mean'],
                                                             'TENDENCY_10': ['sum', 'mean'],
                                                             'TENDENCY_11': ['sum', 'mean'],
                                                             'TENDENCY_12': ['sum', 'mean'],
                                                             'TENDENCY_13': ['sum', 'mean'],
                                                             'TENDENCY_14': ['sum', 'mean'],
                                                             'TENDENCY_15': ['sum', 'mean'],
                                                             'TENDENCY_16': ['sum', 'mean']})
        usr2.columns = usr2.columns.droplevel(0)
        usr2.columns = ['USR_TENDENCY_1_SUM', 'USR_TENDENCY_1_MEAN',
                        'USR_TENDENCY_2_SUM', 'USR_TENDENCY_2_MEAN',
                        'USR_TENDENCY_3_SUM', 'USR_TENDENCY_3_MEAN',
                        'USR_TENDENCY_4_SUM', 'USR_TENDENCY_4_MEAN',
                        'USR_TENDENCY_5_SUM', 'USR_TENDENCY_5_MEAN',
                        'USR_TENDENCY_6_SUM', 'USR_TENDENCY_6_MEAN',
                        'USR_TENDENCY_7_SUM', 'USR_TENDENCY_7_MEAN',
                        'USR_TENDENCY_8_SUM', 'USR_TENDENCY_8_MEAN',
                        'USR_TENDENCY_9_SUM', 'USR_TENDENCY_9_MEAN',
                        'USR_TENDENCY_10_SUM', 'USR_TENDENCY_10_MEAN',
                        'USR_TENDENCY_11_SUM', 'USR_TENDENCY_11_MEAN',
                        'USR_TENDENCY_12_SUM', 'USR_TENDENCY_12_MEAN',
                        'USR_TENDENCY_13_SUM', 'USR_TENDENCY_13_MEAN',
                        'USR_TENDENCY_14_SUM', 'USR_TENDENCY_14_MEAN',
                        'USR_TENDENCY_15_SUM', 'USR_TENDENCY_15_MEAN',
                        'USR_TENDENCY_16_SUM', 'USR_TENDENCY_16_MEAN']
        usr2.reset_index(inplace=True)

        # Step #3. Merged usr1 with usr2
        usr = usr1.merge(usr2, on=['USER_ID_1', 'USER_ID_2'])

        print('- Saving...')
        usr.to_pickle(name__features_user)
        print('- Saved {}'.format(name__features_user))

        return usr

    def generate_product_features(df):
        print('Generate product features')

        if Path(name__features_product).exists():
            print('- {} is existed already. Let\'s load it'.format(name__features_product))
            return pd.read_pickle(name__features_product)

        print('- {} is not existed. Let\'s generate it'.format(name__features_product))

        # Grouped by PRODUCT_ID
        prd = df.groupby([df.PRODUCT_ID]).agg({'LAST_EPISODE': ['sum', 'mean'],
                                               'START_DATE': ['sum', 'mean'],
                                               'TOTAL_EPISODE_CNT': ['sum', 'mean']})
        prd.columns = prd.columns.droplevel(0)
        prd.columns = ['PRD_LAST_EPISODE_SUM', 'PRD_LAST_EPISODE_MEAN',
                       'PRD_START_DATE_SUM', 'PRD_START_DATE_MEAN',
                       'PRD_TOTAL_EPISODE_CNT_SUM', 'PRD_TOTAL_EPISODE_CNT_MEAN']
        prd.reset_index(inplace=True)

        print('- Saving...')
        prd.to_pickle(name__features_product)
        print('- Saved {}'.format(name__features_product))

        return prd

    def generate_user_product_features(df):
        print('Generate user_product features')

        if Path(name__features_user_product).exists():
            print('- {} is existed already. Let\'s load it'.format(name__features_user_product))
            return pd.read_pickle(name__features_user_product)

        print('- {} is not existed. Let\'s generate it'.format(name__features_user_product))

        # Grouped by USER_ID_1, USER_ID_2, USER_ID_3, PRODUCT_ID
        usr_prd = df.groupby([df.USER_ID_1, df.USER_ID_2, df.USER_ID_3, df.PRODUCT_ID])\
            .agg({'USER_ID_1': 'size',
                  'ORDERED': 'sum'})
        # usr_prd.columns = usr_prd.columns.droplevel(0)
        usr_prd.columns = ['UP_VIEW_CNT', 'UP_ORDERED_SUM']
        usr_prd['UP_ORDERED_RATIO'] = pd.Series(usr_prd.UP_ORDERED_SUM / usr_prd.UP_VIEW_CNT).astype(np.float32)
        usr_prd.reset_index(inplace=True)

        print('- Saving...')
        usr_prd.to_pickle(name__features_user_product)
        print('- Saved {}'.format(name__features_user_product))

        return usr_prd

    def generate_features(dtrain, dtest):
        usr = generate_user_features(dtrain)
        prd = generate_product_features(dtrain)
        usr_prd = generate_user_product_features(dtrain)

        dtrain = dtrain.merge(usr, on=['USER_ID_1', 'USER_ID_2', 'USER_ID_3'], how='left') \
            .merge(prd, on=['PRODUCT_ID'], how='left') \
            .merge(usr_prd, on=['USER_ID_1', 'USER_ID_2', 'USER_ID_3', 'PRODUCT_ID'], how='left')

        dtest = dtest.merge(usr, on=['USER_ID_1', 'USER_ID_2', 'USER_ID_3'], how='left') \
            .merge(prd, on=['PRODUCT_ID'], how='left') \
            .merge(usr_prd, on=['USER_ID_1', 'USER_ID_2', 'USER_ID_3', 'PRODUCT_ID'], how='left')

        print('Train Features {}: [{}]'.format(dtrain.shape, ', '.join(dtrain.columns)))
        print('Test Features {}: [{}]'.format(dtest.shape, ', '.join(dtest.columns)))

        return dtrain, dtest

    if Path(name__features_train).exists() and Path(name__features_test).exists():
        print('{} {} are existed already. Let\'s load it'.format(name__features_train, name__features_test))
        return pd.read_pickle(name__features_train), pd.read_pickle(name__features_test)

    # Load target data
    train, test = loader.load()

    # Generate features
    f_train, f_test = generate_features(train, test)

    print('Saving...')
    f_train.to_pickle(name__features_train)
    f_test.to_pickle(name__features_test)
    print('Saved {} {}'.format(name__features_train, name__features_test))

    return f_train, f_test
