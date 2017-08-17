
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from definitions import *


def load():
    print('Load train data')
    train = __preprocess_train()
    # Drop duplicated rows
    train.drop_duplicates(inplace=True)
    # The TAG_1, TAG_2 have only one values {0}, so we drop them
    train.drop(['TAG_1', 'TAG_2'], axis=1, inplace=True)
    print('- Preprocessed train data {}: [{}]'.format(train.shape, ', '.join(train.columns)))

    print('Load test data')
    test = __preprocess_test()
    # The TAG_1, TAG_2 have only one values {0}, so we drop them
    test.drop(['TAG_1', 'TAG_2'], axis=1, inplace=True)
    print('- Preprocessed test data {}: [{}]'.format(test.shape, ', '.join(test.columns)))

    return train, test


def __preprocess(data, bias):
    for col_name in range(7-bias, 10-bias):
        new_col_name = 'USER_ID_{}'.format(col_name-6+bias)
        data.rename(columns={col_name: new_col_name}, inplace=True)
        data[new_col_name] = data[new_col_name].astype(np.str)
    for col_name in range(10-bias, 110-bias):
        new_col_name = 'BUY_PRODUCT_{}'.format(col_name-9+bias)
        data.rename(columns={col_name: new_col_name}, inplace=True)
        data[new_col_name] = data[new_col_name].astype(np.int8)
    for col_name in range(113-bias, 123-bias):
        new_col_name = 'SCHEDULE_{}'.format(col_name-112+bias)
        data.rename(columns={col_name: new_col_name}, inplace=True)
        data[new_col_name] = data[new_col_name].astype(np.int8)
    for col_name in range(123-bias, 141-bias):
        new_col_name = 'GENRE_{}'.format(col_name-122+bias)
        data.rename(columns={col_name: new_col_name}, inplace=True)
        data[new_col_name] = data[new_col_name].astype(np.int8)
    for col_name in range(145-bias, 151-bias):
        new_col_name = 'TAG_{}'.format(col_name-144+bias)
        data.rename(columns={col_name: new_col_name}, inplace=True)
        data[new_col_name] = data[new_col_name].astype(np.int8)
    for col_name in range(151-bias, 167-bias):
        new_col_name = 'TENDENCY_{}'.format(col_name-150+bias)
        data.rename(columns={col_name: new_col_name}, inplace=True)
        data[new_col_name] = data[new_col_name].astype(np.float32)
    data.rename(columns={1-bias: 'PLATFORM_A', 2-bias: 'PLATFORM_B', 3-bias: 'PLATFORM_C', 4-bias: 'PLATFORM_D',
                         5-bias: 'SESSION_CNT', 6-bias: 'PRODUCT_ID', 110-bias: 'TAG', 111-bias: 'COIN_NEEDED',
                         112-bias: 'COMPLETED', 141-bias: 'LAST_EPISODE', 142-bias: 'PUBLISHED', 143-bias: 'START_DATE',
                         144-bias: 'TOTAL_EPISODE_CNT'}, inplace=True)
    if bias == 0:
        data.rename(columns={0: 'ORDERED'}, inplace=True)

    # Fit to memory size
    if bias == 0:
        data.ORDERED = data.ORDERED.astype(np.int8)
    data.PLATFORM_A = data.PLATFORM_A.astype(np.int8)
    data.PLATFORM_B = data.PLATFORM_B.astype(np.int8)
    data.PLATFORM_C = data.PLATFORM_C.astype(np.int8)
    data.PLATFORM_D = data.PLATFORM_D.astype(np.int8)
    data.SESSION_CNT = data.SESSION_CNT.astype(np.int16)
    data.TAG = data.TAG.astype(np.int8)
    data.COIN_NEEDED = data.COIN_NEEDED.astype(np.int8)
    data.COMPLETED = data.COMPLETED.astype(np.int8)
    data.LAST_EPISODE = data.LAST_EPISODE.astype(np.int8)
    data.PUBLISHED = data.PUBLISHED.astype(np.int8)
    data.START_DATE = data.START_DATE.astype(np.int8)
    data.TOTAL_EPISODE_CNT = data.TOTAL_EPISODE_CNT.astype(np.int8)

    return data


def __preprocess_test():
    if Path(name__preprocessed_test + name__pkl_ext).exists():
        print('- {} is existed already. Let\'s load it'.format(name__preprocessed_test + name__pkl_ext))
        return pd.read_pickle(name__preprocessed_test + name__pkl_ext)

    # Read input data
    print('- {} is not existed. Let\'s do preprocessing it'.format(name__preprocessed_test + name__pkl_ext))
    data = pd.read_table(name__input_test_data, header=None)

    # Preprocess (bias is used for ORDERED column)
    data = __preprocess(data, bias=1)

    # Save preprocessed data
    print('- Saving...')
    data.to_pickle(name__preprocessed_test + name__pkl_ext)
    # data.to_csv(name__preprocessed_test + name__csv_ext, index=False)
    print('- Saved {}'.format(name__preprocessed_test + name__pkl_ext))

    return data


def __preprocess_train():
    if Path(name__preprocessed_train + name__pkl_ext).exists():
        print('- {} is existed already. Let\'s load it'.format(name__preprocessed_train + name__pkl_ext))
        return pd.read_pickle(name__preprocessed_train + name__pkl_ext)

    # Read input data
    print('- {} is not existed. Let\'s do preprocessing it'.format(name__preprocessed_train + name__pkl_ext))
    data = pd.read_table(name__input_train_data, header=None)

    # Preprocess
    data = __preprocess(data, bias=0)

    # Save preprocessed data
    print('- Saving...')
    data.to_pickle(name__preprocessed_train + name__pkl_ext)
    # data.to_csv(name__preprocessed_train + name__csv_ext, index=False)
    print('- Saved {}'.format(name__preprocessed_train + name__pkl_ext))

    return data

