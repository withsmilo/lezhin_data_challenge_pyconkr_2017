

name__pkl_ext = '.pkl'
name__csv_ext = '.csv'
name__dat_Ext = '.dat'

path__org = '../data/'
path__generated = '../data/generated/'

name__input_train_data = path__org + 'lezhin_dataset_v2_training.tsv'
name__input_test_data = path__org + 'lezhin_dataset_v2_test_without_label.tsv'
name__preprocessed_train = path__generated + 'preprocessed_train'
name__preprocessed_test = path__generated + 'preprocessed_test'
name__features_user = path__generated + 'features_user' + name__pkl_ext
name__features_product = path__generated + 'features_product' + name__pkl_ext
name__features_user_product = path__generated + 'features_user_product' + name__pkl_ext
name__features_train = path__generated + 'features_train' + name__pkl_ext
name__features_test = path__generated + 'features_test' + name__pkl_ext
name__model = path__generated + 'model' + name__dat_Ext
name__submission = path__generated + 'predicted' + name__csv_ext
