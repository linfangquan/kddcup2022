import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import RobustScaler,StandardScaler
from prepare import prep_env

warnings.simplefilter('ignore')


def simple_diff(series: np.ndarray):
    return series.diff()

def simple_diff2(series: np.ndarray):
    return series.diff(2)

def add_features(df):
    df = fill_data(df)
    df['hour'] = (pd.to_datetime(df['Tmstamp']).dt.hour).astype('int')  # 24
    df['time_index'] = (pd.to_datetime(df['Tmstamp']).dt.minute / 10 + pd.to_datetime(df['Tmstamp']).dt.hour * 6).\
        astype('int')  # 144
    df['tid'] = df['TurbID'] - 1  # 134, note that df['TurbID'] is indexed from 1 not 0
    df = df.fillna(0.)
    return df

def fill_data(df):
    settings = prep_env()
    invalid_cond = (df['Patv'] < 0) | \
                   ((df['Patv'] == 0) & (df['Wspd'] > 2.5))
    df.Patv[invalid_cond] = np.NaN
    df[df['Pab1'] > 89]['Pab1'] = np.NaN
    df[df['Pab2'] > 89]['Pab2'] = np.NaN
    df[df['Pab3'] > 89]['Pab3'] = np.NaN
    df[df['Wdir'] > 180]['Wdir'] = np.NaN
    df[df['Wdir'] < -180]['Wdir'] = np.NaN
    df[df['Ndir'] > 720]['Ndir'] = np.NaN
    df[df['Ndir'] < -720]['Ndir'] = np.NaN
    group = settings['group_config']
    df['gid'] = df.TurbID.apply(lambda x: group[x - 1])
    values = df.groupby(['Day', 'Tmstamp', 'gid'])['Patv'].transform('mean')
    df.Patv = np.where(df.Patv.notnull(), df.Patv, values)
    df = df.interpolate().fillna(method='bfill')
    df.drop(['gid'], axis=1, inplace=True)
    return df

def invalid_data(raw_data):
    return ~(raw_data['Patv'] < 0)

def get_train_data_part(settings):
    print("Loading train data")
    df = pd.read_csv(os.path.join(settings['data_path'], settings['filename']))
    print('Adding features')
    df = add_features(df)
    cols = [c for c in df.columns if c not in settings['remove_features']]
    df = df[cols]
    print("data cols", df.columns)
    training_data_list = {}
    RS_list = {}
    RS_target_list = {}
    cat_col = settings['cat_features']
    part = {}
    for trub_id, i in enumerate(settings['group_config'], 1):
        if i not in part: part[i] = []
        part[i].append(trub_id)
    for i in range(settings['part_num']):
        df_train = df[df.TurbID.isin(part[i])]
        train_features = df_train.drop(columns=['Day', 'TurbID'] + cat_col)
        train_features_cat = df_train[cat_col].to_numpy()
        col_num = len(train_features.columns)
        train_targets = df_train['Patv']
        RS_list[i] = RobustScaler()
        train_features = RS_list[i].fit_transform(train_features)
        RS_target_list[i] = StandardScaler()
        train_targets = RS_target_list[i].fit_transform(train_targets.to_numpy().reshape(-1, 1))
        training_data_list[i] = (train_features, train_features_cat, train_targets)

    print("Loading train data finish, features num :", col_num)
    return training_data_list, col_num, RS_list, RS_target_list

def get_test_data(settings):
    print("Loading test data")
    df = pd.read_csv(settings['path_to_test_x'])
    df_list = []
    for i in range(settings['capacity']):
        tid = i + 1
        df_list.append(df[df.TurbID == tid])
    df = add_features(pd.concat(df_list))
    cols = [c for c in df.columns if c not in settings['remove_features']]
    df = df[cols]
    df_list = []
    for i in range(settings['capacity']):
        tid = i + 1
        df_list.append(df[df.TurbID == tid][-int(settings["input_len"]):])
    print("Loading test data finish")
    return df_list
