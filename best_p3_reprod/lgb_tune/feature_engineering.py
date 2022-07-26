import pandas as pd
import numpy as np
import os
import warnings

warnings.simplefilter('ignore')


def add_features(df):
    df = make_rolling_features(df)
    df = df.interpolate().fillna(method='bfill')
    df['hour'] = pd.to_datetime(df['Tmstamp']).dt.hour.astype('category')
    df['time_index'] = ((pd.to_datetime(df['Tmstamp']).dt.minute % 60) / 10).astype('category')
    df['Tmstamp'] = df['Tmstamp'].map(lambda p: p.replace(":", "")).astype('category')
    return df

def fill_data(df):
    invalid_cond = (df['Patv'] < 0) | \
                   ((df['Patv'] == 0) & (df['Wspd'] > 2.5))
    df.Patv[invalid_cond] = np.NaN
    GRP_GLOBAL = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                  5,
                  6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                  11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
                  15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18,
                  19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
                  23, 23, 23, 23, 23]
    df['gid'] = df.TurbID.apply(lambda x: GRP_GLOBAL[x - 1])
    values = df.groupby(['Day', 'Tmstamp', 'gid'])['Patv'].transform('mean')
    df.Patv = np.where(df.Patv.notnull(), df.Patv, values)
    df.Pab1 = df[['Pab1', 'Pab2', 'Pab3']].mean(axis=1)
    df = df.interpolate().fillna(method='bfill')
    df.drop(['gid'], axis=1, inplace=True)
    return df

def invalid_data(raw_data):
    return ~((raw_data['Patv'] < 0) | \
                   ((raw_data['Patv'] == 0) & (raw_data['Wspd'] > 2.5)) | \
                   ((raw_data['Pab1'] > 89) | (raw_data['Pab2'] > 89) | (raw_data['Pab3'] > 89)) | \
                   ((raw_data['Wdir'] < -180) | (raw_data['Wdir'] > 180) | (raw_data['Ndir'] < -720) |
                    (raw_data['Ndir'] > 720)))

def make_rolling_features(df):
    for fea in ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Patv']:
        id_group = df.groupby('TurbID')[fea]
        for i in [6, 12, 36, 72, 144]:
            df[fea + "_simple_diff"] = id_group.apply(simple_diff)
            df[fea + "_simple_diff2"] = id_group.apply(simple_diff1d)
            df[fea + "_rolling_mean" + str(i)] = id_group.transform(lambda x: x.rolling(i).mean())
            df[fea + "_rolling_max" + str(i)] = id_group.transform(lambda x: x.rolling(i).max())
            df[fea + "_rolling_min" + str(i)] = id_group.transform(lambda x: x.rolling(i).min())
            df[fea + "_rolling_std" + str(i)] = id_group.transform(lambda x: x.rolling(i).std())
            df[fea + "_rolling_mean_diff" + str(i)] = df.groupby('TurbID')[fea + "_rolling_mean" + str(i)].apply(
                simple_diff)
            df[fea + "_rolling_mean_diffn" + str(i)] = df.groupby('TurbID')[fea + "_rolling_mean" + str(i)].apply(
                simple_diff1d)
            df[fea + "_rolling_mean_cal" + str(i)] = df[fea + "_rolling_mean" + str(i)] - df[fea]
        for i in range(5):
            df[fea + "_past" + str(i)] = df[fea].shift(i+1)
    return df


def simple_diff(series: np.ndarray):
    return series.diff()

def simple_diff2(series: np.ndarray):
    return series.diff(2)

def simple_diff1d(series: np.ndarray):
    return series.diff(288)


def get_test_data(settings):
    print("read test data")
    df = pd.read_csv(settings['path_to_test_x'])
    df = fill_data(df)
    print(len(df))
    print("add features")
    df = add_features(df)
    df_list = []
    for i in range(settings['capacity']):
        tid = i + 1
        df_list.append(df[df.TurbID == tid][-288:])
    print("get test data finish")
    return df_list


def get_train_data(settings, from_cache=False):
    print("Loading train data")
    if from_cache:
        df = pd.read_feather('features.f')
    else:
        df = pd.read_csv(os.path.join(settings['data_path'], settings['filename']))
        df = fill_data(df)
        print('Adding features')
        df = add_features(df)
        # df.to_feather('features.f')
    print("Loading train data finish")
    return df

def split_data(df, settings):
    print("splitting data")
    df_val = df[(df['Day'] > settings['train_len'])]
    df_train = df[(df['Day'] <= settings['train_len'])]
    cols = [c for c in df.columns if (c not in settings['remove_features']) and ('target' not in c)]
    x_train, x_val = df_train[cols], df_val[cols]
    cols = [c for c in df.columns if 'target' in c]
    print(cols)
    y_train, y_val = df_train[cols], df_val[cols]
    print("splitting data finish")
    return x_train, x_val, y_train, y_val


def add_target(df, settings):
    print("adding targeting")
    index = 0
    for i in settings['split_part']:
        index += i
        print(i, index)
        df['target' + str(index)] = df.groupby('TurbID')["Patv"].shift(-index)
        df['target' + str(index)] = df.groupby('TurbID')['target' + str(index)].transform(
            lambda x: x.rolling(i).mean())
    df = df[~df['target' + str(index)].isnull()].reset_index()
    print("adding targeting finish")
    return df

def split_data_by_part(df, settings, part_num, part_size):
    print("splitting data")
    df_val = df[(df['Day'] > settings['train_len']) & (df.TurbID < part_size * part_num) & (df.TurbID >= (part_size * (part_num-1)))]
    df_train = df[(df['Day'] <= settings['train_len']) & (df.TurbID < part_size * part_num) & (df.TurbID >= (part_size * (part_num-1)))]
    cols = [c for c in df.columns if (c not in settings['remove_features']) and ('target' not in c)]
    x_train, x_val = df_train[cols], df_val[cols]
    cols = [c for c in df.columns if 'target' in c]
    print(cols)
    y_train, y_val = df_train[cols], df_val[cols]
    print("splitting data finish")
    return x_train, x_val, y_train, y_val

