import warnings
import numpy as np
import torch
from gru_torch_ens.models import BaselineGruModel
from gru_torch_ens.data_processed import get_train_data_part as get_train_gru
from gru_torch_ens.data_processed import get_test_data as get_test_gru

import lightgbm as lgb
from lgb_tune.feature_engineering import get_test_data

warnings.simplefilter('ignore')

import random
import os
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def forecast(settings):
    # type: (dict) -> tuple
    seed_everything(2020)

    settings['train_len'] = 245
    settings['input_len'] = 72
    settings['dropout'] = 0.05
    settings['remove_features'] = ['Tmstamp', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab2', 'Pab3', 'Prtv']

    _, col_num, RS_list, RS_target_list = get_train_gru(settings)
    settings["in_var"] = col_num
    df_list = get_test_gru(settings)
    alpha_dist_drift = 1.0

    preds = []
    model_list = ["model_288"]
    for i in range(settings['capacity']):
        part_no = settings['group_config'][i]
        test = df_list[i]
        cols = [c for c in test.columns if c not in settings['remove_features'] + ['target', 'Day', 'TurbID'] + settings['cat_features']]
        test_cat = test[settings['cat_features']]
        test_cat = test_cat.to_numpy().reshape(-1, settings['input_len'], test_cat.shape[-1])
        test_cat = torch.IntTensor(test_cat).cuda()
        test = test[cols]
        test = RS_list[part_no].transform(test).reshape(-1, settings['input_len'], test.shape[-1])
        test = torch.FloatTensor(test).cuda()
        pred_list = []
        for model_name in model_list:
            path_to_model = settings['checkpoints'] + f'/gru/o288/{model_name}_{part_no}.pth'
            model = BaselineGruModel(settings).cuda()
            model.load_state_dict(torch.load(path_to_model))
            pred_list.append(model(test, test_cat, settings['output_len']).cpu().detach().numpy()[-1])
        pred = np.concatenate(pred_list)
        pred = pred[:, np.newaxis]
        pred = RS_target_list[part_no].inverse_transform(pred)
        l, k, w = np.array(pred), 36, 3
        la, lb = l[:k], l[k:]
        for j in range(1, w):
            lb = np.concatenate((lb, l[k - j:-j]), 1)
        lb = lb.mean(1, keepdims=True)
        preds.append(np.concatenate((la, lb)))
    preds_o288 = np.clip(np.array(preds) * alpha_dist_drift, a_min=100., a_max=1500.) # (134,288,1)

    seed_everything(2020)
    preds = []
    model_list = ["model_288"]
    for i in range(settings['capacity']):
        part_no = settings['group_config'][i]
        test = df_list[i]
        cols = [c for c in test.columns if c not in settings['remove_features'] + ['target', 'Day', 'TurbID'] + settings['cat_features']]
        test_cat = test[settings['cat_features']]
        test_cat = test_cat.to_numpy().reshape(-1, settings['input_len'], test_cat.shape[-1])
        test_cat = torch.IntTensor(test_cat).cuda()
        test = test[cols]
        test = RS_list[part_no].transform(test).reshape(-1, settings['input_len'], test.shape[-1])
        test = torch.FloatTensor(test).cuda()
        pred_list = []
        for model_name in model_list:
            path_to_model = settings['checkpoints'] + f'/gru/o36/{model_name}_{part_no}.pth'
            model = BaselineGruModel(settings).cuda()
            model.load_state_dict(torch.load(path_to_model))
            pred_list.append(model(test, test_cat, settings['output_len']).cpu().detach().numpy()[-1])
        pred = np.concatenate(pred_list)
        pred = pred[:, np.newaxis]
        pred = RS_target_list[part_no].inverse_transform(pred)
        preds.append(pred)
    preds_36 = np.clip(np.array(preds) * alpha_dist_drift, a_min=100., a_max=1500.)  # (134,288,1)

    res_gru = preds_o288
    res_gru[:,:36,:] = preds_36[:,:36,:]

    settings['train_len'] = 200
    settings['split_part'] = [3, 6, 9, 18, 36, 216]
    settings['remove_features'] = ['Day', 'Tmstamp','index']
    preds = []
    df_list = get_test_data(settings)
    path_to_model = settings['checkpoints']
    gbm = {}
    index = 0
    for i in settings['split_part']:
        index += i
        for part_num in range(4):
            model_name = "model_" + str(part_num) + "_" + str(index)
            gbm[model_name] = lgb.Booster(model_file=path_to_model + '/tree/' + model_name)
    for i in range(settings['capacity']):
        df = df_list[i][-1:]
        cols = [c for c in df.columns if c not in settings['remove_features']]
        df = df[cols]
        pred_list = []
        index = 0
        part_num = int(i/34)
        for time_len in settings['split_part']:
            index += time_len
            model_name = "model_" + str(part_num) + "_" + str(index)
            res = gbm[model_name].predict(df, num_iteration=gbm[model_name].best_iteration)
            pred_list.extend([res] * time_len)
        pred = np.concatenate(pred_list)
        pred = pred[:, np.newaxis]
        preds.append(pred)
    res_tree = np.clip(np.array(preds) * alpha_dist_drift, a_min=0., a_max=1521.)

    res = res_gru * 0.5 + res_tree * 0.5

    return res