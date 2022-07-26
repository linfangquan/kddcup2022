from feature_engineering import *
from prepare import prep_env
import lightgbm as lgb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    settings = prep_env()
    path_to_model = settings["checkpoints"]
    df = get_train_data(settings,from_cache=False)
    df = add_target(df,settings)
    for part_num in range(4):
        x_train, x_val, y_train, y_val = split_data_by_part(df, settings,part_num+1,34)
        index = 0
        for i in settings['split_part']:
            index += i

            if index == 288:
                params = {
                    'objective': 'regression',
                    'verbose': -1,
                    'metric': 'rmse',
                    'learning_rate': 0.002,
                    "device": "gpu",
                    'num_leaves': 57,
                    "random_state": 2022,
                    "bagging_freq": 5,
                    "bagging_fraction": 0.6,
                    "feature_fraction": 0.05,
                }
            elif index == 72:
                params = {
                    'objective': 'regression',
                    'verbose': -1,
                    'metric': 'rmse',
                    'learning_rate': 0.02,
                    "device": "gpu",
                    'num_leaves': 7,
                    "random_state": 2022,
                    "bagging_freq": 5,
                    "bagging_fraction": 0.2322,
                    "feature_fraction": 0.6150,
                }
            elif index == 36:
                params = {
                    'objective': 'regression',
                    'verbose': -1,
                    'metric': 'rmse',
                    'learning_rate': 0.01,
                    "device": "gpu",
                    'num_leaves': 18,
                    "random_state": 2022,
                    "bagging_freq": 5,
                    "bagging_fraction": 0.345,
                    "feature_fraction": 0.179,
                }
            elif index == 18:
                params = {
                    'objective': 'regression',
                    'verbose': -1,
                    'metric': 'rmse',
                    'learning_rate': 0.01,
                    "device": "gpu",
                    'num_leaves': 18,
                    "random_state": 2022,
                    "bagging_freq": 5,
                    "bagging_fraction": 0.345,
                    "feature_fraction": 0.179,
                }
            elif index == 9:
                params = {
                    'objective': 'regression',
                    'verbose': -1,
                    'metric': 'rmse',
                    'learning_rate': 0.01,
                    "device": "gpu",
                    'num_leaves': 32,
                    "random_state": 2022,
                    "bagging_freq": 5,
                    "bagging_fraction": 0.32965548292828706,
                    "feature_fraction": 0.4476385065077759,
                }
            else:
                params = {
                    'objective': 'regression',
                    'verbose': -1,
                    'metric': 'rmse',
                    'learning_rate': 0.01,
                    "device": "gpu",
                }

            model_name = "model_" + str(part_num) + "_" + str(index)
            label_name = 'target' + str(index)
            print(f"------------------train  {model_name}---------------------------")
            train_data = lgb.Dataset(x_train, label=y_train[label_name])
            valid_data = lgb.Dataset(x_val, label=y_val[label_name])
            gbm = lgb.train(params,
                            train_data,
                            valid_sets=[train_data, valid_data],
                            num_boost_round=1000,
                            verbose_eval=50,
                            early_stopping_rounds=20,
                            keep_training_booster=True
                            )
            gbm.save_model(path_to_model + '/tree/' + model_name)
