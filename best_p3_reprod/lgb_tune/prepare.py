def prep_env():
    # type: () -> dict

    settings = {
        'data_path': '../data/',
        'filename': 'wtbdata_245days.csv',
        'checkpoints': 'checkpoints',
        'capacity': 134,
        'pred_file': 'predict.py',
        "framework": "base",
        'train_len': 200,
        'split_part':[3,6,9,18,36,216],
        'remove_features': ['Day', 'Tmstamp','index']
    }

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
