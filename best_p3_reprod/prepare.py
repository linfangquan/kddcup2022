# -*-Encoding: utf-8 -*-

def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        'data_path': '../data/',
        'filename': 'wtbdata_245days.csv',
        'checkpoints': "checkpoints",
        "input_len": 72,
        "train_output_len": 288,
        "output_len": 288,
        'seq_pre': 288,
        'start_col': 3,
        'in_var': 10,
        'out_var': 1,
        'day_len': 144,
        'capacity': 134,
        'mode': 'train',
        'train_len':245,
        'part_num': 24,
        'group_config': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5,
                         5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                         11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
                         15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18,
                         19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
                         23, 23, 23, 23, 23],
        # 'group_config': [0] * 134,
        'pred_file': 'predict.py',
        # "framework": "paddlepaddle",
        "framework": "pytorch",
        'stride': 1,
        "gpu": 0,
        'is_debug': True,
        'remove_features': ['Tmstamp', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab2', 'Pab3', 'Prtv'],
        # 'remove_features': ['Tmstamp', 'Etmp', 'Itmp', 'Pab2', 'Pab3', 'Prtv'],
        'cat_features': ['time_index', 'hour', 'tid'],
        'embed_dim': 2,
        'pos_embed_dim': 16,
        "lstm_layer": 2,
        "dropout": 0.05,
        'nheads': 2,
        'nlayers': 4,
        'has_pos_encoder': False
    }

    return settings
