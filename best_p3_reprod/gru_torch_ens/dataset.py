from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, features, features_cat, targets, settings):
        super(TrainDataset, self).__init__()
        self.features = features
        self.targets = targets
        self.features_cat = features_cat
        self.input_len = settings['input_len']
        self.output_len = settings['output_len']

    def __len__(self):
        return len(self.features) - self.output_len - self.input_len + 1

    def __getitem__(self, index):
        output_begin = index + self.input_len
        output_end = index + self.input_len + self.output_len
        return self.features[index: index + self.input_len].astype('float32') \
            , self.features_cat[index: index + self.input_len] \
            , self.targets[output_begin: output_end].reshape(-1).astype('float32')
