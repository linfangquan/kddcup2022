from torch import nn
import torch
import numpy as np


class BaselineGruModel(nn.Module):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGruModel, self).__init__()
        self.input_len = settings["input_len"]
        self.hidC = 3  # self.hidC = settings["in_var"]
        self.num_emb_size = 42
        self.time_emb_size = 6
        self.id_emb_size = 6

        self.pab_emb_size = 6
        self.ndir_emb_size = 6
        self.wdir_emb_size = 6

        self.hidR = 48
        self.num_layers = settings["lstm_layer"]
        self.out_dim = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])

        self.numeric_embedding = nn.Linear(self.hidC, self.num_emb_size)
        self.h0_embedding = nn.Embedding(num_embeddings=144, embedding_dim=self.hidR * settings["lstm_layer"])
        self.time_embedding = nn.Embedding(num_embeddings=144, embedding_dim=self.time_emb_size)
        self.id_embedding = nn.Embedding(num_embeddings=134, embedding_dim=self.id_emb_size)

        # The following three lines are useless, but they cannot be deleted, otherwise it will not be fully reproduced.
        self.pab_embedding = nn.Embedding(num_embeddings=7, embedding_dim=self.pab_emb_size)
        self.ndir_embedding = nn.Embedding(num_embeddings=13, embedding_dim=self.ndir_emb_size)
        self.wdir_embedding = nn.Embedding(num_embeddings=10, embedding_dim=self.wdir_emb_size)

        self.lstm = nn.GRU(input_size=self.num_emb_size + self.time_emb_size,
                           hidden_size=self.hidR, num_layers=self.num_layers, batch_first=False)
        self.projection = nn.Linear(self.hidR + self.id_emb_size, self.out_dim)

    def forward(self, numerical_features, categorical_features=None, output_len=288):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            output_len:
            categorical_features:
            numerical_features:
            x_enc:
        Returns:
            A tensor
        """

        numerical_features = numerical_features.cuda()
        categorical_features = categorical_features.cuda()

        x_enc = numerical_features  # [batch, input_len, hidC]
        x_enc = self.numeric_embedding(x_enc)  # [batch, input_len, num_emb_size]

        h0_time = categorical_features[:, 0, 0]  # input start time [batch]
        h0_emb = self.h0_embedding(h0_time)  # [batch, h0_emb_size]
        h0 = h0_emb.reshape((-1, self.num_layers, self.hidR)).permute(1, 0, 2)

        # generate the x_time feature for output
        x_time = categorical_features[:, :, 0]
        x_time_list = [x_time]
        for i in range(output_len // self.input_len + 1):
            x_time_list.append(x_time_list[-1] + self.input_len)
        x_time = torch.cat(tuple(x_time_list), 1)[:, :self.input_len + output_len] % 144
        x_time_emb = self.time_embedding(x_time)  # [batch, input_len + output_len, time_emb_size]

        tid = categorical_features[:, 0, 2]  # tid = Turb_id - 1
        tid_emb = self.id_embedding(tid)  # [batch, id_emb_dim]
        tid_emb = torch.tile(tid_emb.unsqueeze(1),
                             (1, self.input_len + output_len, 1))  # [batch, input_len + output_len, id_emb_size]

        x = torch.zeros([x_enc.shape[0], output_len, self.num_emb_size]).to(device=numerical_features.device)
        x_enc = torch.cat((x_enc, x), 1)  # [batch, input_len + output_len, num_emb_size]

        x_enc = torch.cat((x_enc, x_time_emb), 2).permute(1, 0, 2)  # [input_len + output_len, batch, num_emb_size + time_emb_size]

        dec, _ = self.lstm(x_enc, h0.contiguous())  # [input_len + output_len, batch, num_emb_size + time_emb_size]

        dec = torch.cat((dec.permute(1, 0, 2), tid_emb), 2)  # [batch, input_len + output_len, hidR + id_emb_size]
        sample = self.projection(self.dropout(dec))  # [batch, input_len + output_len, 1]
        sample = sample[:, -output_len:, -self.out_dim:]  # [batch, output_len, 1]
        return sample.squeeze(2)  # [batch, output_len]
