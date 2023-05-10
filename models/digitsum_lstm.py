# -*- coding: utf-8 -*-
# Author: jlgao HIT-SCIR
import torch.nn as nn
import torch


class DigitsumLstm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(11, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, dropout=0.25)
        self.dense = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs, lengths):
        self.lstm.flatten_parameters()
        inputs = inputs.transpose(0, 1)
        assert inputs.dim() == 2
        inputs = self.embed(inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths.tolist(), enforce_sorted=False)
        h0 = torch.randn(1, lengths.size(0), self.hidden_size, device=inputs.device)
        c0 = torch.randn(1, lengths.size(0), self.hidden_size, device=inputs.device)
        _, (hn, _) = self.lstm(packed_inputs, (h0, c0))
        # hn: num_layers * num_directions, batch, hidden_size
        logits = self.dense(hn)
        logits = self.relu(logits).squeeze(dim=0)
        return logits


# lengths tensor([ 4, 12,  9], device='cuda:0') torch.Size([3]) torch.int64 cuda:0
# import torch
# a=torch.tensor([12,5,7],dtype=torch.int64,device="cuda")
# torch.as_tensor(a, dtype=torch.int64)


if __name__ == '__main__':
    dv = "cpu"
    model = DigitsumLstm(hidden_size=10)
    model.to(dv)
    inputs = torch.tensor([[1, 2, 3, 5, 10, 10], [1, 5, 0, 10, 10, 10]], dtype=torch.long, device=dv)
    lengths = torch.tensor([4, 3], dtype=torch.long, device=dv)
    logits=model(inputs, lengths)
    loss_e=nn.MSELoss(reduction='none').to(dv)
    loss = loss_e(logits.squeeze(dim=-1), torch.tensor([1,2],dtype=torch.float,device=dv))
    loss=loss.mean()
    print(loss)
    loss.backward()