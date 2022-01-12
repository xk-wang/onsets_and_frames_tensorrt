import torch
from torch import nn


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        hidden_size = self.rnn.hidden_size
        num_directions = 2 if self.rnn.bidirectional else 1
        batch_size, sequence_length, input_features = x.shape
        output, _ = self.rnn(x)

        return output

        # return self.rnn(x)[0]

        # 双向模型预测时为什么需要自己添加第二个反向的内容呢？
        # 画下图即可知道，切割时需要反向来保持反向的总体序列一致性
        # if self.training:
        #     return self.rnn(x)[0]
        # else:
        #     # evaluation mode: support for longer sequences that do not fit in memory
        #     batch_size, sequence_length, input_features = x.shape
        #     hidden_size = self.rnn.hidden_size
        #     num_directions = 2 if self.rnn.bidirectional else 1

        #     h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
        #     c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
        #     output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

        #     # forward direction
        #     slices = range(0, sequence_length, self.inference_chunk_length)
        #     for start in slices:
        #         end = start + self.inference_chunk_length
        #         output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

        #     # reverse direction
        #     if self.rnn.bidirectional:
        #         h.zero_()
        #         c.zero_()

        #         for start in reversed(slices):
        #             end = start + self.inference_chunk_length
        #             result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
        #             output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

        #     return output
