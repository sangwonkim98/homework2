import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np

class EncoderLSTM_Att(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, word_embedding_numpy=None):
        super(EncoderLSTM_Att, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

        self.embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))

    def forward(self, input, length):
        embedded = self.embedding(input) # 단어 index를 통해 word embedding matrix에서 vector look up
        output = embedded
        output = pack_padded_sequence(output, lengths=torch.tensor(length), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(output)
        output = pad_packed_sequence(output)[0]  # pack_padded_sequence를 tuple 형태로 변환 [length, batch_size, hidden_size]
        hidden = hidden.permute((1, 0, 2))  # [batch_size, layer * direction, hidden_size]
        hidden = hidden.reshape(hidden.shape[0], -1)  # [batch_size, hidden_size * layer * direction]
        return output, hidden, cell

class DecoderLSTM_Att(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size):
        super(DecoderLSTM_Att, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.att_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax_final = nn.LogSoftmax(dim=1)

    def forward(self, enc_output, input, hidden, cell):
        embed = self.embedding(input)

        output, (hidden, cell) = self.lstm(embed.unsqueeze(0), (hidden, cell))

        enc_output = enc_output.permute((1, 0, 2)) # encoder의 각 단어별 state
        # embed : [batch_size, embedding_size]
        # hidden : [1, batch_size, hidden_size]
        # enc_output : [batch_size, length, hidden_size]
        att_score = torch.bmm(enc_output, hidden.permute((1,2,0)))
        # bmm : batch-wise matrix multiplication
        # [batch_size, length, 1]
        att_dist = self.softmax(att_score)
        # [batch_size, length, 1]
        att_output = enc_output * att_dist  # broadcasting!!
        # att_output : [batch_size, length, hidden_size]
        att_output = torch.sum(att_output, dim=1).unsqueeze(0)  # [1, batch_size, hidden_size]

        hidden = torch.cat((hidden, att_output), dim=-1)  # [1, batch_size, hidden_size*2]
        hidden = self.att_combine(hidden)  # [1, batch_size, hidden_size]

        # hidden : [1, batch_size, hidden_size]
        output = self.softmax_final(self.out(hidden[0]))
        return output, hidden, cell
