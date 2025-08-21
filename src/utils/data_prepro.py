import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class TranslateDataset(Dataset):
    def __init__(self, source, target, device):
        self.src = source
        self.tgt = target
        self.device = device

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.LongTensor(self.src[idx]).to(self.device), torch.LongTensor(self.tgt[idx]).to(self.device)

def collate(batch):
    src, tgt = zip(*batch)
    src_len = []
    tgt_len = []
    for t in src:
        src_len.append(len(t))
    for t in tgt:
        tgt_len.append(len(t))
    src_text = pad_sequence(src, batch_first=True)
    tgt_text = pad_sequence(tgt, batch_first=True)
    src_lengths = torch.tensor(src_len)
    tgt_lengths = torch.tensor(tgt_len)
    return src_text, tgt_text, src_lengths, tgt_lengths