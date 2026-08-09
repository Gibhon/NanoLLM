import torch
from torch.utils.data import Dataset


class NanoLLMDataset(Dataset):
    def __init__(self, token_ids, block_size):
        super().__init__()
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index):
        x = self.token_ids[index : index + self.block_size]
        y = self.token_ids[index + 1 : index + self.block_size + 1]
        return x, y
