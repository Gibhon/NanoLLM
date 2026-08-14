import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# -------This Works but is too memory inefficient :(-------
#
# class TransformerEmbeddingScratch(nn.Module):
#     def __init__(self, vocab_size, embed_dim, max_seq_len):
#         self.vocab_size = vocab_size
#         self.max_seq_len = max_seq_len

#         self.token_embed = nn.Linear(vocab_size, embed_dim, bias=False)
#         self.pos_embed = nn.Linear(max_seq_len, embed_dim, bias=False)

#     def forward(self, x):
#         seq_len = x.size(1)

#         # Shape: (batch_size, seq_len) -> (batch_size, seq_len, vocab_size)
#         # [1, 2, 3] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]   || (3) -> (3, 3) HERE: Vocab in place of 3
#         x_one_hot = F.one_hot(x, num_classes=self.vocab_size).float()

#         positions = torch.arange(0, seq_len, device=x.device)

#         # TODO: Understand the Query
#         # ? Why max_seq_len when positions is only till seq_len anyways
#         # Shape: (seq_len) -> (seq_len, max_seq_len)
#         pos_onehot = F.one_hot(
#             positions,
#             num_classes=self.max_seq_len,
#         ).float()

#         # (seq_len, self.max_seq_len) -> (1, seq_len, max_seq_len)
#         pos_onehot = pos_onehot.unsqueeze(0)

#         # (batch_size, seq_len, vocab_size) @ (vocab, embed) -> (batch, seq, embed)
#         token_embedding = self.token_embed(x_one_hot)
#         # (1, seq, max) @ (max, embed) -> (1, seq, embed)
#         pos_embedding = self.pos_embed(pos_onehot)

#         # 1 is broadcasted to batch_size
#         return token_embedding + pos_embedding


class TransformerEmbeddingScratch(nn.Module):
    def __init__(self, embed_dim, vocab_size, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device)

        tok_embedding = self.token_embed(x)
        pos_embedding = self.pos_embed(positions)
        return tok_embedding + pos_embedding


class AttentionScratch(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        assert (
            embed_dim % n_heads == 0
        ), "Embedding dimension must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # self.query_t = nn.Linear(embed_dim, embed_dim)
        # self.key_t = nn.Linear(embed_dim, embed_dim)
        # self.value_t = nn.Linear(embed_dim, embed_dim)
        self.qkv_t = nn.Linear(embed_dim, embed_dim * 3)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        qkv = self.qkv_t(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Final Shape: (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # (b, n_h, s, s) @ (b, n_h, s, h_d) ==> (b, n_h, s, h_d)
        attended_values = torch.matmul(attention_weights, v)

        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, embed_dim)

        return self.output_projection(attended_values)


class FeedForwardScratch(nn.Module):
    def __init__(self, embed_dim, expansion_factor, dropout):
        super().__init__()
        self.fc1 = nn.Linear(
            embed_dim, embed_dim * expansion_factor
        )  # Order-> embed_dim * feedforward_dim
        self.fc2 = nn.Linear(embed_dim * expansion_factor, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class TransformerBlockScratch(nn.Module):
    def __init__(self, embed_dim, n_heads,dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = AttentionScratch(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForwardScratch(embed_dim, 4, dropout=dropout)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x


class NanoLLMScratch(nn.Module):
    def __init__(self, tokenizer, max_seq_len, embed_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.max_seq_len = max_seq_len
        vocab_size = tokenizer.vocab_size

        self.embedding = TransformerEmbeddingScratch(
            vocab_size=vocab_size, embed_dim=embed_dim, max_seq_len=max_seq_len
        )
        self.blocks = nn.ModuleList(
            [TransformerBlockScratch(embed_dim, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.lang_modeling_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size()[1]
        assert (
            seq_len <= self.max_seq_len
        ), f"Sequence Length{seq_len} exceeds max{self.max_seq_len}"

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.lang_modeling_head(x)

        return logits
