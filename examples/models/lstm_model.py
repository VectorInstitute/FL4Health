import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, vocab_size: int, vocab_dimension: int = 128, lstm_dimension: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_dimension)
        self.lstm_dimension = lstm_dimension
        self.lstm = nn.LSTM(
            input_size=vocab_dimension,
            hidden_size=lstm_dimension,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(2 * lstm_dimension, 4)

    def forward(self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        text_emb = self.embedding(x)
        out, _ = self.lstm(text_emb, hidden)

        out_forward = out[:, -1, : self.lstm_dimension]
        out_reverse = out[:, 0, self.lstm_dimension :]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        return nn.LeakyReLU()(text_fea)

    def init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 4 since the number of layers is 2 and it is bidirectional (so 1 per layer per direction)
        return (torch.zeros(4, batch_size, self.lstm_dimension), torch.zeros(4, batch_size, self.lstm_dimension))
