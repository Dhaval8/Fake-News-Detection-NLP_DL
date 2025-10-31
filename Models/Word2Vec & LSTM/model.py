import torch
import torch.nn as nn

class LSTM_Word2Vec(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_classes=2):
        super(LSTM_Word2Vec, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        pooled_output = torch.mean(lstm_out, dim=1)
        out = self.dropout(pooled_output)
        return self.fc(out)
