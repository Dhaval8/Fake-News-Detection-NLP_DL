import torch
import torch.nn as nn
from transformers import BertModel

class BERT_CNN(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', num_classes=2, dropout=0.3):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3 * 100, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        x = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len] for Conv1d

        x1 = torch.max_pool1d(self.relu(self.conv1(x)), kernel_size=x.size(2) - 1).squeeze(2)
        x2 = torch.max_pool1d(self.relu(self.conv2(x)), kernel_size=x.size(2) - 2).squeeze(2)
        x3 = torch.max_pool1d(self.relu(self.conv3(x)), kernel_size=x.size(2) - 3).squeeze(2)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out