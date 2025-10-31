import torch
import torch.nn as nn
from transformers import BertModel

class BERT_BiLSTM(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', lstm_hidden_dim=128, num_classes=2):
        super(BERT_BiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_state)
        pooled_output = torch.mean(lstm_out, dim=1)
        output = self.dropout(pooled_output)
        logits = self.fc(output)
        return logits
