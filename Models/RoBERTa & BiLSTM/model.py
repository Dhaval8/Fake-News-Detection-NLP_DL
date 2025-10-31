import torch
import torch.nn as nn
from transformers import AutoModel

class RoBERTa_BiLSTM(nn.Module):
    """
    Distil/Roberta + BiLSTM classifier.
    - By default loads the model name passed (use a smaller variant for speed).
    - Transformer parameters are frozen by default (to train only LSTM + classifier).
    """

    def __init__(self,
                 roberta_model_name='distilroberta-base',
                 hidden_dim=128,
                 num_classes=2,
                 dropout=0.3,
                 freeze_transformer=True):
        super(RoBERTa_BiLSTM, self).__init__()

        # Use AutoModel for compatibility
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        transformer_hidden = self.roberta.config.hidden_size

        # Freeze transformer to speed up training (default True)
        if freeze_transformer:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # Optionally: unfreeze last encoder layer(s) if you want some fine-tuning
        # Uncomment to unfreeze last transformer layer:
        # try:
        #     for param in list(self.roberta.encoder.layer[-1].parameters()):
        #         param.requires_grad = True
        # except Exception:
        #     pass

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=transformer_hidden,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        # Initialize classifier weights (optional but helpful)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        returns logits: (batch, num_classes)
        """
        # transformer outputs: last_hidden_state (batch, seq_len, hidden)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq_output = outputs.last_hidden_state

        # pass through BiLSTM
        lstm_out, _ = self.lstm(seq_output)  # (batch, seq_len, hidden_dim*2)

        # simple mean pooling over sequence length
        pooled = torch.mean(lstm_out, dim=1)  # (batch, hidden_dim*2)

        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits
