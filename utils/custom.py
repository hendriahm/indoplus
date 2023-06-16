import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizerFast
class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          ### New layers:
          self.lstm = nn.LSTM(768, 256, batch_first=True,bidirectional=True)
          self.linear = nn.Linear(256*2, 5)
          
    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(
               ids, 
               attention_mask=mask)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
          hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
          linear_output = self.linear(hidden.view(-1,256*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification

          return linear_output

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = CustomBERTModel()
