import model_bert
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768,linear_hiddens=256):
        super(Model, self).__init__()
        self.bert = model_bert.BERTModel(vocab_size, num_hiddens, norm_shape, ffn_num_input,ffn_num_hiddens,
                                         num_heads, num_layers, dropout,max_len, key_size, query_size,
                                         value_size,hid_in_features, mlm_in_features,nsp_in_features)
        self.fc1 = nn.Linear(num_hiddens,linear_hiddens)
        self.bn1 = torch.nn.BatchNorm1d(linear_hiddens)
        self.fc2 = nn.Linear(linear_hiddens, 2)
    def forward(self, tokens, segments, valid_lens=None,pred_positions=None):
        encoded_X,_,_ = self.bert(tokens, segments, valid_lens,pred_positions)
        cls_output = encoded_X[:, 0, :]
        x = self.fc1(cls_output)
        x = self.bn1(x)
        x = nn.functional.relu(x)  # 添加非线性激活
        x = self.fc2(x)
        return x