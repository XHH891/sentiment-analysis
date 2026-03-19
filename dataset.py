import torch
import text_data

def load_data(file_path, delimiter=','):
    labels, texts = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, text = line.split(delimiter, 1)  # 按第一个分隔符拆分
            labels.append(int(label))  # 转换为整数标签
            texts.append(text.strip('"'))  # 去除文本中的引号
    return labels, texts

def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

def get_data_from_paragraph(tokenizer):
    data_from_paragraph = []
    for i in tokenizer:
        tokens, _ = get_tokens_and_segments(i)
        data_from_paragraph.append(tokens)
    return data_from_paragraph

def data_from_tokens_id(tokens, vocab):
    token_ids_list = []
    for token in tokens:
        token_ids = [vocab[t] if t in vocab else vocab['<unk>'] for t in token]
        token_ids_list.append(token_ids)
    return token_ids_list

class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels,vocab,max_len=128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.vocab = vocab
        self.tokenizers = text_data.tokenize(self.texts)
        self.tokenizer = self.tokenizers
        self.tokenizer = get_data_from_paragraph(self.tokenizer)
        self.tokenizer = data_from_tokens_id(self.tokenizer, self.vocab)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label = self.labels[idx]
        token = self.tokenizer[idx]
        print(token)
        if len(token) > self.max_len:
            token = token[:self.max_len]
        else:
            token = token + [self.vocab['<pad>']] * (self.max_len - len(token))
        attention_mask = [1 if token != self.vocab['<pad>'] else 0 for token in token]
        token_tensor = torch.tensor(token, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return (token_tensor,attention_mask_tensor,label_tensor)

def data(batch_size, max_len=128):

    labels,texts = load_data(r"data\waimai_10k.txt")
    labels_cs,text_cs = load_data(r"data\waimai_10k_测试.txt")

    pretrained_vocab = torch.load("vocab.pth")

    dataset = Dataset(texts, labels,pretrained_vocab,max_len)
    dataset_cs = Dataset(text_cs, labels_cs,pretrained_vocab,max_len)

    dataloader_cs = torch.utils.data.DataLoader(dataset_cs, batch_size, shuffle=False,num_workers=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,num_workers=4)
    return dataloader,dataloader_cs,pretrained_vocab

