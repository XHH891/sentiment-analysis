import torch
import dataset
import text_data
import teain
import model

def try_gpu():
    if torch.cuda.is_available():
        return torch.device('cuda:0')  # 使用第一个GPU
    return torch.device('cpu')

def data(file_path,max_len = 128):
    with open(file_path, 'r', encoding='utf-8') as f:
        a = f.readlines()
    a = text_data.tokenize(a)

    vocab = torch.load("vocab.pth", weights_only=False,map_location=torch.device('cpu'))
    b = dataset.get_data_from_paragraph(a)
    c = dataset.data_from_tokens_id(b,vocab)
    d = []
    for i in c:
        if len(i) > max_len:
            d.append(c[:max_len])
        else:
            d.append(i + [vocab['<pad>']] * (max_len - len(i)))
    attention_mask = []
    for sample in d:
        mask = [1 if token != vocab['<pad>'] else 0 for token in sample]
        attention_mask.append(mask)
    return d,attention_mask,vocab

def yy(data,attention_mask,net,devices):

    all_probabilities = []

    for i in range(len(data)):
        data_one = data[i]
        mask_one = attention_mask[i]

        token_tensor = torch.tensor([data_one], dtype=torch.long).to(devices)
        attention_tensor = torch.tensor([mask_one], dtype=torch.long).to(devices)
        segments = torch.zeros_like(token_tensor).to(devices)
        valid_lens = attention_tensor.sum(dim=1)
        net = net.to(devices)
        net.eval()
        with torch.no_grad():
            output = net(token_tensor, segments, valid_lens)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())
    return all_probabilities

data,attention_mask,vocab = data("t.txt")

devices = teain.try_gpu()
net = model.Model(len(vocab),num_hiddens=128,norm_shape=[128],ffn_num_input=128,ffn_num_hiddens=256,
                      num_heads=2,num_layers=8,dropout=0.2,key_size=128,query_size=128,value_size=128,
                      hid_in_features=128,mlm_in_features=128,nsp_in_features=128,linear_hiddens=256)

checkpoint = torch.load("情感分析.pth",map_location=torch.device('cpu'))
net.load_state_dict(checkpoint)

results = yy(data,attention_mask,net,devices)
for i in results:
    a = i.argmax()
    if a == 1:
        print("情感为正面")
    else:
        print("情感为负面")