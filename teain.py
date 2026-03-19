import torch
from torch import nn
import dataset
import model

def try_gpu():
    if torch.cuda.is_available():
        return torch.device('cuda:0')  # 使用第一个GPU
    return torch.device('cpu')

def load_pretrained_bert(model, pretrained_path, freeze_bert=True):
    pretrained_weights = torch.load(pretrained_path)
    model_dict = model.state_dict()
    bert_dict = {k.replace("bert.", ""): v
                 for k, v in model_dict.items()
                 if k.startswith("bert.") and k.replace("bert.", "") in pretrained_weights}
    pretrained_dict = {"bert." + k: v for k, v in pretrained_weights.items() if k in bert_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False
    return model

def cs(train_iter_cs,net,devices):
    a = 0
    b = 0
    net.eval()
    with torch.no_grad():
        for tokens, attention_mask, labels in train_iter_cs:

            tokens = tokens.to(devices)
            attention_mask = attention_mask.to(devices)
            labels = labels.to(devices)

            segments = torch.zeros_like(tokens).to(devices)
            valid_lens = attention_mask.sum(dim=1)

            outputs = net(tokens, segments, valid_lens,pred_positions=None)
            _, predicted = torch.max(outputs, 1)
            a += (predicted == labels).sum().item()
            b += labels.size(0)
    return a / b

def train(train_iter,train_iter_cs ,net, loss, devices, num_steps):
    net.train()
    net = net.to(devices)
    trainer = torch.optim.Adam(net.parameters(), lr=0.0001)
    step_counter = 0
    while step_counter < num_steps:
        for tokens, attention_mask, labels in train_iter:
            tokens = tokens.to(devices)
            attention_mask = attention_mask.to(devices)
            labels = labels.to(devices)
            segments = torch.zeros_like(tokens).to(devices)
            valid_lens = attention_mask.sum(dim=1)
            output = net(tokens, segments, valid_lens,pred_positions=None)
            l = loss(output,labels)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            step_counter += 1
            if step_counter % 100 == 0:
                print(f"第{step_counter + 1}轮,正确率为:, {cs(train_iter_cs,net,devices)},损失:,{l.item():.4f}")

if __name__ == '__main__':
    batch_size, max_len = 512, 128
    train_iter,train_iter_cs, vocab= dataset.data(batch_size, max_len)
    net = model.Model(len(vocab),num_hiddens=768,norm_shape=[768],ffn_num_input=768,ffn_num_hiddens=3072,
                    num_heads=12,num_layers=12,dropout=0.1,key_size=768,query_size=768,value_size=768,
                    hid_in_features=768,mlm_in_features=768,nsp_in_features=768,linear_hiddens=3072)
    net = load_pretrained_bert(net,"bert.pth",False)
    devices = try_gpu()
    loss= nn.CrossEntropyLoss()
    train(train_iter,train_iter_cs, net,loss,devices, 500)