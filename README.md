# 这是我实现的关于外卖评价的情绪判断项目

此项目使用了一个已经预训练好的bert语言模型，外接一个分类头。在此基础上进行微调。

在t.txt文件中输入要判断的外卖评价，运行应用代码即可对句子进行正面或负面的评价

其中dataset.py text_data.py（Vocab类为建立词表，未被使用） 为训练时数据处理代码

EncoderBlock.py FFN.py model.py model_bert.py Multi_Head_Attention.py  rresidual_layer_normalization.py为模型定义代码

teain.py为训练代码

vocab.pth为词表

bert.pth为已经预训练好的bert部分模型
