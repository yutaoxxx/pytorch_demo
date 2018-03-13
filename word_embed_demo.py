#-*—encoding:utf8-*-
'''
@author:yutao
@description:a demo for word_embeding
'''
import torch
import torch.nn as nn
word2ix = {'hello': 0, 'word' : 1}
#参数(词典长度, 词典维数)
embeds = nn.Embedding(2, 5)
hello_idx = torch.LongTenser([word2ix['hello']])
hello_idx = Variable(hello_idx)

hello_embed = embeds(hello_idx)
print(hello_embed)