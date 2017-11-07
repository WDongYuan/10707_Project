import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class BOWIMG(nn.Module):
    def __init__(self,dim_out,dict_size,word_embed_dim,img_dim):
        super(BOWIMG, self).__init__()
        self.fc = nn.Linear(word_embed_dim+img_dim,dim_out)
        self.embed = nn.Embedding(dict_size,word_embed_dim,padding_idx=0)
        self.nonlinear = nn.LogSoftmax()
        self.drop = nn.Dropout()
    def forward(self,q,v):
        out = self.embed(q)
        out = torch.sum(out,dim=1)
        out = torch.cat((v,out),1)
        out = self.fc(out)
        #out = self.drop(out)
        out = self.nonlinear(out)
        return out





