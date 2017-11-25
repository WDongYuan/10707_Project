import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.init as init

class hier_glimpse(nn.Module):
    def __init__(self,glimpse_size,vocab_size,ans_size,embed_size,lstm_hidden_size,channel_size,loc_size,feat_hidden_size,out_hidden_size,drop_out):
        super(hier_glimpse,self).__init__()
        self.glimpse_size = glimpse_size
        self.classifier = nn.Sequential(
                                nn.Linear(lstm_hidden_size*self.glimpse_size + channel_size*self.glimpse_size,out_hidden_size),
                                nn.ReLU(),
                                nn.Dropout(drop_out),
                                nn.Linear(out_hidden_size,ans_size),
                                nn.LogSoftmax()
                                )
        self.text = TextNN(vocab_size,ans_size,embed_size,lstm_hidden_size,drop_out)
        # self.att = []
        # for i in range(self.glimpse_size):
        #     attention = Attention(lstm_hidden_size,channel_size,loc_size,feat_hidden_size,drop_out)
        #     self.att.append(attention)
        # self.attention = Attention(lstm_hidden_size,channel_size,loc_size,feat_hidden_size,drop_out)

        self.att1 = Attention(lstm_hidden_size,channel_size,loc_size,feat_hidden_size,drop_out)
        if self.glimpse_size==2:
            self.att2 = Attention(lstm_hidden_size,channel_size,loc_size,feat_hidden_size,drop_out)

        self.img_size = loc_size * loc_size
        self.channel_size = channel_size
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,q,v,q_length,param):
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        v = v.view(-1,self.channel_size,self.img_size) # (b, c, s)
        q = self.text(q,q_length,param)

        q_att = None
        v_att = None

        a_v1,a_q1 = self.att1(q,v,param)
        q1 = torch.bmm(q.transpose(1,2),a_q1).squeeze() # (b, h, len) * (b, len, 1) -> (b, h, 1)
        v1 = torch.bmm(v,a_v1).squeeze()
        q_att = q1
        v_att = v1

        if self.glimpse_size==2:
            a_v2,a_q2 = self.att2(q,v,param)
            q2 = torch.bmm(q.transpose(1,2),a_q2).squeeze() # (b, h, len) * (b, len, 1) -> (b, h, 1)
            v2 = torch.bmm(v,a_v2).squeeze()
            q_att = torch.cat([q_att,q2],1)
            v_att = torch.cat([v_att,v2],1)

        out = self.classifier(torch.cat([q_att,v_att],1))

        # a_v,a_q = self.attention(q,v,param)
        # q = torch.bmm(q.transpose(1,2),a_q).squeeze() # (b, h, len) * (b, len, 1) -> (b, h, 1)
        # v = torch.bmm(v,a_v).squeeze()
        # out = self.classifier(torch.cat([q,v],1))

        return out

class TextNN(nn.Module):
    def __init__(self,vocab_size,ans_size,embed_size,lstm_hidden_size,drop_out):
        super(TextNN,self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layer = 1
        self.bidirectional_flag = False
        self.direction = 2 if self.bidirectional_flag else 1
        self.question_lstm = nn.LSTM(embed_size, lstm_hidden_size,
                                    num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
        self.drop = nn.Dropout(drop_out)
        init.xavier_uniform(self.embed.weight)
        self._init_lstm(self.question_lstm.weight_ih_l0)
        self._init_lstm(self.question_lstm.weight_hh_l0)

    def forward(self,q,q_length,param):
        batch_size = q.size()[0]
        self.batch_size = batch_size
        q_c_0 = self.init_hidden(param)
        q_h_0 = self.init_hidden(param)
        #LSTM 
        q = F.tanh(self.embed(q))
        q = torch.nn.utils.rnn.pack_padded_sequence(self.drop(q), list(q_length.data.type(torch.LongTensor)), batch_first=True)
        self.question_lstm.flatten_parameters()
        q, (q_h_t,q_c_t) = self.question_lstm(q,(q_h_0,q_c_0)) # (b, l, h)
        del q_h_t,q_c_t
        q, _ = torch.nn.utils.rnn.pad_packed_sequence(q, batch_first=True)
        #q = q.transpose(1,2).contiguous() # (b, h, l)
        q=q.contiguous()
        return q

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform(w)

    def init_hidden(self,param):
        direction = 2 if self.bidirectional_flag else 1
        return autograd.Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.lstm_hidden_size).cuda(async=True),**param)

class Attention(nn.Module):
    def __init__(self,lstm_hidden_size,channel_size,loc_size,feat_hidden_size,drop_out):
        super(Attention,self).__init__()
        ##TODO change the bmm to linear
        self.affi = nn.Linear(lstm_hidden_size,channel_size,bias=False)
        self.linear_i = nn.Linear(channel_size,feat_hidden_size,bias=False)
        self.linear_uniform_init(self.linear_i)

        self.linear_q = nn.Linear(lstm_hidden_size,feat_hidden_size,bias=False)
        self.linear_uniform_init(self.linear_q)

        self.att_i = nn.Linear(feat_hidden_size,1,bias=False)
        self.linear_uniform_init(self.att_i)

        self.att_q = nn.Linear(feat_hidden_size,1,bias=False)
        self.linear_uniform_init(self.att_q)

        self.feat_hidden_size = feat_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.img_size = loc_size * loc_size
        self.channel_size = channel_size
        self.drop = nn.Dropout(drop_out)

    def forward(self,q,v,param):
        #ATT
        batch_size ,seq_size = q.size()[:2]
        out_q = Variable((torch.ones(batch_size,seq_size,1).float()/seq_size).cuda(async=True),**param) # (b,l,1)
        out_i = Variable((torch.ones(batch_size,self.img_size,1).float()/self.img_size).cuda(async=True),**param) # (b,s,1)
        # v = self.drop(v)
        # q = self.drop(q)
        # print(self.lstm_hidden_size)
        # print(self.channel_size)
        # print(seq_size)
        # print(q.size())
        # print(q)
        # print(self.affi(q.view(-1,self.lstm_hidden_size)).size())
        # print(v.size())
        c = F.tanh(torch.bmm(self.affi(q.view(-1,self.lstm_hidden_size)).view(-1,seq_size,self.channel_size),v)) # (b, l, h) dot (h,c) dot (b,c,s) -> (b, l, s)
        out_i = v.transpose(1,2).contiguous()
        out_i = self.linear_i(out_i.view(-1,self.channel_size)).contiguous().view(-1,self.img_size,self.feat_hidden_size).permute(0,2,1)
        # out_i = self.linear_i(out_i.view(-1,self.channel_size)).view(-1,self.feat_hidden_size,self.img_size) # (b, c, s) * (b, c, s) and (b, c, s) dot (k, c) -> (b,k,s)
        out_q = q
        out_q = self.linear_q(out_q.view(-1,self.lstm_hidden_size)).view(-1,seq_size,self.feat_hidden_size).permute(0,2,1) # (b, h, l) * (b, h, l) and (b, h, l) dot (k, h ) -> (b,k,l)
        h_i = F.tanh(out_i + torch.bmm(out_q,c)) # (b, k, s)
        h_q = F.tanh(out_q + torch.bmm(out_i,c.transpose(1,2))) #(b, k, l)
        out_q = self.drop(h_q)
        out_i = self.drop(h_i)
        out_q = F.softmax(self.att_q(out_q.transpose(1,2)).squeeze()).unsqueeze(2) # (b, l)
        out_i = F.softmax(self.att_i(out_i.transpose(1,2)).squeeze()).unsqueeze(2) # (b, s)
        return out_i,out_q

    def linear_uniform_init(self,layer):
        init.xavier_uniform(layer.weight.data)
        if layer.bias is not None:
            init.xavier_uniform(layer.bias.data)
        # init.uniform(layer.weight.data,a=-0.01,b=0.01)
        # init.uniform(layer.bias.data,a=-0.01,b=0.01)

'''
class Attention(nn.Module):
    def __init__(self,stack_size,lstm_hidden_size,channel_size,loc_size,feat_hidden_size,drop_out):
        super(Attention,self).__init__()
        self.stack_size = stack_size
        ##TODO change the bmm to linear
        self.affi = nn.Linear(lstm_hidden_size,channel_size,bias=False)
        self.linear_i = nn.Linear(channel_size,feat_hidden_size,bias=False)
        self.linear_q = nn.Linear(lstm_hidden_size,feat_hidden_size,bias=False)
        self.att_i = nn.Linear(feat_hidden_size,1,bias=False)
        self.att_q = nn.Linear(feat_hidden_size,1,bias=False)
        self.feat_hidden_size = feat_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.img_size = loc_size * loc_size
        self.channel_size = channel_size
        self.drop = nn.Dropout(drop_out)

    def forward(self,q,v,param):
        #ATT
        batch_size ,seq_size = q.size()[:2]
        out_q = Variable((torch.ones(batch_size,seq_size,1).float()/seq_size).cuda(async=True),**param) # (b,l,1)
        out_i = Variable((torch.ones(batch_size,self.img_size,1).float()/self.img_size).cuda(async=True),**param) # (b,s,1)
        # v = self.drop(v)
        # q = self.drop(q)
        c = F.tanh(torch.bmm(self.affi(q.view(-1,self.lstm_hidden_size)).view(-1,seq_size,self.channel_size),v)) # (b, l, h) dot (h,c) dot (b,c,s) -> (b, l, s)
        for i in range(self.stack_size):
            out_i = out_i.expand(batch_size,self.img_size,self.channel_size)*v.transpose(1,2)
            out_i = self.linear_i(out_i.view(-1,self.channel_size)).view(-1,self.feat_hidden_size,self.img_size) # (b, c, s) * (b, c, s) and (b, c, s) dot (k, c) -> (b,k,s)
            out_q = out_q.expand(batch_size,seq_size,self.lstm_hidden_size)*q
            out_q = self.linear_q(out_q.view(-1,self.lstm_hidden_size)).view(-1,self.feat_hidden_size,seq_size) # (b, h, l) * (b, h, l) and (b, h, l) dot (k, h ) -> (b,k,l)
            h_i = F.tanh(out_i + torch.bmm(out_q,c)) # (b, k, s)
            h_q = F.tanh(out_q + torch.bmm(out_i,c.transpose(1,2))) #(b, k, l)
            out_q = self.drop(h_q)
            out_i = self.drop(h_i)
            out_q = F.softmax(self.att_q(out_q.transpose(1,2)).squeeze()).unsqueeze(2) # (b, l)
            out_i = F.softmax(self.att_i(out_i.transpose(1,2)).squeeze()).unsqueeze(2) # (b, s)
        return out_i,out_q
'''
