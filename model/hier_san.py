import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
class hier_san(nn.Module):
    def __init__(self,stack_size,vocab_size,ans_size,embed_size,lstm_hidden_size,channel_size,loc_size,seq_size,feat_hidden_size):
        super(hier_san,self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        self.stack_size = vocab_size
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.out_nonlinear = nn.LogSoftmax()
        self.out_linear = nn.Linear(lstm_hidden_size + channel_size,ans_size)
        self.lstm_layer = 1
        self.bidirectional_flag = False
        self.direction = 2 if self.bidirectional_flag else 1
        self.question_lstm = nn.LSTM(embed_size, lstm_hidden_size,
                                    num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
        ##TODO change the bmm to linear
        self.affi = nn.Linear(lstm_hidden_size,channel_size,bias=False)
        self.linear_i = nn.Linear(channel_size,feat_hidden_size,bias = False)
        self.linear_q = nn.Linear(lstm_hidden_size,feat_hidden_size,bias = False)
        self.att_i = nn.Sequential(
            nn.Linear(feat_hidden_size,1,bias = False),
            nn.Softmax()
        )
        self.att_q = nn.Sequential(
            nn.Linear(feat_hidden_size,1,bias = False),
            nn.Softmax()
        )
        self.feat_hidden_size = feat_hidden_size
        self.seq_size = seq_size
        self.img_size = loc_size * loc_size
        self.channel_size = channel_size

    def forward(self,q,v,q_length,param):
        batch_size ,_,_,_ = v.size()
        self.batch_size = batch_size
        q_c_0 = self.init_hidden(param)
        q_h_0 = self.init_hidden(param)

        v = v.view(-1,self.channel_size,self.img_size) # (b, c, s)
        #LSTM 
        q = self.embed(q)
        q = torch.nn.utils.rnn.pack_padded_sequence(q, list(q_length.data.type(torch.LongTensor)), batch_first=True)
        self.question_lstm.flatten_parameters()
        q, (q_h_t,q_c_t) = self.question_lstm(q,(q_h_0,q_c_0)) # (b, l, h)
        q, _ = torch.nn.utils.rnn.pad_packed_sequence(q, batch_first=True)
        q = q.transpose(1,2).contiguous() # (b, h, l)


        #ATT
        a_q = Variable(torch.ones(batch_size,self.seq_size,1)) # (b,l,1)
        a_i = Variable(torch.ones(batch_size,self.img_size,1)) # (b,s,1)
        c = F.tanh(torch.bmm(self.affi(q.transpose(1,2).contiguous().view(-1,self.lstm_hidden_size)).view(-1,self.seq_size,self.channel_size),v)) # (b, l, h) dot (h,c) dot (b,c,s) -> (b, l, s)

        ##TODO reshuffle the tensor to reduce computation
        for i in range(self.stack_size):
            a_and_i = a_i.expand(self.channel_size,self.img_size)*v
            w_i_i = self.linear_i(a_and_i.transpose(1,2).contiguous().view(-1,self.channel_size)).view(-1,self.feat_hidden_size,self.img_size) # (b, c, s) * (b, c, s) and (b, c, s) dot (k, c) -> (b,k,s)
            a_and_q = q_i.expand(self.lstm_hidden_size,self.seq_size)*q
            w_q_q =self.linear_q(a_and_q.transpose(1,2).contiguous().view(-1,self.lstm_hidden_size)).view(-1,self.feat_hidden_size,self.seq_size) # (b, h, l) * (b, h, l) and (b, h, l) dot (k, h ) -> (b,k,l)
            h_i = F.tanh(w_i_i + torch.bmm(w_q_q,c)) # (b, k, s)
            h_q = F.tanh(w_q_q + torch.bmm(w_i_i,c.transpose(0,1).contiguous())) #(b, k, l)
            a_q = self.att_q(h_q.transpose(1,2).contiguous().view(-1, self.feat_hidden_size)).view(-1,self.seq_size) # (b, l)
            a_i = self.att_i(h_i.transpose(1,2).contiguous().view(-1, self.feat_hidden_size)).view(-1,self.img_size) # (b, s)

        q_star = torch.bmm(q,a_q.unsqueeze(2)).squeeze() # (b, h, len) * (b, len, 1) -> (b, h, 1)
        i_star = torch.bmm(v,q_i.unsqueeze(2)).squeeze()

        out = self.out_nonlinear(self.out_linear(torch.cat([q_star,i_star],1)))

        return out

    def init_hidden(self,param):
        direction = 2 if self.bidirectional_flag else 1
        return autograd.Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.lstm_hidden_size).cuda(async=True),**param)


