import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class hier_san(nn.Module):
    def __init__(self,stack_size,vocab_size,ans_size,embed_size,lstm_hidden_size,channel_size,loc_size,seq_size,feat_hidden_size):
        super(fusion_san,self).__init__()
        self.embed = vocab_size
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
        self.affi = Variable(torch.Tensor(channel_size,lstm_hidden_size))
        self.linear_i = nn.Linear(channel_size,feat_hidden_size,bias = False)
        self.linear_q = nn.Linear(lstm_hidden_size,feat_hidden_size,bias = False)
        self.att_i = nn.Sequential(
            nn.Linear(feat_hidden_size,1,bias = False),
            nn.SoftMax()
        )
        self.att_q = nn.Sequential(
            nn.Linear(feat_hidden_size,1,bias = False),
            nn.SoftMax()
        )
        self.feat_hidden_size = feat_hidden_size
        self.seq_size = seq_size
        self.img_size = loc_size * loc_size

    def forward(self,q,v,q_length,param):
        q_c_0 = self.init_hidden(param)
        q_h_0 = self.init_hidden(param)

        #LSTM 
        embedding = self.embed(q)
        pack_sent = torch.nn.utils.rnn.pack_padded_sequence(embedding, list(q_length.data.type(torch.LongTensor)), batch_first=True)
        self.question_lstm.flatten_parameters()
        q_h_n, (q_h_t,q_c_t) = self.question_lstm(pack_sent,(q_h_0,q_c_0))

        #ATT
        a_q = Variable(torch.ones(1,self.seq_size))
        a_i = Variable(torch.ones(1,self.img_size))
        c = F.tanh(torch.bmm(torch.bmm(q_h_n.transpose(),self.affi),v))

        for i in range(self.stack_size):
            w_i_v =self.linear_i(a_i.expand(self.channel_size,self.img_size))
            w_q_q =self.linear_q(q_i.expand(self.lstm_hidden_size,self.seq_size))
            h_i = F.tanh(w_i_v + torch.bmm(w_q_q,c))
            h_q = F.tanh(w_q_q + torch.bmm(w_i_v,c.transpose()))
            a_q = self.att_q(h_q)
            a_i = self.att_i(h_i)

        q_star = torch.bmm(a_q,q_h_n)
        i_star = torch.bmm(a_i,v)

        out = self.out_nonlinear(self.out_linear(torch.cat([q_star,i_star],1)))

        return out

    def init_hidden(self,param):
        direction = 2 if self.bidirectional_flag else 1
        return autograd.Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.lstm_hidden_size).cuda(async=True),**param)


