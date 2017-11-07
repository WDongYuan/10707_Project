import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RelationalNetwork(nn.Module):
	def __init__(self,voc_size,word_embedding_size,channel,map_w,map_h,answer_voc_size):
		##Embedding
		super(RelationalNetwork, self).__init__()
		self.voc_size = voc_size
		self.word_embedding_size = word_embedding_size
		self.embed = nn.Embedding(self.voc_size, self.word_embedding_size,padding_idx=0)

		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = True
		self.lstm_hidden_size = 500
		self.question_lstm = nn.LSTM(self.word_embedding_size, self.lstm_hidden_size,
									num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)

		##CNN
		self.channel = channel
		self.map_w = map_w
		self.map_h = map_h
		self.obj_num = self.map_w*self.map_h

		self.concat_length = self.channel*2+self.lstm_hidden_size

		##g_mlp
		self.g_mlp_hidden_size = 100
		self.g_mlp = nn.Sequential(
			nn.Linear(self.concat_length,self.g_mlp_hidden_size),
			nn.ReLU(),
			nn.Linear(self.g_mlp_hidden_size,self.g_mlp_hidden_size),
			nn.ReLU())

		##f_mlp
		self.answer_voc_size = answer_voc_size
		self.f_mlp_hidden_size = 100
		self.f_mlp = nn.Sequential(
			nn.Linear(self.g_mlp_hidden_size,self.f_mlp_hidden_size),
			nn.ReLU(),
			nn.Linear(self.f_mlp_hidden_size,self.answer_voc_size),
			nn.ReLU())
		self.LogSoftmax = nn.LogSoftmax()


	def forward(self,sent_batch,conv_map_batch,sents_lengths):
		batch_size , sentlength = sent_batch.size()
		self.batch_size = batch_size
		sent_emb = self.embed(sent_batch)

		##LSTM
		pack_sent = torch.nn.utils.rnn.pack_padded_sequence(sent_emb, list(sents_lengths.data), batch_first=True)
		q_c_0 = self.init_hidden()
		q_h_0 = self.init_hidden()
		q_h_n, (self.q_h_t,self.q_c_t) = self.question_lstm(pack_sent,(self.q_h_0,self.q_c_0))

		##Concat
		lstm_expand = q_h_t.permute(1,0,2).view(self.batch_size,-1,self.hidden_size).expand(self.batch_size,self.obj_num**2,self.hidden_size)
		out = conv_map_batch.permute(0,2,3,1)
		conv_expand_1 = out.unsqueeze(1).expand(self.batch_size,self.obj_num,self.map_h,self.map_w,self.channel)
		conv_expand_1 = conv_expand_1.contiguous().view(self.batch_size,self.obj_num**2,self.channel)
		conv_expand_2 = out.unsqueeze(3).expand(self.batch_size,self.map_h,self.map_w,self.obj_num,self.channel)
		conv_expand_2 = conv_expand_2.contiguous().view(self.batch_size,self.obj_num**2,self.channel)
		out = torch.cat((lstm_expand,conv_expand_1,conv_expand_2),2)

		out = self.g_mlp(out).sum(1).view(self.batch_size,self.concat_length)
		out = self.f_mlp(out)
		out = self.LogSoftmax(out)
		return out

	def init_hidden(self):
		direction = 2 if self.bidirectional_flag else 1
		return autograd.Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.lstm_hidden_size))



