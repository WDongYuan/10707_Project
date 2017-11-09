import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

class RelationalNetwork(nn.Module):
	def __init__(self,voc_size,word_embedding_size,in_channel,out_channel,map_w,map_h,answer_voc_size,lstm_hidden_size,g_mlp_hidden_size):
		##Embedding
		super(RelationalNetwork, self).__init__()
		self.voc_size = voc_size
		self.word_embedding_size = word_embedding_size
		self.embed = nn.Embedding(self.voc_size, self.word_embedding_size,padding_idx=0)

		##Attention
		self.att_linear = nn.Linear((in_channel+lstm_hidden_size)*map_h*map_w,map_h*map_w, bias=False)

		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = False
		self.direction = 2 if self.bidirectional_flag else 1
		self.lstm_hidden_size = lstm_hidden_size
		self.question_lstm = nn.LSTM(self.word_embedding_size, self.lstm_hidden_size,
									num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		# self.q_c_0 = self.init_hidden()
		# self.q_h_0 = self.init_hidden()

		##CNN
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.map_w = map_w
		self.map_h = map_h
		self.kernel_size = 3
		self.padding_size = (self.kernel_size-1) // 2
		self.stride = 1
		self.conv = nn.Sequential(
			nn.Conv2d(in_channel,out_channel,(self.kernel_size,self.kernel_size),stride=self.stride,padding=self.padding_size),
			nn.BatchNorm2d(self.out_channel),
			nn.ReLU())

		self.obj_num = self.map_w*self.map_h
		self.concat_length = self.out_channel*2+self.lstm_hidden_size*self.lstm_layer*self.direction

		##g_mlp
		self.g_mlp_hidden_size = g_mlp_hidden_size
		self.g_mlp = nn.Sequential(
			nn.Linear(self.concat_length,self.g_mlp_hidden_size),
			nn.Tanh())

		##f_mlp
		self.answer_voc_size = answer_voc_size
		self.f_mlp = nn.Sequential(
			nn.Linear(self.g_mlp_hidden_size + self.in_channel + self.lstm_hidden_size ,self.answer_voc_size),
			nn.Tanh())
		self.LogSoftmax = nn.LogSoftmax()

	def forward(self,sent_batch,conv_map_batch,sents_lengths,param):
		batch_size , sentlength = sent_batch.size()
		self.batch_size = batch_size
		sent_emb = self.embed(sent_batch)

		##CNN
		out = self.conv(conv_map_batch)

		##LSTM
		q_c_0 = self.init_hidden(param)
		q_h_0 = self.init_hidden(param)
		pack_sent = torch.nn.utils.rnn.pack_padded_sequence(sent_emb, list(sents_lengths.data.type(torch.LongTensor)), batch_first=True)
		self.question_lstm.flatten_parameters()
		q_h_n, (q_h_t,q_c_t) = self.question_lstm(pack_sent,(q_h_0,q_c_0))

		##Relation
		lstm_expand = q_h_t.permute(1,0,2).contiguous().view(self.batch_size,1,self.lstm_hidden_size*self.lstm_layer*self.direction).expand(self.batch_size,self.obj_num**2,self.lstm_hidden_size*self.lstm_layer*self.direction)
		out = out.permute(0,2,3,1)
		conv_expand_1 = out.unsqueeze(1).expand(self.batch_size,self.obj_num,self.map_h,self.map_w,self.out_channel)
		conv_expand_1 = conv_expand_1.contiguous().view(self.batch_size,self.obj_num**2,self.out_channel)
		conv_expand_2 = out.unsqueeze(3).expand(self.batch_size,self.map_h,self.map_w,self.obj_num,self.out_channel)
		conv_expand_2 = conv_expand_2.contiguous().view(self.batch_size,self.obj_num**2,self.out_channel)
		out = torch.cat((lstm_expand,conv_expand_1,conv_expand_2),2)
		out = self.g_mlp(out.view(-1,self.concat_length)).view(self.batch_size,-1,self.g_mlp_hidden_size).sum(1).squeeze()

		#Attention
		q = q_h_t.permute(1,0,2).view(self.batch_size,-1).unsqueeze(2).expand(self.batch_size,self.lstm_hidden_size,self.map_h*self.map_w)
		i = conv_map_batch.view(self.batch_size,-1,self.map_h*self.map_w)
		attention = torch.cat([i,q],1).view(self.batch_size,-1)
		attention = F.tanh(self.att_linear(attention).view(-1,self.map_h*self.map_w))
		attention = nn.softmax(attention).unsqueeze(1).expand(self.batch_size,self.in_channel+self.lstm_hidden_size,self.map_h*self.map_w)
		attention = (attention*conv_map_batch).sum(2).squeeze()

		#Classifier
		out = torch.cat([attention,out,q_h_t],1)
		out = self.f_mlp(out)
		out = self.LogSoftmax(out)
		return out

	def init_hidden(self,param):
		direction = 2 if self.bidirectional_flag else 1
		return autograd.Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.lstm_hidden_size).cuda(async=True),**param)



