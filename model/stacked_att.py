import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.init import kaiming_uniform
import torch.nn.init as init

class StackAttNetwork(nn.Module):
	def __init__(self,voc_size,word_embedding_size,map_c,map_w,map_h,
		answer_voc_size,lstm_hidden_size,feature_size,drop=0.0):
		
		super(StackAttNetwork, self).__init__()

		self.dropout = nn.Dropout(drop)
		self.img_space = map_w*map_h
		self.map_w = map_w
		self.map_h = map_h
		self.map_c = map_c
		self.feature_size = feature_size
		self.batch_size = 0
		self.answer_voc_size = answer_voc_size

		##Embedding
		self.voc_size = voc_size
		self.word_embedding_size = word_embedding_size
		self.embed = nn.Embedding(self.voc_size, self.word_embedding_size,padding_idx=0)
		# init.xavier_uniform(self.embed.weight)
		init.uniform(self.embed.weight,a=-0.01,b=0.01)

		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = False
		self.direction = 2 if self.bidirectional_flag else 1
		self.lstm_hidden_size = lstm_hidden_size
		self.question_lstm = nn.LSTM(self.word_embedding_size, self.lstm_hidden_size,
									num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.new_lstm_hidden_size = self.lstm_hidden_size*self.direction*self.lstm_layer

		self._init_lstm(self.question_lstm.weight_ih_l0)
		self._init_lstm(self.question_lstm.weight_hh_l0)

		##CNN
		# self.out_c = out_c
		# self.kernel_size = 3
		# self.padding_size = (self.kernel_size-1) // 2
		# self.stride = 1
		# self.conv = nn.Sequential(
		# 	nn.Conv2d(map_c,out_c,(self.kernel_size,self.kernel_size),stride=self.stride,padding=self.padding_size),
		# 	nn.ReLU(),
		# 	self.dropout)

		##Conver image dimension to lstm_hidden_size
		self.convert_d = nn.Sequential(
			nn.Linear(self.map_c,self.new_lstm_hidden_size),
			nn.Tanh())
		self.out_c = self.new_lstm_hidden_size

		##Attention layer1
		self.att1 = Attention(self.new_lstm_hidden_size,self.feature_size,self.out_c,self.map_w,self.map_h,drop)
		##Attention layer2
		self.att2 = Attention(self.new_lstm_hidden_size,self.feature_size,self.out_c,self.map_w,self.map_h,drop)
		##Attention layer3
		# self.att3 = Attention(self.new_lstm_hidden_size,self.feature_size,self.convert_c,self.map_w,self.map_h,drop)

		# self.my_att = MyAttention(self.convert_c,self.map_w,self.map_h,self.lstm_hidden_size)

		##Map to answer space
		self.linear_u = nn.Linear(self.new_lstm_hidden_size,self.answer_voc_size)
		self.log_softmax = nn.LogSoftmax()

	def forward(self,q,img,sents_lengths,param):
		batch_size,_ = q.size()
		self.batch_size = batch_size

		##Word embedding
		q_emb = self.embed(q)

		##LSTM
		# q_c_0 = self.init_hidden(param)
		# q_h_0 = self.init_hidden(param)
		pack_q = torch.nn.utils.rnn.pack_padded_sequence(q_emb, list(sents_lengths.data.type(torch.LongTensor)), batch_first=True)
		self.question_lstm.flatten_parameters()
		# q_h_n, (q_h_t,q_c_t) = self.question_lstm(pack_q,(q_h_0,q_c_0))
		q_h_n, (q_h_t,q_c_t) = self.question_lstm(pack_q)
		vq = q_h_t.permute(1,0,2).contiguous().view(self.batch_size,self.new_lstm_hidden_size)

		##CNN
		# img = self.conv(img)
		# img = img.permute(0,2,3,1).contiguous().view(batch_size,self.map_h*self.map_h,self.out_c)
		# print(img.size())

		img = img.permute(0,2,3,1).contiguous().view(batch_size,self.map_h*self.map_h,self.map_c)

		##Convert image dimension to new_lstm_hidden_size
		img = self.convert_d(img)


		##Attention
		u = self.dropout(vq)
		vi_tilde = self.att1(img,u)
		u = vi_tilde+u

		u = self.dropout(u)
		vi_tilde = self.att2(img,u)
		u = vi_tilde+u
		
		# vi_tilde = self.att3(img,u)
		# u = vi_tilde+u

		##MyAttention
		# u = self.my_att(img,vq)

		##Generate answer
		# print(u.size())
		u = self.dropout(u)
		ans_prob = self.log_softmax(self.linear_u(u))
		return ans_prob

	def init_hidden(self,param):
		return autograd.Variable(torch.rand(self.lstm_layer*self.direction,self.batch_size,self.lstm_hidden_size).cuda(async=True),**param)

	def _init_lstm(self, weight):
		for w in weight.chunk(4, 0):
			init.xavier_uniform(w)

class Attention(nn.Module):
	def __init__(self,lstm_hidden_size,feature_size,out_c,map_w,map_h,drop):
		super(Attention,self).__init__()
		self.dropout = nn.Dropout(drop)

		self.batch_size = 0
		self.lstm_hidden_size = lstm_hidden_size
		self.feature_size = feature_size
		self.out_c = out_c
		self.map_w = map_w
		self.map_h = map_h
		self.img_space = self.map_h*self.map_w

		self.linear_q = nn.Linear(self.lstm_hidden_size,self.feature_size)
		self.linear_uniform_init(self.linear_q)

		self.linear_i = nn.Linear(self.out_c,self.feature_size)
		self.linear_uniform_init(self.linear_i)

		self.tanh = nn.Tanh()

		self.linear_h = nn.Linear(self.feature_size,1)
		self.linear_uniform_init(self.linear_h)

		self.softmax = nn.Softmax()

	def forward(self,vi,vq):
		self.batch_size,_ = vq.size()
		# vi = vi.view(self.batch_size*self.img_space,self.out_c)

		ha = self.tanh(self.linear_i(vi).view(self.batch_size,self.img_space,self.feature_size)+
			self.linear_q(vq).unsqueeze(1).expand(self.batch_size,self.img_space,self.feature_size))
		ha = self.dropout(ha)

		ha = ha.view(-1,self.feature_size)
		pi = self.softmax(self.linear_h(ha).view(self.batch_size,self.img_space)).unsqueeze(1)
		vi_tilde = torch.bmm(pi,vi.view(self.batch_size,self.img_space,self.out_c)).squeeze()

		return vi_tilde
	def linear_uniform_init(self,layer):
		init.uniform(layer.weight.data,a=-0.01,b=0.01)
		init.uniform(layer.bias.data,a=-0.01,b=0.01)

class MyAttention(nn.Module):
	def __init__(self,map_c,map_w,map_h,lstm_hidden_size):
		super(MyAttention,self).__init__()

		self.batch_size = 0
		self.map_c = map_c
		self.map_w = map_w
		self.map_h = map_h
		self.img_space = self.map_h*self.map_w
		self.lstm_hidden_size = lstm_hidden_size
		self.linear = nn.Linear(self.map_c,self.lstm_hidden_size)

		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		# self.sigmoid = nn.Sigmoid()
		# self.tanh = nn.Tanh()

	def forward(self,vi,vq):
		self.batch_size,_ = vq.size()
		vi_c = self.linear(self.relu(vi))
		weight = torch.bmm(self.relu(vi_c),vq.unsqueeze(2)).view(self.batch_size,self.img_space)
		weight = self.softmax(weight)
		# weight = self.sigmoid(weight)
		# weight = self.tanh(weight)
		v_att = torch.bmm(weight.unsqueeze(1),vi).squeeze()
		return v_att


