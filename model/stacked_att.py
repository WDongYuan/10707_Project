import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.init import kaiming_uniform

class StackAttNetwork(nn.Module):
	def __init__(self,voc_size,word_embedding_size,map_c,out_c,map_w,map_h,
		answer_voc_size,lstm_hidden_size,feature_size):
		
		super(StackAttNetwork, self).__init__()

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

		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = False
		self.direction = 2 if self.bidirectional_flag else 1
		self.lstm_hidden_size = lstm_hidden_size
		self.question_lstm = nn.LSTM(self.word_embedding_size, self.lstm_hidden_size,
									num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.new_lstm_hidden_size = self.lstm_hidden_size*self.direction*self.lstm_layer

		##CNN
		self.out_c = out_c
		self.kernel_size = 3
		self.padding_size = (self.kernel_size-1) // 2
		self.stride = 1
		self.conv = nn.Sequential(
			nn.Conv2d(map_c,out_c,(self.kernel_size,self.kernel_size),stride=self.stride,padding=self.padding_size),
			nn.ReLU())

		##Conver image dimension to lstm_hidden_size
		self.convert_d = nn.Sequential(
			nn.Linear(out_c,self.new_lstm_hidden_size),
			nn.Tanh())
		self.convert_c = self.new_lstm_hidden_size

		##Attention layer1
		self.att1 = Attention(self.new_lstm_hidden_size,self.feature_size,self.convert_c,self.map_w,self.map_h)
		##Attention layer2
		self.att2 = Attention(self.new_lstm_hidden_size,self.feature_size,self.convert_c,self.map_w,self.map_h)

		##Map to answer space
		self.linear_u = nn.Linear(self.new_lstm_hidden_size,self.answer_voc_size)
		self.softmax = nn.Softmax()

	def forward(self,q,img,sents_lengths,param):
		batch_size,_ = q.size()
		self.batch_size = batch_size

		##Word embedding
		q_emb = self.embed(q)

		##LSTM
		q_c_0 = self.init_hidden(param)
		q_h_0 = self.init_hidden(param)
		pack_q = torch.nn.utils.rnn.pack_padded_sequence(q_emb, list(sents_lengths.data.type(torch.LongTensor)), batch_first=True)
		self.question_lstm.flatten_parameters()
		q_h_n, (q_h_t,q_c_t) = self.question_lstm(pack_q,(q_h_0,q_c_0))
		vq = q_h_t.permute(1,0,2).contiguous().view(self.batch_size,self.new_lstm_hidden_size)

		##CNN
		img = self.conv(img)
		img = img.permute(0,2,3,1).view(batch_size,self.map_h*self.map_h,self.out_c)
		# print(img.size())

		##Convert image dimension to new_lstm_hidden_size
		img = self.convert_d(img)

		##Attention
		u = vq
		vi_tilde = self.att_1(img,u)
		u = vi_tilde+u
		vi_tilde = self.att_2(img,u)
		u = vi_tilde+u

		##Generate answer
		ans_prob = self.softmax(self.linear_u(u))
		return ans_prob

	def init_hidden(self,param):
		return autograd.Variable(torch.rand(self.lstm_layer*self.direction,self.batch_size,self.lstm_hidden_size).cuda(async=True),**param)

class Attention(nn.Module):
	def __init__(self,lstm_hidden_size,feature_size,convert_c,map_w,map_h):
		super(Attention,self).__init__()
		self.batch_size = 0
		self.lstm_hidden_size = lstm_hidden_size
		self.feature_size = feature_size
		self.convert_c = convert_c
		self.map_w = map_w
		self.map_h = map_h
		self.img_space = self.map_h*self.map_w

		self.linear_q = nn.Linear(self.lstm_hidden_size,self.feature_size)
		self.linear_i = nn.Linear(self.convert_c,self.feature_size,bias=False)
		self.tanh = nn.Tanh()
		self.linear_h = nn.Linear(self.feature_size,1)
		self.softmax = nn.Softmax()

	def forward(vi,vq):
		self.batch_size,_ = vq.size()
		vi = vi.view(self.batch_size*self.img_space,self.convert_c)
		ha = self.tanh(self.linear_i(vi).view(self.batch_size,self.img_space,self.feature_size)+
			self.linear_q(vq).unsqueeze(1).expand(self.batch_size,self.img_space,self.feature_size))
		ha = ha.view(-1,self.feature_size)
		pi = self.softmax(self.linear_h(ha).view(self.batch_size,1,self.img_space))
		vi_tilde = torch.bmm(pi,vi.view(self.batch_size,self.img_space,self.convert_c)).squeeze()

		return vi_tilde


