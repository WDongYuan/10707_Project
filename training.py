from model import baseline
from model import relational_network_model
import sys
sys.path.append('./util')
import data
import utils
from tqdm import *
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import pdb
import torch
import torch.nn.utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import config
import torch.nn as nn


if __name__=="__main__":
    train = True
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    training,train_dict_size = data.get_loader(train=True,full_batch = False)
    val,val_dict_size = data.get_loader(val=True,full_batch= False)

    model = relational_network_model.RelationalNetwork(train_dict_size,config.word_embed_dim,config.output_features,config.output_size,config.output_size,config.max_answers)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],lr = config.initial_lr)

    best_perf = 0.0
    model = nn.parallel.DataParallel(model,[0,1]).cuda()
    var_params = {
        'requires_grad': False,
	    'volatile': False
    }
    val_params = {
        'requires_grad': False,
        'volatile': True
    }
    lr_scheduler = scheduler.StepLR(optimizer, step_size = config.decay_step, gamma = config.decay_size)

    print("data is fully loaded")
    print("lr"+str(config.initial_lr))
    print("embedding lr"+str(config.initial_embed_lr))
    print("decay step %s, size %s" %(str(config.decay_step),str(config.decay_size)))

    for i in tqdm(range(config.epochs)):
        lr_scheduler.step()
        batch_loss = 0
        train_accs = []
        for v,q,a,item,q_len in training:
            q = Variable(q.cuda(async=True),**var_params)
            a = Variable(a.cuda(async=True),**var_params)
            v = Variable(v.cuda(async=True),**var_params)
            q_len = Variable(q_len.cuda(async=True), **var_params)
            o = model(q,v,q_len)
            optimizer.zero_grad()
            loss =(-o*(a/10)).sum(dim=1).mean() # F.nll_loss(o,a)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 20)
            optimizer.step()
            batch_loss += loss.data[0]
            acc = utils.batch_accuracy(o.data,a.data).cpu()
            train_accs.append(acc.view(-1))
        train_acc= torch.cat(train_accs,dim=0).mean()
        print("epoch %s, loss %s, accuracy %s" %(str(i),str(batch_loss/config.batch_size),str(train_acc)))
        if (i+1)%config.val_interval ==0:
            val_accs = []
            for v,q,a,item,q_len in val:
                q = Variable(q.cuda(async=True),**val_params)
                a = Variable(a.cuda(async=True),**val_params)
                v = Variable(v.cuda(async=True),**val_params)
                q_len = Variable(q_len.cuda(async=True), **val_params)
                o = model(q,v,q_len)
                acc = utils.batch_accuracy(o.data,a.data).cpu()
                val_accs.append(acc.view(-1))
            val_acc=torch.cat(val_accs,dim=0).mean()
            print("epoch %s, validation accuracy %s" %(str(i),str(val_acc)))
            if val_acc > best_perf:
                best_perf = val_acc
                torch.save(model,"./best_model.model")
    print("best performance %s" %str(best_perf))
        
    


            
