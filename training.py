from model import baseline
from model import relational_network_model
from model import stacked_att
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
import config
import torch.nn as nn
from datetime import datetime
import time

def save_model(state, filename='saved_model.out'):
    torch.save(state, filename)

if __name__=="__main__":
    load_model = False

    train = True
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print("Loading data...")
    #########################################################################
    training,train_dict_size = data.get_loader(train=True,full_batch = False)
    val,val_dict_size = data.get_loader(val=True,full_batch= False)
    #########################################################################
    # training,train_dict_size = data.get_loader(val=True,full_batch = False)
    # val,val_dict_size = training,train_dict_size
    #########################################################################
    print("Finish loading data!")
    #########################################################################
    # model = relational_network_model.RelationalNetwork(train_dict_size,config.word_embed_dim,config.output_features,config.rn_conv_channel,
    #         config.output_size,config.output_size,config.max_answers,config.lstm_hidden_size,config.g_mlp_hidden_size,config.relation_length)
    #########################################################################
    model = None
    if not load_model:
        feature_size = 500
        model = stacked_att.StackAttNetwork(train_dict_size,config.word_embed_dim,config.output_features,config.rn_conv_channel,
                config.output_size,config.output_size,config.max_answers,config.lstm_hidden_size,feature_size)
    else:
        print("Loading model...")
        model = torch.load("./best_model.model")
    #########################################################################
    lr = float(sys.argv[2])
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],lr = lr)

    best_perf = 0.0
    if int(sys.argv[1]) == 2:    
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        model = nn.parallel.DataParallel(model,[0,1]).cuda()
    elif int(sys.argv[1]) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model.cuda()
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
    print("lr"+str(lr))
    print("decay step %s, size %s" %(str(config.decay_step),str(config.decay_size)))
    for i in tqdm(range(config.epochs)):
        lr_scheduler.step()
        batch_loss = 0
        train_accs = []
        print(datetime.now())
        model.train()
        start_time = time.time()
        sample_counter = 0
        tmp_acc = []
        for v,q,a,item,q_len in training:
            q = Variable(q.cuda(async=True),**var_params)
            a = Variable(a.cuda(async=True),**var_params)
            v = Variable(v.cuda(async=True),**var_params)
            q_len = Variable(q_len.cuda(async=True), **var_params)
            o = model(q,v,q_len,var_params)
            optimizer.zero_grad()
            loss =(-o*(a/10)).sum(dim=1).mean() # F.nll_loss(o,a)
            loss.backward()
            optimizer.step()
            batch_loss += loss.data[0]
            acc = utils.batch_accuracy(o.data,a.data).cpu()
            train_accs.append(acc.view(-1))
            tmp_acc.append(acc.view(-1))
            sample_counter += config.batch_size
            if sample_counter%5000==0:
                print((round(torch.cat(tmp_acc,dim=0).mean(),4),round(loss.data[0]))),
                # print("."),
                tmp_acc = []
            if sample_counter%100000==0:
                print("")
                print(str(sample_counter)+" samples.")
                print("Time: "+str(time.time()-start_time))
                print("############################################")
        train_acc= torch.cat(train_accs,dim=0).mean()
        print("")
        print("epoch %s, loss %s, accuracy %s" %(str(i),str(batch_loss/config.batch_size),str(train_acc)))
        if (i+1)%config.val_interval ==0:
            print("")
            val_accs = []
            model.eval()
            for v,q,a,item,q_len in val:
                q = Variable(q.cuda(async=True),**val_params)
                a = Variable(a.cuda(async=True),**val_params)
                v = Variable(v.cuda(async=True),**val_params)
                q_len = Variable(q_len.cuda(async=True), **val_params)
                o = model(q,v,q_len,val_params)
                acc = utils.batch_accuracy(o.data,a.data).cpu()
                val_accs.append(acc.view(-1))
            val_acc=torch.cat(val_accs,dim=0).mean()
            print("epoch %s, validation accuracy %s" %(str(i),str(val_acc)))
            if val_acc > best_perf:
                best_perf = val_acc
                # torch.save(model,"./best_model.model")
                save_model({'model': model.state_dict(),
                    'optimizer':optimizer.state_dict()},
                    "./my_best_model.model")
    print("best performance %s" %str(best_perf))
        
    


            
