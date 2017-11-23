# from model import baseline
# from model import relational_network_model
from model import stacked_att
import sys
sys.path.append('./util')
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
import numpy as np
########################################
import torch.backends.cudnn as cudnn
import data
import utils
from tqdm import *
########################################

def Validation(model,val,val_params):
    val_accs = []
    model.eval()
    rate = 0.1
    for v,q,a,item,q_len in val:
        # if np.random.random()>rate:
        #     continue
        q = Variable(q.cuda(async=True),**val_params)
        a = Variable(a.cuda(async=True),**val_params)
        v = Variable(v.cuda(async=True),**val_params)
        q_len = Variable(q_len.cuda(async=True), **val_params)
        o = model(q,v,q_len,val_params)
        acc = utils.batch_accuracy(o.data,a.data).cpu()
        val_accs.append(acc.view(-1))
    val_acc=torch.cat(val_accs,dim=0).mean()
    print("validation accuracy: "+ str(val_acc))

def save_model(state, filename='saved_model.out'):
    torch.save(state, filename)

if __name__=="__main__":
    if len(sys.argv)==1:
        print("python training.py train 0.001 not_load")
        print("python training.py validate 0.001 load my_best_model_15.model")
        exit()

    train = True
    if sys.argv[1] == "validate":
        train = False

    # acc_record_file = open("./acc_record_file","w+")
    load_model = True if not train or sys.argv[3]=="load" else False
    load_path = None
    if load_model:
        load_path = sys.argv[4]

    #########################################################################
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    #########################################################################
    print("Loading data...")
    #########################################################################
    # training,train_dict_size = data.get_loader(train=True,full_batch = False)
    # print("train_dict_size: "+str(train_dict_size))
    # val,val_dict_size = data.get_loader(val=True,full_batch= False)
    #########################################################################
    #########################################################################
    # training,train_dict_size = None,17000
    # val,val_dict_size = None,0
    #########################################################################
    training,train_dict_size = data.get_loader(val=True,full_batch = False)
    val,val_dict_size = training,train_dict_size
    #########################################################################
    print("Finish loading data!")
    #########################################################################
    # model = relational_network_model.RelationalNetwork(train_dict_size,config.word_embed_dim,config.output_features,config.rn_conv_channel,
    #         config.output_size,config.output_size,config.max_answers,config.lstm_hidden_size,config.g_mlp_hidden_size,config.relation_length)
    #########################################################################
    model = None
    if train and not load_model:
        feature_size = 512
        model = stacked_att.StackAttNetwork(train_dict_size,config.word_embed_dim,config.output_features,
                config.output_size,config.output_size,config.max_answers,config.lstm_hidden_size,feature_size,config.drop)
        # print(len(list(model.parameters())))
        # print([p.size() for p in list(model.parameters())])
        # print(model)
        # exit()
    elif train and load_model:
        print("Loading model...")
        model = torch.load(load_path)
    elif not train and load_model:
        print("Loading model...")
        model = torch.load(load_path)
        print("Begin validation")
        val_params = {
            'requires_grad': False,
            'volatile': True
        }
        Validation(model,val,val_params)
        exit()
    #########################################################################
    lr = float(sys.argv[2])

    ##Set learning rate for embedding layer
    param = []
    param_l = list(model.parameters())
    param.append({'params': param_l[0], 'lr': lr})
    for i in range(1,len(param_l)):
        param.append({'params': param_l[i],'lr': lr})
    optimizer = optim.Adam(param,lr = lr,weight_decay=0.0005)
    # optimizer = optim.Adam(model.parameters(),lr = lr)
    # optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9)

    best_perf = 0.0
    # if int(sys.argv[1]) == 2:    
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    #     model = nn.parallel.DataParallel(model,[0,1]).cuda()
    # elif int(sys.argv[1]) == 1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     model.cuda()

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
    print("dropout = "+str(config.drop))
    print("decay step %s, size %s" %(str(config.decay_step),str(config.decay_size)))

    # Validation(model,val,val_params,best_perf)
    # exit()

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
            torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
            # for p in model.parameters():
            #     print(torch.mean(torch.abs(p.grad.data)))
            optimizer.step()
            batch_loss += loss.data[0]
            acc = utils.batch_accuracy(o.data,a.data).cpu()
            train_accs.append(acc.view(-1))
            tmp_acc.append(acc.view(-1))
            sample_counter += config.batch_size
            if sample_counter%5000==0:
                print((round(torch.cat(tmp_acc,dim=0).mean(),4),round(loss.data[0],4))),
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
        # acc_record_file.write("train: "+str(batch_loss/config.batch_size)+" "+str(train_acc)+"\n")
        if (i+1)%1 ==0:
            print("Saving model...")
            torch.save(model,"./my_best_model_"+str(i+1)+".model")
        #     best_perf = Validation(model,val,val_params,best_perf,i)
    acc_record_file.close()
    print("best performance %s" %str(best_perf))
        
    


            
