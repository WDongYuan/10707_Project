from model import baseline
from model import relational_network_model
from model import hier_glimpse
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
import torch
import torch.nn.utils
import os
import config
import torch.nn as nn
from datetime import datetime
import time

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

if __name__=="__main__":
    train = True
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print("reading data...")
    #########################################################################
    training,train_dict_size = data.get_loader(train=True,full_batch = False)
    val,val_dict_size = data.get_loader(val=True,full_batch= False)
    #########################################################################
    # training,train_dict_size = data.get_loader(val=True,full_batch = False)
    # val,val_dict_size = training,train_dict_size
    #########################################################################
    print("finish reading data!")

    model = hier_glimpse.hier_glimpse(config.glimpse_size,
                                train_dict_size,
                                config.max_answers,
                                config.word_embed_dim,
                                config.lstm_hidden_size,
                                config.image_embed_dim,
                                config.output_size,
                                config.feat_hidden_size,
                                config.out_hidden_size,
                                config.drop_out
                                )
    if len(sys.argv) == 4:
        model = torch.load(sys.argv[3])
    if int(sys.argv[1]) == 2:    
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        model = nn.parallel.DataParallel(model,[0,1]).cuda()
    elif int(sys.argv[1]) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model.cuda()
    elif int(sys.argv[1]) == 3:
        model = torch.load("model_8.model")
        print("Begin validation")
        val_params = {
            'requires_grad': False,
            'volatile': True
        }
        Validation(model,val,val_params)
        exit()

    lr = float(sys.argv[2])
    #embed_params = list(map(id, model.text.embed.parameters()))
    #base_params = filter(lambda p: id(p) not in embed_params,model.parameters())
    # optimizer = optim.Adam([
    #                         {'params':model.text.embed.parameters(),'lr': config.initial_embed_lr},
    #                         {'params':base_params}
    #                         ],lr = lr)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],lr = lr)

    best_perf = 0.0
    
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
    # print("embedding lr"+str(config.initial_embed_lr))
    print("decay step %s, size %s" %(str(config.decay_step),str(config.decay_size)))
    for i in tqdm(range(config.epochs)):
        lr_scheduler.step()
        batch_loss = 0
        train_accs = []
        print(datetime.now())
        model.train()

        sample_counter = 0
        tmp_acc = []
        start_time = time.time()

        for v,q,a,item,q_len in training:
            q = Variable(q.cuda(async=True),**var_params)
            a = Variable(a.cuda(async=True),**var_params)
            v = Variable(v.cuda(async=True),**var_params)
            q_len = Variable(q_len, **var_params)
            o = model(q,v,q_len,var_params)
            optimizer.zero_grad()
            # print(o.size())
            # print(a.size())
            loss =(-o*(a/10)).sum(dim=1).mean() # F.nll_loss(o,a)
            loss.backward()
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
        print("epoch %s, loss %s, accuracy %s" %(str(i),str(batch_loss/config.batch_size),str(train_acc)))
        torch.save(model,"./curr_model.model")
        if (i+1)%config.val_interval ==0:
            val_accs = []
            model.eval()
            for v,q,a,item,q_len in val:
                q = Variable(q.cuda(async=True),**val_params)
                a = Variable(a.cuda(async=True),**val_params)
                v = Variable(v.cuda(async=True),**val_params)
                q_len = Variable(q_len, **val_params)
                o = model(q,v,q_len,val_params)
                acc = utils.batch_accuracy(o.data,a.data).cpu()
                val_accs.append(acc.view(-1))
            val_acc=torch.cat(val_accs,dim=0).mean()
            print("epoch %s, validation accuracy %s" %(str(i),str(val_acc)))
            if val_acc > best_perf:
                best_perf = val_acc
                torch.save(model,"./best_model"+str(i)+".model")
        print("saving model...")
        torch.save(model,"./model_"+str(i)+".model")
    print("best performance %s" %str(best_perf))
        
    


            
