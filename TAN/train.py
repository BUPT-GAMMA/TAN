import os
import random
import numpy as np
import torch
import logging
import tqdm
import torch.optim as optim
from tqdm import tqdm
from model import TAN
from Optim import ScheduledOptim
from metric import portfolio
from Dataloader import DataLoader
import math
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def get_performance(cascade_crit, regular_crit, pred_cascade, regular_outputs, gold_cascade, smoothing=False, num_class=None):  
    
    gold_cascade = gold_cascade.contiguous().view(-1)
    inte_sta = np.array([0]*opt.num_heads)
    pred_cascade = pred_cascade.view(gold_cascade.size(0),opt.num_heads,opt.user_size).max(1)[0]
    cascade_loss = cascade_crit(pred_cascade, gold_cascade)
    gold_cate = torch.from_numpy(np.array(range(opt.num_heads))).unsqueeze(-1).repeat(1,opt.user_size).view(-1).to(opt.device)
    regular_loss = regular_crit(regular_outputs,gold_cate)
    #cascade prediction
    pred_cascade = pred_cascade.max(1)[1]
    gold_cascade = gold_cascade.contiguous().view(-1)
    n_cascade_correct = pred_cascade.data.eq(gold_cascade.data)
    n_cascade_correct = n_cascade_correct.masked_select((gold_cascade.ne(PAD)*gold_cascade.ne(EOS)).data).sum().float()
    return cascade_loss, regular_loss, n_cascade_correct,inte_sta

def train_epoch(model, training_data, cascade_crit, regular_crit, optimizer):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_len = 0
    multi_inte = np.array([0]*opt.num_heads)
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        batch_len+=1
        # prepare data
        gold_cascade = batch[0][:, 1:]
        n_words = (gold_cascade.ne(PAD)*gold_cascade.ne(EOS)).data.sum().float()
        n_total_words += n_words
        # forward
        optimizer.zero_grad()
        pred_cascade, regular_outputs= model(batch)

        # backward
        loss_cascade, loss_regular, n_node_correct, inte_sta = get_performance(cascade_crit, regular_crit, pred_cascade, regular_outputs, gold_cascade)
        loss = loss_cascade+loss_regular
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()
        # note keeping
        n_total_correct += n_node_correct
        multi_inte+=inte_sta
        total_loss += loss.item()
        torch.cuda.empty_cache() 
    return total_loss/n_total_words, n_total_correct/n_total_words

def eval_epoch(model, validation_data, cascade_crit, regular_crit):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss = 0
    n_total_words = 0
    n_total_correct = 0
    batch_len = 0

    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):
        batch_len+=1
        gold_cascade = batch[0][:, 1:]
        pred_cascade, regular_outputs= model(batch)
        loss_cascade, loss_regular, n_node_correct, inte_sta = get_performance(cascade_crit, regular_crit, pred_cascade, regular_outputs, gold_cascade)
        loss = loss_cascade+loss_regular
        n_words = (gold_cascade.ne(PAD)*gold_cascade.ne(EOS)).data.sum().float()
        n_total_words += n_words
        n_total_correct += n_node_correct
        #print(loss)
        total_loss += loss.item()
        torch.cuda.empty_cache() 
    return total_loss/n_total_words, n_total_correct/n_total_words

def test_epoch(model, test_data, k_list=[1,10,50,100]):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    scores = {}
    scores['MRR'] = 0
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
    n_total_words = 0
    batch_len = 0
    for batch in tqdm(
            test_data, mininterval=2,
            desc='  - (Test) ', leave=False):
        batch_len+=1
        # prepare data
        gold_cascade = batch[0][:, 1:]
        # forward
        pred_cascade, regular_outputs = model(batch)
        user_num,user_size = pred_cascade.size(0), int(pred_cascade.size(1)/opt.num_heads)
        pred_seq = pred_cascade.view(user_num,opt.num_heads,user_size).max(1)[0]
        scores_batch, scores_len = portfolio(pred_seq.detach().cpu().numpy(), gold_cascade.contiguous().view(-1).detach().cpu().numpy(),\
                                             k_list)
        torch.cuda.empty_cache() 
        n_total_words += scores_len
        scores['MRR'] += scores_batch['MRR'] * scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len      
    scores['MRR'] = scores['MRR'] / n_total_words
    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words
    return scores

def train(model, training_data, validation_data, test_data, cascade_crit, regular_crit, optimizer):
    log_train_file = None
    log_valid_file = None
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,node_accuracy\n')
            log_vf.write('epoch,loss,ppl,node_accuracy\n')
    valid_accus = []

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_node_accu = train_epoch(model, training_data, cascade_crit, regular_crit, optimizer)
        print('  - (Training)   loss: {ppl: 3.5f}, node accuracy: {n_accu:3.3f} %, '\
              ' elapse: {elapse:3.3f} min'.format(
                  ppl=train_loss, n_accu=100*train_node_accu, elapse=(time.time()-start)/60))
        # validation
        start = time.time()
        valid_loss, valid_node_accu = eval_epoch(model, validation_data, cascade_crit, regular_crit)
        print('  - (Validation) loss: {ppl: 3.5f}, node accuracy: {n_accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=valid_loss, n_accu=100*valid_node_accu, elapse=(time.time()-start)/60))
        valid_accus += [valid_node_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if epoch_i % 5==0:
            # test
            scores = test_epoch(model, test_data)
            print('  - (Test) ')
            for metric in scores.keys():
                print(metric+' '+str(scores[metric]))

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_node_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_node_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')


        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,ppl=math.exp(min(train_loss, 100)), 
                    accu=100*train_node_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss, ppl=math.exp(min(valid_loss, 100)), 
                    accu=100*valid_node_accu))

class Option(object):
    def __init__(self):
        self.epoch = 100
        self.batch_size = 32
        self.d_model = 160
        self.d_inner_hid = 160
        self.n_warmup_steps = 500
        self.dropout = 0.1
        self.max_len = 501
        self.num_heads = 5
        self.doc_size = 768
        self.Temperature = 3
        self.doc = False
        self.sub_dim = self.d_inner_hid//self.num_heads 
        self.num_blocks = 3
        self.d_user_vec = self.d_model
        self.log = 'temp/tan'
        self.save_model = 'temp/tan'
        self.save_mode = 'best'
        self.device = 'cuda'
        self.user_size = 0
        self.time_unit = 64
        self.tupe = True
        self.decay = True
        self.relative = None
        self.doc = False

opt = Option()
torch.set_num_threads(4)
train_data = DataLoader(data=0, load_dict=False, batch_size=opt.batch_size)
valid_data = DataLoader(data=1,  batch_size=opt.batch_size)
test_data = DataLoader(data=2,  batch_size=opt.batch_size)
opt.user_size = train_data.user_size

Model = TAN(opt)

optimizer = ScheduledOptim(
    optim.Adam(
        Model.parameters(),
        betas=(0.9, 0.98), eps=1e-09),
    64, opt.n_warmup_steps)

def get_cascade_criterion(user_size):
    ''' With PAD token zero weight '''
    weight = torch.ones(user_size)
    weight[PAD] = 0
    weight[EOS] = 0
    return nn.CrossEntropyLoss(weight, size_average=False)

def get_regular_citerion(ninte):
    weight = torch.ones(ninte)
    return nn.CrossEntropyLoss(weight, size_average=False)

cascade_crit = get_cascade_criterion(train_data.user_size)
regular_crit = get_regular_citerion(opt.num_heads)

if opt.device=='cuda':
    Model = Model.cuda()
    cascade_crit = cascade_crit.cuda()
    regular_crit = regular_crit.cuda()

train(Model, train_data, valid_data, test_data, cascade_crit, regular_crit, optimizer)
