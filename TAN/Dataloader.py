import random
import numpy as np
import torch
import pickle
import joblib

PAD = 0
UNK = 2
BOS = 3
EOS = 1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

class DataLoader(object):
    ''' For data iteration '''
    def __init__(
            self, data=0, load_dict=False, cuda=True, batch_size=32, shuffle=True, test=False, with_EOS=True): #data = 0 for train, 1 for valid, 2 for test
        self._batch_size = batch_size
        self._u2idx = {}
        self._idx2u = []
        self.u2idx_dict = "temp/u2idx.data"
        self.idx2u_dict = "temp/idx2u.data"
        self.data = data
        self.test = test
        self.with_EOS = with_EOS
        if not load_dict:
            self._buildIndex()
            with open(self.u2idx_dict, 'wb') as handle:
                pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.idx2u_dict, 'wb') as handle:
                pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.u2idx_dict, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.idx2u_dict, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)
        self._train_data,train_len = self._readFromFile('temp/train.data')
        self._valid_data,valid_len = self._readFromFile('temp/valid.data')
        self._test_data,test_len = self._readFromFile('temp/test.data')
        self.train_size = len(self._train_data)
        self.valid_size = len(self._valid_data)
        self.test_size = len(self._test_data)
        print("training set size:%d   valid set size:%d  testing set size:%d" % (self.train_size, self.valid_size, self.test_size))
        print(self.train_size+self.valid_size+self.test_size)
        print((train_len+valid_len+test_len+0.0)/(self.train_size+self.valid_size+self.test_size))
        print(self.user_size-2)
        self.cuda = cuda
        if self.data == 0:
            self._n_batch = int(np.ceil(len(self._train_data) / batch_size))
        elif self.data == 1:
            self._n_batch = int(np.ceil(len(self._valid_data) / batch_size))
        else:
            self._n_batch = int(np.ceil(len(self._test_data) / batch_size))
        self._iter_count = 0
        self._need_shuffle = shuffle
        if self._need_shuffle:
            random.shuffle(self._train_data)

    def _buildIndex(self):
        #set index to users

        train_user_set = set()
        valid_user_set = set()
        test_user_set = set()
        with open("temp/train.data",'rb') as file:
            train_dict = pickle.load(file)
        with open("temp/test.data",'rb') as file:
            test_dict = pickle.load(file)
        with open("temp/valid.data",'rb') as file:
            valid_dict = pickle.load(file)
        for post in train_dict.keys():
            for user in train_dict[post]['seq']:
                train_user_set.add(user)

        for post in test_dict.keys():
            for user in test_dict[post]['seq']:
                test_user_set.add(user)

        for post in valid_dict.keys():
            for user in valid_dict[post]['seq']:
                valid_user_set.add(user)

        user_set = train_user_set | valid_user_set | test_user_set

        pos = 0
        self._u2idx['<blank>'] = pos
        self._idx2u.append('<blank>')
        pos += 1
        self._u2idx['</s>'] = pos
        self._idx2u.append('</s>')
        pos += 1

        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        self.user_size = len(user_set) + 2
        print("user_size : %d" % (self.user_size))

    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        total_len = 0
        t_data = []
        with open(filename,'rb') as file:
            data_dict = pickle.load(file)
        for post in data_dict.keys():
            userlist = []
            intervallist = []
            for user in data_dict[post]['seq']:
                userlist.append(self._u2idx[user])
            intervallist = data_dict[post]['decay']
            retwe_len = data_dict[post]['len']
            post_topic = data_dict[post]['pre']
            post_id = post
            if len(userlist) > 1 and len(userlist)<=500:
                total_len+=retwe_len
                if len(userlist) != len(intervallist) or len(userlist) != retwe_len:
                    print("error")
                if self.with_EOS:
                    userlist.append(EOS)
                    intervallist.append(0)
                temp_dict = {}
                temp_dict['seq'] = userlist
                temp_dict['topic'] = post_topic
                temp_dict['interval'] = intervallist
                temp_dict['len'] = retwe_len+1
                t_data.append(temp_dict)
        return t_data,total_len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch
    
    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''
            insts.sort(key=lambda x:x['len'],reverse = True)
            max_len = insts[0]['len']+1
            '''
            len_data = np.array([
                inst['len']*[1]+(max_len - inst['len'])*[0]
                for inst in insts])
            '''
            len_data = np.array([
                inst['len'] for inst in insts])
            cascade_data = np.array([
                inst['seq'] + [PAD] * (max_len - inst['len'])
                for inst in insts])
            topic_data = np.array([inst['topic'] for inst in insts])
            interval_data = np.array([
                inst['interval'] + [0] * (max_len - inst['len'])
                for inst in insts])
            cascade_data_tensor = torch.LongTensor(cascade_data)
            interval_data_tensor = torch.LongTensor(interval_data)
            topic_data_tensor = torch.FloatTensor(topic_data)
            len_data_tensor = torch.FloatTensor(len_data)
            if self.cuda:
                interval_data_tensor = interval_data_tensor.cuda()
                len_data_tensor = len_data_tensor.cuda()
                topic_data_tensor = topic_data_tensor.cuda()
                cascade_data_tensor = cascade_data_tensor.cuda()
            return cascade_data_tensor,interval_data_tensor,len_data_tensor,topic_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1
            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size
            if self.data == 0:
                seq_insts = self._train_data[start_idx:end_idx]
            elif self.data == 1:
                seq_insts = self._valid_data[start_idx:end_idx]
            else:
                seq_insts = self._test_data[start_idx:end_idx]
            seq_cascade,seq_interval,seq_len,seq_topic = pad_to_longest(seq_insts)
            return (seq_cascade,seq_interval,seq_len,seq_topic)
        else:
            if self._need_shuffle:
                random.shuffle(self._train_data)
            self._iter_count = 0
            raise StopIteration() 

