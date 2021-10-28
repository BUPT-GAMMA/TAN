import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.in_proj_weight = Parameter(torch.empty(self.num_heads, self.head_dim, 2 *self.head_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(self.num_heads, 2 *self.head_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.num_heads):
            for j in range(2):
                init.xavier_uniform_(self.in_proj_weight[i,:, (self.head_dim * j):(self.head_dim * (j+1))])

        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, pos_weight = None, decay_weight = None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """

        tgt_len, bsz, heads, dim = query.size()
        assert heads*dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, heads, dim]
        assert key.size() == value.size()

        if pos_weight is not None:
            self.scaling = (2*self.head_dim) ** -0.5

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None
        v = query
        q, k = self._in_proj_qkv(query)
        q = q*self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if pos_weight is not None:
            pos_weight = self.scaling*pos_weight
            attn_output_weights += pos_weight

        if pos_weight is not None:
            attn_output_weights += decay_weight

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights.float(), dim=-1,
            dtype=torch.float32 if attn_output_weights.dtype == torch.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.num_heads,self.head_dim)
        #attn_output = self.out_proj(attn_output)

        return attn_output


    def _in_proj_qkv(self, input):
        tgt_len, bsz, heads, subdim = input.size()
        input = input.contiguous().view(tgt_len*bsz,heads,subdim).transpose(0,1)
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        output = torch.matmul(input,weight).squeeze() + bias.unsqueeze(1)
        assert list(output.size()) == [heads, tgt_len*bsz, 2*subdim] 
        output = output.transpose(0,1).view(tgt_len,bsz,heads,2*subdim)
        return output.chunk(2, dim=-1)

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate, group=1):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1, groups = group)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1, groups = group)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        
    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1,1,seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    #print(previous_mask)
    #print(seqs)
    masked_seq = previous_mask * seqs.data.float()
    #print(masked_seq.size())

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq,PAD_tmp],dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2,masked_seq.long(),float('-inf'))
    #print(masked_seq.requires_grad)
    return masked_seq

class TAN(nn.Module):
    def __init__(self, opt, dropout=0.1):
        super(TAN, self).__init__()

        self.user_size = opt.user_size
        self.ninp = opt.d_user_vec
        self.nhid = opt.d_inner_hid
        self.num_heads = opt.num_heads
        self.dev = opt.device
        self.relative = opt.relative
        self.tupe = opt.tupe
        self.decay = opt.decay
        self.T = opt.Temperature
        self.doc = opt.doc
        self.head_dim = self.nhid//self.num_heads
        self.user_emb = nn.Embedding(self.user_size, self.ninp, padding_idx=PAD)
        self.pos_emb = nn.Embedding(opt.max_len, self.head_dim) 
        self.emb_dropout = nn.Dropout(dropout)
        if self.doc is True:
            self.doc_dim = self.head_dim
            self.doc_proj = nn.Linear(opt.doc_size,self.doc_dim)

        if self.decay is True:
            self.time_emb = nn.Embedding(opt.time_unit, self.num_heads)

        if self.relative is not None:
            self.in_proj_bias = Parameter(torch.empty(self.num_heads, 2*opt.relative+1))

        self.prototype = nn.Embedding(self.num_heads, self.head_dim)
        #self.last_layer = PointWiseFeedForward(self.nhid, dropout,self.num_heads)
        self.last_layers = nn.ModuleList()
        
        if self.tupe is True:
            self.pos_proj_weight = Parameter(torch.empty(self.num_heads, self.head_dim, 2 *self.head_dim))
        if self.relative is not None:
            self.in_proj_bias = Parameter(torch.empty(self.num_heads, 2*opt.relative+1))
 
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        #self.forward_layers = nn.ModuleList()
        self.cosine = nn.CosineSimilarity(dim=-1,eps=1e-6)
        for _ in range(opt.num_blocks):
            new_attn_layernorm = nn.LayerNorm(self.head_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = MultiheadAttention(self.nhid, self.num_heads,
                                                        dropout)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = nn.LayerNorm(self.head_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
        for i in range(self.num_heads):
            for j in range(2):
                init.xavier_uniform_(self.pos_proj_weight[i,:, (self.head_dim * j):(self.head_dim * (j+1))])
        
        if self.relative is not None:
            init.constant_(self.in_proj_bias, 0.)

        for _ in range(self.num_heads):
            new_fwd_layer = PointWiseFeedForward(self.head_dim, dropout)
            self.last_layers.append(new_fwd_layer)

    def proj_qk(self, input):
        tgt_len, bsz, heads, subdim = input.size()
        input = input.contiguous().view(tgt_len*bsz,heads,subdim).transpose(0,1)
        weight = self.pos_proj_weight
        output = torch.matmul(input,weight).squeeze()
        assert list(output.size()) == [heads, tgt_len*bsz, 2*subdim] 
        output = output.transpose(0,1).view(tgt_len,bsz,heads,2*subdim)
        return output.chunk(2, dim=-1)

    def log2feats(self, log_seqs, log_intervals, input_doc = False):
        batch_size = log_seqs.size(0)
        max_len = log_seqs.size(1)
        if self.decay is True:
            time_decay = self.time_emb(log_intervals).transpose(1, 2).unsqueeze(-2)
        if self.doc is True:
            doc_emb = self.doc_proj(input_doc)#batch_size*head_dim
            seqs = self.user_emb(log_seqs).view(batch_size,max_len,self.num_heads,self.head_dim)
            weight = F.softmax(self.cosine(doc_emb.unsqueeze(1).unsqueeze(1),seqs)/self.T, dim=-1)#batch_size*max_len*num_heads
            assert list(weight.size()) == [batch_size,max_len,self.num_heads]
            seqs = seqs*(weight.unsqueeze(-1))
        else:
            seqs = self.user_emb(log_seqs).view(batch_size,max_len,self.num_heads,self.head_dim)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        pos_attn = None
        #print(self.pos_emb(torch.LongTensor(positions).to(self.dev)).size())
        if self.tupe is False:
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev)).unsqueeze(-2)
        else:
            positions = self.pos_emb(torch.LongTensor(positions).to(self.dev)).unsqueeze(-2).expand(-1,-1,self.num_heads,-1)
            positions = torch.transpose(positions, 0, 1)
            Q_pos,K_pos = self.proj_qk(positions)
            Q_pos = Q_pos.contiguous().view(max_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            K_pos = K_pos.contiguous().view(max_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            pos_attn = torch.bmm(Q_pos, K_pos.transpose(1, 2))
        seqs = self.emb_dropout(seqs)
        #timeline_mask = torch.cuda.BoolTensor(log_seqs == PAD)
        timeline_mask = (log_seqs == PAD).cpu().numpy()
        timeline_mask = torch.ByteTensor(timeline_mask).to(self.dev)
        tl = seqs.size(1)
        attention_mask = np.triu(np.ones((tl, tl))*float('-inf'), k=1).astype('float32')
        attention_mask = torch.from_numpy(attention_mask).to(self.dev)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            if i == len(self.attention_layers)-1 and self.decay is True:
                mha_outputs = self.attention_layers[i](Q, seqs, seqs,  attn_mask=attention_mask,
                                            key_padding_mask=timeline_mask, pos_weight=pos_attn, decay_weight = time_decay)
            else:
                mha_outputs = self.attention_layers[i](Q, seqs, seqs,  attn_mask=attention_mask,
                                            key_padding_mask=timeline_mask, pos_weight=pos_attn)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            #print(seqs.size())
            seqs = self.forward_layernorms[i](seqs)
        log_feats = torch.zeros(batch_size,max_len,self.num_heads,self.head_dim).to(self.dev)
        for i in range(self.num_heads):
            log_feats[:,:,i,:] = self.last_layers[i](seqs[:,:,i,:].view(batch_size,max_len,self.head_dim))
        return log_feats

    def forward(self, input, generate=False): # for training        
        input_cascade = input[0]
        input_interval = input[1]
        input_len = input[2] 
        input_doc = input[3]
        if not generate:
            input_cascade = input_cascade[:,:-1]
            input_interval = input_interval[:,:-1]
        batch_size = input_cascade.size(0)
        max_len = input_cascade.size(1)
        if self.doc is True:
            log_feats = self.log2feats(input_cascade,input_interval,input_doc) 
        else:
            log_feats = self.log2feats(input_cascade,input_interval) 
        log_feats = log_feats.contiguous().view(batch_size*max_len,self.num_heads,self.head_dim).transpose(0,1)
        candi_user = self.user_emb.weight.t().view(self.num_heads,self.head_dim,-1)#
        #.view(batch_size,max_len,self.num_heads,self.head_dim)
        outputs = torch.matmul(log_feats,candi_user).view(self.num_heads,batch_size,max_len,-1)#num_heads*(max_len*batch_size)*user_size
        outputs = get_previous_user_mask(input_cascade, self.user_size) + outputs
        outputs = outputs.transpose(0,1).transpose(1,2)
        #print(outputs.size())
        regular_outputs = self.cosine(self.prototype.weight,candi_user.transpose(1,2).unsqueeze(-2))
        return outputs.contiguous().view(-1,self.num_heads*self.user_size), regular_outputs.contiguous().view(-1,self.num_heads)
