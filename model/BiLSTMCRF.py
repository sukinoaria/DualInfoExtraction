# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:59:26
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .crf import CRF

class BiLSTMCRF(nn.Module):
    def __init__(self,args,input_size, tagset_size,name,):
        super(BiLSTMCRF, self).__init__()
        print("build BiLSTM-CRF block...%s" % name)
        self.name = name
        self.gpu = args.gpu
        self.use_char = args.use_char

        self.droplstm = nn.Dropout(args.dropout)
        self.bilstm_flag = args.bilstm
        self.lstm_layer = args.lstm_layer
        self.use_crf = args.use_crf
        self.average_batch = args.average_batch_loss

        self.input_size = input_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = args.hidden_dim // 2
        else:
            lstm_hidden = args.hidden_dim

        self.tagset_size = tagset_size

        self.word_extractor = args.word_extractor
        if self.word_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        ## add two more label for downlayer lstm, use original label size for CRF
        self.hidden2tag = nn.Linear(args.hidden_dim, self.tagset_size+2)

        if self.use_crf:
            self.crf = CRF(self.tagset_size, self.gpu)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.lstm = self.lstm.cuda()

    def calculate_loss(self,word_represent, word_seq_lengths,batch_label,mask):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        batch_size = word_represent.size(0)
        seq_len = self.input_size
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, seq_len, hidden_size)
        feature_out = self.droplstm(lstm_out.transpose(1,0))
        ## feature_out (batch_size, seq_len, hidden_size)
        outs = self.hidden2tag(feature_out)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if self.average_batch:
            total_loss = total_loss / batch_size

        return outs,total_loss,tag_seq


    def forward(self,word_represent, word_seq_lengths,mask):
        batch_size = word_represent.size(0)
        seq_len = self.input_size

        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, seq_len, hidden_size)
        feature_out = self.droplstm(lstm_out.transpose(1, 0))
        ## feature_out (batch_size, seq_len, hidden_size)
        outs = self.hidden2tag(feature_out)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return outs,tag_seq

    def sentence_representation(self, word_represent, word_seq_lengths):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, ), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_represent.size(0)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        ## lstm_out (seq_len, seq_len, hidden_size)
        ## feature_out (batch_size, hidden_size)
        feature_out = hidden[0].transpose(1,0).contiguous().view(batch_size,-1)

        feature_list = [feature_out]
        final_feature = torch.cat(feature_list, 1)
        outputs = self.hidden2tag(self.droplstm(final_feature))
        ## outputs: (batch_size, label_alphabet_size)
        return outputs