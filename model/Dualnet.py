# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-13 11:49:38

from __future__ import print_function
from __future__ import absolute_import
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wordrep import WordRep
from .BiLSTMCRF import BiLSTMCRF
from .interunit import InterUnit

class Dualnet(nn.Module):
    def __init__(self, args,data):
        super(Dualnet, self).__init__()

        ###Networks
        self.word_feature_extractor =args.word_extractor
        self.use_char = args.use_char
        self.char_feature_extractor = args.char_extractor
        self.use_crf = args.use_crf

        ## Training
        self.average_batch_loss = args.average_batch_loss
        self.status = args.status
        ### Hyperparameters

        self.cnn_layer = args.cnn_layer
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.char_hidden_dim = args.char_hidden_dim
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.lstm_layer = args.lstm_layer
        self.bilstm = args.bilstm

        self.gpu = args.gpu
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.clip = args.clip
        self.momentum = args.momentum
        self.l2 = args.l2

        #Dual Network Modules...
        self.wordrep = WordRep(args, data)

        # information interaction unit
        self.inter_unit = InterUnit(args.word_emb_dim + args.char_hidden_dim, data.hlabelset_size + 2,
                                    data.llabelset_size + 2, 1, F.relu)

        # component of Dual model
        self.H_ner = BiLSTMCRF(args,input_size=args.char_hidden_dim+args.word_emb_dim,
                              tagset_size=data.hlabelset_size,name='High layer model')
        self.B_ner = BiLSTMCRF(args, input_size=args.char_hidden_dim + args.word_emb_dim,
                              tagset_size=data.llabelset_size, name='bottom layer model')

        print("build Dual sequence labeling network...")

    def calculate_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_hlabel,batch_llabel, mask):
        word_represent = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

        H2BH_outs, H2BH_loss, H2BH_tag_seqs = self.H_ner.calculate_loss(word_represent, word_seq_lengths,
                                                                       batch_hlabel, mask)
        B2HB_outs, B2HB_loss, B2HB_tag_seqs = self.B_ner.calculate_loss(word_represent, word_seq_lengths,
                                                                       batch_llabel, mask)

        high_rep,low_rep = self.inter_unit(word_represent,H2BH_outs,B2HB_outs)

        #concat low level information to do low level tagging
        H2BB_outs, H2BB_loss, H2BB_tag_seqs = self.B_ner.calculate_loss(high_rep, word_seq_lengths,
                                                                       batch_llabel, mask)
        # concat high level information to do high level tagging
        B2HH_outs, B2HH_loss, B2HH_tag_seqs = self.H_ner.calculate_loss(low_rep, word_seq_lengths,
                                                                       batch_hlabel, mask)

        return H2BH_loss,H2BB_loss,B2HB_loss,B2HH_loss,H2BH_tag_seqs,H2BB_tag_seqs,B2HB_tag_seqs,B2HH_tag_seqs

    #todo for a while...
    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        word_represent = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

        H2BH_outs,H2BH_tag_seqs = self.H_ner(word_represent, word_seq_lengths, mask)
        B2HB_outs,B2HB_tag_seqs = self.B_ner(word_represent, word_seq_lengths, mask)

        high_rep, low_rep = self.inter_unit(word_represent, H2BH_outs, B2HB_outs)

        # concat low level information to do low level tagging
        _,H2BB_tag_seqs = self.B_ner(high_rep, word_seq_lengths, mask)
        # concat high level information to do high level tagging
        _,B2HH_tag_seqs = self.H_ner(low_rep, word_seq_lengths, mask)

        return H2BH_tag_seqs, H2BB_tag_seqs, B2HB_tag_seqs, B2HH_tag_seqs


    def show_model_summary(self,logger):
        logger.info(" " + "++" * 20)
        logger.info(" Model Network:")
        logger.info("     Model        use_crf: %s" % (self.use_crf))
        logger.info("     Model word extractor: %s" % (self.word_feature_extractor))
        logger.info("     Model       use_char: %s" % (self.use_char))
        if self.use_char:
            logger.info("     Model char extractor: %s" % (self.char_feature_extractor))
            logger.info("     Model char_hidden_dim: %s" % (self.char_hidden_dim))
        logger.info(" " + "++" * 20)
        logger.info(" Training:")
        logger.info("     Optimizer: %s" % (self.optimizer))
        logger.info("     Iteration: %s" % (self.iteration))
        logger.info("     BatchSize: %s" % (self.batch_size))
        logger.info("     Average  batch   loss: %s" % (self.average_batch_loss))

        logger.info(" " + "++" * 20)
        logger.info(" Hyperparameters:")

        logger.info("     Hyper              lr: %s" % (self.lr))
        logger.info("     Hyper        lr_decay: %s" % (self.lr_decay))
        logger.info("     Hyper         HP_clip: %s" % (self.clip))
        logger.info("     Hyper        momentum: %s" % (self.momentum))
        logger.info("     Hyper              l2: %s" % (self.l2))
        logger.info("     Hyper      hidden_dim: %s" % (self.hidden_dim))
        logger.info("     Hyper         dropout: %s" % (self.dropout))
        logger.info("     Hyper      lstm_layer: %s" % (self.lstm_layer))
        logger.info("     Hyper          bilstm: %s" % (self.bilstm))
        logger.info("     Hyper             GPU: %s" % (self.gpu))
        logger.info("Model SUMMARY END.")
        logger.info("++" * 50)
        sys.stdout.flush()