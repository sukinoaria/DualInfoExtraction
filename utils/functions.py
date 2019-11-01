# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-14 12:23:52
from __future__ import print_function
from __future__ import absolute_import
import sys
import torch
import numpy as np

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet,hlabelset,llabelset, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r', encoding="utf8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    hlabels = []
    llabels = []
    word_Ids = []
    char_Ids = []
    hlabel_Ids = []
    llabel_Ids = []
    ### for sequence labeling data format i.e. CoNLL 2003
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if sys.version_info[0] < 3:
                word = word.decode('utf-8')
            words.append(word)
            if number_normalized:
                word = normalize_word(word)
            hlabel = pairs[-2]
            llabel = pairs[-1]
            hlabels.append(hlabel)
            llabels.append(llabel)
            word_Ids.append(word_alphabet.get_index(word))
            hlabel_Ids.append(hlabelset.get_index(hlabel))
            llabel_Ids.append(llabelset.get_index(llabel))
            ## get char
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            #only use length to jugde if the email is over
            #if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
            instence_texts.append([words, chars, hlabels,llabels])
            instence_Ids.append([word_Ids, char_Ids,hlabel_Ids,llabel_Ids])
            words = []
            chars = []
            hlabels = []
            llabels = []
            word_Ids = []
            char_Ids = []
            hlabel_Ids = []
            llabel_Ids = []
    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
        instence_texts.append([words, chars, hlabels,llabels])
        instence_Ids.append([word_Ids, char_Ids,hlabel_Ids,llabel_Ids])
        words = []
        chars = []
        hlabels = []
        llabels = []
        word_Ids = []
        char_Ids = []
        hlabel_Ids = []
        llabel_Ids = []
    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim

def predict_check(hpred_variable,lpred_variable, hgold_variable,lgold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    hpred = hpred_variable.cpu().data.numpy()
    lpred = lpred_variable.cpu().data.numpy()
    hgold = hgold_variable.cpu().data.numpy()
    lgold = lgold_variable.cpu().data.numpy()

    mask = mask_variable.cpu().data.numpy()

    hoverlaped = (hpred == hgold)
    loverlaped = (lpred == lgold)
    #correct token when both level is True
    alloverlaped = (hoverlaped == True) & (loverlaped == True)
    right_token = np.sum(alloverlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)

    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

def batchify_sequence_labeling_with_label(input_batch_list, gpu,max_sent_length, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)

    #cut too long texts
    words = []
    chars = []
    hlabels = []
    llabels = []
    for sent in input_batch_list:
        word = sent[0] if len(sent[0]) < max_sent_length else sent[0][:max_sent_length]
        char = sent[1] if len(sent[0]) < max_sent_length else sent[1][:max_sent_length]
        hlabel = sent[2] if len(sent[0]) < max_sent_length else sent[2][:max_sent_length]
        llabel = sent[3] if len(sent[0]) < max_sent_length else sent[3][:max_sent_length]

        words.append(word)
        chars.append(char)
        hlabels.append(hlabel)
        llabels.append(llabel)

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    hlabel_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    llabel_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()

    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    for idx, (seq, hlabel,llabel, seqlen) in enumerate(zip(words, hlabels,llabels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        hlabel_seq_tensor[idx, :seqlen] = torch.LongTensor(hlabel)
        llabel_seq_tensor[idx, :seqlen] = torch.LongTensor(llabel)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    hlabel_seq_tensor = hlabel_seq_tensor[word_perm_idx]
    llabel_seq_tensor = llabel_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        hlabel_seq_tensor = hlabel_seq_tensor.cuda()
        llabel_seq_tensor = llabel_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, hlabel_seq_tensor,llabel_seq_tensor, mask


if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
