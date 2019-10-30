from __future__ import print_function
from __future__ import absolute_import
import sys
from .alphabet import Alphabet
from .functions import *
import pickle as pickle

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"

class Data:
    def __init__(self,args):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1

        self.number_normalized = False
        self.norm_word_emb = False
        self.norm_char_emb = False

        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.hlabelset = Alphabet('high_label',True)
        self.llabelset = Alphabet('low_label', True)

        self.tagScheme = "BIO" ## BMES/BIO
        self.split_token = ' ||| '

        ### I/O
        self.train_dir = args.train
        self.dev_dir = args.dev
        self.test_dir = args.test
        self.decode_dir = args.decode_dir

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None

        self.word_emb_dir = args.wordemb
        self.char_emb_dir = args.charemb
        self.word_emb_dim = args.word_emb_dim
        self.char_emb_dim = args.char_emb_dim

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.hlabelset_size = 0
        self.llabelset_size = 0
        self.norm_feature_embs = []

    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding='utf-8').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                hlabel = pairs[-2]
                llabel = pairs[-1]
                self.hlabelset.add(hlabel)
                self.llabelset.add(llabel)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.hlabelset_size = self.hlabelset.size()
        self.llabelset_size = self.llabelset.size()

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.hlabelset.close()
        self.llabelset.close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm: %s, dir: %s"%(self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)
        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s"%(self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)

    def generate_instance(self, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet, self.char_alphabet, self.hlabelset,self.llabelset, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(self.dev_dir, self.word_alphabet, self.char_alphabet, self.hlabelset,self.llabelset, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(self.test_dir, self.word_alphabet, self.char_alphabet,self.hlabelset,self.llabelset, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def write_decoded_results(self, predict_results, name):
        
        sent_num = len(predict_results)
        content_list = []
        if name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test !")
        assert(sent_num == len(content_list))
        fout = open(self.decode_dir,'w')
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, self.decode_dir))


    def load(self,data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self,save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
