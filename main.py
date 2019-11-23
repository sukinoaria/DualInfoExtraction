import gc
import os
import time
import random
import logging

import torch.nn as nn
import torch.optim as optim

from models.Dualnet import Dualnet
from models.B2H import B2H
from models.H2B import H2B

from utils.data import Data
from utils.functions import *
from options import parse_argument
from utils.metric import get_ner_fmeasure

import warnings
warnings.filterwarnings('ignore')

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

logger = logging.getLogger(__name__)

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print("  Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def evaluate(data, model,logger, name,best_dev = -1):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    H2BH_pred_results = []
    H2BB_pred_results = []
    B2HH_pred_results = []
    B2HB_pred_results = []
    hgold_results = []
    lgold_results = []
    ## set modules in eval modules
    model.eval()
    batch_size = model.batch_size
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_hlabel,batch_llabel, mask  = batchify_sequence_labeling_with_label(instance, args.gpu,args.max_sent_length, False)
        if args.model == "DUAL":
            H2BH_tag_seqs, H2BB_tag_seqs, B2HB_tag_seqs, B2HH_tag_seqs = model(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)

            H2BHpred_label, H2BBpred_label, hgold_label, lgold_label = recover_label(H2BH_tag_seqs, H2BB_tag_seqs,batch_hlabel, batch_llabel, mask,
                                                                                     data.hlabelset, data.llabelset,batch_wordrecover)
            B2HHpred_label, B2HBpred_label, _, _ = recover_label(B2HH_tag_seqs, B2HB_tag_seqs,batch_hlabel, batch_llabel, mask,
                                                                 data.hlabelset, data.llabelset,batch_wordrecover)
            H2BH_pred_results += H2BHpred_label
            H2BB_pred_results += H2BBpred_label
            B2HH_pred_results += B2HHpred_label
            B2HB_pred_results += B2HBpred_label
            hgold_results += hgold_label
            lgold_results += lgold_label

        elif args.model == "H2B":
            H2BH_tag_seqs, H2BB_tag_seqs, = model(batch_word, batch_wordlen, batch_char, batch_charlen,batch_charrecover, mask)
            hpred_label, lpred_label, hgold_label, lgold_label = recover_label(H2BH_tag_seqs, H2BB_tag_seqs,batch_hlabel, batch_llabel, mask,
                                                                               data.hlabelset, data.llabelset,batch_wordrecover)
            H2BH_pred_results += hpred_label
            H2BB_pred_results += lpred_label
            hgold_results += hgold_label
            lgold_results += lgold_label

        elif args.model == "B2H":
            B2HB_tag_seqs, B2HH_tag_seqs = model(batch_word, batch_wordlen, batch_char, batch_charlen,batch_charrecover, mask)
            hpred_label, lpred_label, hgold_label, lgold_label = recover_label(B2HH_tag_seqs, B2HB_tag_seqs,
                                                                               batch_hlabel, batch_llabel, mask,
                                                                               data.hlabelset, data.llabelset,
                                                                               batch_wordrecover)
            B2HH_pred_results += hpred_label
            B2HB_pred_results += lpred_label
            hgold_results += hgold_label
            lgold_results += lgold_label

    if args.model == "DUAL":
        H2BH_evals, H2BB_evals, H2B_evals = get_ner_fmeasure(hgold_results, lgold_results, H2BH_pred_results,H2BB_pred_results)
        B2HH_evals, B2HB_evals, B2H_evals = get_ner_fmeasure(hgold_results, lgold_results, B2HH_pred_results,B2HB_pred_results)

    elif args.model == "H2B":
        H2BH_evals, H2BB_evals, H2B_evals = get_ner_fmeasure(hgold_results, lgold_results, H2BH_pred_results,H2BB_pred_results, )
        B2HH_evals, B2HB_evals, B2H_evals = [0, 0, 0, 0], [0, 0, 0], [0, 0, 0]

    elif args.model == "B2H":
        H2BH_evals, H2BB_evals, H2B_evals = [0, 0, 0, 0], [0, 0, 0], [0, 0, 0]
        B2HH_evals, B2HB_evals, B2H_evals = get_ner_fmeasure(hgold_results, lgold_results, B2HH_pred_results,B2HB_pred_results, )

    H2B_results = [H2BH_pred_results, H2BB_pred_results]
    B2H_results = [B2HH_pred_results, B2HB_pred_results]

    logger.info(
        "%s --HIGH layer: H2B MODEL  acc:%.4f , p: %.4f, r: %.4f, f: %.4f ||||| B2H MODEL acc:%.4f , p: %.4f, r: %.4f, f: %.4f ." %
        (name.upper(),H2BH_evals[0], H2BH_evals[1], H2BH_evals[2],H2BH_evals[3], B2HH_evals[0], B2HH_evals[1], B2HH_evals[2], B2HH_evals[3]))

    logger.info(
        "%s --BOT layer: H2B MODEL  p: %.4f, r: %.4f, f: %.4f ||||| B2H MODEL  p: %.4f, r: %.4f, f: %.4f ." %
        (name.upper(),H2BB_evals[0], H2BB_evals[1], H2BB_evals[2], B2HB_evals[0], B2HB_evals[1], B2HB_evals[2]))

    logger.info(
        "%s --ALL layer: H2B MODEL  p: %.4f, r: %.4f, f: %.4f ||||| B2H MODEL  p: %.4f, r: %.4f, f: %.4f .best_f: %.4f" %
        (name.upper(),H2B_evals[0], H2B_evals[1], H2B_evals[2], B2H_evals[0], B2H_evals[1], B2H_evals[2], best_dev))

    print(
        "%s --ALL layer: H2B MODEL  p: %.4f, r: %.4f, f: %.4f ||||| B2H MODEL  p: %.4f, r: %.4f, f: %.4f .best_f: %.4f" %
        (name.upper(),H2B_evals[0], H2B_evals[1], H2B_evals[2], B2H_evals[0], B2H_evals[1], B2H_evals[2], best_dev))

    return H2B_evals,B2H_evals, H2B_results,B2H_results

def train(args,data,model):
    logger.info("Training modules...")
    model.show_model_summary(logger)
    print("Training Parameters:%s",args)

    if args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.l2)
    elif args.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        print("Optimizer illegal: %s"%(args.optimizer))
        exit(1)
    best_dev = 0

    ## start training
    for idx in range(args.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        #print("Epoch: %s/%s" %(idx,modules.iteration))
        if args.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, args.lr_decay, args.lr)
        instance_count = 0
        sample_loss = 0
        total_loss = 0
        sample_whole_token = 0
        sample_H2B_high_right_token = 0
        sample_H2B_bot_right_token = 0
        sample_H2B_all_right_token = 0

        sample_B2H_high_right_token = 0
        sample_B2H_bot_right_token = 0
        sample_B2H_all_right_token = 0
        random.shuffle(data.train_Ids)

        model.train()
        model.zero_grad()
        batch_size = args.batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1

        for batch_id in range(total_batch):

            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_hlabel,batch_llabel, mask  =\
                batchify_sequence_labeling_with_label(instance, args.gpu,args.max_sent_length,True)
            instance_count += 1

            if args.model == 'DUAL':
                H2BH_loss, H2BB_loss, B2HB_loss, B2HH_loss, H2BH_tag_seqs, H2BB_tag_seqs, B2HB_tag_seqs, B2HH_tag_seqs = model.calculate_loss(
                    batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_hlabel, batch_llabel,mask)

                H2B_whole, H2B_high_right, H2B_bot_right, H2B_all_right = predict_check(H2BH_tag_seqs, H2BB_tag_seqs,batch_hlabel, batch_llabel,mask)
                sample_whole_token += H2B_whole

                sample_H2B_high_right_token += H2B_high_right
                sample_H2B_bot_right_token += H2B_bot_right
                sample_H2B_all_right_token += H2B_all_right

                _, B2H_high_right, B2H_bot_right, B2H_all_right = predict_check(B2HH_tag_seqs, B2HB_tag_seqs,batch_hlabel, batch_llabel,mask)
                sample_B2H_high_right_token += B2H_high_right
                sample_B2H_bot_right_token += B2H_bot_right
                sample_B2H_all_right_token += B2H_all_right

                loss = args.H2BH*H2BH_loss + args.H2BB*H2BB_loss + args.B2HB*B2HB_loss + args.B2HH*B2HH_loss
            elif args.model == 'H2B':
                H2BH_loss, H2BB_loss, H2BH_tag_seqs, H2BB_tag_seqs = model.calculate_loss(batch_word, batch_wordlen,batch_char, batch_charlen,
                                                                                          batch_charrecover,batch_hlabel, batch_llabel,mask)
                H2B_whole, H2B_high_right, H2B_bot_right, H2B_all_right = predict_check(H2BH_tag_seqs, H2BB_tag_seqs,
                                                                                        batch_hlabel, batch_llabel,
                                                                                        mask)
                sample_whole_token += H2B_whole
                sample_H2B_high_right_token += H2B_high_right
                sample_H2B_bot_right_token += H2B_bot_right
                sample_H2B_all_right_token += H2B_all_right

                loss = args.H2BH * H2BH_loss + args.H2BB * H2BB_loss
            elif args.model == 'B2H':
                B2HB_loss, B2HH_loss, B2HB_tag_seqs, B2HH_tag_seqs = model.calculate_loss(batch_word, batch_wordlen,batch_char, batch_charlen,
                                                                                          batch_charrecover,batch_hlabel, batch_llabel,mask)

                B2H_whole, B2H_high_right, B2H_bot_right, B2H_all_right = predict_check(B2HH_tag_seqs, B2HB_tag_seqs,batch_hlabel, batch_llabel,mask)
                sample_whole_token += B2H_whole
                sample_B2H_high_right_token += B2H_high_right
                sample_B2H_bot_right_token += B2H_bot_right
                sample_B2H_all_right_token += B2H_all_right
                loss = args.B2HB * B2HB_loss + args.B2HH * B2HH_loss

            sample_loss += loss.item()
            total_loss += loss.item()
            #if end%(10*args.batch_size) == 0:
            if end % (10*args.batch_size) == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f;Token Num:%s ||| H2B Hacc:%.4f;Bacc: %.4f;"
                      "ALLacc:%.4f|||||B2H Hacc:%.4f;Bacc:%.4f;ALLacc:%.4f"
                      % (end, temp_cost, sample_loss, sample_whole_token,
                         (sample_H2B_high_right_token + 0.)/ sample_whole_token,(sample_H2B_bot_right_token + 0.)/ sample_whole_token,
                         (sample_H2B_all_right_token + 0.)/ sample_whole_token,(sample_B2H_high_right_token + 0.)/ sample_whole_token,
                         (sample_B2H_bot_right_token + 0.)/ sample_whole_token,(sample_B2H_all_right_token + 0.) / sample_whole_token))

                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0

                sample_whole_token = 0
                sample_H2B_high_right_token = 0
                sample_H2B_bot_right_token = 0
                sample_H2B_all_right_token = 0

                sample_B2H_high_right_token = 0
                sample_B2H_bot_right_token = 0
                sample_B2H_all_right_token = 0

            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f;Token Num:%s ||| H2B Hacc:%.4f;Bacc: %.4f;"
              "ALLacc:%.4f|||||B2H Hacc:%.4f;Bacc:%.4f;ALLacc:%.4f"
              % (end, temp_cost, sample_loss, sample_whole_token,
                 (sample_H2B_high_right_token + 0.) / sample_whole_token,(sample_H2B_bot_right_token + 0.) / sample_whole_token,
                 (sample_H2B_all_right_token + 0.) / sample_whole_token,(sample_B2H_high_right_token + 0.) / sample_whole_token,
                 (sample_B2H_bot_right_token + 0.) / sample_whole_token,(sample_B2H_all_right_token + 0.) / sample_whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        logger.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (idx, epoch_cost, train_num / epoch_cost, total_loss))
        #print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)

        # continue
        if args.model == 'DUAL':
            H2B_evals,B2H_evals, H2B_results,B2H_results= evaluate(data, model,logger, "dev",best_dev=best_dev)
            current_score = B2H_evals[2]
        elif args.model == 'H2B':
            H2B_evals, _,_,_ = evaluate(data, model,logger, "dev",best_dev=best_dev)
            current_score = H2B_evals[2]
        elif args.model == 'B2H':
            B2H_evals, _,_,_ = evaluate(data, model,logger, "dev",best_dev=best_dev)
            current_score = B2H_evals[2]

        if current_score > best_dev:
            print("New f score %f > previous %f ,Save current best modules in file:%s" % (current_score,best_dev,args.load_model_name))
            torch.save(model.state_dict(), args.load_model_name)
            best_dev = current_score
        gc.collect()

def load_model_decode(args,data,model,name):
    print("Load Model from file... ")

    model.load_state_dict(torch.load(args.load_model_name,map_location='cpu'))

    print("Decode %s data ..."% name)
    H2B_evals,B2H_evals, H2B_results,B2H_results= evaluate(data, model, name)

    return H2B_results

if __name__ == '__main__':

    #process args ...
    args = parse_argument()
    args.gpu = torch.cuda.is_available()
    args.load_model_name += "_"+args.model

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='log/output_'+args.model+'.log')
    logger.info("\n\n")
    logger.info("Start Status Logging...")

    #Load data
    data = Data(args)
    if args.load_data :
        print("Load data from Pkl file...")
        data.load(args.load_data_name)
    else:
        print("Generating Pickle data file...")
        data.build_alphabet(data.train_dir)
        data.build_alphabet(data.dev_dir)
        data.build_alphabet(data.test_dir)
        data.fix_alphabet()

        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        print("Saving pkl data %s..." % args.load_data_name)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        data.save(args.load_data_name)

    #initial modules...
    if args.model == "DUAL":
        model = Dualnet(args,data)
    elif args.model == "B2H":
        model = B2H(args, data)
    elif args.model == "H2B":
        model = H2B(args, data)
    else:
        raise ValueError("Invalid Model Type!!!")

    if args.gpu:
        model.to(torch.device("cuda"))

    print("Seed num:" ,seed_num)
    status = args.status.lower()

    if status == 'train':
        print("MODEL: train")
        train(args,data,model)

    elif status == 'test':
        print("MODEL: test")
        print("Load Model from file: ", args.load_model_name)
        model.load_state_dict(torch.load(args.load_model_name, map_location='cpu'))
        if args.gpu:
            model.to(torch.device("cuda"))
        model.show_model_summary(logger)
        evaluate(data, model,logger, "test",best_dev=0)

    elif status == 'decode':
        print("MODEL: decode")
        decode_results = load_model_decode(args,data,model,'test')
        data.write_decoded_results(decode_results, 'test')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")