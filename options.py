import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')

    ###PATH Settings###
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--load_data', choices=[False, True], help='update algorithm', default=True)
    parser.add_argument('--load_data_name', default="output/dataset")
    parser.add_argument('--load_model_name', default="output/last.cpt")

    parser.add_argument('--train', default="dataset/2layershort/train.txt")
    parser.add_argument('--dev', default="dataset/2layershort/dev.txt" )
    parser.add_argument('--test', default="dataset/2layershort/test.txt")

    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--decode_dir', default='output/decode.txt')

    parser.add_argument('--wordemb',  help='Embedding for words', default='dataset/glove.6B.100d.txt')
    parser.add_argument('--charemb',  help='Embedding for chars', default=None)
    parser.add_argument('--char_emb_dim', help='Embedding dim for chars', default=30)
    parser.add_argument('--word_emb_dim', help='Embedding dim for words', default=100)

    parser.add_argument('--max_sent_length', type=int, default=200)

    ###Networks
    parser.add_argument('--word_extractor', type=str,choices=['LSTM','CNN','GRU'], help='word level feature extractor', default='LSTM')
    parser.add_argument('--use_char',type=bool,default=True)
    parser.add_argument('--char_extractor', type=str,choices=['LSTM','CNN','GRU'], default='LSTM')
    parser.add_argument('--use_crf', type=bool, default=True)

    ## Training
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--iteration', type=int, default=50)
    parser.add_argument('--average_batch_loss', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='SGD')

    #Multi-Task Loss Hypers
    parser.add_argument('--H2BH', type=float, default=1.)
    parser.add_argument('--H2BB', type=float, default=1.)
    parser.add_argument('--B2HB', type=float, default=1.)
    parser.add_argument('--B2HH', type=float, default=1.)

    ### Hyperparameters
    parser.add_argument('--cnn_layer', type=int, default=4)
    parser.add_argument('--char_hidden_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--lstm_layer', type=int, default=1)
    parser.add_argument('--bilstm', type=bool, default=True)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--clip', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--l2', type=float, default=1e-8)

    args = parser.parse_args()
    return args