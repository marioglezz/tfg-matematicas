import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='src/NGCF/',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='src/Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='src/NGCF/',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='movielens_1M',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='ngcf',
                        help='Specify the name of model (ngcf).')
    
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[1, 2, 5, 10, 20, 50, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    
    return parser.parse_args()
