import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    # General options
    parser.add_argument("--model", required=True, type=str,
                        help="Name of model; nasa_lstm")
    parser.add_argument("--device", default='cuda', type=str,
                        help="cuda or cpu")
    parser.add_argument("--dataset", default='nasa', type=str,
                        help="Dataset for training; nasa")
    parser.add_argument("--test_only", default=False, action='store_true',
                        help="Network test only")
    parser.add_argument("--train_set", default=None, type=str,
                        help="Specified training set; battery ids separated with comma(,) are required.")
    parser.add_argument("--columns", default='V,I,T,dt', type=str,
                        help="Data columns for training; column names separated with comma(,) are required.")
    
    parser.add_argument("--continue_training", default=None, type=str,
                        help="Continue interrupted training; Directory for log files is needed.")
    parser.add_argument("--test_model", default=None, type=str,
                        help="Test a trained model; Directory for log files is needed.")
    parser.add_argument("--test_epoch", default=None, type=int,
                        help="Epoch number of a trained model")
    parser.add_argument("--log_test_history", default=False, action='store_true',
                        help="Log test history")
    parser.add_argument("--comment", default=None, type=str,
                        help="Comments on training the model")
    
    # Model options
    parser.add_argument("--rnn_input_size", default=None, type=int,
                        help="RNN input dimension; default input size is the number of data columns")
    parser.add_argument("--rnn_hidden_size", default=16, type=int,
                        help="RNN hidden layer dimension")
    parser.add_argument("--rnn_n_layer", default=2, type=int,
                        help="RNN number of layers")
    parser.add_argument("--use_aggregation", default=False, action='store_true',
                        help="Use feature aggregation on the output for RNN")
    
    # Training options
    parser.add_argument("--num_epochs", default=100, type=int,
                        help="Total number of training epochs")
    parser.add_argument("--max_train_time", default=None, type=int,
                        help="Maximum training time; write in seconds")
    
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--sliding_window", default=32, type=int)
    parser.add_argument("--target", default='soh', type=str,
                        help="Target column; soh/capacity")
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--soh_loss_weight", default=1., type=float,
                        help="Weight for the SOH loss")
    parser.add_argument("--summary_steps", default=100, type=int,
                        help="Training summary frequency")
    
    return parser.parse_args()