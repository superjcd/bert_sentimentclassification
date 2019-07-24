import torch


class BaseConfigs(object):
    data_dir = 'data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model = 'bert-base-chinese'
    do_lower_case = True
    max_seq_length = 128
    train_batch_size = 64
    learning_rate = 5e-5
    warmup_proportion = 0.1
    num_train_epochs = 10
    output_dir = load_dir = 'output'


class CustomConfigs(BaseConfigs):
    load_dir = None

