'''
 单机训练
'''
import os
import torch
import logging.config
import yaml
from  argparse import ArgumentParser
from torch.nn import CrossEntropyLoss
from tqdm import trange, tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from prepare_data import WeiboProcessor, convert_examples_to_features
from settings import CustomConfigs, BaseConfigs



with open('configs.yaml', 'r') as f:
    logging.config.dictConfig(yaml.load(f.read()))

def main():
    bc = BaseConfigs()
    cc = CustomConfigs()
    # 准备数据处理类 - weiboprocessor
    data_process = WeiboProcessor()
    label_list = data_process.get_labels()
    num_labels = len(label_list)
    parser = ArgumentParser()
    parser.add_argument('--load_model', action='store_true', help='load model from local or not') # 保存模型与否
    # 增加其他参数， 来覆盖CustomConfigs
    args = parser.parse_args()
    if args.load_model:
        if os.path.exists(os.path.join(cc.output_dir, WEIGHTS_NAME)):
            logging.info('从本地加载模型')
            tokenizer = BertTokenizer.from_pretrained(cc.output_dir, do_lower_case=bc.do_lower_case)
            model = BertForSequenceClassification.from_pretrained(cc.output_dir, num_labels=num_labels).to(bc.device)
        else:
            raise ValueError(f'{cc.output_dir} 下面没有对应的模型')
    else:
        logging.info('下载并使用pytorch-pretrainec-model')
        tokenizer = BertTokenizer.from_pretrained(cc.bert_model, do_lower_case=bc.do_lower_case)
        model = BertForSequenceClassification.from_pretrained(cc.bert_model, num_labels=num_labels).to(bc.device)

    # 从本地获取所有数据数据
    train_examples = data_process.get_train_examples(cc.data_dir)
    # 将数据加载为需要都格式
    train_features = convert_examples_to_features(
        train_examples, label_list, cc.max_seq_length, tokenizer)
    # 需要被加入模型的参数
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    # 准备dataloader
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)  # 当你定义了randomsampler之后， 就不需要在定义datalodaer时， 定义shuffle=True
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bc.train_batch_size)
    # 权重衰减
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # 不对bias等不进行权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=cc.learning_rate)
    for i in trange(int(cc.num_train_epochs), desc="Epoch"):
        logging.info(f'The epoch now is {i}')
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(cc.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
        # 每个 epoch 结束以后都需要保存一下模型
        logging.info('we are going to save model')
        model_to_save = model.module if hasattr(model, 'module') else model
        # 保存模型
        output_model_file = os.path.join(cc.output_dir, WEIGHTS_NAME) # 使用特定都模型名称， 以方便使用from_pretrained
        torch.save(model_to_save.state_dict(), output_model_file)
        # 保存config
        output_config_file = os.path.join(cc.output_dir, CONFIG_NAME)
        model_to_save.config.to_json_file(output_config_file)
        # 保存tokenizer
        tokenizer.save_vocabulary(cc.output_dir)



if __name__ == '__main__':
    main()




