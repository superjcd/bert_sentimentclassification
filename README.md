# 使用Bert进行文本分类-以微博情感分析为例
  使用预训练好的bert模型， 我们可以进行文本分类（包括情感分析）、句子关系判断(包括文本蕴含、文本匹配等)、序列标注（包括命名实体识别等）等迁移学习任务。
  本例， 我基于[pytorch-transforms](https://github.com/huggingface/pytorch-transformers)中的分类examples改写的。
  另外， 如果对其他对文本分类方法感兴趣， 比如BiLSTM及textCNN， 可以参考[这里](https://github.com/superjcd/sentimentclassification)。
  
### 文件构成
```
├── classifer.py      文本分类主程序
├── configs.yaml      logging的配置文件
├── data              微博分类数据
├── logs              logging输出
├── output            模型保存目录
├── prepare_data.py   数据处理
└── settings.py       配置文件
```

### 使用方式
  在settings.py中配置完相应参数之后，可以进行文本分类模型训练：
  如果是初次训练：
```
   python classifer.py
```
  从output目录加载已经训练好的模型
```
   python classifer.py --load_model
```

### 说明
  由于bert模型的比较大， 有大量的参数， 所以训练过程会非常缓慢（事实上只有cpu的我并没有完成训练）， 如果没有多块GPU，建议放弃！ 


