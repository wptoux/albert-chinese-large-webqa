# albert-chinese-large-webqa
基于百度webqa与dureader数据集训练的Albert Large QA模型

## 数据来源
+ 百度WebQA 1.0数据集
+ 百度Dureader数据集

## 训练方法
整理后形成类似squad数据集的形式，包含训练数据705139条，验证数据69638条。基于google提供的albert chinese large模型进行finetune。最终f1约0.7

+ 参数
  + learning_rate 1e-5
  + max_seq_length 512
  + max_query_length 50
  + max_answer_length 300
  + doc_stride 256
  + num_train_epochs 2
  + warmup_steps 1000
  + per_gpu_train_batch_size 8
  + gradient_accumulation_steps 3
  + n_gpu 2 (Nvidia Tesla P100)
  
## Metric
![metric](https://github.com/wptoux/albert-chinese-large-webqa/raw/master/webqa-tb.png)

## 使用方法
```
from transformers import AutoModelForQuestionAnswering, BertTokenizer

model = AutoModelForQuestionAnswering.from_pretrained('./model/albert-chinese-large-qa')
tokenizer = BertTokenizer.from_pretrained('./model/albert-chinese-large-qa')
```

## 存在的问题
transformers实现的SquadExample类缺乏对中文的支持，导致其推理结果会存在问题，所以Metric中的F1和Exact会比真实结果低。但是这个不会影响到训练。
