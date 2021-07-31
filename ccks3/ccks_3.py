from paddlenlp.data import Stack, Tuple, Pad
import paddlenlp as ppnlp
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
import numpy as np
import os
import json
import time
import paddle
import paddlenlp
from functools import partial


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()



def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
def convert_example(example, tokenizer, max_seq_length=80, is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids
train_tmp = []
dev_tmp = []
num = 0 
a = []
for line in open('data/Xeon3NLP_round1_train_20210524.txt','r'):
    t = json.loads(line)
    for j in t['candidate']:
        l = dict()
        l['query'] = str(t['query'])
        l['title'] = str(j['text'])
        a.append(len(l['title']))
        if j['label'] == '不匹配':
            l['label'] = 0
        elif j['label'] == '完全匹配':
            l['label'] = 2
        else:
            l['label'] = 1
        if num < 18000:
            train_tmp.append(l)
         else:
            dev_tmp.append(l)
        
    num += 1
num

from paddlenlp.datasets import load_dataset
def read(filename):
    for line in filename:
        yield {'query': line['query'], 'title': line['title'], 'label':line['label']}

train_ds = load_dataset(read, filename=train_tmp, lazy=False)
dev_ds = load_dataset(read, filename=dev_tmp, lazy=False)
for idx, example in enumerate(train_ds):
    if idx <= 5:
        print(example)

pretrained_model = paddlenlp.transformers.RobertaModel.from_pretrained('roberta-wwm-ext')

tokenizer = paddlenlp.transformers.RobertaTokenizer.from_pretrained('roberta-wwm-ext')

import paddle.nn as nn

class PointwiseMatching(nn.Layer):
   
    # 此处的 pretained_model 在本例中会被 ERNIE1.0 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 3)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        return logits
model = PointwiseMatching(pretrained_model)

batch_size = 16
# 训练过程中的最大学习率
learning_rate = 2e-5 
# 训练轮次
epochs = 3
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 1e-4

# 训练集的样本转换函数
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=80)



batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]

train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
    

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

global_step = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 500 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    evaluate(model, criterion, metric, dev_data_loader)
    

save_dir = os.path.join("model")
os.makedirs(save_dir)

save_param_path = os.path.join(save_dir, 'model_state.pdparams')
paddle.save(model.state_dict(), save_param_path)
tokenizer.save_pretrained(save_dir)
batchify_test_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
): [data for data in fn(samples)]

num = 0
fo = open("submit_addr_match_runid.txt", "w")
for line in open('data/Xeon3NLP_round1_test_20210524.txt','r'):
    t = json.loads(line)
    for j in range(len(t['candidate'])):
        tmp = []
        l = dict()
        l['query'] = t['query']
        l['title'] = t['candidate'][j]['text']
        l['label'] = 0
        input_ids, token_type_ids,label = convert_example(l, tokenizer, max_seq_length=80)
        tmp.append((input_ids, token_type_ids))
        input_ids, token_type_ids = batchify_test_fn(tmp)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        idx = paddle.argmax(logits, axis=1).numpy()
        if idx[0] == 0:
            t['candidate'][j]['label'] = '不匹配'
        elif idx[0] == 1:
            t['candidate'][j]['label'] = '部分匹配'
        else:
            t['candidate'][j]['label'] = '完全匹配'
    if num < 10:
        print(t)
    if num % 500 == 0:
        print(num)
    fo.write(json.dumps(t,ensure_ascii=False))
    fo.write('\n')
    num +=1
