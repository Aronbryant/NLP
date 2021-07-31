from os import get_exec_path, getenv
import  random
import numpy as np
from pandas.core.indexes.api import all_indexes_same
import torch
import logging
import pandas as pd
from torch.nn import parameter
from torch.nn.modules import dropout
logging.basicConfig(level=logging.INFO,format='%(asctime)-15s %(levelname)s: %(message)s',filename = 'output.log')

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

gpu=1
use_cuda = gpu>=0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device('cuda',gpu)
else:
    device = torch.device('cpu')
logging.info("Use cuda: %s,gpu id : %d.",use_cuda,gpu)

#把数据分成十份
fold_num = 10
data_file = './dataset/train_set.csv'
def all_data2fola(fold_num,num = 10000):
    fold_data = []
    f = pd.read_csv(data_file,sep = '\t',encoding='UTF-8')
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]
    total = len(labels)
    index = list(range(total))
    np.random.shuffle(index)
    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])
    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)
    all_index = [[]for _ in range(fold_num)]#储存十个fold对应的index
    for label,data in label2id.items():
        batch_size = int(len(data)/fold_num)
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size +1 if i < other else batch_size
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)
    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]
        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start:end]
            fold_labels = labels + other_labels[start:end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels
        assert batch_size == len(fold_labels)
        index = list(range(batch_size))
        np.random.shuffle(index)
        shuffle_fola_texts = []
        shuffle_fola_labels = []
        for i in index:
            shuffle_fola_texts.append(fold_texts[i])
            shuffle_fola_labels.append(fold_labels[i])
        data = {'label':shuffle_fola_labels,'text':shuffle_fola_texts}
        fold_data.append(data)
    logging.info("Fold len %s",str([len(data[label]) for data in fold_data]))
    return fold_data
fold_data = all_data2fola(10)
#拆分训练集验证集，读取测试机
fold_id = 9
dev_data = fold_data[fold_id]
train_texts = []
train_labels = []
for i in range(fold_id):
    data = fold_id[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])

train_data = {'label':train_labels,'text':train_texts}
#读取测试集
test_data_file = './dataset/test_a.csv'
f = pd.read_csv(test_data_file,sep = '\t',encoding='UTF-8')
texts = f['text'].tolist()
test_data = {'label':[0]*len(texts),'texts':texts}
#创建vocab
from collections import Counter
from transformers import BasicTokenizer, modelcard
basick_tokenizer = BasicTokenizer()
class Vocab():
    def __init__(self,train_data):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]','[UNK]']
        self._id2extword = ['[PAD]','[UNK]']
        self._id2label = []
        self.target_names = []
        self.build_vocab(train_data)
        reverse = lambda x:dict(zip(x,range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._label2id)
        logging.info('Build vocab:word %d,label %d.' %(self.word_size,self.label_size)) 
    def build_vocab(self,data):
        self.word_counter = Counter()
        for text in data['text']:
            words = text.spilt()
            for word in words:
                self.word_counter +=1
        for word,count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)
        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}
        self.label_counter = Counter(data['label'])
        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])
    def load_pretrained_embs(self,embfile):
        with open(embfile,encoding='UTF-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count,embedding_dim = int(items[0]),int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index,embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:],dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index +=1
        embeddings[self.unk] = embeddings[self.unk]/word_count
        embeddings = embeddings / np.std(embeddings)
        reverse = lambda x:dict(zip(x,range(len(x))))
        self._exteord2id = reverse(self._id2extword)
        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings
    def word2id(self,xs):
        if isinstance(xs,list):
            return [self._word2id.get(x,self.unk) for x in xs]

        return self._word2id
    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)
    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)
    @property
    def word_size(self):
        return len(self._id2word)
    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)
vocab = Vocab(train_data)
#定义模型
import torch.nn as nn
import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self,hidden_size):
        super(Attention,self).__init__()
        self.weight = nn.parameter(torch.Tensor(hidden_size,hidden_size))
        self.weight.data.normal_(mean = 0.0,std = 0.05)
        self.bias = nn.parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size,dtype = np.float32)
        self.bias.data.copy_(torch.from_numpy(b))
        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0,std=0.05)
    def forward(self,batch_hidden,batch_masks):
        key = torch.matmul(batch_hidden,self.weight) + self.bias
        outputs = torch.matmul(key,self.query)
        masked_outputs = outputs.masked_fill((1-batch_masks).bool(),float(-1e32))
        attn_scores = F.softmax(masked_outputs,dim=1)
        masked_attn_scores = attn_scores.masked_fill((1-batch_masks).bool(),0.0)
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1),key).squeeze(1)
        return batch_outputs,attn_scores
word2vec_path = 'word2vec.txt'
dropout = 0.15
class WordCNNEncoder(nn.Module):
    def __init__(self,vocab):
        super(WordCNNEncoder).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 100
        self.word_embed = nn.Embedding(vocab.word_size,self.word_dims,padding_idx=0)
        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size ,word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))
        self.extword_embed = nn.Embedding(extword_size,word_dims,padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False
        input_size = self.word_dims
        self.filter_sizes = [2,3,4]
        self.out_channel = 100

        self.convs = nn.ModuleList([nn.Conv2d(1,self.out_channel,(filter_size,input_size),bias=True)
        for filter_size in self.filter_sizes])
    def forward(self,word_ids,extword_ids):
        sen_nums,sent_len = word_ids.shape
        word_embed = self.word_embed(word_ids)
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed+extword_embed
        if self.training:
            batch_embed = self.dropout(batch_embed)
        batch_embed.unsqueeze(1)

        pooled_outputs = []
        for i in range(len(self.filter_szies)):
            filter_height = sent_len - self.filter_sizes[i] +1
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)
            mp = nn.MaxPool2d((filter_height,1))
            pooled = mp(hidden).reshape(sen_nums,self.out_channel)
            pooled_outputs.append(pooled)
        reps = torch.cat(pooled_outputs,dim  = 1)

        if self.training:
            reps = self.dropout(reps)
        return reps
#定义sentencoder
sent_hidden_size = 256
sent_num_layers = 2
class SentEncoder(nn.Module):
    def __init__(self,sent_rep_size):
        super(SentEncoder,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.sent_lstm - nn.LSTM(
            input_size = sent_rep_size,
            hidden_size = sent_hidden_size,
            num_layers = sent_num_layers,
            batch_first = True,
            bidirectional = True   
        )
    def forward(self,sent_reps,sent_masks):
        sent_hiddens,_ = self.sent_lstm(sent_reps)
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)
        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)
        return sent_hiddens
#定义整个模型
class Model(nn.Module):
    def __init__(self,vocab):
        super(Model,self).__init__()
        self.sent_reps_size = 300
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordCNNEncoder(vocab)
        parameters.extend(list(filter(lambda p: p.requires_grad,self.word_encoder.parameters())))
        self.sent_encoder = SentEncoder(self.sent_reps_size)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad,self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))
        # doc_rep_size
        self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        logging.info('Build model with cnn word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))
    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b * doc_len * sentence_len
        # batch_masks : b * doc_len * sentence_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        # batch_inputs1: sentence_num * sentence_len
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  
        # batch_inputs2: sentence_num * sentence_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)
        # batch_masks: sentence_num * sentence_len 
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  
        # sent_reps: sentence_num * sentence_rep_size
        # sen_num * (3*out_channel) =  sen_num * 300
        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2) 
        
        
        # sent_reps：b * doc_len * sent_rep_size
        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  
        # batch_masks：b * doc_len * max_sent_len
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  
        # sent_masks：b * doc_len any(2) 表示在 第二个维度上判断
        # 表示如果如果一个句子中有词 true，那么这个句子就是 true，用于给 lstm 过滤
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len
        # sent_hiddens: b * doc_len * num_directions * hidden_size
        # sent_hiddens:  batch, seq_len, 2 * hidden_size
        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  
        
        
        # doc_reps: b * (2 * hidden_size)
        # atten_scores: b * doc_len
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  
        
        # b * num_labels
        batch_outputs = self.out(doc_reps)  

        return batch_outputs


model = Model(vocab)
# 把文章划分为句子

def sentence_split(text,vocab,max_sent_len = 256,max_segmet=16):
    words = text.strip().split()
    document_len = len(words)
    index = list(range(0,document_len,max_sent_len))
    index.append(document_len)
    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]:index[i+1]]
        assert  len(segment) > 0
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        segments.append([len(segment),segment])
    assert len(segments) > 0
    if len(segments) > max_segmet:
        segment_ = int(max_segmet / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments
def get_examples(data,vocab,max_sent_len = 256,max_segmet=8):
    label2id = vocab.label2id
    examples = []
    for text,label in zip(data['text'],data['label']):
        id = label2id(label)
        sents_words = sentence_split(text, vocab, max_sent_len, max_segmet)
        doc = []
        for sent_len,sent_words in sents_words:
            word_ids = vocab.word2id(sent_words)
            # 把 word 转为 ext id
            extword_ids = vocab.extword2id(sent_words)
            doc.append([sent_len, word_ids, extword_ids])
        examples.append([id, len(doc), doc])

    logging.info('Total %d docs.' % len(examples))
    return examples
#定义batch_slice
def batch_silce(data,batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_bathc_size = batch_size if i < batch_num - 1 else len(data) - batch_size*i
        docs = [data[i*batch_size + b] for b in range(cur_bathc_size)]

        yield docs
def data_iter(data,batch_size,shuffle=True,noise =1.0):
    batched_data = []
    if shuffle:
        np.random.shuffle(data)
        lengths = [example[1] for example in data]
        noisy_lengths = [-(l + np.random.random.uniform(-noise,noise))for l in lengths]
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data
    batched_data.extend(list(batch_size(sorted_data,batch_size)))
    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch
#metric

# some function
from sklearn.metrics import f1_score, precision_score, recall_score


def get_score(y_ture, y_pred):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_ture, y_pred, average='macro') * 100
    p = precision_score(y_ture, y_pred, average='macro') * 100
    r = recall_score(y_ture, y_pred, average='macro') * 100

    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2)

# 保留 n 位小数点
def reformat(num, n):
    return float(format(num, '0.' + str(n) + 'f'))
#定义训练和计算方法
import time 
from sklearn.metrics import classification_report
clip = 5.0
epochs = 17
early_stops = 3
log_interval = 50
test_batch_size = 128
train_batch_size = 128
save_model = './cnn0.bin'
save_test = './cnn0.csv'
class Trainer():
    def __init__(self,model,vocab):
        self.model = model
        self.report = True
        self.train_data = get_examples(train_data,vocab)
        self.batch_num = int(np.ceil(len(self.train_data) / float(train_batch_size)))
        self.dev_data = get_examples(dev_data,vocab)
        self.test_data = get_examples(test_data,vocab)
        self.criterion = nn.CrossEntropyLoss()
        self.target_names = vocab.target_names
        self.optimizer = Optimizer(model.all_parameters)
        self.step = 0
        self.early_stop = -1
        self.best_train_f1,self.best_dev_f1 = 0,0
        self.last_epoch = epochs

    def train(self):
        for epoch in range(epochs):
            train_f1 = self._train(epoch)
            dev_f1 = self._eval(epoch)
            if self.best_dev_f1 <= dev_f1:
                logging.info(
                    "Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(),save_model)
                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == early_stops:
                    logging.info(
                        "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
                            epoch - early_stops, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break    
    def test(self):
        self.model.load_state_dict(torch.load(save_model))
        self._eval(self.last_epoch + 1, test=True)
    
    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            # batch_inputs: (batch_inputs1, batch_inputs2, batch_masks)
            # 形状都是：batch_size * doc_len * sent_len
            # batch_labels: batch_size
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            # batch_outputs：b * num_labels
            batch_outputs = self.model(batch_inputs)
            # criterion 是 CrossEntropyLoss，真实标签的形状是：N
            # 预测标签的形状是：(N,C)
            loss = self.criterion(batch_outputs, batch_labels)
            
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value
            # 把预测值转换为一维，方便下面做 classification_report，计算 f1
            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()
            self.optimizer.zero_grad()

            self.step += 1

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                
                lrs = self.optimizer.get_lr()
                logging.info(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / log_interval,
                        elapsed / log_interval))
                
                losses = 0
                start_time = time.time()
                
            batch_idx += 1
            
        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat 保留 4 位数字
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)

        logging.info(
            '| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f}'.format(epoch, score, f1,
                                                                                  overall_losses,
                                                                                  during_time))
        # 如果预测和真实的标签都包含相同的类别数目，才能调用 classification_report                                                                        
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1
# build optimizer
learning_rate = 2e-4
decay = .75
decay_step = 1000


class Optimizer:
    def __init__(self, model_parameters):
        self.all_params = []
        self.optims = []
        self.schedulers = []

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=learning_rate)
                self.optims.append(optim)

                l = lambda step: decay ** (step // decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)

            else:
                Exception("no nameed parameters.")

        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res
    # 这里验证集、测试集都使用这个函数，通过 test 来区分使用哪个数据集
    def _eval(self, epoch, test=False):
        self.model.eval()
        start_time = time.time()
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                            # batch_inputs: (batch_inputs1, batch_inputs2, batch_masks)
            # 形状都是：batch_size * doc_len * sent_len
            # batch_labels: batch_size                                                                  
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                # batch_outpu
                # ts：b * num_labels                                                                  
                batch_outputs = self.model(batch_inputs)
                # 把预测值转换为一维，方便下面做 classification_report，计算 f1                                                                  
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, f1 = get_score(y_true, y_pred)

            during_time = time.time() - start_time
            
            if test:
                df = pd.DataFrame({'label': y_pred})
                df.to_csv(save_test, index=False, sep=',')
            else:
                logging.info('| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'.format(epoch, score, f1, during_time))
                                                                                  
                if set(y_true) == set(y_pred) and self.report:
                    report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                    logging.info('\n' + report)

        return f1

    
    # data 参数就是 get_examples() 得到的，经过了分 batch
    # batch_data是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)
    # 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
    def batch2tensor(self, batch_data):
        '''
            [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]]
        '''
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            # doc_data 代表一篇新闻，是一个 tuple: (label, 句子数量，doc)
            # doc_data[0] 是 label
            doc_labels.append(doc_data[0])
            # doc_data[1] 是 这篇文章的句子数量
            doc_lens.append(doc_data[1])
            # doc_data[2] 是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
            # 所以 sent_data[0] 表示每个句子的长度（单词个数）
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            # 取出这篇新闻中最长的句子长度（单词个数）
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)
        
        # 取出最长的句子数量
        max_doc_len = max(doc_lens)
        # 取出这批 batch 数据中最长的句子长度（单词个数）
        max_sent_len = max(doc_max_sent_len)
        # 创建 数据
        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                # batch_data[b][2] 表示一个 list，是一篇文章中的句子
                sent_data = batch_data[b][2][sent_idx] #sent_data 表示一个句子
                for word_idx in range(sent_data[0]): # sent_data[0] 是句子长度(单词数量)
                    # sent_data[1] 表示 word_ids
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    # # sent_data[2] 表示 extword_ids
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    # mask 表示 哪个位置是有词，后面计算 attention 时，没有词的地方会被置为 0                                               
                    batch_masks[b, sent_idx, word_idx] = 1

        if use_cuda:
            batch_inputs1 = batch_inputs1.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels


# train
trainer = Trainer(model, vocab)
trainer.train()