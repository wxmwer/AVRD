# -*- encoding: utf-8 -*-
#使用deepseek生成学生和教师的特征文本后，训练个模型，保存为pt文件
import random
import time
from typing import Dict, List
import os
import pandas as pd
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, logging

import warnings
warnings.filterwarnings("ignore")


def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集"""

    def load_snli_data(path):
        with jsonlines.open(path, 'r') as f:
            return [line.get('origin') for line in f]

    def load_AVRD_data(path):
        with open(path, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                lines.append(line.strip('\n'))
        return lines

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]

    def load_LLM_data(path):
        with open(path, "r") as f:
            lines = []
            for line in f.readlines():
                lines.append(line.strip('\n'))
        return lines

    assert name in ["snli", "AVRD", "sts", "LLM"]
    if name == 'snli':
        return load_snli_data(path)
    elif name == 'AVRD':
        return load_AVRD_data(path)
    elif name == 'sts':
        return load_sts_data(path)
    else:
        return load_LLM_data(path)


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, data: List,  tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        inputs = self.tokenizer([f"{text}<|im_end|>", f"{text}<|im_end|>"], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        return inputs
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""

    def __init__(self, data: List, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        inputs = self.tokenizer(f"{text}<|im_end|>", max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        return inputs
    def __getitem__(self, index: int):
        da = self.data[index]
        return self.text_2_id([da[0]]), self.text_2_id([da[1]]), int(da[2])

class Qwen(nn.Module):
    def __init__(self, pooling):
        super(Qwen, self).__init__()
        model_path = '/root/autodl-tmp/qwen/pretrained_model/Qwen2.5-7b'
        config = AutoConfig.from_pretrained(model_path)
        # config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = DROPOUT  # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT
        config.use_cache = False
        config.pretraining_tp = 1
        # config.use_sliding_window_attention = False
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model=model_path, config=config, torch_dtype=torch.bfloat16).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.bos_token is None:  # qwen没有bos_token，要设置一下，不然dpo train时会报错。
            self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.eos_token})
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        self.pooling = pooling

    def get_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        if self.pooling == 'cls':
            result = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[2])
            eos_token_id = self.tokenizer.eos_token_id
            for i in range(last_hidden_state.shape[0]):
                eos_token_index = torch.nonzero((input_ids[i] == eos_token_id) & (attention_mask[i] == 1))
                # eos_token_index = eos_token_index.nonzero(as_tuple=True)[1]
                # 选取从 0 到 index[i] 的元素
                if eos_token_index.numel() == 0:
                    result[i] = last_hidden_state[i, -1]
                else:
                    eos_token_index = eos_token_index[0].item()
                    # print('eos_token_index=',eos_token_index)
                    # print('last_hidden_state[i, eos_token_index] shape=',last_hidden_state[i, eos_token_index].shape)
                    # print('result[i] shape=',result[i].shape)
                    result[i] = last_hidden_state[i, eos_token_index]
            return result

        if self.pooling == 'pooler':
            result = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[2])
            eos_token_id = self.tokenizer.eos_token_id
            for i in range(last_hidden_state.shape[0]):
                eos_token_index = torch.nonzero((input_ids[i] == eos_token_id) & (attention_mask[i] == 1))
                # eos_token_index = eos_token_index.nonzero(as_tuple=True)[1]
                # 选取从 0 到 index[i] 的元素
                if eos_token_index.numel() == 0:

                    result[i] = np.mean(last_hidden_state[i, :])
                else:
                    eos_token_index = eos_token_index[0].item()
                    # print('eos_token_index=',eos_token_index)
                    # print('last_hidden_state[i, eos_token_index] shape=',last_hidden_state[i, eos_token_index].shape)
                    # print('result[i] shape=',result[i].shape)
                    result[i] = np.mean(last_hidden_state[i, 0:eos_token_index+1])
            return result  # [batch, 3584]

        if self.pooling == 'last-avg':
            # last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            # return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
            last = outputs.last_hidden_state
            return ((last * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

        if self.pooling == 'first-last-avg':
            # first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            # last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]
            # first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            # last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            # avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            # return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
            first = outputs.hidden_states[1]
            last = outputs.hidden_states[-1]
            pooled_result = ((first + last) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                -1).unsqueeze(-1)
            return pooled_result

class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""
    def __init__(self, d_model=3584, dropout=0.1):
        super(SimcseModel, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.activation = nn.GELU()
        self.linear3 = nn.Linear(d_model, 1024)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.activation(x)
        # x = self.linear2(x)
        # x = self.activation(x)
        # x = self.linear3(x)
        return x
        # super(SimcseModel, self).__init__()
        # self.linear1 = nn.Linear(d_model, d_model//2)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(d_model//2, D_MODEL)
        # self.activation = nn.GELU()

    # def forward(self, x):
    #     x = self.linear1(x)
    #     x = self.activation(x)
    #     x = self.dropout(x)
    #     x = self.linear2(x)
    #     return x
def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    print('y_pred shape=', y_pred.shape)
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    print('y_true shape=', y_true.shape)
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    print('sim shape=', sim.shape)

    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss


def eval(model, qwen_model, dataloader) -> float:
    """模型评估函数
    批量预测, batch结果拼接, 一次性求spearman相关度
    """

    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            #source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = qwen_model.get_embedding(source_input_ids, source_attention_mask)
            source_pred = model(source_pred.to(DEVICE))
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
            #target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
            target_pred = qwen_model.get_embedding(target_input_ids, target_attention_mask)
            target_pred = model(target_pred.to(DEVICE))
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim.to(DEVICE)), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef train
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(model, train_dl, dev_dl, qwen_model, optimizer) -> None:
    """模型训练函数"""
    global best

    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        #token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)

        input = qwen_model.get_embedding(input_ids, attention_mask)
        out = model(input.to(DEVICE))
        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            corrcoef = eval(model, qwen_model, dev_dl)
            print(f'loss: {loss.item():.4f}, corrcoef:{corrcoef:.4f}')
            #model.train()
            if best < corrcoef:
                best = corrcoef
                # 检查文件是否存在，如果存在则删除
                if os.path.exists(SAVE_PATH):
                    try:
                        os.remove(SAVE_PATH)
                        print(f"已删除存在的文件: {SAVE_PATH}")
                    except Exception as e:
                        print(f"删除文件时出现错误: {e}")
                # 保存状态字典到数据盘
                try:
                    torch.save(model.state_dict(), SAVE_PATH)
                    print(f"模型已成功保存到 {SAVE_PATH}")
                except Exception as e:
                    print(f"保存模型时出现错误: {e}")
                print(f"higher corrcoef: {best:.4f} in batch: {batch_idx}", 'save model')
def save_list_to_txt(lst, file_path):
    try:
        with open(file_path, 'w') as file:
            for item in lst:
                file.write(str(item) + '\n')
        print(f"列表已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
if __name__ == '__main__':
    # stu_path = 'summaries_stu_deepseek.txt'
    # teacher_path = 'summaries_teacher_deepseek.txt'
    # stu_data = pd.read_csv(stu_path, sep='\t')
    # teacher_data = pd.read_csv(teacher_path, sep='\t')
    # data_all = []
    # for i in range(len(stu_data)):
    #     data_all.append(stu_data.loc[i,'Summary_Stu'])
    # for i in range(len(teacher_data)):
    #     data_all.append(teacher_data.loc[i,'Summary_Teacher'])
    # save_list_to_txt(data_all, 'datasets/llm_sentences.txt')

    # 基本参数
    LR = 1e-5
    EPOCHS = 10
    # SAMPLES = 150400  #150400/64=2350    #85.1927%的样本  train_data_AVRD + train_data_snli  #10000
    BATCH_SIZE = 4
    DROPOUT = 0.3
    MAXLEN = 60
    D_MODEL = 512 #SimcseModel网络的输出，即词向量的最终维度
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预训练模型目录
    m_path = [  # 'OwnEnc_epo3_lr1e_bert_chinese_pytorch',
        # 'OwnEnc_epo3_lr3e_bert_chinese_pytorch',
        # 'OwnEnc_epo3_lr1e_bert_wwm_ext_chinese_pytorch',
        # 'OwnEnc_epo3_lr3e_bert_wwm_ext_chinese_pytorch',
        # 'OwnEnc_epo3_lr1e_roberta_wwm_ext_chinese_pytorch',
        'Qwen2.5-7b',
        # 'OwnEnc_epo5_lr1e_roberta_wwm_ext_chinese_pytorch',
        # 'OwnEnc_epo5_lr3e_roberta_wwm_ext_chinese_pytorch'
    ]
    # 数据目录
    base_url = '/root/autodl-tmp/qwen/'
    SNIL_TRAIN = base_url + 'datasets/cnsd-snli/train.txt'
    AVRD_TRAIN = base_url + 'datasets/AVRD-TXT/text_sentences.txt'
    STS_TRAIN = base_url + 'datasets/STS-B/cnsd-sts-train.txt'
    STS_DEV = base_url + 'datasets/STS-B/cnsd-sts-dev.txt'
    STS_TEST = base_url + 'datasets/STS-B/cnsd-sts-test.txt'
    LLM_TRAIN = base_url + 'datasets/llm_sentences.txt'

    train_data_snli = load_data('snli', SNIL_TRAIN)
    train_data_sts = load_data('sts', STS_TRAIN)
    train_data_AVRD = load_data('AVRD', AVRD_TRAIN)
    #train_data_LLM = load_data('LLM', LLM_TRAIN)
    # print(len(train_data_snli),len(train_data_sts),len(train_data_AVRD))
    train_data = train_data_AVRD  # + train_data_snli   # 两个数据集组合 146828+29713=176541
    #train_data = train_data_LLM
    random.seed(18)
    ##random.shuffle(train_data)
    # train_data = random.sample(train_data, SAMPLES) # 随机采样

    dev_data = load_data('sts', STS_DEV)
    test_data = load_data('sts', STS_TEST)
    # random.shuffle(dev_data)
    # random.shuffle(test_data)
    ReSulT = dict()
    for m_path_ in m_path:
        ReSulT[m_path_] = dict()
        for POOLING_ in ['cls']:  # ['cls', 'pooler', 'first-last-avg', 'last-avg']
            qwen_model = Qwen(POOLING_).to(DEVICE)
            tokenizer = qwen_model.tokenizer
            train_dataloader = DataLoader(TrainDataset(train_data, tokenizer), batch_size=BATCH_SIZE)  # "shuffle" default: False
            dev_dataloader = DataLoader(TestDataset(dev_data, tokenizer), batch_size=BATCH_SIZE)
            test_dataloader = DataLoader(TestDataset(test_data, tokenizer), batch_size=BATCH_SIZE)
            ReSulT[m_path_][POOLING_] = dict()
            print(f'===============pooling: {POOLING_}, model: {m_path_}+MLP=================')
            # load model
            assert POOLING_ in ['cls', 'pooler', 'last-avg', 'first-last-avg']
            model = SimcseModel().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # 参数存放位置
            name = f'Qwen_mlp_3584.pt'
            SAVE_PATH = f'/root/autodl-tmp/qwen/LLM_result/{name}'  # _own

            # train
            best = 0
            for epoch in range(EPOCHS):
                print(f'epoch: {epoch}')
                train(model, train_dataloader, dev_dataloader, qwen_model, optimizer)  # 保存条件不变，不能只看损失减少。
            # print(f'train is finished, best model is saved at {SAVE_PATH}')

            # eval
            #model.load_state_dict(torch.load(SAVE_PATH))
            dev_corrcoef = eval(model, qwen_model, dev_dataloader)
            test_corrcoef = eval(model, qwen_model, test_dataloader)
            # print(f'dev_corrcoef: {dev_corrcoef:.4f}')
            # print(f'test_corrcoef: {test_corrcoef:.4f}')

            # result
            print(f'------train is finished, best model is saved at {SAVE_PATH}-----------')
            print(
                f'-----------best: {best:.4f}, test_corrcoef: {test_corrcoef:.4f}, dev_corrcoef: {dev_corrcoef:.4f}--------')
            ReSulT[m_path_][POOLING_]['best_'] = f'{best:.4f}-{name}'
            ReSulT[m_path_][POOLING_]['dev_corrcoef'] = f'{dev_corrcoef:.4f}-{name}'
            ReSulT[m_path_][POOLING_]['test_corrcoef'] = f'{test_corrcoef:.4f}-{name}'
