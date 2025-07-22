# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

import os
import datetime
import warnings
warnings.filterwarnings("ignore")

class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = DROPOUT  # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            # last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            # return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
            last = out.last_hidden_state
            return ((last * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            # return out.last_hidden_state

        if self.pooling == 'first-last-avg':
            # first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            # last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]
            # first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            # last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            # avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            # return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
            first = out.hidden_states[1]
            last = out.hidden_states[-1]
            pooled_result = ((first + last) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                -1).unsqueeze(-1)
            return pooled_result


# 文本信息设置时间率（授课学期、发表时间）
# 生成每位教师的信息生成时间列表（降序，如[2021,2018,2000]）
# 依据每位教师产生自身信息的时间（相对时间）设置时间率(如，[1，rate, rate^2])

# time_DataFrame={te_co,te_pa},time_Str={'SKXQ','FBSJ'}
def Time_rate(time_DataFrame, Rate=0.9, time_Str='SKXQ'):
    # 教师课程信息中，授课学期信息
    #time_DataFrame
    time_DataFrameR = time_DataFrame[['教工号加密', time_Str]].drop_duplicates().sort_values(['教工号加密', time_Str],
                                                                                        ascending=False).set_index(
        '教工号加密')

    # 最终生成的授课学期的数据格式
    SKXQ_time_rate = pd.DataFrame(columns=['教工号加密', time_Str, 'time_rate'])

    # 时间率参数
    rate = Rate

    # 遍历每位老师的时间列表
    for i in time_DataFrame['教工号加密'].unique():
        SKXQ_uni = time_DataFrameR.loc[i, time_Str]

        # 防止教师只有一个时间,提取的信息类型type(SKXQ_uni)=‘str’，此时不能用list()函数
        if isinstance(SKXQ_uni, str):
            c = [SKXQ_uni]
        else:
            c = list(SKXQ_uni)

        # 生成时间列表对应的时间率权重
        c_rate = [round(rate ** count, 2) for count in range(len(c))]
        c_index = [i for count in range(len(c))]

        # 转成DataFrame，便于接下来使用
        c_pd = pd.DataFrame(np.array([c_index, c, c_rate]).T, columns=['教工号加密', time_Str, 'time_rate'])
        SKXQ_time_rate = pd.concat([SKXQ_time_rate, c_pd])

    # 将时间率权重参数（授课学期，发表时间）加到教师信息（课程，论文）中
    time_DataFrame = pd.merge(time_DataFrame, SKXQ_time_rate, on=['教工号加密', time_Str], how='left')

    return time_DataFrame
#将文本转化为BERT句向量
def List_vector(TEXT, model):
    TEXT = str(TEXT)
    if len(TEXT) != 0 :
        source = tokenizer([TEXT], padding=True, truncation=True, max_length=40, return_tensors="pt")
        source_input_ids = source.get('input_ids').to(DEVICE)
        source_attention_mask = source.get('attention_mask').to(DEVICE)
        source_token_type_ids = source.get('token_type_ids').to(DEVICE)
        with torch.no_grad():
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids).to(DEVICE)
        return source_pred.squeeze(0).cpu().numpy()
    return np.zeros(768) #.tolist()


# 导入Excel词向量时出现的错误解决办法
def nan_vect(x):
    if isinstance(x, float):
        return np.zeros(768)
    return x


# 整合教师的时间信息向量
def Time_vector_sum(Vector_name, time_DataFrame, vect_DF):
    # 生成带时间率权重的向量
    if Vector_name in ['课程分词向量', '论文译文分词向量']:
        Vector_name1 = Vector_name[:-2]
        Vector_name2 = Vector_name + '-时间'
        Vector_bert = 'KCMC' if Vector_name1 == "课程分词" else 'LWTM_译文'
        ##过滤词语,放在这里运算量较大
        # time_DataFrame[Vector_name1]=time_DataFrame[Vector_name1].map(lambda x: Dictionary_filter(x,vocabs=Vocabs))
        # 利用model，生成词向量
        # time_DataFrame[Vector_name]=time_DataFrame[Vector_bert].map(lambda x: List_vector(x,model=Model))
        time_DataFrame = pd.merge(time_DataFrame, vect_DF.rename(columns={'sentence': Vector_bert, 'vector': Vector_name})
                                  , on=Vector_bert, how='left')
        time_DataFrame[Vector_name] = time_DataFrame[Vector_name].map(lambda x: nan_vect(x))
        time_DataFrame['time_rate'] = time_DataFrame['time_rate'].astype('float')
        time_DataFrame[Vector_name2] = time_DataFrame[Vector_name] * time_DataFrame['time_rate']
        # 构造最终生成的数据结构
        pd_time_DataFrame = pd.DataFrame(columns=['教工号加密', Vector_name1, Vector_name, Vector_name2])
        # 遍历每位教师，对各项求和（均值），groupby函数对分词向量列不起作用，只能遍历分组
        for i in time_DataFrame['教工号加密'].unique():
            time_DataFrame_sub = time_DataFrame[time_DataFrame['教工号加密'] == i]
            # 分词列表进行加和
            # 分词向量直接求均值
            sum_text = time_DataFrame_sub[Vector_name1].sum()
            sum_vect = time_DataFrame_sub[Vector_name].mean()
            # 带时间率权重的向量求加权权重的平均值
            sum_rate = time_DataFrame_sub['time_rate'].sum()
            sum_veti = time_DataFrame_sub[Vector_name2].sum()
            sum_veti = sum_veti / sum_rate
            # 生成dataframe，利于后面操作，注意这里生成的格式：'课程分词':[sum_text]
            pd_time_DataFrame_sub = pd.DataFrame(
                {'教工号加密': i, Vector_name1: [sum_text], Vector_name: [sum_vect], Vector_name2: [sum_veti]})
            pd_time_DataFrame = pd.concat([pd_time_DataFrame, pd_time_DataFrame_sub])

        return pd_time_DataFrame

    elif Vector_name == '方向分词向量':
        Vector_name1 = Vector_name[:-2]
        Vector_bert = '方向汇总'
        pd_time_DataFrame = pd.DataFrame(columns=['教工号加密', Vector_name1, Vector_name])

        ##过滤词语,放在这里运算量较大
        # time_DataFrame[Vector_name1]=time_DataFrame[Vector_name1].map(lambda x: Dictionary_filter(x,vocabs=Vocabs))
        # 利用model，生成词向量
        # time_DataFrame[Vector_name]=time_DataFrame[Vector_bert].map(lambda x: List_vector(x,model=Model))
        time_DataFrame = pd.merge(time_DataFrame,
                                  vect_DF.rename(columns={'sentence': Vector_bert, 'vector': Vector_name})
                                  , on=Vector_bert, how='left')
        time_DataFrame[Vector_name] = time_DataFrame[Vector_name].map(lambda x: nan_vect(x))

        # 遍历每位教师，对各项求和（均值），groupby函数对分词向量列不起作用，只能遍历分组
        for i in time_DataFrame['教工号加密'].unique():
            time_DataFrame_sub = time_DataFrame[time_DataFrame['教工号加密'] == i]

            # 分词列表进行加和
            # 分词向量直接求均值
            sum_text = time_DataFrame_sub[Vector_name1].sum()
            sum_vect = time_DataFrame_sub[Vector_name].mean()

            pd_time_DataFrame_sub = pd.DataFrame({'教工号加密': i, Vector_name1: [sum_text], Vector_name: [sum_vect]})
            pd_time_DataFrame = pd.concat([pd_time_DataFrame, pd_time_DataFrame_sub])

        return pd_time_DataFrame

#相似度计算函数
def sim_vec(vector1,vector2):
    
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

#匹配准则使用前的判断条件
def is_match(list_t,list_s1,list_s2,list_s3):
    return set(list_s1).issubset(set(list_t)) or set(list_s2).issubset(set(list_t)) or set(list_s3).issubset(set(list_t))

#匹配准则使用时的相似度计算
def sim_vec_match(vector_t,vector_s1,vector_s2,vector_s3):
    s1=np.dot(vector_t,vector_s1)/(np.linalg.norm(vector_t)*np.linalg.norm(vector_s1))
    s2=np.dot(vector_t,vector_s2)/(np.linalg.norm(vector_t)*np.linalg.norm(vector_s2))
    s3=np.dot(vector_t,vector_s3)/(np.linalg.norm(vector_t)*np.linalg.norm(vector_s3))
    return pd.DataFrame([s1,s2,s3]).max()[0]


# 教师推荐（池化准则、匹配准则）
def Recommended_criteria(pd_te_co, pd_te_di, pd_te_pa,  # 三个教师合并后向量{te_co2,te_di2,te_pa2}
                         da_qu,
                         N=20,  # 池化准则中的参数n,每个学生需求下截取的推荐名单长度
                         Pool_reco_size=12,  # 池化推荐人数
                         Match_reco_size=3,  # 匹配推荐人数
                         Consider_matching_criteria=True,  # 是否考虑匹配准则
                         # Consider_time=True, #是否计算没有时间率的方案
                         Consider_time=True  # 是否计算没有时间率的方案
                         ):
    # 设置参数
    n = N  # 池化准则中的参数n,每个学生需求下截取的推荐名单长度
    pool_reco_size = Pool_reco_size  # 池化推荐人数
    match_reco_size = Match_reco_size  # 匹配推荐人数
    st_te_reco_all = []  # 不考虑时间率时，所有学生的教师推荐名单（ID）
    st_te_reco_all_time = []  # 考虑时间率时，所有学生的教师推荐名单（ID）

    # 把教师向量（包含时间）保存到文件
    # te_pool_notime = pd.merge(pd_te_co[['教工号加密', '课程分词向量']], pd_te_di[['教工号加密', '方向分词向量']], on='教工号加密', how='left')
    # te_pool_notime = pd.merge(te_pool_notime, pd_te_pa[['教工号加密', '论文译文分词向量']], on='教工号加密', how='left')
    # csv_file_tea_notime = 'datasets/LLM_data/Teacher_vector_notime.csv'
    # # 检查文件是否存在，若不存在则创建
    # if not os.path.exists(csv_file_tea_notime):
    #     open(csv_file_tea_notime, 'w').close()
    # try:
    #     te_pool_notime.to_csv(csv_file_tea_notime, index=False, encoding='utf-8-sig')
    #     print("教师向量数据已成功保存到Teacher_vector_notime.csv文件中。")
    # except Exception as e:
    #     print(f"保存Teacher_vector_notime.csv文件时出现错误: {e}")

    # -------------------------------------------------------------------------------------------------------------------------------
    # 考虑时间的教师表示
    if Consider_time:
    # 池化准则需要用到的教师数据格式
        te_time_pool = pd.merge(pd_te_co[['教工号加密', '课程分词向量-时间']], pd_te_di[['教工号加密', '方向分词向量']], on='教工号加密', how='left')
        te_time_pool = pd.merge(te_time_pool, pd_te_pa[['教工号加密', '论文译文分词向量-时间']], on='教工号加密', how='left')

        # # 把教师向量（包含时间）保存到文件
        # csv_file_tea_time = 'datasets/LLM_data/Teacher_vector_time.csv'
        # # 检查文件是否存在，若不存在则创建
        # if not os.path.exists(csv_file_tea_time):
        #     open(csv_file_tea_time, 'w').close()
        # try:
        #     te_time_pool.to_csv(csv_file_tea_time, index=False, encoding='utf-8-sig')
        #     print("教师向量数据已成功保存到Teacher_vector_time.csv文件中。")
        # except Exception as e:
        #     print(f"保存Teacher_vector_time.csv文件时出现错误: {e}")


        # 匹配准则需要用到的教师数据格式
        te_time_match = pd.concat(
            [pd_te_co[['教工号加密', '课程分词', '课程分词向量-时间']].rename(columns={'课程分词': '分词', '课程分词向量-时间': '分词向量'}),
             pd_te_di[['教工号加密', '方向分词', '方向分词向量']].rename(columns={'方向分词': '分词', '方向分词向量': '分词向量'}),
             pd_te_pa[['教工号加密', '论文译文分词', '论文译文分词向量-时间']].rename(columns={'论文译文分词': '分词', '论文译文分词向量-时间': '分词向量'})])
        # 遍历每位学生
        for i in range(len(da_qu)):

            st_co_vect = da_qu.iloc[i, -3]
            st_di_vect = da_qu.iloc[i, -2]
            st_pa_vect = da_qu.iloc[i, -1]

            st_co_text = da_qu.iloc[i, -6]
            st_di_text = da_qu.iloc[i, -5]
            st_pa_text = da_qu.iloc[i, -4]

            # 池化推荐

            te_time_pool['课程-池化相似度'] = te_time_pool[['课程分词向量-时间', '方向分词向量', '论文译文分词向量-时间']].applymap(
                lambda x: sim_vec(x, st_co_vect)).max(axis=1)
            te_time_pool['方向-池化相似度'] = te_time_pool[['课程分词向量-时间', '方向分词向量', '论文译文分词向量-时间']].applymap(
                lambda x: sim_vec(x, st_di_vect)).max(axis=1)
            te_time_pool['论文-池化相似度'] = te_time_pool[['课程分词向量-时间', '方向分词向量', '论文译文分词向量-时间']].applymap(
                lambda x: sim_vec(x, st_pa_vect)).max(axis=1)

            st_co_te = te_time_pool[['教工号加密', '课程-池化相似度']].sort_values('课程-池化相似度', ascending=False).iloc[:n, :]
            st_di_te = te_time_pool[['教工号加密', '方向-池化相似度']].sort_values('方向-池化相似度', ascending=False).iloc[:n, :]
            st_pa_te = te_time_pool[['教工号加密', '论文-池化相似度']].sort_values('论文-池化相似度', ascending=False).iloc[:n, :]

            st_te_pool = pd.concat([st_co_te.rename(columns={'课程-池化相似度': '池化相似度'}),
                                    st_di_te.rename(columns={'方向-池化相似度': '池化相似度'}),
                                    st_pa_te.rename(columns={'论文-池化相似度': '池化相似度'})]).groupby(['教工号加密']).agg(
                ['count', max]).reset_index(drop=False).sort_values(
                [('池化相似度', 'count'), ('池化相似度', 'max')], ascending=False)
            st_te_pool_reco = list(st_te_pool.iloc[:pool_reco_size, 0])

            # 匹配推荐
            if Consider_matching_criteria:
                te_time_match['co_is_match'] = te_time_match['分词'].map(
                    lambda x: is_match(x, st_co_text, st_di_text, st_pa_text))

                st_te_match = te_time_match[te_time_match['co_is_match'] & (~te_time_match['教工号加密'].isin(st_te_pool_reco))]
                st_te_match['匹配相似度'] = st_te_match['分词向量'].map(
                    lambda x: sim_vec_match(x, st_co_vect, st_di_vect, st_pa_vect))

                st_te_match = st_te_match[['教工号加密', '匹配相似度']].groupby(['教工号加密']).agg(['count', max]).reset_index(
                    drop=False).sort_values(
                    [('匹配相似度', 'count'), ('匹配相似度', 'max')], ascending=False)

                if len(st_te_match) > match_reco_size:
                    st_te_match_reco = list(st_te_match.iloc[:match_reco_size, 0])
                else:
                    st_te_match_reco = list(st_te_match.iloc[:, 0])
            else:
                st_te_match_reco = []

            st_te_reco = st_te_pool_reco + st_te_match_reco
            st_te_reco_all_time.append(st_te_reco)
            print('\r考虑时间率时，第{}位学生已结束'.format(i + 1), end='')
        # print('\n')
    # ---------------------------------------------------------------------------------------------------------------------------
    # 未考虑时间的教师表示
    else:
        # 池化准则需要用到的教师数据格式
        te_pool = pd.merge(pd_te_co[['教工号加密', '课程分词向量']], pd_te_di[['教工号加密', '方向分词向量']], on='教工号加密', how='left')
        te_pool = pd.merge(te_pool, pd_te_pa[['教工号加密', '论文译文分词向量']], on='教工号加密', how='left')
        # 匹配准则需要用到的教师数据格式
        te_match = pd.concat([pd_te_co[['教工号加密', '课程分词', '课程分词向量']].rename(columns={'课程分词': '分词', '课程分词向量': '分词向量'}),
                              pd_te_di[['教工号加密', '方向分词', '方向分词向量']].rename(columns={'方向分词': '分词', '方向分词向量': '分词向量'}),
                              pd_te_pa[['教工号加密', '论文译文分词', '论文译文分词向量']].rename(
                                  columns={'论文译文分词': '分词', '论文译文分词向量': '分词向量'})])
        # 遍历每位学生
        for i in range(len(da_qu)):
            # 学生三个需求的向量
            st_co_vect = da_qu.iloc[i, -3]
            st_di_vect = da_qu.iloc[i, -2]
            st_pa_vect = da_qu.iloc[i, -1]

            # 学生三个需求的文本
            st_co_text = da_qu.iloc[i, -6]
            st_di_text = da_qu.iloc[i, -5]
            st_pa_text = da_qu.iloc[i, -4]

            # 池化推荐
            # 生成每个学生需求与所有教师的相似度
            te_pool['课程-池化相似度'] = te_pool[['课程分词向量', '方向分词向量', '论文译文分词向量']].applymap(
                lambda x: sim_vec(x, st_co_vect)).max(axis=1)
            te_pool['方向-池化相似度'] = te_pool[['课程分词向量', '方向分词向量', '论文译文分词向量']].applymap(
                lambda x: sim_vec(x, st_di_vect)).max(axis=1)
            te_pool['论文-池化相似度'] = te_pool[['课程分词向量', '方向分词向量', '论文译文分词向量']].applymap(
                lambda x: sim_vec(x, st_pa_vect)).max(axis=1)

            # 取学生每个需求与所有教师的相似度列表中最高的n个
            st_co_te = te_pool[['教工号加密', '课程-池化相似度']].sort_values('课程-池化相似度', ascending=False).iloc[:n, :]
            st_di_te = te_pool[['教工号加密', '方向-池化相似度']].sort_values('方向-池化相似度', ascending=False).iloc[:n, :]
            st_pa_te = te_pool[['教工号加密', '论文-池化相似度']].sort_values('论文-池化相似度', ascending=False).iloc[:n, :]

            # 整合三个需求，在3n长度的序列中，依据频率和相似度排序，选前pool_reco_size个教师作为池化推荐
            st_te_pool = pd.concat([st_co_te.rename(columns={'课程-池化相似度': '池化相似度'}),
                                    st_di_te.rename(columns={'方向-池化相似度': '池化相似度'}),
                                    st_pa_te.rename(columns={'论文-池化相似度': '池化相似度'})]).groupby(['教工号加密']).agg(
                ['count', max]).reset_index(drop=False).sort_values(
                [('池化相似度', 'count'), ('池化相似度', 'max')], ascending=False)

            # 最终的池化推荐教师名单（ID）
            st_te_pool_reco = list(st_te_pool.iloc[:pool_reco_size, 0])

            # 匹配推荐
            if Consider_matching_criteria:
                # 生成每条教师信息是否满足匹配推荐的信息（只要学生三个需求满足一个即可）
                te_match['co_is_match'] = te_match['分词'].map(lambda x: is_match(x, st_co_text, st_di_text, st_pa_text))

                # 筛选出满足匹配推荐和名字未出现在池化推荐的教师信息
                st_te_match = te_match[te_match['co_is_match'] & (~te_match['教工号加密'].isin(st_te_pool_reco))]

                # 生成匹配相似度（学生三个需求与教师信息最大的）
                st_te_match['匹配相似度'] = st_te_match['分词向量'].map(
                    lambda x: sim_vec_match(x, st_co_vect, st_di_vect, st_pa_vect))

                # 依据序列中频率和相似度排序
                st_te_match = st_te_match[['教工号加密', '匹配相似度']].groupby(['教工号加密']).agg(['count', max]).reset_index(
                    drop=False).sort_values(
                    [('匹配相似度', 'count'), ('匹配相似度', 'max')], ascending=False)

                # 选前match_reco_size个教师作为匹配推荐
                if len(st_te_match) > match_reco_size:
                    # 最终的匹配推荐教师名单（ID）
                    st_te_match_reco = list(st_te_match.iloc[:match_reco_size, 0])
                else:
                    # 最终的匹配推荐教师名单（ID）
                    st_te_match_reco = list(st_te_match.iloc[:, 0])
            else:
                st_te_match_reco = []

            # 最终的教师推荐名单（ID）
            st_te_reco = st_te_pool_reco + st_te_match_reco
            # 添加到所有学生的教师推荐名单（ID）
            st_te_reco_all.append(st_te_reco)
            print('\r没有考虑时间率时，第{}位学生已结束'.format(i + 1), end='')
        # print('\n')
    return st_te_reco_all_time, st_te_reco_all
def Recommended_criteria_no_pooling(pd_te_co, pd_te_di, pd_te_pa,  # 三个教师合并后向量{te_co2,te_di2,te_pa2}
                         da_qu,
                         N=20,  # 池化准则中的参数n,每个学生需求下截取的推荐名单长度
                         Pool_reco_size=12,  # 池化推荐人数
                         Match_reco_size=3,  # 匹配推荐人数
                         Consider_matching_criteria=True,  # 是否考虑匹配准则
                         # Consider_time=True, #是否计算没有时间率的方案
                         Consider_time=True  # 是否计算没有时间率的方案
                         ):
    # 设置参数
    n = N  # 池化准则中的参数n,每个学生需求下截取的推荐名单长度
    pool_reco_size = Pool_reco_size  # 池化推荐人数
    match_reco_size = Match_reco_size  # 匹配推荐人数
    st_te_reco_all = []  # 不考虑时间率时，所有学生的教师推荐名单（ID）
    st_te_reco_all_time = []  # 考虑时间率时，所有学生的教师推荐名单（ID） 
    if Consider_time:
    # 池化准则需要用到的教师数据格式
        te_time_pool = pd.merge(pd_te_co[['教工号加密', '课程分词向量-时间']], pd_te_di[['教工号加密', '方向分词向量']], on='教工号加密', how='left')
        te_time_pool = pd.merge(te_time_pool, pd_te_pa[['教工号加密', '论文译文分词向量-时间']], on='教工号加密', how='left')

        # # 把教师向量（包含时间）保存到文件
        # csv_file_tea_time = 'datasets/LLM_data/Teacher_vector_time.csv'
        # # 检查文件是否存在，若不存在则创建
        # if not os.path.exists(csv_file_tea_time):
        #     open(csv_file_tea_time, 'w').close()
        # try:
        #     te_time_pool.to_csv(csv_file_tea_time, index=False, encoding='utf-8-sig')
        #     print("教师向量数据已成功保存到Teacher_vector_time.csv文件中。")
        # except Exception as e:
        #     print(f"保存Teacher_vector_time.csv文件时出现错误: {e}")


        # 匹配准则需要用到的教师数据格式
        te_time_match = pd.concat(
            [pd_te_co[['教工号加密', '课程分词', '课程分词向量-时间']].rename(columns={'课程分词': '分词', '课程分词向量-时间': '分词向量'}),
             pd_te_di[['教工号加密', '方向分词', '方向分词向量']].rename(columns={'方向分词': '分词', '方向分词向量': '分词向量'}),
             pd_te_pa[['教工号加密', '论文译文分词', '论文译文分词向量-时间']].rename(columns={'论文译文分词': '分词', '论文译文分词向量-时间': '分词向量'})])
        te_time_pool['combined'] = te_time_pool.apply(lambda row: np.concatenate([row['课程分词向量-时间'], row['方向分词向量'], row['论文译文分词向量-时间']]), axis=1)
        da_qu['combined'] = da_qu.apply(lambda row: np.concatenate([row['课程_分词向量'], row['方向_分词向量'], row['论文_分词向量']]), axis=1)
        # 遍历每位学生
        for i in range(len(da_qu)):
            st_vect = da_qu.iloc[i, -1]
            te_time_pool['相似度'] = te_time_pool[['combined']].applymap(lambda x: sim_vec(x, st_vect)).max(axis=1)
            
            st_co_te = te_time_pool[['教工号加密', '相似度']].sort_values('相似度', ascending=False)
            st_te_pool_reco = list(st_co_te.iloc[:pool_reco_size, 0])
            st_te_reco_all_time.append(st_te_pool_reco)
    return st_te_reco_all_time

def AP(ranked_list, ground_truth):
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs * 2 / len(ground_truth)
    else:
        return 0


def MAP(ranked_list, ground_truth):  # {st_te_reco_all_name,st_te_truth_all_name}
    M_a_p = []
    for i in range(len(ranked_list)):
        m_a_p = AP(ranked_list[i], ground_truth[i])
        M_a_p.append(m_a_p)
    MAP_value = sum(M_a_p) / len(ranked_list)
    return MAP_value


def hit1(gt_items, pred_items):
    count = 0
    for item in pred_items:
        if item in gt_items:
            count += 1
    return count


def HR(ranked_list, ground_truth, len_ground_truth=3):  # {st_te_reco_all_name,st_te_truth_all_name,真实列表为3个教师}
    H_r = []
    for i in range(len(ranked_list)):
        h_r = hit1(ground_truth[i], ranked_list[i])
        H_r.append(h_r)
    HR_value = sum(H_r) / (len_ground_truth * len(ranked_list))
    return HR_value


def AUC_0_1(ranked_list, ground_truth):
    reco_true = []
    reco_true_len = []
    for i in range(len(ranked_list)):
        stran = ranked_list[i]
        sttan = ground_truth[i]
        reco_true_sub = [name for name in stran if name in sttan]
        reco_true.append(reco_true_sub)
        reco_true_len.append(len(reco_true_sub))
    AUC0 = 1 - reco_true_len.count(0) / len(ranked_list)
    AUC1 = (reco_true_len.count(2) + reco_true_len.count(3)) / len(ranked_list)
    return AUC0, AUC1, reco_true


def Evaluation_metrics(reco_ID_all, trut_Name_all, ID_Name):  # {st_te_reco_all_time or st_te_reco_all}
    if len(reco_ID_all) != 0:
        # 将所有学生的教师推荐名单中的教师ID换成教师姓名
        reco_Name_all = pd.DataFrame(reco_ID_all).replace(ID_Name).values.tolist()

        AUC_value = AUC_0_1(reco_Name_all, trut_Name_all)
        AUC0_value, AUC1_value, recom_true = AUC_value  # [0], AUC_value[1]

        MAP_value = MAP(reco_Name_all, trut_Name_all)

        HR_value = HR(reco_Name_all, trut_Name_all)

        return [AUC0_value, AUC1_value, MAP_value, HR_value], recom_true
    else:
        return [0.0, 0.0, 0.0, 0.0], [0]


def Model_run_time(model_name, te_co_, te_di_, te_pa_, da_qu_,trut_Name_all_, ID_Name_,
                   Rate_=0.9, N_=20, Pool_reco_size_=13, Match_reco_size_=2, Consider_time_=True, match_crit=True):
    # Seed=18
    ##记录时间
    Now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')  # -%H-%M
    ##模型训练
    # model_ = word2vec.Word2Vec(sentences=Sentences,min_count=1,window=Window,sg=1
    #                         ,hs=Hs,seed=Seed,workers=16,size=Size)#vector_size or size
    ##模型保存
    # model_path='{}_model_seed{}_hs_wind{}_size{}.model'.format(Now_time,Seed,Window,Size)
    # model_.save('..\\result_vector\\Parameter_tuning\\'+model_path)
    ##词典读取
    # vocabs_= model_.wv.index2word

    reco_all = Recommended_criteria_no_pooling(pd_te_co=te_co_, pd_te_di=te_di_, pd_te_pa=te_pa_, da_qu=da_qu_,
                                    N=N_, Pool_reco_size=Pool_reco_size_, Match_reco_size=Match_reco_size_,
                                    Consider_matching_criteria=match_crit, Consider_time=Consider_time_)
    Eval_value_time, recom_true = Evaluation_metrics(reco_ID_all=reco_all, trut_Name_all=trut_Name_all_, ID_Name=ID_Name_)
    return [Rate_, N_, Pool_reco_size_, Match_reco_size_] + Eval_value_time, recom_true
    # reco_all = Recommended_criteria(pd_te_co=te_co_, pd_te_di=te_di_, pd_te_pa=te_pa_, da_qu=da_qu_,
    #                                 N=N_, Pool_reco_size=Pool_reco_size_, Match_reco_size=Match_reco_size_,
    #                                 Consider_matching_criteria=match_crit, Consider_time=Consider_time_)
    # st_te_reco_all_time, st_te_reco_all = reco_all[0], reco_all[1]

    # Eval_value_time, recom_true = Evaluation_metrics(reco_ID_all=st_te_reco_all_time, trut_Name_all=trut_Name_all_,
    #                                                  ID_Name=ID_Name_)
    # Eval_value_NOtime, recom_true_NOtime = Evaluation_metrics(reco_ID_all=st_te_reco_all, trut_Name_all=trut_Name_all_,
    #                                                           ID_Name=ID_Name_)

    # return [Now_time, model_name, Rate_, N_, Pool_reco_size_,
    #         Match_reco_size_] + Eval_value_time + Eval_value_NOtime, recom_true

def eval_vector(x):
    str_x = x.replace('\n','').replace('  ',' ').replace('  ',' ').replace(' ',',')#.replace('  ',' ')
    str_x = str_x.replace(',,',',').replace(',,',',').replace(',,',',')
    if str_x[1] == ',':
        return np.array(eval('[' + str_x[2:]))
    elif str_x[1] == '-':
        return np.array(eval(str_x))

if __name__ == '__main__':
    base_url = '/root/autodl-tmp/qwen/'
    #读取数据文件夹下所有数据
    dir_name = base_url + 'datasets/data'
    dir_names = os.listdir(dir_name)
    datas = ['data_' + i[:4] for i in dir_names]

    for fname, file in zip(datas, dir_names):
        print(fname, ":", file)
        name = os.path.join(dir_name, file)
        exec(fname + '= pd.read_excel(name)')
    da_co = data_Cour.copy()
    da_di = data_Dire.copy()
    da_pa = data_Pape.copy()
    da_qu = data_2Que.copy()
    da_qu_new = data_3Que.copy()
    da_te = data_Teac.copy()
    # 经过Excel储存后读取，分词列的列表值外层被套上了字符串格式，去除一下
    da_co['课程分词'] = da_co['课程分词'].map(lambda x: eval(x))
    da_di[['YJFX_词语', 'ZXKMC_词语', 'FXKMC_词语', '方向分词']] = da_di[['YJFX_词语', 'ZXKMC_词语', 'FXKMC_词语', '方向分词']].applymap(
        lambda x: eval(x))
    da_pa[['论文译文分词', '论文分词']] = da_pa[['论文译文分词', '论文分词']].applymap(lambda x: eval(x))
    #da_qu[['课程_分词', '方向_分词', '论文_分词']] = da_qu[['课程_分词', '方向_分词', '论文_分词']].applymap(lambda x: eval(x))

    #可推荐教师的文本信息整合
    # 提取可推荐教师的编号唯一值和姓名
    te_unique = da_te[['教工号加密', 'SKJS']].drop_duplicates()

    # #把te_unique保存到文件
    # csv_file = 'datasets/LLM_data/TeacherID_Name.csv'
    # # 检查文件是否存在，若不存在则创建
    # if not os.path.exists(csv_file):
    #     open(csv_file, 'w').close()
    # try:
    #     te_unique.to_csv(csv_file, index=False, encoding='utf-8-sig')
    #     print("教师姓名和ID数据已成功保存到TeacherID_Name.csv文件中。")
    # except Exception as e:
    #     print(f"保存TeacherID_Name.csv文件时出现错误: {e}")

    # 生成教师ID（key）和姓名（value）典
    ID_NAME = dict(zip(te_unique['教工号加密'], te_unique['SKJS']))
    # 所有学生的真实教师名单
    st_te_truth_all_name = da_qu_new[['指导老师1', '指导老师2', '指导老师3']].values.tolist()
    # 可推荐教师的课程信息（课程分词、授课学期），填充课程分词空值为[]，授课学期空值为'2020-2021'
    te_co = pd.merge(te_unique, da_co[['教工号加密', '课程分词', 'SKXQ', 'KCMC']], on='教工号加密', how='left')
    te_co['课程分词'].fillna('[]', inplace=True)
    te_co['KCMC'].fillna('', inplace=True)
    te_co.loc[te_co['课程分词'] == '[]', '课程分词'] = te_co.loc[te_co['课程分词'] == '[]', '课程分词'].map(lambda x: eval(x))
    te_co['SKXQ'].fillna('2020-2021', inplace=True)
    # 可推荐教师的方向信息（方向分词），填充方向分词空值为[]
    da_di[['YJFX', 'ZXKMC', 'FXKMC']] = da_di[['YJFX', 'ZXKMC', 'FXKMC']].fillna('')
    da_di['方向汇总'] = da_di['YJFX'] + '，' + da_di['ZXKMC'] + '，' + da_di['FXKMC']
    te_di = pd.merge(te_unique, da_di[['教工号加密', '方向分词', '方向汇总']], on='教工号加密', how='left')
    te_di['方向分词'].fillna('[]', inplace=True)
    te_di.loc[te_di['方向分词'] == '[]', '方向分词'] = te_di.loc[te_di['方向分词'] == '[]', '方向分词'].map(lambda x: eval(x))
    te_di['方向汇总'].fillna('', inplace=True)
    # 可推荐教师的论文信息（论文分词、发表时间），填充论文分词空值为[]，发表时间空值为'2021-01-01'
    te_pa = pd.merge(te_unique, da_pa[['教工号加密', '论文译文分词', 'FBSJ', 'LWTM_译文']], on='教工号加密', how='left')
    te_pa['论文译文分词'].fillna('[]', inplace=True)
    te_pa['LWTM_译文'].fillna('', inplace=True)
    te_pa.loc[te_pa['论文译文分词'] == '[]', '论文译文分词'] = te_pa.loc[te_pa['论文译文分词'] == '[]', '论文译文分词'].map(lambda x: eval(x))
    te_pa['FBSJ'].fillna('2021-01-01', inplace=True)
    print(te_unique.shape, te_co.shape, te_di.shape, te_pa.shape)

    #BERT句向量
    # 基本参数
    DROPOUT = 0.3
    MAXLEN = 40
    # POOLING = 'cls' #'last-avg'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预训练模型目录
    # 预模型无影响
    # BERT = 'pretrained_model/bert_pytorch'
    # BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_chinese_pytorch'
    # ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
    BERT_AVRD = base_url+'pretrained_model/Own_encoder'
    model_path = BERT_AVRD

    # Load pretrained model/tokenizer
    # tok无影响
    tokenizer = BertTokenizer.from_pretrained(model_path)  # model_path#'pretrained_model/bert_wwm_ext_chinese_pytorch'
    # tokenizer = BertTokenizer.from_pretrained('pretrained_model/bert_wwm_ext_chinese_pytorch')#model_path#

    ## 微调后参数存放位置
    ##SAVE_PATH = './saved_model/simcse_unsup.pt'#_own
    ##SAVE_PATH = './saved_model/simcse_unsup_own_pooling_last-avg.pt'
    ##SAVE_PATH = './saved_model/ScUnOw_MEnc_epo3_lr1e_bert_Pcl_L1e.pt'
    # SAVE_PATH = './saved_model/ScUnOw_MEnc_epo3_lr3e_roberta_wwm_ext_Pfi.pt'
    # Save_Path = ['ScUnOw_MEnc_epo3_lr3e_roberta_wwm_ext_Pfi.pt',
    # 'ScUnOw_MEnc_epo3_lr3e_roberta_wwm_ext_Pla.pt',
    # 'ScUnOw_MEnc_epo3_lr3e_roberta_wwm_ext_Pcl.pt',
    # 'ScUnOw_MEnc_epo3_lr3e_roberta_wwm_ext_Ppo.pt',
    # 'ScUnOw_MEnc_epo3_lr3e_bert_Pfi.pt',
    # 'ScUnOw_MEnc_epo3_lr3e_bert_Ppo.pt',
    # 'ScUnOw_MEnc_epo3_lr1e_bert_Pfi.pt',
    # 'ScUnOw_MEnc_epo3_lr1e_bert_Ppo.pt',
    # 'ScUnOw_MEnc_epo3_lr1e_bert_wwm_ext_Pfi.pt',
    # 'ScUnOw_MEnc_epo3_lr1e_bert_wwm_ext_Ppo.pt',
    # 'ScUnOw_MEnc_epo3_lr3e_bert_wwm_ext_Pfi.pt',
    # 'ScUnOw_MEnc_epo3_lr3e_bert_wwm_ext_Ppo.pt',
    # 'ScUnOw_MEnc_epo3_lr1e_roberta_wwm_ext_Pfi.pt',
    # 'ScUnOw_MEnc_epo3_lr1e_roberta_wwm_ext_Ppo.pt',
    # 'ScUnOw_MEnc_epo5_lr1e_roberta_wwm_ext_Pfi.pt',
    # 'ScUnOw_MEnc_epo5_lr1e_roberta_wwm_ext_Ppo.pt',
    # 'ScUnOw_MEnc_epo5_lr3e_roberta_wwm_ext_Pfi.pt',
    # 'ScUnOw_MEnc_epo5_lr3e_roberta_wwm_ext_Ppo.pt']
    #师生向量
    Save_Path0 = ['ScUnOw_MEnc_epo3_lr3e_roberta_wwm_ext_Pcl_230625epo1-06.pt']
    Save_Path1 = ['ScUnOw_MEnc_epo3_lr3e_roberta_wwm_ext_Ppo_230625epo1-20.pt']
    Save_Path = ['ScUnOw_MEnc_epo3_lr3e_roberta_wwm_ext_Pcl_LLM_oldandnewData.pt']
    Rate_list = [0.88, 0.89, 0.90, 0.92, 0.95, 1] #0.9, 0.92, 0.95, 1
    for i in range(len(Save_Path)):
        starttime1 = datetime.datetime.now()
        Save_Path_ = Save_Path[i]
        model_name = Save_Path_
        SAVE_PATH = base_url + 'LLM_result/' + Save_Path_
        # for iPO in ['cls', 'pooler', 'first-last-avg', 'last-avg']:
        #     if iPO[:2] == Save_Path_[39:41]:
        #         POOLING = iPO
        ## load model
        for POOLING in ['cls']:
            #assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
            model = SimcseModel(pretrained_model=model_path, pooling=POOLING).to(DEVICE)
            # 获取模型自身的状态字典键
            model_state_dict = model.state_dict()
            loaded_state_dict = torch.load(SAVE_PATH, map_location=DEVICE)
            # 过滤掉多余的键
            filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k in model_state_dict}
            ## eval
            model.load_state_dict(filtered_state_dict)
            model.eval()
            # 句向量
            # 生成使用教师句子的BERT句向量，避免重复句子造成计算损耗
            lite = list(pd.concat([te_co['KCMC'], te_di['方向汇总'], te_pa['LWTM_译文']], axis=0).drop_duplicates())
            lite_df = pd.DataFrame(lite, columns=['sentence'])
            lite_df['sentence'] = lite_df['sentence'].map(lambda x: str(x))
            lite_df['vector'] = lite_df['sentence'].map(lambda x: List_vector(x, model))
            # 调查问卷中学生的向量表示
            da_qu_1 = da_qu.copy()
            da_qu_1[['课程_分词向量', '方向_分词向量', '论文_分词向量']] = da_qu_1[['课程名称', '研究方向', '论文关键词']].applymap(
               lambda x: List_vector(x, model=model))
            da_qu_1.drop(columns=['课程_分词', '方向_分词', '论文_分词'], inplace=True)
            endtime1 = datetime.datetime.now()
            print('\r========================{},RunTime: {}s========================='.format(
                'model: ' + str(i) + '--' + model_name + '向量读取完成', (endtime1 - starttime1).seconds))
    
            # # 把学生向量保存到文件
            # csv_file_stu = 'datasets/LLM_data/Stu_vector.csv'
            # # 检查文件是否存在，若不存在则创建
            # if not os.path.exists(csv_file_stu):
            #     open(csv_file_stu, 'w').close()
            # try:
            #     da_qu_1.to_csv(csv_file_stu, index=False, encoding='utf-8-sig')
            #     print("学生向量数据已成功保存到Stu_vector.csv文件中。")
            # except Exception as e:
            #     print(f"保存Stu_vector.csv文件时出现错误: {e}")
    
    
            for Parameter_2 in Rate_list:
                starttime2 = datetime.datetime.now()
                lite_df_ = lite_df.copy()
                # 可推荐教师的文本最终表示向量（整合时间信息）
                # 加上时间率
                # Rate_ = Parameter_2  # 0.9
                # te_co1 = Time_rate(Rate=Rate_, time_DataFrame=te_co, time_Str='SKXQ')
                # te_di1 = te_di.copy()
                # te_pa1 = Time_rate(Rate=Rate_, time_DataFrame=te_pa, time_Str='FBSJ')
                # # 教师向量
                # te_co2 = Time_vector_sum(Vector_name='课程分词向量', time_DataFrame=te_co1, vect_DF=lite_df_)
                # te_di2 = Time_vector_sum(Vector_name='方向分词向量', time_DataFrame=te_di1, vect_DF=lite_df_)
                # te_pa2 = Time_vector_sum(Vector_name='论文译文分词向量', time_DataFrame=te_pa1, vect_DF=lite_df_)
                # endtime2 = datetime.datetime.now()
                # print('\r****************{} 运行结束,RunTime: {}s*******************'.format(
                #     '时间率: ' + str(Parameter_2) + '--' + '完成', (endtime2 - starttime2).seconds))
                # for Parameter_1 in [[13, 2]]:
                #     Parameter_11 = Parameter_1[0]  # Pool_reco_size_
                #     Parameter_12 = Parameter_1[1]  # Match_reco_size_
                #
                #     for Parameter_3 in [15]:
                #         starttime = datetime.datetime.now()
                #         Parameter_tuning0, _ = Model_run_time(model_name,
                #                                               te_co_=te_co2, te_di_=te_di2, te_pa_=te_pa2, da_qu_=da_qu_1,
                #                                               Rate_=Parameter_2, N_=Parameter_3,
                #                                               Pool_reco_size_=Parameter_11, Match_reco_size_=Parameter_12,
                #                                               Consider_time_=True, match_crit=True,
                #                                               trut_Name_all_=st_te_truth_all_name, ID_Name_=ID_NAME)
                #
                #         print(Parameter_tuning0)
    
                #使用pt文件时使用，不使用pt文件时使用上面注释的代码
                # lite_df_['vector'] = lite_df_['vector'].map(lambda x: eval_vector(x))
                # da_qu_1[['课程_分词向量', '方向_分词向量', '论文_分词向量']] = da_qu_1[['课程_分词向量', '方向_分词向量', '论文_分词向量']].applymap(
                #     lambda x: eval_vector(x))
                # 加上时间率
                Rate_ = Parameter_2
                te_co1 = Time_rate(Rate=Rate_, time_DataFrame=te_co, time_Str='SKXQ')
                te_di1 = te_di.copy()
                te_pa1 = Time_rate(Rate=Rate_, time_DataFrame=te_pa, time_Str='FBSJ')
                # 教师向量
                te_co2 = Time_vector_sum(Vector_name='课程分词向量', time_DataFrame=te_co1, vect_DF=lite_df_)
                te_di2 = Time_vector_sum(Vector_name='方向分词向量', time_DataFrame=te_di1, vect_DF=lite_df_)
                te_pa2 = Time_vector_sum(Vector_name='论文译文分词向量', time_DataFrame=te_pa1, vect_DF=lite_df_)
                #model_name='2333'
                reco_ID_all_Model, reco_true_model=Model_run_time(model_name,
                                                            te_co_=te_co2, te_di_=te_di2, te_pa_=te_pa2, da_qu_=da_qu_1,
                                                            Rate_=Rate_, N_=20, Pool_reco_size_=20, Match_reco_size_=2,
                                                            Consider_time_=True, match_crit=False,
                                                            trut_Name_all_=st_te_truth_all_name,ID_Name_=ID_NAME)
                print(POOLING)
                print('\n',reco_ID_all_Model)
                