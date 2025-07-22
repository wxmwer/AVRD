# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, logging
import os
import warnings

warnings.filterwarnings("ignore")


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
        self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # if self.tokenizer.bos_token is None:  # qwen没有bos_token，要设置一下，不然dpo train时会报错。
        #     self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.eos_token})
        #     self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        self.pooling = pooling

    def get_embedding(self, input_ids, attention_mask):
        """
        :param input_ids:
        :param attention_mask:
        :return:
        """
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

        if self.pooling == 'cls':
            result = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[2])
            for i in range(last_hidden_state.shape[0]):
                result[i] = last_hidden_state[i, -1]
            return result
        if self.pooling == 'mean':
            result = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[2])
            for i in range(last_hidden_state.shape[0]):
                result[i] = torch.mean(last_hidden_state[i, 1:])
            return result  # [batch, 3584]


# 文本信息设置时间率（授课学期、发表时间）
# 生成每位教师的信息生成时间列表（降序，如[2021,2018,2000]）
# 依据每位教师产生自身信息的时间（相对时间）设置时间率(如，[1，rate, rate^2])

# time_DataFrame={te_co,te_pa},time_Str={'SKXQ','FBSJ'}
def Time_rate(time_DataFrame, Rate=0.85, time_Str='SKXQ'):
    # 教师课程信息中，授课学期信息
    # time_DataFrame
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


# 将文本转化为Qwen句向量
def List_vector(TEXT, qwen_model, tokenizer):
    TEXT = str(TEXT)
    if len(TEXT) != 0:
        source = tokenizer(f"{TEXT}<|im_end|>", padding=True, truncation=True, max_length=40, return_tensors="pt")
        source = {key: value.to(DEVICE) for key, value in source.items()}
        source_input_ids = source.get('input_ids').to(DEVICE)
        source_attention_mask = source.get('attention_mask').to(DEVICE)
        # print(TEXT)
        with torch.no_grad():
            source_pred = qwen_model.get_embedding(source_input_ids, source_attention_mask)
        return source_pred.squeeze().cpu().numpy()
    return np.zeros(QWEN_DIM)  # .tolist()


# 导入Excel词向量时出现的错误解决办法
def nan_vect(x):
    if isinstance(x, float):
        return np.zeros(QWEN_DIM)
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
        time_DataFrame = pd.merge(time_DataFrame,
                                  vect_DF.rename(columns={'sentence': Vector_bert, 'vector': Vector_name})
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


if __name__ == '__main__':
    # 读取数据文件夹下所有数据
    base_url = '/root/autodl-tmp/qwen/'
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
    # da_qu[['课程名称', '研究方向', '论文关键词']] = da_qu[['课程名称', '研究方向', '论文关键词']].applymap(lambda x: eval(x))

    te_unique = da_te[['教工号加密', 'SKJS']].drop_duplicates()
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
    # print(te_unique.shape, te_co.shape, te_di.shape, te_pa.shape)

    # BERT句向量
    # 基本参数
    DROPOUT = 0.3
    POOLING = 'mean'
    QWEN_DIM = 3584
    # POOLING = 'cls' #'last-avg'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_url = '/root/autodl-tmp/qwen/'

    # 大模型读取
    qwen_model = Qwen(POOLING)
    qwen_model.to(DEVICE)
    tokenizer = qwen_model.tokenizer

    # 调查问卷中学生的向量表示
    da_qu_1 = da_qu.copy()
    da_qu_1[['课程_分词向量', '方向_分词向量', '论文_分词向量']] = da_qu_1[['课程名称', '研究方向', '论文关键词']].applymap(
        lambda x: List_vector(x, qwen_model, tokenizer))
    da_qu_1.drop(columns=['课程_分词', '方向_分词', '论文_分词'], inplace=True)

    file_stu_notime = 'qwen_stu_vector_mean.h5'
    # 检查文件是否存在，若不存在则创建
    if not os.path.exists(file_stu_notime):
        open(file_stu_notime, 'w').close()
    try:
        # da_qu_1.to_csv(file_stu_notime, index=False, encoding='utf-8-sig')
        da_qu_1.to_hdf(file_stu_notime, key='stu', mode='w')
        print("学生向量数据已成功保存到qwen_stu_vector_mean.h5文件中。")
    except Exception as e:
        print(f"保存qwen_stu_vector_mean.h5文件时出现错误: {e}")

    # 生成使用教师句子的BERT句向量，避免重复句子造成计算损耗
    lite = list(pd.concat([te_co['KCMC'], te_di['方向汇总'], te_pa['LWTM_译文']], axis=0).drop_duplicates())
    lite_df = pd.DataFrame(lite, columns=['sentence'])
    lite_df['sentence'] = lite_df['sentence'].map(lambda x: str(x))
    lite_df['vector'] = lite_df['sentence'].map(lambda x: List_vector(x, qwen_model, tokenizer))
    Rate_ = 0.85
    # 加上时间率
    te_co1 = Time_rate(Rate=Rate_, time_DataFrame=te_co, time_Str='SKXQ')
    te_di1 = te_di.copy()
    te_pa1 = Time_rate(Rate=Rate_, time_DataFrame=te_pa, time_Str='FBSJ')
    # 教师向量
    te_co2 = Time_vector_sum(Vector_name='课程分词向量', time_DataFrame=te_co1, vect_DF=lite_df)
    te_di2 = Time_vector_sum(Vector_name='方向分词向量', time_DataFrame=te_di1, vect_DF=lite_df)
    te_pa2 = Time_vector_sum(Vector_name='论文译文分词向量', time_DataFrame=te_pa1, vect_DF=lite_df)

    te_time_pool = pd.merge(te_co2[['教工号加密', '课程分词向量']], te_di2[['教工号加密', '方向分词向量']], on='教工号加密', how='left')
    te_time_pool = pd.merge(te_time_pool, te_pa2[['教工号加密', '论文译文分词向量']], on='教工号加密', how='left')

    te_time_match = pd.concat(
        [te_co2[['教工号加密', '课程分词', '课程分词向量-时间']].rename(columns={'课程分词': '分词', '课程分词向量-时间': '分词向量'}),
         te_di2[['教工号加密', '方向分词', '方向分词向量']].rename(columns={'方向分词': '分词', '方向分词向量': '分词向量'}),
         te_pa2[['教工号加密', '论文译文分词', '论文译文分词向量-时间']].rename(columns={'论文译文分词': '分词', '论文译文分词向量-时间': '分词向量'})])

    # 把教师向量（包含时间）保存到文件
    file_tea_time = 'qwen_teacher_vector_time_mean.h5'
    # 检查文件是否存在，若不存在则创建
    if not os.path.exists(file_tea_time):
        open(file_tea_time, 'w').close()
    try:
        # te_time_pool.to_csv(file_tea_time, index=False, encoding='utf-8-sig')
        te_time_pool.to_hdf(file_tea_time, key='teacher', mode='w')
        te_time_match.to_hdf(file_tea_time, key='teacher_match', mode='a')
        print("教师向量数据已成功保存到qwen_teacher_vector_time_mean.h5文件中。")
    except Exception as e:
        print(f"保存qwen_teacher_vector_time_mean.h5文件时出现错误: {e}")

# 根据key从 HDF5 文件中读取特定的 DataFrame
# loaded_df1 = pd.read_hdf(file_path, key='df1')
# loaded_df2 = pd.read_hdf(file_path, key='df2')
