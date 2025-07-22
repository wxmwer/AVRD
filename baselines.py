import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import jieba
import datetime
import warnings
warnings.filterwarnings("ignore")


# 文本信息设置时间率（授课学期、发表时间）
# 生成每位教师的信息生成时间列表（降序，如[2021,2018,2000]）
# 依据每位教师产生自身信息的时间（相对时间）设置时间率(如，[1，rate, rate^2])

# time_DataFrame={te_co,te_pa},time_Str={'SKXQ','FBSJ'}
def Time_rate(time_DataFrame, Rate=0.9, time_Str='SKXQ'):
    # 教师课程信息中，授课学期信息
    time_DataFrame
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
#过滤分词信息中不属于模型词典的词语
def Dictionary_filter(List,vocabs):
    List_out=[i for i in List if i in vocabs]
    return List_out

#将过滤后的文本词语列表转化为词向量
def List_vector(List,model):
    if len(List) != 0:
        vector=model.wv[List].mean(axis=0)
        return vector#.tolist()
    return np.zeros(model.wv.vectors.shape[1])#.tolist()


# 整合教师的时间信息向量
def Time_vector_sum(Vector_name, time_DataFrame, Model, Vocabs):
    # 生成带时间率权重的向量
    if Vector_name in ['课程分词向量', '论文分词向量']:  # 译文
        Vector_name1 = Vector_name[:-2]
        Vector_name2 = Vector_name + '-时间'

        ##过滤词语,放在这里运算量较大
        # time_DataFrame[Vector_name1]=time_DataFrame[Vector_name1].map(lambda x: Dictionary_filter(x,vocabs=Vocabs))
        # 利用model，生成词向量
        time_DataFrame[Vector_name] = time_DataFrame[Vector_name1].map(lambda x: List_vector(x, model=Model))

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
        pd_time_DataFrame = pd.DataFrame(columns=['教工号加密', Vector_name1, Vector_name])

        ##过滤词语,放在这里运算量较大
        # time_DataFrame[Vector_name1]=time_DataFrame[Vector_name1].map(lambda x: Dictionary_filter(x,vocabs=Vocabs))
        # 利用model，生成词向量
        time_DataFrame[Vector_name] = time_DataFrame[Vector_name1].map(lambda x: List_vector(x, model=Model))

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
                         Consider_no_time=False  # 是否计算没有时间率的方案
                         ):
    # 设置参数
    n = N  # 池化准则中的参数n,每个学生需求下截取的推荐名单长度
    pool_reco_size = Pool_reco_size  # 池化推荐人数
    match_reco_size = Match_reco_size  # 匹配推荐人数
    st_te_reco_all = []  # 不考虑时间率时，所有学生的教师推荐名单（ID）
    st_te_reco_all_time = []  # 考虑时间率时，所有学生的教师推荐名单（ID）
    # -------------------------------------------------------------------------------------------------------------------------------
    # 考虑时间的教师表示
    # if Consider_time:
    # 池化准则需要用到的教师数据格式
    te_time_pool = pd.merge(pd_te_co[['教工号加密', '课程分词向量-时间']], pd_te_di[['教工号加密', '方向分词向量']], on='教工号加密', how='left')
    te_time_pool = pd.merge(te_time_pool, pd_te_pa[['教工号加密', '论文分词向量-时间']], on='教工号加密', how='left')  # 译文
    # 匹配准则需要用到的教师数据格式
    te_time_match = pd.concat(
        [pd_te_co[['教工号加密', '课程分词', '课程分词向量-时间']].rename(columns={'课程分词': '分词', '课程分词向量-时间': '分词向量'}),
         pd_te_di[['教工号加密', '方向分词', '方向分词向量']].rename(columns={'方向分词': '分词', '方向分词向量': '分词向量'}),
         pd_te_pa[['教工号加密', '论文分词', '论文分词向量-时间']].rename(columns={'论文分词': '分词', '论文分词向量-时间': '分词向量'})])  # 译文 译文 译文 译文
    # 遍历每位学生
    for i in range(len(da_qu)):

        st_co_vect = da_qu.iloc[i, -3]
        st_di_vect = da_qu.iloc[i, -2]
        st_pa_vect = da_qu.iloc[i, -1]

        st_co_text = da_qu.iloc[i, -6]
        st_di_text = da_qu.iloc[i, -5]
        st_pa_text = da_qu.iloc[i, -4]

        # 池化推荐
        te_time_pool['课程-池化相似度'] = te_time_pool[['课程分词向量-时间', '方向分词向量', '论文分词向量-时间']].applymap(
            lambda x: sim_vec(x, st_co_vect)).max(axis=1)  # 译文
        te_time_pool['方向-池化相似度'] = te_time_pool[['课程分词向量-时间', '方向分词向量', '论文分词向量-时间']].applymap(
            lambda x: sim_vec(x, st_di_vect)).max(axis=1)  # 译文
        te_time_pool['论文-池化相似度'] = te_time_pool[['课程分词向量-时间', '方向分词向量', '论文分词向量-时间']].applymap(
            lambda x: sim_vec(x, st_pa_vect)).max(axis=1)  # 译文

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
    if Consider_no_time:
        # 池化准则需要用到的教师数据格式
        te_pool = pd.merge(pd_te_co[['教工号加密', '课程分词向量']], pd_te_di[['教工号加密', '方向分词向量']], on='教工号加密', how='left')
        te_pool = pd.merge(te_pool, pd_te_pa[['教工号加密', '论文分词向量']], on='教工号加密', how='left')  # 译文
        # 匹配准则需要用到的教师数据格式
        te_match = pd.concat([pd_te_co[['教工号加密', '课程分词', '课程分词向量']].rename(columns={'课程分词': '分词', '课程分词向量': '分词向量'}),
                              pd_te_di[['教工号加密', '方向分词', '方向分词向量']].rename(columns={'方向分词': '分词', '方向分词向量': '分词向量'}),
                              pd_te_pa[['教工号加密', '论文分词', '论文分词向量']].rename(
                                  columns={'论文分词': '分词', '论文分词向量': '分词向量'})])  # 译文 译文 译文 译文
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
            te_pool['课程-池化相似度'] = te_pool[['课程分词向量', '方向分词向量', '论文分词向量']].applymap(
                lambda x: sim_vec(x, st_co_vect)).max(axis=1)  # 译文
            te_pool['方向-池化相似度'] = te_pool[['课程分词向量', '方向分词向量', '论文分词向量']].applymap(
                lambda x: sim_vec(x, st_di_vect)).max(axis=1)  # 译文
            te_pool['论文-池化相似度'] = te_pool[['课程分词向量', '方向分词向量', '论文分词向量']].applymap(
                lambda x: sim_vec(x, st_pa_vect)).max(axis=1)  # 译文

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

        return [round(AUC0_value, 4), round(AUC1_value, 4), round(MAP_value, 4), round(HR_value, 4)], recom_true
    else:
        return [0.0, 0.0, 0.0, 0.0], []
if __name__ == '__main__':
    dir_name='datasets/data'
    dir_names=os.listdir(dir_name)
    datas = ['data_' + i[:4] for i in dir_names if i[i.rfind('.')+1:] in ['xlsx']]
    for fname,file in zip(datas,dir_names):
        print(fname,":",file)
        name = os.path.join(dir_name,file)
        exec(fname +'= pd.read_excel(name)')
    da_co=data_Cour.copy()
    da_di=data_Dire.copy()
    da_pa=data_Pape.copy()
    da_qu=data_3Que.copy()
    da_te=data_Teac.copy()
    # 经过Excel储存后读取，分词列的列表值外层被套上了字符串格式，去除一下
    da_co['课程分词'] = da_co['课程分词'].map(lambda x: eval(x))
    da_di[['YJFX_词语', 'ZXKMC_词语', 'FXKMC_词语', '方向分词']] = da_di[['YJFX_词语', 'ZXKMC_词语', 'FXKMC_词语', '方向分词']].applymap(
        lambda x: eval(x))
    da_pa[['论文译文分词', '论文分词']] = da_pa[['论文译文分词', '论文分词']].applymap(lambda x: eval(x))
    da_qu[['课程_分词', '方向_分词', '论文_分词']] = da_qu[['课程_分词', '方向_分词', '论文_分词']].applymap(lambda x: eval(x))
    # 统计每个训练样本的单词长度（预处理时），
    # 现将课程样本自身复制四倍长度
    # 研究方向样本自身复制两倍长度
    # 论文方向样本保持不变
    # da_co.loc[(da_co['length_课程']>1) & (da_co['length_课程']<7) , '课程分词']=da_co.loc[(da_co['length_课程']>1) & (da_co['length_课程']<7) , '课程分词']*4
    # da_di.loc[(da_di['length_方向']>1)          , '方向分词']=da_di.drop_duplicates(['YJFX','ZXKMC','FXKMC']).loc[(da_di['length_方向']>1)          , '方向分词']*2
    # da_pa.loc[(da_pa['length_论文']>2) & (da_pa['length_论文']<32) , '论文分词']=da_pa.loc[(da_pa['length_论文']>2) &\
    #                            (da_pa['length_论文']<32) , '论文分词'] #译文 译文 译文 译文 译文 译文
    # 提取可推荐教师的编号唯一值和姓名
    te_unique = da_te[['教工号加密', 'SKJS']].drop_duplicates()
    # 生成教师ID（key）和姓名（value）典
    ID_NAME = dict(zip(te_unique['教工号加密'], te_unique['SKJS']))
    # 所有学生的真实教师名单
    st_te_truth_all_name = da_qu[['指导老师1', '指导老师2', '指导老师3']].values.tolist()
    # 可推荐教师和学生的文本信息整合，与本文模型处理不同

    # 教师文本信息
    # 可推荐教师的课程信息（课程分词）
    te_co_tfidf = pd.merge(te_unique, da_co.drop_duplicates(['教工号加密', 'KCMC'])[['教工号加密', '课程分词']].rename(
        columns={'课程分词': '教师信息分词'})  # .drop_duplicates(['教工号加密','KCMC'])
                           , on='教工号加密', how='left')
    te_co_tfidf['教师信息分词'].fillna('[]', inplace=True)
    te_co_tfidf.loc[te_co_tfidf['教师信息分词'] == '[]', '教师信息分词'] = te_co_tfidf.loc[
        te_co_tfidf['教师信息分词'] == '[]', '教师信息分词'].map(lambda x: eval(x))
    te_co_tfidf = te_co_tfidf.groupby(['教工号加密', 'SKJS']).sum().reset_index(drop=False)
    te_co_tfidf['信息来源'] = '课程'

    # 可推荐教师的方向信息（方向分词）
    te_di_tfidf = pd.merge(te_unique,
                           da_di.drop_duplicates(['教工号加密', 'YJFX', 'ZXKMC', 'FXKMC'])[['教工号加密', '方向分词']].rename(
                               columns={'方向分词': '教师信息分词'})  # .drop_duplicates(['教工号加密','YJFX','ZXKMC','FXKMC'])
                           , on='教工号加密', how='left')
    te_di_tfidf['教师信息分词'].fillna('[]', inplace=True)
    te_di_tfidf.loc[te_di_tfidf['教师信息分词'] == '[]', '教师信息分词'] = te_di_tfidf.loc[
        te_di_tfidf['教师信息分词'] == '[]', '教师信息分词'].map(lambda x: eval(x))
    te_di_tfidf = te_di_tfidf.groupby(['教工号加密', 'SKJS']).sum().reset_index(drop=False)
    te_di_tfidf['信息来源'] = '方向'

    # 可推荐教师的论文信息（论文分词、发表时间）
    te_pa_tfidf = pd.merge(te_unique, da_pa.drop_duplicates(['教工号加密', 'LWTM'])[['教工号加密', '论文译文分词']].rename(
        columns={'论文译文分词': '教师信息分词'})  # .drop_duplicates(['教工号加密','LWTM_译文'])
                           , on='教工号加密', how='left')  # _译文 译文 译文
    te_pa_tfidf['教师信息分词'].fillna('[]', inplace=True)
    te_pa_tfidf.loc[te_pa_tfidf['教师信息分词'] == '[]', '教师信息分词'] = te_pa_tfidf.loc[
        te_pa_tfidf['教师信息分词'] == '[]', '教师信息分词'].map(lambda x: eval(x))
    te_pa_tfidf = te_pa_tfidf.groupby(['教工号加密', 'SKJS']).sum().reset_index(drop=False)
    te_pa_tfidf['信息来源'] = '论文'

    # 合并三类信息，并填充分词信息空值为[]
    te_info_tfidf = pd.concat([te_co_tfidf, te_di_tfidf, te_pa_tfidf])
    # te_inf_tfidf['教师信息分词'].fillna('[]',inplace=True)
    # te_info_tfidf.loc[te_info_tfidf['教师信息分词']=='[]','教师信息分词']=te_info_tfidf.loc[te_info_tfidf['教师信息分词']=='[]','教师信息分词'].map(lambda x: eval(x))
    ##求出每位教师的所有信息集合
    # te_text=te_info.groupby(['教工号加密','SKJS']).sum().reset_index(drop=False)

    # 学生文本信息
    da_qu_tfidf = da_qu.copy()
    # da_qu_tfidf['学生信息分词']=da_qu['课程_分词']+da_qu['方向_分词']+da_qu['论文_分词']
    # st_text=list(da_qu_tfidf['学生信息分词'])
    #print(te_unique.shape, te_co_tfidf.shape, te_di_tfidf.shape, te_pa_tfidf.shape, te_info_tfidf.shape)  # , te_text.shape, len(st_text))

    from gensim.models import TfidfModel
    from gensim.corpora import Dictionary
    from gensim import similarities


    def TFIDF_recom(st_info, te_info, dictionary, corpus, model, N=15, sim_size_pool=13, sim_size_matc=2,
                    match_recom=True):
        # st_info是DataFrame

        # 对稀疏向量建立索引
        index = similarities.SparseMatrixSimilarity(model[corpus], num_features=len(
            dictionary.keys()))  # MatrixSimilarity(model[corpus])#
        reco_ID_all_tfidf = []

        for i in range(len(st_info)):
            # 对测试文本分词
            # dic_text_list = list(jieba.cut(text))
            st_co_dic_text = st_info.iloc[i, -3]
            st_di_dic_text = st_info.iloc[i, -2]
            st_pa_dic_text = st_info.iloc[i, -1]
            # dic_text_list=st_co_dic_text+st_di_dic_text+st_pa_dic_text

            ## 制作测试文本的词袋
            st_co_doc_vec = dictionary.doc2bow(st_co_dic_text)
            st_di_doc_vec = dictionary.doc2bow(st_di_dic_text)
            st_pa_doc_vec = dictionary.doc2bow(st_pa_dic_text)
            # doc_text_vec = dictionary.doc2bow(dic_text_list)

            # 获取语料库每个文档中每个词的tfidf值，即用tfidf=model模型训练语料库
            # tfidf = TfidfModel(corpus)

            sim_co = index[model[st_co_doc_vec]]  # 相当于sim = index.get_similarities(model[doc_text_vec])
            sim_di = index[model[st_di_doc_vec]]
            sim_pa = index[model[st_pa_doc_vec]]
            # sim = index[model[doc_text_vec]]
            # 按照相似度来排序
            # sim_sorted = sorted(enumerate(sim,1), key=lambda x: -x[1]) #enumerate(sim,1)里面的1表示索引从1开始
            # next(iter(enumerate(sim,2))) 为 (2, 0.0)
            # 相当于sorted(enumerate(sim), key=lambda x: x[1], reverse=True

            # sim_sorted = sorted(zip(dataset_teacherID,sim), key=lambda x: -x[1])
            # sim_sorted = pd.DataFrame(zip(corpusIndex,sim),columns=['教工号加密','相似度']).sort_values(['教工号加密','相似度'],ascending=False).drop_duplicates(
            #                    ['教工号加密']).sort_values(['相似度'],ascending=False)
            corpusIndex = list(te_info['教工号加密'])
            corpusIndex1 = list(te_info['信息来源'])
            sim_co_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_co), columns=['教工号加密', '信息来源', '相似度'])
            sim_di_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_di), columns=['教工号加密', '信息来源', '相似度'])
            sim_pa_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_pa), columns=['教工号加密', '信息来源', '相似度'])

            # 池化准则
            sim_co_sorted = sim_co_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]
            sim_di_sorted = sim_di_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]
            sim_pa_sorted = sim_pa_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]

            sim_sorted_pool = pd.concat([sim_co_sorted, sim_di_sorted, sim_pa_sorted]).groupby(['教工号加密']).agg(
                ['count', max]).reset_index(drop=False).sort_values(
                [('相似度', 'count'), ('相似度', 'max')], ascending=False)
            sim_list_pool = list(sim_sorted_pool['教工号加密'])[:sim_size_pool]

            if match_recom:
                # 匹配准则
                st_sim_merge = pd.merge(sim_co_pd, sim_di_pd, on=['教工号加密', '信息来源'], how='left')
                st_sim_merge = pd.merge(st_sim_merge, sim_pa_pd, on=['教工号加密', '信息来源'], how='left')
                st_sim_merge['匹配相似度'] = st_sim_merge.iloc[:, -3:].max(axis=1)

                te_info['is_match'] = te_info['教师信息分词'].map(
                    lambda x: is_match(x, st_co_dic_text, st_di_dic_text, st_pa_dic_text))
                # 筛选出满足匹配推荐和名字未出现在池化推荐的教师信息
                sim_sorted_match = te_info.loc[
                    te_info['is_match'] & (~te_info['教工号加密'].isin(sim_list_pool)), ['教工号加密', '信息来源']]
                sim_sorted_match = pd.merge(sim_sorted_match, st_sim_merge[['教工号加密', '信息来源', '匹配相似度']],
                                            on=['教工号加密', '信息来源'], how='left')
                # 依据序列中频率和相似度排序
                sim_sorted_match = sim_sorted_match[['教工号加密', '匹配相似度']].groupby(['教工号加密']).agg(
                    ['count', max]).reset_index(drop=False).sort_values(
                    [('匹配相似度', 'count'), ('匹配相似度', 'max')], ascending=False)

                # 选前match_reco_size个教师作为匹配推荐
                if len(sim_sorted_match) > sim_size_matc:
                    # 最终的匹配推荐教师名单（ID）
                    sim_list_match = list(sim_sorted_match.iloc[:sim_size_matc, 0])
                else:
                    # 最终的匹配推荐教师名单（ID）
                    sim_list_match = list(sim_sorted_match.iloc[:, 0])
            else:
                sim_list_match = []

            sim_list = sim_list_pool + sim_list_match
            reco_ID_all_tfidf.append(sim_list)
            print('\rTFIDF模型，第{}个学生已完成'.format(i + 1), end='')
        return reco_ID_all_tfidf


    daset_teac_tfidf = list(te_info_tfidf['教师信息分词'])
    Dct_tfidf = Dictionary(daset_teac_tfidf)  # fit dictionary
    Corpus_tfidf = [Dct_tfidf.doc2bow(line) for line in daset_teac_tfidf]  # convert corpus to BoW format

    # TF-IDF也是一种数据表示文本的方式
    # model_tfidf = TfidfModel(Corpus_tfidf)
    # model_tfidf.save("pre-translation_results/2025-3-27-data170-model.tfidf")
    # print('pre-translation_results/2025-3-27-data170-model.tfidf 保存成功！')
    model_tfidf = TfidfModel.load("pre-translation_results/2025-3-27-data170-model.tfidf")
    reco_list_ID = TFIDF_recom(st_info=da_qu_tfidf, te_info=te_info_tfidf,
                               dictionary=Dct_tfidf, corpus=Corpus_tfidf, model=model_tfidf,
                               match_recom=True, sim_size_pool=13, sim_size_matc=2)
    reco_ID_all_tfidf, reco_true_tfidf = Evaluation_metrics(reco_ID_all=reco_list_ID,
                                                            trut_Name_all=st_te_truth_all_name, ID_Name=ID_NAME)
    print('\n', reco_ID_all_tfidf)
 

    #LSA(LSI)+TFIDF
    from gensim.models import LsiModel


    def LSA_TFIDF_recom(st_info, te_info, dictionary, corpus, model, N=15, sim_size_pool=13, sim_size_matc=2,
                        match_recom=True):
        # st_info是DataFrame

        # 对稀疏向量建立索引
        index = similarities.MatrixSimilarity(model[corpus])
        # index.save('/tmp/deerwester.index')
        # index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
        reco_ID_all_tfidf = []

        for i in range(len(st_info)):
            # 对测试文本分词
            # dic_text_list = list(jieba.cut(text))
            st_co_dic_text = st_info.iloc[i, -3]
            st_di_dic_text = st_info.iloc[i, -2]
            st_pa_dic_text = st_info.iloc[i, -1]
            # dic_text_list=st_co_dic_text+st_di_dic_text+st_pa_dic_text

            ## 制作测试文本的词袋
            st_co_doc_vec = dictionary.doc2bow(st_co_dic_text)
            st_di_doc_vec = dictionary.doc2bow(st_di_dic_text)
            st_pa_doc_vec = dictionary.doc2bow(st_pa_dic_text)
            # doc_text_vec = dictionary.doc2bow(dic_text_list)

            # 用Tfidf值-------------------*********----model_tfidf------上面的TFIDF数据和模型，全局变量--------------
            st_co_doc_vec = model_tfidf[st_co_doc_vec]
            st_di_doc_vec = model_tfidf[st_di_doc_vec]
            st_pa_doc_vec = model_tfidf[st_pa_doc_vec]

            # 获取语料库每个文档中每个词的tfidf值，即用tfidf=model模型训练语料库
            # tfidf = TfidfModel(corpus)

            sim_co = index[model[st_co_doc_vec]]  # 相当于sim = index.get_similarities(model[doc_text_vec])
            sim_di = index[model[st_di_doc_vec]]
            sim_pa = index[model[st_pa_doc_vec]]
            # sim = index[model[doc_text_vec]]
            # 按照相似度来排序
            # sim_sorted = sorted(enumerate(sim,1), key=lambda x: -x[1]) #enumerate(sim,1)里面的1表示索引从1开始
            # next(iter(enumerate(sim,2))) 为 (2, 0.0)
            # 相当于sorted(enumerate(sim), key=lambda x: x[1], reverse=True

            # sim_sorted = sorted(zip(dataset_teacherID,sim), key=lambda x: -x[1])
            # sim_sorted = pd.DataFrame(zip(corpusIndex,sim),columns=['教工号加密','相似度']).sort_values(['教工号加密','相似度'],ascending=False).drop_duplicates(
            #                    ['教工号加密']).sort_values(['相似度'],ascending=False)
            corpusIndex = list(te_info['教工号加密'])
            corpusIndex1 = list(te_info['信息来源'])
            sim_co_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_co), columns=['教工号加密', '信息来源', '相似度'])
            sim_di_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_di), columns=['教工号加密', '信息来源', '相似度'])
            sim_pa_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_pa), columns=['教工号加密', '信息来源', '相似度'])

            # 池化准则
            sim_co_sorted = sim_co_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]
            sim_di_sorted = sim_di_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]
            sim_pa_sorted = sim_pa_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]

            sim_sorted_pool = pd.concat([sim_co_sorted, sim_di_sorted, sim_pa_sorted]).groupby(['教工号加密']).agg(
                ['count', max]).reset_index(drop=False).sort_values(
                [('相似度', 'count'), ('相似度', 'max')], ascending=False)
            sim_list_pool = list(sim_sorted_pool['教工号加密'])[:sim_size_pool]

            if match_recom:
                # 匹配准则
                st_sim_merge = pd.merge(sim_co_pd, sim_di_pd, on=['教工号加密', '信息来源'], how='left')
                st_sim_merge = pd.merge(st_sim_merge, sim_pa_pd, on=['教工号加密', '信息来源'], how='left')
                st_sim_merge['匹配相似度'] = st_sim_merge.iloc[:, -3:].max(axis=1)

                te_info['is_match'] = te_info['教师信息分词'].map(
                    lambda x: is_match(x, st_co_dic_text, st_di_dic_text, st_pa_dic_text))
                # 筛选出满足匹配推荐和名字未出现在池化推荐的教师信息
                sim_sorted_match = te_info.loc[
                    te_info['is_match'] & (~te_info['教工号加密'].isin(sim_list_pool)), ['教工号加密', '信息来源']]
                sim_sorted_match = pd.merge(sim_sorted_match, st_sim_merge[['教工号加密', '信息来源', '匹配相似度']],
                                            on=['教工号加密', '信息来源'], how='left')
                # 依据序列中频率和相似度排序
                sim_sorted_match = sim_sorted_match[['教工号加密', '匹配相似度']].groupby(['教工号加密']).agg(
                    ['count', max]).reset_index(drop=False).sort_values(
                    [('匹配相似度', 'count'), ('匹配相似度', 'max')], ascending=False)

                # 选前match_reco_size个教师作为匹配推荐
                if len(sim_sorted_match) > sim_size_matc:
                    # 最终的匹配推荐教师名单（ID）
                    sim_list_match = list(sim_sorted_match.iloc[:sim_size_matc, 0])
                else:
                    # 最终的匹配推荐教师名单（ID）
                    sim_list_match = list(sim_sorted_match.iloc[:, 0])
            else:
                sim_list_match = []

            sim_list = sim_list_pool + sim_list_match
            reco_ID_all_tfidf.append(sim_list)
            print('\rLSA_TFIDF模型，第{}个学生已完成'.format(i + 1), end='')
        return reco_ID_all_tfidf


    # model_lsa_tfidf = LsiModel(model_tfidf[Corpus_tfidf], id2word=Dct_tfidf, num_topics=430, decay=1.0, power_iters=3, extra_samples=50)
    # model_lsa_tfidf.save('pre-translation_results/2025-3-27-num_top430-pow_ite3-ext_sam50+tfidf.lsi')
    model_lsa_tfidf = LsiModel.load(
        'pre-translation_results/2025-3-27-num_top430-pow_ite3-ext_sam50+tfidf.lsi')
    reco_list_ID_lsa_tfidf = LSA_TFIDF_recom(st_info=da_qu_tfidf, te_info=te_info_tfidf,
                                             dictionary=Dct_tfidf, corpus=Corpus_tfidf, model=model_lsa_tfidf,
                                             match_recom=True, sim_size_pool=13, sim_size_matc=2)
    reco_ID_all_lsa_tfidf, reco_true_lsa_tfidf = Evaluation_metrics(reco_ID_all=reco_list_ID_lsa_tfidf,
                                                                    trut_Name_all=st_te_truth_all_name, ID_Name=ID_NAME)
    print('\n', reco_ID_all_lsa_tfidf)
 

#LSA(LSI)
    def LSA_recom(st_info, te_info, dictionary, corpus, model, N=15, sim_size_pool=13, sim_size_matc=2,
                  match_recom=True):
        # st_info是DataFrame

        # 对稀疏向量建立索引
        index = similarities.MatrixSimilarity(model[corpus])
        # index.save('/tmp/deerwester.index')
        # index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
        reco_ID_all_tfidf = []

        for i in range(len(st_info)):
            # 对测试文本分词
            # dic_text_list = list(jieba.cut(text))
            st_co_dic_text = st_info.iloc[i, -3]
            st_di_dic_text = st_info.iloc[i, -2]
            st_pa_dic_text = st_info.iloc[i, -1]
            # dic_text_list=st_co_dic_text+st_di_dic_text+st_pa_dic_text

            ## 制作测试文本的词袋
            st_co_doc_vec = dictionary.doc2bow(st_co_dic_text)
            st_di_doc_vec = dictionary.doc2bow(st_di_dic_text)
            st_pa_doc_vec = dictionary.doc2bow(st_pa_dic_text)
            # doc_text_vec = dictionary.doc2bow(dic_text_list)

            ##用Tfidf值-----------------*********----model_tfidf------上面的TFIDF数据和模型，全局变量--------------
            # st_co_doc_vec = model_tfidf[st_co_doc_vec]
            # st_di_doc_vec = model_tfidf[st_di_doc_vec]
            # st_pa_doc_vec = model_tfidf[st_pa_doc_vec]

            # 获取语料库每个文档中每个词的tfidf值，即用tfidf=model模型训练语料库
            # tfidf = TfidfModel(corpus)

            sim_co = index[model[st_co_doc_vec]]  # 相当于sim = index.get_similarities(model[doc_text_vec])
            sim_di = index[model[st_di_doc_vec]]
            sim_pa = index[model[st_pa_doc_vec]]
            # sim = index[model[doc_text_vec]]
            # 按照相似度来排序
            # sim_sorted = sorted(enumerate(sim,1), key=lambda x: -x[1]) #enumerate(sim,1)里面的1表示索引从1开始
            # next(iter(enumerate(sim,2))) 为 (2, 0.0)
            # 相当于sorted(enumerate(sim), key=lambda x: x[1], reverse=True

            # sim_sorted = sorted(zip(dataset_teacherID,sim), key=lambda x: -x[1])
            # sim_sorted = pd.DataFrame(zip(corpusIndex,sim),columns=['教工号加密','相似度']).sort_values(['教工号加密','相似度'],ascending=False).drop_duplicates(
            #                    ['教工号加密']).sort_values(['相似度'],ascending=False)
            corpusIndex = list(te_info['教工号加密'])
            corpusIndex1 = list(te_info['信息来源'])
            sim_co_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_co), columns=['教工号加密', '信息来源', '相似度'])
            sim_di_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_di), columns=['教工号加密', '信息来源', '相似度'])
            sim_pa_pd = pd.DataFrame(zip(corpusIndex, corpusIndex1, sim_pa), columns=['教工号加密', '信息来源', '相似度'])

            # 池化准则
            sim_co_sorted = sim_co_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]
            sim_di_sorted = sim_di_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]
            sim_pa_sorted = sim_pa_pd.sort_values(['教工号加密', '相似度'], ascending=False).drop_duplicates(
                ['教工号加密']).sort_values(['相似度'], ascending=False).iloc[:N, :]

            sim_sorted_pool = pd.concat([sim_co_sorted, sim_di_sorted, sim_pa_sorted]).groupby(['教工号加密']).agg(
                ['count', max]).reset_index(drop=False).sort_values(
                [('相似度', 'count'), ('相似度', 'max')], ascending=False)
            sim_list_pool = list(sim_sorted_pool['教工号加密'])[:sim_size_pool]

            if match_recom:
                # 匹配准则
                st_sim_merge = pd.merge(sim_co_pd, sim_di_pd, on=['教工号加密', '信息来源'], how='left')
                st_sim_merge = pd.merge(st_sim_merge, sim_pa_pd, on=['教工号加密', '信息来源'], how='left')
                st_sim_merge['匹配相似度'] = st_sim_merge.iloc[:, -3:].max(axis=1)

                te_info['is_match'] = te_info['教师信息分词'].map(
                    lambda x: is_match(x, st_co_dic_text, st_di_dic_text, st_pa_dic_text))
                # 筛选出满足匹配推荐和名字未出现在池化推荐的教师信息
                sim_sorted_match = te_info.loc[
                    te_info['is_match'] & (~te_info['教工号加密'].isin(sim_list_pool)), ['教工号加密', '信息来源']]
                sim_sorted_match = pd.merge(sim_sorted_match, st_sim_merge[['教工号加密', '信息来源', '匹配相似度']],
                                            on=['教工号加密', '信息来源'], how='left')
                # 依据序列中频率和相似度排序
                sim_sorted_match = sim_sorted_match[['教工号加密', '匹配相似度']].groupby(['教工号加密']).agg(
                    ['count', max]).reset_index(drop=False).sort_values(
                    [('匹配相似度', 'count'), ('匹配相似度', 'max')], ascending=False)

                # 选前match_reco_size个教师作为匹配推荐
                if len(sim_sorted_match) > sim_size_matc:
                    # 最终的匹配推荐教师名单（ID）
                    sim_list_match = list(sim_sorted_match.iloc[:sim_size_matc, 0])
                else:
                    # 最终的匹配推荐教师名单（ID）
                    sim_list_match = list(sim_sorted_match.iloc[:, 0])
            else:
                sim_list_match = []

            sim_list = sim_list_pool + sim_list_match
            reco_ID_all_tfidf.append(sim_list)
            print('\rLSA模型，第{}个学生已完成'.format(i + 1), end='')
        return reco_ID_all_tfidf


    # model_lsa = LsiModel(Corpus_tfidf, id2word=Dct_tfidf, num_topics=480, decay=1.0, power_iters=1, extra_samples=100)
    # model_lsa.save('pre-translation_results/2025-3-27-num_top480-pow_ite1-ext_sam100.lsi')
    model_lsa = LsiModel.load('pre-translation_results/2025-3-27-num_top480-pow_ite1-ext_sam100.lsi')
    reco_list_ID_lsa = LSA_recom(st_info=da_qu_tfidf, te_info=te_info_tfidf,
                                 dictionary=Dct_tfidf, corpus=Corpus_tfidf, model=model_lsa,
                                 match_recom=True, sim_size_pool=13, sim_size_matc=2)
    reco_ID_all_lsa, reco_true_lsa = Evaluation_metrics(reco_ID_all=reco_list_ID_lsa,
                                                        trut_Name_all=st_te_truth_all_name, ID_Name=ID_NAME)
    print('\n', reco_ID_all_lsa)
 
#word2vec MODEL
    from gensim.models import word2vec

    sent_co = list(da_co.drop_duplicates('KCMC').loc[(da_co['length_课程'] > 1) & (da_co['length_课程'] < 7), '课程分词'] * 4)
    sent_di = list(da_di.drop_duplicates(['YJFX', 'ZXKMC', 'FXKMC']).loc[(da_di['length_方向'] > 1), '方向分词'] * 2)
    sent_pa = list(da_pa.drop_duplicates('LWTM').loc[(da_pa['length_论文'] > 2) & (da_pa['length_论文'] < 32), '论文分词'])  # 译文 译文 译文
    sentences = sent_co + sent_di + sent_pa
    ## 模型训练完成
    ##2022-08-26-14-45_model_seed18_hs_wind13_size300.model 0.88 20 13 2 0.6 0.2941 0.2484 0.3353 0 0 0 0 13 300
    model_ = word2vec.Word2Vec(sentences,min_count=1,window=13,sg=1,hs=1,seed=18,workers=16, vector_size=300)#vector_size or size
    model_.save('pre-translation_results/2025-3-27_model_seed18_hs_wind7_size340.model')
    model_ = word2vec.Word2Vec.load('pre-translation_results/2025-3-27_model_seed18_hs_wind7_size340.model')

    # 可推荐教师的课程信息（课程分词、授课学期），填充课程分词空值为[]，授课学期空值为'2020-2021'
    te_co = pd.merge(te_unique, da_co[['教工号加密', '课程分词', 'SKXQ']], on='教工号加密', how='left')
    te_co['课程分词'].fillna('[]', inplace=True)
    te_co.loc[te_co['课程分词'] == '[]', '课程分词'] = te_co.loc[te_co['课程分词'] == '[]', '课程分词'].map(lambda x: eval(x))
    te_co['SKXQ'].fillna('2020-2021', inplace=True)

    # 可推荐教师的方向信息（方向分词），填充方向分词空值为[]
    te_di = pd.merge(te_unique, da_di[['教工号加密', '方向分词']], on='教工号加密', how='left')
    te_di['方向分词'].fillna('[]', inplace=True)
    te_di.loc[te_di['方向分词'] == '[]', '方向分词'] = te_di.loc[te_di['方向分词'] == '[]', '方向分词'].map(lambda x: eval(x))

    # 可推荐教师的论文信息（论文分词、发表时间），填充论文分词空值为[]，发表时间空值为'2021-01-01'
    te_pa = pd.merge(te_unique, da_pa[['教工号加密', '论文分词', 'FBSJ']], on='教工号加密', how='left')  # 译文
    te_pa['论文分词'].fillna('[]', inplace=True)  # 译文
    te_pa.loc[te_pa['论文分词'] == '[]', '论文分词'] = te_pa.loc[te_pa['论文分词'] == '[]', '论文分词'].map(lambda x: eval(x))  # 译文 译文 译文 译文
    te_pa['FBSJ'].fillna('2021-01-01', inplace=True)

    # 模型读取
    model_past = word2vec.Word2Vec.load('pre-translation_results/2025-3-27_model_seed18_hs_wind7_size340.model')
    # 词典读取
    vocabs_past = model_past.wv.index_to_key
    vocabs_past = set(vocabs_past)

    # 教师分词
    te_co['课程分词'] = te_co['课程分词'].map(lambda x: Dictionary_filter(x, vocabs=vocabs_past))
    te_di['方向分词'] = te_di['方向分词'].map(lambda x: Dictionary_filter(x, vocabs=vocabs_past))
    te_pa['论文分词'] = te_pa['论文分词'].map(lambda x: Dictionary_filter(x, vocabs=vocabs_past))  # 译文 译文

    # 学生分词
    da_qu_model = da_qu.copy()
    da_qu_model[['课程_分词', '方向_分词', '论文_分词']] = da_qu_model[['课程_分词', '方向_分词', '论文_分词']].applymap(
        lambda x: Dictionary_filter(x, vocabs=vocabs_past))


    # 词向量模型训练完，仅调整时间率和长度参数的函数
    def Model_run_time(model_name, model_, vocabs_,
                       te_co_=te_co, te_di_=te_di, te_pa_=te_pa, da_qu_=da_qu_model,
                       Rate_=0.88, N_=20, Pool_reco_size_=13, Match_reco_size_=2,
                       Consider_no_time_=True, match_crit=True,
                       trut_Name_all_=st_te_truth_all_name, ID_Name_=ID_NAME):
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

        # 可推荐教师的文本最终表示向量（整合时间信息）
        # 加上时间率
        te_co1 = Time_rate(Rate=Rate_, time_DataFrame=te_co_, time_Str='SKXQ')
        te_di1 = te_di_.copy()
        te_pa1 = Time_rate(Rate=Rate_, time_DataFrame=te_pa_, time_Str='FBSJ')
        # 教师向量
        te_co2 = Time_vector_sum(Vector_name='课程分词向量', time_DataFrame=te_co1, Model=model_, Vocabs=vocabs_)
        te_di2 = Time_vector_sum(Vector_name='方向分词向量', time_DataFrame=te_di1, Model=model_, Vocabs=vocabs_)
        te_pa2 = Time_vector_sum(Vector_name='论文分词向量', time_DataFrame=te_pa1, Model=model_, Vocabs=vocabs_)  # 译文

        # 调查问卷中学生的向量表示
        da_qu_[['课程_分词向量', '方向_分词向量', '论文_分词向量']] = da_qu_[['课程_分词', '方向_分词', '论文_分词']].applymap(
            lambda x: List_vector(x, model=model_))

        reco_all = Recommended_criteria(pd_te_co=te_co2, pd_te_di=te_di2, pd_te_pa=te_pa2, da_qu=da_qu_,
                                        N=N_, Pool_reco_size=Pool_reco_size_, Match_reco_size=Match_reco_size_,
                                        Consider_matching_criteria=match_crit, Consider_no_time=Consider_no_time_)
        st_te_reco_all_time, st_te_reco_all = reco_all[0], reco_all[1]

        Eval_value_time, recom_true = Evaluation_metrics(reco_ID_all=st_te_reco_all_time, trut_Name_all=trut_Name_all_,
                                                         ID_Name=ID_Name_)
        Eval_value_NOtime, recom_true_NOtime = Evaluation_metrics(reco_ID_all=st_te_reco_all,
                                                                  trut_Name_all=trut_Name_all_, ID_Name=ID_Name_)

        return [Now_time, model_name, Rate_, N_, Pool_reco_size_,
                Match_reco_size_] + Eval_value_time + Eval_value_NOtime, recom_true, st_te_reco_all_time, st_te_reco_all, \
               te_co2, te_di2, te_pa2


    # 没有匹配准则
    model_name = '2025-3-27_model_seed18_hs_wind7_size340.model'
    model_ = word2vec.Word2Vec.load('pre-translation_results/2025-3-27_model_seed18_hs_wind7_size340.model')
    vocabs_ = model_.wv.index_to_key

    reco_ID_all_Model, reco_true_model, st_te_reco_all_time_model, st_te_reco_all_model, \
    pd_te_co1, pd_te_di1, pd_te_pa1 = Model_run_time(model_name, model_, vocabs_,
                                                     te_co_=te_co, te_di_=te_di, te_pa_=te_pa, da_qu_=da_qu_model,
                                                     Rate_=0.88, N_=20, Pool_reco_size_=13, Match_reco_size_=2,
                                                     Consider_no_time_=True, match_crit=False,
                                                     trut_Name_all_=st_te_truth_all_name, ID_Name_=ID_NAME)
    print('\n没有匹配准则时的结果：', reco_ID_all_Model[-8:])

    # 有匹配准则
    reco_ID_all_Model1, reco_true_model1, st_te_reco_all_time_model1, st_te_reco_all_model1, \
    pd_te_co, pd_te_di, pd_te_pa = Model_run_time(model_name, model_, vocabs_,
                                                  te_co_=te_co, te_di_=te_di, te_pa_=te_pa, da_qu_=da_qu_model,
                                                  Rate_=0.88, N_=20, Pool_reco_size_=13, Match_reco_size_=2,
                                                  Consider_no_time_=True, match_crit=True,
                                                  trut_Name_all_=st_te_truth_all_name, ID_Name_=ID_NAME)
    print('\n有匹配准则时的结果：', reco_ID_all_Model1[-8:])
