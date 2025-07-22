# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import os
import datetime
import warnings

warnings.filterwarnings("ignore")


# 文本信息设置时间率（授课学期、发表时间）
# 生成每位教师的信息生成时间列表（降序，如[2021,2018,2000]）
# 依据每位教师产生自身信息的时间（相对时间）设置时间率(如，[1，rate, rate^2])

# time_DataFrame={te_co,te_pa},time_Str={'SKXQ','FBSJ'}


# 导入Excel词向量时出现的错误解决办法
def nan_vect(x):
    if isinstance(x, float):
        return np.zeros(D_MODEL)
    return x


# 相似度计算函数
def sim_vec(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


# 匹配准则使用前的判断条件
def is_match(list_t, list_s1, list_s2, list_s3):
    return set(list_s1).issubset(set(list_t)) or set(list_s2).issubset(set(list_t)) or set(list_s3).issubset(
        set(list_t))


# 匹配准则使用时的相似度计算
def sim_vec_match(vector_t, vector_s1, vector_s2, vector_s3):
    s1 = np.dot(vector_t, vector_s1) / (np.linalg.norm(vector_t) * np.linalg.norm(vector_s1))
    s2 = np.dot(vector_t, vector_s2) / (np.linalg.norm(vector_t) * np.linalg.norm(vector_s2))
    s3 = np.dot(vector_t, vector_s3) / (np.linalg.norm(vector_t) * np.linalg.norm(vector_s3))
    return pd.DataFrame([s1, s2, s3]).max()[0]


# 教师推荐（池化准则、匹配准则）
def Recommended_criteria(te_time_pool,  # 三个教师合并后向量{te_co2,te_di2,te_pa2}
                         te_time_match,
                         da_qu,
                         N=20,  # 池化准则中的参数n,每个学生需求下截取的推荐名单长度
                         Pool_reco_size=12,  # 池化推荐人数
                         Match_reco_size=3,  # 匹配推荐人数
                         Consider_matching_criteria=True,  # 是否考虑匹配准则
                         ):
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

    # 池化准则需要用到的教师数据格式
    # te_time_pool = pd.merge(pd_te_co[['教工号加密', '课程分词向量-时间']], pd_te_di[['教工号加密', '方向分词向量']], on='教工号加密', how='left')
    # te_time_pool = pd.merge(te_time_pool, pd_te_pa[['教工号加密', '论文译文分词向量-时间']], on='教工号加密', how='left')

    # 把教师向量（包含时间）保存到文件
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
    # te_time_match = pd.concat(
    #     [pd_te_co[['教工号加密', '课程分词', '课程分词向量-时间']].rename(columns={'课程分词': '分词', '课程分词向量-时间': '分词向量'}),
    #      pd_te_di[['教工号加密', '方向分词', '方向分词向量']].rename(columns={'方向分词': '分词', '方向分词向量': '分词向量'}),
    #      pd_te_pa[['教工号加密', '论文译文分词', '论文译文分词向量-时间']].rename(columns={'论文译文分词': '分词', '论文译文分词向量-时间': '分词向量'})])
    # 遍历每位学生
    for i in range(len(da_qu)):

        st_co_vect = da_qu.iloc[i, -3]
        st_di_vect = da_qu.iloc[i, -2]
        st_pa_vect = da_qu.iloc[i, -1]

        st_co_text = da_qu.iloc[i, -6]
        st_di_text = da_qu.iloc[i, -5]
        st_pa_text = da_qu.iloc[i, -4]

        # 池化推荐
        te_time_pool['课程-池化相似度'] = te_time_pool[['课程分词向量', '方向分词向量', '论文译文分词向量']].applymap(
            lambda x: sim_vec(x, st_co_vect)).max(axis=1)
        te_time_pool['方向-池化相似度'] = te_time_pool[['课程分词向量', '方向分词向量', '论文译文分词向量']].applymap(
            lambda x: sim_vec(x, st_di_vect)).max(axis=1)
        te_time_pool['论文-池化相似度'] = te_time_pool[['课程分词向量', '方向分词向量', '论文译文分词向量']].applymap(
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


def Model_run_time(te_time_pool, te_time_match, da_qu_, trut_Name_all_, ID_Name_, Rate_=0.9, N_=20, Pool_reco_size_=13,
                   Match_reco_size_=2, match_crit=True):
    Now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')  # -%H-%M
    ##模型训练
    # model_ = word2vec.Word2Vec(sentences=Sentences,min_count=1,window=Window,sg=1
    #                         ,hs=Hs,seed=Seed,workers=16,size=Size)#vector_size or size
    ##模型保存
    # model_path='{}_model_seed{}_hs_wind{}_size{}.model'.format(Now_time,Seed,Window,Size)
    # model_.save('..\\result_vector\\Parameter_tuning\\'+model_path)
    ##词典读取
    # vocabs_= model_.wv.index2word

    st_te_reco_all_time = Recommended_criteria(te_time_pool, te_time_match, da_qu_,
                                               N=N_, Pool_reco_size=Pool_reco_size_, Match_reco_size=Match_reco_size_,
                                               Consider_matching_criteria=match_crit)

    Eval_value_time, recom_true = Evaluation_metrics(reco_ID_all=st_te_reco_all_time, trut_Name_all=trut_Name_all_,
                                                     ID_Name=ID_Name_)

    return [Now_time, Rate_, N_, Pool_reco_size_, Match_reco_size_] + Eval_value_time, recom_true


def eval_vector(x):
    str_x = x.replace('\n', '').replace('  ', ' ').replace('  ', ' ').replace(' ', ',')  # .replace('  ',' ')
    str_x = str_x.replace(',,', ',').replace(',,', ',').replace(',,', ',')
    if str_x[1] == ',':
        return np.array(eval('[' + str_x[2:]))
    elif str_x[1] == '-':
        return np.array(eval(str_x))


if __name__ == '__main__':
    # 读取数据文件夹下所有数据
    base_url = '/root/autodl-tmp/qwen/'
    # dir_name = base_url + 'datasets/data'
    dir_name = 'datasets/data'
    dir_names = os.listdir(dir_name)
    datas = ['data_' + i[:4] for i in dir_names]

    for fname, file in zip(datas, dir_names):
        print(fname, ":", file)
        name = os.path.join(dir_name, file)
        exec(fname + '= pd.read_excel(name)')
    #da_co = data_Cour.copy()
    #da_di = data_Dire.copy()
    #da_pa = data_Pape.copy()
    #da_qu = data_2Que.copy()
    da_qu_new = data_3Que.copy()
    da_te = data_Teac.copy()
    # 经过Excel储存后读取，分词列的列表值外层被套上了字符串格式，去除一下
    #da_co['课程分词'] = da_co['课程分词'].map(lambda x: eval(x))
    #da_di[['YJFX_词语', 'ZXKMC_词语', 'FXKMC_词语', '方向分词']] = da_di[['YJFX_词语', 'ZXKMC_词语', 'FXKMC_词语', '方向分词']].applymap(
    #    lambda x: eval(x))
    #da_pa[['论文译文分词', '论文分词']] = da_pa[['论文译文分词', '论文分词']].applymap(lambda x: eval(x))
    #da_qu[['课程_分词', '方向_分词', '论文_分词']] = da_qu[['课程_分词', '方向_分词', '论文_分词']].applymap(lambda x: eval(x))

    # 可推荐教师的文本信息整合
    # 提取可推荐教师的编号唯一值和姓名
    te_unique = da_te[['教工号加密', 'SKJS']].drop_duplicates()

    # 把te_unique保存到文件
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

    # BERT句向量
    # 基本参数
    DROPOUT = 0.3
    D_MODEL = 512  # SimcseModel网络的输出，即词向量的最终维度
    # POOLING = 'cls' #'last-avg'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # base_url = '/root/autodl-tmp/qwen/LLM_result/'
    # 师生向量
    base_url = 'pretrained_model/'
    Save_Path = ['Qwen_mlp_3584_three_layers.pt']
    for i in range(len(Save_Path)):
        model_name = Save_Path[i]
        SAVE_PATH = base_url + Save_Path[i]


        # 调查问卷中学生的向量表示# 读取 HDF5 文件
        print('Start load qwen_stu_vector.h5')

        try:
            stu_file_path = 'datasets/qwen_stu_vector.h5'
            da_qu_1 = pd.read_hdf(stu_file_path, key='stu')
            print("qwen_stu_vector.h5成功读取到 DataFrame!")
        # print(df.head())
        except FileNotFoundError:
            print("错误: 文件未找到，请检查文件路径。")
        except KeyError:
            print("错误: 指定的键在 HDF5 文件中不存在。")
        except Exception as e:
            print(f"发生未知错误: {e}")
        print('Start load qwen_teacher_vector_time.h5')
        try:
            teacher_file_path = 'datasets/qwen_teacher_vector_time.h5'
            te_time_pool = pd.read_hdf(teacher_file_path, key='teacher')
            te_time_match = pd.read_hdf(teacher_file_path, key='teacher_match')
            print("qwen_teacher_vector_time.h5成功读取到 DataFrame!")
        # print(df.head())
        except FileNotFoundError:
            print("错误: 文件未找到，请检查文件路径。")
        except KeyError:
            print("错误: 指定的键在 HDF5 文件中不存在。")
        except Exception as e:
            print(f"发生未知错误: {e}")

        for i in range(len(da_qu_1)):
            tensor_3 = torch.tensor(da_qu_1.iloc[i, -3], dtype=torch.float)
            tensor_2 = torch.tensor(da_qu_1.iloc[i, -2], dtype=torch.float)
            tensor_1 = torch.tensor(da_qu_1.iloc[i, -1], dtype=torch.float)

            da_qu_1.iat[i, -3] = tensor_3.squeeze().cpu().detach().numpy()
            da_qu_1.iat[i, -2] = tensor_2.squeeze().cpu().detach().numpy()
            da_qu_1.iat[i, -1] = tensor_1.squeeze().cpu().detach().numpy()

        for i in range(len(te_time_pool)):
            tensor_3 = torch.tensor(te_time_pool.iloc[i, -3], dtype=torch.float)
            tensor_2 = torch.tensor(te_time_pool.iloc[i, -2], dtype=torch.float)
            tensor_1 = torch.tensor(te_time_pool.iloc[i, -1], dtype=torch.float)

            te_time_pool.iat[i, -3] = tensor_3.squeeze().cpu().detach().numpy()
            te_time_pool.iat[i, -2] = tensor_2.squeeze().cpu().detach().numpy()
            te_time_pool.iat[i, -1] = tensor_1.squeeze().cpu().detach().numpy()
        for i in range(len(te_time_match)):
            tensor_1 = torch.tensor(te_time_match.iloc[i, -1], dtype=torch.float)
            te_time_match.iat[i, -1] = tensor_1.squeeze().cpu().detach().numpy()

        rate_1 = 0.9
        reco_ID_all_Model, reco_true_model = Model_run_time(te_time_pool, te_time_match, da_qu_1,
                                                            st_te_truth_all_name, ID_NAME, Rate_=rate_1, N_=15,
                                                            Pool_reco_size_=13, Match_reco_size_=2, match_crit=False)
        print('\n', reco_ID_all_Model)
