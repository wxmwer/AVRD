import os
import pandas as pd
from openai import OpenAI

if __name__ == '__main__':
    dir_name = 'datasets/data'
    filename_course= 'Course_information_word_segmentation_And_Teacher_Course_Representation_Information.xlsx'
    filename_dir = 'Direction_Information_Word_segmentation_And_Teacher_Direction_Representation_Information.xlsx'
    filename_paper = 'Paper_Information_Word_segmentation_And_Teacher_Paper_Representation_Information.xlsx'
    filename_teacher = 'Teacher_Name_Colleage(one-to-many)_Information.xlsx'
    name_course = os.path.join(dir_name, filename_course)
    name_dir = os.path.join(dir_name, filename_dir)
    name_paper = os.path.join(dir_name, filename_paper)
    name_teacher = os.path.join(dir_name, filename_teacher)

    try:
        # 读取 Excel 文件
        teacher_course = pd.read_excel(name_course)
        teacher_dir = pd.read_excel(name_dir)
        teacher_paper = pd.read_excel(name_paper)
        da_teacher = pd.read_excel(name_teacher)

        te_unique = da_teacher[['教工号加密', 'SKJS']].drop_duplicates()
        # 生成教师ID（key）和姓名（value）典
        ID_NAME = dict(zip(te_unique['教工号加密'], te_unique['SKJS']))

        teacher_course_tfidf = pd.merge(te_unique, teacher_course.drop_duplicates(['教工号加密', 'KCMC'])[['教工号加密', 'KCMC']], on='教工号加密', how='left')
        teacher_course_tfidf = teacher_course_tfidf.groupby(['教工号加密', 'SKJS'])['KCMC'].agg('，'.join).reset_index(drop=False)

        teacher_dir_tfidf = pd.merge(te_unique, teacher_dir.drop_duplicates(['教工号加密', 'YJFX', 'ZXKMC', 'FXKMC'])
                                [['教工号加密', 'YJFX', 'ZXKMC', 'FXKMC']], on='教工号加密', how='left')
        teacher_dir_tfidf['方向信息'] = teacher_dir_tfidf[['YJFX', 'ZXKMC', 'FXKMC']].apply(
            lambda row: '，'.join([str(val) for val in row if pd.notna(val)]), axis=1)
        teacher_dir_tfidf = teacher_dir_tfidf.groupby(['教工号加密', 'SKJS'])['方向信息'].agg('，'.join).reset_index(drop=False)
        teacher_dir_tfidf['方向信息'] = teacher_dir_tfidf['方向信息'].replace('', '未查询到相关信息')


        teacher_paper_tfidf = pd.merge(te_unique, teacher_paper.drop_duplicates(['教工号加密', 'LWTM_译文'])[['教工号加密', 'LWTM_译文']]
                               , on='教工号加密', how='left')  # _译文 译文 译文
        teacher_paper_tfidf['LWTM_译文'] = teacher_paper_tfidf['LWTM_译文'].fillna('未查询到相关信息')
        #teacher_paper_tfidf = teacher_paper_tfidf[['LWTM']].apply(lambda row: '，'.join([str(val) for val in row if pd.notna(val)]), axis=1)
        teacher_paper_tfidf = teacher_paper_tfidf.groupby(['教工号加密', 'SKJS'])['LWTM_译文'].agg('，'.join).reset_index(drop=False)
        teacher_info= pd.merge(teacher_course_tfidf, teacher_dir_tfidf[['教工号加密', '方向信息']], on='教工号加密', how='left')
        teacher_info = pd.merge(teacher_info, teacher_paper_tfidf[['教工号加密', 'LWTM_译文']], on='教工号加密', how='left')
        teacher_info['LWTM_译文'] = teacher_info['LWTM_译文'].apply(lambda x: f'"{x}"')
        teacher_info = teacher_info.rename(columns={'SKJS':'姓名', 'KCMC':'课程信息', 'LWTM_译文':'论文信息' })
        teacher_keywords_list = teacher_info[['教工号加密', '课程信息', '方向信息', '论文信息']].values.tolist()
    except Exception as e:
        print(f"Error processing data: {e}")
    sample = '''
                这位老师讲授的课程包括数信息管理专题问题研究、商务数据分析、数据分析与挖掘、管理信息系统、网络营销理论与实务等课程，从事的研究方向包括商务智能与数据挖掘、能源系统建模、物流与供应链管理、管理科学与工程。
                发表的论文包括“出口贸易隐含碳变化驱动因素分析模型及应用”，“利用效率模型估计不良产出影子价格的研究进展“，“基于方向距离函数的中国航班延误机场效率测度“，
                “原油价格波动分析与预测模型综述“，“边际CO2减排成本：来自上海工业部门替代影子价格估计的结果“，“基于ICA的支持向量回归原油价格预测方法“，
                “可再生能源电力系统的可靠性和经济性评价“。
                '''
    sample_res = '''
                这位老师在数据分析、数据挖掘技术方面较为突出，从事的研究方向包括商务智能与数据挖掘、能源系统建模、物流与供应链管理等，主要的研究兴趣是使用数据分析与挖掘技术研究多个领域的科学问题，包括能能源价格预测与分析、能源政策制定、效率测度、碳排放建模等。
                '''
    except_list = []
    text_list = []
    teacher_ids = []

    client = OpenAI(api_key="sk-670d18e7cfb74f26b0d309a00740a18e", base_url="https://api.deepseek.com")
    with open('summaries_teacher_deepseek.txt', 'w', encoding='utf-8') as file:
        file.write("TeacherID\tSummary_Teacher\n")
        for i in range(len(teacher_keywords_list)):
            teacherid = teacher_keywords_list[i][0]
            teacher_keywords = teacher_keywords_list[i][1:]  # 每个学生的三条关键词对应的语句

            try:
                prompt = f"""
                                根据以下导师的所有信息，撰写一段总结，概括该导师的专业特长和科研方向。请确保总结简洁、全面，并能够突出导师的关键特点。尽量控制在250字以内，使用正式的专业语言。

                                具体信息在分隔符'--------------------------'和'--------------------------'内：
                                --------------------------
                                这位导师教授的课程{teacher_keywords[0]},主要的研究方向{teacher_keywords[1]}，发表的科研论文{teacher_keywords[2]}
                                --------------------------
                                请根据以上信息生成对该导师的总结。
                                """

                messages = [
                    {"role": "user",
                     "content": "作为一个研究生招生专家，你需要帮助教务负责人为学生推荐导师，快速总结导师的专业特长和研究兴趣，生成一段连贯性的话，不要包含两段及以上的话"},
                    {"role": "assistant", "content": "当然，为了更好地理解您的任务，请给我一些专业特长和研究兴趣生成的案例"},
                    {"role": "user", "content": f"这里是一个例子：\n{sample}\n这是生成的结果: \n{sample_res}"},
                    {"role": "assistant",
                     "content": "好的，我已经充分理解您的需求，请给我您需要生成的导师信息，我将为您生成满意的结果"},
                    {"role": "user", "content": prompt}
                ]

                # stu_ids.append(stuid)  # 保存 UserID 用于后续匹配输出
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=250,
                    stream=False
                )
                resp = response.choices[0].message.content
                resp = " ".join([line.strip() for line in resp.splitlines() if line.strip()])  # 合并成一段话
                file.write(f"{teacherid}\t{resp}\n")
                print(f'No.{i+1}: {teacherid} finish!')
            except Exception as e:
                print(f"Error processing output for TeacherID {teacherid}: {e}")
                except_list.append(teacherid)  # 记录出错的 stuID
                continue
    # 此时文件已经自动关闭，内容已保存
    print("文件写入完成，已自动关闭。")