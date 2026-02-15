import openai
import json
from openai import OpenAI
import os
from tqdm import tqdm


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def generate_gpt4_response(text1, text2, select):
    prompt_0 = f"""
# CONTEXT #
你是西交利物浦大学招生办的主要负责人，主管招生办工作人员的业务能力。
# OBJECTIVE #
<answer>
{text1}
</answer>
<standard>
{text2}
<\standsrd>
根据<standard>与<\standsrd>之间的内容为参考，依据完整性的原则以语义为标准为<answer>与</answer>之间的答案进行简要评价，若答案中缺失参考内容中的内容则被视为不够完整。请先给出基于完整性的评价并进行评分，满分为5分，其中5分为表现最好，2.5分为中等水平，1分为最差，答案的长短、位置不能影响你的评分，不能因为答案长完整性得分就高，要根据内容语义的完整来打分。
# STYLE #
基于依据完整性与准确性进行评分
# TONE #
严谨客观
# RESPONSE #
你的回答必须严格参照一下示例输出：[“完整性”:”<a>score</a >”,“基于完整性的评价”:”<integrity>explain</integrity>”]。其中<integrity>与</integrity>之间的应该是基于完整性对此答案的评价，<a>与</a >应该是此答案完整性的得分。
"""
    prompt_1 = f'''
    # CONTEXT #
    你是西交利物浦大学招生办的主要负责人，主管招生办工作人员的业务能力。
    # OBJECTIVE #
    <answer>
    {text1}
    </answer>
    <standard>
    {text2}
    <\standsrd>
    根据<standard>与<\standsrd>之间的内容为参考，依据准确性的原则以语义为标准为<answer>与</answer>之间的答案进行简要评价，若答案中缺失参考内容中的内容则被视为不够准确。请先给出基于准确性的评价并进行评分，满分为5分，其中5分为表现最好，2.5分为中等水平，1分为最差，答案的长短、位置不能影响你的评分，不能因为答案长准确性得分就高，要根据内容语义的准确来打分。
    # STYLE #
    基于依据准确性与准确性进行评分
    # TONE #
    严谨客观
    # RESPONSE #
    你的回答必须严格参照一下示例输出：[“基于准确性的评价”:”<accuracy>explain</accuracy>”，“准确性”:”<a>score</a>”]。其中<accuracy>与</accuracy>之间的应该是基于准确性对此答案的简要评价，<a>与</a >应该是此答案准确性的得分。

    '''

    if select == 1:
        prompt = prompt_1
    else:
        prompt = prompt_0
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": 'system',
             "content": '你是西交利物浦大学招生办的主要负责人，主管招生办工作人员的业务能力，擅长为作答进行打分。'},
            {"role": 'user', "content": prompt},
        ],
        temperature=0.01,
        max_tokens=150
    )

    return response.choices[0].message.content

def evaluate(path, store_path, num):
    answer_address = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\answer.json"
    with open(f'{answer_address}', 'r', encoding='utf-8') as json_file:
        answer_data = json.load(json_file)
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in tqdm(filenames, position=0, desc='file', leave=False, colour='red'):
            for i in tqdm(range(2), position=1, desc=f'{filename}', leave=False, colour='blue'):
                # 读取 JSON 文件
                path_filename = os.path.join(dirpath, filename)
                with open(path_filename, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    responses = {}

                    for key in tqdm(data, position=2, desc='question', leave=False, colour='green'):
                        text1 = data[key]
                        text2 = answer_data.get(key, None)
                        if text2:  # 确保有相应的标准进行比较
                            response = generate_gpt4_response(text1=text2, text2=text1, select=num)
                            # print(response)
                            responses[key] = response  # 将结果存入字典中

                    # 将结果保存到一个 JSON 文件中
                    path_filename_result = os.path.join(store_path, filename)
                    with open(f"{path_filename_result}_{i}", 'w', encoding='utf-8') as outfile:
                        json.dump(responses, outfile, ensure_ascii=False, indent=4)

    return 0;

def calculate(path, key):
    dict = {}


    for dirpath, dirnames, filenames in os.walk(path):
        list_result = []
        print(filenames)
        list_case = []
        for filename in filenames:

            dict_lcase = {}
            path_filename = os.path.join(dirpath,filename)
            with open(path_filename, 'r', encoding='utf-8') as file:
                dict_score = json.load(file)
            sum = 0
            count = 0
            dict_lower = {}

            for k, v in dict_score.items():
                pattern = r'(准确性”:|完整性”:|准确性":).*?(\d+(?:\.\d+)?)'
                import re

                # 搜索匹配项
                match = re.search(pattern, v)
                # 如果找到匹配项，则打印结果
                if match:
                    first_number = match.group(2)
                    num = float(first_number)
                    sum += num
                    if num < 2:

                        dict_lower[k] = v
                    count += 1
                else:
                    count +=0
                    print(f'{k} : {v} in {filename} is fail, the count of it is {count}, the sum of which is {sum}')
                # list_v = v.split('，')
                # print(list_v[0])
                #
                # print(type(v))
                # import re
                # score = re.findall(r'(\d+(?:\.\d)?)', list_v[0])
                # print(score)

            dict_lcase[filename] = dict_lower
            dict_scase = {}

            list_case.append(dict_lower)
            dict_count = {}



            list_result.append(dict_lcase)
            average = sum / count
            dict[filename] = average
        list_result.append(dict)
        # 初始化一个空字典来存储每个问题的出现次数
        question_count = {}
        print(list_case)
        for dict_lcase in list_case:
            for key, value in dict_lcase.items():
                if key in question_count:
                    question_count[key] += 1
                else:
                    question_count[key] = 1


                # if isinstance(value, type(v_dict)):
                #     print('true')

        # 将问题按照出现次数从高到低排序
        sorted_question_count = sorted(question_count.items(), key=lambda x: x[1], reverse=True)

        # 打印每个问题的出现次数
        for question, count in sorted_question_count:
            print(f"'{question}': {count}")

        with open(f'{path}_result', 'w',encoding='utf-8') as file:
            json.dump(list_result, file, ensure_ascii=False, indent=4)
        # 遍历列表中的每个字典
        # for item in list_result:
        #     for key, value in item.items():
        #         print(key)
        #         if isinstance(value, dict):
        #             for question in value:
        #                 if question in question_count:
        #                     question_count[question] += 1
        #                 else:
        #                     question_count[question] = 1
        # # 将问题按照出现次数从高到低排序
        # sorted_question_count = sorted(question_count.items(), key=lambda x: x[1], reverse=True)

        # 打印每个问题的出现次数
        # for question, count in sorted_question_count:
        #     print(f"'{question}': {count}")
    return dict

# with open('answer_along2.json', 'r', encoding='utf-8') as json_file:
#     answer_data = json.load(json_file)
# 
# responses = {}
# 
# for key in data:
#     text1 = data[key]
#     text2 = answer_data.get(key, None)
#     if text2:  # 确保有相应的标准进行比较
#         response = generate_gpt4_response(text1=text2, text2=text1)
#         print(response)
#         responses[key] = response  # 将结果存入字典中
# 
# # 将结果保存到一个 JSON 文件中
# with open('responses_along2_wanzheng.json', 'w', encoding='utf-8') as outfile:
#     json.dump(responses, outfile, ensure_ascii=False, indent=4)
#
path_answer = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\answer"
path_1 = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_result\accuracy_new"
path_0 = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_result\integrity_new"
# evaluate(path_answer, path_0, 1)
evaluate(path_answer, path_1, 0)
print("所有响应已保存到 responses_new 文件中。")