import os
import asyncio
import aiofiles
import json
import time
import jwt
import requests
from tqdm import tqdm
import chardet



def read_file(file_path):
   try:
       with open(file_path, 'r',encoding='utf-8') as file:
           contents = file.read()
           return contents
   except FileNotFoundError:
       print(f"The file {file_path} was not found.")
       return None
   except IOError as e:
       print(f"An error occurred while reading the file: {e}")
       return None

def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    token = jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},)

    return token

api_key = os.environ.get("GLM_API_KEY", "")
if api_key:
    print(generate_token(api_key, 5400000))



def glm(file_path):
    # remove_blank_lines(file_path)
    # API URL 和 你的 API Key
    api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    api_key = os.environ.get("GLM_API_KEY", "")  # 从环境变量读取
    token = generate_token(api_key, 21600000)
    text = read_file(file_path)
    print(text)
    prompt = f"# CONTEXT（上下文） #你是一位西交利物浦大学资深的招生负责人，擅长从文本中提取关键信息并进行总结,请你阅读以下文本材料 <{text}>，该文本可能为中文或英文且包含多个知识点 # OBJECTIVE（目标） #请你对于文本中的每一句话或者每一段话进行总结，总结成每句话，句子由主谓宾构成，主语不能用代词，每个知识点可以有冗余，但是所有知识点可以概括整个文本，一定要对每个句子或者几个句子都进行总结，要全面,若存在数字内容应在总结中体现，知识点尽可能多，不能遗漏任意一段文本，对于存在并列关系的内容请分别概括总结，总结包含文本的全部细节，对于没有整段话或者没有完整句子，请你也进行提取总结，知识点要能完整体现出文本，如果文本中存在一段话是具体细节，请你作为单独知识点保留输出。生成的是连续的、不被中断的listofstring格式,生成中文回答 # STYLE（风格） #风格要求尽可能严谨学术并且全面 # TONE（语调） #说服性 # RESPONSE（响应） #你的回答只需要包含总结，以下是一个生成的具体例子：['总结的知识点1', '总结的知识点2', ... ，'总结的知识点n'],请严格按照这个格式生成，不要有换行符号等符号出现,每个知识点是一个字符串，知识点与知识点之间用逗号隔开而不被其他内容中断"

    # 准备请求的头部和数据
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }



    data = {
        "model": "glm-4-0520",
        "messages": [
            {
                "role": "system",
                "content": "你是一位资深的招生负责人，擅长从文本中提取关键信息并进行总结。"
            },
            {
                "role": "user",
                "content": prompt
            },
            ],
        "temperature": 0.01,
        "max_tokens": 4095,
        "top_p": 0.70

            }


    try:
        # 发送POST请求
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # 将触发HTTPError，如果状态码是400-499
    except requests.HTTPError as err:
        print(f"Request to {api_url} failed with status {err.response.status_code}: {err}")
    except requests.RequestException as e:
        print(f"Request to {api_url} failed with error: {e}")

    # 检查响应状态码
    if response.status_code == 200:
        # 成功获取数据
        data = response.json()  # 如果响应是JSON格式
        headers = response.headers  # 获取头部信息
    else:
        # 请求失败，打印错误信息
        print(f"Request failed with status {response.status_code}")
    try:
        ms = response.json()["choices"]
    except KeyError:
        print(KeyError)
        return []
        # 可以在这里添加错误处理逻辑
    #print(ms[0]['message']['content'])
    print(type(ms[0]['message']['content']))
    return ms[0]['message']['content']
def remove_blank_lines(file_path):
    """
    删除指定文件中的所有空行。

    参数:
    file_path (str): 要处理的文件的路径。
    """
    try:
        # 读取文件内容open(file_path, 'rb')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

        # 过滤掉空行
        filtered_lines = [line for line in lines if line.strip()]

        # 将过滤后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(filtered_lines)

        print(f"已删除 {file_path} 中的所有空行。")

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

def merge_and_sort_dicts(dict1, dict2):
    # 合并字典
    merged_dict = {**dict1, **dict2}
    # 按key排序
    sorted_dict = dict(sorted(merged_dict.items(), key=lambda item: item[0]))
    return sorted_dict

#glm(r'C:\Users\刘佳睿\Desktop\summer_surf\zipuai\file.txt')

folder_path = r'C:\Users\刘佳睿\Desktop\summer_surf\zipuai\text'
read_file(folder_path)
path_177 = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\text\1e51097984423f48a5329b2306f7e56a.txt"
list_177 = [path_177]
# fp = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\430.txt"
# glm(fp)
# print("++++++++++=")

mass = [69, 685]
lack_update = [221, 430, 818]
lack = [638, 642, 1155, 1158, 1301]
json_s = []
list_lack = []
list_still = [1099, 221, 818,962]
list_no_condition = [749, 228, 1396, 1355,1323, 1026, 101]
list_nonsence = [685, 177]
file_dict_path = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\file_dict.json"
final2_path = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final2.json"
final3_path = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final3.json"
new_text_path = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\new_text"
final_all_path = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_all.json"
with open(file_dict_path,'r', encoding="utf-8") as file:
    file_dict = json.load(file)
with open(final2_path, 'r', encoding="utf-8") as file:
    text_final2 = json.load(file)
with open(final3_path,'r', encoding="utf-8") as file:
    text_final3 = json.load(file)



merged_dict = {**text_final3, **text_final2}
# 按key排序
sorted_dict = dict(sorted(merged_dict.items(), key=lambda item: item[0]))

final_all_payh = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_all.json"
with open(final_all_payh, 'w', encoding='utf-8') as f:
    json.dump(sorted_dict, f, ensure_ascii=False, indent=4)
with open(final_all_payh,'r', encoding="utf-8") as file:
    final_all = json.load(file)

print(type(final_all))
print(type(file_dict))
print(file_dict)
print(len(sorted_dict))
for k, value in sorted_dict.items():
    print('the type of the k is ', type(k))
    if value is None or value == "" or value == []:

        print(file_dict.get(str(k)),' ',str(k))
        list_lack.append(k)

print(list_lack)
print(len(list_lack))
for c in list_lack:
    n = file_dict.get(str(c))
    p = rf"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\new_text\{n}"
    # print(f'{c} ',str(c) ,f" {n}\n",read_file(p))


for c in tqdm(list_no_condition):
    print('1111',type(c))
    h_sentance = []
    n = file_dict.get(str(c))
    try:
        p = rf"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\new_text\{n}"

        # remove_blank_lines(p)
        list_t = glm(p)
        for line in list_t.split('\n'):
            # print(line,"1111")
            h_sentance.append(line)
    except KeyError as e:
        h_sentance = []
    except AttributeError as e:
        h_sentance = []
    print(h_sentance)
    dict_t = {n: h_sentance}
    sorted_dict[str(c)] = h_sentance

# 对键进行排序，转换为整数进行比较
sorted_keys = sorted(sorted_dict.keys(), key=int)

# 根据排序后的键创建新的字典
sorted_dict_new = {key: sorted_dict[key] for key in sorted_keys}

json_s.append(sorted_dict_new)
# 创建新文件夹用于存储HTML文件
output_folder = r'C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_all'
file_path = output_folder  # 请替换为实际路径
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(json_s, f, ensure_ascii=False, indent=4)
#

# #
# async def process_file(filename,map,list_e,list_ty,list_r,list_t,json_s):
#
#         try:
#             # 检查文件扩展名是否为.txt
#             if (filename.endswith('.txt')):
#                 print("the filename is",filename)
#                 count = map.get(filename)
#                 print(f"the count is :{count}")
#                 # 构建完整的文件路径
#                 file_path = os.path.join(folder_path, filename)
#                 list_t = glm(file_path)
#
#         except IOError as e:
#             list_t = []
#             print(f"error information is {e}")
#             list_e.append(filename)
#         except Exception as e:
#             list_t = []
#             print(f"the information of exception is {e}")
#             list_e.append(filename)
#
#         for line in list_t.split('\n'):
#             # print(line,"1111")
#             h_sentance.append(line)
#             try:
#                 list_r = eval(list_t)
#             except TypeError as e:
#                 list_ty.append(list_r)
#                 list_r = []
#                 list_e.append(filename)
#                 print(f"报错信息： {e}")
#             except SyntaxError as e:
#                 list_ty.append(list_r)
#                 list_r = []
#                 list_e.append(filename)
#                 print(f"报错信息： {e}")
#
#             dict_t = {count: list_r}
#             json_s.append(dict_t)
#             count = count + 1
#
#
# async def main(folder_path, map):
#     list_e = []
#     list_ty = []
#     list_r = []
#     # 遍历文件夹
#     count = 0
#     list_t = []
#     json_s = []
#     tasks = []
#     for filename in os.listdir(folder_path):
#         if os.path.isfile(os.path.join(folder_path, filename)):
#             task = asyncio.create_task(process_file(os.path.join(folder_path, filename), map,list_e,list_ty,list_r,list_t,json_s))
#             tasks.append(task)
#
#     # 等待所有任务完成
#     await asyncio.gather(*tasks)
#     # 创建新文件夹用于存储HTML文件
#     output_folder = r'C:\Users\刘佳睿\Desktop\summer_surf\zipuai\fault\sentance'
#     file_path = output_folder  # 请替换为实际路径
#     with open(file_path, 'w', encoding='utf-8') as f:
#         json.dump(json_s, f, ensure_ascii=False, indent=4)
#     print(f"the files that cause error or exception are: {list_e}")
#     print(f"the files that caues the typeerror are: {list_ty}")

# 存储文件名和其对应的索引
file_index_map = {}

# 遍历文件夹内的所有文件
# for index, filename in enumerate(sorted(os.listdir(folder_path)), start=1):
#     if filename.endswith('.txt'):
#         # 将文件名映射到其索引
#         file_index_map[os.path.join(folder_path, filename)] = index
#
# output_folder = r'C:\Users\刘佳睿\Desktop\summer_surf\zipuai\map'
# file_path_map = output_folder  # 请替换为实际路径
# with open(file_path_map , 'w', encoding='utf-8') as f:
#     json.dump(file_index_map, f, ensure_ascii=False, indent=4)
# 打印结果
# for filename, index in file_index_map.items():
    #print(f"文件名: {os.path.join(folder_path, filename)}, 索引: {index}")
# 运行主函数

# asyncio.run(main(folder_path, file_index_map))
