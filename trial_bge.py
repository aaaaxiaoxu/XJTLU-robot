import string
import os
os.environ['TRANSFORMERS_CACHE'] = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\bge"
import jwt
import hashlib
import time
import qianfan
from transformers import AutoTokenizer, AutoModel
import torch
import json
from tqdm import tqdm
import numpy as np
import faiss
from searcher import Websearcher, find_closest_text_and_extract_content
import requests
from openai import OpenAI
import pickle
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]


path_token = ''
def generate_token(apikey: str, exp_seconds: int):
    global path_token
    path_token = r'D:\database\token.txt'
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
    # 将token保存到本地文件
    with open(path_token, 'w') as file:
        file.write(str(token))
    return token


def bge_emboding(str):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\bge\models--BAAI--bge-large-zh-v1.5\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116")
    model = AutoModel.from_pretrained(
        r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\bge\models--BAAI--bge-large-zh-v1.5\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116")
    model.eval()

    # Tokenize sentences
    max_length = 512
    encoded_input = tokenizer(str, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        #print(model_output, f"type of it is{type(model_output)}")
        #print(len(model_output[0][0]), f"type of it is{type(model_output[0][0])}")
        # Perform pooling. In this case, cls pooling.
    return model_output

class VectorArray(np.ndarray):
    def __int__(self, origin = None):
        super().__init__()
        self.origin = origin

    def __new__(cls, input_array, info=None, origin=None):
        # 创建数组实例
        obj = np.asarray(input_array).view(cls)
        # 存储额外的信息
        obj.info = info
        obj.origin = origin
        return obj

    def __array_finalize__(self, obj):
        # 这个函数在创建数组时被调用，可以用来初始化继承的属性
        if obj is None: return
        self.info = getattr(obj, 'info', None)
    def to_dict(self):
        return {"name": self, "value": self.origin}


def database_build(str):
    vector = {}
    database = []

    count = 0
    map = {}
    filename = fr"{str}"
    with open(filename, 'r', encoding="utf-8") as file:
        data = json.load(file)
        print(len(data))
        for key, value in tqdm(data.items()):
            list_map = []
            list = []
            if value == 0:
                v_read = bge_emboding('0').last_hidden_state[:, 0, :].squeeze(0)
                v_read_array = v_read.numpy()

                vector[count] = v_read_array.tolist()
                list.append(vector)
                list_map.append(count)
                count += 1

            else:
                for v in value:
                    v_read = bge_emboding(v).last_hidden_state[:, 0, :].squeeze(0)
                    v_read_array = v_read.numpy()

                    vector[count] = v_read_array.tolist()
                    list.append(vector)
                    list_map.append(count)
                    count += 1
            print(len(vector))
            map[key] = list_map
            print(key,'len of map is ', len(map))
        for k ,v in map.items():
            print(f"the v of K:{k} is ", v)
    # downlord = r"D:\database\vectoe_file_new.json"
    # with open(downlord, 'w', encoding='utf-8') as file:
    #     json.dump(vector, file, ensure_ascii=False, indent=4)
    dl_map = r"D:\database\map.json"
    with open(dl_map, 'w', encoding='utf-8') as file:
        json.dump(map, file, ensure_ascii=False, indent=4)
    return 0

#
# vector = {}
# database = []
# filename = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final(2).json"
# with open(filename,'r',encoding="utf-8") as file:
#     data = json.load(file)
#     for key, value in data.items():
#         list = []
#         for v in value:
#             v_read = VectorArray(bge_emboding(v).pooler_output, origin=key)
#             list.append(v)
#             database.append(v)
#         vector[key] = list
# downlord = r"D:\database"
# with open(downlord, 'w', encoding='utf-8') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)

# v = VectorArray(bge_emboding(sentences).last_hidden_state[0], origin=1)
# print(v[0], '\n', v.origin)


path_index = r"D:\database\index.faiss"
def add_index(database, dim):
    path_index = r"D:\database\index.faiss"
    index = faiss.IndexFlatL2(dim)
    v_all = []
    for k,v in database.items():
        v_modi = np.array(v, dtype='float32')
        v_all.append(v_modi)
    v_input = np.stack(v_all)
    index.add(v_input)
    faiss.write_index(index, path_index)
    return path_index


def search(str, k, index, map):
    v_read = bge_emboding(str).last_hidden_state[:, 0, :].squeeze(0)
    v = v_read.numpy()
    v = v.astype(np.float32)
    D,I = index.search(np.array([v]), k)
    list_I = I.tolist()
    web_list = []
    for value in list_I[0]:
        for k,v in map.items():
            for num in v:
                if value == num:
                    web_list.append(k)
    unique_list = list(set(web_list))
    return unique_list

# filename = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final3.json"
# database_build(filename)

# json_file_path = r"D:\database\vectoe_file.json"

# # 加载 JSON 数据
# with open(json_file_path, 'r') as file:
#     embeddings_dict = json.load(file)
# print(len(embeddings_dict))
# # 处理数据，移除第一个维度
# processed_embeddings = {}
# for key, embedding in embeddings_dict.items():
#     # 将列表转换为张量
#     tensor = torch.tensor(embedding)
#     # 移除第一个维度
#     squeezed_tensor = tensor.squeeze(0)
#     # 将张量转换回列表
#     processed_embedding = squeezed_tensor.tolist()
#     processed_embeddings[key] = processed_embedding
# print(len(processed_embeddings))
# # 保存处理后的数据
# with open(r'D:\database\processed_embeddings.json', 'w') as file:
#     db = json.dump(processed_embeddings, file, indent=4)
# with open(r'D:\database\vectoe_file.json', 'r') as file:
#     db = json.load(file)
# with open(r"D:\database\map.json", 'r') as file:
#     map = json.load(file)
# index = add_index(db, 1024)
# print(type(map["1"]))

# str = "介绍"
# vec = bge_emboding(str)
# web_list = search(str, 4, index, map)
# print(web_list)

def find_text(web_list, str):
    text = []
    filename = fr"{str}"
    with open(filename, 'r', encoding="utf-8") as file:
        data = json.load(file)
    for web in web_list:
        for k,v in data.items():
            if k == web:
                text.append(v)
    result_string = ' '
    for i in text:
        for j in i:
            result_string = result_string + "\n" +j
    return result_string

key_mom = os.environ.get("GLM_API_KEY", "")

def parse_sse_line(line):
    """解析SSE行数据，提取事件名称和数据"""
    if line.startswith("data:"):
        # 去除data:前缀，并返回剩余部分
        return line[5:].strip()

def answer_of_gml(str, num = 4):
    path_index = r"D:\database\index.faiss"
    path_token = r'D:\database\token.txt'
    text_path = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final3.json"
    # with open(r"D:\database\vectoe_file_new.json", 'r') as file:
    #     db = json.load(file)
    with open(r"D:\database\map.json", 'r') as file:
        map = json.load(file)
    index = faiss.read_index(path_index)
    web_list = search(str, num, index, map)
    info = find_text(web_list, text_path)
    # 网页搜索
    searcher = Websearcher(str, 15)
    # 提取所需内容
    extracted_texts = searcher.extract_content()
    text, url = find_closest_text_and_extract_content(extracted_texts, str)
    with open(path_token, 'r') as file:
        token = file.read()


    api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

#     prompt = f'''
# # CONTEXT #
# <query>
# {str.strip()}
# <\query>；
# # <sourse>
# # {text.strip()}
# # <\sourse>
# <info>
# {info.strip()}
# <\info>。
# <query>与<\query>之间的是咨询的招生问题。<info>与<\info>和<sourse>与<\sourse>之间的是关于西交利物浦大学的一些信息。
# # OBJECTIVE #
# 根据<info>与<\info>之间的信息以及<sourse>与<\sourse>之间的信息，回答<query>与<\query>之间的问题。只需要利用<info>与<\info>之间关于问题有关的信息对问题进行回答。<info>与<\info>之间与问题无关的信息则不要出现在回答之中，你的回答只需要包含对问题的解答。
# 请严格遵循以上要求！
# # STYLE #
# 请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
# # TONE #
# 严谨、流畅
# # RESPONSE #
# 你的回答将以咨询问题被回答的形式给出,你的回答只需要包含对问题的解答，请严格遵守这一要求。
# # AUDIENCE #
# 你的受众是来咨询西交利物浦大学信息的学生以及学生家长。text.strip()
# '''

    prompt_1 = f'''
    # 上下文 #
    <query>
    {str.strip()}
    <\query>；
    <sourse>
    {text.strip()}
    <\sourse>
    <info>
    {info.strip()}
    <\info>。
    <query>与<\query>之间的是咨询的招生问题。<info>与<\info>和<sourse>与<\sourse>之间的是关于西交利物浦大学的一些信息。
    # 目标 #
    根据<info>与<\info>之间的信息以及<sourse>与<\sourse>之间的信息，回答<query>与<\query>之间的问题。只需要利用<info>与<\info>之间关于问题有关的信息对问题进行回答。<info>与<\info>之间与问题无关的信息则不要出现在回答之中，你的回答只需要包含对问题的解答。
    请严格遵循以上要求！
    # 风格 #
    请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
    # 语气 #
    严谨、流畅
    # 回答 #
    你的回答将以咨询问题被回答的形式给出,你的回答不需要分点只需要包含回答的内容，请严格遵守这一要求。
    # 受众 #
    你的受众是来咨询西交利物浦大学信息的学生以及学生家长。
    '''



    data = {
        "model": "glm-4-0520",
        "tools": [{
            "type": 'web_search',
            "web_search": {
                "enable": False,
                "search_result": True,
                "search_query": f"西交利物浦大学 {str}"

            }
        }],
        "messages": [
            {
                "role": "system",
                "content": "你是一位资深的招生负责人，有丰富的工作经验."
            },

            {
                "role": "user",
                "content": prompt_1
            },
        ],
        "stream": True,
        "temperature": 0.01,
        "max_tokens": 4095,
        "top_p": 0.70,


    }

    try:
        # 发送POST请求
        response = requests.post(api_url, headers=headers, json=data, stream=True)

        # 获取Content-Type头部
        content_type = response.headers.get('Content-Type', 'Unknown')
        print(f"响应的格式是: {content_type}")

        # 检查响应状态码
        if response.status_code == 200:
            s = ''
            print(response.status_code)
            try:
                for line in response.iter_lines():
                    if line:
                        # 解码每行数据
                        decoded_line = line.decode('utf-8',errors='ignore')
                        # 解析SSE行
                        data = parse_sse_line(decoded_line)
                        if data:
                            json_data = json.loads(data)
                            # print(json_data['choices'][0]['delta']['content'], end='')
                            s+=json_data['choices'][0]['delta']['content']

            except Exception as e:
                print(" ")

            # # 成功获取数据
            # data = response.json()  # 如果响应是JSON格式
            # headers = response.headers  # 获取头部信息
            # print(type(data))
        else:
            # 请求失败，打印错误信息
            print(f"Request failed with status {response.status_code}")

        response.raise_for_status()  # 这行代码会抛出HTTPError异常，如果状态码不是2xx
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        print("Response content:", response.text)
        try:
            error_json = response.json()
            print("Error details:", error_json)
        except ValueError:
            pass  # Ignore if the response is not JSON format

    # try:
    #     ms = response.json()["choices"]
    # except KeyError:
    #     return []
    #
    # print(type(ms))
    # print(type(ms[0]['message']['content']))
    return s

# token = generate_token(key_mom, 21600000)

# with open(r"D:\database\vectoe_file_new.json", 'r') as file:
#     db = json.load(file)
# add_index(db, 1024)

# question = ". 硕士阶段西浦有入学奖学金吗？. 西浦有入学奖学金吗？"
# answer_of_gml(question, num=4)
# with open(r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\question(4).json",'r',encoding='utf-8') as  file:
#     dict_question = json.load(file)
# print('1111111111')
# print(len(dict_question))
# dict_glm_answer = {}
# for q,a_s in tqdm(dict_question.items()):
#     a_glm = answer_of_gml(q, num=5)
#     dict_glm_answer[q] = a_glm
# dict_glm_answer_json = []
# dict_glm_answer_json.append(dict_glm_answer)
# with open(r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\question_glm_final(broken).json",'w',encoding='utf-8') as file:
#     json.dump(dict_glm_answer_json, file, ensure_ascii=False, indent=4)

def answer_of_deepseek(str):
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
    prompt = f'''
    # 上下文 #
    <query>
    {str.strip()}
    <\query>；
    <query>与<\query>之间的是咨询的招生问题。
    # 目标 #
    根据回答<query>与<\query>之间的问题。你的回答只需要包含对问题的解答。
    请严格遵循以上要求！
    # 风格 #
    请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
    # 语气 #
    严谨、流畅
    # 回答 #
    你的回答将以咨询问题被回答的形式给出,你的回答不需要分点只需要包含回答的内容，请严格遵守这一要求。
    # 受众 #
    你的受众是来咨询西交利物浦大学信息的学生以及学生家长。
    '''
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位资深的招生负责人，有丰富的工作经验."},
            {"role": "user", "content": f"{prompt}", 'temperature':0.01},
        ],
        stream=False,

    )
    return response.choices[0].message.content
# answer_of_deepseek("介绍会计专业")

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    secret_key = os.environ.get("BAIDU_SECRET_KEY", "")
    assess_key = os.environ.get("BAIDU_ACCESS_KEY", "")
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": assess_key, "client_secret": secret_key}
    return str(requests.post(url, params=params).json().get("access_token"))

def answer_of_ernie(str):

    prompt = f'''
        # 上下文 #
        <query>
        {str.strip()}
        <\query>；
        <query>与<\query>之间的是咨询的招生问题。
        # 目标 #
        根据回答<query>与<\query>之间的问题。你的回答只需要包含对问题的解答。
        请严格遵循以上要求！
        # 风格 #
        请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
        # 语气 #
        严谨、流畅
        # 回答 #
        你的回答将以咨询问题被回答的形式给出,你的回答不需要分点只需要包含回答的内容，请严格遵守这一要求。
        # 受众 #
        你的受众是来咨询西交利物浦大学信息的学生以及学生家长。
        '''
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + get_access_token()
    # 注意message必须是奇数条
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload).json()

    return response['result']

# def answer_of_baichuan(str):
#     prompt = f'''
#             # 上下文 #
#             <query>
#             {str.strip()}
#             <\query>；
#             <query>与<\query>之间的是咨询的招生问题。
#             # 目标 #
#             根据回答<query>与<\query>之间的问题。你的回答只需要包含对问题的解答。
#             请严格遵循以上要求！
#             # 风格 #
#             请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
#             # 语气 #
#             严谨、流畅
#             # 回答 #
#             你的回答将以咨询问题被回答的形式给出,你的回答不需要分点只需要包含回答的内容，请严格遵守这一要求。
#             # 受众 #
#             你的受众是来咨询西交利物浦大学信息的学生以及学生家长。
#             '''
#     import torch
#     from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
#     model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='v1.0.5')
#     tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto",
#                                               trust_remote_code=True, torch_dtype=torch.float16)
#     model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",
#                                                  trust_remote_code=True, torch_dtype=torch.float16)
#     model.generation_config = GenerationConfig.from_pretrained(model_dir)
#     messages = []
#     messages.append({"role": "user", "content": f"{prompt}"})
#     response = model.chat(tokenizer, messages)
#     print(response)
#     return 0
# # answer_of_baichuan("hello")
#
# def answer_of_monkey(string):
#     prompt = f'''# 上下文 #
# <query>
# {string.strip()}
# <\query>；
# <query>与<\query>之间的是咨询的招生问题。
# # 目标 #
# 根据回答<query>与<\query>之间的问题。你的回答只需要包含对问题的解答。
# 请严格遵循以上要求！
# # 风格 #
# 请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
# # 语气 #
# 严谨、流畅
# # 回答 #
# 你的回答将以咨询问题被回答的形式给出,你的回答不需要分点只需要包含回答的内容，请严格遵守这一要求。
# # 受众 #
# 你的受众是来咨询西交利物浦大学信息的学生以及学生家长。
# '''
#
#     # 定义请求头，可能需要根据API的要求进行调整
#     headers = {
#         'Content-Type': 'application/json',
#         # 其他可能需要的headers
#     }
#     t = int(time.time())
#     timestamp = str(t)
#     ak = os.environ.get("MONKEY_AK", "")
#     sk = os.environ.get("MONKEY_SK", "")
#     signature_string = ak + '+' + sk + '+' + timestamp
#
#     # 定义请求体，这通常是一个JSON对象，具体内容取决于API的要求
#     data = {'apikey': f'{ak}',
#             'timestamp': f'{timestamp}',
#             'signature': hashlib.md5(signature_string.encode('utf-8')).hexdigest(),
#             'model': 'uclai-large',
#             "messages": [{"role": "user","content": f"{prompt}"}]
#             }
#
#     # # 将数据转换为JSON格式
#     # json_data = requests.json.dumps(data)
#
#     # 发送POST请求
#     response = requests.post('https://open-ka.mobvoi.com/api/chat/v2/chat', headers=headers, data=data)
#
#     # 检查请求是否成功
#     if response.status_code == 200:
#         # 处理响应数据
#         print(response.text)
#     else:
#         print('Failed to connect:', response.status_code)
#     return response

# answer_of_monkey('hello')
def answer_of_yi_34b(str):
    prompt = f'''# 上下文 #
    <query>
    {str.strip()}
    <\query>；
    <query>与<\query>之间的是咨询的招生问题。
    # 目标 #
    根据回答<query>与<\query>之间的问题。你的回答只需要包含对问题的解答。
    请严格遵循以上要求！
    # 风格 #
    请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
    # 语气 #
    严谨、流畅
    # 回答 #
    你的回答将以咨询问题被回答的形式给出,你的回答不需要分点只需要包含回答的内容，请严格遵守这一要求。
    # 受众 #
    你的受众是来咨询西交利物浦大学信息的学生以及学生家长。
    '''
    import openai
    from openai import OpenAI
    API_BASE = "https://api.lingyiwanwu.com/v1"
    API_KEY = os.environ.get("YI_API_KEY", "")

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=API_KEY,
        base_url=API_BASE)

    completion = client.chat.completions.create(
        model="yi-34b-chat-0205",
        messages=[{"role": "user", "content": f"{prompt}"}])
    return completion.choices[0].message.content

def answer_of_xinhuo(str):
    from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
    from sparkai.core.messages import ChatMessage
    prompt = f'''# 上下文 #
    <query>
    {str.strip()}
    <\query>；
    <query>与<\query>之间的是咨询的招生问题。
    # 目标 #
    根据回答<query>与<\query>之间的问题。你的回答只需要包含对问题的解答。
    请严格遵循以上要求！
    # 风格 #
    请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
    # 语气 #
    严谨、流畅
    # 回答 #
    你的回答将以咨询问题被回答的形式给出,你的回答不需要分点只需要包含回答的内容，请严格遵守这一要求。
    # 受众 #
    你的受众是来咨询西交利物浦大学信息的学生以及学生家长。
    '''
    # 星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
    SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
    # 星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
    SPARKAI_APP_ID = os.environ.get("SPARKAI_APP_ID", "")
    SPARKAI_API_SECRET = os.environ.get("SPARKAI_API_SECRET", "")
    SPARKAI_API_KEY = os.environ.get("SPARKAI_API_KEY", "")
    # 星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
    SPARKAI_DOMAIN = 'generalv3.5'

    if __name__ == '__main__':
        spark = ChatSparkLLM(
            spark_api_url=SPARKAI_URL,
            spark_app_id=SPARKAI_APP_ID,
            spark_api_key=SPARKAI_API_KEY,
            spark_api_secret=SPARKAI_API_SECRET,
            spark_llm_domain=SPARKAI_DOMAIN,
            streaming=False,
        )
        messages = [ChatMessage(
            role="user",
            content=f'{prompt}'
        )]
        handler = ChunkPrintHandler()
        a = spark.generate([messages], callbacks=[handler])
        return a.dict()['generations'][0][0]['text']

def answer_of_gemini(str):
    prompt = f'''# 上下文 #
        <query>
        {str.strip()}
        <\query>；
        <query>与<\query>之间的是咨询的招生问题。
        # 目标 #
        根据回答<query>与<\query>之间的问题。你的回答只需要包含对问题的解答。
        请严格遵循以上要求！
        # 风格 #
        请以一位西交利物浦大学资深的招生负责人与人交流的口吻回答问题。
        # 语气 #
        严谨、流畅
        # 回答 #
        你的回答将以咨询问题被回答的形式给出,你的回答不需要分点只需要包含回答的内容，请严格遵守这一要求。
        # 受众 #
        你的受众是来咨询西交利物浦大学信息的学生以及学生家长。
        '''
    # setup
    import google.generativeai as genai
    from IPython.display import display
    from IPython.display import Markdown

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""), transport='rest')

    # 查询模型
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"{prompt}")

    return response.text

# Refer to the document for workspace information: https://help.aliyun.com/document_detail/2746874.html

import random
from http import HTTPStatus
import dashscope


def call_stream_with_messages():
    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    messages = [
        {'role': 'user', 'content': 'Introduce the capital of China'}]
    responses = dashscope.Generation.call(
        'qwen2-72b-instruct',
        messages=messages,
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message"  format.
        stream=False,
        # output_in_full=True  # get streaming output incrementally
    )
    result = responses.output.choices[0].message.content
    print(responses.output.choices[0].message.content)
    print(type(responses.output.choices[0].message.content))
    return result

# call_stream_with_messages()


with open(r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\question_glm_final.json",'r',encoding='utf-8') as file:
    dict_question = json.load(file)
# with open(r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\question_final.json",'r',encoding='utf-8') as  file:
#     dict_question = json.load(file)
dict_deepseek_answer = {}
dict_ernie_answer = {}
dict_yi_answer = {}
dict_xh_answer = {}
dict_ge_answer = {}
# for q,a_s in tqdm(dict_question.items()):
#     # a_deepseek = answer_of_deepseek(q)
#     # a_ernie = answer_of_ernie(q)
#     # a_yi = answer_of_yi_34b(q)
#     a_geni = answer_of_gemini(q)
#     # a_xh = answer_of_xinhuo(q)
#     # dict_deepseek_answer[q] = a_deepseek
#     # dict_ernie_answer[q] = a_ernie
#     # dict_yi_answer[q] = a_yi
#     # dict_xh_answer[q] = a_xh
#     dict_ge_answer[q] = a_geni
#
# # dict_deepseek_answer_json = []
# # dict_ernie_answer_json = []
# dict_yi_answer_json = []
# dict_xh_answer_json = []
# dict_ge_answer_json = []
# # dict_deepseek_answer_json.append(dict_deepseek_answer)
# # dict_ernie_answer_json.append(dict_ernie_answer)
# # dict_yi_answer_json.append(dict_yi_answer)
# # dict_xh_answer_json.append(dict_xh_answer)
# dict_ge_answer_json.append(dict_ge_answer)
# # with open(r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\question_yi_final.json",'w',encoding='utf-8') as file:
# #     json.dump(dict_yi_answer_json, file, ensure_ascii=False, indent=4)
# # with open(r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\question_xh_final.json",'w',encoding='utf-8') as file:
# #     json.dump(dict_xh_answer_json, file, ensure_ascii=False, indent=4)
# with open(r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\question_ge_final.json",'w',encoding='utf-8') as file:
#     json.dump(dict_ge_answer_json, file, ensure_ascii=False, indent=4)

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

path_1 = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_result\accuracy"
path_0 = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_result\integrity"
path_accuracy_new = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_result\integrity_new"
path_integrity_new = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\final_result\accuracy_new"
print(calculate(path_accuracy_new,1))
print(calculate(path_integrity_new,1))

# list_path = [path_0,path_1]
# list_dict = []
# dict_average = {}
# dict_path = {'准确性':path_0, '完整性':path_1}
# for k, v in dict_path.items():
#     list_dict.append(calculate(v, k))
# print(list_dict)

