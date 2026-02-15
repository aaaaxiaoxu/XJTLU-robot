import re
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
os.environ['TRANSFORMERS_CACHE'] = r"C:\Users\刘佳睿\Desktop\summer_surf\zipuai\bge"
class Websearcher:
    def __init__(self, query, page_num):
        self.query = query
        self.page_num = page_num
        self.html = self.get_search_results()

    def get_search_results(self):
        url = f"https://www.baidu.com/s"
        params = {
            "wd": f"西交利物浦大学 {self.query}",
            "pn": self.page_num * 1
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        return response.text

    def extract_content(self):
        soup = BeautifulSoup(self.html, 'html.parser')
        results = {}

        for div in soup.find_all('div', class_='c-container'):
            # 跳过广告
            ad_span = div.find('a', class_='c-color-gray')
            if ad_span and ad_span.find('span') and ad_span.find('span').text == '广告':
                continue

            # 查找 has-tts='true' 的块
            tts_div = div.find('div', attrs={"has-tts": "true"})
            if not tts_div:
                continue

            # 获取 h3 标签下的 a 标签链接
            link_tag = tts_div.find('h3').find('a')
            if not link_tag:
                continue
            link = link_tag['href']

            # 获取符合条件的 span 块中的文本
            spans = div.find_all('span', class_=re.compile(r'content-right.*'))
            for span in spans:
                results[link] = span.text.strip()

        # 去重
        unique_results = {}
        for link, text in results.items():
            if link not in unique_results:
                unique_results[link] = text

        return unique_results


class Websearcher2:
    def __init__(self, query, page_num):
        self.query = query
        self.page_num = page_num
        self.html = self.get_search_results()

    def get_search_results(self):
        url = f"https://www.baidu.com/s"
        params = {
            "wd": f"{self.query}",
            "pn": self.page_num * 1
        }
        from random import randint

        USER_AGENTS = [
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
            "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
            "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
            "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
            "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
            "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
            "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
            "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
            "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
        ]

        random_agent = USER_AGENTS[randint(0, len(USER_AGENTS) - 1)]
        headers = {
            'User-Agent': random_agent,
        }

        # headers = {
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        # }
        try:
            response = requests.get(url, params=params, headers=headers)
            response.encoding = 'utf-8'

        except:
            for i in range(4):  # 循环去请求网站
                response = requests.get(url, params=params, headers=headers)
                response.encoding = 'utf-8'
                if response.status_code == 200:
                    break

        return response.text

    def extract_content(self):
        soup = BeautifulSoup(self.html, 'html.parser')
        results = {}

        for div in soup.find_all('div', class_='c-container'):
            # 跳过广告
            ad_span = div.find('a', class_='c-color-gray')
            if ad_span and ad_span.find('span') and ad_span.find('span').text == '广告':
                continue

            # 查找 has-tts='true' 的块
            tts_div = div.find('div', attrs={"has-tts": "true"})
            if not tts_div:
                continue

            # 获取 h3 标签下的 a 标签链接
            link_tag = tts_div.find('h3').find('a')
            if not link_tag:
                continue
            link = link_tag['href']

            # 获取符合条件的 span 块中的文本
            spans = div.find_all('span', class_=re.compile(r'content-right.*'))
            for span in spans:
                results[link] = span.text.strip()

        # 去重
        unique_results = {}
        for link, text in results.items():
            if link not in unique_results:
                unique_results[link] = text

        return unique_results



def encode_texts(model, tokenizer, texts):
    # Tokenize texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Take the mean of the last hidden states (mean pooling)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

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
    return model_output.last_hidden_state.mean(dim=1).numpy()
def find_closest_text_and_extract_content(data_dict, query):
    c = 0
    # Step 1: Convert dictionary values to a list and filter out invalid entries
    texts = [text for text in data_dict.values() if isinstance(text, str) and text.strip()]

    if not texts:
        c+=1
        # 如果 texts 为空，返回 None，表示没有有效的内容
        print(f'the count of errors is {c}')
        return 'None', None
    # Step 2: Convert each text to a vector using the BGE model
    vectors = bge_emboding(texts)

    # Step 3: Convert the query to a vector
    query_vector = bge_emboding([query])[0]

    # Step 4: Use faiss to find the closest vector to the query
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    distances, indices = index.search(np.array([query_vector]), 1)

    # Step 5: Get the closest text and its corresponding URL
    closest_text = texts[indices[0][0]]
    url = [key for key, value in data_dict.items() if value == closest_text][0]

    # Step 6: Access the URL and extract the natural language text
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove all link tags
    for a_tag in soup.find_all('a'):
        a_tag.decompose()

    # Extract natural language text
    text = soup.get_text(separator=' ', strip=True)

    return text, url
if __name__ == "__main__":
    # 使用示例
    websearcher = Websearcher("入学条件", 15)
    result = websearcher.extract_content()

    query = "西交利物浦大学入学条件"
    text, url = find_closest_text_and_extract_content(result, query)
    print(f"URL是: {url}")
    print(f"该URL提取的文本是: {text}")


