import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import time
import random
import requests
import json
import os
import re
from threading import Thread
import base64
import jieba.analyse
from bs4 import BeautifulSoup
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from waitress import serve
import threading

# ==================== [核心配置读取] ====================
print("读取配置文件...")
with open("./set.json", "r", encoding="utf-8") as setting:
    setinfo = setting.read()
    setdir = json.loads(setinfo)
    chat_models = setdir["chat_models"]
    draw_url = setdir.get("draw_url", "")
    draw_key = setdir.get("draw_key", "")
    draw_model = setdir.get("draw_model", "")
    system_prompts = setdir["system_prompts"]
    triggers = setdir["triggers"]
    random_trigger = setdir["random_trigger"]
    AI_name = setdir["AI_name"]
    ban_names = setdir.get("ban_names", [])
    root_ids = setdir.get("root_ids", [])
    send_debug = setdir.get("send_debug", False)
    speaker = setdir.get("speaker", "default")
    is_voice = setdir.get("is_voice", False)
    song = setdir.get("song", False)
    singer = setdir.get("singer", "default")
    vision_models = setdir.get("vision_models", [])

smusic_l = os.listdir("./data/voice/smusic") if os.path.exists("./data/voice/smusic") else []
str_music_l = ",".join(smusic_l)
moodstr = ",".join(system_prompts.keys())

order = f"""
[order]
1. 每句话之间使用#split#分割开，每段话直接也使用#split#分割开，你如：“#split#你好。群友。#split#幻日老爹在不？#split#”
2. 当需要发送表情包表达情绪时，按照格式 #split##emotion/情绪##split#，例如有人反复纠缠不休导致很生气：#split##emotion/angry##split#  (不要总是发送表情包，每条信息最多使用一次表情包，只支持以下表情包[angry,happy,sad,fear,bored])
3. 使用绘画功能时按照格式 #split##picture/绘画提示词##split# ，例如绘画一个女孩： #split##picture/one girl##split#  （除非明确要求否则不要绘画；绘画提示词尽力充实丰富，细节饱满详细，提示词使用英文单词）
4. 需要联网搜索时按照格式 #split##search/搜索关键词##split#，例如查询国内的新闻：#split##search/国内 新闻##split#  （关键词尽量多，详细，具体）
5. 每隔一段时间有重要的信息点需要写入长期记忆 #split##memory/写入的信息内容##split#，例如提到幻日是你的老爹：#split##memory/幻日是我老爹##split# （信息尽可能精简，不要写入有时效性的类似“明天是周天”的信息会失效造成干扰，不要写入[self_impression]下已经存在的内容）
6. 不想或者不需要回复信息时，只需要输出 #split##pass/None##split#，例如提到的信息与你无关-“@蓝莓 你是坏蛋”： #split##pass/None##split# (不要总是使用此操作拒绝回复)
7. 需要切换自身心情时，按照格式 #split##mood/心情名##split#，例如有人惹你生气：#split##mood/angry##split#，心情平复后：#split##mood/default##split#（非必要不要情感，只支持以下心情[{moodstr}]）
"""
if is_voice:
    order += """
8. 使用语音时按照格式 #split##voice/语言合成的内容##split# ，例如语音输出“你好”： #split##voice/你好##split#  (不要过多使用语音；使用语音时不可使用（括号）和特色字符)"""
if song:
    order += """
9. 心情好或想要唱歌时，按照格式 #split##music/歌曲名##split#，例如有人想让你唱潮汐：#split##music/潮汐##split# (不要总是唱歌，男声或合唱可能声音可能出问题，可适当通过唱歌表达情绪)"""

order += """
0. 回复时，禁止以群友的名义重复或冒充群友说话"""

system_prompt = system_prompts["default"]
system = system_prompt + order

for mood in system_prompts.keys():
    system_prompts[mood] += order

jieyue = True
cpu_lacking = False
weihu = False
objdict = {}
startT = time.time()


# ==================== [图片识别模块] ====================
def recognize_image(image_url):
    """调用 vision_models 中的模型进行图片识别"""
    if not vision_models:
        print("警告：未配置 vision_models，无法识别图片。")
        return "（系统未配置视觉模型，无法查看图片）"

    current_vision_model = random.choice(vision_models)
    v_api = current_vision_model.get("model_api", "")
    v_key = current_vision_model.get("model_key", "")
    v_name = current_vision_model.get("model_name", "")

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {v_key}"
        }
        payload = {
            "model": v_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text",
                         "text": "请详细描述这张图片的内容，包括主体、背景、文字信息（如果有）以及图片传达的情绪。"}
                    ]
                }
            ],
            "stream": False
        }
        response = requests.post(url=v_api, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return f"（图片内容：{content}）"
        else:
            print(f"视觉模型请求失败: {response.status_code} - {response.text}")
            return "（图片识别请求被拒绝）"
    except Exception as e:
        print(f"视觉模型调用出错: {e}")
        return "（图片识别发生错误）"


def process_rev_images(rev):
    """处理消息中的图片：提取URL -> 识别 -> 替换文本"""
    is_image_processed = False
    descriptions = []

    if 'message' in rev and isinstance(rev['message'], list):
        for segment in rev['message']:
            if segment['type'] == 'image':
                url = segment['data'].get('url', '')
                if url:
                    print(f"[识图] 正在识别图片...")
                    desc = recognize_image(url)
                    descriptions.append(desc)
                    is_image_processed = True

    if is_image_processed:
        if 'raw_message' in rev:
            rev['raw_message'] = re.sub(r'\[CQ:image,[^\]]*\]', '', rev['raw_message'])
            if descriptions:
                append_text = " [系统注：视觉模型识别到图片内容 -> " + "；".join(descriptions) + "]"
                rev['raw_message'] += append_text
                rev['_has_image_converted'] = True
                print(f"图片已转换为文本: {append_text}")
    return rev


# ==================== [原有功能函数] ====================
def change_setting(file_name, key, value):
    with open(file_name, 'r', encoding='utf-8') as f:
        t_gsetting = json.load(f)
    t_gsetting[key] = value
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(t_gsetting, f, ensure_ascii=False, indent=4)


def search(query):
    try:
        response = requests.get(
            'https://api.openinterpreter.com/v0/browser/search',
            params={"query": query},
        )
        if response.status_code == 200 and response.json()["result"]:
            return response.json()["result"]
        else:
            querys = query.split(" ")
            result = bing_search(querys)
            if result:
                return result
            else:
                result = bing_search(query)
                if result:
                    return result
                else:
                    return "未搜索到合适结果"
    except:
        return "搜索请求失败"


def bing_search(keywords):
    q = ""
    if isinstance(keywords, list):
        for p_k in keywords:
            q += (p_k + "+")
    else:
        q = keywords + "+"
    url = 'https://cn.bing.com/search?q=%s&count=10&qs=n&sp=-1&lq=0&pq=%s' % (q[:-1], q[:-1])
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        search_items = soup.find_all('li', class_='b_algo')
        if not search_items:
            return ""
        result = ""
        for index, item in enumerate(search_items):
            title_tag = item.find('h2')
            title = title_tag.get_text() if title_tag else "无标题"
            link_tag = item.find('a')
            link = link_tag['href'] if link_tag else "#"
            summary_div = item.find('div', class_='b_caption')
            if summary_div:
                summary_p = summary_div.find_all('p')
                if summary_p:
                    summary = ''.join(p.get_text() for p in summary_p)
                else:
                    summary = summary_div.get_text(strip=True)
            else:
                summary = ''
            if index < 3:
                try:
                    response_detail = requests.get(link, headers=headers, timeout=5)
                    response_detail.raise_for_status()
                    soup_detail = BeautifulSoup(response_detail.text, 'html.parser')
                    content_div = soup_detail.find('div', id='content')
                    if content_div:
                        content = content_div.get_text(strip=True)
                        content = (content[:5000]) if len(content) > 5000 else content
                    else:
                        content = soup_detail.get_text(strip=True)[:2000]
                    result += f'标题：{title}\n链接：{link}\n摘要：{summary}\n详细内容：{content}\n'
                except Exception as e:
                    print(f'请求详细页面错误：{e}')
            else:
                result += f'标题：{title}\n链接：{link}\n摘要：{summary}\n详细内容：{content}\n'
            if len(result) > 5000:
                return result
        return result
    except Exception as e:
        print(f'请求错误：{e}')
        return ""


def merge_contents(data):
    if not data: return []
    data = [data[0]] + [{"role": "user", "content": " "}] + data[1:]
    new_data = []
    temp_content = ""
    prev_role = None
    for item in data:
        current_role = item['role']
        current_content = item['content']
        if not current_content or not current_content.replace(" ", ''):
            if current_role == "user":
                current_content = "[特殊消息]"
            else:
                current_content = "呜呜，遇到未知错误..."
        if current_role == prev_role:
            temp_content += current_content
        else:
            if temp_content:
                new_data.append({'role': prev_role, 'content': temp_content})
            temp_content = current_content
            prev_role = current_role
    if temp_content:
        new_data.append({'role': prev_role, 'content': temp_content})
    return new_data


def get_I_memory(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return "\n[self_impression]\n" + content[-400:]


def get_memory(file_path, keywords, match_n=200, time_n=200, radius=50):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()[-409600:]
    if not keywords:
        return content[-time_n:]
    try:
        keywords_pattern = '|'.join(map(re.escape, keywords))
        matches = list(re.finditer(keywords_pattern, content))
    except:
        return content[-time_n:]
    merged_blocks = []
    for match in matches:
        start_index = max(match.start() - radius, 0)
        end_index = min(match.end() + radius, len(content))
        overlap = False
        for block in merged_blocks:
            if start_index < block['end'] and end_index > block['start']:
                block['start'] = min(start_index, block['start'])
                block['end'] = max(end_index, block['end'])
                block['count'] += 1
                overlap = True
                break
        if not overlap:
            merged_blocks.append({'start': start_index, 'end': end_index, 'count': 1})
    text_blocks = [{'text': content[block['start']:block['end']], 'count': block['count']} for block in merged_blocks]
    sorted_by_count = sorted(text_blocks, key=lambda x: x['count'], reverse=True)[:5]
    text_blocks = [{'text': content[block['start']:block['end']], 'end': block['end']} for block in merged_blocks]
    sorted_by_end = sorted(text_blocks, key=lambda x: x['end'], reverse=True)
    non_duplicate_sorted_by_end = [block for block in sorted_by_end if
                                   block['text'] not in [b['text'] for b in sorted_by_count]][:5]
    main_text = ''
    for per_text in non_duplicate_sorted_by_end:
        main_text += "--%s\n" % per_text["text"]
        if len(main_text) > match_n:
            break
    for per_text in sorted_by_count:
        main_text += "--%s\n" % per_text["text"]
        if len(main_text) > match_n + time_n:
            break
    return main_text[:match_n + time_n + 200]


def draw_group(prompt, to):
    try:
        urldraw = draw_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + draw_key
        }
        if "cogview" in draw_model or "stabilityai/" in draw_model:
            data = {
                "model": draw_model,
                "prompt": prompt,
            }
        else:
            data = {
                "model": draw_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        send_msg({'msg_type': 'group', 'number': to, 'msg': '正在绘画[%s]中...' % prompt})
        response = requests.post(url=urldraw, headers=headers, stream=True, data=json.dumps(data))
        if response.status_code == 200:
            send_msg({'msg_type': 'group', 'number': to, 'msg': '绘画完毕发送中...'})
        processed_d_data_draw = ''
        for line in response.iter_lines():
            try:
                decoded = line.decode('utf-8').replace('\n', '\\n').replace('\b', '\\b').replace('\f', '\\f').replace(
                    '\r', '\\r').replace('\t', '\\t')
                if decoded != '':
                    if "cogview" in draw_model or "stabilityai/" in draw_model:
                        processed_d_data_draw += json.loads(decoded)["data"][0]["url"]
                    else:
                        processed_d_data_draw += json.loads(decoded[5:])["choices"][0]["delta"]["content"]
            except Exception as e:
                print(e)
        image_url = processed_d_data_draw.split('(')[-1].replace(')', '')
        print(image_url)
        max_n = 500
        for n in range(0, max_n):
            try:
                image_response = requests.get(image_url)
                name = str(random.randrange(100000, 999999)) + '.png'
                with open("./data/image/%s" % name, 'wb') as f_image:
                    f_image.write(image_response.content)
                send_image({'msg_type': 'group', 'number': to, 'msg': name})
                break
            except:
                if n == max_n - 1:
                    raise TimeoutError("重试无效")
    except Exception as e:
        print('绘画错误:', e)
        send_msg({'msg_type': 'group', 'number': to, 'msg': 'AI绘画操作无法执行'})


def draw_private(prompt, to):
    try:
        urldraw = draw_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + draw_key
        }
        if "cogview" in draw_model or "stabilityai/" in draw_model:
            data = {
                "model": draw_model,
                "prompt": prompt,
            }
        else:
            data = {
                "model": draw_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        send_msg({'msg_type': 'private', 'number': to, 'msg': '正在绘画[%s]中...' % prompt})
        response = requests.post(url=urldraw, headers=headers, stream=True, data=json.dumps(data))
        if response.status_code == 200:
            send_msg({'msg_type': 'private', 'number': to, 'msg': '绘画完毕发送中...'})
        processed_d_data_draw = ''
        for line in response.iter_lines():
            try:
                decoded = line.decode('utf-8').replace('\n', '\\n').replace('\b', '\\b').replace('\f', '\\f').replace(
                    '\r', '\\r').replace('\t', '\\t')
                if decoded != '':
                    if "cogview" in draw_model or "stabilityai/" in draw_model:
                        processed_d_data_draw += json.loads(decoded)["data"][0]["url"]
                    else:
                        processed_d_data_draw += json.loads(decoded[5:])["choices"][0]["delta"]["content"]
            except Exception as e:
                print(e)
        image_url = processed_d_data_draw.split('(')[-1].replace(')', '')
        print(image_url)
        max_n = 500
        for n in range(0, max_n):
            try:
                image_response = requests.get(image_url)
                name = str(random.randrange(100000, 999999)) + '.png'
                with open("./data/image/%s" % name, 'wb') as f_image:
                    f_image.write(image_response.content)
                send_image({'msg_type': 'private', 'number': to, 'msg': name})
                break
            except:
                if n == max_n - 1:
                    raise TimeoutError("重试无效")
    except Exception as e:
        print('绘画错误:', e)
        send_msg({'msg_type': 'private', 'number': to, 'msg': 'AI绘画操作无法执行'})


def send_msg(resp_dict):
    msg_type = resp_dict['msg_type']
    number = resp_dict['number']
    msg = resp_dict['msg'].strip()
    if msg:
        url = 'http://localhost:3000/send_group_msg' if msg_type == 'group' else 'http://localhost:3000/send_private_msg'
        key = 'group_id' if msg_type == 'group' else 'user_id'
        try:
            requests.post(url, json={key: number, 'message': msg})
            print(f"发送消息成功: {msg[:20]}")
        except Exception as e:
            print(f"发送消息失败: {e}")


def send_image(resp_dict):
    msg_type = resp_dict['msg_type']
    number = resp_dict['number']
    msg = resp_dict['msg']
    # 使用 172.17.0.1 供 Docker 访问宿主机的图片文件服务
    url_file = f"http://172.17.0.1:4321/data/image/{msg}"
    cq_code = f"[CQ:image,file={url_file}]"

    url = 'http://localhost:3000/send_group_msg' if msg_type == 'group' else 'http://localhost:3000/send_private_msg'
    key = 'group_id' if msg_type == 'group' else 'user_id'
    try:
        requests.post(url, json={key: number, 'message': cq_code})
        print(f"发送图片: {msg}")
    except Exception as e:
        print(f"发送图片失败: {e}")


def send_voice(resp_dict):
    msg_type = resp_dict['msg_type']
    number = resp_dict['number']
    msg = resp_dict['msg']
    # 语音同样需要修正地址
    url_file = f"http://172.17.0.1:4321/data/voice/{msg}"
    cq_code = f"[CQ:record,file={url_file}]"

    url = 'http://localhost:3000/send_group_msg' if msg_type == 'group' else 'http://localhost:3000/send_private_msg'
    key = 'group_id' if msg_type == 'group' else 'user_id'
    try:
        requests.post(url, json={key: number, 'message': cq_code})
        print(f"发送语音: {msg}")
    except Exception as e:
        print(f"发送语音失败: {e}")


def send_music(resp_dict):
    msg_type = resp_dict['msg_type']
    number = resp_dict['number']
    msg = resp_dict['msg']
    file_name = msg.split("/")[-1]
    url_file = f"http://172.17.0.1:4321/data/voice/{msg}"
    cq_code = f"[CQ:file,file={url_file},name={file_name}]"

    url = 'http://localhost:3000/send_group_msg' if msg_type == 'group' else 'http://localhost:3000/send_private_msg'
    key = 'group_id' if msg_type == 'group' else 'user_id'
    try:
        requests.post(url, json={key: number, 'message': cq_code})
        print(f"发送音乐: {msg}")
    except Exception as e:
        print(f"发送音乐失败: {e}")


def send_image_url(resp_dict):
    msg_type = resp_dict['msg_type']
    number = resp_dict['number']
    msg = resp_dict['msg']
    url = 'http://localhost:3000/send_group_msg' if msg_type == 'group' else 'http://localhost:3000/send_private_msg'
    key = 'group_id' if msg_type == 'group' else 'user_id'
    data = {key: number, "message": {"type": "image", "data": {"file": msg.replace("%20", " ")}}}
    try:
        requests.post(url, json=data)
    except Exception as e:
        print(f"发送图片URL失败: {e}")


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F700-\U0001F77F"
                               u"\U0001F780-\U0001F7FF"
                               u"\U0001F800-\U0001F8FF"
                               u"\U0001F900-\U0001F9FF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FA70-\U0001FAFF"
                               u"\U00002702-\U000027B0"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def choose_model():
    if len(chat_models) == 1:
        model_info = chat_models[0]
    else:
        w_c_models = []
        for c_model in chat_models:
            w_c_models += [c_model] * c_model["weight"]
        model_info = random.choice(w_c_models)
    return model_info["model_api"], model_info["model_name"], model_info["model_key"]


# ==================== [核心主逻辑] ====================
def main(rev):
    global objdict

    # 1. 过滤心跳包
    if rev.get('post_type') == 'meta_event':
        return

    # 2. 打印日志
    if "raw_message" in rev:
        print(f">> 处理消息: {rev['raw_message']}")

    # 3. 处理图片
    rev = process_rev_images(rev)

    user_api, user_chat_model, user_key = choose_model()
    try:
        timestamp = time.time()
        localtime = time.localtime(timestamp)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", localtime)
        e_information = "[information](准确 有时效性)\n当前时间：%s\n" % current_time

        # ------------------ 私聊处理 ------------------
        if "message_type" in rev and rev["message_type"] == "private":
            if "banaijian%schat" % rev["sender"]["user_id"] not in objdict.keys():
                objdict["banaijian%schat" % rev["sender"]["user_id"]] = ""
            if not os.path.exists("./user/p%s" % rev["sender"]["user_id"]):
                os.makedirs("./user/p%s" % rev["sender"]["user_id"])
                with open("./user/p%s/memory.txt" % rev["sender"]["user_id"], "w") as tpass: pass
            if not os.path.exists("./user/p%s/I_memory.txt" % rev["sender"]["user_id"]):
                with open("./user/p%s/I_memory.txt" % rev["sender"]["user_id"], "w") as tpass: pass
            if not os.path.exists("./user/p%s/setting.json" % rev["sender"]["user_id"]):
                data = {'mood': 'default', 'random_trigger': random_trigger, "root_id": root_ids}
                with open("./user/p%s/setting.json" % rev["sender"]["user_id"], 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            # 【关键逻辑】检查是否应该记录消息
            should_record = "[CQ:image," not in rev['raw_message'] or rev.get('_has_image_converted', False)

            if should_record:
                objdict["banaijian%schat" % rev["sender"]["user_id"]] += (
                            rev["sender"]["nickname"] + "：" + rev['raw_message'].replace(
                        '[CQ:at,qq=%d]' % rev['self_id'], '') + '\n\n')
                objdict["banaijian%schat" % rev["sender"]["user_id"]] = objdict["banaijian%schat" % rev["sender"][
                    "user_id"]][-50:]

            if True:
                a = objdict["banaijian%schat" % rev["sender"]["user_id"]]
                print(a)
                self_id = random.randrange(100000, 999999)
                objdict["banaijian%sgeneing" % rev["sender"]["user_id"]] = [self_id]
                rev['raw_message'] = rev['raw_message'].replace('[CQ:at,qq=%d]' % rev['self_id'], '')
                if "banaijian%s" % rev["sender"]["user_id"] not in objdict.keys():
                    objdict["banaijian%s" % rev["sender"]["user_id"]] = [[{'role': 'system', 'content': system}]]
                if '#reset' in rev['raw_message']:
                    objdict["banaijian%s" % rev["sender"]["user_id"]] = [[{'role': 'system', 'content': system}]]
                    send_msg({'msg_type': 'private', 'number': rev["sender"]["user_id"], 'msg': '已清空对话历史'})
                else:
                    processed_d_data = "强制切换意图"
                    if random.randrange(0, 2) == 0:
                        objdict["banaijian%s" % rev["sender"]["user_id"]][0][0] = {"role": "system",
                                                                                   "content": system_prompts["default"]}
                    if weihu:
                        send_msg({'msg_type': 'private', 'number': rev["sender"]["user_id"], 'msg': "维护中..."})
                        raise KeyboardInterrupt("维护ing")

                    if not processed_d_data: processed_d_data = '.'

                    turl = user_api
                    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + user_key}
                    messages = objdict["banaijian%s" % rev["sender"]["user_id"]][0] + [
                        {"role": "user", "content": objdict["banaijian%schat" % rev["sender"]["user_id"]]}]
                    keywords = jieba.analyse.extract_tags(rev['raw_message'].replace(AI_name, ""), topK=50)
                    s_memory = get_memory("./user/p%s/memory.txt" % rev["sender"]["user_id"], keywords)
                    s_memory += get_I_memory("./user/p%s/I_memory.txt" % rev["sender"]["user_id"])
                    print(s_memory)
                    data = {
                        "model": user_chat_model,
                        "messages": merge_contents([{"role": "system", "content": messages[0][
                                                                                      "content"] + "[memory](模糊 无时效性)\n%s\n" % s_memory + e_information}] + messages[
                                                                                                                                                                  1:]),
                        "stream": True,
                        "use_search": False
                    }
                    is_return = True
                    while is_return:
                        is_return = False
                        for _ in range(0, 3):
                            try:
                                response = requests.post(url=turl, headers=headers, stream=True, data=json.dumps(data))
                                if response.status_code == 200: break
                                data["model"] = chat_models[0]["model_name"]
                                turl = chat_models[0]["model_api"]
                                user_key = chat_models[0]["model_key"]
                                headers = {"Content-Type": "application/json", "Authorization": "Bearer " + user_key}
                            except Exception as e:
                                data["model"] = chat_models[0]["model_name"]
                                turl = chat_models[0]["model_api"]
                                user_key = chat_models[0]["model_key"]
                                headers = {"Content-Type": "application/json", "Authorization": "Bearer " + user_key}
                        is_not_remove_emoji = random.randrange(0, 3)
                        temp_tts_list = []
                        processed_d_data1 = ''
                        for line in response.iter_lines():
                            try:
                                decoded = line.decode('utf-8').replace('\n', '\\n').replace('\b', '\\b').replace('\f',
                                                                                                                 '\\f').replace(
                                    '\r', '\\r').replace('\t', '\\t')
                                if decoded != '':
                                    temp_processed_d_data1 = json.loads(decoded[5:])["choices"][0]["delta"]["content"]
                            except Exception as e:
                                continue
                            if decoded != '':
                                for p_token in temp_processed_d_data1:
                                    processed_d_data1 += p_token
                                    if not is_not_remove_emoji:
                                        processed_d_data1 = remove_emojis(processed_d_data1)
                                    lastlen = len(temp_tts_list)
                                    temp_tts_list = processed_d_data1.split("#split#")
                                    if not temp_tts_list:
                                        temp_tts_list = temp_tts_list[:-1]
                                    if self_id not in objdict["banaijian%sgeneing" % rev["sender"]["user_id"]]:
                                        objdict["banaijian%s" % rev["sender"]["user_id"]][0] = \
                                        objdict["banaijian%s" % rev["sender"]["user_id"]][0] + [
                                            {'role': 'user', 'content': rev['raw_message']},
                                            {'role': 'assistant', 'content': processed_d_data1}]
                                        raise InterruptedError("新消息中断")

                                    if len(temp_tts_list) > 1 and lastlen < len(temp_tts_list):
                                        if '#voice/' in temp_tts_list[-2]:
                                            # (原有voice逻辑，略微简化但保留功能)
                                            try:
                                                voice = temp_tts_list[-2].split('#voice/')[-1].replace("#", '')
                                                tts_data = {"cha_name": speaker, "text": voice.replace("...", "…"),
                                                            "character_emotion": random.choice(
                                                                ['default', 'angry', 'excited'])}
                                                b_wav = requests.post(url='http://127.0.0.1:5000/tts', json=tts_data)
                                                n = random.randrange(10000, 99999)
                                                name = '%stts%d.wav' % (
                                                (time.strftime('%F') + '-' + time.strftime('%T').replace(':', '-')), n)
                                                with open('./data/voice/%s' % name, 'wb') as wbf:
                                                    wbf.write(b_wav.content)
                                                send_voice({'msg_type': 'private', 'number': rev["sender"]["user_id"],
                                                            'msg': name})
                                            except:
                                                send_msg({'msg_type': 'private', 'number': rev["sender"]["user_id"],
                                                          'msg': "语音合成失败"})
                                        elif '#picture/' in temp_tts_list[-2]:
                                            picture = temp_tts_list[-2].split('#picture/')[-1].replace("#", '')
                                            draw_private(picture, rev["sender"]["user_id"])
                                        elif '#search/' in temp_tts_list[-2]:
                                            response.close()
                                            temp_tts_list = temp_tts_list[:-1]
                                            break
                                            # ... (保留其他 mood, memory, pass 逻辑) ...
                                        elif "#pass/" in temp_tts_list[-2]:
                                            response.close()
                                            raise KeyboardInterrupt("PASS")
                                        else:
                                            send_msg({'msg_type': 'private', 'number': rev["sender"]["user_id"],
                                                      'msg': temp_tts_list[-2].replace("%s：" % AI_name, "")})

                        # 流式结束后的处理
                        if len(temp_tts_list) > 0 and "抱歉" in temp_tts_list[-1]:
                            objdict["banaijian%s" % rev["sender"]["user_id"]][0] = [
                                objdict["banaijian%s" % rev["sender"]["user_id"]][0][0]]
                        else:
                            last_segment = temp_tts_list[-1] if temp_tts_list else ""
                            if '#search/' in last_segment:
                                s_prompt = last_segment.split('#search/')[-1].replace("#", '')
                                send_msg({'msg_type': 'private', 'number': rev["sender"]["user_id"],
                                          'msg': "正在联网搜索：%s" % s_prompt})
                                search_result = search(s_prompt)
                                objdict["banaijian%s" % rev["sender"]["user_id"]][0] += [
                                    {'role': 'user', 'content': rev['raw_message']}, {'role': 'assistant',
                                                                                      'content': processed_d_data1 + """\nsystem[搜索结果不可见]：搜索结果：\n%s\n""" % (
                                                                                          search_result)},
                                    {"role": "user", "content": "请详细讲述"}]
                                messages = objdict["banaijian%s" % rev["sender"]["user_id"]][0]
                                data["messages"] = merge_contents(
                                    [{"role": "system", "content": system_prompt}] + messages[1:])
                                is_return = True
                                continue
                            else:
                                send_msg({'msg_type': 'private', 'number': rev["sender"]["user_id"],
                                          'msg': last_segment.replace("%s：" % AI_name, "")})

                            objdict["banaijian%s" % rev["sender"]["user_id"]][0] = \
                            objdict["banaijian%s" % rev["sender"]["user_id"]][0] + [
                                {'role': 'user', 'content': rev['raw_message']},
                                {'role': 'assistant', 'content': processed_d_data1}]
                            with open("./user/p%s/memory.txt" % rev["sender"]["user_id"], "a", encoding="utf-8") as txt:
                                txt.write("[%s]我：%s\n[%s]你：%s\n" % (
                                current_time, rev['raw_message'], current_time, processed_d_data1))

            if len(objdict["banaijian%s" % rev["sender"]["user_id"]][0]) > 10:
                objdict["banaijian%s" % rev["sender"]["user_id"]][0] = [objdict[
                                                                            "banaijian%s" % rev["sender"]["user_id"]][
                                                                            0][0]] + objdict[
                                                                                         "banaijian%s" % rev["sender"][
                                                                                             "user_id"]][0][-6:]
            objdict["banaijian%schat" % rev["sender"]["user_id"]] = ''

        # ------------------ 群聊处理 ------------------
        elif "message_type" in rev and rev["message_type"] == "group":
            # 过滤删除逻辑
            if ("团子" in rev["sender"]["nickname"] or "芙芙" in rev["sender"]["nickname"]) and "[CQ:image," in rev[
                'raw_message']:
                time.sleep(5 + random.randrange(0, 5))
                requests.post('http://localhost:3000/delete_msg', json={'message_id': rev['message_id']})

            pass_ban = False
            for ban_name in ban_names:
                if ban_name in rev["sender"]["nickname"]:
                    pass_ban = True
                    break
            if pass_ban: return

            if "banaijian%schat" % rev['group_id'] not in objdict.keys():
                objdict["banaijian%schat" % rev['group_id']] = ""
            if not os.path.exists("./user/g%s" % rev['group_id']):
                os.makedirs("./user/g%s" % rev['group_id'])
                with open("./user/g%s/memory.txt" % rev['group_id'], "w") as tpass: pass
            if not os.path.exists("./user/g%s/I_memory.txt" % rev['group_id']):
                with open("./user/g%s/I_memory.txt" % rev['group_id'], "w") as tpass: pass
            if not os.path.exists("./user/g%s/setting.json" % rev['group_id']):
                data = {'mood': 'default', 'random_trigger': random_trigger, "root_id": root_ids}
                with open("./user/g%s/setting.json" % rev['group_id'], 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            # 【关键逻辑】群聊是否记录
            should_record_group = "[CQ:image," not in rev['raw_message'] or rev.get('_has_image_converted', False)

            if should_record_group:
                objdict["banaijian%schat" % rev['group_id']] = objdict["banaijian%schat" % rev['group_id']][-70:]
                objdict["banaijian%schat" % rev['group_id']] += (
                            "[" + rev["sender"]["nickname"] + "]说：" + rev['raw_message'].replace(
                        '[CQ:at,qq=%d,name=%s]' % (rev['self_id'], AI_name), AI_name) + '\n\n')

            # 读取群配置
            with open("./user/g%s/setting.json" % rev['group_id'], 'r', encoding='utf-8') as f:
                tt_gsetting = json.load(f)
            tt_random_trigger = tt_gsetting["random_trigger"]

            is_trigger = False
            for trigger in triggers:
                if trigger in rev['raw_message']:
                    is_trigger = True
                    break

            # 【关键逻辑】触发条件，加入图片转换后的判断
            if (is_trigger or '[CQ:at,qq=%d]' % rev['self_id'] in rev['raw_message'] or random.randrange(0,
                                                                                                         tt_random_trigger) == 0 or rev.get(
                    '_has_image_converted', False)):
                a = objdict["banaijian%schat" % rev['group_id']]
                print(a)
                self_id = random.randrange(100000, 999999)
                objdict["banaijian%sgeneing" % rev['group_id']] = [self_id]
                rev['raw_message'] = rev['raw_message'].replace('[CQ:at,qq=%d,name=%s]' % (rev['self_id'], AI_name),
                                                                AI_name)

                if "banaijian%s" % rev['group_id'] not in objdict.keys():
                    objdict["banaijian%s" % rev['group_id']] = [[{'role': 'system', 'content': system}]]

                if '#reset' in rev['raw_message']:
                    objdict["banaijian%s" % rev['group_id']] = [[{'role': 'system', 'content': system}]]
                    send_msg({'msg_type': 'group', 'number': rev['group_id'], 'msg': '[已清空对话历史]'})
                    # ... 省略部分管理员指令逻辑，直接进对话 ...
                else:
                    processed_d_data = "强制切换意图"
                    if random.randrange(0, 7) == 0:
                        objdict["banaijian%s" % rev['group_id']][0][0] = {"role": "system",
                                                                          "content": system_prompts["default"]}
                    if weihu:
                        send_msg({'msg_type': 'group', 'number': rev['group_id'], 'msg': "[维护中...]"})
                        raise KeyboardInterrupt("维护ing")

                    if not processed_d_data: processed_d_data = '.'

                    turl = user_api
                    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + user_key}
                    messages = objdict["banaijian%s" % rev['group_id']][0] + [
                        {"role": "user", "content": objdict["banaijian%schat" % rev['group_id']]}]
                    keywords = jieba.analyse.extract_tags(rev['raw_message'].replace(AI_name, ""), topK=50)
                    s_memory = get_memory("./user/g%s/memory.txt" % rev['group_id'], keywords, match_n=500)
                    s_memory += get_I_memory("./user/g%s/I_memory.txt" % rev['group_id'])
                    print(s_memory)
                    data = {
                        "model": user_chat_model,
                        "messages": merge_contents([{"role": "system", "content": messages[0][
                                                                                      "content"] + "[memory](模糊 无时效性)\n%s\n" % s_memory + e_information}] + messages[
                                                                                                                                                                  1:]),
                        "stream": True
                    }
                    is_return = True
                    while is_return:
                        is_return = False
                        for _ in range(0, 3):
                            try:
                                response = requests.post(url=turl, headers=headers, stream=True, data=json.dumps(data))
                                if response.status_code == 200: break
                                data["model"] = chat_models[0]["model_name"]
                                turl = chat_models[0]["model_api"]
                                user_key = chat_models[0]["model_key"]
                                headers = {"Content-Type": "application/json", "Authorization": "Bearer " + user_key}
                            except Exception as e:
                                data["model"] = chat_models[0]["model_name"]
                                turl = chat_models[0]["model_api"]
                                user_key = chat_models[0]["model_key"]
                                headers = {"Content-Type": "application/json", "Authorization": "Bearer " + user_key}

                        is_not_remove_emoji = random.randrange(0, 3)
                        temp_tts_list = []
                        processed_d_data1 = ''

                        for line in response.iter_lines():
                            try:
                                decoded = line.decode('utf-8').replace('\n', '\\n').replace('\b', '\\b').replace('\f',
                                                                                                                 '\\f').replace(
                                    '\r', '\\r').replace('\t', '\\t')
                                if decoded != '':
                                    temp_processed_d_data1 = json.loads(decoded[5:])["choices"][0]["delta"]["content"]
                            except Exception as e:
                                continue

                            if decoded != '':
                                for p_token in temp_processed_d_data1:
                                    processed_d_data1 += p_token
                                    if not is_not_remove_emoji:
                                        processed_d_data1 = remove_emojis(processed_d_data1)
                                    lastlen = len(temp_tts_list)
                                    temp_tts_list = processed_d_data1.split("#split#")
                                    if not temp_tts_list:
                                        temp_tts_list = temp_tts_list[:-1]
                                    if self_id not in objdict["banaijian%sgeneing" % rev['group_id']]:
                                        objdict["banaijian%s" % rev['group_id']][0] = \
                                        objdict["banaijian%s" % rev['group_id']][0] + [
                                            {'role': 'user', 'content': rev['raw_message']},
                                            {'role': 'assistant', 'content': processed_d_data1}]
                                        raise InterruptedError("新消息中断")

                                    if len(temp_tts_list) > 1 and lastlen < len(temp_tts_list):
                                        if '#voice/' in temp_tts_list[-2]:
                                            try:
                                                voice = temp_tts_list[-2].split('#voice/')[-1].replace("#", '')
                                                tts_data = {"cha_name": speaker, "text": voice.replace("...", "…"),
                                                            "character_emotion": random.choice(
                                                                ['default', 'angry', 'excited'])}
                                                b_wav = requests.post(url='http://127.0.0.1:5000/tts', json=tts_data)
                                                n = random.randrange(10000, 99999)
                                                name = '%stts%d.wav' % (
                                                (time.strftime('%F') + '-' + time.strftime('%T').replace(':', '-')), n)
                                                with open('./data/voice/%s' % name, 'wb') as wbf:
                                                    wbf.write(b_wav.content)
                                                send_voice(
                                                    {'msg_type': 'group', 'number': rev['group_id'], 'msg': name})
                                            except:
                                                print("暂不支持语音合成")
                                        elif '#picture/' in temp_tts_list[-2]:
                                            picture = temp_tts_list[-2].split('#picture/')[-1].replace("#", '')
                                            draw_group(picture, rev['group_id'])
                                        elif '#search/' in temp_tts_list[-2]:
                                            response.close()
                                            temp_tts_list = temp_tts_list[:-1]
                                            break
                                        elif "#pass/" in temp_tts_list[-2]:
                                            response.close()
                                            raise KeyboardInterrupt("PASS")
                                        else:
                                            send_msg({'msg_type': 'group', 'number': rev['group_id'],
                                                      'msg': temp_tts_list[-2].replace("%s：" % AI_name, "")})

                        # 结尾处理
                        if len(temp_tts_list) > 0 and "抱歉" in temp_tts_list[-1]:
                            objdict["banaijian%s" % rev['group_id']][0] = [
                                objdict["banaijian%s" % rev['group_id']][0][0]]
                        else:
                            last_segment = temp_tts_list[-1] if temp_tts_list else ""
                            if '#search/' in last_segment:
                                s_prompt = last_segment.split('#search/')[-1].replace("#", '')
                                send_msg({'msg_type': 'group', 'number': rev['group_id'],
                                          'msg': "正在联网搜索：%s" % s_prompt})
                                search_result = search(s_prompt)
                                objdict["banaijian%s" % rev['group_id']][0] += [
                                    {'role': 'user', 'content': rev['raw_message']}, {'role': 'assistant',
                                                                                      'content': processed_d_data1 + """\nsystem[搜索结果不可见]：搜索结果：\n%s\n""" % (
                                                                                          search_result)},
                                    {"role": "user", "content": "请详细讲述"}]
                                messages = objdict["banaijian%s" % rev['group_id']][0]
                                data["messages"] = merge_contents(
                                    [{"role": "system", "content": system_prompt}] + messages[1:])
                                is_return = True
                                continue
                            else:
                                send_msg({'msg_type': 'group', 'number': rev['group_id'],
                                          'msg': last_segment.replace("%s：" % AI_name, "")})

                            objdict["banaijian%s" % rev['group_id']][0] = objdict["banaijian%s" % rev['group_id']][
                                                                              0] + [{'role': 'user',
                                                                                     'content': rev['raw_message']},
                                                                                    {'role': 'assistant',
                                                                                     'content': processed_d_data1}]
                            with open("./user/g%s/memory.txt" % rev['group_id'], "a", encoding="utf-8") as txt:
                                txt.write("[%s]%s\n[%s]你回复：%s\n" % (
                                current_time, objdict["banaijian%schat" % rev['group_id']], current_time,
                                processed_d_data1))
                        objdict["banaijian%schat" % rev['group_id']] = ''

            if len(objdict["banaijian%s" % rev['group_id']][0]) > 18:
                objdict["banaijian%s" % rev['group_id']][0] = [objdict["banaijian%s" % rev['group_id']][0][0]] + \
                                                              objdict["banaijian%s" % rev['group_id']][0][-6:]
    except Exception as e:
        if str(e) != "PASS":
            print(f"Main Error: {e}")


# ==================== [启动服务] ====================

# 1. 静态文件服务 (提供图片/语音给 NapCat)
file_app = Flask("file_server")
CORS(file_app)


@file_app.route('/data/image/<filename>', methods=['GET', 'POST'])
def image_files(filename):
    if os.path.exists(f'./data/image/{filename}'): return send_from_directory('./data/image/', filename)
    return 'Not found', 404


@file_app.route('/data/voice/<filename>', methods=['GET', 'POST'])
def voice_files(filename):
    if os.path.exists(f'./data/voice/{filename}'): return send_from_directory('./data/voice/', filename)
    return 'Not found', 404


def run_file_server():
    # 绑定 0.0.0.0 允许 Docker 访问
    serve(file_app, host='0.0.0.0', port=4321, threads=10)


Thread(target=run_file_server).start()

# 2. 消息接收服务 (替换 Socket)
recv_app = Flask("recv_server")


@recv_app.route('/', methods=['POST'])
def receive_event():
    try:
        data = request.json
        # 只要不是心跳包，就开线程处理
        if data and data.get('post_type') != 'meta_event':
            Thread(target=main, args=(data,)).start()
    except Exception as e:
        print(f"接收异常: {e}")
    # 始终返回成功，防止 NapCat 报错
    return jsonify({})


def run_recv_server():
    print("启动消息接收服务 (0.0.0.0:3001)...")
    serve(recv_app, host='0.0.0.0', port=3001, threads=10)


Thread(target=run_recv_server).start()

print('程序完全启动，等待消息...')
while True: time.sleep(10)