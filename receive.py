import socket
import json
import time

ListenSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ListenSocket.bind(('127.0.0.1', 3001))
ListenSocket.listen(100)

# 【修复】标准的 HTTP 响应头，使用 \r\n 分隔，并返回空的 JSON {}
# 这样 NapCat 就不会报错 "Invalid header value char" 了
HttpResponseHeader = 'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}'


def request_to_json(msg):
    try:
        # 自动识别 HTTP 头和 Body 的分隔符
        if "\r\n\r\n" in msg:
            dicstrmsg = msg.split("\r\n\r\n")[-1]
        elif "\n\n" in msg:
            dicstrmsg = msg.split("\n\n")[-1]
        else:
            return None

        return json.loads(dicstrmsg)
    except:
        return None


# 需要循环执行，返回值为json格式
def rev_msg():  # json or None
    Client, Address = ListenSocket.accept()
    try:
        # 使用 errors='ignore' 防止部分特殊字符导致 decode 报错
        Request = Client.recv(10240).decode(encoding='utf-8', errors='ignore')

        rev_json = request_to_json(Request)

        # 【新增】只有收到非心跳包(meta_event)的数据时才打印，证明 Socket 通了
        if rev_json and rev_json.get('post_type') != 'meta_event':
            print(f"[Receive] 收到数据: Type={rev_json.get('post_type')}")

    except Exception as e:
        print(f"接收出错: {e}")
        rev_json = None

    # 发送修复后的响应
    Client.sendall(HttpResponseHeader.encode(encoding='utf-8'))
    Client.close()

    return rev_json