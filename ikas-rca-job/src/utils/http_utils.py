# utils/http_utils.py
import requests

def notify_http_api(task_id, task_status, msg, end_time, properties_config):
    """
    调用 HTTP 接口通知其他程序计算结果
    """
    url = properties_config["rca_http_api_url"]
    payload = {
        "taskId": task_id,
        "taskStatus": task_status,
        "msg": msg,
        "endTime": end_time
    }
    headers = {"Content-Type": "application/json"}

    try:
        # 发送PUT请求
        response = requests.put(url, json=payload, headers=headers)
        response.raise_for_status()  # 如果响应状态码不是200，则抛出异常
        print(f"------发生请求到后端-----:{payload}")
        print(f"{payload}")
        print(f"{response}")
        return response
    except requests.exceptions.RequestException as e:
        print(f"HTTP PUT请求失败: {e}")
        return None

