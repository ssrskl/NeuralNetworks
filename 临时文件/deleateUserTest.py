# 设置API相关信息
import requests

api_url = 'http://10.168.91.142/hub/api/'
api_token = '2696c2a4daf84e98af60bd6780858570'
headers = {
    'Authorization': f'token {api_token}',
}

r = requests.delete(
    "http://10.168.91.142/hub/api/users/gufs20230401637",
    headers=headers,
)

print(r.status_code)