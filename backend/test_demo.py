import pytest
import json
from http.server import HTTPServer
from threading import Thread
import requests
from demo import RequestHandler, run_server

# 启动测试服务器的辅助函数
def start_test_server():
    server = Thread(target=run_server, args=('localhost', 8888), daemon=True)
    server.start()
    return server

@pytest.fixture(scope="module", autouse=True)
def server():
    # 启动测试服务器
    server_thread = start_test_server()
    # 等待服务器启动
    import time
    time.sleep(1)
    yield
    # 测试结束后无需额外操作，daemon线程会随主进程退出

BASE_URL = "http://localhost:8888/api/user/ask"

def test_post_correct_request():
    """测试正确的POST请求"""
    data = {"question": "高血压需要注意什么？"}
    response = requests.post(
        BASE_URL,
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "response" in result
    assert "cases" in result
    assert isinstance(result["cases"], list)
    assert len(result["cases"]) > 0


def test_get_method():
    """测试使用GET方法访问接口"""
    response = requests.get(BASE_URL)
    assert response.status_code == 405
    assert "message" in response.json()

def test_missing_content_type():
    """测试缺少Content-Type头"""
    data = {"question": "测试"}
    response = requests.post(
        BASE_URL,
        json=data,
        headers={"Content-Type": "text/plain"}
    )
    
    assert response.status_code == 415
    assert "error" in response.json()

def test_missing_content_length():
    """测试缺少Content-Length头（通过不传递数据模拟）"""
    response = requests.post(
        BASE_URL,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 400
    assert "error" in response.json()

def test_invalid_json():
    """测试无效的JSON格式"""
    response = requests.post(
        BASE_URL,
        data="{invalid json}",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 400
    assert "error" in response.json()

def test_missing_question_field():
    """测试请求体缺少question字段"""
    data = {"other_field": "value"}
    response = requests.post(
        BASE_URL,
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 400
    assert "error" in response.json()