from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import re
import time  # ✅ 用来模拟大模型耗时

class RequestHandler(BaseHTTPRequestHandler):
    # 解析JSON请求体
    def _parse_json_body(self, length):
        try:
            body = self.rfile.read(length).decode('utf-8')
            return json.loads(body)
        except json.JSONDecodeError:
            return None

    # 发送JSON响应
    def _send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        # 如果前端是在 localhost 上用 fetch，可以顺便加一行 CORS（可选）
        # self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    # 处理POST请求
    def do_POST(self):
        # 仅处理指定路径
        if self.path != '/api/user/ask':
            self._send_json_response(
                {'error': '路径不存在'},
                status_code=404
            )
            return

        # 检查Content-Type是否为JSON
        content_type = self.headers.get('Content-Type', '')
        if not re.search(r'application/json', content_type):
            self._send_json_response(
                {'error': '请使用application/json格式'},
                status_code=415
            )
            return

        # 读取并解析请求体
        try:
            content_length = int(self.headers['Content-Length'])
        except (KeyError, ValueError):
            self._send_json_response(
                {'error': '缺少Content-Length头'},
                status_code=400
            )
            return

        request_data = self._parse_json_body(content_length)
        if not request_data or 'question' not in request_data:
            self._send_json_response(
                {'error': '请求体缺少question字段'},
                status_code=400
            )
            return

        question = request_data.get("question", "")
        print(f"收到问题：{question}")

        # ✅ 模拟大模型思考耗时：停顿 5 秒
        time.sleep(5)

        # mock 主回答
        mock_response = (
            "得了高血压平时需要注意以下几点："
            "1. 饮食方面，控制食盐摄入量，每天不超过 6 克，避免吃太油腻的食物，"
            "多吃新鲜绿色蔬菜水果和有机食物，还可以适量用党参泡水喝，因为党参有降血脂、降血压等作用；"
            "2. 适度增强体育锻炼，提高身体素质；"
            "3. 保持情绪平和，避免激动，减轻精神压力，不要过度紧张；"
            "4. 若通过生活方式调整后血压控制效果不佳，应在医生指导下配合降压药物治疗。"
        )

        # mock 参考案例
        mock_cases = [
            {
                "id": 1,
                "question": "我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗?",
                "answer": (
                    "高血压病人可以口服党参的。党参有降血脂、降血压的作用，可以帮助改善心血管状况，"
                    "但仍需注意血压监测，遵医嘱服药。"
                ),
            },
            {
                "id": 2,
                "question": "我是一个中学教师，最近体检发现高血压，该怎么治疗、需要注意什么？",
                "answer": (
                    "高血压患者首先要注意控制食盐摄入量，每天不超过 6 克，注意避免油腻饮食，多吃蔬菜水果；"
                    "同时保持规律作息与适量运动，必要时在医生指导下使用降压药物。"
                ),
            },
        ]

        # 返回响应：response + cases
        self._send_json_response({
            "response": mock_response,
            "cases": mock_cases,
        })

    # 处理其他请求方法
    def do_GET(self):
        self._send_json_response(
            {'message': '请使用POST方法访问 /api/user/ask 接口'},
            status_code=405
        )

def run_server(host='0.0.0.0', port=886):
    server_address = (host, port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"服务器启动，监听 {host}:{port} ...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器正在关闭...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
