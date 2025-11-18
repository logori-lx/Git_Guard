from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import re

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
        mock_response = "得了高血压平时需要注意以下几点：1. 饮食方面，控制食盐摄入量，每天不超过 6 克，避免吃太油腻的食物，多吃新鲜绿色蔬菜水果和有机食物，还可以适量用党参泡水喝，因为党参有降血脂、降血压等作用；2. 适度增强体育锻炼，提高身体素质；3. 保持情绪平和，避免激动，减轻精神压力，不要过度紧张；4. 若通过生活方式调整后血压控制效果不佳，应在医生指导下配合降压药物治疗。" 
        mock_cases = [
                {
                    "id": 1,
                    "question": "我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗?",
                    "answer": "高血压病人可以口服党参的。党参有降血脂，降血压的作用，可以彻底消除血液中的垃圾，从而对冠心病以及心血管疾病的患者都有一定的稳定预防工作作用，因此平时口服党参能远离三高的危害。另外党参除了益气养血，降低中枢神经作用，调整消化系统功能，健脾补肺的功能。感谢您的进行咨询，期望我的解释对你有所帮助。"
                },
                {
                    "id": 2,
                    "question": "我是一位中学教师，平时身体健康，最近学校组织健康检查，结果发觉我是高血压，去年还没有这种情况，我很担心，这边我主要想进行咨询一下高血压应当怎样治疗？麻烦医生指导一下，谢谢。",
                    "answer": "高血压患者首先要注意控制食盐摄入量，每天不超过六克，注意不要吃太油腻的食物，多吃新鲜的绿色蔬菜水果，多吃有机食物，注意增强体育锻炼，增加身体素质，同时压力不要过大，精神不要紧张，效果不佳的话，可以积极配合以降压药物控制血压治疗，情绪平时保持平和，不易激动。"
                }
            ]

        # 返回响应
        self._send_json_response({
            'response': mock_response,
            'cases': mock_cases
        })

    # 处理其他请求方法
    def do_GET(self):
        self._send_json_response(
            {'message': '请使用POST方法访问/api/user/ask接口'}, 
            status_code=405
        )

def run_server(host='0.0.0.0', port=886):
    server_address = (host, port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"服务器启动，监听 {host}:{port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器正在关闭...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()