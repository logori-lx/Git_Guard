# Document Link
https://rp7bg9wtwt.feishu.cn/wiki/Vaj9wFrpIikJwgkoZ3ZcHaemnvh
# Backend Usage
start server:
```
cd backend && python demo.py
```
post request
```
curl -X POST http://localhost:8080/api/user/ask 
  -H "Content-Type: application/json" 
  -d '{"question": "得了高血压平时需要注意什么？"}'
```
