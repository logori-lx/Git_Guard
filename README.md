# Document Link
https://rp7bg9wtwt.feishu.cn/wiki/Vaj9wFrpIikJwgkoZ3ZcHaemnvh

# Data preprocessing
tag the original csv data with related_disease_1, related_disease_2
```
cd data_processing
python disease_name_preprocess.py
python data_preprocessing.py
```
# One-key deployment
As docker compose has been used, you can run both frontend and backend with the command below
```
# in project root directory 
# ATTENTION: You need to ensure the backend image has been built
# by executing the cmd: cd backend && docker build . -t backend
# before executing the command below to start both frontend and backend service
docker-compose up -d
```
stop both the frontend and backend with the command below
```
docker-compose down
```

# Backend Usage
## Method1
Before running server you need to set the api_key of 智谱清言 in evironmental variable
```
MEDICAL_RAG= {your api key you get from https://bigmodel.cn/}
```
Then place database to backend/DATA
```
// path should be existed below
backend\DATA\chroma_db
```
start server:
```
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
## Method2
build your docker image
```
cd backend && docker build . -t backend
```
run server in docker
```
docker run -d   --name backend   -p 8000:8000    -e "MEDICAL_RAG={fill your api key you get from 智谱清言}"    backend:latest
```
## Check server start or not
post request
```
curl -X POST http://localhost:8000/api/user/ask 
  -H "Content-Type: application/json" 
  -d '{"question": "得了高血压平时需要注意什么？"}'
```
# FrontEnd Usage
1. Install dependencies
npm install
2. Run the frontend
npm run dev
Frontend will listen to 8000 port and run on:
➡️ http://localhost:5173/