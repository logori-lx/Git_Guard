# Document Link
https://rp7bg9wtwt.feishu.cn/wiki/Vaj9wFrpIikJwgkoZ3ZcHaemnvh

# Data preprocessing
tag the original csv data with related_disease_1, related_disease_2
```
cd data_processing
python disease_name_preprocess.py
python data_preprocessing.py
```
# Backend Usage
Before running server you need to set the api_key of 智谱清言 in evironmental variable
```
MEDICAL_RAG= {your api key you get from https://bigmodel.cn/}
```
Then place database to project/data_processing/DATA
```
// path should be existed below
data_processing\DATA\chroma_db
```
start server:
```
cd project && uvicorn main:app --reload --host 0.0.0.0 --port 8080
```
post request
```
curl -X POST http://localhost:8080/api/user/ask 
  -H "Content-Type: application/json" 
  -d '{"question": "得了高血压平时需要注意什么？"}'
```
# FrontEnd Usage
1. Install dependencies
npm install
2. Run the frontend
npm run dev
Frontend will run on:

➡️ http://localhost:5173/