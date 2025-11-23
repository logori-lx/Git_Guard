import axios from "axios";

export interface ReferenceCase {
  id: number;
  question: string;
  answer: string;
}

// 后端原始返回结构
export interface RawResponse {
  response: string;
  cases?: ReferenceCase[];
}

// 给前端 UI 用的结构
export interface AskResponse {
  response: string;
  referenceCases: ReferenceCase[];
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const resp = await axios.post("/api/user/ask", { question });
  const data = resp.data as RawResponse;

  return {
    response: data.response,
    referenceCases: Array.isArray(data.cases) ? data.cases : [],
  };
}
