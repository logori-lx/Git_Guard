// src/services/chat.ts
import axios from "axios";

/**
 * The context returned by the backend, each item in the structure
 */
export interface BackendContextItem {
  ask: string;
  answer: string;
  department?: string;
}

/**
 * The original return structure from the backend (the one you just sent me)
 *
 * {
 *   "answer": "……Summary of answers……",
 *   "context": [
 *     { "ask": "...", "answer": "...", "department": "..." },
 *     ...
 *   ]
 * }
 */
export interface BackendResponse {
  answer: string;
  context?: BackendContextItem[];
}

/**
* For the front-end use of askQuestion:
* Return the back-end's answer/context as is.
* The ChatPage already uses res.answer / res.context to retrieve the fields.
 */
export async function askQuestion(
  question: string
): Promise<BackendResponse> {
  const resp = await axios.post("/api/user/ask", { question });
  const data = resp.data || {};

  return {
    answer: data.answer ?? "",
    context: Array.isArray(data.context) ? data.context : [],
  };
}
