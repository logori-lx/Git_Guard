// src/pages/Chat/ChatPage.tsx
import React, { useEffect, useState } from "react";
import MessageList from "../../shared/MessageList";
import ChatInput from "../../shared/ChatInput";
import { askQuestion } from "../../services/chat";
import { v4 as uuidv4 } from "uuid";
import type { Message } from "../../shared/MessageItem";

interface HistoryItem {
  id: string;
  title: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [sessionId, setSessionId] = useState("");

  // Backend request in progress (waiting for API response)
  const [isSending, setIsSending] = useState(false);
  // In the front-end typewriter animation
  const [isTyping, setIsTyping] = useState(false);

  /* -------------------- initialization -------------------- */
  useEffect(() => {
    startNewSession();
    const saved = JSON.parse(localStorage.getItem("chat-history") || "[]");
    setHistory(saved);
  }, []);

  /* -------------------- Save the current session to local -------------------- */
  useEffect(() => {
    if (!sessionId) return;
    localStorage.setItem("session-" + sessionId, JSON.stringify(messages));
  }, [messages, sessionId]);

  /* -------------------- Update the history titles on the left. -------------------- */
  useEffect(() => {
    if (!sessionId) return;
    const firstUser = messages.find((m) => m.role === "user");
    if (!firstUser) return;

    setHistory((prev) => {
      const others = prev.filter((h) => h.id !== sessionId);
      const cur: HistoryItem = {
        id: sessionId,
        title:
          firstUser.text.length > 12
            ? firstUser.text.slice(0, 12) + "..."
            : firstUser.text,
      };
      const updated = [cur, ...others];
      localStorage.setItem("chat-history", JSON.stringify(updated));
      return updated;
    });
  }, [messages, sessionId]);

  /* -------------------- New Dialogue -------------------- */
  const startNewSession = () => {
    const id = uuidv4();
    setSessionId(id);
    setIsSending(false);
    setIsTyping(false);

    setMessages([
      {
        id: "welcome",
        role: "assistant",
        text:
          "欢迎使用 Medical RAG 医疗健康咨询助手。请用自然语言描述你的不适或疑问，例如“高血压能吃党参吗？”、“长期胃痛该挂什么科？”。",
      },
    ]);
  };

  /* -------------------- Load historical sessions -------------------- */
  const loadHistory = (id: string) => {
    const saved = JSON.parse(localStorage.getItem("session-" + id) || "[]");
    if (!saved || saved.length === 0) return;

    setSessionId(id);
    setIsSending(false);
    setIsTyping(false);
    setMessages(saved);
  };

  /* -------------------- Delete historical sessions -------------------- */
  const deleteHistory = (id: string) => {
    const updated = history.filter((h) => h.id !== id);
    setHistory(updated);
    localStorage.setItem("chat-history", JSON.stringify(updated));
    localStorage.removeItem("session-" + id);

    if (id === sessionId) {
      if (updated.length > 0) {
        loadHistory(updated[0].id);
      } else {
        startNewSession();
      }
    }
  };

  /* -------------------- Sending problem -------------------- */
  const send = async (text: string) => {
    if (isSending || isTyping) return; // Do not send duplicates while requesting or typing.

    const userMsg: Message = { id: uuidv4(), role: "user", text };
    const loadingMsg: Message = {
      id: "loading-" + Date.now(),
      role: "assistant",
      text: "",
      loading: true,
    };

    // Integrate "user message + loading placeholder" all at once
    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setIsSending(true);

    try {
      // ✅ Now, askQuestion returns {answer, context}.
      const res = await askQuestion(text);

      const aiAnswer = res.answer ?? "";
      const rawContext = Array.isArray(res.context) ? res.context : [];

      // ✅ Mapped to a unified ReferenceCase structure for easier UI use.
      const referenceCases = rawContext.map((c: any, index: number) => ({
        id: index + 1,
        question: c.ask ?? "",
        answer: c.answer ?? "",
        department: c.department ?? "",
      }));

      const finalId = uuidv4();

      // Replace the loading message
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? {
                id: finalId,
                role: "assistant",
                text: aiAnswer,
                referenceCases: referenceCases.length
                  ? referenceCases
                  : undefined,
              }
            : m
        )
      );

      // Start the front-end typewriter animation
      setIsTyping(true);
    } catch (e) {
      console.error(e);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? {
                id: uuidv4(),
                role: "assistant",
                text: "请求失败，请稍后再试。",
              }
            : m
        )
      );
      setIsTyping(false);
    } finally {
      // The backend request has ended (regardless of success or failure).
      setIsSending(false);
    }
  };


  /* -------------------- The last answer that asked for a typewriter was found. -------------------- */
  const lastAssistant = [...messages]
    .filter((m) => m.role === "assistant" && !m.loading)
    .slice(-1)[0];
  const lastAssistantId = lastAssistant?.id;

  const inputDisabled = isSending || isTyping;

  return (
    <div className="app-shell">
      {/* top */}
      <header className="header">
        <div className="header-inner">
          <h1>Medical RAG 医疗健康咨询助手</h1>
          <p>
            面向普通用户的专业健康咨询工具，结合检索增强生成（RAG）技术，为常见健康问题提供权威且易懂的回答。
          </p>
        </div>
      </header>

      <div className="main-container">
        {/* left-hand history bar */}
        <aside className="sidebar">
          <div className="sidebar-title">历史会话</div>

          <button className="history-new-btn" onClick={startNewSession}>
            新对话
          </button>

          {history.map((h) => (
            <div
              key={h.id}
              className="history-item"
              onClick={() => loadHistory(h.id)}
              title={h.title}
            >
              <span className="history-item-title">{h.title}</span>
              <button
                className="history-delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteHistory(h.id);
                }}
              >
                ×
              </button>
            </div>
          ))}
        </aside>

        {/* right-side chat area */}
        <div className="chat-container">
          <div className="message-list" id="scroll-container">
            <div className="message-center">
              <MessageList
                messages={messages}
                lastAssistantId={lastAssistantId}
                onLastTypingDone={() => setIsTyping(false)}
              />
            </div>
          </div>

          <div className="input-area">
            <div className="input-row">
              <ChatInput
                onSend={send}
                disabled={inputDisabled}
                placeholder="输入你的问题，例如：得了高血压平时需要注意什么？"
              />
            </div>

            {/* ⭐ This is the "Editing" status message you're looking for, which only appears while waiting for the backend.*/}
            {isSending && (
              <div className="input-status">
                编辑中…（模型正在生成回答，请稍候）
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
