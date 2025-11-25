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

  // 后端请求中（等待接口返回）
  const [isSending, setIsSending] = useState(false);
  // 前端打字机动画中
  const [isTyping, setIsTyping] = useState(false);

  // ✅ 新增：是否展示左侧历史
  const [showHistory, setShowHistory] = useState(true);

  /* -------------------- 初始化 -------------------- */
  useEffect(() => {
    startNewSession();
    const saved = JSON.parse(localStorage.getItem("chat-history") || "[]");
    setHistory(saved);

    // 小屏默认隐藏历史
    if (window.innerWidth < 900) {
      setShowHistory(false);
    }
  }, []);

  /* -------------------- 保存当前会话到本地 -------------------- */
  useEffect(() => {
    if (!sessionId) return;
    localStorage.setItem("session-" + sessionId, JSON.stringify(messages));
  }, [messages, sessionId]);

  /* -------------------- 更新左侧历史标题 -------------------- */
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

  /* -------------------- 新对话 -------------------- */
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

  /* -------------------- 加载历史会话 -------------------- */
  const loadHistory = (id: string) => {
    const saved = JSON.parse(localStorage.getItem("session-" + id) || "[]");
    if (!saved || saved.length === 0) return;

    setSessionId(id);
    setIsSending(false);
    setIsTyping(false);
    setMessages(saved);
  };

  /* -------------------- 删除历史会话 -------------------- */
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

  /* -------------------- 发送问题 -------------------- */
  const send = async (text: string) => {
    if (isSending || isTyping) return; // 正在请求或打字中时禁止重复发送

    const userMsg: Message = { id: uuidv4(), role: "user", text };
    const loadingMsg: Message = {
      id: "loading-" + Date.now(),
      role: "assistant",
      text: "",
      loading: true,
    };

    // 一次性把“用户消息 + loading 占位”塞进去
    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setIsSending(true);

    try {
      const res: any = await askQuestion(text);

      // 按你最新后端结构解析：answer + context[ ]（ask/answer/department）
      const rawCases = Array.isArray(res.context) ? res.context : [];
      const referenceCases =
        rawCases.length > 0
          ? rawCases.map((c: any, index: number) => ({
              id: index + 1,
              question: c.ask ?? "",
              answer: c.answer ?? "",
              department: c.department ?? "",
            }))
          : [];

      const finalId = uuidv4();

      // 替换 loading 那条消息
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? {
                id: finalId,
                role: "assistant",
                text: res.answer ?? "",
                referenceCases:
                  referenceCases.length > 0 ? referenceCases : undefined,
              }
            : m
        )
      );

      // 开始前端打字机动画
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
      // 后端请求结束（不管成功失败）
      setIsSending(false);
    }
  };

  /* -------------------- 找到需要打字机的最后一条回答 -------------------- */
  const lastAssistant = [...messages]
    .filter((m) => m.role === "assistant" && !m.loading)
    .slice(-1)[0];
  const lastAssistantId = lastAssistant?.id;

  const inputDisabled = isSending || isTyping;

  /* -------------------- 切换历史显示 -------------------- */
  const toggleHistory = () => {
    setShowHistory((v) => !v);
  };

  return (
    <div className="app-shell">
      {/* 顶部 */}
      <header className="header">
        <div className="header-inner">
          <div className="header-top-row">
            <div>
              <h1>Medical RAG 医疗健康咨询助手</h1>
              <p>
                面向普通用户的专业健康咨询工具，结合检索增强生成（RAG）技术，为常见健康问题提供权威且易懂的回答。
              </p>
            </div>

            {/* 右上角：显示 / 隐藏 历史按钮，桌面 & 手机都能用 */}
<button
  onClick={() => setShowHistory(!showHistory)}
  className="toggle-history-btn"
>
  {showHistory ? "隐藏历史" : "显示历史"}
</button>
          </div>
        </div>
      </header>

      <div className="main-container">
        {/* 左侧历史栏：完全由 showHistory 控制是否渲染 */}
        {showHistory && (
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
        )}

        {/* 右侧聊天区（会根据是否有 sidebar 自然铺满剩余宽度） */}
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
