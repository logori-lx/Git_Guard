import React, { useEffect, useRef, useState } from "react";

export type Role = "user" | "assistant" | "system";

export interface ReferenceCase {
  id: number;
  question: string;
  answer: string;
}

export interface Message {
  id: string;
  role: Role;
  text: string;
  referenceCases?: ReferenceCase[];
  loading?: boolean;
}

const TYPING_INTERVAL = 15;       // 打字速度：数值越大越慢
const TYPING_MAX_LENGTH = 1200;   // 超过这个长度直接展示全部文本

interface MessageItemProps {
  msg: Message;
  /** 是否允许这一条执行打字机（只有最后一条 assistant 是 true） */
  typingEnabled?: boolean;
  /** 打字完成时调用（只对最后一条 assistant 生效） */
  onTypingDone?: () => void;
}

const MessageItem: React.FC<MessageItemProps> = ({
  msg,
  typingEnabled = false,
  onTypingDone,
}) => {
  const [typedText, setTypedText] = useState(msg.text);
  const [isTypingDone, setIsTypingDone] = useState(true);
  const [showRef, setShowRef] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  const isUser = msg.role === "user";

  // 用 ref 保存回调，避免它进 useEffect 依赖导致重复打字
  const doneRef = useRef<(() => void) | undefined>(onTypingDone);
  useEffect(() => {
    doneRef.current = onTypingDone;
  }, [onTypingDone]);

  /* ---------- 用户消息：不打字机，不显示复制 ---------- */
  if (isUser) {
    return <div className="user-bubble">{msg.text}</div>;
  }

  const hasRef =
    msg.referenceCases &&
    Array.isArray(msg.referenceCases) &&
    msg.referenceCases.length > 0;

  /* ---------- AI 消息打字机 ---------- */
  useEffect(() => {
    // 不启用打字机 或 正在 loading：直接展示
    if (!typingEnabled || msg.loading) {
      setTypedText(msg.text);
      setIsTypingDone(true);
      return;
    }

    // 文本太长：直接展示，避免动画太久
    if (msg.text.length > TYPING_MAX_LENGTH) {
      setTypedText(msg.text);
      setIsTypingDone(true);
      doneRef.current?.();
      return;
    }

    let index = 0;
    setTypedText("");
    setIsTypingDone(false);

    const timer = setInterval(() => {
      index++;
      setTypedText(msg.text.slice(0, index));

      // 每次打出新内容时，把滚动条拉到底（只对最后一条开启 typing 的消息生效）
      const outer = document.getElementById("scroll-container");
      if (outer) {
        outer.scrollTop = outer.scrollHeight;
      }

      if (index >= msg.text.length) {
        clearInterval(timer);
        setIsTypingDone(true);
        doneRef.current?.();
      }
    }, TYPING_INTERVAL);

    return () => clearInterval(timer);
  }, [msg.text, msg.loading, typingEnabled]);

  /* 显示“参考案例”的条件：
     1）有参考案例
     2）不是 loading
     3）要么这条不打字（旧消息），要么打字已经结束（当前最后一条） */
  const canShowRefUI = hasRef && !msg.loading && (!typingEnabled || isTypingDone);

  // 展开/收起参考案例：展开时自动滚到底
  const handleToggleRef = () => {
    setShowRef((prev) => {
      const next = !prev;
      if (!prev && next) {
        setTimeout(() => {
          const outer = document.getElementById("scroll-container");
          if (outer) {
            outer.scrollTop = outer.scrollHeight;
          }
        }, 0);
      }
      return next;
    });
  };

  // 复制回答文本（这里只复制主回答 msg.text，不包含案例；如果想连案例一起复制可以改逻辑）
  const handleCopy = async () => {
    try {
      await navigator.clipboard?.writeText(msg.text || "");
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 1500);
    } catch (e) {
      console.error("复制失败:", e);
    }
  };

  return (
    <div className="ai-card">
      {/* 主回答 */}
      {msg.loading ? (
        <div className="loading-dots">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
      ) : (
        <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.65 }}>
          {typedText}
          {/* 打字中才显示光标，打完自动消失 */}
          {typingEnabled && !isTypingDone && (
            <span className="typewriter-cursor" />
          )}
        </div>
      )}

      {/* 参考案例：等打完字再出现，展开时自动滚到底 */}
      {canShowRefUI && (
        <>
          <div className="context-toggle" onClick={handleToggleRef}>
            {showRef
              ? "隐藏参考案例 Reference cases ▲"
              : "显示参考案例 Reference cases ▼"}
          </div>

          <div className={`context-box ${showRef ? "" : "collapsed"}`}>
            <div
              style={{
                fontWeight: 600,
                fontSize: 13,
                marginBottom: 6,
                color: "#4b5563",
              }}
            >
              参考案例（Reference cases）
            </div>

            {msg.referenceCases!.map((c, index) => (
              <div
                key={c.id ?? index}
                style={{
                  marginBottom: 12,
                  whiteSpace: "pre-wrap",
                  lineHeight: 1.6,
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  Case {index + 1}
                </div>
                <div>
                  <strong>Question:</strong> {c.question}
                </div>
                <div>
                  <strong>Answer:</strong> {c.answer}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* 复制按钮：放在回答 + 参考案例之后，靠右对齐 */}
      {!msg.loading && (
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            marginTop: canShowRefUI ? 8 : 6,
          }}
        >
          <button
            type="button"
            className="copy-btn"
            onClick={handleCopy}
            disabled={!msg.text}
          >
            {isCopied ? "已复制" : "复制"}
          </button>
        </div>
      )}
    </div>
  );
};

export default MessageItem;
