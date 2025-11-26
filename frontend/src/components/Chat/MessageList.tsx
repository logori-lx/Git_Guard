// src/components/Chat/MessageList.tsx
import React, { useEffect, useRef } from "react";
import MessageItem, { Message } from "./MessageItem";

const MessageList: React.FC<{ messages: Message[] }> = ({ messages }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Each time messages changes, it automatically rolls to the bottom (append behavior)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    // Minimal delay to ensure element rendering
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }, [messages]);

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-auto px-4 py-6 bg-white"
      style={{ minHeight: 0 }}
    >
      <div className="max-w-3xl mx-auto">
        {messages.map((m) => (
          <MessageItem key={m.id} msg={m} />
        ))}
      </div>
    </div>
  );
};

export default MessageList;
