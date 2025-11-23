import React, { useEffect } from "react";
import MessageItem, { Message } from "./MessageItem";

interface Props {
  messages: Message[];
  lastAssistantId?: string;
  onLastTypingDone?: () => void;
}

const MessageList: React.FC<Props> = ({
  messages,
  lastAssistantId,
  onLastTypingDone,
}) => {
  useEffect(() => {
    const outer = document.getElementById("scroll-container");
    if (!outer) return;
    requestAnimationFrame(() => {
      outer.scrollTop = outer.scrollHeight;
    });
  }, [messages]);

  return (
    <>
      {messages.map((msg) => (
        <MessageItem
          key={msg.id}
          msg={msg}
          typingEnabled={msg.id === lastAssistantId} // 只有最后一条才允许打字
          onTypingDone={msg.id === lastAssistantId ? onLastTypingDone : undefined}
        />
      ))}
    </>
  );
};

export default MessageList;
