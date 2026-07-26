import { useState, useEffect, useCallback } from "react";
import { askQuestion } from "../api/askApi";
import { getChatMessages } from "../api/chatsApi";

import { useAsyncAction } from "./useAsyncAction";

let msgIdCounter = 0;
const nextMsgId = () => `local-${++msgIdCounter}`;
export function fromApiMessage(m) {
  return {
    id: m.message_id,
    role: m.role,
    text: m.content,
    citations: m.citations ?? undefined,
    isError: false,
  };
}
/**
 * Owns the message list for whichever chat is active, plus sending a
 * new question. Needs to know the active chat (and whether it's a
 * fresh "New chat", to trigger the title refresh) and gets two small
 * callbacks from useChats so it can tell the chat list about that --
 * see the props note in App.jsx for why it's wired this way instead
 * of useMessages importing useChats directly.
 */
export function useMessages(activeChatId, { isNewChat, onFirstMessageSent, onMessageSent }) {
  const [messages, setMessages] = useState([]);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const { run, error } = useAsyncAction();

  // Load message history whenever the active chat changes.
  useEffect(() => {
    if (!activeChatId) return;
    (async () => {
      setIsLoadingMessages(true);
      await run(async () => {
        const history = await getChatMessages(activeChatId);
        setMessages(history.map(fromApiMessage));
      });
      setIsLoadingMessages(false);
    })();
  }, [activeChatId, run]);

  const handleSend = useCallback(
    async (question) => {
      const chatId = activeChatId;
      const wasNewChat = isNewChat;

      setMessages((prev) => [
        ...prev,
        { id: nextMsgId(), role: "user", text: question },
      ]);
      setIsSending(true);

      try {
        const data = await askQuestion(chatId, question);
        setMessages((prev) => [
          ...prev,
          {
            id: nextMsgId(),
            role: "assistant",
            text: data.answer,
            citations: data.citations,
          },
        ]);

        // The backend auto-titles new chats from the first question --
        // let useChats know so the sidebar picks up the new title.
        if (wasNewChat) {
          await onFirstMessageSent?.();
        } else {
          onMessageSent?.(chatId);
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: nextMsgId(),
            role: "assistant",
            text: `Something went wrong: ${err.message}`,
            isError: true,
          },
        ]);
      } finally {
        setIsSending(false);
      }
    },
    [activeChatId, isNewChat, onFirstMessageSent, onMessageSent],
  );

  return { messages, isLoadingMessages, isSending, error, handleSend };
}