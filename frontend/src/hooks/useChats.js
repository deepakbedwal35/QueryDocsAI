import { useState, useEffect, useCallback } from "react";
import {
  listChats,
  createChat,
  renameChat,
  deleteChat,
} from "../api/chatsApi";
import { useAsyncAction } from "./useAsyncAction";

/**
 * Owns the chat list + which chat is active, and every chat-level
 * CRUD action (create, rename, delete). Nothing about messages or
 * documents lives here -- see useMessages / useDocuments.
 */
export function useChats() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isLoadingChats, setIsLoadingChats] = useState(true);
  const { run, error, setError } = useAsyncAction();

  const activeChat = chats.find((c) => c.chat_id === activeChatId) ?? null;

  // Initial load: fetch this device's chats, or create the first one.
  useEffect(() => {
    (async () => {
      await run(async () => {
        const existing = await listChats();
        if (existing.length > 0) {
          setChats(existing);
          setActiveChatId(existing[0].chat_id);
        } else {
          const chat = await createChat();
          setChats([chat]);
          setActiveChatId(chat.chat_id);
        }
      });
      setIsLoadingChats(false);
    })();
  }, [run]);

  const handleNewChat = useCallback(async () => {
    await run(async () => {
      const chat = await createChat();
      setChats((prev) => [chat, ...prev]);
      setActiveChatId(chat.chat_id);
    });
  }, [run]);

  const handleSelectChat = useCallback((chatId) => {
    setActiveChatId(chatId);
  }, []);

  const handleRenameChat = useCallback(
    async (chatId, newTitle) => {
      // Optimistic update, corrected if the request fails.
      setChats((prev) =>
        prev.map((c) => (c.chat_id === chatId ? { ...c, title: newTitle } : c)),
      );
      await run(() => renameChat(chatId, newTitle));
    },
    [run],
  );

  const handleDeleteChat = useCallback(
    async (chatId) => {
      const result = await run(() => deleteChat(chatId));
      if (result === undefined) return; // run() already set error, bail out

      setChats((prevChats) => {
        const remaining = prevChats.filter((c) => c.chat_id !== chatId);

        if (chatId !== activeChatId) {
          return remaining;
        }

        if (remaining.length > 0) {
          setActiveChatId(remaining[0].chat_id);
          return remaining;
        }

        // No chats left -- create a fresh one so the app always has
        // an active chat. Fire-and-forget; the effect below or a
        // direct call handles setting it once created.
        createChat().then((chat) => {
          setChats([chat]);
          setActiveChatId(chat.chat_id);
        });
        return remaining;
      });
    },
    [run, activeChatId],
  );

  // Called by useMessages after the first message in a "New chat" is
  // sent, since the backend auto-titles the chat server-side and the
  // sidebar needs to pick that title up.
  const refreshChats = useCallback(async () => {
    await run(async () => {
      const refreshed = await listChats();
      setChats(refreshed);
    });
  }, [run]);

  // Called by useMessages to bump a chat's updated_at locally after a
  // non-first message, so the sidebar's "most recent" ordering (if any)
  // stays accurate without a full refetch.
  const touchChat = useCallback((chatId) => {
    setChats((prev) =>
      prev.map((c) =>
        c.chat_id === chatId
          ? { ...c, updated_at: new Date().toISOString() }
          : c,
      ),
    );
  }, []);

  return {
    chats,
    activeChatId,
    activeChat,
    isLoadingChats,
    error,
    setError,
    handleNewChat,
    handleSelectChat,
    handleRenameChat,
    handleDeleteChat,
    refreshChats,
    touchChat,
  };
}