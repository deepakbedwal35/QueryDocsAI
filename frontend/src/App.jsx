import { useEffect, useState } from "react";
import ChatHistorySidebar from "./components/ChatHistorySidebar";
import ChatWindow from "./components/ChatWindow";
import InputBar from "./components/InputBar";
import DocumentsSidebar from "./components/DocumentsSidebar";
import ThemeToggle from "./components/ThemeToggle";
import { askQuestion } from "./api/askApi";
import {
  listChats,
  createChat,
  renameChat,
  deleteChat,
  getChatMessages,
} from "./api/chatsApi";
import {
  listDocuments,
  uploadDocument,
  deleteDocument,
} from "./api/documentsApi";

let msgIdCounter = 0;
const nextMsgId = () => `local-${++msgIdCounter}`;

// Maps a backend message row into the shape ChatWindow expects.
function fromApiMessage(m) {
  return {
    id: m.message_id,
    role: m.role,
    text: m.content,
    citations: m.citations ?? undefined,
    isError: false,
  };
}

export default function App() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);

  const [isLoadingChats, setIsLoadingChats] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);
  const [documents, setDocuments] = useState([]);

  const activeChat = chats.find((c) => c.chat_id === activeChatId) ?? null;

  // Initial load: fetch this device's chats, or create the first one.
  useEffect(() => {
    (async () => {
      try {
        const existing = await listChats();
        if (existing.length > 0) {
          setChats(existing);
          setActiveChatId(existing[0].chat_id);
        } else {
          const chat = await createChat();
          setChats([chat]);
          setActiveChatId(chat.chat_id);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoadingChats(false);
      }
    })();
  }, []);

  // Load message history whenever the active chat changes.
  useEffect(() => {
    if (!activeChatId) return;
    (async () => {
      setIsLoadingMessages(true);
      try {
        const history = await getChatMessages(activeChatId);
        setMessages(history.map(fromApiMessage));
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoadingMessages(false);
      }
    })();
  }, [activeChatId]);

  // Load documents whenever the active chat changes.
  useEffect(() => {
    if (!activeChatId) return;
    (async () => {
      setIsLoadingDocuments(true);
      try {
        const docs = await listDocuments(activeChatId);
        setDocuments(docs);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoadingDocuments(false);
      }
    })();
  }, [activeChatId]);

  const handleNewChat = async () => {
    try {
      const chat = await createChat();
      setChats((prev) => [chat, ...prev]);
      setActiveChatId(chat.chat_id);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleSelectChat = (chatId) => {
    setActiveChatId(chatId);
  };

  const handleUploadDocument = async (file) => {
    try {
      const doc = await uploadDocument(activeChatId, file);
      setDocuments((prev) => [...prev, doc]);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDeleteDocument = async (docId) => {
    try {
      await deleteDocument(activeChatId, docId);
      setDocuments((prev) => prev.filter((d) => d.document_id !== docId));
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRenameChat = async (chatId, newTitle) => {
    // Optimistic update, corrected if the request fails.
    setChats((prev) =>
      prev.map((c) => (c.chat_id === chatId ? { ...c, title: newTitle } : c)),
    );
    try {
      await renameChat(chatId, newTitle);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDeleteChat = async (chatId) => {
    try {
      await deleteChat(chatId);
    } catch (err) {
      setError(err.message);
      return;
    }

    const remaining = chats.filter((c) => c.chat_id !== chatId);

    if (chatId !== activeChatId) {
      setChats(remaining);
      return;
    }

    if (remaining.length > 0) {
      setChats(remaining);
      setActiveChatId(remaining[0].chat_id);
    } else {
      const chat = await createChat();
      setChats([chat]);
      setActiveChatId(chat.chat_id);
    }
  };

  const handleSend = async (question) => {
    const chatId = activeChatId;
    const wasNewChat = activeChat?.title === "New chat";

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

      // The backend auto-titles new chats from the first question —
      // refresh the sidebar list so the title shows up there too.
      if (wasNewChat) {
        const refreshed = await listChats();
        setChats(refreshed);
      } else {
        setChats((prev) =>
          prev.map((c) =>
            c.chat_id === chatId
              ? { ...c, updated_at: new Date().toISOString() }
              : c,
          ),
        );
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
  };

  if (isLoadingChats) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background text-sm text-foreground-muted">
        Loading...
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      <ChatHistorySidebar
        chats={chats.map((c) => ({ id: c.chat_id, title: c.title }))}
        activeChatId={activeChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onRenameChat={handleRenameChat}
        onDeleteChat={handleDeleteChat}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-6 py-3">
          <span className="text-sm font-medium text-foreground-muted">
            {activeChat?.title ?? "Ask My Papers"}
          </span>
          <ThemeToggle />
        </header>

        {error && (
          <div className="border-b border-danger bg-background-highlight px-6 py-2 text-xs text-danger">
            {error}
          </div>
        )}

        {isLoadingMessages ? (
          <div className="flex flex-1 items-center justify-center text-sm text-foreground-muted">
            Loading messages...
          </div>
        ) : (
          <ChatWindow messages={messages} isLoading={isSending} />
        )}

        <InputBar onSend={handleSend} isLoading={isSending} />
      </div>

      <DocumentsSidebar
        documents={documents}
        onUpload={handleUploadDocument}
        onDelete={handleDeleteDocument}
        isLoading={isLoadingDocuments}
      />
    </div>
  );
}
