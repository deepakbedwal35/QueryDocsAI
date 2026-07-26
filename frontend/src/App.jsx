import ChatHistorySidebar from "./components/ChatHistorySidebar";
import ChatWindow from "./components/ChatWindow";
import InputBar from "./components/InputBar";
import DocumentsSidebar from "./components/DocumentsSidebar";
import ThemeToggle from "./components/ThemeToggle";
import { useChats } from "./hooks/useChats";
import { useMessages } from "./hooks/useMessages";
import { useDocuments } from "./hooks/useDocuments";

export default function App() {
  const {
    chats, activeChatId, activeChat, isLoadingChats, error: chatsError, handleNewChat, handleSelectChat,
    handleRenameChat, handleDeleteChat, refreshChats,touchChat,
  } = useChats();

  const {
    messages,
    isLoadingMessages,
    isSending,
    error: messagesError,
    handleSend,
  } = useMessages(activeChatId, {
    isNewChat: activeChat?.title === "New chat",
    onFirstMessageSent: refreshChats,
    onMessageSent: touchChat,
  });

  const {
    documents,
    isLoadingDocuments,
    error: documentsError,
    handleUploadDocument,
    handleDeleteDocument,
  } = useDocuments(activeChatId);

  // Any of the three hooks can produce an error; show whichever fired
  // most recently. See the props note for why this isn't one shared
  // error state instead.
  const error = messagesError || documentsError || chatsError;

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
            {activeChat?.title ?? "QueryDocsAI"}
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