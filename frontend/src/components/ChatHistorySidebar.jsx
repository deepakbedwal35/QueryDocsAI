import { Plus } from "lucide-react";
import { ChatListItem } from "./left/ChatListItem";
export default function ChatHistorySidebar({
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
  onRenameChat,
  onDeleteChat,
}) {
  return (
    <aside className="flex h-full w-64 shrink-0 flex-col bg-background-muted">
      <div className="flex items-center justify-between px-4 py-4">
        <span className="font-heading text-sm font-bold tracking-wide text-foreground">
          Ask My Papers
        </span>
      </div>

      <div className="px-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground transition-colors hover:bg-background-highlight cursor-pointer"
        >
          <Plus size={16} />
          New chat
        </button>
      </div>

      <nav className="mt-4 flex-1 space-y-1 overflow-y-auto px-3">
        <p className="px-1 pb-1 text-xs font-medium uppercase tracking-wide text-foreground-muted">
          Recent
        </p>
        {chats.map((chat) => (
          <ChatListItem
            key={chat.id}
            chat={chat}
            isActive={chat.id === activeChatId}
            onSelect={onSelectChat}
            onRename={onRenameChat}
            onDelete={onDeleteChat}
          />
        ))}
      </nav>
    </aside>
  );
}
