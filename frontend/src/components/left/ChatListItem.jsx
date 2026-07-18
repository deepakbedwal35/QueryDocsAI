import { useEffect, useRef, useState } from "react";
import { MoreHorizontal } from "lucide-react";
import { ChatOptionsMenu } from "./ChatOptionsMenu";

export function ChatListItem({ chat, isActive, onSelect, onRename, onDelete }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [draftTitle, setDraftTitle] = useState(chat.title);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isEditing) inputRef.current?.focus();
  }, [isEditing]);

  const commitRename = () => {
    const trimmed = draftTitle.trim();
    if (trimmed) onRename(chat.id, trimmed);
    setIsEditing(false);
  };

  const cancelRename = () => {
    setDraftTitle(chat.title);
    setIsEditing(false);
  };

  if (isEditing) {
    return (
      <div className="flex items-center gap-1 rounded-md px-2 py-1.5">
        <input
          ref={inputRef}
          value={draftTitle}
          onChange={(e) => setDraftTitle(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitRename();
            if (e.key === "Escape") cancelRename();
          }}
          className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground focus:outline-none"
        />
      </div>
    );
  }

  return (
    <div className="group relative">
      <button
        onClick={() => onSelect(chat.id)}
        className={`flex w-full items-center rounded-md py-2 pl-3 pr-8 text-left text-sm transition-colors cursor-pointer ${
          isActive
            ? "bg-primary text-background"
            : "text-foreground-muted hover:bg-background-highlight hover:text-foreground"
        }`}
      >
        <span className="truncate">{chat.title}</span>
      </button>

      <button
        onClick={(e) => {
          e.stopPropagation();
          setMenuOpen((open) => !open);
        }}
        aria-label="Chat options"
        className={`absolute right-1 top-1/2 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded transition-opacity cursor-pointer ${
          isActive ? "text-background" : "text-foreground-muted"
        } ${menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100"} `}
      >
        <MoreHorizontal size={14} />
      </button>

      {menuOpen && (
        <ChatOptionsMenu
          onRename={() => {
            setMenuOpen(false);
            setIsEditing(true);
          }}
          onDelete={() => {
            setMenuOpen(false);
            onDelete(chat.id);
          }}
          onClose={() => setMenuOpen(false)}
        />
      )}
    </div>
  );
}
