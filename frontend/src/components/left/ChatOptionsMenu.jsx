import { useEffect, useRef } from "react";
import { Pencil, Trash2 } from "lucide-react";

export function ChatOptionsMenu({ onRename, onDelete, onClose }) {
  const menuRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [onClose]);

  return (
    <div
      ref={menuRef}
      className="absolute right-0 top-8 z-10 w-36 overflow-hidden rounded-md bg-background shadow-md"
    >
      <button
        onClick={onRename}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-foreground transition-colors hover:bg-background-highlight cursor-pointer"
      >
        <Pencil size={14} />
        Rename
      </button>
      <button
        onClick={onDelete}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-danger transition-colors hover:bg-background-highlight cursor-pointer"
      >
        <Trash2 size={14} />
        Delete
      </button>
    </div>
  );
}
