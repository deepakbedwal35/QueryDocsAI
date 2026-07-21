import { useState } from "react";
import { Send } from "lucide-react";

export default function InputBar({ onSend, isLoading }) {
  const [value, setValue] = useState("");

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="shrink-0 px-6 py-4">
      <div className="mx-auto flex max-w-2xl items-end gap-2 rounded-2xl border border-border bg-background-highlight px-3 py-2 shadow-sm">
        <textarea
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your papers..."
          disabled={isLoading}
          className="max-h-40 flex-1 resize-none bg-transparent py-1.5 text-sm text-foreground focus:outline-none disabled:opacity-60"
        />

        <button
          onClick={handleSend}
          disabled={!value.trim() || isLoading}
          aria-label="Send message"
          className="flex h-8 w-8 p-1 shrink-0 items-center justify-center rounded-full bg-primary text-background transition-colors hover:bg-primary-hover disabled:opacity-40 cursor-pointer"
        >
          <Send size={16} strokeWidth={2.5} />
        </button>
      </div>
    </div>
  );
}
