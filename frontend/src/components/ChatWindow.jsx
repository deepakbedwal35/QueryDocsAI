import { useEffect, useRef } from "react";

// Citations are already rendered as pills below the message, so the
// inline [chunk: id] tags the model leaves in the text are stripped
// here for display only (the raw text with tags is still what's
// stored in state / sent to any logging).
const CITATION_TAG_PATTERN = /\s?\[chunk:\s*[\w-]+\s*\]/g;

function stripCitationTags(text) {
  return text
    .replace(CITATION_TAG_PATTERN, "")
    .replace(/\s+([.,;:])/g, "$1")
    .trim();
}

export default function ChatWindow({ messages, isLoading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center px-6">
        <div className="text-center">
          <p className="font-heading text-lg font-bold text-foreground">
            Ask something about your papers
          </p>
          <p className="mt-1 text-sm text-foreground-muted">
            Answers are grounded in the documents on the right, with citations.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-2xl flex-col gap-4 px-6 py-8">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-primary text-background"
                  : msg.isError
                    ? "border border-danger bg-background-highlight text-danger"
                    : "border border-border bg-background-highlight text-foreground"
              }`}
            >
              <p className="whitespace-pre-wrap">
                {msg.role === "assistant"
                  ? stripCitationTags(msg.text)
                  : msg.text}
              </p>
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {msg.citations.map((c) => (
                    <span
                      key={c.chunk_id}
                      title={c.excerpt}
                      className="rounded-full px-2 py-0.5 text-xs text-foreground-muted"
                    >
                      {c.paper_title}
                      {c.page != null ? `, p.${c.page}` : ""}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-2xl  bg-background-highlight px-4 py-2.5 text-sm text-foreground-muted">
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
