import { useRef, useState } from "react";
import { File, Plus, Trash2, Loader2 } from "lucide-react";

function FileIcon() {
  return <File className="h-4 w-4 shrink-0" />;
}

export default function DocumentsSidebar({
  documents,
  onUpload,
  onDelete,
  isLoading,
}) {
  const fileInputRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    try {
      await onUpload(file);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col bg-background-muted">
      <div className="px-4 py-4">
        <span className="font-heading text-sm font-bold tracking-wide text-foreground">
          Documents
        </span>
        <p className="mt-0.5 text-xs text-foreground-muted">
          {isLoading ? "Loading..." : `${documents.length} files in this chat`}
        </p>
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto px-3">
        {documents.map((doc) => (
          <div
            key={doc.document_id}
            className="group flex items-start gap-2 rounded-md px-3 py-2 text-sm text-foreground-muted transition-colors hover:border-border hover:bg-background-highlight hover:text-foreground"
          >
            <FileIcon />
            <div className="min-w-0 flex-1">
              <p className="truncate">{doc.filename}</p>
              <p className="text-xs text-foreground-muted">
                {doc.page_count != null
                  ? `${doc.page_count} pages, ${doc.chunk_count} chunks`
                  : `${doc.chunk_count} chunks`}
              </p>
            </div>
            <button
              onClick={() => onDelete(doc.document_id)}
              aria-label={`Delete ${doc.filename}`}
              className="mt-0.5 shrink-0 rounded p-0.5 text-foreground-muted opacity-0 transition-opacity hover:text-danger group-hover:opacity-100 cursor-pointer"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>

      <div className="p-3">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileChange}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="flex w-full items-center justify-center gap-2 rounded-md border border-dashed border-border-muted px-3 py-2 text-sm text-foreground-muted transition-colors hover:border-primary hover:bg-background hover:text-foreground disabled:opacity-50 cursor-pointer"
        >
          {isUploading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Plus className="h-4 w-4" />
          )}
          {isUploading ? "Uploading..." : "Upload document"}
        </button>
      </div>
    </aside>
  );
}
