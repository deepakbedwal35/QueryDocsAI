import { File, Plus } from "lucide-react";

const dummyDocs = [
  { id: "1", name: "wiersma-2017-consciousness.pdf", pages: 12 },
  { id: "2", name: "koch-tononi-2016-ncc-review.pdf", pages: 20 },
  { id: "3", name: "alkire-2019-anesthesia.pdf", pages: 8 },
];

function FileIcon() {
  return <File className="h-4 w-4 shrink-0" />;
}

export default function DocumentsSidebar() {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col bg-background-muted">
      <div className="px-4 py-4">
        <span className="font-heading text-sm font-bold tracking-wide text-foreground">
          Documents
        </span>
        <p className="mt-0.5 text-xs text-foreground-muted">
          {dummyDocs.length} files in this chat
        </p>
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto px-3">
        {dummyDocs.map((doc) => (
          <div
            key={doc.id}
            className="flex items-start gap-2 rounded-md  px-3 py-2 text-sm text-foreground-muted transition-colors hover:border-border hover:bg-background-highlight hover:text-foreground cursor-pointer"
          >
            <FileIcon />
            <div className="min-w-0">
              <p className="truncate">{doc.name}</p>
              <p className="text-xs text-foreground-muted">{doc.pages} pages</p>
            </div>
          </div>
        ))}
      </div>

      <div className=" p-3">
        <button className="flex w-full items-center justify-center gap-2 rounded-md border border-dashed border-border-muted px-3 py-2 text-sm text-foreground-muted transition-colors hover:border-primary hover:bg-background hover:text-foreground cursor-pointer">
          <Plus className="h-4 w-4" />
          Upload document
        </button>
      </div>
    </aside>
  );
}
