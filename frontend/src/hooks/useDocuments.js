import { useState, useEffect, useCallback } from "react";
import {
  listDocuments,
  uploadDocument,
  deleteDocument,
} from "../api/documentsApi";
import { useAsyncAction } from "./useAsyncAction";

/**
 * Owns the uploaded-document list for whichever chat is active, plus
 * upload/delete actions. Independent of useMessages/useChats other
 * than needing the active chat id.
 */
export function useDocuments(activeChatId) {
  const [documents, setDocuments] = useState([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const { run, error } = useAsyncAction();

  // Load documents whenever the active chat changes.
  useEffect(() => {
    if (!activeChatId) return;
    (async () => {
      setIsLoadingDocuments(true);
      await run(async () => {
        const docs = await listDocuments(activeChatId);
        setDocuments(docs);
      });
      setIsLoadingDocuments(false);
    })();
  }, [activeChatId, run]);

  const handleUploadDocument = useCallback(
    async (file) => {
      await run(async () => {
        const doc = await uploadDocument(activeChatId, file);
        setDocuments((prev) => [...prev, doc]);
      });
    },
    [activeChatId, run],
  );

  const handleDeleteDocument = useCallback(
    async (docId) => {
      await run(async () => {
        await deleteDocument(activeChatId, docId);
        setDocuments((prev) => prev.filter((d) => d.document_id !== docId));
      });
    },
    [activeChatId, run],
  );

  return {
    documents,
    isLoadingDocuments,
    error,
    handleUploadDocument,
    handleDeleteDocument,
  };
}