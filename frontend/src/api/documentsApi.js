import { getDeviceId } from "./deviceId";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Upload a PDF document to a chat.
 * Uses raw fetch instead of apiFetch because we need multipart/form-data,
 * not application/json.
 */
export async function uploadDocument(chatId, file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE_URL}/chats/${chatId}/documents`, {
    method: "POST",
    headers: {
      "X-Device-Id": getDeviceId(),
    },
    body: form,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // not JSON, use statusText
    }
    throw new Error(`Upload failed (${res.status}): ${detail}`);
  }

  return res.json();
}

/**
 * List documents uploaded to a chat.
 */
export async function listDocuments(chatId) {
  const res = await fetch(`${API_BASE_URL}/chats/${chatId}/documents`, {
    headers: {
      "Content-Type": "application/json",
      "X-Device-Id": getDeviceId(),
    },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // not JSON
    }
    throw new Error(`Request failed (${res.status}): ${detail}`);
  }

  return res.json();
}

/**
 * Delete a document from a chat.
 */
export async function deleteDocument(chatId, docId) {
  const res = await fetch(
    `${API_BASE_URL}/chats/${chatId}/documents/${docId}`,
    {
      method: "DELETE",
      headers: {
        "X-Device-Id": getDeviceId(),
      },
    },
  );

  if (!res.ok && res.status !== 204) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // not JSON
    }
    throw new Error(`Delete failed (${res.status}): ${detail}`);
  }
}
