import { apiFetch } from "./client";

/** GET /chats — list this device's chats, most recently updated first. */
export function listChats() {
  return apiFetch("/chats");
}

/** POST /chats — create a new chat. */
export function createChat(title = "New chat") {
  return apiFetch("/chats", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

/** PATCH /chats/{id} — rename a chat. */
export function renameChat(chatId, title) {
  return apiFetch(`/chats/${chatId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

/** DELETE /chats/{id} — delete a chat (cascades to its messages). */
export function deleteChat(chatId) {
  return apiFetch(`/chats/${chatId}`, { method: "DELETE" });
}

/** GET /chats/{id}/messages — full message history for a chat. */
export function getChatMessages(chatId) {
  return apiFetch(`/chats/${chatId}/messages`);
}
