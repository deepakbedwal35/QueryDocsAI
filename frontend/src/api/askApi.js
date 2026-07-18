import { apiFetch } from "./client";

/**
 * Calls POST /ask on the backend for a given chat.
 *
 * @param {string} chatId
 * @param {string} question
 * @returns {Promise<{answer: string, citations: Array, answer_found: boolean}>}
 */
export function askQuestion(chatId, question) {
  return apiFetch("/ask", {
    method: "POST",
    body: JSON.stringify({ chat_id: chatId, question }),
  });
}
