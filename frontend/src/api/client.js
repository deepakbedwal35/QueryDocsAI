import { getDeviceId } from "./deviceId";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Shared fetch wrapper: base URL, JSON handling, and the X-Device-Id
 * header every backend route requires. Throws on non-2xx with the
 * backend's error detail when available.
 */
export async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-Device-Id": getDeviceId(),
      ...options.headers,
    },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON, fall back to statusText
    }
    throw new Error(`Request failed (${res.status}): ${detail}`);
  }

  if (res.status === 204) return null;
  return res.json();
}
