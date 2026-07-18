const STORAGE_KEY = "askMyPapers.deviceId";

function generateUuid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  // Fallback for older browsers without crypto.randomUUID
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Returns this browser's persistent anonymous device id, creating
 * and storing one on first call. This is the only "identity" the
 * app has — no login. Sent as X-Device-Id on every API request.
 */
export function getDeviceId() {
  let id = localStorage.getItem(STORAGE_KEY);
  if (!id) {
    id = generateUuid();
    localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}
