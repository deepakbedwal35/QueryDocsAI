import {useState, useCallback} from "react"
/**
 * Wraps an async function with a shared error-state pattern, so every
 * handler doesn't need its own try/catch { setError(err.message) }.
 *
 * Usage:
 *   const { run, error, setError } = useAsyncAction();
 *   const handleDelete = (id) => run(() => deleteChat(id));
 *
 * `error` is local to THIS hook instance -- see note in App.jsx about
 * how multiple hooks share one visible error banner.
 */

export function useAsyncAction() {
  const [error, setError] = useState(null);
 
  const run = useCallback(async (fn, { onError } = {}) => {
    try {
      return await fn();
    } catch (err) {
      setError(err.message);
      onError?.(err);
      return undefined;
    }
  }, []);
 
  return { run, error, setError };
}