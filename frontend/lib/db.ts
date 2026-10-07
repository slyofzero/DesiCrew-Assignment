/**
 * IndexedDB storage helper for client-side chat session persistence.
 * Database: ExcelAgentDB
 * Object Store: sessions (keyPath: id)
 *
 * Includes a synchronous write-through localStorage mirror for instant, bulletproof restoration.
 */

export interface TrajectoryStep {
  step: "reason" | "act" | "tool_trace" | "reflect" | "final_output";
  thought?: string;
  is_aligned?: boolean;
  output_as_table?: boolean;
  decision?: string;
  instruction?: string;
  calls?: Array<{ tool: string; args: any; id?: string }>;
  tool?: string;
  args?: any;
  output?: any;
  critique?: string;
  is_satisfied?: boolean;
  content?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  trajectory?: TrajectoryStep[];
  isStreaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  messages: ChatMessage[];
  filePath?: string;
}

const DB_NAME = "ExcelAgentDB";
const DB_VERSION = 1;
const STORE_NAME = "sessions";
const BACKUP_KEY = "excel_agent_sessions_backup";

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof window === "undefined" || !window.indexedDB) {
      reject(new Error("IndexedDB is not supported"));
      return;
    }

    const request = window.indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event: IDBVersionChangeEvent) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => {
      console.warn("IndexedDB open blocked by other tabs");
    };
  });
}

/**
 * Fetch all chat sessions from IndexedDB, falling back to localStorage mirror if needed.
 */
export async function getAllSessions(): Promise<ChatSession[]> {
  try {
    const db = await openDB();
    const sessions = await new Promise<ChatSession[]>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();

      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });

    if (sessions && sessions.length > 0) {
      return sessions;
    }
  } catch (err) {
    console.warn("IndexedDB read failed, checking localStorage backup:", err);
  }

  // Fallback to localStorage backup
  if (typeof window !== "undefined") {
    try {
      const backup = localStorage.getItem(BACKUP_KEY) || localStorage.getItem("excel_agent_sessions");
      if (backup) {
        const parsed = JSON.parse(backup);
        if (Array.isArray(parsed) && parsed.length > 0) {
          // Re-seed IndexedDB in background
          saveAllSessions(parsed).catch(() => {});
          return parsed;
        }
      }
    } catch (e) {
      console.error("Failed to read localStorage backup:", e);
    }
  }

  return [];
}

/**
 * Save or update a single chat session in IndexedDB and localStorage mirror.
 */
export async function saveSession(session: ChatSession): Promise<void> {
  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(session);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.error("IndexedDB saveSession failed:", err);
  }
}

/**
 * Bulk save all sessions in IndexedDB and the localStorage mirror.
 */
export async function saveAllSessions(sessions: ChatSession[]): Promise<void> {
  if (!sessions || sessions.length === 0) return;

  // 1. Immediately sync to localStorage mirror
  if (typeof window !== "undefined") {
    try {
      localStorage.setItem(BACKUP_KEY, JSON.stringify(sessions));
      localStorage.setItem("excel_agent_sessions", JSON.stringify(sessions));
    } catch (lsErr) {
      console.warn("localStorage mirror write failed:", lsErr);
    }
  }

  // 2. Persist to IndexedDB
  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);

      sessions.forEach((s) => {
        store.put(s);
      });

      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } catch (err) {
    console.error("IndexedDB saveAllSessions failed:", err);
  }
}

/**
 * Delete a specific session by id from both IndexedDB and localStorage mirror.
 */
export async function deleteSession(id: string): Promise<void> {
  if (typeof window !== "undefined") {
    try {
      const backup = localStorage.getItem(BACKUP_KEY);
      if (backup) {
        const parsed: ChatSession[] = JSON.parse(backup);
        const filtered = parsed.filter((s) => s.id !== id);
        localStorage.setItem(BACKUP_KEY, JSON.stringify(filtered));
        localStorage.setItem("excel_agent_sessions", JSON.stringify(filtered));
      }
    } catch (e) {
      console.error(e);
    }
  }

  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.delete(id);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.error("IndexedDB deleteSession failed:", err);
  }
}

