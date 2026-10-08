/**
 * IndexedDB storage helper for Question 2: Support Assistant chat session persistence.
 * Database: SupportAssistantDB
 * Object Store: sessions (keyPath: id)
 */

export interface SupportTrajectoryStep {
  step: "reason" | "act" | "tool_trace" | "reflect" | "final_output";
  thought?: string;
  decision?: string;
  instruction?: string;
  tool?: string;
  args?: any;
  output?: any;
  content?: string;
}

export interface SupportChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  trajectory?: SupportTrajectoryStep[];
  thoughts?: string[];
  retrievedChunks?: Array<{
    source: string;
    section: string;
    score: number;
    content: string;
  }>;
}

export interface SupportChatSession {
  id: string;
  title: string;
  createdAt: string;
  messages: SupportChatMessage[];
}

const DB_NAME = "SupportAssistantDB";
const DB_VERSION = 1;
const STORE_NAME = "sessions";
const BACKUP_KEY = "support_assistant_sessions_backup";

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
      console.warn("SupportAssistantDB open blocked by other tabs");
    };
  });
}

export async function getSupportSessions(): Promise<SupportChatSession[]> {
  try {
    const db = await openDB();
    const sessions = await new Promise<SupportChatSession[]>((resolve, reject) => {
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
    console.warn("SupportAssistantDB read failed, checking localStorage backup:", err);
  }

  if (typeof window !== "undefined") {
    try {
      const backup = localStorage.getItem(BACKUP_KEY);
      if (backup) {
        const parsed = JSON.parse(backup);
        if (Array.isArray(parsed) && parsed.length > 0) {
          saveAllSupportSessions(parsed).catch(() => {});
          return parsed;
        }
      }
    } catch (e) {
      console.error("Failed to read localStorage backup:", e);
    }
  }

  return [];
}

export async function saveSupportSession(session: SupportChatSession): Promise<void> {
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
    console.error("SupportAssistantDB saveSession failed:", err);
  }
}

export async function saveAllSupportSessions(sessions: SupportChatSession[]): Promise<void> {
  if (!sessions || sessions.length === 0) return;

  if (typeof window !== "undefined") {
    try {
      localStorage.setItem(BACKUP_KEY, JSON.stringify(sessions));
    } catch (lsErr) {
      console.warn("localStorage mirror write failed:", lsErr);
    }
  }

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
    console.error("SupportAssistantDB saveAllSessions failed:", err);
  }
}

export async function deleteSupportSession(id: string): Promise<void> {
  if (typeof window !== "undefined") {
    try {
      const backup = localStorage.getItem(BACKUP_KEY);
      if (backup) {
        const parsed: SupportChatSession[] = JSON.parse(backup);
        const filtered = parsed.filter((s) => s.id !== id);
        localStorage.setItem(BACKUP_KEY, JSON.stringify(filtered));
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
    console.error("SupportAssistantDB deleteSession failed:", err);
  }
}
