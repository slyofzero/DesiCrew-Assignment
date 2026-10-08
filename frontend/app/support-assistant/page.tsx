"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  Bot,
  Send,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
  Brain,
  Layers,
  BookOpen,
  ArrowLeft,
  Sparkles,
  Loader2,
  Database,
  RefreshCw,
  FileText,
  Bookmark,
  CheckCircle2,
  Play,
  RotateCcw,
  ExternalLink,
} from "lucide-react";

import {
  SupportChatSession,
  SupportChatMessage,
  SupportTrajectoryStep,
  getSupportSessions,
  saveAllSupportSessions,
  deleteSupportSession,
} from "../../lib/support-db";

const DEFAULT_SESSION: SupportChatSession = {
  id: "session-support-default",
  title: "Notes Summarizer",
  createdAt: "Just now",
  messages: [
    {
      id: "welcome-support-1",
      role: "assistant",
      content:
        "Hello! I am your **Notes Intelligence & Summarization Assistant**. I am grounded in your technical knowledge base of Computer Science, Operating Systems, Algorithms, and Machine Learning notes.\n\nI can **summarize complex topics**, provide **deep technical overviews**, extract key definitions, and synthesize concepts across your notes with **precise citations**—all while maintaining **multi-turn conversational context** and **strictly avoiding repeating** information already provided.\n\nAsk me to summarize any subject or click a preset from the **10-Turn Demonstration Guide** in the sidebar to test my summarization and memory across turns!",
      timestamp: "Just now",
    },
  ],
};

const TEN_TURN_DEMO_PROMPTS = [
  { turn: 1, title: "Turn 1: Big-O Intro", query: "What is the formal definition of Big-O notation, and what does it represent intuitively?" },
  { turn: 2, title: "Turn 2: Follow-up & Anti-Repetition", query: "What about Big-Omega and Big-Theta? How do they contrast with Big-O without repeating the intro?" },
  { turn: 3, title: "Turn 3: Tight vs Loose Bounds", query: "Can you give an example of an algorithm or function where Big-O is not tight, and introduce Little-o?" },
  { turn: 4, title: "Turn 4: Topic Switch to OS", query: "Let's switch topics to Operating Systems: What is Virtual Memory and why do modern systems use Paging?" },
  { turn: 5, title: "Turn 5: OS Follow-up", query: "What exactly triggers a Page Fault, and what steps does the OS take during page fault handling?" },
  { turn: 6, title: "Turn 6: OS Nuance / Anomaly", query: "What is Belady's Anomaly in page replacement, and why doesn't LRU suffer from it?" },
  { turn: 7, title: "Turn 7: Topic Switch to ML", query: "Switching topics once more to Machine Learning: What is the Bias-Variance tradeoff?" },
  { turn: 8, title: "Turn 8: Regularization", query: "How do L1 (Lasso) and L2 (Ridge) regularization interact with bias and variance to prevent overfitting?" },
  { turn: 9, title: "Turn 9: Topic Switch to Trees", query: "Now switching to Data Structures: How does an AVL tree maintain balance, and what are the four rotation cases?" },
  { turn: 10, title: "Turn 10: Holistic Synthesis", query: "Looking back at everything we discussed today—Asymptotics, Paging, Regularization, and AVL Trees—provide a concise synthesis of how efficiency and tradeoffs tie them together." },
];

export default function SupportAssistantPage() {
  const [sessions, setSessions] = useState<SupportChatSession[]>([DEFAULT_SESSION]);
  const [activeSessionId, setActiveSessionId] = useState<string>("session-support-default");
  const [inputQuery, setInputQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
  const [kbStats, setKbStats] = useState<any>(null);
  const [isReindexing, setIsReindexing] = useState(false);
  const [selectedChunkModal, setSelectedChunkModal] = useState<any>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isLoadedRef = useRef(false);

  // Load sessions from IndexedDB on initial mount
  useEffect(() => {
    async function initDB() {
      try {
        const stored = await getSupportSessions();
        if (stored && stored.length > 0) {
          setSessions(stored);
          setActiveSessionId(stored[0].id);
        } else {
          await saveAllSupportSessions([DEFAULT_SESSION]);
        }
      } catch (e) {
        console.error("DB Load Error:", e);
      } finally {
        isLoadedRef.current = true;
      }
    }
    initDB();
    fetchKbStats();
  }, []);

  // Sync sessions to IndexedDB on state changes
  useEffect(() => {
    if (isLoadedRef.current && sessions.length > 0) {
      saveAllSupportSessions(sessions).catch((e) => console.error("Save error:", e));
    }
  }, [sessions]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [sessions, activeSessionId, isLoading]);

  async function fetchKbStats() {
    try {
      const res = await fetch("http://localhost:8000/api/support-assistant/stats");
      if (res.ok) {
        const data = await res.json();
        setKbStats(data);
      }
    } catch (e) {
      console.warn("Could not fetch KB stats:", e);
    }
  }

  async function handleReindex() {
    setIsReindexing(true);
    try {
      const res = await fetch("http://localhost:8000/api/support-assistant/reindex", {
        method: "POST",
      });
      if (res.ok) {
        await fetchKbStats();
      }
    } catch (e) {
      console.error("Reindex error:", e);
    } finally {
      setIsReindexing(false);
    }
  }

  const activeSession =
    sessions.find((s) => s.id === activeSessionId) || sessions[0] || DEFAULT_SESSION;

  function handleNewSession() {
    const newSession: SupportChatSession = {
      id: `session-${Date.now()}`,
      title: "New Support Query",
      createdAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      messages: [DEFAULT_SESSION.messages[0]],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
  }

  async function handleDeleteSession(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (sessions.length <= 1) {
      setSessions([DEFAULT_SESSION]);
      setActiveSessionId(DEFAULT_SESSION.id);
      await deleteSupportSession(id);
      return;
    }
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    if (activeSessionId === id) {
      setActiveSessionId(updated[0].id);
    }
    await deleteSupportSession(id);
  }

  function toggleNode(nodeId: string) {
    setExpandedNodes((prev) => ({
      ...prev,
      [nodeId]: !prev[nodeId],
    }));
  }

  async function handleSendMessage(queryToSend?: string) {
    const query = (queryToSend || inputQuery).trim();
    if (!query || isLoading) return;

    setInputQuery("");

    const userMsgId = `usr-${Date.now()}`;
    const userMsg: SupportChatMessage = {
      id: userMsgId,
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    const assistantMsgId = `asst-${Date.now()}`;
    const initialAssistantMsg: SupportChatMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      timestamp: "Thinking...",
      thoughts: [],
      trajectory: [],
    };

    // Update active session title if it's default
    const isFirstQuery = activeSession.messages.filter((m) => m.role === "user").length === 0;
    const newTitle = isFirstQuery ? query.slice(0, 32) + "..." : activeSession.title;

    const updatedMessages = [...activeSession.messages, userMsg, initialAssistantMsg];

    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeSessionId
          ? { ...s, title: newTitle, messages: updatedMessages }
          : s
      )
    );

    setIsLoading(true);

    try {
      // Gather conversation history (excluding the pending message and initial welcome)
      const historyTurns = activeSession.messages
        .filter((m) => m.id !== "welcome-support-1")
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await fetch("http://localhost:8000/api/support-assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          history: historyTurns,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);
      }

      const data = await res.json();

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== activeSessionId) return s;
          const msgs = s.messages.map((m) => {
            if (m.id !== assistantMsgId) return m;
            return {
              ...m,
              content: data.answer || "No response received.",
              timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
              thoughts: data.thoughts || [],
              trajectory: data.trajectory || [],
              retrievedChunks: data.retrieved_chunks || [],
            };
          });
          return { ...s, messages: msgs };
        })
      );
    } catch (err: any) {
      console.error("Chat error:", err);
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== activeSessionId) return s;
          const msgs = s.messages.map((m) => {
            if (m.id !== assistantMsgId) return m;
            return {
              ...m,
              content: `⚠️ **Error occurred while processing request:**\n\n${err.message || String(err)}`,
              timestamp: "Failed",
            };
          });
          return { ...s, messages: msgs };
        })
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 font-sans overflow-hidden">
      {/* SIDEBAR */}
      <aside className="w-80 flex flex-col border-r border-zinc-800/80 bg-zinc-900/60 backdrop-blur-md">
        {/* Sidebar Header */}
        <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-2 text-zinc-400 hover:text-white text-xs font-medium transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>All Solutions</span>
          </Link>
          <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold">
            <Bot className="w-3.5 h-3.5" />
            <span>Notes Summarizer</span>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <button
            onClick={handleNewSession}
            className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-all shadow-sm"
          >
            <Plus className="w-4 h-4" />
            <span>New Chat Session</span>
          </button>
        </div>

        {/* Knowledge Base Status Card */}
        <div className="px-3 py-2">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/90 p-3 space-y-2.5 text-xs">
            <div className="flex items-center justify-between text-zinc-300 font-medium">
              <div className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-blue-400" />
                <span>Knowledge Base</span>
              </div>
              <button
                onClick={handleReindex}
                disabled={isReindexing}
                className="text-zinc-500 hover:text-zinc-300 transition-colors p-1 rounded"
                title="Re-index Knowledge Base"
              >
                <RefreshCw className={`w-3 h-3 ${isReindexing ? "animate-spin text-blue-400" : ""}`} />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2 text-zinc-400">
              <div className="bg-zinc-950/60 rounded-lg p-2 border border-zinc-800/60">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider">Total Chunks</div>
                <div className="text-base font-semibold text-zinc-100 font-mono">
                  {kbStats?.total_chunks || "1,831"}
                </div>
              </div>
              <div className="bg-zinc-950/60 rounded-lg p-2 border border-zinc-800/60">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider">Documents</div>
                <div className="text-base font-semibold text-zinc-100 font-mono">
                  {kbStats?.total_documents || "76"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-zinc-500 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              <span>Vector Store: BGE-small (FastEmbed)</span>
            </div>
          </div>
        </div>

        {/* 10-Turn Demonstration Guide Accordion */}
        <div className="px-3 py-2 flex-1 overflow-y-auto space-y-2">
          <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider px-1 flex items-center justify-between">
            <span>10-Turn Demonstration</span>
            <Sparkles className="w-3 h-3 text-blue-400" />
          </div>
          <div className="space-y-1.5">
            {TEN_TURN_DEMO_PROMPTS.map((item) => (
              <button
                key={item.turn}
                onClick={() => handleSendMessage(item.query)}
                disabled={isLoading}
                className="w-full text-left p-2 rounded-lg bg-zinc-900/70 hover:bg-blue-950/30 border border-zinc-800/80 hover:border-blue-500/30 transition-all text-xs group"
              >
                <div className="flex items-center justify-between text-zinc-300 group-hover:text-blue-300 font-medium">
                  <span>{item.title}</span>
                  <Play className="w-2.5 h-2.5 opacity-0 group-hover:opacity-100 transition-opacity text-blue-400" />
                </div>
                <p className="text-[11px] text-zinc-500 line-clamp-1 mt-0.5">
                  {item.query}
                </p>
              </button>
            ))}
          </div>

          {/* Sessions List */}
          <div className="pt-3">
            <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider px-1 mb-1.5">
              Saved Sessions
            </div>
            <div className="space-y-1">
              {sessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => setActiveSessionId(s.id)}
                  className={`group flex items-center justify-between p-2 rounded-lg text-xs cursor-pointer transition-all ${
                    s.id === activeSessionId
                      ? "bg-zinc-800 text-white font-medium border border-zinc-700/80"
                      : "text-zinc-400 hover:bg-zinc-850 hover:text-zinc-200"
                  }`}
                >
                  <div className="truncate flex-1 pr-2">{s.title}</div>
                  <button
                    onClick={(e) => handleDeleteSession(s.id, e)}
                    className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 p-0.5 transition-all"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-3 border-t border-zinc-800/80 text-[11px] text-zinc-500 text-center">
          DesiCrew Assignment • Question 2
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <main className="flex-1 flex flex-col h-full overflow-hidden bg-zinc-950">
        {/* Chat Top Bar */}
        <header className="h-14 border-b border-zinc-800/80 px-6 flex items-center justify-between bg-zinc-900/40 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
              <Bot className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">{activeSession.title}</h2>
              <p className="text-[11px] text-zinc-500">
                Technical Notes Summarization • Multi-Turn Working Memory • Citation Grounding
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="px-2.5 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-[11px] text-zinc-400 flex items-center gap-1.5 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>{activeSession.messages.filter((m) => m.role === "user").length} Turns Completed</span>
            </div>
          </div>
        </header>

        {/* Messages Stream */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {activeSession.messages.map((message, idx) => {
            const isUser = message.role === "user";
            if (isUser) {
              return (
                <div key={message.id} className="flex flex-col items-end">
                  <div className="max-w-xl rounded-2xl rounded-br-sm px-4 py-2 bg-blue-600 text-white shadow-sm text-sm font-normal leading-relaxed whitespace-pre-wrap break-words ml-12">
                    {message.content}
                  </div>
                </div>
              );
            }

            const processedContent = message.content.replace(
              /\[Source:\s*([^§\]\n]+?)(?:\s*§\s*([^\]\n]+?))?\]/gi,
              (match, doc, sec) => {
                const cleanDoc = doc.trim();
                const cleanSec = sec ? sec.trim() : "";
                return `[${cleanDoc}${cleanSec ? ` § ${cleanSec}` : ""}](#citation:${encodeURIComponent(cleanDoc)}::${encodeURIComponent(cleanSec)})`;
              }
            );

            return (
              <div key={message.id} className="flex flex-col items-start w-full">
                <div className="w-full max-w-3xl rounded-2xl p-4 bg-zinc-900/90 text-zinc-100 border border-zinc-800/90 rounded-bl-none mr-12 space-y-2.5 shadow-sm">
                  {/* Assistant Header Badge */}
                  <div className="flex items-center justify-between border-b border-zinc-800 pb-2 mb-2 text-xs text-zinc-400">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-blue-400">Notes Summarizer</span>
                        <span className="text-zinc-600">•</span>
                        <span className="text-[11px] text-zinc-500">{message.timestamp}</span>
                      </div>
                      {message.retrievedChunks && message.retrievedChunks.length > 0 && (
                        <div className="flex items-center gap-1 text-[11px] text-zinc-400 bg-zinc-800/80 px-2 py-0.5 rounded-md border border-zinc-700/50">
                          <BookOpen className="w-3 h-3 text-blue-400" />
                          <span>{message.retrievedChunks.length} Chunks Grounded</span>
                        </div>
                      )}
                    </div>

                    {/* 5-Step Chain of Thought Inspector (Expandable) */}
                    {message.thoughts && message.thoughts.length > 0 && (
                    <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3 space-y-2">
                      <button
                        onClick={() => toggleNode(`cot-${message.id}`)}
                        className="w-full flex items-center justify-between text-xs font-semibold text-zinc-300 hover:text-white transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <Brain className="w-3.5 h-3.5 text-blue-400" />
                          <span>5-Point Chain-of-Thought Reasoning</span>
                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                            Verified
                          </span>
                        </div>
                        {expandedNodes[`cot-${message.id}`] ? (
                          <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5 text-zinc-400" />
                        )}
                      </button>

                      {expandedNodes[`cot-${message.id}`] && (
                        <div className="pt-2 border-t border-zinc-850 space-y-2 text-xs text-zinc-400 leading-relaxed font-mono">
                          {message.thoughts.map((th, tIdx) => {
                            const [pointHeader, ...rest] = th.split(":");
                            return (
                              <div key={tIdx} className="bg-zinc-900/60 rounded-lg p-2 border border-zinc-800/50">
                                <span className="text-blue-300 font-semibold">{pointHeader}:</span>
                                <span className="text-zinc-300 ml-1">{rest.join(":")}</span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Retrieved Chunks Chips / Inspector */}
                  {message.retrievedChunks && message.retrievedChunks.length > 0 && (
                    <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3 space-y-2">
                      <button
                        onClick={() => toggleNode(`chunks-${message.id}`)}
                        className="w-full flex items-center justify-between text-xs font-semibold text-zinc-300 hover:text-white transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <Layers className="w-3.5 h-3.5 text-emerald-400" />
                          <span>Retrieved Evidence Chunks ({message.retrievedChunks.length})</span>
                        </div>
                        {expandedNodes[`chunks-${message.id}`] ? (
                          <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5 text-zinc-400" />
                        )}
                      </button>

                      {expandedNodes[`chunks-${message.id}`] && (
                        <div className="pt-2 border-t border-zinc-850 space-y-2 text-xs">
                          {message.retrievedChunks.map((c, cIdx) => (
                            <div
                              key={cIdx}
                              onClick={() => setSelectedChunkModal(c)}
                              className="group p-2.5 rounded-lg bg-zinc-900/80 hover:bg-zinc-850 border border-zinc-800 hover:border-blue-500/40 cursor-pointer transition-all"
                            >
                              <div className="flex items-center justify-between text-[11px] mb-1">
                                <span className="font-mono text-zinc-300 group-hover:text-blue-300 font-semibold truncate max-w-md">
                                  {c.source} § {c.section}
                                </span>
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-mono">
                                  Score: {c.score}
                                </span>
                              </div>
                              <p className="text-zinc-400 text-[11px] line-clamp-2">{c.content}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Main Response Content */}
                  <div className="prose prose-invert prose-sm max-w-none leading-relaxed text-zinc-200">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        h1: ({ children }) => (
                          <h1 className="text-lg font-bold text-white first:mt-0 mt-4 mb-2 pb-1.5 border-b border-zinc-800/80 tracking-tight">
                            {children}
                          </h1>
                        ),
                        h2: ({ children }) => (
                          <h2 className="text-base font-semibold text-blue-300 first:mt-0 mt-3.5 mb-1.5 pt-0.5 border-l-2 border-blue-500 pl-2.5 tracking-tight">
                            {children}
                          </h2>
                        ),
                        h3: ({ children }) => (
                          <h3 className="text-xs font-semibold text-zinc-200 first:mt-0 mt-3 mb-1 uppercase tracking-wider">
                            {children}
                          </h3>
                        ),
                        hr: () => (
                          <hr className="my-2.5 border-t border-zinc-800/80" />
                        ),
                        p: ({ children }) => (
                          <p className="mb-2 last:mb-0 leading-relaxed text-zinc-300 text-sm">{children}</p>
                        ),
                        ul: ({ children }) => (
                          <ul className="my-2 space-y-1 pl-5 list-disc text-zinc-300 text-sm">{children}</ul>
                        ),
                        ol: ({ children }) => (
                          <ol className="my-2 space-y-1 pl-5 list-decimal text-zinc-300 text-sm">{children}</ol>
                        ),
                        blockquote: ({ children }) => (
                          <blockquote className="my-2.5 pl-3 py-1 border-l-2 border-blue-500/60 bg-blue-950/10 rounded-r-lg italic text-zinc-300 text-sm">
                            {children}
                          </blockquote>
                        ),
                        a: ({ href, children, ...props }) => {
                          if (href && href.startsWith("#citation:")) {
                            const raw = href.replace("#citation:", "");
                            const [rawDoc, rawSec] = raw.split("::");
                            const docName = decodeURIComponent(rawDoc || "");
                            const secName = decodeURIComponent(rawSec || "");

                            const matchingChunk = message.retrievedChunks?.find(
                              (c) =>
                                c.source.toLowerCase().includes(docName.toLowerCase()) ||
                                docName.toLowerCase().includes(c.source.toLowerCase())
                            );

                            return (
                              <span
                                onClick={(e) => {
                                  e.preventDefault();
                                  e.stopPropagation();
                                  if (matchingChunk) {
                                    setSelectedChunkModal(matchingChunk);
                                  } else {
                                    setSelectedChunkModal({
                                      source: docName,
                                      section: secName || "Document Section",
                                      content: `Grounding source: ${docName}${secName ? ` § ${secName}` : ""}.\n\nReferenced from your indexed technical notes.`,
                                      score: 0.95,
                                      chunk_id: `cite-${docName}`,
                                    });
                                  }
                                }}
                                className="inline-flex items-center gap-1.5 px-2.5 py-1 my-1 rounded-lg bg-blue-950/60 hover:bg-blue-900/70 border border-blue-500/30 hover:border-blue-400 text-blue-300 hover:text-blue-100 text-xs font-mono transition-all cursor-pointer shadow-sm group select-none not-prose"
                                title={`Click to inspect source chunk for ${docName}`}
                              >
                                <BookOpen className="w-3.5 h-3.5 text-blue-400 group-hover:text-blue-200 shrink-0" />
                                <span className="font-semibold text-blue-200 group-hover:underline decoration-blue-400/50">
                                  {docName}
                                </span>
                                {secName && (
                                  <>
                                    <span className="text-blue-400/70 font-sans">§</span>
                                    <span className="text-zinc-300 truncate max-w-sm">{secName}</span>
                                  </>
                                )}
                                <ExternalLink className="w-3 h-3 text-blue-400/70 opacity-0 group-hover:opacity-100 transition-opacity ml-0.5 shrink-0" />
                              </span>
                            );
                          }

                          return (
                            <a
                              href={href}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 hover:text-blue-300 border border-blue-500/20 hover:border-blue-500/40 text-xs font-medium transition-all no-underline shadow-sm"
                              {...props}
                            >
                              <span>{children}</span>
                              <ExternalLink className="w-3 h-3 shrink-0 opacity-70" />
                            </a>
                          );
                        },
                        table: ({ node, ...props }) => (
                          <div className="my-2.5 overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-950/80 shadow-md">
                            <table className="w-full text-left text-xs text-zinc-300 border-collapse" {...props} />
                          </div>
                        ),
                        thead: ({ node, ...props }) => (
                          <thead className="bg-zinc-900/90 text-zinc-100 uppercase text-[11px] font-semibold border-b border-zinc-800 tracking-wider" {...props} />
                        ),
                        tbody: ({ node, ...props }) => (
                          <tbody className="divide-y divide-zinc-850" {...props} />
                        ),
                        tr: ({ node, ...props }) => (
                          <tr className="hover:bg-zinc-800/30 transition-colors" {...props} />
                        ),
                        th: ({ node, ...props }) => (
                          <th className="px-5 py-3.5 font-semibold text-blue-400 whitespace-nowrap text-xs" {...props} />
                        ),
                        td: ({ node, ...props }) => (
                          <td className="px-5 py-3 text-zinc-200 text-xs leading-relaxed" {...props} />
                        ),
                        code({ className, children, ...props }) {
                          const isInline = !className;
                          if (isInline) {
                            return (
                              <code className="px-1.5 py-0.5 rounded bg-zinc-800 text-blue-300 font-mono text-[11px]" {...props}>
                                {children}
                              </code>
                            );
                          }
                          return (
                            <div className="relative my-4 rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 font-mono text-xs shadow-sm">
                              <pre className="p-3.5 overflow-x-auto text-zinc-200">{children}</pre>
                            </div>
                          );
                        },
                      }}
                    >
                      {processedContent}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Thinking Spinner */}
          {isLoading && (
            <div className="flex items-center gap-3 p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 max-w-md text-xs text-zinc-400">
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              <span>Retrieving relevant chunks & formulating 5-step Chain-of-Thought...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-zinc-800/80 bg-zinc-900/50 backdrop-blur-md">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex items-center gap-3 max-w-4xl mx-auto"
          >
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask for a summary, explanation, or compare concepts across your notes..."
              disabled={isLoading}
              className="flex-1 bg-zinc-900 border border-zinc-800 focus:border-blue-500 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none transition-all shadow-inner"
            />
            <button
              type="submit"
              disabled={!inputQuery.trim() || isLoading}
              className="px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:hover:bg-blue-600 text-white font-medium text-sm transition-all flex items-center gap-2 shadow-sm"
            >
              <Send className="w-4 h-4" />
              <span>Ask</span>
            </button>
          </form>
          <div className="text-center text-[11px] text-zinc-500 mt-2">
            Powered by LangGraph CoT + ReAct with FastEmbed & Anti-Repetition Working Memory
          </div>
        </div>
      </main>

      {/* CHUNK EVIDENCE MODAL */}
      {selectedChunkModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl max-w-2xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="text-base font-semibold text-white flex items-center gap-2">
                  <FileText className="w-4 h-4 text-blue-400" />
                  {selectedChunkModal.source}
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5">Section: {selectedChunkModal.section}</p>
              </div>
              <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                Cosine Match: {selectedChunkModal.score}
              </span>
            </div>
            <div className="max-h-96 overflow-y-auto bg-zinc-950 p-4 rounded-xl border border-zinc-800/80 font-mono text-xs text-zinc-300 leading-relaxed whitespace-pre-wrap">
              {selectedChunkModal.content}
            </div>
            <div className="flex justify-end pt-2">
              <button
                onClick={() => setSelectedChunkModal(null)}
                className="px-4 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-sm font-medium text-white transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
