"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  FileSpreadsheet,
  Upload,
  Send,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
  Brain,
  Wrench,
  CheckCircle2,
  XCircle,
  FileText,
  Loader2,
  Table,
  ArrowLeft,
  Sparkles,
} from "lucide-react";

import {
  ChatSession,
  ChatMessage,
  TrajectoryStep,
  getAllSessions,
  saveAllSessions,
  deleteSession as deleteSessionFromDB,
} from "../../lib/db";

const DEFAULT_SESSION: ChatSession = {
  id: "session-default",
  title: "Inventory Analysis",
  createdAt: "Just now",
  messages: [
    {
      id: "welcome-1",
      role: "assistant",
      content:
        "Hello! I am your Excel Data Intelligence Agent. I can inspect your workbook, run custom Python calculations, retrieve cell ranges, and verify my outputs through self-reflection.\n\nAsk me anything about your dataset or upload an Excel file (`.xlsx`) to get started!",
      timestamp: "Just now",
    },
  ],
  filePath: "data/data.xlsx",
};

export default function ExcelAgentPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([DEFAULT_SESSION]);
  const [activeSessionId, setActiveSessionId] = useState<string>("session-default");
  const [inputQuery, setInputQuery] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string>("data.xlsx");
  const [activeFilePath, setActiveFilePath] = useState<string>("data/data.xlsx");
  const [activeSheetMetadata, setActiveSheetMetadata] = useState<any>(null);
  const [showMetadataModal, setShowMetadataModal] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isLoadedRef = useRef(false);

  // Initialize sessions from IndexedDB on client load
  useEffect(() => {
    async function initDB() {
      try {
        const storedSessions = await getAllSessions();
        if (storedSessions && storedSessions.length > 0) {
          setSessions(storedSessions);
          setActiveSessionId(storedSessions[0].id);
          if (storedSessions[0].filePath) {
            setActiveFilePath(storedSessions[0].filePath);
          }
        } else {
          // If IndexedDB is empty, check legacy localStorage migration or seed DEFAULT_SESSION
          const legacy = localStorage.getItem("excel_agent_sessions");
          if (legacy) {
            try {
              const parsed = JSON.parse(legacy);
              if (Array.isArray(parsed) && parsed.length > 0) {
                setSessions(parsed);
                setActiveSessionId(parsed[0].id);
                if (parsed[0].filePath) setActiveFilePath(parsed[0].filePath);
                await saveAllSessions(parsed);
                isLoadedRef.current = true;
                return;
              }
            } catch (e) {
              console.error("Migration error:", e);
            }
          }
          await saveAllSessions([DEFAULT_SESSION]);
        }
      } catch (err) {
        console.error("Failed to load sessions from IndexedDB:", err);
      } finally {
        isLoadedRef.current = true;
      }
    }
    initDB();
  }, []);

  // Sync sessions to IndexedDB whenever sessions update (ONLY after initial load completes)
  useEffect(() => {
    if (!isLoadedRef.current) return;
    if (sessions.length > 0) {
      saveAllSessions(sessions).catch((err) =>
        console.error("Failed to persist sessions to IndexedDB:", err)
      );
    }
  }, [sessions]);

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [sessions, isStreaming]);

  // Active session helper (always guaranteed to return a valid ChatSession)
  const currentSession =
    sessions.find((s) => s.id === activeSessionId) || sessions[0] || DEFAULT_SESSION;

  // Create new chat
  const handleNewChat = () => {
    const newId = "session-" + Date.now();
    const newSession: ChatSession = {
      id: newId,
      title: `Analysis ${sessions.length + 1}`,
      createdAt: new Date().toLocaleTimeString(),
      messages: [
        {
          id: "welcome-" + Date.now(),
          role: "assistant",
          content:
            "Started a new session. How can I help you analyze your dataset today?",
          timestamp: new Date().toLocaleTimeString(),
        },
      ],
      filePath: activeFilePath,
    };
    const updated = [newSession, ...sessions];
    setSessions(updated);
    setActiveSessionId(newId);
    saveAllSessions(updated);
  };

  // Delete chat
  const handleDeleteChat = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const remaining = sessions.filter((s) => s.id !== id);
    await deleteSessionFromDB(id);

    if (remaining.length === 0) {
      // If user deletes the last/only conversation, reset to a fresh clean session
      const freshSession: ChatSession = {
        id: "session-" + Date.now(),
        title: "Inventory Analysis",
        createdAt: new Date().toLocaleTimeString(),
        messages: [
          {
            id: "welcome-" + Date.now(),
            role: "assistant",
            content:
              "Chat reset. How can I help you analyze your dataset today?",
            timestamp: new Date().toLocaleTimeString(),
          },
        ],
        filePath: activeFilePath,
      };
      setSessions([freshSession]);
      setActiveSessionId(freshSession.id);
      await saveAllSessions([freshSession]);
    } else {
      setSessions(remaining);
      if (activeSessionId === id) {
        setActiveSessionId(remaining[0].id);
      }
      await saveAllSessions(remaining);
    }
  };

  // Fetch metadata
  const handleFetchMetadata = async () => {
    try {
      const res = await fetch(
        `http://localhost:8000/api/excel-agent/metadata?file_path=${encodeURIComponent(
          activeFilePath
        )}`
      );
      if (res.ok) {
        const data = await res.json();
        setActiveSheetMetadata(data);
        setShowMetadataModal(true);
      } else {
        alert("Failed to load metadata.");
      }
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend API (ensure FastAPI is running on port 8000).");
    }
  };

  // Handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    setIsUploading(true);

    try {
      const res = await fetch("http://localhost:8000/api/excel-agent/upload", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploadedFileName(data.filename);
        setActiveFilePath(data.saved_path);

        // Update session's file path
        setSessions((prev) =>
          prev.map((s) =>
            s.id === activeSessionId ? { ...s, filePath: data.saved_path } : s
          )
        );

        // Add confirmation message
        const uploadMsg: ChatMessage = {
          id: "upload-" + Date.now(),
          role: "assistant",
          content: `Successfully uploaded **${data.filename}**. All subsequent queries in this chat will target this workbook.`,
          timestamp: new Date().toLocaleTimeString(),
        };

        setSessions((prev) =>
          prev.map((s) =>
            s.id === activeSessionId
              ? { ...s, messages: [...s.messages, uploadMsg] }
              : s
          )
        );
      } else {
        alert("Upload failed. Please ensure the file is a valid Excel workbook.");
      }
    } catch (err) {
      console.error(err);
      alert("Connection error while uploading to backend.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // Toggle trajectory node expansion
  const toggleNode = (nodeId: string) => {
    setExpandedNodes((prev) => ({
      ...prev,
      [nodeId]: !prev[nodeId],
    }));
  };

  // Send query & stream trajectory
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputQuery.trim() || isStreaming) return;

    const query = inputQuery.trim();
    setInputQuery("");

    // Create user message
    const userMsgId = "user-" + Date.now();
    const userMsg: ChatMessage = {
      id: userMsgId,
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString(),
    };

    // Create placeholder assistant message
    const assistantMsgId = "assistant-" + Date.now();
    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      timestamp: new Date().toLocaleTimeString(),
      trajectory: [],
      isStreaming: true,
    };

    // Update active session title if it's the first query
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          const isFirstQuery = s.messages.filter((m) => m.role === "user").length === 0;
          return {
            ...s,
            title: isFirstQuery ? query.slice(0, 26) + "..." : s.title,
            messages: [...s.messages, userMsg, assistantMsg],
          };
        }
        return s;
      })
    );

    setIsStreaming(true);

    try {
      // Build prior conversation history (only query and final assistant answer pairs)
      const historyPayload = currentSession.messages
        .filter((m) => m.content && m.content.trim() !== "" && !m.id.startsWith("welcome-"))
        .map((m) => ({ role: m.role, content: m.content }));

      const response = await fetch(
        "http://localhost:8000/api/excel-agent/chat/stream",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query,
            file_path: activeFilePath,
            history: historyPayload,
          }),
        }
      );

      if (!response.ok || !response.body) {
        throw new Error("Streaming connection failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      const currentTrajectory: TrajectoryStep[] = [];
      let finalContent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.replace("data: ", "").trim();
          if (!jsonStr) continue;

          try {
            const event = JSON.parse(jsonStr);

            if (event.type === "reason") {
              currentTrajectory.push({
                step: "reason",
                thought: event.thought,
                is_aligned: event.is_aligned,
                output_as_table: event.output_as_table,
                decision: event.decision,
                instruction: event.instruction,
              });
            } else if (event.type === "act") {
              currentTrajectory.push({
                step: "act",
                calls: event.calls,
              });
            } else if (event.type === "tool_trace") {
              if (Array.isArray(event.outputs)) {
                event.outputs.forEach((item: any) => {
                  currentTrajectory.push({
                    step: "tool_trace",
                    tool: item.tool,
                    output: item.output,
                  });
                });
              }
            } else if (event.type === "reflect") {
              currentTrajectory.push({
                step: "reflect",
                critique: event.critique,
                is_satisfied: event.is_satisfied,
              });
            } else if (event.type === "final_output") {
              finalContent = event.answer;
              currentTrajectory.push({
                step: "final_output",
                content: event.answer,
                output_as_table: event.output_as_table,
              });
            }

            // Real-time update into state
            setSessions((prev) =>
              prev.map((s) => {
                if (s.id === activeSessionId) {
                  return {
                    ...s,
                    messages: s.messages.map((m) => {
                      if (m.id === assistantMsgId) {
                        return {
                          ...m,
                          content: finalContent || m.content,
                          trajectory: [...currentTrajectory],
                        };
                      }
                      return m;
                    }),
                  };
                }
                return s;
              })
            );
          } catch (parseErr) {
            console.error("Error parsing SSE JSON:", parseErr, jsonStr);
          }
        }
      }

      // Mark streaming as complete
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === activeSessionId) {
            return {
              ...s,
              messages: s.messages.map((m) =>
                m.id === assistantMsgId ? { ...m, isStreaming: false } : m
              ),
            };
          }
          return s;
        })
      );
    } catch (err: any) {
      console.error("Chat streaming error:", err);
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === activeSessionId) {
            return {
              ...s,
              messages: s.messages.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content:
                        m.content ||
                        "Sorry, I encountered an error communicating with the agent server. Please make sure the FastAPI server is running.",
                      isStreaming: false,
                    }
                  : m
              ),
            };
          }
          return s;
        })
      );
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-emerald-500 selection:text-black overflow-hidden">
      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileUpload}
        accept=".xlsx,.xls"
        className="hidden"
      />

      {/* LEFT SIDEBAR: Sessions & Workbooks */}
      <aside className="w-72 bg-zinc-900/80 border-r border-zinc-800 flex flex-col justify-between shrink-0">
        <div>
          {/* Top Branding */}
          <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Suite Home
            </Link>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Q1 Agent
            </span>
          </div>

          {/* New Chat Button */}
          <div className="p-3">
            <button
              onClick={handleNewChat}
              className="flex items-center justify-center gap-2 w-full py-2.5 px-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm transition-all shadow-lg shadow-emerald-950/20"
            >
              <Plus className="w-4 h-4" />
              New Analysis Session
            </button>
          </div>

          {/* Active Dataset Pill */}
          <div className="px-3 py-2">
            <div className="p-2.5 rounded-xl bg-zinc-800/60 border border-zinc-700/60 text-xs flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <span className="text-zinc-400 font-medium flex items-center gap-1.5">
                  <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
                  Active File
                </span>
                <button
                  onClick={handleFetchMetadata}
                  className="text-[11px] text-emerald-400 hover:underline flex items-center gap-1"
                >
                  <Table className="w-3 h-3" /> Schema
                </button>
              </div>
              <p className="font-mono text-zinc-200 truncate" title={uploadedFileName}>
                {uploadedFileName}
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="mt-1 flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg bg-zinc-700/60 hover:bg-zinc-700 text-zinc-200 text-xs transition-colors"
              >
                {isUploading ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Upload className="w-3.5 h-3.5" />
                )}
                Upload New Excel (.xlsx)
              </button>
            </div>
          </div>

          {/* Sessions List */}
          <div className="px-3 py-2">
            <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider px-2">
              Conversations
            </span>
            <div className="mt-1 space-y-1 max-h-[48vh] overflow-y-auto pr-1">
              {sessions.map((sess) => {
                const isActive = sess.id === activeSessionId;
                return (
                  <div
                    key={sess.id}
                    onClick={() => {
                      setActiveSessionId(sess.id);
                      if (sess.filePath) setActiveFilePath(sess.filePath);
                    }}
                    className={`group flex items-center justify-between px-3 py-2 rounded-xl text-xs cursor-pointer transition-colors ${
                      isActive
                        ? "bg-zinc-800 text-white font-medium border border-zinc-700"
                        : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <Sparkles
                        className={`w-3.5 h-3.5 shrink-0 ${
                          isActive ? "text-emerald-400" : "text-zinc-500"
                        }`}
                      />
                      <span className="truncate">{sess.title}</span>
                    </div>
                    <button
                      onClick={(e) => handleDeleteChat(sess.id, e)}
                      className="opacity-0 group-hover:opacity-100 hover:text-red-400 p-1 transition-opacity"
                      title="Delete chat"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-zinc-800 text-[11px] text-zinc-400 flex items-center justify-between">
          <span>Excel Agent v1.0</span>
          <span className="flex items-center gap-1 text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
            Online
          </span>
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <main className="flex-1 flex flex-col bg-zinc-950 overflow-hidden">
        {/* Top Navbar */}
        <header className="h-14 border-b border-zinc-800/80 px-6 flex items-center justify-between shrink-0 bg-zinc-900/40 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              <FileSpreadsheet className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">
                {currentSession.title}
              </h2>
              <p className="text-[11px] text-zinc-400">
                Reason &rarr; Act &rarr; Tools &rarr; Reflect &rarr; Synthesize
              </p>
            </div>
          </div>
          <button
            onClick={handleFetchMetadata}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700/80 transition-colors"
          >
            <Table className="w-3.5 h-3.5 text-emerald-400" />
            Workbook Schema
          </button>
        </header>

        {/* Chat Messages Log */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {currentSession.messages.map((msg) => {
              const isUser = msg.role === "user";
              return (
                <div
                  key={msg.id}
                  className={`flex gap-4 ${isUser ? "justify-end" : "justify-start"}`}
                >
                  {!isUser && (
                    <div className="w-8 h-8 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center shrink-0 text-emerald-400 mt-1">
                      <Brain className="w-4 h-4" />
                    </div>
                  )}

                  <div className={`flex flex-col max-w-[85%] ${isUser ? "items-end" : "items-start"}`}>
                    {/* User bubble */}
                    {isUser ? (
                      <div className="rounded-2xl px-4 py-2.5 bg-emerald-600 text-white text-sm shadow-md">
                        {msg.content}
                      </div>
                    ) : (
                      <div className="w-full space-y-3">
                        {/* Interactive Agent Trajectory Steps (Placed ABOVE agent response) */}
                        {msg.trajectory && msg.trajectory.length > 0 && (() => {
                          const trajKey = `${msg.id}-trajectory-box`;
                          // If streaming or no final output yet, default to expanded; once completed, default to collapsed
                          const isFinished = Boolean(msg.content && !msg.isStreaming);
                          const isBoxOpen = expandedNodes[trajKey] ?? !isFinished;

                          return (
                            <div className="rounded-xl border border-zinc-800/90 bg-zinc-900/40 p-3 space-y-2 text-xs transition-all">
                              <button
                                onClick={() => toggleNode(trajKey)}
                                className="w-full flex items-center justify-between font-semibold text-zinc-400 hover:text-zinc-200 uppercase tracking-wider text-[10px] transition-colors"
                              >
                                <span className="flex items-center gap-1.5">
                                  <Sparkles className="w-3 h-3 text-emerald-400" />
                                  <span>Agent Execution Trajectory ({msg.trajectory.length} steps)</span>
                                </span>
                                <span className="flex items-center gap-1 text-[11px] font-normal normal-case text-zinc-400">
                                  {isBoxOpen ? (
                                    <>
                                      <span>Hide trace</span>
                                      <ChevronDown className="w-3.5 h-3.5" />
                                    </>
                                  ) : (
                                    <>
                                      <span>View reasoning & tool trace</span>
                                      <ChevronRight className="w-3.5 h-3.5" />
                                    </>
                                  )}
                                </span>
                              </button>

                              {isBoxOpen && (
                                <div className="space-y-1.5 pt-1 border-t border-zinc-800/60 mt-1">
                                  {msg.trajectory.map((step, idx) => {
                                    const stepKey = `${msg.id}-step-${idx}`;
                                    const isExpanded = expandedNodes[stepKey] ?? true;

                                    if (step.step === "reason") {
                                      return (
                                        <div
                                          key={stepKey}
                                          className="rounded-lg bg-zinc-900 border border-zinc-800 p-2.5 transition-all"
                                        >
                                          <button
                                            onClick={() => toggleNode(stepKey)}
                                            className="w-full flex items-center justify-between text-left font-medium text-purple-400"
                                          >
                                            <span className="flex items-center gap-1.5">
                                              <Brain className="w-3.5 h-3.5" />
                                              Reasoning Thought
                                              {step.is_aligned === false ? (
                                                <span className="text-[10px] px-1.5 py-0.2 rounded bg-red-500/10 text-red-400 border border-red-500/20 font-mono">
                                                  Out of Scope
                                                </span>
                                              ) : (
                                                <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                                                  Excel Scope
                                                </span>
                                              )}
                                              {step.output_as_table && (
                                                <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono flex items-center gap-1">
                                                  <Table className="w-2.5 h-2.5" /> Table Format
                                                </span>
                                              )}
                                            </span>
                                            <span className="flex items-center gap-1 text-[10px] text-zinc-400">
                                              Decision: {step.decision}
                                              {isExpanded ? (
                                                <ChevronDown className="w-3 h-3" />
                                              ) : (
                                                <ChevronRight className="w-3 h-3" />
                                              )}
                                            </span>
                                          </button>
                                          {isExpanded && (
                                            <div className="mt-2 text-zinc-300 pl-5 border-l border-purple-500/20 text-[11px] leading-relaxed">
                                              <p>{step.thought}</p>
                                              {step.instruction && (
                                                <p className="mt-1 text-zinc-400 italic">
                                                  Instruction: {step.instruction}
                                                </p>
                                              )}
                                            </div>
                                          )}
                                        </div>
                                      );
                                    }

                                    if (step.step === "act") {
                                      return (
                                        <div
                                          key={stepKey}
                                          className="rounded-lg bg-zinc-900 border border-zinc-800 p-2.5"
                                        >
                                          <button
                                            onClick={() => toggleNode(stepKey)}
                                            className="w-full flex items-center justify-between text-left font-medium text-amber-400"
                                          >
                                            <span className="flex items-center gap-1.5">
                                              <Wrench className="w-3.5 h-3.5" />
                                              Action: Tool Invocation
                                            </span>
                                            {isExpanded ? (
                                              <ChevronDown className="w-3 h-3 text-zinc-400" />
                                            ) : (
                                              <ChevronRight className="w-3 h-3 text-zinc-400" />
                                            )}
                                          </button>
                                          {isExpanded && step.calls && (
                                            <div className="mt-2 pl-5 border-l border-amber-500/20 space-y-1 text-[11px]">
                                              {step.calls.map((call, cIdx) => (
                                                <div key={cIdx} className="font-mono text-zinc-300">
                                                  <span className="text-amber-300 font-semibold">
                                                    {call.tool}
                                                  </span>
                                                  ({JSON.stringify(call.args)})
                                                </div>
                                              ))}
                                            </div>
                                          )}
                                        </div>
                                      );
                                    }

                                    if (step.step === "tool_trace") {
                                      return (
                                        <div
                                          key={stepKey}
                                          className="rounded-lg bg-zinc-900 border border-zinc-800 p-2.5"
                                        >
                                          <button
                                            onClick={() => toggleNode(stepKey)}
                                            className="w-full flex items-center justify-between text-left font-medium text-cyan-400"
                                          >
                                            <span className="flex items-center gap-1.5">
                                              <FileText className="w-3.5 h-3.5" />
                                              Observation Trace: {step.tool}
                                            </span>
                                            {isExpanded ? (
                                              <ChevronDown className="w-3 h-3 text-zinc-400" />
                                            ) : (
                                              <ChevronRight className="w-3 h-3 text-zinc-400" />
                                            )}
                                          </button>
                                          {isExpanded && (
                                            <div className="mt-2 pl-5 border-l border-cyan-500/20">
                                              <pre className="p-2 rounded bg-zinc-950 font-mono text-[10px] text-zinc-300 overflow-x-auto max-h-48 whitespace-pre-wrap">
                                                {typeof step.output === "object"
                                                  ? JSON.stringify(step.output, null, 2)
                                                  : String(step.output)}
                                              </pre>
                                            </div>
                                          )}
                                        </div>
                                      );
                                    }

                                    if (step.step === "reflect") {
                                      return (
                                        <div
                                          key={stepKey}
                                          className={`rounded-lg border p-2.5 ${
                                            step.is_satisfied
                                              ? "bg-emerald-950/20 border-emerald-800/40 text-emerald-400"
                                              : "bg-red-950/20 border-red-800/40 text-red-400"
                                          }`}
                                        >
                                          <div className="flex items-center gap-2 font-medium">
                                            {step.is_satisfied ? (
                                              <CheckCircle2 className="w-3.5 h-3.5" />
                                            ) : (
                                              <XCircle className="w-3.5 h-3.5" />
                                            )}
                                            <span>Critic Self-Reflection Review</span>
                                          </div>
                                          <p className="mt-1 pl-5 text-zinc-300 text-[11px] leading-relaxed">
                                            {step.critique}
                                          </p>
                                        </div>
                                      );
                                    }

                                    return null;
                                  })}
                                </div>
                              )}
                            </div>
                          );
                        })()}

                        {/* Streaming Indicator */}
                        {msg.isStreaming && !msg.content && (
                          <div className="flex items-center gap-2 text-xs text-zinc-400 bg-zinc-900/60 p-3 rounded-xl border border-zinc-800">
                            <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
                            <span>Synthesizing verified observations...</span>
                          </div>
                        )}

                        {/* Assistant Final Content (Markdown View with Rich Table Styling) */}
                        {msg.content && (
                          <div className="rounded-2xl px-5 py-4 bg-zinc-900 border border-zinc-800 text-zinc-200 text-sm leading-relaxed prose prose-invert prose-emerald max-w-none shadow-sm overflow-x-auto">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm, remarkMath]}
                              rehypePlugins={[rehypeKatex]}
                              components={{
                                table: ({ node, ...props }) => (
                                  <div className="my-3 overflow-x-auto rounded-xl border border-zinc-700/80 bg-zinc-950/60 shadow-md">
                                    <table className="w-full text-left text-xs text-zinc-300 border-collapse" {...props} />
                                  </div>
                                ),
                                thead: ({ node, ...props }) => (
                                  <thead className="bg-zinc-800/80 text-zinc-100 uppercase text-[11px] font-semibold border-b border-zinc-700 tracking-wider" {...props} />
                                ),
                                tbody: ({ node, ...props }) => (
                                  <tbody className="divide-y divide-zinc-800" {...props} />
                                ),
                                tr: ({ node, ...props }) => (
                                  <tr className="hover:bg-zinc-800/40 transition-colors" {...props} />
                                ),
                                th: ({ node, ...props }) => (
                                  <th className="px-4 py-3 font-semibold text-emerald-400" {...props} />
                                ),
                                td: ({ node, ...props }) => (
                                  <td className="px-4 py-2.5 whitespace-nowrap text-zinc-200" {...props} />
                                ),
                              }}
                            >
                              {msg.content}
                            </ReactMarkdown>
                          </div>
                        )}
                      </div>
                    )}

                    <span className="text-[10px] text-zinc-400 mt-1 px-1">
                      {msg.timestamp}
                    </span>
                  </div>
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* INPUT PROMPT BAR */}
        <div className="p-4 border-t border-zinc-800/80 bg-zinc-900/50 backdrop-blur-sm shrink-0">
          <form
            onSubmit={handleSendMessage}
            className="max-w-4xl mx-auto relative flex items-center"
          >
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask an analytical question or calculation (e.g. 'What is the median opening stock?' or 'How many gaming items were sold?')"
              disabled={isStreaming}
              className="w-full bg-zinc-900 border border-zinc-700/80 rounded-2xl py-3.5 pl-4 pr-12 text-sm text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 disabled:opacity-50 transition-all shadow-inner"
            />
            <button
              type="submit"
              disabled={isStreaming || !inputQuery.trim()}
              className="absolute right-2 p-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-40 disabled:hover:bg-emerald-600 transition-colors shadow-md"
            >
              {isStreaming ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </form>
          <p className="text-center text-[11px] text-zinc-400 mt-2">
            The agent autonomously runs python code, inspects cells, filters datasets, and performs self-reflection verification.
          </p>
        </div>
      </main>

      {/* SCHEMA / METADATA MODAL */}
      {showMetadataModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6">
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl max-w-2xl w-full max-h-[80vh] flex flex-col overflow-hidden shadow-2xl">
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
              <div className="flex items-center gap-2 text-white font-semibold text-sm">
                <Table className="w-4 h-4 text-emerald-400" />
                Workbook Schema & Columns
              </div>
              <button
                onClick={() => setShowMetadataModal(false)}
                className="text-zinc-400 hover:text-white text-xs px-2 py-1 rounded-lg bg-zinc-800"
              >
                Close
              </button>
            </div>
            <div className="p-4 overflow-y-auto space-y-4">
              <pre className="p-3 rounded-xl bg-zinc-950 font-mono text-xs text-zinc-300 whitespace-pre-wrap">
                {JSON.stringify(activeSheetMetadata, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
