import Link from "next/link";
import { FileSpreadsheet, Bot, BrainCircuit, ArrowRight } from "lucide-react";

export default function Home() {
  const routes = [
    {
      title: "Excel Data Intelligence Agent",
      href: "/excel-agent",
      badge: "Question 1",
      icon: FileSpreadsheet,
      description:
        "Autonomous multi-turn agent capable of data introspection, cell range querying, Python sandbox code execution, statistical computation, and real-time self-reflection loop.",
      color: "from-emerald-500/20 to-teal-500/10 border-emerald-500/30 text-emerald-400",
      buttonColor: "bg-emerald-600 hover:bg-emerald-500 text-white",
    },
    {
      title: "Document-Aware Support Assistant",
      href: "/support-assistant",
      badge: "Question 2",
      icon: Bot,
      description:
        "Context-grounded support agent featuring multi-turn working memory, anti-repetition guardrails, and precise knowledge retrieval.",
      color: "from-blue-500/20 to-indigo-500/10 border-blue-500/30 text-blue-400",
      buttonColor: "bg-blue-600 hover:bg-blue-500 text-white",
    },
    {
      title: "Intelligent Document Processing (IDP)",
      href: "/idp-pipeline",
      badge: "Question 3",
      icon: BrainCircuit,
      description:
        "High-accuracy extraction pipeline with per-field confidence scoring, schema validation, and confidence-driven human-in-the-loop review triage.",
      color: "from-purple-500/20 to-pink-500/10 border-purple-500/30 text-purple-400",
      buttonColor: "bg-purple-600 hover:bg-purple-500 text-white",
    },
  ];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col justify-between selection:bg-emerald-500 selection:text-black">
      <div className="max-w-6xl mx-auto px-6 py-16 w-full">
        {/* Header */}
        <div className="flex flex-col items-center text-center space-y-4 mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-zinc-900 border border-zinc-800 text-zinc-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            DesiCrew Agentic Engineering Assignment
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white max-w-2xl">
            Autonomous Agent & IDP Intelligence Suite
          </h1>
          <p className="text-zinc-400 max-w-xl text-base md:text-lg">
            LangGraph-orchestrated systems featuring dynamic self-reflection, verified execution traces, and human-in-the-loop triage.
          </p>
        </div>

        {/* Feature Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {routes.map((card) => {
            const Icon = card.icon;
            return (
              <div
                key={card.href}
                className="group relative flex flex-col justify-between rounded-2xl bg-zinc-900/60 border border-zinc-800/80 p-6 backdrop-blur-sm transition-all duration-200 hover:border-zinc-700 hover:bg-zinc-900/90"
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-3 rounded-xl bg-zinc-800/80 border border-zinc-700/50 text-zinc-200 group-hover:scale-105 transition-transform">
                      <Icon className="w-6 h-6" />
                    </div>
                    <span className="text-xs px-2.5 py-0.5 rounded-full font-mono bg-zinc-800 text-zinc-400 border border-zinc-700">
                      {card.badge}
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2 group-hover:text-emerald-400 transition-colors">
                    {card.title}
                  </h3>
                  <p className="text-zinc-400 text-sm leading-relaxed mb-6">
                    {card.description}
                  </p>
                </div>

                <Link
                  href={card.href}
                  className={`inline-flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl text-sm font-medium transition-all ${card.buttonColor}`}
                >
                  Open Application
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                </Link>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-zinc-900 py-6 text-center text-xs text-zinc-600">
        Engineered with FastAPI, LangGraph, Python 3.12, and Next.js
      </footer>
    </div>
  );
}
