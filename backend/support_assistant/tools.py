import fnmatch
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

try:
    from support_assistant.vector_store import DEFAULT_DATA_DIR, vector_store
except ImportError:
    from vector_store import DEFAULT_DATA_DIR, vector_store

KB_ROOT = Path(DEFAULT_DATA_DIR).resolve()


def _sanitize_path(subpath: str) -> Path:
    """Ensure paths remain within the knowledge base root for safety."""
    cleaned = subpath.strip().replace("\\", "/").lstrip("/")
    target = (KB_ROOT / cleaned).resolve()
    if not str(target).startswith(str(KB_ROOT)):
        return KB_ROOT
    return target


@tool
def list_directory(subpath: str = "") -> dict[str, Any]:
    """List subdirectories and files in the knowledge base directory or subfolder.
    Use this tool when you need to discover available topics, categories, or notes.
    Args:
        subpath: Relative folder path inside the knowledge base (e.g., 'Algorithms', 'Operating Systems', or '' for root).
    """
    target = _sanitize_path(subpath)
    if not target.exists():
        return {"error": f"Directory not found: {subpath}"}
    if not target.is_dir():
        return {"error": f"Path is not a directory: {subpath}"}

    directories = []
    files = []
    excluded = {".git", ".obsidian", ".agent", ".venv", "__pycache__", "node_modules", "imgs", "drawings", "templates"}

    try:
        for entry in sorted(target.iterdir()):
            if entry.name.startswith(".") or entry.name.lower() in excluded:
                continue
            rel = str(entry.relative_to(KB_ROOT)).replace("\\", "/")
            if entry.is_dir():
                directories.append(rel)
            elif entry.suffix.lower() in {".md", ".txt", ".pdf", ".ipynb"}:
                files.append(rel)

        return {
            "current_path": str(target.relative_to(KB_ROOT)).replace("\\", "/") if target != KB_ROOT else "/",
            "directories": directories[:50],
            "files": files[:80],
            "total_files": len(files),
            "total_dirs": len(directories),
        }
    except Exception as err:
        return {"error": f"Failed listing directory: {err}"}


@tool
def read_document(file_path: str, start_line: int = 1, max_lines: int = 80) -> dict[str, Any]:
    """Read the content of a specific document or notes file within the knowledge base.
    Use this when you need deeper context, the full section, or code/math examples from a specific file.
    Args:
        file_path: Relative path to the file (e.g. 'Algorithms/Asymptotic Notation.md').
        start_line: 1-indexed starting line number (default 1).
        max_lines: Maximum number of lines to return (default 80, max 150).
    """
    target = _sanitize_path(file_path)
    if not target.exists() or not target.is_file():
        return {"error": f"File does not exist: {file_path}"}

    ext = target.suffix.lower()
    if ext not in {".md", ".txt", ".pdf", ".ipynb"}:
        return {"error": f"Unsupported file type: {ext}"}

    max_lines = min(max_lines, 150)

    try:
        if ext in {".md", ".txt"}:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            total_lines = len(lines)
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, start_idx + max_lines)
            selected = lines[start_idx:end_idx]

            return {
                "file_path": str(target.relative_to(KB_ROOT)).replace("\\", "/"),
                "total_lines": total_lines,
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "content": "\n".join(selected),
            }

        elif ext == ".pdf":
            import pymupdf

            doc = pymupdf.open(target)
            total_pages = len(doc)
            page_text = ""
            for p in range(min(5, total_pages)):
                page_text += f"\n--- Page {p + 1} ---\n" + doc[p].get_text()
            doc.close()
            return {
                "file_path": str(target.relative_to(KB_ROOT)).replace("\\", "/"),
                "total_pages": total_pages,
                "content": page_text[:3000],
            }

        return {"error": f"Cannot format file: {file_path}"}
    except Exception as err:
        return {"error": f"Failed reading file {file_path}: {err}"}


@tool
def search_documents_by_name(pattern: str) -> dict[str, Any]:
    """Find notes or document files matching a filename keyword or wildcard pattern.
    Use this when you want to quickly locate files relevant to a subject (e.g. 'Sorting', 'Tree', 'Bayes', 'Memory').
    Args:
        pattern: Keyword or wildcard pattern to match against filenames (e.g. '*Sorting*' or 'Dynamic').
    """
    if not pattern:
        return {"matches": []}

    search_pat = f"*{pattern.strip()}*" if not any(c in pattern for c in ["*", "?"]) else pattern
    matches = []
    excluded = {".git", ".obsidian", ".agent", ".venv", "__pycache__", "node_modules", "imgs", "drawings", "templates"}

    try:
        for root, dirs, files in os.walk(KB_ROOT):
            dirs[:] = [d for d in dirs if d.lower() not in excluded and not d.startswith(".")]
            for file in files:
                if fnmatch.fnmatch(file.lower(), search_pat.lower()):
                    fp = Path(root) / file
                    rel = str(fp.relative_to(KB_ROOT)).replace("\\", "/")
                    matches.append(rel)

        return {
            "query_pattern": pattern,
            "total_matches": len(matches),
            "matches": matches[:30],
        }
    except Exception as err:
        return {"error": f"Failed searching file names: {err}"}


@tool
def query_knowledge_base(query: str, top_k: int = 5) -> dict[str, Any]:
    """Perform targeted semantic similarity vector search across all indexed notes chunks.
    Use this if you want to look up additional concepts or refined queries beyond the initially retrieved context.
    Args:
        query: Conceptual or technical search query.
        top_k: Number of most relevant chunks to return (default 5).
    """
    try:
        results = vector_store.search(query, top_k=top_k)
        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "source": r["source"],
                    "section": r["section"],
                    "score": r["score"],
                    "content": r["content"],
                }
                for r in results
            ],
        }
    except Exception as err:
        return {"error": f"Semantic vector search failed: {err}"}


ALL_TOOLS = [
    list_directory,
    read_document,
    search_documents_by_name,
    query_knowledge_base,
]
