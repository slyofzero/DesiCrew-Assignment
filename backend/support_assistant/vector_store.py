import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

logger = logging.getLogger(__name__)

# Default knowledge base directories
BACKEND_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BACKEND_DIR / "data" / "support_kb"
DEFAULT_DATA_DIR = os.getenv("RAG_CHATBOT_DATA_DIR", str(BACKEND_DIR / "data"))

# Supported file extensions
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".ipynb"}
EXCLUDED_DIRS = {".git", ".obsidian", ".agent", ".venv", "__pycache__", "node_modules", "imgs", "drawings", "templates"}


class DocumentChunk:
    def __init__(self, chunk_id: str, source: str, section: str, content: str):
        self.chunk_id = chunk_id
        self.source = source
        self.section = section
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.section,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            source=data["source"],
            section=data["section"],
            content=data["content"],
        )


class VectorStore:
    def __init__(self, data_dir: str | Path | None = None, storage_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.storage_dir = Path(storage_dir or STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.storage_dir / "embeddings.npz"
        self.metadata_file = self.storage_dir / "chunks_metadata.json"

        self._embedding_model: TextEmbedding | None = None
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None  # shape (N, D), float32 normalized

        # Attempt to load cached index on initialization
        self.load_index()

    @property
    def embedding_model(self) -> TextEmbedding:
        if self._embedding_model is None:
            # Using lightweight, fast quantized BGE-small (384 dimensions)
            self._embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedding_model

    def _extract_text_and_sections_md(self, text: str, rel_path: str) -> list[tuple[str, str]]:
        """Split Markdown text into sections based on headings and paragraph chunks."""
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []  # list of (section_name, content_block)
        current_section = "Overview"
        current_block: list[str] = []

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if header_match:
                if current_block:
                    joined = "\n".join(current_block).strip()
                    if len(joined) > 20:
                        sections.append((current_section, joined))
                    current_block = []
                current_section = header_match.group(2).strip()
            else:
                current_block.append(line)

        if current_block:
            joined = "\n".join(current_block).strip()
            if len(joined) > 20:
                sections.append((current_section, joined))

        if not sections and text.strip():
            sections.append(("Content", text.strip()))

        return sections

    def _chunk_text(self, text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
        """Robust sliding window chunking by character count with guaranteed forward progression."""
        if not text or len(text.strip()) == 0:
            return []

        text = text.strip()
        if len(text) <= chunk_size:
            return [text]

        step = max(100, chunk_size - chunk_overlap)
        chunks = []
        for start in range(0, len(text), step):
            sub = text[start : start + chunk_size].strip()
            if len(sub) > 40:
                chunks.append(sub)
            if start + chunk_size >= len(text):
                break

        return chunks

    def _load_document_chunks(self, file_path: Path) -> list[DocumentChunk]:
        """Extract chunks with section and source metadata from a given file."""
        chunks: list[DocumentChunk] = []
        try:
            rel_path = str(file_path.relative_to(self.data_dir)).replace("\\", "/")
        except ValueError:
            rel_path = file_path.name

        ext = file_path.suffix.lower()

        try:
            if ext in {".md", ".txt"}:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                sections = self._extract_text_and_sections_md(text, rel_path)
                for sec_idx, (section_title, sec_content) in enumerate(sections):
                    subchunks = self._chunk_text(sec_content)
                    for c_idx, subchunk in enumerate(subchunks):
                        cid = f"{hashlib.md5(rel_path.encode()).hexdigest()[:8]}_{sec_idx}_{c_idx}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=cid,
                                source=rel_path,
                                section=section_title,
                                content=subchunk,
                            )
                        )

            elif ext == ".pdf":
                try:
                    import pymupdf

                    doc = pymupdf.open(file_path)
                    for page_num in range(len(doc)):
                        page_text = doc[page_num].get_text().strip()
                        if not page_text:
                            continue
                        subchunks = self._chunk_text(page_text)
                        for c_idx, subchunk in enumerate(subchunks):
                            cid = f"{hashlib.md5(rel_path.encode()).hexdigest()[:8]}_p{page_num + 1}_{c_idx}"
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=cid,
                                    source=rel_path,
                                    section=f"Page {page_num + 1}",
                                    content=subchunk,
                                )
                            )
                    doc.close()
                except Exception as pdf_err:
                    logger.warning(f"Failed parsing PDF {file_path}: {pdf_err}")

            elif ext == ".ipynb":
                try:
                    nb_content = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
                    cells = nb_content.get("cells", [])
                    for cell_idx, cell in enumerate(cells):
                        cell_type = cell.get("cell_type", "code")
                        source_text = "".join(cell.get("source", [])).strip()
                        if not source_text:
                            continue
                        cid = f"{hashlib.md5(rel_path.encode()).hexdigest()[:8]}_c{cell_idx}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=cid,
                                source=rel_path,
                                section=f"Cell {cell_idx + 1} ({cell_type})",
                                content=source_text,
                            )
                        )
                except Exception as nb_err:
                    logger.warning(f"Failed parsing notebook {file_path}: {nb_err}")

        except Exception as err:
            logger.error(f"Error reading file {file_path}: {err}")

        return chunks

    def build_index(self, force: bool = False) -> dict[str, Any]:
        """Scan knowledge base directory, build chunks, calculate vector embeddings, and save to disk."""
        if not force and self.load_index():
            return {
                "status": "cached",
                "chunks_count": len(self.chunks),
                "data_dir": str(self.data_dir),
            }

        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist.")
            return {"status": "error", "message": f"Data directory {self.data_dir} not found."}

        all_chunks: list[DocumentChunk] = []
        scanned_files = 0

        for root, dirs, files in os.walk(self.data_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIRS and not d.startswith(".")]

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    file_path = Path(root) / file
                    file_chunks = self._load_document_chunks(file_path)
                    if file_chunks:
                        all_chunks.extend(file_chunks)
                        scanned_files += 1

        if not all_chunks:
            logger.warning("No document chunks extracted.")
            self.chunks = []
            self.embeddings = None
            return {"status": "empty", "scanned_files": scanned_files, "chunks_count": 0}

        # Vectorize all chunks in batches
        texts_to_embed = [
            f"Document: {c.source} | Section: {c.section}\nContent: {c.content}"
            for c in all_chunks
        ]
        logger.info(f"Embedding {len(texts_to_embed)} chunks...")
        embeddings_list = []
        batch_size = 128
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        for b_idx in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[b_idx : b_idx + batch_size]
            batch_emb = list(self.embedding_model.embed(batch))
            embeddings_list.extend(batch_emb)
            current_batch_num = b_idx // batch_size + 1
            if current_batch_num % 2 == 0 or current_batch_num == total_batches:
                print(f"[VectorStore] Indexed batch {current_batch_num}/{total_batches} ({len(embeddings_list)}/{len(texts_to_embed)} chunks)", flush=True)

        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        # Normalize vectors for fast cosine similarity via dot product
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = embeddings_matrix / norms

        self.chunks = all_chunks
        self.embeddings = normalized_embeddings

        # Save to disk
        np.savez_compressed(self.index_file, embeddings=self.embeddings)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2)

        logger.info(f"Indexed {len(self.chunks)} chunks across {scanned_files} files.")
        return {
            "status": "success",
            "scanned_files": scanned_files,
            "chunks_count": len(self.chunks),
            "data_dir": str(self.data_dir),
        }

    def load_index(self) -> bool:
        """Load cached embeddings and chunk metadata if present on disk."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                npz_data = np.load(self.index_file)
                self.embeddings = npz_data["embeddings"]
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    meta_list = json.load(f)
                    self.chunks = [DocumentChunk.from_dict(d) for d in meta_list]
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}")
                return False
        return False

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the top_k most relevant chunks using cosine similarity."""
        if self.embeddings is None or not self.chunks:
            # Build index if not ready
            self.build_index()

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Embed user query
        query_emb = list(self.embedding_model.embed([query]))[0]
        query_vec = np.array(query_emb, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Compute dot product (cosine similarity)
        scores = np.dot(self.embeddings, query_vec)

        # Select top_k indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        # Sort top indices in descending order
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "section": chunk.section,
                "content": chunk.content,
                "score": round(float(scores[idx]), 4),
            })
        return results

    def get_stats(self) -> dict[str, Any]:
        """Return high level stats of the vector store."""
        unique_sources = list({c.source for c in self.chunks})
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(unique_sources),
            "data_dir": str(self.data_dir),
            "is_indexed": self.embeddings is not None and len(self.chunks) > 0,
            "documents": unique_sources[:50],  # preview
        }


# Global singleton instance for easy import
vector_store = VectorStore()
