"""MedRAG retriever initialization and querying."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

from .config import PipelineConfig
from .environment import ensure_medrag_repo


def _load_medrag_utils(repo_dir: Path) -> ModuleType:
    """Load MedRAG's src/utils.py without colliding with this project's src package."""

    utils_path = repo_dir / "src" / "utils.py"
    if not utils_path.exists():
        raise RuntimeError(f"MedRAG utils.py not found at {utils_path}")

    repo_src = str(repo_dir / "src")
    repo_root = str(repo_dir)
    for path in (repo_src, repo_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location("medrag_source_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MedRAG utilities from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MedRAGRetrievalManager:
    def __init__(self, config: PipelineConfig):
        self.config = config
        repo_dir = ensure_medrag_repo(config)
        self.medrag_utils = _load_medrag_utils(repo_dir)
        self._patch_dense_index_dir()
        self.systems: dict[str, object] = {}

    @property
    def corpus_names(self) -> dict:
        return self.medrag_utils.corpus_names

    @property
    def retriever_names(self) -> dict:
        return self.medrag_utils.retriever_names

    def _patch_dense_index_dir(self) -> None:
        retriever_cls = self.medrag_utils.Retriever
        if getattr(retriever_cls, "_arc_index_patch_applied", False):
            return

        original_init = retriever_cls.__init__
        config = self.config

        def patched_init(self, retriever_name, corpus_name, db_dir, HNSW=False, **kwargs):
            original_init(self, retriever_name, corpus_name, db_dir, HNSW=HNSW, **kwargs)
            index_dir = getattr(self, "index_dir", None)
            if index_dir and "/drive/" in str(index_dir):
                local_idx = (
                    config.local_embed_dir
                    / corpus_name
                    / "index"
                    / retriever_name.replace("/", "_")
                )
                local_idx.mkdir(parents=True, exist_ok=True)
                self.index_dir = str(local_idx)

        retriever_cls.__init__ = patched_init
        retriever_cls._arc_index_patch_applied = True

    def init_retriever(self, retriever_key: str):
        print(f"\n=== Init {retriever_key} ===")
        start = time.time()
        system = self.medrag_utils.RetrievalSystem(
            retriever_name=retriever_key,
            corpus_name=self.config.resolved_corpus_name,
            db_dir=str(self.config.db_dir),
            HNSW=False,
            cache=False,
        )
        self.systems[retriever_key] = system
        elapsed = (time.time() - start) / 60
        print(f"{retriever_key} ready in {elapsed:.1f} min.")
        return system

    def init_all(self) -> None:
        for retriever in self.config.retrievers:
            self.init_retriever(retriever)

    def retrieve_all(self, question_text: str, k: int | None = None) -> dict[str, list[dict]]:
        if not self.systems:
            self.init_all()
        top_k = k or self.config.top_k
        results: dict[str, list[dict]] = {}
        for key, system in self.systems.items():
            snippets, scores = system.retrieve(question_text, k=top_k, id_only=False)
            results[key] = [
                {
                    "rank": i + 1,
                    "score": float(scores[i]),
                    "document_id": snippet.get("id"),
                    "title": snippet.get("title", ""),
                    "text": snippet.get("content", snippet.get("contents", "")),
                    "source_corpus": self.config.resolved_corpus_name,
                    "retriever": key,
                }
                for i, snippet in enumerate(snippets)
            ]
            print(f"  {key:12s} -> {len(results[key])} snippets")
        return results
