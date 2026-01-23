"""
mini-SWE-agent runner for SWE-bench with optional dense-only semantic retrieval (hosted NVIDIA nv-embed-v1).

What this file gives you:
- mode: baseline | semantic
- semantic retrieval: chunk repo inside container -> embed on HOST via NVIDIA API -> FAISS IndexFlatIP
- robust cache on disk (repo+base_commit+indexing config), so multiple instances from same repo reuse the same index
- optional structure priors to avoid "all tests": prefer-src weighting + per-bucket quota
- rag-only: run retrieval and save artifacts without running the agent

Usage (retrieval-only debug):
  export NVIDIA_API_KEY="..."
  python3 swebench_semantic.py --subset lite --split dev --slice 0:1 \
    -o runs/semantic_nvembed_dev --mode semantic --top-k 8 --include-tests \
    --prefer-src --rag-only \
    --model "openai/qwen/qwen3-coder-480b-a35b-instruct"
"""
import concurrent.futures
import hashlib
import json
import os
import random
import re
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent.run.extra.rag_dense_helper import nvidia_embed
from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

_GLOBAL_EMB_CACHE_LOCK = threading.Lock()
_GLOBAL_EMB_CACHE: dict[str, list[float]] = {}

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""
app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}

_OUTPUT_FILE_LOCK = threading.Lock()

class ProgressTrackingAgent(DefaultAgent):
    """Wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()

def get_swebench_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name

def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)

    if env_config["environment_class"] in ["docker", "swerex_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name

    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out.get("returncode") != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env

def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    with _OUTPUT_FILE_LOCK:
        output_data: Dict[str, Any] = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))

def remove_from_preds_file(output_path: Path, instance_id: str):
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))

# -----------------------------
# Semantic retrieval utilities
# -----------------------------

@dataclass
class SemanticCfg:
    top_k: int = 8
    include_tests: bool = True
    chunk_chars: int = 2400
    chunk_overlap: int = 300
    max_files: int = 5000
    max_bytes: int = 200_000_000
    batch_size: int = 32
    prefer_src: bool = True
    src_boost: float = 1.15
    test_downweight: float = 0.85
    quota_src: Optional[int] = None  # default derived from top_k
    quota_test: Optional[int] = None  # default derived from top_k
    rag_only: bool = False
    exclude_dotfiles: bool = True
    exclude_meta_markdown: bool = True

def _cache_key_repo_level(instance: dict, cfg: SemanticCfg) -> str:
    """
    Repo-level caching: multiple instances from same repo+commit reuse the same FAISS index.
    """
    repo = instance.get("repo", "")
    base = instance.get("base_commit", "") or ""
    s = (
        f"repo={repo}::base={base}::include_tests={cfg.include_tests}"
        f"::chunk_chars={cfg.chunk_chars}::chunk_overlap={cfg.chunk_overlap}"
        f"::max_files={cfg.max_files}::max_bytes={cfg.max_bytes}"
        f"::exclude_dotfiles={cfg.exclude_dotfiles}"
        f"::exclude_meta_markdown={cfg.exclude_meta_markdown}"
        f"::embed_model=nvidia/nv-embed-v1"
    )
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _normalize_for_hash(text: str) -> str:
    # Normalize only enough to be stable across minor whitespace differences.
    # Keep semantics. Don't over-normalize.
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Optional: trim trailing spaces per line
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    return t.strip()

def _chunk_id(text: str) -> str:
    t = _normalize_for_hash(text)
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def _load_vec(cache_dir: Path, cid: str) -> Optional[List[float]]:
    p = cache_dir / f"{cid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))["v"]
    except Exception:
        return None

def _save_vec(cache_dir: Path, cid: str, vec: List[float]) -> None:
    p = cache_dir / f"{cid}.json"
    p.write_text(json.dumps({"v": vec}), encoding="utf-8")

def _get_repo_chunks_from_container(
    env: Environment,
    *,
    include_tests: bool,
    chunk_chars: int,
    chunk_overlap: int,
    max_files: int,
    max_bytes: int,
    exclude_dotfiles: bool,
    exclude_meta_markdown: bool,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runs inside the SWE-bench container to:
      - walk /testbed
      - chunk files
      - return: (repo_map_text, chunks, stats)

    IMPORTANT:
    - No outer f-string. We do safe string replacement for params.
    """
    script = r"""
python3 - <<'PY'
import json, pathlib, re, bisect
from pathlib import Path

REPO_ROOT="/testbed"
INCLUDE_TESTS=__INCLUDE_TESTS__
CHUNK_CHARS=__CHUNK_CHARS__
CHUNK_OVERLAP=__CHUNK_OVERLAP__
MAX_FILES=__MAX_FILES__
MAX_BYTES=__MAX_BYTES__
EXCLUDE_DOTFILES=__EXCLUDE_DOTFILES__
EXCLUDE_META_MD=__EXCLUDE_META_MD__

# Directories to always skip (directory parts only!)
# Directories to always skip (directory names only, not paths)
SKIP_DIRS = set([
    # Version control
    ".git", ".hg", ".svn",

    # Python caches / tooling
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".hypothesis",

    # Virtual envs / builds
    ".venv", "venv", "env",
    "build", "dist", "out",

    # JS / frontend
    "node_modules", "bower_components",

    # Docs / examples / data (very high volume, low signal)
    "docs", "doc", "documentation",
    "examples", "example",
    "benchmarks", "benchmark",
    "notebooks", "notebook",
    "tutorials", "tutorial",
    "samples", "sample",
    "demo", "demos",
    "paper",

    # Test data / fixtures (usually huge)
    "fixtures", "fixture",
    "testdata", "test_data",
    "data", "datasets",

    # CI / meta
    ".github", ".gitlab",
    ".circleci", ".azure-pipelines","ci",

    # Packaging / infra
    ".eggs", ".idea", ".vscode",

    # Binary / misc
    "vendor", "third_party", "external",
])

# Note: .github is commonly docs/issue templates; skip if you want (kept here as default skip)
# If you'd rather keep it, remove ".github".
SKIP_DIRS.add(".github")

if not INCLUDE_TESTS:
    SKIP_DIRS |= set(["test","tests"])

SKIP_EXT = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",

    # Archives / compressed
    ".zip", ".tar", ".gz", ".xz", ".7z",

    # Media
    ".mp4", ".mp3", ".wav", ".avi", ".mov",

    # Binaries / compiled
    ".so", ".dylib", ".dll", ".exe", ".bin",

    # ML artifacts
    ".pt", ".pth", ".onnx", ".ckpt",

    # Data
    ".csv", ".tsv", ".parquet", ".h5", ".hdf5", ".npz",

    # PDFs
    ".pdf",
}

META_FILENAMES = {
    "readme.md", "readme.rst",
    "changelog.md", "changelog.rst",
    "license", "license.md", "license.txt",
    "code_of_conduct.md",
    "contributing.md",
    "authors", "authors.md",
    "copyright",
    "notice",
}

def is_text_file(p: pathlib.Path) -> bool:
    try:
        data = p.read_bytes()
    except Exception:
        return False
    if b"\x00" in data[:4096]:
        return False
    return True

def should_index_file(rel_path: str) -> bool:
    p = Path(rel_path)

    # 1) Exclude dot-directories and dotfiles
    if EXCLUDE_DOTFILES:
        if any(part.startswith(".") for part in p.parts):
            return False
        if p.name.startswith("."):
            return False

    # 2) Exclude noisy meta markdown/files
    if EXCLUDE_META_MD:
        if p.name.lower() in META_FILENAMES:
            return False

    return True

def iter_files():
    root = pathlib.Path(REPO_ROOT)
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        parts = set(p.parts)
        if any(d in parts for d in SKIP_DIRS):
            continue
            
        if any(part.endswith(".egg-info") for part in p.parts):
            continue

        if p.suffix.lower() in SKIP_EXT:
            continue

        rel = p.relative_to(root).as_posix()

        # APPLY your filter (this was missing before)
        if not should_index_file(rel):
            continue

        yield p, rel

def chunk_text(text: str):
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + CHUNK_CHARS)
        yield start, end, text[start:end]
        if end == n:
            break
        start = max(0, end - CHUNK_OVERLAP)

# collect files (capped)
all_files=[]
total_bytes=0
for p, rel in iter_files():
    try:
        b = p.stat().st_size
    except Exception:
        continue
    total_bytes += b
    all_files.append((p, rel, b))
    if len(all_files) >= MAX_FILES or total_bytes >= MAX_BYTES:
        break

# chunk + track top-level "chunk counts" (real chunks, not files)
chunks=[]
skipped_nontext=0
top_counts={}
for p, rel, _b in all_files:
    try:
        if not is_text_file(p):
            skipped_nontext += 1
            continue
        txt = p.read_text(errors="ignore")
    except Exception:
        continue

    line_offsets=[0]
    for m in re.finditer(r"\n", txt):
        line_offsets.append(m.start()+1)

    for s,e,ct in chunk_text(txt):
        ls = bisect.bisect_right(line_offsets, s)
        le = bisect.bisect_right(line_offsets, e)

        chunks.append({
            "path": rel,
            "line_start": int(ls),
            "line_end": int(max(ls, le)),
            "text": ct
        })

        top = rel.split("/", 1)[0] if "/" in rel else rel
        top_counts[top] = top_counts.get(top, 0) + 1

items = sorted(top_counts.items(), key=lambda x: (-x[1], x[0]))
repo_map_lines=[]
repo_map_lines.append("Repository layout (high-level):")
for k,v in items[:40]:
    repo_map_lines.append(f"  - {k} ({v} chunks)")
repo_map_lines.append("")
repo_map_lines.append("Hint: retrieved code snippets below are labeled with __filename and line spans.")
repo_map_text="\n".join(repo_map_lines)

stats = {
    "repo_root": REPO_ROOT,
    "include_tests": INCLUDE_TESTS,
    "exclude_dotfiles": EXCLUDE_DOTFILES,
    "exclude_meta_markdown": EXCLUDE_META_MD,
    "n_files_considered": len(all_files),
    "n_chunks": len(chunks),
    "skipped_nontext_files": skipped_nontext,
    "top_level_chunk_counts": top_counts,
    "max_files": MAX_FILES,
    "max_bytes": MAX_BYTES,
    "chunk_chars": CHUNK_CHARS,
    "chunk_overlap": CHUNK_OVERLAP,
}

print(json.dumps({
    "ok": True,
    "repo_map_text": repo_map_text,
    "stats": stats,
    "chunks": chunks
}, ensure_ascii=False))
PY
"""

    script = (
        script.replace("__INCLUDE_TESTS__", "True" if include_tests else "False")
        .replace("__CHUNK_CHARS__", str(int(chunk_chars)))
        .replace("__CHUNK_OVERLAP__", str(int(chunk_overlap)))
        .replace("__MAX_FILES__", str(int(max_files)))
        .replace("__MAX_BYTES__", str(int(max_bytes)))
        .replace("__EXCLUDE_DOTFILES__", "True" if exclude_dotfiles else "False")
        .replace("__EXCLUDE_META_MD__", "True" if exclude_meta_markdown else "False")
    )

    out = env.execute(script)
    if out.get("returncode") != 0:
        raise RuntimeError(f"Chunking failed: {out}")

    raw = out.get("output") or ""
    try:
        payload = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Chunking output was not JSON: {e}\nRAW_OUTPUT_HEAD:\n{raw[:2000]}")

    if not payload.get("ok"):
        raise RuntimeError(f"Chunking not ok: {payload}\nRAW_OUTPUT_HEAD:\n{raw[:2000]}")

    return payload["repo_map_text"], payload["chunks"], payload["stats"]

def _require_faiss():
    if faiss is None:
        raise RuntimeError("faiss is not installed on host. Install with: pip install faiss-cpu")

def _l2_normalize(mat: List[List[float]]) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:
        out = []
        for v in mat:
            s = 0.0
            for x in v:
                s += float(x) * float(x)
            norm = (s ** 0.5) if s > 0 else 1.0
            out.append([float(x) / norm for x in v])
        return out

    arr = np.asarray(mat, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms

def _embed_in_batches(texts: List[str], input_type: str, batch_size: int) -> List[List[float]]:
    vecs: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"[semantic] Starting embedding: {total_batches} total batches", flush=True)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Print only every 100th batch
        if batch_num % 100 == 0:
            print(f"[semantic] Embedding batch {batch_num}/{total_batches}", flush=True)

        v = nvidia_embed(batch, input_type=input_type)
        vecs.extend(v)

    print("[semantic] Finished embedding\n", flush=True)
    return vecs

def _chunk_id(repo: str, path: str, line_start: int, line_end: int, text: str) -> str:
    """
    Stable chunk id across commits:
    - MUST NOT include base_commit
    - Uses repo + file path + span + content hash
    """
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{repo}::{path}::{line_start}-{line_end}::{h}"

def _load_vec(cid: str) -> Optional[List[float]]:
    with _GLOBAL_EMB_CACHE_LOCK:
        return _GLOBAL_EMB_CACHE.get(cid)

def _store_vec(cid: str, vec: List[float]) -> None:
    with _GLOBAL_EMB_CACHE_LOCK:
        _GLOBAL_EMB_CACHE[cid] = vec
        
def build_or_load_semantic_index_host(
    env: Environment,
    cache_root: Path,
    instance: dict,
    cfg: SemanticCfg,
) -> Dict[str, Any]:
    """
    Builds FAISS index on HOST:
      1) chunk inside container -> get chunks list
      2) embed on host via NVIDIA API (passage)
      3) FAISS IndexFlatIP over normalized vectors
      4) cache artifacts under cache_root/<key>/

    This version is FIXED and SAFE:
      - no undefined global_emb_cache
      - never passes Optional vectors into _l2_normalize / numpy
      - reuses embeddings across commits (within the same python run) using a stable chunk-id
    """
    _require_faiss()
    key = _cache_key_repo_level(instance, cfg)
    idx_dir = cache_root / key
    idx_dir.mkdir(parents=True, exist_ok=True)

    repo_map_path = idx_dir / "repo_map.txt"
    chunks_path = idx_dir / "chunks.json"
    faiss_path = idx_dir / "faiss.index"
    meta_path = idx_dir / "index_meta.json"

    # If full cache exists for THIS repo+commit+cfg, reuse it
    if repo_map_path.exists() and chunks_path.exists() and faiss_path.exists() and meta_path.exists():
        return {
            "key": key,
            "repo_map_path": str(repo_map_path),
            "chunks_path": str(chunks_path),
            "faiss_path": str(faiss_path),
            "meta_path": str(meta_path),
            "rebuilt": False,
        }

    # 1) Chunk inside container
    repo_map_text, chunks, chunk_stats = _get_repo_chunks_from_container(
        env,
        include_tests=cfg.include_tests,
        chunk_chars=cfg.chunk_chars,
        chunk_overlap=cfg.chunk_overlap,
        max_files=cfg.max_files,
        max_bytes=cfg.max_bytes,
        exclude_dotfiles=cfg.exclude_dotfiles,
        exclude_meta_markdown=cfg.exclude_meta_markdown,
    )

    texts = [c.get("text", "") for c in chunks]
    if not texts or all(not t.strip() for t in texts):
        raise RuntimeError("No chunks produced; check skip rules or repo contents.")

    # 2) Embed on host with cross-commit reuse (in-memory global cache)
    #
    # IMPORTANT: This assumes you have these helpers defined elsewhere in the file:
    #   - _chunk_id(repo: str, path: str, line_start: int, line_end: int, text: str) -> str
    #   - _load_vec(cid: str) -> Optional[List[float]]
    #   - _store_vec(cid: str, vec: List[float]) -> None
    #
    # If you don't, add them exactly as I sent earlier.
    repo = instance.get("repo", "") or ""
    cids: List[str] = []
    for c in chunks:
        cids.append(
            _chunk_id(
                repo,
                c.get("path", ""),
                int(c.get("line_start", 1)),
                int(c.get("line_end", c.get("line_start", 1) or 1)),
                c.get("text", ""),
            )
        )

    vecs: List[Optional[List[float]]] = [None] * len(texts)
    missing_idx: List[int] = []
    missing_texts: List[str] = []

    for i, cid in enumerate(cids):
        v = _load_vec(cid)
        if v is None:
            missing_idx.append(i)
            missing_texts.append(texts[i])
        else:
            vecs[i] = v

    if missing_texts:
        logger.info(
            f"[semantic] Embedding {len(missing_texts)}/{len(texts)} NEW chunks via NVIDIA API "
            f"(batch_size={cfg.batch_size})..."
        )
        new_vecs = _embed_in_batches(missing_texts, input_type="passage", batch_size=cfg.batch_size)
        if len(new_vecs) != len(missing_texts):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(new_vecs)} vecs for {len(missing_texts)} texts"
            )

        for pos, v in zip(missing_idx, new_vecs):
            vecs[pos] = v
            _store_vec(cids[pos], v)
    else:
        logger.info(f"[semantic] Reused cached embeddings for all {len(texts)} chunks (no API calls).")

    # Ensure complete, and convert Optional -> concrete
    final_vecs: List[List[float]] = []
    for i, v in enumerate(vecs):
        if v is None:
            raise RuntimeError(f"Internal error: missing embedding for chunk index {i} ({chunks[i].get('path')})")
        final_vecs.append(v)

    # 3) Normalize + build FAISS
    vecs_norm = _l2_normalize(final_vecs)

    try:
        import numpy as np  # type: ignore
    except Exception:
        raise RuntimeError("numpy is required with faiss. Install: pip install numpy faiss-cpu")

    emb = np.asarray(vecs_norm, dtype="float32")
    if emb.ndim != 2 or emb.shape[0] != len(chunks):
        raise RuntimeError(f"Embedding matrix shape unexpected: {emb.shape} (n_chunks={len(chunks)})")

    dim = int(emb.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # 4) Cache artifacts
    repo_map_path.write_text(repo_map_text, encoding="utf-8")
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(faiss_path))

    meta = {
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "key": key,
        "embed_model": "nvidia/nv-embed-v1 (hosted via NVIDIA API)",
        "dim": dim,
        "n_chunks": len(chunks),
        "chunk_stats": chunk_stats,
        "cfg": {
            "include_tests": cfg.include_tests,
            "chunk_chars": cfg.chunk_chars,
            "chunk_overlap": cfg.chunk_overlap,
            "max_files": cfg.max_files,
            "max_bytes": cfg.max_bytes,
            "batch_size": cfg.batch_size,
            "exclude_dotfiles": cfg.exclude_dotfiles,
            "exclude_meta_markdown": cfg.exclude_meta_markdown,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "key": key,
        "repo_map_path": str(repo_map_path),
        "chunks_path": str(chunks_path),
        "faiss_path": str(faiss_path),
        "meta_path": str(meta_path),
        "rebuilt": True,
    }

def _path_weight(path: str, cfg: SemanticCfg) -> float:
    p = path.replace("\\", "/")
    if cfg.prefer_src and p.startswith(("src/", "sqlfluff/", "lib/")):
        return cfg.src_boost
    if p.startswith(("test/", "tests/", "docs/", "fixtures/")):
        return cfg.test_downweight
    return 1.0

def _bucket(path: str) -> str:
    p = path.replace("\\", "/")
    return p.split("/", 1)[0] if "/" in p else p

def semantic_retrieve_topk_host(
    index_path: Path,
    chunks_path: Path,
    query: str,
    cfg: SemanticCfg,
) -> List[Dict[str, Any]]:
    _require_faiss()

    chunks: List[Dict[str, Any]] = json.loads(chunks_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))

    q_vec = nvidia_embed([query], input_type="query")
    q_norm = _l2_normalize(q_vec)

    try:
        import numpy as np  # type: ignore
    except Exception:
        raise RuntimeError("numpy is required with faiss. Install: pip install numpy")

    q = np.asarray(q_norm, dtype="float32")

    pool = max(cfg.top_k * 10, 50)
    pool = min(pool, len(chunks))
    scores, ids = index.search(q, pool)

    cands: List[Dict[str, Any]] = []
    for s, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        c = chunks[idx]
        w = _path_weight(c["path"], cfg)
        cands.append(
            {
                "path": c["path"],
                "line_start": c.get("line_start", 1),
                "line_end": c.get("line_end", c.get("line_start", 1)),
                "text": c["text"],
                "score": float(s),
                "score_weighted": float(s) * float(w),
            }
        )

    cands.sort(key=lambda x: x["score_weighted"], reverse=True)

    quota_src = cfg.quota_src if cfg.quota_src is not None else max(1, cfg.top_k // 2)
    quota_test = cfg.quota_test if cfg.quota_test is not None else max(1, cfg.top_k // 4)

    picked: List[Dict[str, Any]] = []
    used = defaultdict(int)

    def limit_for(b: str) -> int:
        if b == "src":
            return quota_src
        if b in ("test", "tests"):
            return quota_test
        return cfg.top_k

    for c in cands:
        b = _bucket(c["path"])
        if used[b] < limit_for(b):
            picked.append(c)
            used[b] += 1
        if len(picked) >= cfg.top_k:
            break

    if len(picked) < cfg.top_k:
        already = {(x["path"], x["line_start"], x["line_end"]) for x in picked}
        for c in cands:
            key = (c["path"], c["line_start"], c["line_end"])
            if key in already:
                continue
            picked.append(c)
            if len(picked) >= cfg.top_k:
                break

    return picked[: cfg.top_k]

def format_retrieved_context(repo_map_text: str, chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(repo_map_text.strip())
    lines.append("")
    lines.append("Retrieved context (dense semantic search):")
    lines.append("Note: each snippet is grounded with __filename and line span.")
    lines.append("")

    for i, c in enumerate(chunks, start=1):
        path = c["path"]
        ls = c.get("line_start", 1)
        le = c.get("line_end", ls)
        score = c.get("score", 0.0)
        wscore = c.get("score_weighted", score)
        lines.append(f"[{i}] __filename: {path}")
        lines.append(f"    __span: L{ls}-L{le}    __score: {score:.4f}    __wscore: {wscore:.4f}")
        lines.append("```")
        lines.append((c.get("text") or "").rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def build_task_with_retrieval(problem_statement: str, retrieved_context: str) -> str:
    return (
        "You are fixing a real repository bug. Use the repo layout and retrieved file snippets to localize the change.\n"
        "When you refer to code, use the __filename markers to orient yourself.\n\n"
        + retrieved_context
        + "\n\n"
        + "Issue description:\n"
        + problem_statement.strip()
        + "\n"
    )

def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    *,
    mode: str,
    semcfg: SemanticCfg,
) -> None:
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    base_task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    extra_info = None
    env = None
    exit_status: Any = None
    result: Any = None

    try:
        env = get_sb_environment(config, instance)
        task = base_task
        retrieval_meta = None

        if mode == "semantic":
            progress_manager.update_instance_status(instance_id, "Semantic retrieval: chunk -> embed -> FAISS")
            cache_root = output_dir / "_semantic_index_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            # Global cache across *all* repos/commits (content-addressed)
            global_emb_cache = (cache_root.parent / "_semantic_emb_cache")
            global_emb_cache.mkdir(parents=True, exist_ok=True)

            idx_info = build_or_load_semantic_index_host(env, cache_root, instance, semcfg)

            repo_map_text = Path(idx_info["repo_map_path"]).read_text(encoding="utf-8", errors="ignore")
            chunks = semantic_retrieve_topk_host(
                index_path=Path(idx_info["faiss_path"]),
                chunks_path=Path(idx_info["chunks_path"]),
                query=base_task,
                cfg=semcfg,
            )

            retrieved_context = format_retrieved_context(repo_map_text, chunks)

            (instance_dir / "retrieved_context.txt").write_text(retrieved_context, encoding="utf-8")
            (instance_dir / "retrieval.json").write_text(
                json.dumps(
                    {
                        "mode": "semantic",
                        "embed_model": "nvidia/nv-embed-v1 (hosted via NVIDIA API)",
                        "top_k": semcfg.top_k,
                        "include_tests": semcfg.include_tests,
                        "chunk_chars": semcfg.chunk_chars,
                        "chunk_overlap": semcfg.chunk_overlap,
                        "max_files": semcfg.max_files,
                        "max_bytes": semcfg.max_bytes,
                        "batch_size": semcfg.batch_size,
                        "prefer_src": semcfg.prefer_src,
                        "src_boost": semcfg.src_boost,
                        "test_downweight": semcfg.test_downweight,
                        "quota_src": semcfg.quota_src,
                        "quota_test": semcfg.quota_test,
                        "exclude_dotfiles": semcfg.exclude_dotfiles,
                        "exclude_meta_markdown": semcfg.exclude_meta_markdown,
                        "index": idx_info,
                        "results_preview": [
                            {
                                "path": c["path"],
                                "line_start": c.get("line_start"),
                                "line_end": c.get("line_end"),
                                "score": c.get("score"),
                                "score_weighted": c.get("score_weighted"),
                            }
                            for c in chunks
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            task = build_task_with_retrieval(base_task, retrieved_context)
            retrieval_meta = {"index": idx_info}

            if semcfg.rag_only:
                exit_status, result = "RAG_ONLY", ""
                extra_info = extra_info or {}
                extra_info["retrieval"] = retrieval_meta
                return

        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )

        progress_manager.update_instance_status(instance_id, "Running agent")
        exit_status, result = agent.run(task)

        if retrieval_meta is not None:
            extra_info = extra_info or {}
            extra_info["retrieval"] = retrieval_meta

    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}

    finally:
        if env and hasattr(env, "stop"):
            env.stop()

        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)

def filter_instances(
    instances: List[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> List[dict]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)

    before_filter = len(instances)
    if filter_spec:
        instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")

    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances

@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5')", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    mode: str = typer.Option("baseline", "--mode", help="Run mode: baseline | semantic", rich_help_panel="Retrieval"),
    top_k: int = typer.Option(8, "--top-k", help="Top-k chunks to inject (semantic mode)", rich_help_panel="Retrieval"),
    include_tests: bool = typer.Option(True, "--include-tests/--exclude-tests", help="Index tests directories too", rich_help_panel="Retrieval"),
    chunk_chars: int = typer.Option(2400, "--chunk-chars", help="Chunk size in characters", rich_help_panel="Retrieval"),
    chunk_overlap: int = typer.Option(300, "--chunk-overlap", help="Chunk overlap in characters", rich_help_panel="Retrieval"),
    max_files: int = typer.Option(5000, "--max-files", help="Cap number of files indexed", rich_help_panel="Retrieval"),
    max_bytes: int = typer.Option(200_000_000, "--max-bytes", help="Cap total bytes indexed", rich_help_panel="Retrieval"),
    batch_size: int = typer.Option(32, "--embed-batch-size", help="Embedding batch size (hosted API)", rich_help_panel="Retrieval"),
    rag_only: bool = typer.Option(False, "--rag-only", help="Only run retrieval + save artifacts, do not run agent", rich_help_panel="Retrieval"),
    prefer_src: bool = typer.Option(True, "--prefer-src/--no-prefer-src", help="Bias retrieval toward src/ (weighting + quotas)", rich_help_panel="Retrieval"),
    src_boost: float = typer.Option(1.15, "--src-boost", help="Score multiplier for src/ paths", rich_help_panel="Retrieval"),
    test_downweight: float = typer.Option(0.85, "--test-downweight", help="Score multiplier for test/docs/fixtures paths", rich_help_panel="Retrieval"),
    quota_src: int = typer.Option(-1, "--quota-src", help="Max chunks from src bucket in top-k (-1 auto)", rich_help_panel="Retrieval"),
    quota_test: int = typer.Option(-1, "--quota-test", help="Max chunks from test bucket in top-k (-1 auto)", rich_help_panel="Retrieval"),
    exclude_dotfiles: bool = typer.Option(True, "--exclude-dotfiles/--include-dotfiles", help="Skip dot dirs/files like .gitignore, .github/*", rich_help_panel="Retrieval"),
    exclude_meta_markdown: bool = typer.Option(True, "--exclude-meta-md/--include-meta-md", help="Skip README/CHANGELOG/LICENSE etc.", rich_help_panel="Retrieval"),
    model: str | None = typer.Option(None, "-m", "--model", help="Agent model to use (LLM)", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type (docker or singularity)", rich_help_panel="Advanced"),
) -> None:
    if mode not in {"baseline", "semantic"}:
        raise typer.BadParameter("mode must be one of: baseline, semantic")

    if mode == "semantic" and not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("Missing NVIDIA_API_KEY env var (export NVIDIA_API_KEY=...)")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]

    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())

    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    semcfg = SemanticCfg(
        top_k=top_k,
        include_tests=include_tests,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        max_files=max_files,
        max_bytes=max_bytes,
        batch_size=batch_size,
        prefer_src=prefer_src,
        src_boost=src_boost,
        test_downweight=test_downweight,
        quota_src=None if quota_src < 0 else quota_src,
        quota_test=None if quota_test < 0 else quota_test,
        rag_only=rag_only,
        exclude_dotfiles=exclude_dotfiles,
        exclude_meta_markdown=exclude_meta_markdown,
    )

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: Dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                    mode=mode,
                    semcfg=semcfg,
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

if __name__ == "__main__":
    app()