"""
mini-SWE-agent runner for SWE-bench with reproducible retrieval experiments:

Modes:
  - bm25: repo-local BM25 retrieval (inject top-k)
  - hybrid: BM25 top-N candidates -> embeddings -> fusion (rrf/linear/semantic-only) -> inject top-k

Key goals:
  - One script for fair comparisons: identical indexing + chunking across modes.
  - Repo-local indexing inside the SWE-bench container at /testbed.
  - Optional tests indexing: --include-tests (default) or --exclude-tests.
  - --rag-only saves retrieved context without running the agent.

Outputs per instance:
  - retrieval.json        (all retrieval params + selected chunks + scores)
  - retrieved_context.txt (injected context)
  - index_stats.json      (what was indexed)
  - <instance_id>.traj.json (agent trajectory)
  - preds.json            (patches for harness; skipped in rag-only)

Recommended first debug run:
  python3 swebench_retrieval_runner.py --subset lite --split dev --slice 0:1 \
    --output runs/debug_one --workers 1 --mode hybrid --rag-only --include-tests
"""
import concurrent.futures
import json
import math
import random
import re
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

# ----------------------------
# CLI / App
# ----------------------------
_HELP_TEXT = """Run mini-SWE-agent on SWE-bench (baseline, BM25, or hybrid retrieval)."""

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

# ----------------------------
# Agent wrapper for progress
# ----------------------------

class ProgressTrackingAgent(DefaultAgent):
    """Wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()

# ----------------------------
# SWE-bench env
# ----------------------------

def get_swebench_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name

def _get_exec_streams(out: Any) -> Tuple[str, str, int]:
    """Normalize env.execute() output."""
    if isinstance(out, dict):
        rc = int(out.get("returncode", out.get("rc", 0) or 0))
        if "stdout" in out or "stderr" in out:
            return str(out.get("stdout", "")), str(out.get("stderr", "")), rc
        if "output" in out and isinstance(out["output"], str):
            return out["output"], str(out.get("stderr", "")), rc
        if "stdout_lines" in out and isinstance(out["stdout_lines"], list):
            return "\n".join(out["stdout_lines"]), "\n".join(out.get("stderr_lines", []) or []), rc
        # Fallback: dump dict into stderr for debugging
        return "", json.dumps(out, ensure_ascii=False, indent=2), rc
    if isinstance(out, str):
        return out, "", 0
    return "", f"Unsupported execute() output type: {type(out)}; value={out!r}", 1

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
        stdout, stderr, rc = _get_exec_streams(out)
        if rc != 0:
            raise RuntimeError(f"Error executing startup command (rc={rc}).\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}")
    return env

# ----------------------------
# preds.json I/O
# ----------------------------

def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str) -> None:
    with _OUTPUT_FILE_LOCK:
        output_data: Dict[str, Any] = {}
        if output_path.exists():
            try:
                output_data = json.loads(output_path.read_text())
            except Exception:
                output_data = {}
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))


def remove_from_preds_file(output_path: Path, instance_id: str) -> None:
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        try:
            output_data = json.loads(output_path.read_text())
        except Exception:
            return
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))

# ----------------------------
# Retrieval data structures
# ----------------------------

@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str       # injected text (original)
    score_text: str # scoring text (optionally cleaned)

@dataclass
class ScoredChunk:
    chunk: Chunk
    bm25: float
    cosine: float
    bm25_norm: float
    rrf: float
    final: float
    rank: int

# ----------------------------
# Text cleaning for scoring
# ----------------------------

def strip_top_level_docstring(text: str) -> str:
    s = text.lstrip("\ufeff \t\r\n")
    if s.startswith('"""') or s.startswith("'''"):
        quote = s[:3]
        end = s.find(quote, 3)
        if end != -1:
            return s[end + 3 :]
    return text

def strip_line_comments(text: str) -> str:
    out_lines: List[str] = []
    for i, line in enumerate(text.splitlines(True)):
        if i == 0 and line.startswith("#!"):
            out_lines.append(line)
            continue
        if i <= 1 and "coding" in line and line.strip().startswith("#"):
            out_lines.append(line)
            continue
        if line.lstrip().startswith("#"):
            continue
        m = re.match(r"^(.*?)(\s+#.*)$", line)
        if m:
            out_lines.append(m.group(1).rstrip() + "\n")
        else:
            out_lines.append(line)
    return "".join(out_lines)

def make_score_text(original: str, *, strip_docstrings: bool, strip_comments_opt: bool) -> str:
    s = original
    if strip_docstrings:
        s = strip_top_level_docstring(s)
    if strip_comments_opt:
        s = strip_line_comments(s)
    return s

# ----------------------------
# BM25 scoring
# ----------------------------

def _tokenize_for_bm25(text: str) -> List[str]:
    parts = re.split(r"[^A-Za-z0-9_]+", text)
    return [p.lower() for p in parts if p]


def bm25_rank(query: str, docs: List[str]) -> List[float]:
    """Try rank_bm25; fallback to a pure Python BM25-like implementation."""
    if not docs:
        return []
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        corpus = [_tokenize_for_bm25(d) for d in docs]
        bm25 = BM25Okapi(corpus)
        return list(map(float, bm25.get_scores(_tokenize_for_bm25(query))))
    except Exception:
        tokenized_docs = [_tokenize_for_bm25(d) for d in docs]
        N = len(tokenized_docs)
        if N == 0:
            return []
        df: Dict[str, int] = {}
        doc_lens = [len(toks) for toks in tokenized_docs]
        avgdl = sum(doc_lens) / max(1, N)
        for toks in tokenized_docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        def idf(t: str) -> float:
            n = df.get(t, 0)
            return math.log(1 + (N - n + 0.5) / (n + 0.5))

        k1 = 1.5
        b = 0.75
        q_toks = _tokenize_for_bm25(query)

        scores: List[float] = []
        for toks, dl in zip(tokenized_docs, doc_lens):
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q_toks:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + k1 * (1 - b + b * (dl / (avgdl + 1e-9)))
                s += idf(t) * (f * (k1 + 1)) / (denom + 1e-9)
            scores.append(float(s))
        return scores

def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

# ----------------------------
# Embeddings (cached + thread-safe)
# ----------------------------

_EMBEDDER_CACHE_LOCK = threading.Lock()
_EMBEDDER_CACHE: Dict[Tuple[str, str], "Embedder"] = {}

class Embedder:
    """Transformers mean pooling embedder, cached per (model_name, device)."""

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._encode_lock = threading.Lock()

    @staticmethod
    def _mean_pool(last_hidden, attention_mask, torch_mod):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode(self, texts: List[str], max_length: int = 512, batch_size: int = 16):
        torch = self.torch
        if not texts:
            raise ValueError("Embedder.encode() called with empty texts list.")

        vecs = []
        with self._encode_lock:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    tok = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    tok = {k: v.to(self.device) for k, v in tok.items()}
                    out = self.model(**tok)
                    pooled = self._mean_pool(out.last_hidden_state, tok["attention_mask"], torch)
                    pooled = pooled / pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
                    vecs.append(pooled.detach().cpu())
        return torch.cat(vecs, dim=0)

def _resolve_embed_device(embed_device: str) -> str:
    embed_device = (embed_device or "auto").lower().strip()
    if embed_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("embed_device must be one of: auto, cpu, cuda")
    if embed_device == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def get_cached_embedder(model_name: str, embed_device: str) -> Embedder:
    dev = _resolve_embed_device(embed_device)
    key = (model_name, dev)
    with _EMBEDDER_CACHE_LOCK:
        if key in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[key]
        emb = Embedder(model_name, device=dev)
        _EMBEDDER_CACHE[key] = emb
        return emb

def cosine_sim_matrix(query_vec, doc_vecs) -> List[float]:
    """query_vec: (1,d), doc_vecs: (N,d), vectors are normalized => cosine = dot."""
    q = query_vec.squeeze(0)  # (d,)
    sims = (doc_vecs @ q).tolist()
    return [float(s) for s in sims]

# ----------------------------
# RRF fusion
# ----------------------------

def rrf_fusion(rank_a: List[int], rank_b: List[int], *, k: int = 60, w_a: float = 1.0, w_b: float = 1.0) -> List[float]:
    if len(rank_a) != len(rank_b):
        raise ValueError("rank lists must have same length")
    out: List[float] = []
    for ra, rb in zip(rank_a, rank_b):
        out.append((w_a / (k + ra)) + (w_b / (k + rb)))
    return out

# ----------------------------
# Container-side chunk collection
# ----------------------------

def build_lines_score(lines_inject, strip_docstrings: bool, strip_comments: bool):
    # IMPORTANT: do NOT remove lines; only blank them out
    lines_score = lines_inject[:]  # same length

    if strip_comments:
        # naive: remove inline comments; you can improve later
        lines_score = [ln.split("#", 1)[0] + ("\n" if ln.endswith("\n") else "") for ln in lines_score]

    if strip_docstrings:
        in_triple = False
        triple_tok = None

        for i, ln in enumerate(lines_score):
            # detect triple quotes (simple but works well enough)
            if not in_triple:
                if "'''" in ln or '"""' in ln:
                    # choose which token starts first in the line
                    idx1 = ln.find("'''") if "'''" in ln else 10**9
                    idx2 = ln.find('"""') if '"""' in ln else 10**9
                    triple_tok = "'''" if idx1 < idx2 else '"""'

                    # if it opens and closes on same line -> blank only docstring span (simpler: blank whole line)
                    if ln.count(triple_tok) >= 2:
                        lines_score[i] = "\n" if ln.endswith("\n") else ""
                    else:
                        in_triple = True
                        lines_score[i] = "\n" if ln.endswith("\n") else ""
            else:
                # we are inside docstring block
                lines_score[i] = "\n" if ln.endswith("\n") else ""
                if triple_tok and triple_tok in ln:
                    in_triple = False
                    triple_tok = None

    return lines_score

def collect_repo_chunks_in_container(
    env: Environment,
    repo_root: str,
    include_exts: List[str],
    exclude_dir_regex: str,
    exclude_file_regex: str,
    exclude_packaging_files: bool,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    chunk_lines: int,
    chunk_overlap: int,
    strip_docstrings_for_scoring: bool,
    strip_comments_for_scoring: bool,
) -> Tuple[List[Chunk], Dict[str, Any]]:
    payload = {
        "repo_root": repo_root,
        "include_exts": include_exts,
        "exclude_dir_regex": exclude_dir_regex,
        "exclude_file_regex": exclude_file_regex,
        "exclude_packaging_files": bool(exclude_packaging_files),
        "max_files": max_files,
        "max_file_bytes": max_file_bytes,
        "max_total_bytes": max_total_bytes,
        "chunk_lines": chunk_lines,
        "chunk_overlap": chunk_overlap,
        "strip_docstrings_for_scoring": bool(strip_docstrings_for_scoring),
        "strip_comments_for_scoring": bool(strip_comments_for_scoring),
    }
    cfg_json = json.dumps(payload)

    cmd = f"""python - <<'PY'
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass
import os, re, json

cfg = json.loads({cfg_json!r})
root = cfg["repo_root"]
include_exts = set([e.lower() for e in cfg["include_exts"]])
exclude_dir_re = re.compile(cfg["exclude_dir_regex"]) if cfg["exclude_dir_regex"] else None
exclude_file_re = re.compile(cfg["exclude_file_regex"]) if cfg["exclude_file_regex"] else None

exclude_packaging = bool(cfg.get("exclude_packaging_files", False))
PACKAGING_FILES = {{
    "setup.py","pyproject.toml","setup.cfg","requirements.txt","requirements-dev.txt","tox.ini","pdm.lock","poetry.lock"
}}

max_files = int(cfg["max_files"])
max_file_bytes = int(cfg["max_file_bytes"])
max_total_bytes = int(cfg["max_total_bytes"])
chunk_lines = int(cfg["chunk_lines"])
chunk_overlap = int(cfg["chunk_overlap"])

strip_doc = bool(cfg.get("strip_docstrings_for_scoring", True))
strip_comments = bool(cfg.get("strip_comments_for_scoring", False))

total_bytes = 0
n_files = 0
n_skipped_dir = 0
n_skipped_ext = 0
n_skipped_file_re = 0
n_skipped_packaging = 0
n_skipped_size = 0
n_errors = 0
n_chunks = 0

def normpath(p: str) -> str:
    return p.replace(os.sep, "/")

def build_lines_score(lines_inject, strip_docstrings: bool, strip_comments: bool):
    # IMPORTANT: do NOT remove lines; only blank them out
    lines_score = list(lines_inject)  # same length

    if strip_comments:
        new_lines = []
        for ln in lines_score:
            # keep shebang / encoding lines
            if ln.startswith("#!"):
                new_lines.append(ln); continue
            if "coding" in ln and ln.strip().startswith("#"):
                new_lines.append(ln); continue

            # full-line comments -> blank line
            if ln.lstrip().startswith("#"):
                new_lines.append("\\n" if ln.endswith("\\n") else "")
                continue

            # inline comments -> strip after '#'
            left = ln.split("#", 1)[0]
            new_lines.append(left + ("\\n" if ln.endswith("\\n") else ""))
        lines_score = new_lines

    if strip_docstrings:
        in_triple = False
        triple_tok = None

        for i, ln in enumerate(lines_score):
            if not in_triple:
                if "'''" in ln or '\"\"\"' in ln:
                    idx1 = ln.find("'''") if "'''" in ln else 10**9
                    idx2 = ln.find('\"\"\"') if '\"\"\"' in ln else 10**9
                    triple_tok = "'''" if idx1 < idx2 else '\"\"\"'

                    # opens + closes on same line -> blank the line (simple + safe)
                    if ln.count(triple_tok) >= 2:
                        lines_score[i] = "\\n" if ln.endswith("\\n") else ""
                    else:
                        in_triple = True
                        lines_score[i] = "\\n" if ln.endswith("\\n") else ""
            else:
                lines_score[i] = "\\n" if ln.endswith("\\n") else ""
                if triple_tok and triple_tok in ln:
                    in_triple = False
                    triple_tok = None

    return lines_score

def should_exclude_dir(path: str) -> bool:
    global n_skipped_dir
    if not exclude_dir_re:
        return False
    p = normpath(path)
    if exclude_dir_re.search(p):
        n_skipped_dir += 1
        return True
    return False

def should_exclude_file(relpath: str, base: str) -> bool:
    global n_skipped_file_re, n_skipped_packaging
    rp = normpath(relpath)
    if exclude_file_re and exclude_file_re.search(rp):
        n_skipped_file_re += 1
        return True
    if exclude_packaging and base.lower() in PACKAGING_FILES:
        n_skipped_packaging += 1
        return True
    return False

def iter_files():
    global n_skipped_ext
    for dirpath, dirnames, filenames in os.walk(root):
        if should_exclude_dir(dirpath):
            dirnames[:] = []
            continue
        keep = []
        for d in dirnames:
            full = os.path.join(dirpath, d)
            if should_exclude_dir(full):
                continue
            keep.append(d)
        dirnames[:] = keep

        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if include_exts and ext not in include_exts:
                n_skipped_ext += 1
                continue
            yield os.path.join(dirpath, fn), fn

for fp, base in iter_files():
    if n_files >= max_files:
        break
    try:
        st = os.stat(fp)
        if st.st_size > max_file_bytes:
            n_skipped_size += 1
            continue
        if total_bytes + st.st_size > max_total_bytes:
            break

        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()

    except Exception:
        n_errors += 1
        continue

    rel = fp
    if rel.startswith(root):
        rel = rel[len(root):].lstrip(os.sep)
    rel = normpath(rel)

    if should_exclude_file(rel, base):
        continue

    total_bytes += st.st_size
    n_files += 1

    # Build TWO parallel arrays with identical line count
    lines_inject = text_content.splitlines(True)
    lines_score  = build_lines_score(lines_inject, strip_docstrings=strip_doc, strip_comments=strip_comments)

    step = max(1, chunk_lines - chunk_overlap)
    n = len(lines_inject)

    for start in range(0, n, step):
        end = min(n, start + chunk_lines)

        inj_txt = "".join(lines_inject[start:end])
        sc_txt  = "".join(lines_score[start:end]

        obj = {{
            "path": rel,
            "start_line": start+1,
            "end_line": end,
            "text": inj_txt,
            "score_text": sc_txt,
        }}
        print(json.dumps(obj, ensure_ascii=True))
        n_chunks += 1

        if end >= n:
            break

stats = {{
    "repo_root": root,
    "n_files_indexed": n_files,
    "n_chunks_emitted": n_chunks,
    "bytes_indexed": total_bytes,
    "skipped_dirs": n_skipped_dir,
    "skipped_ext": n_skipped_ext,
    "skipped_file_regex": n_skipped_file_re,
    "skipped_packaging_files": n_skipped_packaging,
    "skipped_too_large": n_skipped_size,
    "read_errors": n_errors,
}}
print("===INDEX_STATS===" + json.dumps(stats, ensure_ascii=False))
PY"""

    out = env.execute(cmd)
    stdout, stderr, rc = _get_exec_streams(out)
    if rc != 0:
        raise RuntimeError(f"Chunk collector failed (rc={rc}).\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}")

    chunks: List[Chunk] = []
    stats: Dict[str, Any] = {"note": "no stats emitted"}

    for line in stdout.splitlines():
        if line.startswith("===INDEX_STATS==="):
            try:
                stats = json.loads(line[len("===INDEX_STATS===") :])
            except Exception:
                stats = {"note": "failed to parse stats"}
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        score_text = str(obj.get("score_text", ""))
        if not score_text.strip():
            continue

        chunks.append(
            Chunk(
                path=str(obj.get("path", "")),
                start_line=int(obj.get("start_line", 1)),
                end_line=int(obj.get("end_line", 1)),
                text=str(obj.get("text", "")),
                score_text=score_text,
            )
        )

    return chunks, stats


# ----------------------------
# Prompt building
# ----------------------------

def format_retrieved_context(scored: List[ScoredChunk], max_snippet_chars: int) -> str:
    blocks: List[str] = []
    for s in scored:
        txt = s.chunk.text
        if len(txt) > max_snippet_chars:
            txt = txt[:max_snippet_chars] + "\n... [truncated]\n"
        header = (
            f"[{s.rank}] {s.chunk.path}:{s.chunk.start_line}-{s.chunk.end_line} | "
            f"bm25={s.bm25:.3f} bm25_norm={s.bm25_norm:.3f} cos={s.cosine:.3f} "
            f"rrf={s.rrf:.6f} final={s.final:.6f}"
        )
        blocks.append(header + "\n" + txt)
    return "\n\n".join(blocks)

def build_augmented_prompt(problem: str, retrieved_ctx: str) -> str:
    if not retrieved_ctx.strip():
        return problem.strip()
    return (
        problem.strip()
        + "\n\n"
        + "## REPO CONTEXT (retrieved from /testbed)\n"
        + "You are given snippets from the target repository. Use them to localize and fix the bug.\n"
        + "Do NOT modify tests unless the task explicitly requires it.\n"
        + "Return ONLY a unified git diff.\n\n"
        + retrieved_ctx
        + "\n\n"
        + "## OUTPUT REQUIREMENTS\n"
        + "- Output MUST be a single unified git diff patch starting with 'diff --git'.\n"
        + "- Do NOT include explanations.\n"
        + "- Keep changes minimal and consistent with repo style.\n"
    )

# ----------------------------
# Retrieval implementations
# ----------------------------

def retrieve_bm25_only(
    env: Environment,
    query: str,
    *,
    repo_root: str,
    include_exts: List[str],
    exclude_dir_regex: str,
    exclude_file_regex: str,
    exclude_packaging_files: bool,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    chunk_lines: int,
    chunk_overlap: int,
    strip_docstrings_for_scoring: bool,
    strip_comments_for_scoring: bool,
    top_k: int,
    max_snippet_chars: int,
) -> Tuple[str, Dict[str, Any]]:
    chunks, index_stats = collect_repo_chunks_in_container(
        env=env,
        repo_root=repo_root,
        include_exts=include_exts,
        exclude_dir_regex=exclude_dir_regex,
        exclude_file_regex=exclude_file_regex,
        exclude_packaging_files=exclude_packaging_files,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        chunk_lines=chunk_lines,
        chunk_overlap=chunk_overlap,
        strip_docstrings_for_scoring=strip_docstrings_for_scoring,
        strip_comments_for_scoring=strip_comments_for_scoring,
    )
    if not chunks:
        return "", {"index_stats": index_stats, "note": "no chunks collected"}

    docs = [c.score_text for c in chunks]
    bm25_scores = bm25_rank(query, docs)
    if not bm25_scores:
        return "", {"index_stats": index_stats, "note": "no bm25 scores produced"}

    idx_sorted = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)
    idx_sorted = idx_sorted[: max(1, min(top_k, len(idx_sorted)))]

    # For bm25-only, cosine/rrf are 0; final uses bm25 only (also store norm)
    picked_bm25 = [float(bm25_scores[i]) for i in idx_sorted]
    picked_norm = _minmax_norm(picked_bm25)

    scored: List[ScoredChunk] = []
    for r, (i, b, bn) in enumerate(zip(idx_sorted, picked_bm25, picked_norm), start=1):
        scored.append(
            ScoredChunk(
                chunk=chunks[i],
                bm25=float(b),
                cosine=0.0,
                bm25_norm=float(bn),
                rrf=0.0,
                final=float(b),
                rank=r,
            )
        )

    retrieved_text = format_retrieved_context(scored, max_snippet_chars=max_snippet_chars)
    meta = {
        "mode": "bm25",
        "repo_root": repo_root,
        "index_stats": index_stats,
        "n_chunks": len(chunks),
        "top_k": top_k,
        "selected": [
            {
                "rank": s.rank,
                "path": s.chunk.path,
                "start_line": s.chunk.start_line,
                "end_line": s.chunk.end_line,
                "bm25": s.bm25,
                "bm25_norm": s.bm25_norm,
                "final": s.final,
            }
            for s in scored
        ],
    }
    return retrieved_text, meta

def retrieve_hybrid(
    env: Environment,
    query: str,
    *,
    repo_root: str,
    include_exts: List[str],
    exclude_dir_regex: str,
    exclude_file_regex: str,
    exclude_packaging_files: bool,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    chunk_lines: int,
    chunk_overlap: int,
    strip_docstrings_for_scoring: bool,
    strip_comments_for_scoring: bool,
    bm25_top_n: int,
    top_k: int,
    fusion: str,  # rrf | linear | semantic
    alpha: float,  # used in linear
    rrf_k: int,
    rrf_w_bm25: float,
    rrf_w_emb: float,
    embed_model: str,
    embed_device: str,
    embed_max_length: int,
    embed_batch_size: int,
    max_snippet_chars: int,
) -> Tuple[str, Dict[str, Any]]:
    chunks, index_stats = collect_repo_chunks_in_container(
        env=env,
        repo_root=repo_root,
        include_exts=include_exts,
        exclude_dir_regex=exclude_dir_regex,
        exclude_file_regex=exclude_file_regex,
        exclude_packaging_files=exclude_packaging_files,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        chunk_lines=chunk_lines,
        chunk_overlap=chunk_overlap,
        strip_docstrings_for_scoring=strip_docstrings_for_scoring,
        strip_comments_for_scoring=strip_comments_for_scoring,
    )
    if not chunks:
        return "", {"index_stats": index_stats, "note": "no chunks collected"}

    docs = [c.score_text for c in chunks]
    bm25_scores = bm25_rank(query, docs)
    if not bm25_scores:
        return "", {"index_stats": index_stats, "note": "no bm25 scores produced"}

    idx_sorted = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)
    top_idx = idx_sorted[: max(1, min(bm25_top_n, len(idx_sorted)))]

    cand_chunks = [chunks[i] for i in top_idx]
    cand_bm25 = [float(bm25_scores[i]) for i in top_idx]
    cand_bm25_norm = _minmax_norm(cand_bm25)

    # Embeddings for candidates
    emb = get_cached_embedder(embed_model, embed_device=embed_device)
    q_vec = emb.encode([f"query: {query}"], max_length=embed_max_length, batch_size=1)
    c_vecs = emb.encode(
        [f"passage: {c.score_text}" for c in cand_chunks],
        max_length=embed_max_length,
        batch_size=embed_batch_size,
    )
    cosines = cosine_sim_matrix(q_vec, c_vecs)

    fusion = (fusion or "rrf").lower().strip()
    if fusion not in {"rrf", "linear", "semantic"}:
        raise ValueError("fusion must be one of: rrf, linear, semantic")

    if fusion == "semantic":
        finals = [float(cosines[i]) for i in range(len(cand_chunks))]
        used_rrf = [0.0 for _ in cand_chunks]
    elif fusion == "linear":
        finals = [float(cand_bm25_norm[i]) + float(alpha) * float(cosines[i]) for i in range(len(cand_chunks))]
        used_rrf = [0.0 for _ in cand_chunks]
    else:
        # ranks (1 = best)
        bm25_order = sorted(range(len(cand_chunks)), key=lambda i: cand_bm25[i], reverse=True)
        bm25_rank_list = [0] * len(cand_chunks)
        for r, i in enumerate(bm25_order, start=1):
            bm25_rank_list[i] = r

        emb_order = sorted(range(len(cand_chunks)), key=lambda i: cosines[i], reverse=True)
        emb_rank_list = [0] * len(cand_chunks)
        for r, i in enumerate(emb_order, start=1):
            emb_rank_list[i] = r

        used_rrf = rrf_fusion(
            bm25_rank_list,
            emb_rank_list,
            k=rrf_k,
            w_a=rrf_w_bm25,
            w_b=rrf_w_emb,
        )
        finals = [float(used_rrf[i]) for i in range(len(cand_chunks))]

    order = sorted(range(len(cand_chunks)), key=lambda i: finals[i], reverse=True)
    order = order[: max(1, min(top_k, len(order)))]

    scored: List[ScoredChunk] = []
    for r, i in enumerate(order, start=1):
        scored.append(
            ScoredChunk(
                chunk=cand_chunks[i],
                bm25=float(cand_bm25[i]),
                cosine=float(cosines[i]),
                bm25_norm=float(cand_bm25_norm[i]),
                rrf=float(used_rrf[i]) if used_rrf else 0.0,
                final=float(finals[i]),
                rank=r,
            )
        )

    retrieved_text = format_retrieved_context(scored, max_snippet_chars=max_snippet_chars)
    meta = {
        "mode": "hybrid",
        "fusion": fusion,
        "repo_root": repo_root,
        "index_stats": index_stats,
        "n_chunks": len(chunks),
        "bm25_top_n": bm25_top_n,
        "top_k": top_k,
        "alpha": alpha,
        "rrf": {"k": rrf_k, "w_bm25": rrf_w_bm25, "w_emb": rrf_w_emb},
        "embed_model": embed_model,
        "embed_device": emb.device,
        "strip_docstrings_for_scoring": strip_docstrings_for_scoring,
        "strip_comments_for_scoring": strip_comments_for_scoring,
        "exclude_dir_regex": exclude_dir_regex,
        "exclude_file_regex": exclude_file_regex,
        "exclude_packaging_files": exclude_packaging_files,
        "selected": [
            {
                "rank": s.rank,
                "path": s.chunk.path,
                "start_line": s.chunk.start_line,
                "end_line": s.chunk.end_line,
                "bm25": s.bm25,
                "bm25_norm": s.bm25_norm,
                "cosine": s.cosine,
                "rrf": s.rrf,
                "final": s.final,
            }
            for s in scored
        ],
    }
    return retrieved_text, meta

# ----------------------------
# Instance processing
# ----------------------------

def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    *,
    mode: str,  # baseline | bm25 | hybrid
    rag_only: bool,
    repo_root: str,
    include_exts: List[str],
    exclude_dir_regex: str,
    exclude_file_regex: str,
    exclude_packaging_files: bool,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    chunk_lines: int,
    chunk_overlap: int,
    strip_docstrings_for_scoring: bool,
    strip_comments_for_scoring: bool,
    bm25_top_n: int,
    top_k: int,
    fusion: str,
    alpha: float,
    rrf_k: int,
    rrf_w_bm25: float,
    rrf_w_emb: float,
    embed_model: str,
    embed_device: str,
    embed_max_length: int,
    embed_batch_size: int,
    max_snippet_chars: int,
) -> None:
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model_obj = get_model(config=config.get("model", {}))
    base_task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent: Optional[ProgressTrackingAgent] = None
    extra_info: Optional[Dict[str, Any]] = None
    env: Optional[Environment] = None
    result = ""
    exit_status: Any = "unknown"

    try:
        env = get_sb_environment(config, instance)

        retrieval_meta: Optional[Dict[str, Any]] = None
        retrieved_ctx = ""

        if mode == "baseline":
            if rag_only:
                exit_status = "rag_only_baseline_noop"
                result = ""
                progress_manager.update_instance_status(instance_id, "Baseline rag-only: nothing to retrieve")
            else:
                progress_manager.update_instance_status(instance_id, "Agent run (baseline)")
                agent = ProgressTrackingAgent(
                    model_obj, env, progress_manager=progress_manager, instance_id=instance_id, **config.get("agent", {})
                )
                exit_status, result = agent.run(base_task)

        elif mode == "bm25":
            progress_manager.update_instance_status(instance_id, "RAG: indexing + BM25")
            retrieved_ctx, retrieval_meta = retrieve_bm25_only(
                env=env,
                query=base_task,
                repo_root=repo_root,
                include_exts=include_exts,
                exclude_dir_regex=exclude_dir_regex,
                exclude_file_regex=exclude_file_regex,
                exclude_packaging_files=exclude_packaging_files,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                chunk_lines=chunk_lines,
                chunk_overlap=chunk_overlap,
                strip_docstrings_for_scoring=strip_docstrings_for_scoring,
                strip_comments_for_scoring=strip_comments_for_scoring,
                top_k=top_k,
                max_snippet_chars=max_snippet_chars,
            )
            (instance_dir / "retrieval.json").write_text(json.dumps(retrieval_meta, indent=2, ensure_ascii=False))
            (instance_dir / "retrieved_context.txt").write_text(retrieved_ctx, encoding="utf-8")
            if isinstance(retrieval_meta, dict) and "index_stats" in retrieval_meta:
                (instance_dir / "index_stats.json").write_text(
                    json.dumps(retrieval_meta["index_stats"], indent=2, ensure_ascii=False)
                )

            if rag_only:
                exit_status = "rag_only"
                result = ""
                progress_manager.update_instance_status(instance_id, "RAG-only: saved retrieval logs")
            else:
                task = build_augmented_prompt(base_task, retrieved_ctx)
                progress_manager.update_instance_status(instance_id, "Agent run (BM25)")
                agent = ProgressTrackingAgent(
                    model_obj, env, progress_manager=progress_manager, instance_id=instance_id, **config.get("agent", {})
                )
                exit_status, result = agent.run(task)

        elif mode == "hybrid":
            progress_manager.update_instance_status(instance_id, "RAG: indexing + BM25 + embeddings + fusion")
            retrieved_ctx, retrieval_meta = retrieve_hybrid(
                env=env,
                query=base_task,
                repo_root=repo_root,
                include_exts=include_exts,
                exclude_dir_regex=exclude_dir_regex,
                exclude_file_regex=exclude_file_regex,
                exclude_packaging_files=exclude_packaging_files,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                chunk_lines=chunk_lines,
                chunk_overlap=chunk_overlap,
                strip_docstrings_for_scoring=strip_docstrings_for_scoring,
                strip_comments_for_scoring=strip_comments_for_scoring,
                bm25_top_n=bm25_top_n,
                top_k=top_k,
                fusion=fusion,
                alpha=alpha,
                rrf_k=rrf_k,
                rrf_w_bm25=rrf_w_bm25,
                rrf_w_emb=rrf_w_emb,
                embed_model=embed_model,
                embed_device=embed_device,
                embed_max_length=embed_max_length,
                embed_batch_size=embed_batch_size,
                max_snippet_chars=max_snippet_chars,
            )
            (instance_dir / "retrieval.json").write_text(json.dumps(retrieval_meta, indent=2, ensure_ascii=False))
            (instance_dir / "retrieved_context.txt").write_text(retrieved_ctx, encoding="utf-8")
            if isinstance(retrieval_meta, dict) and "index_stats" in retrieval_meta:
                (instance_dir / "index_stats.json").write_text(
                    json.dumps(retrieval_meta["index_stats"], indent=2, ensure_ascii=False)
                )

            if rag_only:
                exit_status = "rag_only"
                result = ""
                progress_manager.update_instance_status(instance_id, "RAG-only: saved retrieval logs")
            else:
                task = build_augmented_prompt(base_task, retrieved_ctx)
                progress_manager.update_instance_status(instance_id, "Agent run (HYBRID)")
                agent = ProgressTrackingAgent(
                    model_obj, env, progress_manager=progress_manager, instance_id=instance_id, **config.get("agent", {})
                )
                exit_status, result = agent.run(task)

        else:
            raise ValueError("mode must be baseline, bm25, or hybrid")

    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}

    finally:
        if env and hasattr(env, "stop"):
            try:
                env.stop()
            except Exception:
                pass

        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )

        # In rag-only mode: do not write empty patch (keeps harness clean)
        if not rag_only:
            model_name = ""
            try:
                model_name = getattr(model_obj, "config", {}).get("model_name", "")  # type: ignore
            except Exception:
                model_name = str(getattr(model_obj, "model_name", ""))
            update_preds_file(output_dir / "preds.json", instance_id, model_name, result)

        progress_manager.on_instance_end(instance_id, str(exit_status))

# ----------------------------
# Filtering
# ----------------------------

def filter_instances(
    instances: List[dict],
    *,
    filter_spec: str,
    slice_spec: str = "",
    shuffle: bool = False,
) -> List[dict]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)

    before = len(instances)
    if filter_spec:
        instances = [it for it in instances if re.match(filter_spec, it["instance_id"])]
    if len(instances) != before:
        logger.info(f"Instance filter: {before} -> {len(instances)} instances")

    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        logger.info(f"Instance slice -> {len(instances)} instances")
    return instances

# ----------------------------
# Main
# ----------------------------

# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset or dataset path", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice spec (e.g., '0:5')", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Worker threads", rich_help_panel="Basic"),
    model: Optional[str] = typer.Option(None, "-m", "--model", help="Model name", rich_help_panel="Basic"),
    model_class: Optional[str] = typer.Option(None, "--model-class", help="Model class", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "swebench.yaml", "--config", help="Config file", rich_help_panel="Basic"),
    environment_class: Optional[str] = typer.Option(None, "--environment-class", help="docker/singularity", rich_help_panel="Advanced"),

    # ---- Experiment mode ----
    mode: str = typer.Option("baseline", "--mode", help="baseline | bm25 | hybrid", rich_help_panel="Retrieval"),
    rag_only: bool = typer.Option(False, "--rag-only", help="Only run retrieval + save logs, do not run agent", rich_help_panel="Retrieval"),

    repo_root: str = typer.Option("/testbed", "--repo-root", help="Repo root inside container", rich_help_panel="Retrieval"),

    # tests toggle -> implemented via exclude_dir_regex preset
    include_tests: bool = typer.Option(True, "--include-tests/--exclude-tests", help="Index tests/ directories (recommended)", rich_help_panel="Retrieval"),

    include_exts: str = typer.Option(
        ".py,.pyi,.js,.ts,.java,.go,.rs,.cpp,.c,.h,.hpp",
        "--include-exts",
        help="Comma-separated extensions to index",
        rich_help_panel="Retrieval"
    ),
    exclude_file_regex: str = typer.Option(
        r"(?i)(^|/)(changelog|changes|history|release[-_ ]notes|news|readme|license|copying|contributing|code_of_conduct)(\..*)?$"
        r"|(^|/)\.?(github|gitlab|circleci)(/|$)"
        r"|(^|/)(docs?|doc|site|website|_build)(/|$)"
        ,
        "--exclude-file-regex",
        help="Regex for file paths (relative) to exclude.",
        rich_help_panel="Retrieval"
    ),
    exclude_packaging_files: bool = typer.Option(
        True,
        "--exclude-packaging-files",
        help="Exclude common packaging/config files (setup.py, pyproject.toml, requirements.txt, tox.ini, ...).",
        rich_help_panel="Retrieval"
    ),

    max_files: int = typer.Option(6000, "--max-files", help="Max files to index", rich_help_panel="Retrieval"),
    max_file_bytes: int = typer.Option(400000, "--max-file-bytes", help="Skip files larger than this", rich_help_panel="Retrieval"),
    max_total_bytes: int = typer.Option(30000000, "--max-total-bytes", help="Stop indexing after this many bytes", rich_help_panel="Retrieval"),
    chunk_lines: int = typer.Option(90, "--chunk-lines", help="Lines per chunk", rich_help_panel="Retrieval"),
    chunk_overlap: int = typer.Option(20, "--chunk-overlap", help="Line overlap between chunks", rich_help_panel="Retrieval"),

    strip_docstrings_for_scoring: bool = typer.Option(True, "--strip-docstrings-for-scoring/--keep-docstrings-for-scoring", help="Strip top-level module docstring for scoring text", rich_help_panel="Retrieval"),
    strip_comments_for_scoring: bool = typer.Option(False, "--strip-comments-for-scoring/--keep-comments-for-scoring", help="Strip line comments for scoring text", rich_help_panel="Retrieval"),

    # BM25 / hybrid selection sizes
    bm25_top_n: int = typer.Option(30, "--bm25-top-n", help="BM25 candidate pool size (hybrid only)", rich_help_panel="Retrieval"),
    top_k: int = typer.Option(8, "--top-k", help="Final snippets injected", rich_help_panel="Retrieval"),

    # Fusion controls (hybrid only)
    fusion: str = typer.Option("rrf", "--fusion", help="hybrid fusion: rrf | linear | semantic", rich_help_panel="Retrieval"),
    alpha: float = typer.Option(2.0, "--alpha", help="linear fusion: bm25_norm + alpha*cos", rich_help_panel="Retrieval"),
    rrf_k: int = typer.Option(60, "--rrf-k", help="RRF constant k", rich_help_panel="Retrieval"),
    rrf_w_bm25: float = typer.Option(1.0, "--rrf-w-bm25", help="RRF weight for BM25 rank", rich_help_panel="Retrieval"),
    rrf_w_emb: float = typer.Option(1.0, "--rrf-w-emb", help="RRF weight for embedding rank", rich_help_panel="Retrieval"),

    # Embeddings (hybrid only)
    embed_model: str = typer.Option("intfloat/e5-small-v2", "--embed-model", help="HF embedding model", rich_help_panel="Retrieval"),
    embed_device: str = typer.Option("auto", "--embed-device", help="auto|cpu|cuda", rich_help_panel="Retrieval"),
    embed_max_length: int = typer.Option(512, "--embed-max-length", help="Embedding tokenizer max length", rich_help_panel="Retrieval"),
    embed_batch_size: int = typer.Option(16, "--embed-batch-size", help="Embedding batch size", rich_help_panel="Retrieval"),

    max_snippet_chars: int = typer.Option(4500, "--max-snippet-chars", help="Truncate snippet text injected", rich_help_panel="Retrieval"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    if not redo_existing and not rag_only and (output_path / "preds.json").exists():
        try:
            existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        except Exception:
            existing_instances = []
        if existing_instances:
            logger.info(f"Skipping {len(existing_instances)} existing instances")
            instances = [it for it in instances if it["instance_id"] not in existing_instances]

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

    include_exts_list = [e.strip().lower() for e in include_exts.split(",") if e.strip()]
    if mode in {"bm25", "hybrid"} and not include_exts_list:
        raise typer.BadParameter("include_exts cannot be empty when mode != baseline")

    # Exclude regex presets (this is the key: tests toggle actually changes behavior)
    exclude_dir_regex_with_tests = r"(^|/)(test_data|docs?|examples?|benchmarks?|\.git|\.github|dist|build|node_modules|venv|\.venv)(/|$)"
    exclude_dir_regex_no_tests   = r"(^|/)(tests?|test_data|docs?|examples?|benchmarks?|\.git|\.github|dist|build|node_modules|venv|\.venv)(/|$)"
    exclude_dir_regex = exclude_dir_regex_with_tests if include_tests else exclude_dir_regex_no_tests

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: Dict[concurrent.futures.Future, str]) -> None:
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
            futures: Dict[concurrent.futures.Future, str] = {}
            for instance in instances:
                fut = executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                    mode=mode.lower().strip(),
                    rag_only=rag_only,
                    repo_root=repo_root,
                    include_exts=include_exts_list,
                    exclude_dir_regex=exclude_dir_regex,
                    exclude_file_regex=exclude_file_regex,
                    exclude_packaging_files=exclude_packaging_files,
                    max_files=max_files,
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    chunk_lines=chunk_lines,
                    chunk_overlap=chunk_overlap,
                    strip_docstrings_for_scoring=strip_docstrings_for_scoring,
                    strip_comments_for_scoring=strip_comments_for_scoring,
                    bm25_top_n=bm25_top_n,
                    top_k=top_k,
                    fusion=fusion,
                    alpha=alpha,
                    rrf_k=rrf_k,
                    rrf_w_bm25=rrf_w_bm25,
                    rrf_w_emb=rrf_w_emb,
                    embed_model=embed_model,
                    embed_device=embed_device,
                    embed_max_length=embed_max_length,
                    embed_batch_size=embed_batch_size,
                    max_snippet_chars=max_snippet_chars,
                )
                futures[fut] = instance["instance_id"]

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