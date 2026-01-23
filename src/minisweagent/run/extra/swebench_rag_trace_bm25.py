"""
mini-SWE-agent runner for SWE-bench with reproducible retrieval experiments:

Adds Trace-to-Retrieve for bm25:
  - Run FAIL_TO_PASS tests in container
  - Parse trace (nodeid/exception/frames/import errors/assertion diffs)
  - Build trace queries
  - BM25 retrieval uses problem statement, trace queries, or both
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

_HELP_TEXT = """Run mini-SWE-agent on SWE-bench (baseline or BM25 retrieval), with optional Trace-to-Retrieve."""

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

import re

_TEST_PATTERNS = (
    r"^test/",
    r"^tests/",
    r"/test_",
    r"_test\.py$",
    r"::test",
)

_GENERIC_ERRORS = {
    "AssertionError", "Exception", "Error", "FAILED", "ERROR",
    "ImportError", "ModuleNotFoundError", "KeyError",
    "TypeError", "ValueError",
}

def _is_testish_query(q: str) -> bool:
    q = (q or "").strip().replace("\\", "/")
    return any(re.search(p, q) for p in _TEST_PATTERNS)

def _is_too_generic(q: str) -> bool:
    q = (q or "").strip()
    return (q in _GENERIC_ERRORS) or (len(q.split()) == 1 and q.endswith("Error"))

def filter_trace_queries(qs: List[str], *, exclude_tests: bool = True) -> List[str]:
    out: List[str] = []
    for q in qs or []:
        if not q or not q.strip():
            continue
        if exclude_tests and _is_testish_query(q):
            continue
        if _is_too_generic(q):
            continue
        out.append(q.strip())
    # dedupe, preserve order
    return list(dict.fromkeys(out))

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
    text: str
    score_text: str

@dataclass
class ScoredChunk:
    chunk: Chunk
    bm25: float
    bm25_norm: float
    final: float
    rank: int

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
# Chunk collection inside container
# ----------------------------

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
    lines_score = list(lines_inject)  # same length

    if strip_comments:
        new_lines = []
        for ln in lines_score:
            if ln.startswith("#!"):
                new_lines.append(ln); continue
            if "coding" in ln and ln.strip().startswith("#"):
                new_lines.append(ln); continue
            if ln.lstrip().startswith("#"):
                new_lines.append("\\n" if ln.endswith("\\n") else "")
                continue
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

    lines_inject = text_content.splitlines(True)
    lines_score  = build_lines_score(lines_inject, strip_docstrings=strip_doc, strip_comments=strip_comments)

    step = max(1, chunk_lines - chunk_overlap)
    n = len(lines_inject)

    for start in range(0, n, step):
        end = min(n, start + chunk_lines)
        inj_txt = "".join(lines_inject[start:end])
        sc_txt  = "".join(lines_score[start:end])
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
                stats = json.loads(line[len("===INDEX_STATS==="):])
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
            f"bm25={s.bm25:.3f} bm25_norm={s.bm25_norm:.3f} final={s.final:.6f}"
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

# ============================================================
# Trace-to-Retrieve (NEW): run tests, parse trace, build queries
# ============================================================

def _safe_one_line(s: str, max_len: int = 240) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s

def ensure_editable_install_if_needed(env: Environment, repo_root: str = "/testbed") -> Dict[str, Any]:
    """
    Some images fail because importlib.metadata.version("pkg") can't find dist metadata
    unless 'pip install -e .' is executed. We'll try it if importing package fails.
    Safe: if it’s already installed, this is quick.
    """
    cmd = f"""bash -lc '
set -e
cd {repo_root}
python - << "PY"
import sys
try:
    import importlib.metadata as md
    # Not knowing package name here — so just exit 0 and let failures surface in pytest.
    print("python_ok", sys.version)
except Exception as e:
    print("python_meta_err", repr(e))
PY
'
"""
    out = env.execute(cmd)
    stdout, stderr, rc = _get_exec_streams(out)
    return {"rc": rc, "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-2000:]}

def checkout_base_and_apply_test_patch(env: Environment, instance: dict, repo_root: str = "/testbed") -> None:
    base = instance.get("base_commit", "")
    patch = instance.get("test_patch", "") or ""
    # Apply exactly like your manual pipeline: reset/clean -> checkout base -> apply test_patch
    cmd = f"""bash -lc '
set -e
cd {repo_root}
git reset --hard
git clean -fdx
git checkout -f {base}
if [ -n "{patch.strip().replace('"', '\\"')}" ]; then
  cat > /tmp/test.patch << "PATCH"
{patch}
PATCH
  git apply /tmp/test.patch
fi
echo "OK: base_commit + test_patch applied"
'
"""
    out = env.execute(cmd)
    stdout, stderr, rc = _get_exec_streams(out)
    if rc != 0:
        raise RuntimeError(f"Failed to checkout/apply test patch (rc={rc}).\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}")

def run_fail_to_pass_and_capture_trace(
    env: Environment,
    instance: dict,
    *,
    repo_root: str,
    max_tests: int,
    pytest_extra_args: str,
) -> Dict[str, Any]:
    """
    Runs up to max_tests FAIL_TO_PASS tests, captures combined output.
    Returns:
      {
        "ran": [nodeid...],
        "rcs": [int...],
        "raw": "combined output",
      }
    """
    f2p = instance.get("FAIL_TO_PASS") or instance.get("fail_to_pass") or []
    if isinstance(f2p, str):
        s = f2p.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith('"[') and s.endswith(']"')):
            try:
                f2p = json.loads(s)
            except Exception:
                f2p = [f2p]
        else:
            f2p = [f2p]
    tests = list(f2p)[:max_tests]
    if not tests:
        return {"ran": [], "rcs": [], "raw": ""}

    combined = []
    rcs = []
    ran = []

    for t in tests:
        ran.append(t)

        cmd = f"""bash -lc '
set -e
cd {repo_root}
pytest -q --tb=long -vv {pytest_extra_args} "{t}"
'
"""
        out = env.execute(cmd)
        stdout, stderr, rc = _get_exec_streams(out)
        rcs.append(int(rc))

        extra = pytest_extra_args.strip()
        cmd_line = f"pytest -q --tb=long -vv {extra} {t}".strip()
        combined.append(f"### PYTEST_CMD: {cmd_line}\n")
        combined.append(stdout)

        if stderr.strip():
            combined.append("\n[STDERR]\n" + stderr + "\n")

        # Stop after first failure (best signal for trace→retrieve)
        if rc != 0:
            break

    return {
        "ran": ran,
        "rcs": rcs,
        "raw": "\n".join(combined),
    }

def _extract_inner_exception(txt: str) -> tuple[Optional[str], Optional[str]]:
    """
    Prefer the real underlying exception if it's embedded inside a wrapper assertion.
    Handles patterns like:
      <Result AttributeError("...")>
      AttributeError: ...
    """
    # Pattern A: Click Result wrapper
    m = re.search(r"Result\s+([A-Za-z_][A-Za-z0-9_\.]*)\(\s*([\"'])(.+?)\2\s*\)", txt)
    if m:
        etype = m.group(1).split(".")[-1]
        emsg = m.group(3).strip()
        return etype, emsg

    # Pattern B: normal Python exception line anywhere
    m = re.search(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*Error)\s*:\s*(.+)\s*$", txt, flags=re.M)
    if m:
        etype = m.group(1).split(".")[-1]
        emsg = m.group(2).strip()
        return etype, emsg

    return None, None

def parse_trace_text(trace_txt: str) -> Dict[str, Any]:
    """
    Parses pytest output for:
      - normal FAILED tests
      - collection errors: "ERROR collecting ..."
      - import errors while loading conftest
    Extracts:
      - failing_nodeid (best-effort)
      - exception type/message
      - frames from both "file.py:line: in func" and "file.py:line: Exception"
    """
    txt = trace_txt or ""
    lines = txt.splitlines()

    def norm(p: str) -> str:
        return (p or "").replace("\\", "/").replace("/testbed/", "")

    failing_nodeid = None
    import_error = None

    # 1) FAILED <nodeid>
    m = re.search(r"^FAILED\s+(\S+)", txt, flags=re.M)
    if m:
        failing_nodeid = m.group(1)

    # 2) ERROR collecting <file>
    if not failing_nodeid:
        m = re.search(r"^ERROR collecting\s+([A-Za-z0-9_./\\-]+)", txt, flags=re.M)
        if m:
            failing_nodeid = norm(m.group(1))

    # 3) Summary line: "ERROR test/foo.py - KeyError: ..."
    if not failing_nodeid:
        m = re.search(r"^ERROR\s+([A-Za-z0-9_./\\-]+)\s+-\s+([A-Za-z_][A-Za-z0-9_\.]*):\s*(.*)$", txt, flags=re.M)
        if m:
            failing_nodeid = norm(m.group(1))

    # 4) ImportError while loading conftest
    m = re.search(r"^ImportError while loading conftest\s+'([^']+)'", txt, flags=re.M)
    if m:
        import_error = norm(m.group(1))

    # 5) exception type/message
    exc_type = None
    exc_msg = None

    inner_t, inner_m = _extract_inner_exception(txt)
    if inner_t:
        exc_type, exc_msg = inner_t, inner_m


    # prefer the last "E   Type: message" if multiple
    for ln in lines:
        m2 = re.match(r"^E\s+([A-Za-z_][A-Za-z0-9_\.]*):\s*(.*)\s*$", ln)
        if m2:
            exc_type = m2.group(1).split(".")[-1]
            exc_msg = m2.group(2).strip()

    # if still none, try summary "ERROR ... - Type: message"
    if exc_type is None:
        m = re.search(r"^ERROR\s+[A-Za-z0-9_./\\-]+\s+-\s+([A-Za-z_][A-Za-z0-9_\.]*):\s*(.*)$", txt, flags=re.M)
        if m:
            exc_type = m.group(1).split(".")[-1]
            exc_msg = m.group(2).strip()

    # assertion fallback
    if exc_type is None and "AssertionError" in txt:
        exc_type = "AssertionError"
        exc_msg = ""

    # 6) frames: support both formats
    frames: List[Dict[str, Any]] = []

    # format A: file.py:123: in func
    pat_in = re.compile(r"^\s*([A-Za-z0-9_./\\-]+\.py)\s*:\s*(\d+)\s*:\s*in\s*(.*)\s*$")

    # format B: file.py:123: ExceptionType  (collection errors often show this)
    pat_exc = re.compile(r"^\s*([A-Za-z0-9_./\\-]+\.py)\s*:\s*(\d+)\s*:\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*$")

    for ln in lines:
        m3 = pat_in.match(ln)
        if m3:
            frames.append({"file": norm(m3.group(1)), "line": int(m3.group(2)), "in": m3.group(3).strip()})
            continue
        m4 = pat_exc.match(ln)
        if m4:
            frames.append({"file": norm(m4.group(1)), "line": int(m4.group(2)), "in": ""})

    kind = "pytest_failure"
    if import_error:
        kind = "import_error"
    elif "ERROR collecting" in txt or "Interrupted: 1 error during collection" in txt:
        kind = "collection_error"

    return {
        "raw_kind": kind,
        "failing_nodeid": failing_nodeid,
        "import_error": import_error,
        "exception": {"type": exc_type, "message": exc_msg},
        "frames": frames[:40],
        "raw_tail": "\n".join(lines[-140:]),
    }
def build_trace_queries(parsed: Dict[str, Any]) -> List[str]:
    qs: List[str] = []

    def add(q: str) -> None:
        q = _safe_one_line(q, 220)
        if q and q not in qs:
            qs.append(q)

    nodeid = (parsed.get("failing_nodeid") or "").strip()
    exc = parsed.get("exception") or {}
    exc_t = (exc.get("type") or "").strip()
    exc_m = (exc.get("message") or "").strip()
    tail = parsed.get("raw_tail") or ""
    frames = parsed.get("frames") or []

    # 1) strongest: exception type + message
    if exc_t and exc_m:
        add(f"{exc_t} {exc_m}")
    elif exc_t:
        add(exc_t)

    # 2) pull “X has no attribute Y” structure -> add X and Y explicitly
    m = re.search(r"'([A-Za-z_][A-Za-z0-9_]*)'.+no attribute\s+'([A-Za-z_][A-Za-z0-9_]*)'", exc_m)
    if m:
        add(m.group(1))              # HookRelay
        add(m.group(2))              # load_default_config
        add(f"{m.group(1)} {m.group(2)}")

    # 3) failing test info
    if nodeid:
        add(nodeid)
        add(Path(nodeid).name)       # commands_test.py::...

    # 4) top frame file(s)
    for fr in frames[:3]:
        fp = (fr.get("file") or "").strip()
        if fp:
            add(fp)
            add(Path(fp).name)

    # 5) last resort: scan tail for Result AttributeError(...) if parse missed it
    m = re.search(r"Result\s+([A-Za-z_][A-Za-z0-9_\.]*)\(\s*[\"'](.+?)[\"']\s*\)", tail)
    if m:
        add(f"{m.group(1).split('.')[-1]} {m.group(2)}")

    # remove noisy short tokens
    bad = {"default", "pytest", "pluggy", "cachedir", "rootdir", "plugins"}
    qs = [q for q in qs if len(q) >= 6 and q.lower() not in bad]

    return qs[:10]

# ----------------------------
# Retrieval (BM25) with multi-query fusion (NEW)
# ----------------------------

def _rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    rank_lists: each is doc indices sorted by descending score for that query.
    returns dict idx -> fused score
    """
    fused: Dict[int, float] = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + r)
    return fused

def retrieve_bm25(
    env: Environment,
    queries: List[str],
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
    fuse: str,
    rrf_k: int,
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

    # compute per-query bm25 scores
    per_q_scores: List[List[float]] = []
    per_q_sorted: List[List[int]] = []

    for q in queries:
        scores = bm25_rank(q, docs)
        if not scores:
            continue
        per_q_scores.append(scores)
        per_q_sorted.append(sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True))

    if not per_q_scores:
        return "", {"index_stats": index_stats, "note": "no bm25 scores produced"}

    fuse = (fuse or "max").lower().strip()
    if fuse not in ("max", "rrf"):
        fuse = "max"

    if fuse == "max":
        fused_scores = [0.0] * len(chunks)
        for scores in per_q_scores:
            for i, s in enumerate(scores):
                if s > fused_scores[i]:
                    fused_scores[i] = float(s)
        idx_sorted = sorted(range(len(chunks)), key=lambda i: fused_scores[i], reverse=True)
    else:
        fused_rrf = _rrf_fuse(per_q_sorted, k=rrf_k)
        idx_sorted = sorted(fused_rrf.keys(), key=lambda i: fused_rrf[i], reverse=True)

    idx_sorted = idx_sorted[: max(1, min(top_k, len(idx_sorted)))]

    # for logging, store “final” score as fused (max or rrf)
    if fuse == "max":
        picked_scores = [float((0.0 if i >= len(chunks) else fused_scores[i])) for i in idx_sorted]
    else:
        picked_scores = [float(_rrf_fuse(per_q_sorted, k=rrf_k).get(i, 0.0)) for i in idx_sorted]

    picked_norm = _minmax_norm(picked_scores)

    scored: List[ScoredChunk] = []
    for r, (i, b, bn) in enumerate(zip(idx_sorted, picked_scores, picked_norm), start=1):
        scored.append(
            ScoredChunk(
                chunk=chunks[i],
                bm25=float(b),
                bm25_norm=float(bn),
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
        "queries": queries,
        "bm25_fuse": fuse,
        "rrf_k": rrf_k,
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
    mode: str,
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
    top_k: int,
    max_snippet_chars: int,
    # trace-to-retrieve
    query_source: str,
    trace_max_tests: int,
    pytest_extra_args: str,
    bm25_fuse: str,
    rrf_k: int,
    editable_install: bool,
) -> None:
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model_obj = get_model(config=config.get("model", {}))
    def get_problem_text(instance: dict) -> str:
        for k in ("problem_statement", "problem", "issue", "instruction", "text", "prompt"):
            v = instance.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    base_task = get_problem_text(instance)
    (instance_dir / "base_task.txt").write_text(
        base_task,
        encoding="utf-8",
        errors="backslashreplace",
    )

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

        trace_raw = ""
        trace_parsed = None
        trace_queries: List[str] = []

        # ---------- NEW: prepare repo state + optionally run trace ----------
        if mode == "bm25":
            progress_manager.update_instance_status(instance_id, "Checkout base + apply test_patch")
            checkout_base_and_apply_test_patch(env, instance, repo_root=repo_root)

            if editable_install:
                progress_manager.update_instance_status(instance_id, "Editable install (pip -e .)")
                out = env.execute(f"bash -lc 'cd {repo_root} && pip install -e .'")
                stdout, stderr, rc = _get_exec_streams(out)
                (instance_dir / "editable_install.txt").write_text(
                    f"RC={rc}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n", encoding="utf-8"
                )

            qs_source = (query_source or "problem").lower().strip()
            if qs_source in ("trace", "both"):
                progress_manager.update_instance_status(instance_id, "Run FAIL_TO_PASS to collect trace")
                tr = run_fail_to_pass_and_capture_trace(
                    env,
                    instance,
                    repo_root=repo_root,
                    max_tests=trace_max_tests,
                    pytest_extra_args=pytest_extra_args,
                )
                trace_raw = tr.get("raw", "")
                (instance_dir / "trace.txt").write_text(trace_raw, encoding="utf-8", errors="backslashreplace")

                trace_parsed = parse_trace_text(trace_raw)
                (instance_dir / "trace.json").write_text(json.dumps(trace_parsed, indent=2, ensure_ascii=False))

                raw_trace_queries = build_trace_queries(trace_parsed)
                trace_queries = filter_trace_queries(raw_trace_queries, exclude_tests=True)

                (instance_dir / "trace_queries.json").write_text(
                    json.dumps(
                        {
                            "raw": raw_trace_queries,
                            "filtered": trace_queries,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )


        # ---------- baseline ----------
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

        # ---------- bm25 (with query-source: problem/trace/both) ----------
        elif mode == "bm25":
            qs_source = (query_source or "problem").lower().strip()
            if qs_source not in ("problem", "trace", "both"):
                qs_source = "problem"

            queries: List[str] = []
            if base_task.strip():
                queries.append(base_task.strip())  # include problem statement
            if qs_source in ("trace", "both"):
                # fallback: if trace parsing failed, still include raw tail as last resort
                if trace_queries:
                    queries.extend(trace_queries)
                elif trace_parsed:
                # Only use fallback if it contains real code signal (not just pytest header / test paths)
                    fallback = trace_parsed.get("raw_tail") or ""
                    fallback = _safe_one_line(fallback, 220)
                if fallback and not _is_testish_query(fallback) and not fallback.startswith("### PYTEST_CMD:"):
                    queries.append(fallback)

            # If somehow empty, fallback to problem statement
            if not queries and base_task.strip():
                queries = [base_task.strip()]

            progress_manager.update_instance_status(instance_id, f"RAG: indexing + BM25 ({qs_source})")
            retrieved_ctx, retrieval_meta = retrieve_bm25(
                env=env,
                queries=queries,
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
                fuse=bm25_fuse,
                rrf_k=rrf_k,
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

        else:
            raise ValueError("mode must be baseline or bm25")

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
    mode: str = typer.Option("baseline", "--mode", help="baseline | bm25", rich_help_panel="Retrieval"),
    rag_only: bool = typer.Option(False, "--rag-only", help="Only run retrieval + save logs, do not run agent", rich_help_panel="Retrieval"),

    repo_root: str = typer.Option("/testbed", "--repo-root", help="Repo root inside container", rich_help_panel="Retrieval"),

    include_tests: bool = typer.Option(True, "--include-tests/--exclude-tests", help="Index tests/ directories", rich_help_panel="Retrieval"),

    include_exts: str = typer.Option(
        ".py,.pyi,.js,.ts,.java,.go,.rs,.cpp,.c,.h,.hpp",
        "--include-exts",
        help="Comma-separated extensions to index",
        rich_help_panel="Retrieval"
    ),
    exclude_file_regex: str = typer.Option(
        r"(?i)(^|/)(changelog|changes|history|release[-_ ]notes|news|readme|license|copying|contributing|code_of_conduct)(\..*)?$"
        r"|(^|/)\.?(github|gitlab|circleci)(/|$)"
        r"|(^|/)(docs?|doc|site|website|_build)(/|$)",
        "--exclude-file-regex",
        help="Regex for file paths (relative) to exclude.",
        rich_help_panel="Retrieval"
    ),
    exclude_packaging_files: bool = typer.Option(
        True,
        "--exclude-packaging-files",
        help="Exclude common packaging/config files.",
        rich_help_panel="Retrieval"
    ),

    max_files: int = typer.Option(6000, "--max-files", help="Max files to index", rich_help_panel="Retrieval"),
    max_file_bytes: int = typer.Option(400000, "--max-file-bytes", help="Skip files larger than this", rich_help_panel="Retrieval"),
    max_total_bytes: int = typer.Option(30000000, "--max-total-bytes", help="Stop indexing after this many bytes", rich_help_panel="Retrieval"),
    chunk_lines: int = typer.Option(90, "--chunk-lines", help="Lines per chunk", rich_help_panel="Retrieval"),
    chunk_overlap: int = typer.Option(20, "--chunk-overlap", help="Line overlap", rich_help_panel="Retrieval"),

    strip_docstrings_for_scoring: bool = typer.Option(True, "--strip-docstrings-for-scoring/--keep-docstrings-for-scoring", help="Strip docstrings for scoring text", rich_help_panel="Retrieval"),
    strip_comments_for_scoring: bool = typer.Option(False, "--strip-comments-for-scoring/--keep-comments-for-scoring", help="Strip comments for scoring text", rich_help_panel="Retrieval"),

    top_k: int = typer.Option(8, "--top-k", help="Final snippets injected", rich_help_panel="Retrieval"),
    max_snippet_chars: int = typer.Option(4500, "--max-snippet-chars", help="Truncate injected text", rich_help_panel="Retrieval"),

    # -------- Trace-to-Retrieve knobs --------
    query_source: str = typer.Option(
        "problem",
        "--query-source",
        help="problem | trace | both  (what BM25 queries use)",
        rich_help_panel="Trace-to-Retrieve"
    ),
    trace_max_tests: int = typer.Option(
        1,
        "--trace-max-tests",
        help="Run up to N FAIL_TO_PASS tests to collect trace (usually 1 is enough).",
        rich_help_panel="Trace-to-Retrieve"
    ),
    pytest_extra_args: str = typer.Option(
        "",
        "--pytest-extra-args",
        help="Extra pytest args appended (e.g. '--maxfail=1').",
        rich_help_panel="Trace-to-Retrieve"
    ),
    editable_install: bool = typer.Option(
        False,
        "--editable-install",
        help="Run 'pip install -e .' inside container before running tests (fixes PackageNotFoundError cases).",
        rich_help_panel="Trace-to-Retrieve"
    ),

    bm25_fuse: str = typer.Option(
        "max",
        "--bm25-fuse",
        help="How to fuse multiple query results: max | rrf",
        rich_help_panel="Trace-to-Retrieve"
    ),
    rrf_k: int = typer.Option(
        60,
        "--rrf-k",
        help="RRF constant (only used if --bm25-fuse rrf).",
        rich_help_panel="Trace-to-Retrieve"
    ),
) -> None:
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
    mode_norm = (mode or "baseline").lower().strip()

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
                    mode=mode_norm,
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
                    top_k=top_k,
                    max_snippet_chars=max_snippet_chars,
                    query_source=query_source,
                    trace_max_tests=trace_max_tests,
                    pytest_extra_args=pytest_extra_args,
                    bm25_fuse=bm25_fuse,
                    rrf_k=rrf_k,
                    editable_install=editable_install,
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


