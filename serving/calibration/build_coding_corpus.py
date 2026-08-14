#!/usr/bin/env python3
"""Build a deterministic, local-only coding calibration corpus.

The builder reads committed material from the NemoClaw-Thor, manyforge, and
manyforge_specs checkouts. It never invokes a network client. The output is a
tokenizer-independent messages JSONL; render it with the released model's
tokenizer before passing it to ModelOpt.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SEED = 386027
SOURCE_SIZE = 768
SELECTED_SIZE = 512

CATEGORY_RATIOS = {
    "implementation": 0.30,
    "debugging_testing": 0.20,
    "code_review": 0.15,
    "architecture_refactoring": 0.15,
    "shell_tool_json": 0.10,
    "ci_config_docs_repo": 0.10,
}

# Character bins are only used to ensure that the corpus contains a useful
# long-context tail. There is deliberately no tokenizer-level truncation here.
LENGTH_RATIOS = {
    "short": 0.35,       # under roughly 1.5K tokens
    "medium": 0.30,      # roughly 1.5K-4K tokens
    "long": 0.25,        # roughly 4K-10K tokens
    "very_long": 0.10,   # roughly 10K-20K+ tokens
}

CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".cu", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".jsx", ".kt", ".lua", ".php", ".py", ".rb", ".rs",
    ".scala", ".sql", ".ts", ".tsx",
}
CONFIG_SUFFIXES = {".json", ".jsonl", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}
SHELL_SUFFIXES = {".sh", ".bash", ".zsh", ".fish", ".ps1"}
ALLOWED_SUFFIXES = CODE_SUFFIXES | CONFIG_SUFFIXES | DOC_SUFFIXES | SHELL_SUFFIXES

EXCLUDED_PARTS = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".venv",
    "__pycache__", "blobs", "build", "coverage", "dist", "generated", "node_modules",
    "target", "vendor", "venv",
}
EXCLUDED_NAMES = {
    ".env", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock",
    "uv.lock", "Cargo.lock",
}
EXCLUDED_PATH_FRAGMENTS = {
    "/benchmarks/",
    "/smoke-evidence/",
    "serving/agentic-bench/",
    "smoke_corpus",
}

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
ASSIGNMENT_SECRET = re.compile(
    r"(?im)^(\s*(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*)([^\s#]+)"
)

SYSTEM_PROMPTS = {
    "implementation": [
        "You are a coding agent. Implement the requested change precisely and keep it testable.",
        "Act as a senior software engineer. Produce a focused implementation with maintainable code.",
        "Work inside an existing repository. Respect its conventions and make the smallest complete change.",
        "Implement the repository task and preserve compatibility with surrounding components.",
    ],
    "debugging_testing": [
        "Diagnose failures from code and test evidence, then provide a concrete validated resolution.",
        "Act as a debugging engineer. Trace the cause before proposing the repair and verification.",
        "Use the available failure evidence to isolate regressions and strengthen the tests.",
        "Investigate the reported behavior and return a reproducible technical resolution.",
    ],
    "code_review": [
        "Review code for correctness, regressions, maintainability, and missing validation.",
        "Act as a senior reviewer. Prioritize actionable defects and explain their impact clearly.",
        "Audit the change carefully, focusing on behavioral risks and concrete evidence.",
        "Perform a rigorous code review and keep findings technically specific.",
    ],
    "architecture_refactoring": [
        "Design and explain software architecture with explicit boundaries, tradeoffs, and migration steps.",
        "Act as a software architect. Preserve contracts while simplifying the implementation.",
        "Reason about component ownership, interfaces, failure modes, and incremental refactoring.",
        "Develop a practical architecture proposal grounded in the existing repository.",
    ],
    "shell_tool_json": [
        "Produce exact shell commands or structured tool payloads that are safe and reproducible.",
        "Operate as a coding agent using terminal tools and structured data with precise syntax.",
        "Return runnable automation and validate assumptions before destructive operations.",
        "Use shell, JSON, or configuration tools carefully and report observable results.",
    ],
    "ci_config_docs_repo": [
        "Explore the repository and update CI, configuration, or technical documentation consistently.",
        "Trace repository conventions before modifying build, deployment, or documentation artifacts.",
        "Maintain operational configuration and docs as executable, verifiable engineering artifacts.",
        "Inspect the repository structure and provide an accurate integration-oriented result.",
    ],
}

USER_PROMPTS = {
    "implementation": [
        "Implement this repository task from the recorded engineering change: {title}",
        "Apply the following requested change while preserving surrounding behavior: {title}",
        "Produce the implementation for this scoped task: {title}",
        "Complete this code change and include the relevant implementation details: {title}",
    ],
    "debugging_testing": [
        "Diagnose and resolve this failure or test gap: {title}",
        "Use the repository evidence to fix and validate the following issue: {title}",
        "Investigate this regression and show the concrete test-oriented resolution: {title}",
        "Trace the cause of this behavior and provide the corresponding repair: {title}",
    ],
    "code_review": [
        "Review the following repository artifact and report the substantive engineering findings: {title}",
        "Audit this implementation or investigation for correctness and operational risk: {title}",
        "Perform a focused code review using this repository context: {title}",
        "Identify the important defects, constraints, and verification evidence in: {title}",
    ],
    "architecture_refactoring": [
        "Explain or implement the architecture and boundary decisions for: {title}",
        "Develop a maintainable design or refactoring plan for: {title}",
        "Reason through the interfaces, ownership, and migration implications of: {title}",
        "Provide the architecture-level result captured by this repository artifact: {title}",
    ],
    "shell_tool_json": [
        "Provide the exact automation or structured tool interaction for: {title}",
        "Turn this operational requirement into safe runnable commands or configuration: {title}",
        "Produce the shell, JSON, or tool-oriented implementation for: {title}",
        "Show the precise repository automation needed for: {title}",
    ],
    "ci_config_docs_repo": [
        "Inspect the repository and provide the CI, configuration, or documentation result for: {title}",
        "Update or explain the integration and operational artifact for: {title}",
        "Use repository context to complete this build, configuration, or documentation task: {title}",
        "Document and validate the repository workflow represented by: {title}",
    ],
}


@dataclass(frozen=True)
class Candidate:
    category: str
    repo: str
    revision: str
    path: str
    source_type: str
    title: str
    assistant: str
    language: str

    @property
    def digest(self) -> str:
        material = "\0".join(
            [self.category, self.repo, self.revision, self.path, self.source_type, self.title, self.assistant]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @property
    def content_digest(self) -> str:
        return hashlib.sha256(self.assistant.encode("utf-8")).hexdigest()


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def allocate(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    result = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(ratios, key=lambda name: (-(raw[name] - result[name]), name))
    for name in order[:remaining]:
        result[name] += 1
    return result


def language_for(path: str) -> str:
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "dockerfile"
    return {
        ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
        ".jsx": "javascript", ".rs": "rust", ".c": "c", ".h": "c-cpp", ".cc": "cpp",
        ".cpp": "cpp", ".hpp": "cpp", ".cu": "cuda", ".go": "go", ".java": "java",
        ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell",
        ".json": "json", ".jsonl": "jsonl", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".ini": "ini", ".cfg": "config", ".md": "markdown",
        ".mdx": "markdown", ".rst": "rst", ".txt": "text", ".sql": "sql",
    }.get(suffix, "text")


def safe_path(path: str) -> bool:
    item = Path(path)
    normalized = "/" + path.lower().lstrip("/")
    if any(fragment in normalized for fragment in EXCLUDED_PATH_FRAGMENTS):
        return False
    lower_parts = {part.lower() for part in item.parts}
    if lower_parts & EXCLUDED_PARTS:
        return False
    if item.name in EXCLUDED_NAMES or item.name.startswith(".env."):
        return False
    suffix = item.suffix.lower()
    return suffix in ALLOWED_SUFFIXES or item.name.lower().startswith("dockerfile")


def sanitize(text: str) -> str | None:
    if "\x00" in text or any(pattern.search(text) for pattern in SECRET_PATTERNS[-1:]):
        return None
    text = text.replace("/home/tndlux", "$HOME")
    for pattern in SECRET_PATTERNS[:-1]:
        text = pattern.sub("[REDACTED_TOKEN]", text)
    text = ASSIGNMENT_SECRET.sub(r"\1[REDACTED]", text)
    text = re.sub(r"\n{5,}", "\n\n\n", text)
    return text.strip()


def length_bin(chars: int) -> str:
    if chars < 6_000:
        return "short"
    if chars < 16_000:
        return "medium"
    if chars < 40_000:
        return "long"
    return "very_long"


def file_categories(path: str, text: str) -> list[str]:
    lower = path.lower()
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    first = text[:4_000].lower()
    result: list[str] = []

    if suffix in CODE_SUFFIXES and not re.search(r"(^|/)(tests?|testdata|fixtures?)(/|$)", lower):
        result.append("implementation")
    if re.search(r"(^|/)(tests?|testdata|fixtures?)(/|$)", lower) or any(
        word in lower for word in ("debug", "failure", "incident", "smoke", "test", "benchmark", "evidence")
    ):
        result.append("debugging_testing")
    if any(word in lower for word in ("audit", "review", "investigation", "finding", "security", "report")) or any(
        phrase in first for phrase in ("root cause", "review finding", "observed failure", "regression")
    ):
        result.append("code_review")
    if any(word in lower for word in ("architecture", "design", "adr", "spec", "plan", "migration", "refactor")):
        result.append("architecture_refactoring")
    if suffix in SHELL_SUFFIXES | CONFIG_SUFFIXES or "tool" in lower or "scripts/" in lower:
        result.append("shell_tool_json")
    if (
        lower.startswith(".github/")
        or "docker" in name
        or suffix in CONFIG_SUFFIXES
        or suffix in DOC_SUFFIXES
        or any(word in lower for word in ("config", "deploy", "setup", "readme", "workflow", "runbook"))
    ):
        result.append("ci_config_docs_repo")

    return list(dict.fromkeys(result))


def excerpt_variants(text: str) -> list[str]:
    """Keep natural documents while providing multiple non-overlapping long windows."""
    text = text.strip()
    if len(text) <= 64_000:
        return [text]
    variants = []
    window = 56_000
    for start in range(0, min(len(text), window * 3), window):
        chunk = text[start : start + window]
        if start:
            newline = chunk.find("\n")
            if newline >= 0:
                chunk = chunk[newline + 1 :]
        variants.append(chunk.strip())
    return [value for value in variants if len(value) >= 400]


def file_candidates(repo: Path, repo_name: str, revision: str) -> Iterable[Candidate]:
    for relative in run_git(repo, "ls-files").splitlines():
        if not safe_path(relative):
            continue
        try:
            # Read the committed object, not the working tree. This keeps the
            # corpus reproducible from the revision recorded in the manifest
            # and excludes unrelated local edits.
            raw = run_git(repo, "show", f"{revision}:{relative}")
            if len(raw) > 1_500_000:
                continue
        except subprocess.CalledProcessError:
            continue
        cleaned = sanitize(raw)
        if cleaned is None or len(cleaned) < 400:
            continue
        categories = file_categories(relative, cleaned)
        if not categories:
            continue
        for index, excerpt in enumerate(excerpt_variants(cleaned)):
            title = f"{repo_name}/{relative}"
            if index:
                title += f" (section {index + 1})"
            for category in categories:
                yield Candidate(
                    category=category,
                    repo=repo_name,
                    revision=revision,
                    path=relative,
                    source_type="repository_file",
                    title=title,
                    assistant=excerpt,
                    language=language_for(relative),
                )


def classify_commit(subject: str, paths: list[str]) -> list[str]:
    lower = subject.lower() + " " + " ".join(paths).lower()
    categories: list[str] = []
    if any(word in lower for word in ("fix", "bug", "test", "debug", "regression", "failure", "smoke")):
        categories.append("debugging_testing")
    if any(word in lower for word in ("review", "audit", "security", "hardening", "finding")):
        categories.append("code_review")
    if any(word in lower for word in ("architecture", "design", "adr", "spec", "refactor", "migration")):
        categories.append("architecture_refactoring")
    if any(Path(path).suffix.lower() in SHELL_SUFFIXES | CONFIG_SUFFIXES for path in paths):
        categories.append("shell_tool_json")
    if any(
        path.startswith(".github/")
        or "docker" in path.lower()
        or Path(path).suffix.lower() in DOC_SUFFIXES | CONFIG_SUFFIXES
        for path in paths
    ):
        categories.append("ci_config_docs_repo")
    if not categories or any(Path(path).suffix.lower() in CODE_SUFFIXES for path in paths):
        categories.append("implementation")
    return list(dict.fromkeys(categories))


def commit_candidates(repo: Path, repo_name: str, revision: str) -> Iterable[Candidate]:
    hashes = run_git(repo, "log", "--format=%H", "-n", "600").splitlines()
    for commit in hashes:
        subject = run_git(repo, "show", "-s", "--format=%s", commit).strip()
        body = run_git(repo, "show", "-s", "--format=%b", commit).strip()
        paths = [
            path for path in run_git(repo, "show", "--format=", "--name-only", commit).splitlines()
            if path and safe_path(path)
        ]
        if not paths:
            continue
        patch = run_git(
            repo, "show", "--format=", "--no-ext-diff", "--no-color", "--unified=3", "--", *paths[:24]
        )
        cleaned = sanitize(patch)
        if cleaned is None or len(cleaned) < 500:
            continue
        if len(cleaned) > 72_000:
            cleaned = cleaned[:72_000].rsplit("\n", 1)[0] + "\n[large patch excerpt ends here]"
        request = subject if not body else f"{subject}\n\n{body}"
        language_counts = collections.Counter(language_for(path) for path in paths)
        language = language_counts.most_common(1)[0][0]
        for category in classify_commit(subject, paths):
            yield Candidate(
                category=category,
                repo=repo_name,
                revision=revision,
                path=",".join(paths[:8]),
                source_type="git_change",
                title=request,
                assistant=cleaned,
                language=language,
            )


def choose_template(options: list[str], digest: str) -> str:
    return options[int(digest[:8], 16) % len(options)]


def candidate_to_record(candidate: Candidate) -> dict:
    digest = candidate.digest
    system = choose_template(SYSTEM_PROMPTS[candidate.category], digest)
    user = choose_template(USER_PROMPTS[candidate.category], digest[8:]).format(title=candidate.title)
    assistant = candidate.assistant
    return {
        "id": f"coding-{digest[:20]}",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "category": candidate.category,
            "source_type": candidate.source_type,
            "repo": candidate.repo,
            "revision": candidate.revision,
            "path": candidate.path,
            "language": candidate.language,
            "assistant_chars": len(assistant),
            "length_bin": length_bin(len(assistant)),
            "content_sha256": hashlib.sha256(assistant.encode("utf-8")).hexdigest(),
        },
    }


def select_balanced(candidates: list[Candidate], total: int, seed: int) -> list[Candidate]:
    category_targets = allocate(total, CATEGORY_RATIOS)
    length_targets = allocate(total, LENGTH_RATIOS)
    selected: list[Candidate] = []
    used_content: set[str] = set()
    used_origins: collections.Counter[tuple[str, str]] = collections.Counter()
    actual_lengths: collections.Counter[str] = collections.Counter()

    for category, category_target in category_targets.items():
        pool = [candidate for candidate in candidates if candidate.category == category]
        rng = random.Random(seed ^ int(hashlib.sha256(category.encode()).hexdigest()[:12], 16))
        rng.shuffle(pool)
        picked = 0
        while picked < category_target:
            viable = [
                candidate for candidate in pool
                if candidate.content_digest not in used_content
                and used_origins[(candidate.repo, candidate.path)] < 2
            ]
            if not viable:
                viable = [candidate for candidate in pool if candidate.content_digest not in used_content]
            if not viable:
                raise RuntimeError(
                    f"Not enough unique candidates for {category}: need {category_target}, selected {picked}"
                )

            def score(candidate: Candidate) -> tuple:
                bin_name = length_bin(len(candidate.assistant))
                shortage = length_targets[bin_name] - actual_lengths[bin_name]
                origin_use = used_origins[(candidate.repo, candidate.path)]
                tie = hashlib.sha256(f"{seed}:{candidate.digest}".encode()).hexdigest()
                return (-shortage, origin_use, tie)

            choice = min(viable, key=score)
            selected.append(choice)
            used_content.add(choice.content_digest)
            used_origins[(choice.repo, choice.path)] += 1
            actual_lengths[length_bin(len(choice.assistant))] += 1
            picked += 1

    random.Random(seed).shuffle(selected)
    return selected


def select_subset(source: list[Candidate], total: int, seed: int) -> list[Candidate]:
    # Reuse the same balancing algorithm but constrain candidates to the source pool.
    return select_balanced(source, total, seed)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize(records: list[dict]) -> dict:
    categories = collections.Counter(record["metadata"]["category"] for record in records)
    lengths = collections.Counter(record["metadata"]["length_bin"] for record in records)
    languages = collections.Counter(record["metadata"]["language"] for record in records)
    repos = collections.Counter(record["metadata"]["repo"] for record in records)
    return {
        "records": len(records),
        "categories": dict(sorted(categories.items())),
        "length_bins": dict(sorted(lengths.items())),
        "languages": dict(languages.most_common()),
        "repos": dict(sorted(repos.items())),
        "assistant_chars": sum(record["metadata"]["assistant_chars"] for record in records),
    }


def validate(records: list[dict], expected_total: int) -> None:
    if len(records) != expected_total:
        raise RuntimeError(f"Expected {expected_total} records, found {len(records)}")
    expected_categories = allocate(expected_total, CATEGORY_RATIOS)
    actual_categories = collections.Counter(record["metadata"]["category"] for record in records)
    if dict(actual_categories) != expected_categories:
        raise RuntimeError(f"Category mismatch: expected {expected_categories}, found {dict(actual_categories)}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate record ids detected")
    content_hashes = [record["metadata"]["content_sha256"] for record in records]
    if len(content_hashes) != len(set(content_hashes)):
        raise RuntimeError("Duplicate assistant source content detected")
    for record in records:
        if [message["role"] for message in record["messages"]] != ["system", "user", "assistant"]:
            raise RuntimeError(f"Invalid message roles in {record['id']}")
        serialized = json.dumps(record, ensure_ascii=False)
        for pattern in SECRET_PATTERNS:
            if pattern.search(serialized):
                raise RuntimeError(f"Secret-like content survived sanitization in {record['id']}")


def parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemoclaw-thor", type=Path, default=default_repo)
    parser.add_argument("--manyforge", type=Path, default=default_repo.parent / "manyforge")
    parser.add_argument("--manyforge-specs", type=Path, default=default_repo.parent / "manyforge_specs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repos = {
        "NemoClaw-Thor": args.nemoclaw_thor.resolve(),
        "manyforge": args.manyforge.resolve(),
        "manyforge_specs": args.manyforge_specs.resolve(),
    }
    revisions: dict[str, str] = {}
    candidates: list[Candidate] = []
    for name, path in repos.items():
        if not (path / ".git").exists():
            raise RuntimeError(f"Required local git checkout not found: {path}")
        revision = run_git(path, "rev-parse", "HEAD").strip()
        revisions[name] = revision
        candidates.extend(file_candidates(path, name, revision))
        candidates.extend(commit_candidates(path, name, revision))

    # Exact duplicate candidates can arise when a file belongs to multiple source routes.
    unique = {candidate.digest: candidate for candidate in candidates}
    all_candidates = list(unique.values())
    source_candidates = select_balanced(all_candidates, SOURCE_SIZE, args.seed)
    selected_candidates = select_subset(source_candidates, SELECTED_SIZE, args.seed)
    source_records = [candidate_to_record(candidate) for candidate in source_candidates]
    selected_records = [candidate_to_record(candidate) for candidate in selected_candidates]
    validate(source_records, SOURCE_SIZE)
    validate(selected_records, SELECTED_SIZE)

    output_dir = args.output_dir.resolve()
    source_path = output_dir / "coding-source-768.messages.jsonl"
    selected_path = output_dir / "coding-selected-512.messages.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(source_path, source_records)
    write_jsonl(selected_path, selected_records)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "network_required": False,
        "tokenizer_rendered": False,
        "truncation": None,
        "category_ratios": CATEGORY_RATIOS,
        "length_ratios": LENGTH_RATIOS,
        "source_revisions": revisions,
        "candidate_count": len(all_candidates),
        "source": summarize(source_records),
        "selected": summarize(selected_records),
        "files": {
            source_path.name: hashlib.sha256(source_path.read_bytes()).hexdigest(),
            selected_path.name: hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
