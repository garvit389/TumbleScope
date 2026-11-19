import ast
from pathlib import Path
import re

import pytest

from core.schemas import INCONCLUSIVE_CAUSES, REASON_CODES, SKIP_CODES, STAGE2_FAIL_REASONS


ROOT = Path(__file__).resolve().parents[1]
CODE_FIELDS = {"reasons", "reason_codes", "fail_reasons", "stage2_fail_reasons",
               "inconclusive_cause", "skip_code"}
CANONICAL = set(REASON_CODES + STAGE2_FAIL_REASONS + SKIP_CODES + INCONCLUSIVE_CAUSES)


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return node.slice.value
    return None


def _argument(call, name, position):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return call.args[position] if len(call.args) > position else None


def _code_values(node):
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value in CODE_FIELDS:
                yield value
    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(_name(target) in CODE_FIELDS for target in targets):
            yield node.value
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("append", "extend"):
            if _name(node.func.value) in CODE_FIELDS:
                yield from node.args
        constructor = _name(node.func)
        if constructor in ("QualityReport", "Verdict"):
            yield _argument(node, "reason_codes", 3 if constructor == "QualityReport" else 4)
        if constructor in ("EstimatorResult", "_unavailable"):
            status = _argument(node, "status", 1 if constructor == "EstimatorResult" else 2)
            skipped = isinstance(status, ast.Constant) and status.value == "skipped"
            if skipped or (constructor == "_unavailable" and status is None):
                yield _argument(node, "error", 3 if constructor == "EstimatorResult" else 1)


def _literal_violations(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in CANONICAL:
            yield (node.lineno, node.value)
        for value in _code_values(node):
            if value is None:
                continue
            for item in ast.walk(value):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    yield (item.lineno, item.value)


def test_reason_codes_match_readme():
    readme = next(path for path in ROOT.glob("*.md") if path.name.lower() == "readme.md")
    line = next(entry for entry in readme.read_text(encoding="utf-8").splitlines()
                if entry.startswith("Reason codes:"))
    assert tuple(re.findall(r"\x60([a-z_]+)\x60", line)) == REASON_CODES
    assert set(REASON_CODES).isdisjoint(STAGE2_FAIL_REASONS)


def test_runtime_code_uses_canonical_constants():
    removed = "alias_" + "suspected"
    for folder in ("verdict", "reconciliation", "preprocessing", "classical_analysis", "pipeline"):
        for source in (ROOT / folder).rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            assert removed not in text, source
            assert not (violations := list(_literal_violations(text))), (source, violations)

    documents = list((ROOT / "docs").rglob("*.md"))
    documents.extend(path for path in ROOT.glob("*.md") if path.name.lower() == "readme.md")
    for source in documents:
        assert removed not in source.read_text(encoding="utf-8"), source

    for folder in ("app", "core", "evaluation", "ingestion", "ml", "physics", "scripts"):
        for source in (ROOT / folder).rglob("*.py"):
            assert removed not in source.read_text(encoding="utf-8"), source


@pytest.mark.parametrize("source", [
    'reasons.append("invented")',
    'summary["stage2_fail_reasons"] = ["invented"]',
    'return EstimatorResult(method, "skipped", [], error="invented")',
    'return Verdict(label, None, None, [], ["invented"], summary, text)',
    'return {"inconclusive_cause": "invented"}',
])
def test_guard_catches_unknown_literal_codes(source):
    assert (2, "invented") in list(_literal_violations("def example():\n    " + source))


def test_guard_accepts_constants_and_technical_errors():
    source = '''
def example():
    reasons = [LOW_QUALITY]
    reasons.extend([CANDIDATE_INADEQUATE])
    return EstimatorResult(method, "failed", [], error="technical error")
'''
    assert list(_literal_violations(source)) == []
