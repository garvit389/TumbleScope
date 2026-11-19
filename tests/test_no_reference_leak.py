import ast
from inspect import signature
from pathlib import Path
import re

import pytest

from pipeline.controller import analyze


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MODULE = "ingestion.reference"
REFERENCE_NAME = "ReferenceLabel"


def _path_parts(node, bindings):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [part for part in re.split(r"[/\\]+", node.value.casefold()) if part and part != "."]
    if isinstance(node, ast.Name):
        return bindings.get(node.id, [])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _path_parts(node.left, bindings) + _path_parts(node.right, bindings)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Path":
            return [part for arg in node.args for part in _path_parts(arg, bindings)]
        if isinstance(func, ast.Attribute) and func.attr in ("join", "joinpath"):
            return _path_parts(func.value, bindings) + [
                part for arg in node.args for part in _path_parts(arg, bindings)
            ]
    return []


def _reference_access(tree):
    paths = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    paths[target.id] = _path_parts(node.value, paths)
        parts = _path_parts(node, paths)
        if ("data", "reference") in list(zip(parts, parts[1:])):
            yield node.lineno
        if isinstance(node, ast.Constant) and node.value == FORBIDDEN_MODULE:
            yield node.lineno


def test_inference_has_no_reference_import_or_argument():
    folders = ("ml", "physics", "verdict", "reconciliation", "classical_analysis",
               "preprocessing", "pipeline", "app")
    for folder in folders:
        for source in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            assert not (lines := list(_reference_access(tree))), (source, lines)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(not item.name.startswith(FORBIDDEN_MODULE) for item in node.names), source
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert not module.startswith(FORBIDDEN_MODULE), source
                    assert all(item.name != REFERENCE_NAME for item in node.names), source
                    if module == "ingestion":
                        assert all(item.name != "reference" for item in node.names), source
                elif isinstance(node, ast.Attribute):
                    assert node.attr != REFERENCE_NAME, source
    assert tuple(signature(analyze).parameters) == ("obs", "cfg", "mode", "progress")


@pytest.mark.parametrize("source", [
    'open("data/reference/labels.csv")',
    r'open("data\\reference\\labels.csv")',
    'DATA = ROOT / "data"\n(DATA / "reference" / "labels.csv").read_bytes()',
    'Path("data").joinpath("reference", "labels.csv").read_text()',
    'importlib.import_module("ingestion.reference")',
])
def test_guard_finds_direct_reference_paths_and_dynamic_imports(source):
    assert list(_reference_access(ast.parse(source)))
