import ast
from pathlib import Path


def test_single_feature_builder_definition():
    root = Path(__file__).resolve().parents[1]
    modules = ("app", "classical_analysis", "core", "evaluation", "ingestion", "ml", "physics",
               "pipeline", "preprocessing", "reconciliation", "scripts", "verdict")
    definitions = [
        path.relative_to(root)
        for module in modules
        for path in (root / module).rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_cnn_windows"
    ]
    assert definitions == [Path("ml/features.py")]
