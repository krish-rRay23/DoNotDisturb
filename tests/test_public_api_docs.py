"""Public functions keep a docstring and annotations. This does not check a result."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "adaptive_plasticity"


def _public_functions(tree: ast.AST):
    found = []

    def walk(nodes, prefix: str) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                found.append((f"{prefix}{node.name}", node))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                walk(node.body, prefix=node.name + ".")

    walk(tree.body, "")
    return found


def _missing_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    missing = []
    args = node.args
    names = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    for arg in names:
        if arg.arg in {"self", "cls"}:
            continue
        if arg.annotation is None:
            missing.append(arg.arg)
    if args.vararg is not None and args.vararg.annotation is None:
        missing.append("*" + args.vararg.arg)
    if args.kwarg is not None and args.kwarg.annotation is None:
        missing.append("**" + args.kwarg.arg)
    if node.returns is None:
        missing.append("return")
    return missing


def test_public_functions_have_docstrings_and_annotations():
    gaps = []
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, node in _public_functions(tree):
            if not ast.get_docstring(node):
                gaps.append(f"{path.name}:{name} docstring")
            missing = _missing_annotations(node)
            if missing:
                gaps.append(f"{path.name}:{name} {missing}")
    assert gaps == []
