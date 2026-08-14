"""Every `a.something` a handler reads must be an argument its own subparser registers.

WHY THIS EXISTS

`--out-gene-key` was added to the `background` subparser and read by the `annotate` handler,
because the edit that inserted it anchored on `--gene-key` and `background` happens to define one
too, earlier in the file. argparse is perfectly happy: the option exists, just on the wrong
command. Nothing fails until someone runs `annotate` and it reaches the line - which, in a
ten-library sweep, was after clustering and after the background had been built.

That is the third defect of this shape in one sitting. The other two:

    from . import declaration        in a module the orchestrator runs as a SCRIPT (scQC)
    from .emit import annotate_obs   scoped to the --out-h5ad branch, used in the --report branch

All three are invisible to a unit test of the thing itself, all three fire only when a
particular path runs, and all three cost a long job. `--help` catches none of them: the option
is registered, the import is syntactically fine, the branch is never taken.

So this is static, and it reads the parser rather than trusting it.

    python tests/test_cli_wiring.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


SRC = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)

#: argparse sets these itself, or the handler is entitled to them regardless of the subparser.
ALWAYS = {"fn", "cmd"}


def dest_of(call):
    """The attribute name argparse will store this add_argument under."""
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    name = call.args[0].value
    return name.lstrip("-").replace("-", "_")


def build_parser_map():
    """{handler_name: {dest, ...}}, by reading the add_parser/add_argument calls in line order.

    The parser is built inline in `main`, not in a named factory, so this walks the whole module
    and relies on source order - which is what argparse relies on too.
    """
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)]
    calls.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    out, current = {}, None
    for call in calls:
        attr = getattr(call.func, "attr", None)
        if attr == "add_parser" and call.args and isinstance(call.args[0], ast.Constant):
            current = f"::{call.args[0].value}"
            out.setdefault(current, set())
        elif attr == "add_argument" and current:
            d = dest_of(call)
            if d:
                out[current].add(d)
        elif attr == "set_defaults" and current:
            for kw in call.keywords:
                if kw.arg == "fn" and isinstance(kw.value, ast.Name):
                    out[kw.value.id] = out.pop(current)
                    current = kw.value.id
    return {k: v for k, v in out.items() if not k.startswith("::")}


def reads_of(handler):
    """Every `a.<attr>` the handler reads, where `a` is its single parameter."""
    fn = next((n for n in ast.walk(TREE)
               if isinstance(n, ast.FunctionDef) and n.name == handler), None)
    if fn is None or not fn.args.args:
        return set()
    param = fn.args.args[0].arg
    out = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == param):
            out.add(node.attr)
    return out


print("\n1 - the parser map reads")
pmap = build_parser_map()
check("handlers were found", len(pmap) >= 6, f"{len(pmap)}: {sorted(pmap)}")
check("annotate's handler is among them", "_annotate" in pmap, str(sorted(pmap)))

print("\n2 - every attribute a handler reads is registered on ITS subparser")
for handler, dests in sorted(pmap.items()):
    missing = sorted(reads_of(handler) - dests - ALWAYS)
    check(f"{handler}", not missing,
          f"reads {missing} which its subparser does not define" if missing else
          f"{len(dests)} options")

print("\n3 - the specific regression: --out-gene-key belongs to annotate, not background")
check("annotate registers it", "out_gene_key" in pmap.get("_annotate", set()))
check("background does NOT", "out_gene_key" not in pmap.get("_background", set()))
check("and annotate is what reads it", "out_gene_key" in reads_of("_annotate"))

print("\n" + "=" * 64)
if fails:
    print(f"cli wiring: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("cli wiring OK - no handler reads an option its own subparser does not define")
