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


def optional_dests():
    """Every dest argparse can hand the handler as None.

    A dest is optional when it declares `default=None`, or declares no default at all with no
    `action` to supply one and no `required=True` to guarantee a value. That is argparse's own
    rule, read off the same calls argparse reads.
    """
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "add_argument"]
    out = set()
    for call in calls:
        d = dest_of(call)
        if not d:
            continue
        kw = {k.arg: k.value for k in call.keywords}
        if "default" in kw:
            if isinstance(kw["default"], ast.Constant) and kw["default"].value is None:
                out.add(d)
            continue
        if "action" in kw:
            continue
        req = kw.get("required")
        if isinstance(req, ast.Constant) and req.value is True:
            continue
        out.add(d)
    return out


print("\n0 - an optional argument is never coerced as though it were required")
# WHAT THIS CAUGHT. `--gap-min` defaults to None, which does not mean "no bar" but "the bar
# classify picks for this weight source". A provenance record built it with `float(a.gap_min)`
# and raised TypeError - on PBS 702931, AFTER an eight-resolution sweep had been walked, at the
# last statement before anything was written. Coercing an optional is a whole class of that
# failure and it is invisible until the option is left out, which is the common case.
# A COERCION UNDER A NONE-CHECK IS CORRECT AND MUST NOT BE FLAGGED. The first version of this
# check flagged `None if a.gap_min is None else float(a.gap_min)` - the fix for the very defect
# it was written for - because it read the call and not the guard around it. A gate that fires
# on correct behaviour gets switched off, and this file has now learned that twice.
def _guarded(fn, param, attr):
    """Call nodes inside an `if`/`ifexp` whose test asks whether `param.attr` is None."""
    safe = set()
    for n in ast.walk(fn):
        if not isinstance(n, (ast.If, ast.IfExp)):
            continue
        t = ast.unparse(n.test)
        if f"{param}.{attr}" not in t or "None" not in t:
            continue
        for part in ((n.body if isinstance(n.body, list) else [n.body])
                     + (n.orelse if isinstance(n.orelse, list) else [n.orelse])):
            safe |= {id(c) for c in ast.walk(part) if isinstance(c, ast.Call)}
    return safe


_opt = optional_dests()
_bad = []
for _h in [n.name for n in ast.walk(TREE)
           if isinstance(n, ast.FunctionDef) and n.name.startswith("_")]:
    _fn = next(n for n in ast.walk(TREE)
               if isinstance(n, ast.FunctionDef) and n.name == _h)
    if not _fn.args.args:
        continue
    _p = _fn.args.args[0].arg
    for _n in ast.walk(_fn):
        if (isinstance(_n, ast.Call) and getattr(_n.func, "id", "") in ("float", "int")
                and len(_n.args) == 1 and isinstance(_n.args[0], ast.Attribute)
                and getattr(_n.args[0].value, "id", "") == _p
                and _n.args[0].attr in _opt
                and id(_n) not in _guarded(_fn, _p, _n.args[0].attr)):
            _bad.append(f"{_h}: {ast.unparse(_n)}")
check("no float()/int() over an argument that can be None", not _bad, "; ".join(_bad))
check("the rule found some optionals to check against", len(_opt) > 10, f"{len(_opt)} optionals")

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
