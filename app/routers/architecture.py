"""Architecture tab endpoints: browse project, read sources, build import graph."""

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.paths import LLOYD_HOME


router = APIRouter()

_ARCH_ALLOWED_ROOTS = [LLOYD_HOME]

_ARCH_SKIP_DIRS = {
    ".git", "node_modules", ".venvs", "__pycache__",
    "sessions", "logs", "dist", ".next", ".cache",
}

_ARCH_SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}

_ARCH_LANG_MAP = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".json": "json",
    ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".css": "css", ".html": "html",
}


def _arch_safe_path(raw: str) -> Path:
    """Resolve and validate that a path is under an allowed root."""
    resolved = Path(raw).resolve()
    for root in _ARCH_ALLOWED_ROOTS:
        root_str = str(root.resolve())
        if resolved == root.resolve() or str(resolved).startswith(root_str + "/"):
            return resolved
    raise HTTPException(status_code=403, detail="Path not allowed")


@router.get("/api/architecture/browse")
async def architecture_browse(path: str = ""):
    """Browse project directory structure for the Architecture tab."""
    if not path:
        path = str(LLOYD_HOME)
    safe = _arch_safe_path(path)
    if not safe.exists() or not safe.is_dir():
        return JSONResponse({"entries": []})

    entries = []
    try:
        for entry in sorted(safe.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            name = entry.name
            if name.startswith(".") or name in _ARCH_SKIP_DIRS:
                continue
            if entry.is_dir():
                try:
                    children = sum(1 for c in entry.iterdir()
                                   if not c.name.startswith(".") and c.name not in _ARCH_SKIP_DIRS)
                except PermissionError:
                    children = 0
                entries.append({"name": name, "path": str(entry), "type": "dir", "children": children})
            elif entry.is_file():
                entries.append({"name": name, "path": str(entry), "type": "file", "size": entry.stat().st_size})
    except PermissionError:
        pass
    return JSONResponse({"entries": entries})


@router.get("/api/architecture/read")
async def architecture_read(path: str = ""):
    """Read a source file for the Architecture tab."""
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    safe = _arch_safe_path(path)
    if not safe.exists() or not safe.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    language = _ARCH_LANG_MAP.get(safe.suffix.lower(), "text")
    try:
        content = safe.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read error: {e}")
    return JSONResponse({
        "path": str(safe),
        "content": content,
        "language": language,
        "lineCount": content.count("\n") + 1,
    })


@router.get("/api/architecture/graph")
async def architecture_graph():
    """Build import dependency graph for the project."""
    root = LLOYD_HOME.resolve()

    files: dict[str, Path] = {}  # relative_path -> absolute_path
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in _ARCH_SOURCE_EXTENSIONS:
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in _ARCH_SKIP_DIRS for part in rel_parts):
            continue
        rel = str(p.relative_to(root))
        files[rel] = p

    py_import_re = re.compile(
        r'^\s*(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w., ]+))',
        re.MULTILINE,
    )
    ts_import_re = re.compile(
        r'''import\s+(?:(?:\{[^}]*\}|[\w*]+(?:\s*,\s*\{[^}]*\})?)\s+from\s+)?['"]([^'"]+)['"]''',
        re.MULTILINE,
    )
    ts_named_re = re.compile(
        r'''import\s+\{([^}]*)\}\s+from\s+['"]([^'"]+)['"]''',
        re.MULTILINE,
    )

    def resolve_py_import(module: str) -> str | None:
        parts = module.split(".")
        candidate = root / (parts[0] + ".py")
        if candidate.exists():
            return str(candidate.relative_to(root))
        candidate = root / parts[0] / "__init__.py"
        if candidate.exists():
            return str(candidate.relative_to(root))
        return None

    def resolve_ts_import(spec: str, source_file: Path) -> str | None:
        if not spec.startswith("."):
            return None
        base = (source_file.parent / spec).resolve()
        if base.is_file() and base.suffix in _ARCH_SOURCE_EXTENSIONS:
            try:
                return str(base.relative_to(root))
            except ValueError:
                return None
        for ext in (".ts", ".tsx", ".js", ".jsx"):
            candidate = base.parent / (base.name + ext)
            if candidate.exists():
                try:
                    return str(candidate.relative_to(root))
                except ValueError:
                    pass
        for idx in ("index.ts", "index.tsx", "index.js"):
            candidate = base / idx
            if candidate.exists():
                try:
                    return str(candidate.relative_to(root))
                except ValueError:
                    pass
        return None

    nodes: dict[str, dict] = {}
    links: list[dict] = []

    for rel, filepath in files.items():
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lang = _ARCH_LANG_MAP.get(filepath.suffix, "text")
        import_count = 0

        if lang == "python":
            for m in py_import_re.finditer(content):
                from_module, from_symbols, import_modules = m.group(1), m.group(2), m.group(3)
                if from_module:
                    target = resolve_py_import(from_module)
                    symbols = [s.strip() for s in from_symbols.split(",")]
                    if target and target in files:
                        links.append({"source": str(filepath), "target": str(files[target]), "symbols": symbols})
                        import_count += 1
                elif import_modules:
                    for mod in import_modules.split(","):
                        target = resolve_py_import(mod.strip())
                        if target and target in files:
                            links.append({"source": str(filepath), "target": str(files[target]), "symbols": []})
                            import_count += 1
        elif lang in ("typescript", "javascript"):
            symbol_map: dict[str, list[str]] = {}
            for m in ts_named_re.finditer(content):
                symbols = [s.strip().split(" as ")[0].strip() for s in m.group(1).split(",") if s.strip()]
                symbol_map[m.group(2)] = symbols
            for m in ts_import_re.finditer(content):
                spec = m.group(1)
                target = resolve_ts_import(spec, filepath)
                if target and target in files:
                    links.append({"source": str(filepath), "target": str(files[target]), "symbols": symbol_map.get(spec, [])})
                    import_count += 1

        exports: list[str] = []
        if lang == "python":
            for em in re.finditer(r'^\s*(?:def|class)\s+(\w+)', content, re.MULTILINE):
                exports.append(em.group(1))
        elif lang in ("typescript", "javascript"):
            for em in re.finditer(r'export\s+(?:default\s+)?(?:function|class|const|let|var|interface|type|enum)\s+(\w+)', content, re.MULTILINE):
                exports.append(em.group(1))

        nodes[rel] = {
            "id": str(filepath),
            "path": str(filepath),
            "count": import_count,
            "lang": lang,
            "exports": exports[:20],
        }

    node_list = list(nodes.values())
    return JSONResponse({
        "nodes": node_list,
        "links": links,
        "totalImports": sum(n["count"] for n in node_list),
        "totalNodes": len(node_list),
        "totalLinks": len(links),
    })
