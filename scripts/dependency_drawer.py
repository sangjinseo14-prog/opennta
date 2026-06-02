import ast
import colorsys
import os
import shutil
import sys

from graphviz import Digraph, ExecutableNotFound

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.join(REPO_ROOT, "src", "opennta")
ROOT_PACKAGE = "opennta"

# Modules whose edges (import connections) are omitted from the graph.
# These are shared/utility packages imported almost everywhere; drawing their
# edges adds noise without insight. The nodes themselves are still shown,
# only the lines to/from them are dropped.
EDGE_EXCLUDED_PREFIXES = (
    "opennta.common",
    "opennta.application.common",
)


def normalize_module_name(module_name: str) -> str:
    if module_name.endswith(".__init__"):
        return module_name[:-9]
    return module_name


def is_package_file(path: str) -> bool:
    return os.path.basename(path) == "__init__.py"


SKIP_DIRS = {
    "__pycache__", "site-packages",
    "env", "venv", ".venv", ".env",
    ".git", ".hg", ".svn",
    ".idea", ".vscode", ".tox", ".mypy_cache", ".pytest_cache",
    "build", "dist", ".eggs", "node_modules",
    "tests", "test",
}


def is_test_file(filename: str) -> bool:
    return filename.startswith("test_") or filename.endswith("_test.py")


def _path_segments(path: str):
    # os.sep is "/" on macOS/Linux and "\\" on Windows.
    # normpath collapses redundant separators first.
    return os.path.normpath(path).split(os.sep)


def build_path_to_module_map():
    modules = {}
    for folder, dirs, files in os.walk(BASE_DIR):
        # Prune noise dirs in-place so os.walk does not descend into them.
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        # Also skip this folder if any segment is a skip dir (defensive).
        if any(seg in SKIP_DIRS for seg in _path_segments(folder)):
            continue

        for f in files:
            if not f.endswith(".py"):
                continue

            if is_test_file(f):
                continue

            full_path = os.path.join(folder, f)
            rel_path = os.path.relpath(full_path, BASE_DIR)

            filename = os.path.basename(full_path)

            if filename == "__init__.py":
                rel_module = os.path.dirname(rel_path).replace(os.sep, ".").replace("/", ".")
            else:
                rel_module = rel_path.replace(os.sep, ".").replace("/", ".")[:-3]

            module_name = f"{ROOT_PACKAGE}.{rel_module}".rstrip(".")
            modules[full_path] = module_name

    return modules


modules = build_path_to_module_map()


CURATED_COLORS = {
    "__main__":                "#E74C3C",  # red
    "application":             "#9B59B6",  # purple
    "tracking":                "#2ECC71",  # green
    "analysis":                "#1ABC9C",  # teal      (separated from green)
    "analysis.numerical_field": "#E67E22",  # orange    (separated from analysis)
    "numerical":               "#A0522D",  # sienna/brown
    "common":                  "#3498DB",  # blue
    "tests":                   "#FF6FA3",  # pink
}

def pastel_color_for_index(index):
    hue = (index * 137.508) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.55, 0.85)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def get_group_name(module_name: str) -> str:
    """
    Group every module by the innermost package (folder) it lives in, i.e. its
    parent package with the leading "opennta." stripped:
      - opennta.__main__                                      -> __main__
      - opennta.analysis.msd_calculator                       -> analysis
      - opennta.analysis.mle.families                         -> analysis.mle
      - opennta.application.tab_analysis.dialogs.unet.dialog  -> application.tab_analysis.dialogs.unet
    """
    clean = normalize_module_name(module_name)
    parts = clean.split(".")

    # Root-package file (e.g. opennta.__main__) has no folder below the root.
    if len(parts) <= 2:
        return "__main__"

    # Drop the module's own name; the remaining path (minus the root) is the folder.
    return ".".join(parts[1:-1])


# Build dynamic group -> color map.
# Use CURATED_COLORS where available, fall back to pastel_color_for_index for the rest.
# Only real (non-__init__) modules become nodes, so derive groups from those to
# avoid creating empty clusters for packages that hold only sub-packages.
all_groups = sorted({
    get_group_name(m)
    for path, m in modules.items()
    if not is_package_file(path)
})
GROUP_COLORS = {}
_fallback_idx = 0
for _group in all_groups:
    if _group in CURATED_COLORS:
        GROUP_COLORS[_group] = CURATED_COLORS[_group]
    else:
        GROUP_COLORS[_group] = pastel_color_for_index(_fallback_idx)
        _fallback_idx += 1


def resolve_relative_import(base_module, level, module, is_package=False):
    """Resolve relative imports such as ``from .sub import X``.

    ``base_module`` is the dotted name of the importing module (without any
    ``.__init__`` suffix). The leading dots are resolved against that module's
    package (its ``__package__``):

      - For a regular module the package is its parent, so we drop the module's
        own name.
      - For a package (an ``__init__.py``) the module name *is* the package, so
        we keep every part. Pass ``is_package=True`` in that case.

    ``level`` is the number of leading dots and ``module`` the text after them
    (may be empty/None for ``from . import X``).
    """
    parts = base_module.split(".")

    # Reduce to the importing module's package (its __package__).
    if not is_package:
        parts = parts[:-1]

    # Too many leading dots to stay inside the project tree.
    if level > len(parts):
        return None

    # Each extra dot beyond the first walks one package upward.
    for _ in range(level - 1):
        if parts:
            parts.pop()

    if module:
        return normalize_module_name(".".join(parts) + "." + module)
    else:
        return normalize_module_name(".".join(parts))


normalized_module_set = {
    normalize_module_name(m)
    for path, m in modules.items()
    if not is_package_file(path)
}

lowercase_map = {
    normalize_module_name(m).lower(): normalize_module_name(m)
    for m in modules.values()
}

# Packages (folders with __init__.py) are not graph nodes themselves, but we
# need to recognise them so re-exported names can be followed (see below).
package_module_set = {
    normalize_module_name(m)
    for path, m in modules.items()
    if is_package_file(path)
}

all_module_set = normalized_module_set | package_module_set


def build_reexport_map():
    """Map names re-exported by package ``__init__.py`` files to their origin.

    When a package's ``__init__.py`` does ``from .submodule import Thing`` (the
    "folder import that hands a function through __init__" case), importing
    ``Thing`` from the *package* really depends on wherever ``Thing`` is
    defined — not on the ``__init__`` itself. This records
    ``(package, exported_name) -> origin_module`` so those edges can be
    reconnected to the real source file.

    ``origin_module`` may itself be a package that re-exports the name again
    (chained re-exports, e.g. ``application`` -> ``main_window`` package ->
    ``main_window.main_window``); :func:`resolve_reexport` follows the chain.
    """
    reexports = {}
    for path, mod in modules.items():
        if not is_package_file(path):
            continue

        package = normalize_module_name(mod)

        try:
            with open(path, encoding="utf-8-sig") as fp:
                tree = ast.parse(fp.read())
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue

            # Resolve the module the names are pulled from.
            if node.level > 0:
                base = resolve_relative_import(
                    package, node.level, node.module, is_package=True
                )
            elif node.module and node.module.startswith(ROOT_PACKAGE):
                base = normalize_module_name(node.module)
            else:
                # Imports from third-party / stdlib packages are not our concern.
                continue

            if not base or not base.startswith(ROOT_PACKAGE):
                continue

            for alias in node.names:
                if alias.name == "*":
                    continue

                # The name as seen by code importing this package.
                exported = alias.asname or alias.name

                # ``from pkg import name`` where ``pkg.name`` is itself a module
                # (e.g. ``from . import submodule``) re-exports that submodule;
                # otherwise ``name`` is a symbol defined in ``base``.
                candidate = normalize_module_name(f"{base}.{alias.name}")
                origin = candidate if candidate in all_module_set else base

                reexports[(package, exported)] = origin

    return reexports


def resolve_reexport(package, symbol, _seen=None):
    """Follow ``symbol`` re-exported by ``package`` to the module defining it.

    Returns the origin module name, or ``None`` if ``package`` does not
    re-export ``symbol``. Re-exports can be chained through nested packages, so
    this recurses while each resolved origin is itself a package that keeps
    re-exporting the same name.
    """
    if _seen is None:
        _seen = set()

    key = (package, symbol)
    if key in _seen:  # guard against import cycles
        return None
    _seen.add(key)

    origin = REEXPORTS.get(key)
    if origin is None:
        return None

    if origin in package_module_set:
        deeper = resolve_reexport(origin, symbol, _seen)
        if deeper is not None:
            return deeper

    return origin


REEXPORTS = build_reexport_map()


def build_dependency_edges():
    edges = set()

    for path, mod in modules.items():
        if is_package_file(path):
            continue

        normalized_source = normalize_module_name(mod)

        try:
            with open(path, encoding="utf-8-sig") as fp:
                tree = ast.parse(fp.read())
        except Exception:
            continue

        for node in ast.walk(tree):

            # import opennta.xxx
            if isinstance(node, ast.Import):
                for n in node.names:
                    normalized_target = normalize_module_name(n.name)
                    if normalized_target in normalized_module_set:
                        edges.add((normalized_source, normalized_target))

            # from xxx import y
            elif isinstance(node, ast.ImportFrom):

                base_module = None

                if node.level > 0:
                    resolved = resolve_relative_import(mod, node.level, node.module)
                    if resolved and resolved.startswith(ROOT_PACKAGE):
                        base_module = resolved
                else:
                    if node.module:
                        if node.module.startswith(ROOT_PACKAGE):
                            base_module = node.module
                        else:
                            # Try: from tracking import X -> opennta.tracking.X
                            base_module = f"{ROOT_PACKAGE}.{node.module}"

                if base_module:
                    normalized_base = normalize_module_name(base_module)
                    if normalized_base in normalized_module_set:
                        edges.add((normalized_source, normalized_base))

                    for alias in node.names:
                        if alias.name == "*":
                            continue

                        full_name = normalize_module_name(base_module + "." + alias.name)

                        if full_name in normalized_module_set:
                            edges.add((normalized_source, full_name))
                        else:
                            key = full_name.lower()
                            if key in lowercase_map:
                                edges.add((normalized_source, lowercase_map[key]))
                            else:
                                # Not a submodule: the name may be a symbol
                                # re-exported through the package's __init__.
                                # Reconnect the edge to where it is defined.
                                origin = resolve_reexport(
                                    normalized_base, alias.name
                                )
                                if (
                                    origin in normalized_module_set
                                    and origin != normalized_source
                                ):
                                    edges.add((normalized_source, origin))

    return edges


def is_init_module(mod_name):
    for path, mod in modules.items():
        if normalize_module_name(mod) == mod_name:
            return is_package_file(path)
    return False


def is_edge_excluded(mod_name: str) -> bool:
    """True if a module belongs to a package whose edges are omitted (see EDGE_EXCLUDED_PREFIXES)."""
    clean = normalize_module_name(mod_name)
    for prefix in EDGE_EXCLUDED_PREFIXES:
        if clean == prefix or clean.startswith(prefix + "."):
            return True
    return False


edges = build_dependency_edges()
edges = {
    (s, t)
    for (s, t) in edges
    if not is_init_module(s) and not is_init_module(t)
    and not is_edge_excluded(s) and not is_edge_excluded(t)
}


def render_dependency_graph(output="opennta_deps"):
    g = Digraph("opennta_deps", format="svg")
    g.attr(rankdir="LR")

    g.attr(
        "graph",
        fontname="Helvetica",
        bgcolor="white",
        splines="spline",
        nodesep="0.2",
        ranksep="0.4",
        concentrate="false",
    )
    g.attr(
        "node",
        fontname="Helvetica",
        fontsize="9",
        margin="0.06,0.03",
        height="0.25",
    )
    g.attr("edge", color="#9AA0A6", arrowsize="0.6", penwidth="0.7")

    with g.subgraph(name="cluster_legend") as leg:
        leg.attr(label="Legend", labelloc="t", style="rounded", color="lightgrey")
        for group, color in GROUP_COLORS.items():
            if group == "__main__":
                label = ROOT_PACKAGE
            else:
                label = f"{ROOT_PACKAGE}.{group}"

            leg.node(
                f"legend_{group.replace('.', '_')}",
                label=label,
                shape="box",
                style="rounded,filled",
                fillcolor=color,
                fontcolor="white",
            )

    for group in sorted(GROUP_COLORS):
        with g.subgraph(name=f"cluster_{group.replace('.', '_')}") as c:
            c.attr(
                label=f"{ROOT_PACKAGE}.{group}" if group != "__main__" else ROOT_PACKAGE,
                style="rounded",
                color=GROUP_COLORS[group],
                penwidth="2",
            )

            for mod in sorted(normalized_module_set):
                if get_group_name(mod) != group:
                    continue

                short = mod.replace(ROOT_PACKAGE + ".", "")
                c.node(
                    mod,
                    label=short,
                    shape="box",
                    style="rounded,filled",
                    fillcolor=GROUP_COLORS[group],
                    fontcolor="white",
                )

    for src, tgt in sorted(edges):
        g.edge(src, tgt)

    try:
        out_path = g.render(output, cleanup=True)
    except ExecutableNotFound:
        _print_graphviz_install_hint()
        raise
    print("Graph generated:", out_path)
    print(f"Modules: {len(normalized_module_set)}, Deps: {len(edges)}")


def _print_graphviz_install_hint():
    msg = ["", "Graphviz 'dot' executable was not found on PATH."]
    if sys.platform == "darwin":
        msg += [
            "  macOS:  brew install graphviz",
            "  If installed via Homebrew on Apple Silicon but still not found,",
            "  ensure /opt/homebrew/bin is on your PATH (IDEs sometimes miss it).",
        ]
    elif sys.platform.startswith("linux"):
        msg += [
            "  Debian/Ubuntu: sudo apt-get install graphviz",
            "  Fedora/RHEL:   sudo dnf install graphviz",
        ]
    elif sys.platform.startswith("win"):
        msg += [
            "  Windows: winget install Graphviz.Graphviz  (or download from graphviz.org)",
            "  Then add the Graphviz 'bin' folder to your PATH.",
        ]
    if not shutil.which("dot"):
        msg.append("  ('dot' is not on PATH in the current shell.)")
    print("\n".join(msg), file=sys.stderr)


def print_imports_of_file(file_rel_path):
    full_path = os.path.join(REPO_ROOT, file_rel_path)
    if full_path not in modules:
        print("Not a module:", file_rel_path)
        return

    module_name = modules[full_path]
    normalized_source = normalize_module_name(module_name)
    print(f"\n==== {file_rel_path} ({normalized_source}) ====")

    with open(full_path, encoding="utf-8-sig") as fp:
        tree = ast.parse(fp.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                print(f"import {n.name}")

        elif isinstance(node, ast.ImportFrom):
            dots = "." * node.level
            module = node.module or ""
            names = ", ".join(a.name for a in node.names)
            print(f"from {dots}{module} import {names}")


if __name__ == "__main__":
    print("Found modules:", len(modules))
    print("Edges:", len(edges))
    print("Groups:", len(GROUP_COLORS))

    print("\nExample: generating graph...")
    render_dependency_graph()

    print("\nExample: printing imports of files...")
    print_imports_of_file("src/opennta/__main__.py")
    print_imports_of_file("src/opennta/tracking/fitting.py")
