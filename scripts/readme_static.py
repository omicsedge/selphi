"""Write docs/README_static.md: the README with still images in place of its animations.

Each animation becomes one still per step (its caption below it); the terminal demo becomes the logs it replays.
Run from the repository root after the README or its animations change:
    python3 scripts/readme_static.py [path/to/chrome-headless-shell]
"""
import ast
import glob
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "static")
OUT_MD = os.path.join(ROOT, "docs", "README_static.md")


def chrome() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    found = glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell"))
    if not found:
        sys.exit("give the path of chrome-headless-shell (or chromium) as the first argument")
    return sorted(found)[-1]


def constants(script: str) -> dict:
    """DUR, STEPS and CAPS of an animation script, without running it."""
    tree = ast.parse(open(os.path.join(ROOT, "scripts", script)).read())
    names: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, (ast.Name, ast.Tuple)) for t in node.targets):
            text = ast.unparse(node)
            if re.match(r"^(DUR|S\d|SEND|STEPS|CAPS)\b", text):
                exec(text, {}, names)
    return names


def stills(svg_name: str, script: str) -> list[tuple[str, str, str]]:
    """One PNG per step, taken at 80% of the step, once its motion has settled: (file, step name, caption)."""
    c = constants(script)
    svg = open(os.path.join(ROOT, "img", svg_name)).read()
    width, height = (int(v) for v in re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).groups())
    base = svg_name.removesuffix("_light.svg")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, ((step, start, end), cap) in enumerate(zip(c["STEPS"], c["CAPS"]), 1):
            t = (start + 0.8 * (end - start)) * c["DUR"]
            page = os.path.join(tmp, "frame.html")
            with open(page, "w") as f:
                f.write(f"<!doctype html><html><body style='margin:0'>{svg}<script>"
                        f"const s=document.querySelector('svg');s.pauseAnimations();s.setCurrentTime({t:.3f});"
                        "</script></body></html>")
            png = f"{base}_{i}.png"
            subprocess.run([chrome(), "--headless", "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1.5",
                            f"--window-size={width},{height}", f"--screenshot={os.path.join(OUT_DIR, png)}",
                            f"file://{page}"], check=True, capture_output=True)
            out.append((png, step, cap))
    return out


def still_block(frames: list[tuple[str, str, str]]) -> str:
    parts = []
    for i, (png, step, cap) in enumerate(frames, 1):
        parts.append(f"**{i}. {step}.** {cap}\n\n<img src=\"static/{png}\" alt=\"{step}\" width=\"900\"/>")
    return "\n\n".join(parts)


def demo_block() -> str:
    parts = []
    for run in ("chip", "wgs", "lowcov"):
        folder = os.path.join(ROOT, "img", "demo_logs", run)
        cmd = open(os.path.join(folder, "cmd.txt")).read().strip()
        log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", open(os.path.join(folder, "ansi.txt")).read()).rstrip()
        parts.append(f"```text\n$ {cmd}\n{log}\n```")
    return "\n\n".join(parts)


def picture(name: str) -> re.Pattern:
    return re.compile(r'<p align="center">\s*<picture>(?:(?!</picture>).)*?img/' + name
                      + r'_light\.svg(?:(?!</picture>).)*?</picture>\s*</p>', re.S)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    md = open(os.path.join(ROOT, "README.md")).read()
    for name, script in (("how", "readme_explainer.py"), ("lowcov", "readme_lowcov.py")):
        block = still_block(stills(f"{name}_light.svg", script))
        md, n = picture(name).subn(lambda _: block, md)
        assert n == 1, f"{name}: {n} animations found in README.md"
    md, n = picture("demo").subn(lambda _: demo_block(), md)
    assert n == 1, f"demo: {n} animations found in README.md"
    # the page lives in docs/: paths into the repository go one level up
    md = re.sub(r'(src|srcset)="img/', r'\1="../img/', md)
    md = re.sub(r"\]\((?!https?:|#|mailto:)", "](../", md)
    note = "*The [README](../README.md) with still images instead of animations.*\n\n"
    md = re.sub(r"(</p>\n\n)", lambda m: m.group(1) + note, md, count=1)
    open(OUT_MD, "w").write(md)
    print(f"wrote {OUT_MD} and {len(os.listdir(OUT_DIR))} images in {OUT_DIR}")


if __name__ == "__main__":
    main()
