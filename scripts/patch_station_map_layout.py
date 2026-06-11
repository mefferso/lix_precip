#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "scripts" / "build_lix_obs_maps.py",
    ROOT / "scripts" / "build_lix_obs_archive_maps.py",
]

REPLACEMENTS = {
    # Left column: locator inset directly above legend. Main map starts exactly
    # where the left column ends, and all three share the same total vertical span.
    'fig.add_axes([0.03, 0.05, 0.16, 0.52])':
        'fig.add_axes([0.01, 0.05, 0.17, 0.59])',
    'fig.add_axes([0.03, 0.60, 0.16, 0.22])':
        'fig.add_axes([0.01, 0.64, 0.17, 0.24])',

    # Main full-area map: pull down to the lower border and expand left/right.
    'fig.add_axes([0.06, 0.08, 0.88, 0.80] if is_regional else [0.22, 0.05, 0.75, 0.82])':
        'fig.add_axes([0.06, 0.08, 0.88, 0.80] if is_regional else [0.18, 0.05, 0.80, 0.83])',

    # Remove the small data-source/footer line printed under the main map.
    '    ax.text(0.02, -0.015, config["desc"], transform=ax.transAxes, fontsize=10, ha="left", va="top")\n':
        '',
}


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    if text == original:
        print(f"No layout patch changes needed for {path.relative_to(ROOT)}")
        return False

    path.write_text(text, encoding="utf-8")
    print(f"Patched station map layout in {path.relative_to(ROOT)}")
    return True


def main() -> None:
    changed = False
    for target in TARGETS:
        if not target.exists():
            raise FileNotFoundError(target)
        changed = patch_file(target) or changed

    if not changed:
        print("Station map layout patch was already applied.")


if __name__ == "__main__":
    main()
