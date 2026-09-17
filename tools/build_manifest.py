"""Build the review manifest for every release-candidate file."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def classify(path: Path) -> tuple[str, str]:
    parts = path.parts
    if parts[:2] == ("results", "published"):
        if path.suffix.lower() in {".png", ".svg"}:
            return "published figure", "curated project result"
        return "published data", "curated generated result"
    if parts and parts[0] == "src":
        return "source code", "final code mainline"
    if parts and parts[0] == "configs":
        return "configuration", "final experiment configuration"
    if parts and parts[0] == "tests":
        return "test", "release validation"
    if parts and parts[0] == "tools":
        return "release tool", "release preparation"
    if parts and parts[0] == ".github":
        return "automation", "release validation"
    if path.name in {"LICENSE", "CITATION.cff"}:
        return "legal/citation", "release documentation"
    return "documentation", "release documentation"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("RELEASE_MANIFEST.csv"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        rel_path = path.relative_to(root)
        if path.resolve() == output.resolve():
            continue
        usage, source = classify(rel_path)
        rows.append(
            {
                "path": rel_path.as_posix(),
                "type": path.suffix.lower().lstrip(".") or "file",
                "size_bytes": path.stat().st_size,
                "purpose": usage,
                "source_category": source,
                "sha256": digest(path),
            }
        )
    rows.append(
        {
            "path": output.relative_to(root).as_posix(),
            "type": "csv",
            "size_bytes": "",
            "purpose": "self-describing manifest",
            "source_category": "generated review material",
            "sha256": "SELF",
        }
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} manifest rows to {output}")


if __name__ == "__main__":
    main()
