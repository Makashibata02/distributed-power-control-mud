"""Fail closed when a release candidate contains privacy or packaging hazards."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image


TEXT_EXTENSIONS = {
    ".cff",
    ".csv",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".svg",
}
FORBIDDEN_EXTENSIONS = {".docx", ".pptx", ".pdf", ".zip", ".rar", ".7z"}
MAX_FILE_BYTES = 10 * 1024 * 1024
PATTERNS = {
    "windows_absolute_path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]"),
    "home_directory": re.compile(r"/(?:Users|home)/[^/\s]+/", re.IGNORECASE),
    "email_address": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
    "phone_or_student_number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{11}(?!\d)"),
    "credential_assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"][^'\"]+"
    ),
}
SKIP_CONTENT_SCAN = {"PRIVACY_AUDIT.md", "RELEASE_MANIFEST.csv", "audit_release.py"}


def relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def audit(root: Path) -> tuple[list[str], list[str], int, int]:
    findings: list[str] = []
    warnings: list[str] = []
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not relative(root, path).startswith("results/generated/")
        and not relative(root, path).startswith(".git/")
    ]
    total_bytes = sum(path.stat().st_size for path in files)
    for path in files:
        rel = relative(root, path)
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_EXTENSIONS:
            findings.append(f"forbidden file type: {rel}")
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.append(f"file exceeds 10 MiB: {rel}")
        if path.name in SKIP_CONTENT_SCAN:
            continue
        if suffix in TEXT_EXTENSIONS or path.name == "LICENSE":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                findings.append(f"text file is not UTF-8: {rel}")
                continue
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{label}: {rel}")
            if suffix == ".svg" and re.search(r"<(?:metadata|dc:creator|dc:date)\b", text):
                findings.append(f"SVG metadata remains: {rel}")
        if suffix == ".png":
            with Image.open(path) as image:
                sensitive_keys = {key for key in image.info if key.lower() in {"author", "creator", "description", "software", "exif", "xml:com.adobe.xmp"}}
            if sensitive_keys:
                findings.append(f"PNG metadata remains ({', '.join(sorted(sensitive_keys))}): {rel}")
    citation = root / "CITATION.cff"
    if citation.exists() and "github.com/OWNER/" in citation.read_text(encoding="utf-8"):
        warnings.append("CITATION.cff contains the expected pre-upload OWNER placeholder.")
    return sorted(set(findings)), warnings, len(files), total_bytes


def write_report(path: Path, findings: list[str], warnings: list[str], file_count: int, total_bytes: int) -> None:
    lines = [
        "# Privacy and Release Audit",
        "",
        f"- Files scanned: {file_count}",
        f"- Candidate size: {total_bytes / 1024 / 1024:.2f} MiB",
        f"- Blocking findings: {len(findings)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Blocking findings",
        "",
    ]
    lines.extend([f"- {item}" for item in findings] or ["- None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines.extend(
        [
            "",
            "## Checks performed",
            "",
            "- Forbidden office, archive, and PDF file types",
            "- Files larger than 10 MiB",
            "- Absolute user or home-directory paths",
            "- Email addresses and 11-digit personal identifiers",
            "- Common credential assignments",
            "- SVG author/date metadata and sensitive PNG metadata",
            "",
            "The author name and Fudan University affiliation are intentionally retained under the approved attribution policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=Path("PRIVACY_AUDIT.md"))
    args = parser.parse_args()
    root = args.root.resolve()
    report = args.report if args.report.is_absolute() else root / args.report
    findings, warnings, file_count, total_bytes = audit(root)
    write_report(report, findings, warnings, file_count, total_bytes)
    if findings:
        raise SystemExit(f"Release audit failed with {len(findings)} blocking finding(s).")
    print(f"Release audit passed: {file_count} files, {total_bytes / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
