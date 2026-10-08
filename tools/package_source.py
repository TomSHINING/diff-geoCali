"""Create clean GitHub source archives without binaries, data or build caches."""

import argparse
import hashlib
from pathlib import Path
import tarfile
import zipfile


def source_files(root):
    root_names = {
        ".gitignore", "README.md", "setup.py", "pyproject.toml",
        "MANIFEST.in", "requirements-build.txt", "source_only_forward_projector.py",
    }
    files = [root / name for name in sorted(root_names)]
    suffixes = {".py", ".cu", ".md", ".json"}
    for directory in ("csrc", "cbct_backprojector", "examples", "docs", "tools"):
        files.extend(
            path for path in (root / directory).rglob("*")
            if path.is_file() and path.suffix in suffixes
            and "__pycache__" not in path.parts
        )
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("A required source file is missing")
    if not (root / "docs/source_manifest.json").is_file():
        raise FileNotFoundError("docs/source_manifest.json must exist before packaging")
    return sorted(set(files))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root.parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = source_files(root)
    stem = "cbct-projectors-cuda-source"
    zip_path = args.output_dir / f"{stem}.zip"
    tar_path = args.output_dir / f"{stem}.tar.gz"
    expected = {f"{root.name}/{path.relative_to(root).as_posix()}": path.read_bytes() for path in files}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in expected.items():
            archive.writestr(name, data)
    with tarfile.open(tar_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"{root.name}/{path.relative_to(root).as_posix()}")

    with zipfile.ZipFile(zip_path) as archive:
        assert set(archive.namelist()) == set(expected)
        assert archive.testzip() is None
        assert all(archive.read(name) == data for name, data in expected.items())
    with tarfile.open(tar_path, "r:gz") as archive:
        assert set(archive.getnames()) == set(expected)
        assert all(archive.extractfile(name).read() == data for name, data in expected.items())

    hashes = []
    for path in (zip_path, tar_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {path.name}\n")
        print(f"Verified {path}: {len(files)} source files, {path.stat().st_size} bytes")
    (args.output_dir / f"{stem}.sha256").write_text("".join(hashes), encoding="utf-8")


if __name__ == "__main__":
    main()
