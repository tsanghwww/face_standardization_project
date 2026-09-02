"""CPU-only tests for Phase3.1 backbone artifact hashing."""

from __future__ import annotations

import tempfile
from pathlib import Path

from phase3.preflight_sd15_backbone import tree_hash


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="phase31-preflight-") as temporary:
        root = Path(temporary)
        (root / "unet").mkdir()
        (root / "unet" / "config.json").write_text("{}", encoding="utf-8")
        first_hash, first_files = tree_hash(root)
        second_hash, second_files = tree_hash(root)
        assert first_hash == second_hash
        assert first_files == second_files
        (root / ".cache").mkdir()
        (root / ".cache" / "ignored").write_text("metadata", encoding="utf-8")
        assert tree_hash(root)[0] == first_hash
        (root / "unet" / "config.json").write_text('{"changed": true}', encoding="utf-8")
        assert tree_hash(root)[0] != first_hash
    print("PHASE3.1 BACKBONE PREFLIGHT TESTS PASSED")


if __name__ == "__main__":
    main()
