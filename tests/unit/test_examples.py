"""The upload example against a stand-in client. All data is synthetic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "upload_units_openai.py"


class FakeOpenAI:
    instances: ClassVar[list[FakeOpenAI]] = []

    def __init__(self) -> None:
        self.attached: list[dict[str, Any]] = []
        FakeOpenAI.instances.append(self)
        self.files = SimpleNamespace(create=lambda file, purpose: SimpleNamespace(id=f"file-{len(file.read())}"))
        self.vector_stores = SimpleNamespace(
            create=lambda name: SimpleNamespace(id="vs-1"),
            files=SimpleNamespace(create=lambda **kwargs: self.attached.append(kwargs)),
        )


def _run(output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("openai")
    fake.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    FakeOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setattr(sys, "argv", ["upload", "--output-dir", str(output)])
    spec = importlib.util.spec_from_file_location("upload_units_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def _manifest(output: Path, rows: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "unit.md").write_text("text", encoding="utf-8")
    (output / "unit_manifest.json").write_text(json.dumps({"version": "v1", "units": rows}), encoding="utf-8")


def test_each_unit_is_uploaded_with_its_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = {"doc_id": "d", "unit_id": "d:1#1", "page_start": 1, "section_1": "1"}
    _manifest(tmp_path, [{"unit_id": "d:1#1", "path": "unit.md", "metadata": metadata}])
    _run(tmp_path, monkeypatch)
    attached = FakeOpenAI.instances[0].attached
    assert len(attached) == 1
    assert attached[0]["attributes"] == metadata
    assert attached[0]["chunking_strategy"]["static"]["chunk_overlap_tokens"] == 0


def test_too_many_attributes_stop_before_any_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = {f"key_{n}": "value" for n in range(17)}
    _manifest(tmp_path, [{"unit_id": "u", "path": "unit.md", "metadata": metadata}])
    with pytest.raises(SystemExit, match="16"):
        _run(tmp_path, monkeypatch)
    assert FakeOpenAI.instances == []


def test_path_outside_the_output_folder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    _manifest(tmp_path / "out", [{"unit_id": "u", "path": "../secret.md", "metadata": {}}])
    with pytest.raises(SystemExit, match="outside"):
        _run(tmp_path / "out", monkeypatch)
    assert FakeOpenAI.instances[0].attached == []
