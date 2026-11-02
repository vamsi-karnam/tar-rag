"""Tesseract OCR for pages with no text layer.

``StructuredPdfExtractor`` finds pages with no text and passes each one here
as a PNG image, and the text comes back into the pipeline. It works on any
callable with the same shape, so a caller can supply another engine with
``ocr=``.

The English model ships in ``tar_rag/data/tessdata``. See
``THIRD_PARTY_NOTICES`` for its licence.
"""

from __future__ import annotations

import re
from importlib.resources import files
from io import BytesIO
from typing import Any

from ..errors import MissingExtractorDependency

# A hex prefix read as the letter O. The body must be a real hex value, so a
# word like "Oxford" or "oxide" is left alone.
_MISREAD_HEX_RE = re.compile(
    r"\b[Oo][xX](?=(?:[0-9A-Fa-f_]*\d[0-9A-Fa-f_]*|[0-9A-F_]+)\b|-(?![A-Za-z]))"
)


def repair_hex_prefix(text: str) -> str:
    """Restore ``0x`` where OCR read the zero as a letter O."""
    return _MISREAD_HEX_RE.sub("0x", text)


def bundled_tessdata_dir() -> str:
    """Return the folder holding the shipped language data, with a trailing slash."""
    return str(files("tar_rag") / "data" / "tessdata").rstrip("/\\") + "/"


class TesseractOcr:
    """Read a page image with Tesseract, through ``tesserocr``.

    Parameters
    ----------
    lang:
        Tesseract language code. Only ``eng`` ships with tar-rag. Other
        languages need ``tessdata_dir``.
    tessdata_dir:
        Folder holding ``<lang>.traineddata``. Defaults to the shipped data.
    """

    def __init__(self, *, lang: str = "eng", tessdata_dir: str | None = None) -> None:
        self.lang = lang
        self.tessdata_dir = tessdata_dir
        self._api: Any = None

    def __getstate__(self) -> dict[str, Any]:
        # The engine handle cannot cross a process boundary. Each worker opens its own.
        return {"lang": self.lang, "tessdata_dir": self.tessdata_dir, "_api": None}

    def _engine(self) -> Any:
        if self._api is None:
            try:
                import tesserocr
            except ImportError as exc:
                raise MissingExtractorDependency(
                    f"TesseractOcr could not load 'tesserocr': {exc}. "
                    "See the README for platforms with no wheel."
                ) from exc
            path = self.tessdata_dir or bundled_tessdata_dir()
            try:
                self._api = tesserocr.PyTessBaseAPI(path=path, lang=self.lang, psm=tesserocr.PSM.AUTO)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Tesseract could not start with data folder {path!r} and language "
                    f"{self.lang!r}: {exc}. Pass tessdata_dir with a plain ASCII path."
                ) from exc
        return self._api

    def __call__(self, png: bytes, page_number: int) -> str:
        from PIL import Image

        engine = self._engine()
        with Image.open(BytesIO(png)) as image:
            engine.SetImage(image.convert("L"))
            return repair_hex_prefix(str(engine.GetUTF8Text()))
