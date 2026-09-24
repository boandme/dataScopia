from __future__ import annotations

import csv

# The 4-byte UTF-32 BOMs must be checked before the 2-byte UTF-16 ones, since a UTF-32LE BOM
# (FF FE 00 00) starts with the same two bytes as a UTF-16LE BOM (FF FE).
_BOM_ENCODINGS: list[tuple[bytes, str]] = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
]

_DELIMITER_CANDIDATES = [",", ";", "\t", "|"]

# ZIP local-file-header / empty-archive / spanned-archive signatures, and gzip's magic number.
# A Kaggle file named "data.csv" occasionally turns out to actually be a compressed archive
# server-side; decoding its raw bytes as text produces exactly the "PK" - prefixed, box-glyph-
# riddled garbage column names this module exists to prevent.
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_GZIP_SIGNATURE = b"\x1f\x8b"


def detect_binary_archive(data: bytes) -> str | None:
    """Return "zip", "gzip", or None. Checked before any text decoding is attempted, since a
    compressed archive's bytes will otherwise "successfully" decode (via the latin-1 fallback)
    into plausible-looking but meaningless text instead of raising anywhere."""
    if data.startswith(_ZIP_SIGNATURES):
        return "zip"
    if data.startswith(_GZIP_SIGNATURE):
        return "gzip"
    return None


def _looks_like_bom_less_utf16(data: bytes) -> str | None:
    """Detect UTF-16 text saved without a byte-order mark (some export tools omit it).

    This matters because a NUL byte (0x00) is independently valid UTF-8 (it just means U+0000),
    so ASCII text encoded as UTF-16 - which is exactly "each character, then a 0x00 byte" for
    UTF-16LE - decodes as UTF-8 *without raising*, one real character glued to a NUL control
    character each time. That NUL renders as an empty box in a browser, producing precisely the
    "half-legible text riddled with boxes" pattern this exists to catch before it reaches the
    user. Genuine UTF-8/Latin text essentially never contains a literal NUL byte, so a high NUL
    fraction is a reliable signal on its own; which half of each pair is null then says LE or BE.
    """
    sample = data[:4096]
    if len(sample) < 8:
        return None
    null_fraction = sample.count(0) / len(sample)
    if null_fraction < 0.2:
        return None
    even_nulls = sum(1 for i in range(0, len(sample) - 1, 2) if sample[i] == 0)
    odd_nulls = sum(1 for i in range(1, len(sample), 2) if sample[i] == 0)
    return "utf-16-be" if even_nulls > odd_nulls else "utf-16-le"


def detect_text_encoding(data: bytes) -> str:
    """Best-effort detection of a byte sample's text encoding.

    Roughly a fifth to a third of real-world Kaggle CSVs are not UTF-8 - Windows-1252/Latin-1
    exports from Excel with accented characters, the occasional UTF-16 export - and naively
    decoding those as UTF-8 with errors="replace" silently turns every non-ASCII byte into a
    replacement character, corrupting column names and values alike into unreadable garbage
    without ever raising an error to signal the problem. This tries, in order: BOM sniffing
    (unambiguous when present), a BOM-less-UTF-16 heuristic (strict UTF-8 would otherwise accept
    this input and silently mangle it - see `_looks_like_bom_less_utf16`), strict UTF-8 (the
    common case), charset-normalizer's statistical detection, then cp1252 (the most common
    Windows/Excel export encoding), and finally latin-1 as a last resort - which can decode any
    byte sequence, so it must never be tried before the others or it would mask a real mismatch
    as a "successful" decode full of wrong characters.
    """
    for bom, encoding in _BOM_ENCODINGS:
        if data.startswith(bom):
            return encoding

    utf16_guess = _looks_like_bom_less_utf16(data)
    if utf16_guess:
        try:
            data.decode(utf16_guess)
            return utf16_guess
        except UnicodeDecodeError:
            pass

    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        import charset_normalizer

        matches = charset_normalizer.from_bytes(data[:200_000])
        best = matches.best()
        if best is not None and best.encoding:
            sample = data[:200_000]
            non_ascii_fraction = sum(1 for b in sample if b > 127) / len(sample) if sample else 0.0
            tie_count = sum(1 for m in matches if abs(m.chaos - best.chaos) < 1e-9)
            # Mostly-ASCII text with just a sprinkling of non-ASCII bytes (accented names, a
            # currency symbol, ...) is exactly the case where several single-byte Western
            # encodings come out statistically tied - often without cp1252 itself even among the
            # candidates - and which tied option .best() returns is not stable across process
            # runs (it depends on internal iteration order, which Python's per-process hash
            # randomization can shuffle). Only default to cp1252 - the overwhelmingly common
            # real-world case for THAT specific ambiguity - when the sample fits that signature;
            # a script that's mostly non-ASCII (Cyrillic, CJK, Greek, ...) has plenty of signal
            # for charset-normalizer to work with, and forcing cp1252 there would corrupt it.
            if tie_count > 1 and non_ascii_fraction < 0.15:
                data.decode("cp1252")
                return "cp1252"
            data.decode(best.encoding)
            return best.encoding
    except Exception:
        pass

    try:
        data.decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError:
        pass

    return "latin-1"


def sniff_delimiter(sample_text: str, fallback: str = ",") -> str:
    """Detect the field delimiter of a delimited-text sample.

    A ".csv" extension is not a reliable promise of a comma delimiter - many real exports
    (especially from European-locale Excel) use semicolons, and some use tabs or pipes. Reading
    those with a hardcoded comma collapses the whole file into a single, useless column whose
    "name" is the entire header line. Tries Python's csv.Sniffer first (handles quoted fields
    correctly), then falls back to a simple per-candidate frequency count over the first line if
    sniffing is inconclusive, so genuinely ambiguous input still gets a reasonable answer instead
    of raising.
    """
    head = "\n".join(sample_text.splitlines()[:50])
    try:
        dialect = csv.Sniffer().sniff(head, delimiters="".join(_DELIMITER_CANDIDATES))
        return dialect.delimiter
    except csv.Error:
        pass

    first_line = sample_text.splitlines()[0] if sample_text else ""
    counts = {d: first_line.count(d) for d in _DELIMITER_CANDIDATES}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else fallback


def decode_bytes(data: bytes, encoding: str | None = None) -> str:
    """Decode a byte sample with a known-good (detected) encoding rather than errors="replace",
    so a mismatched encoding is corrected rather than silently corrupted."""
    encoding = encoding or detect_text_encoding(data)
    try:
        return data.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return data.decode("latin-1")


def unwrap_binary_archive(data: bytes, truncated: bool) -> tuple[bytes, bool]:
    """If `data` is actually a zip/gzip archive rather than plain delimited text, extract the
    real tabular content from inside it. Returns `(data, truncated)` unchanged for anything that
    isn't a recognized archive. Raises ValueError with an actionable message when the archive
    can't be safely unwrapped - e.g. a byte-capped streamed sample that was cut off before a
    zip's trailing central directory, which makes the archive unreadable no matter how it's
    approached - so the caller can surface a clear reason instead of a confusing parse failure
    or, worse, silently misreading the raw archive bytes as garbled "text"."""
    import gzip
    import zipfile

    archive_type = detect_binary_archive(data)
    if archive_type is None:
        return data, truncated
    if truncated:
        raise ValueError(
            "This file is a compressed archive rather than plain delimited text, and the "
            'streamed sample was cut off before the archive could be safely read. Use "Run full '
            'analysis" or download the dataset to analyze it.'
        )
    if archive_type == "gzip":
        try:
            return gzip.decompress(data), False
        except OSError as exc:
            raise ValueError(f"Could not decompress this gzip-compressed file: {exc}") from exc

    import io as _io

    try:
        with zipfile.ZipFile(_io.BytesIO(data)) as zf:
            candidates = [
                name
                for name in zf.namelist()
                if not name.endswith("/") and any(name.lower().endswith(ext) for ext in (".csv", ".tsv"))
            ]
            if not candidates:
                raise ValueError(
                    "This file is a ZIP archive that doesn't contain a CSV/TSV file that can be "
                    "analyzed directly. Download the dataset to inspect its contents."
                )
            inner_name = sorted(candidates, key=lambda name: zf.getinfo(name).file_size, reverse=True)[0]
            return zf.read(inner_name), False
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Could not read this ZIP-compressed file: {exc}") from exc


def looks_like_garbled_text(text: str, sample_size: int = 4096) -> bool:
    """Last-resort sanity check: even after BOM/UTF-16/charset-normalizer/cp1252 detection, some
    file could still be genuinely binary content this module doesn't specifically recognize (an
    unsupported archive format, a corrupted download, ...). latin-1 never raises, so nothing
    upstream would otherwise catch that case - it would just silently hand the caller a dataframe
    full of nonsense column names. This flags text whose control-character density is far above
    what any real delimited text file would ever contain, so the caller can raise a clear error
    instead of presenting garbage as if it were a successfully analyzed dataset."""
    sample = text[:sample_size]
    if not sample:
        return False
    control_count = sum(1 for c in sample if ord(c) < 32 and c not in "\t\n\r")
    return (control_count / len(sample)) > 0.05
