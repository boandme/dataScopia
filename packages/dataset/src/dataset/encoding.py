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


def detect_text_encoding(data: bytes) -> str:
    """Best-effort detection of a byte sample's text encoding.

    Roughly a fifth to a third of real-world Kaggle CSVs are not UTF-8 - Windows-1252/Latin-1
    exports from Excel with accented characters, the occasional UTF-16 export - and naively
    decoding those as UTF-8 with errors="replace" silently turns every non-ASCII byte into a
    replacement character, corrupting column names and values alike into unreadable garbage
    without ever raising an error to signal the problem. This tries, in order: BOM sniffing
    (unambiguous when present), strict UTF-8 (the common case and the only fallback here that
    can actually fail), charset-normalizer's statistical detection, then cp1252 (the most common
    Windows/Excel export encoding), and finally latin-1 as a last resort - which can decode any
    byte sequence, so it must never be tried before the others or it would mask a real mismatch
    as a "successful" decode full of wrong characters.
    """
    for bom, encoding in _BOM_ENCODINGS:
        if data.startswith(bom):
            return encoding

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
