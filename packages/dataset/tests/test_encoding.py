from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gzip
import io
import os
import zipfile

import pytest

from dataset.encoding import (  # noqa: E402
    decode_bytes,
    detect_binary_archive,
    detect_text_encoding,
    looks_like_garbled_text,
    sniff_delimiter,
    unwrap_binary_archive,
)
from dataset.loader import _read_delimited  # noqa: E402
from dataset.streaming import dataframe_from_delimited_sample  # noqa: E402


class TestDetectTextEncoding:
    def test_plain_utf8_detected_as_utf8(self):
        assert detect_text_encoding(b"a,b,c\n1,2,3\n") == "utf-8"

    def test_utf8_bom_detected(self):
        data = b"\xef\xbb\xbfa,b,c\n1,2,3\n"
        assert detect_text_encoding(data) == "utf-8-sig"

    def test_utf16_le_bom_detected(self):
        data = "a,b,c\n1,2,3\n".encode("utf-16")
        assert detect_text_encoding(data) == "utf-16"

    def test_cp1252_accented_text_decodes_without_replacement_chars(self):
        # A realistic-size cp1252 sample (accented French text), the classic case that silently
        # corrupts into replacement characters under a hardcoded utf-8 + errors="replace" decode.
        e_acute, e_grave, a_grave = chr(0xE9), chr(0xE8), chr(0xE0)
        header = f"id,ville,plat,commentaire\n".encode("cp1252")
        line = f"{{i}},Paris,Caf{e_acute} Jos{e_acute},Tr{e_grave}s bon accueil {a_grave} Paris\n"
        data = header + b"".join(line.format(i=i).encode("cp1252") for i in range(600))
        decoded = decode_bytes(data)
        assert "�" not in decoded
        assert e_acute in decoded
        assert a_grave in decoded

    def test_small_sample_with_sparse_accents_still_prefers_cp1252(self):
        # Regression: with only one or two non-ASCII bytes in a small sample, charset-normalizer
        # reports several near-identical single-byte encodings (cp1250, mac_latin2, cp775, ...)
        # as statistically tied and may pick a sibling other than cp1252, which is overwhelmingly
        # the real-world default for this ambiguity class (Windows/Excel CSV exports).
        e_acute = chr(0xE9)
        header = f"nom;ville;prix;quantit{e_acute}\n".encode("cp1252")
        rows = b"".join(f"Produit{i};Paris;{i}.5;{i}\n".encode("cp1252") for i in range(50))
        data = header + rows
        assert detect_text_encoding(data) == "cp1252"
        decoded = decode_bytes(data)
        assert "�" not in decoded

    def test_predominantly_non_ascii_script_is_not_forced_to_cp1252(self):
        # The cp1252 tie-break default must be scoped to mostly-ASCII text with sparse accents -
        # a script that's mostly non-ASCII (Cyrillic here) has plenty of signal for real
        # detection, and forcing cp1252 on it would corrupt every character.
        words = ["Привет", "Мир", "Компания", "Заказ", "Статус"]
        rows = ["id,name,note"]
        for i in range(300):
            w1, w2 = words[i % len(words)], words[(i + 1) % len(words)]
            rows.append(f"{i},{w1} {w2},{w2} {w1}")
        text = "\n".join(rows) + "\n"
        data = text.encode("windows-1251")
        assert detect_text_encoding(data) != "cp1252"
        assert decode_bytes(data) == text


class TestBomLessUtf16:
    """A NUL byte is independently valid UTF-8 (U+0000), so ASCII text saved as UTF-16 without a
    byte-order mark decodes as UTF-8 without ever raising - one real character glued to a NUL
    control character each time, which renders as a box in a browser between every letter. This
    is the exact "half-legible text riddled with boxes" pattern reported as a live bug."""

    def test_utf16_le_without_bom_detected_and_decoded_cleanly(self):
        text = "school_name,student_id\nLincoln High,1\nOakwood,2\n"
        data = text.encode("utf-16-le")
        assert detect_text_encoding(data) in ("utf-16-le",)
        assert decode_bytes(data) == text

    def test_utf16_be_without_bom_detected_and_decoded_cleanly(self):
        text = "school_name,student_id\nLincoln High,1\nOakwood,2\n"
        data = text.encode("utf-16-be")
        assert detect_text_encoding(data) in ("utf-16-be",)
        assert decode_bytes(data) == text

    def test_full_pipeline_recovers_real_column_names_from_bom_less_utf16(self):
        text = "school_name,student_id\nLincoln High,1\nOakwood,2\n"
        data = text.encode("utf-16-le")
        df = dataframe_from_delimited_sample(data, truncated=False, sep=",")
        assert list(df.columns) == ["school_name", "student_id"]

    def test_plain_ascii_csv_not_misdetected_as_utf16(self):
        # A normal, dense CSV should never contain enough NUL bytes to trip the UTF-16 heuristic.
        data = ("a,b,c\n" + "\n".join(f"{i},{i*2},{i*3}" for i in range(500))).encode("utf-8")
        assert detect_text_encoding(data) == "utf-8"


class TestBinaryArchiveUnwrapping:
    """Some Kaggle files named "data.csv" turn out to actually be zip/gzip-compressed archives;
    naively decoding the raw compressed bytes as text produces exactly the "PK"-prefixed,
    box-glyph-riddled garbage column names reported as a live bug."""

    def _make_zip(self, inner_name: str, inner_bytes: bytes) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(inner_name, inner_bytes)
        return buf.getvalue()

    def test_detects_zip_signature(self):
        zip_bytes = self._make_zip("data.csv", b"a,b\n1,2\n")
        assert detect_binary_archive(zip_bytes) == "zip"

    def test_detects_gzip_signature(self):
        gz_bytes = gzip.compress(b"a,b\n1,2\n")
        assert detect_binary_archive(gz_bytes) == "gzip"

    def test_plain_csv_not_flagged_as_archive(self):
        assert detect_binary_archive(b"a,b,c\n1,2,3\n") is None

    def test_zip_wrapped_csv_extracted_and_parsed_correctly(self):
        zip_bytes = self._make_zip("data.csv", b"id,name\n1,Alice\n2,Bob\n")
        df = dataframe_from_delimited_sample(zip_bytes, truncated=False, sep=",")
        assert list(df.columns) == ["id", "name"]
        assert df.shape == (2, 2)

    def test_gzip_wrapped_csv_extracted_and_parsed_correctly(self):
        gz_bytes = gzip.compress(b"id,name\n1,Alice\n2,Bob\n")
        df = dataframe_from_delimited_sample(gz_bytes, truncated=False, sep=",")
        assert list(df.columns) == ["id", "name"]

    def test_truncated_zip_raises_clear_error_instead_of_garbage(self):
        zip_bytes = self._make_zip("data.csv", b"id,name\n" + b"1,Alice\n" * 500)
        with pytest.raises(ValueError, match="compressed archive"):
            unwrap_binary_archive(zip_bytes[:80], truncated=True)

    def test_zip_with_no_tabular_file_raises_clear_error(self):
        zip_bytes = self._make_zip("readme.txt", b"just some notes")
        with pytest.raises(ValueError, match="doesn't contain a CSV/TSV"):
            unwrap_binary_archive(zip_bytes, truncated=False)

    def test_loader_path_also_unwraps_zip(self):
        zip_bytes = self._make_zip("data.csv", b"id,name\n1,Alice\n")
        df = _read_delimited(zip_bytes, default_sep=",")
        assert list(df.columns) == ["id", "name"]


class TestLooksLikeGarbledText:
    def test_normal_delimited_text_not_flagged(self):
        text = "a\tb\tc\n" + "\n".join(f"{i}\tfoo\tbar" for i in range(200))
        assert not looks_like_garbled_text(text)

    def test_random_binary_content_is_flagged(self):
        # latin-1 can decode any byte sequence "successfully", so random binary content is the
        # case this check exists to catch after every real encoding has been ruled out.
        garbage = os.urandom(2000).decode("latin-1")
        assert looks_like_garbled_text(garbage)

    def test_random_binary_pipeline_raises_instead_of_returning_garbage(self):
        with pytest.raises(ValueError, match="doesn't look like readable delimited text"):
            dataframe_from_delimited_sample(os.urandom(2000), truncated=False, sep=",")


class TestSniffDelimiter:
    def test_comma_detected(self):
        assert sniff_delimiter("a,b,c\n1,2,3\n") == ","

    def test_semicolon_detected(self):
        assert sniff_delimiter("a;b;c\n1;2;3\n") == ";"

    def test_tab_not_misdetected_as_comma_when_fields_contain_commas(self):
        # A tab-delimited file whose text fields happen to contain commas must not be sniffed
        # as comma-delimited, or every row collapses into far too many ragged columns.
        text = "name\tnotes\nAlice\tworks in NY, sometimes remote\nBob\tbased in SF, CA\n"
        assert sniff_delimiter(text, fallback="\t") in ("\t",)


class TestFullPipeline:
    def test_semicolon_cp1252_csv_yields_real_column_names_not_one_garbled_column(self):
        i_circ, e_acute = chr(0xEE), chr(0xE9)
        header = f"identifiant;ville;prix_unitaire;quant{i_circ}t{e_acute};r{e_acute}gion\n".encode("cp1252")
        rows = b"".join(f"{i};Paris;{i}.5;{i};{i_circ}le-de-France\n".encode("cp1252") for i in range(300))
        data = header + rows

        df = dataframe_from_delimited_sample(data, truncated=False, sep=",")
        assert list(df.columns) == ["identifiant", "ville", "prix_unitaire", f"quant{i_circ}t{e_acute}", f"r{e_acute}gion"]
        assert df.shape == (300, 5)

        df_upload = _read_delimited(data, default_sep=",")
        assert list(df_upload.columns) == list(df.columns)

    def test_plain_utf8_comma_csv_unaffected(self):
        data = b"a,b,c\n1,2,3\n4,5,6\n"
        df = dataframe_from_delimited_sample(data, truncated=False, sep=",")
        assert list(df.columns) == ["a", "b", "c"]
        assert df.shape == (2, 3)

    def test_tsv_with_embedded_commas_in_fields_stays_two_columns(self):
        data = b"name\tnotes\nAlice\tworks in NY, sometimes remote\nBob\tbased in SF, CA\n"
        df = dataframe_from_delimited_sample(data, truncated=False, sep="\t")
        assert list(df.columns) == ["name", "notes"]
        assert df.shape == (2, 2)
