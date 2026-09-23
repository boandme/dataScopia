from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.encoding import decode_bytes, detect_text_encoding, sniff_delimiter  # noqa: E402
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
