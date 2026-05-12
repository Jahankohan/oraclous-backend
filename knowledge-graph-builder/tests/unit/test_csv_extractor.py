"""
Unit tests for csv_extractor.py

Covers:
- 5-column / 100-row CSV: schema inference, sample_rows, row_count
- TSV auto-detection via csv.Sniffer
- Type inference: int, float, bool, date, str
- Empty values handled gracefully
"""

import csv
import os
import tempfile

import pytest

from app.services.csv_extractor import _infer_type, extract_csv

# ─── _infer_type unit tests ───────────────────────────────────────────────────


class TestInferType:
    @pytest.mark.unit
    def test_infers_int(self):
        assert _infer_type(["1", "2", "3", "100"]) == "int"

    @pytest.mark.unit
    def test_infers_float(self):
        assert _infer_type(["1.5", "2.0", "3.14"]) == "float"

    @pytest.mark.unit
    def test_infers_bool(self):
        assert _infer_type(["true", "false", "True", "False"]) == "bool"

    @pytest.mark.unit
    def test_infers_bool_yes_no(self):
        assert _infer_type(["yes", "no"]) == "bool"

    @pytest.mark.unit
    def test_infers_date(self):
        assert _infer_type(["2024-01-01", "2025-06-15"]) == "date"

    @pytest.mark.unit
    def test_infers_str(self):
        assert _infer_type(["hello", "world", "foo"]) == "str"

    @pytest.mark.unit
    def test_empty_values_return_str(self):
        assert _infer_type(["", "", ""]) == "str"

    @pytest.mark.unit
    def test_mixed_int_str_returns_str(self):
        assert _infer_type(["1", "two", "3"]) == "str"


# ─── extract_csv unit tests ───────────────────────────────────────────────────


def _write_csv(rows: list[list[str]], delimiter: str = ",") -> str:
    """Write rows to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        for row in rows:
            writer.writerow(row)
    return path


class TestExtractCSVBasic:
    @pytest.mark.unit
    def test_5_columns_100_rows(self):
        header = ["id", "name", "score", "active", "joined"]
        rows = [header]
        for i in range(100):
            rows.append(
                [
                    str(i),
                    f"user_{i}",
                    str(float(i) * 1.5),
                    "true",
                    f"2024-{(i % 12) + 1:02d}-01",
                ]
            )

        path = _write_csv(rows)
        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)

        assert result["columns"] == header
        assert result["row_count"] == 100
        assert len(result["sample_rows"]) == 5
        assert result["sample_rows"][0]["id"] == "0"

        schema = result["schema"]
        assert schema["id"] == "int"
        assert schema["name"] == "str"
        assert schema["score"] == "float"
        assert schema["active"] == "bool"
        assert schema["joined"] == "date"

    @pytest.mark.unit
    def test_sample_rows_capped_at_5(self):
        header = ["x"]
        rows = [header] + [[str(i)] for i in range(50)]
        path = _write_csv(rows)
        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)
        assert len(result["sample_rows"]) == 5

    @pytest.mark.unit
    def test_row_count_is_accurate(self):
        header = ["a", "b"]
        rows = [header] + [["1", "2"]] * 37
        path = _write_csv(rows)
        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)
        assert result["row_count"] == 37

    @pytest.mark.unit
    def test_sample_rows_are_dicts(self):
        header = ["col1", "col2"]
        rows = [header, ["hello", "world"]]
        path = _write_csv(rows)
        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)
        assert isinstance(result["sample_rows"][0], dict)
        assert result["sample_rows"][0]["col1"] == "hello"

    @pytest.mark.unit
    def test_schema_keys_match_columns(self):
        header = ["alpha", "beta", "gamma"]
        rows = [header, ["foo", "1", "3.14"]]
        path = _write_csv(rows)
        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)
        assert set(result["schema"].keys()) == set(header)


class TestExtractCSVTSV:
    @pytest.mark.unit
    def test_tsv_auto_detection(self):
        """TSV file with .csv extension should be auto-detected by Sniffer."""
        header = ["col_a", "col_b", "col_c"]
        rows = [header, ["1", "hello", "3.14"], ["2", "world", "2.71"]]

        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            for row in rows:
                writer.writerow(row)

        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)

        assert result["columns"] == header
        assert result["row_count"] == 2

    @pytest.mark.unit
    def test_explicit_tsv_extension(self):
        """File with .tsv extension is always parsed as tab-delimited."""
        header = ["x", "y"]
        rows = [header, ["10", "20"]]

        fd, path = tempfile.mkstemp(suffix=".tsv")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            for row in rows:
                writer.writerow(row)

        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)

        assert result["columns"] == header
        assert result["row_count"] == 1
        assert result["schema"]["x"] == "int"
        assert result["schema"]["y"] == "int"


class TestExtractCSVEdgeCases:
    @pytest.mark.unit
    def test_empty_values_in_column_treated_as_str(self):
        header = ["val"]
        rows = [header] + [[""], [""]] + [["hello"]]
        path = _write_csv(rows)
        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)
        assert result["schema"]["val"] == "str"

    @pytest.mark.unit
    def test_single_row(self):
        header = ["n"]
        rows = [header, ["42"]]
        path = _write_csv(rows)
        try:
            result = extract_csv(path)
        finally:
            os.unlink(path)
        assert result["row_count"] == 1
        assert result["schema"]["n"] == "int"
