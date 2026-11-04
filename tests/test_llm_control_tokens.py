"""Token kontrol / markup tool-call mentah tidak boleh sampai ke user.

Terukur di :8890: satu jawaban diawali
`<|inc_tool_call_ml|_c_calls> <|inc_tool_call_ml|invoke name="...">` karena
model gagal memformat tool-call dan membocorkan penanda protokolnya. Teks
seperti itu bukan jawaban dan membuat aplikasi terasa rusak.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import llm  # noqa: E402


LEAKS = [
    '<|inc_tool_call_ml|_c_calls> <|inc_tool_call_ml|invoke name="x">Sisa cuti Maya tidak tersedia.',
    '<|endoftext|>Halo, ada yang bisa saya bantu?',
    'Jawaban <|im_start|>assistant<|im_end|> selesai.',
    '<tool_call>{"name":"get_team"}</tool_call>Data tim Anda kosong.',
]


def test_control_tokens_are_stripped():
    for raw in LEAKS:
        cleaned = llm._clean_text(raw)
        assert "<|" not in cleaned and "|>" not in cleaned, cleaned
        assert "tool_call" not in cleaned.lower(), cleaned
        assert cleaned, "isi jawaban tidak boleh ikut terhapus"


def test_normal_text_is_untouched():
    for raw in ("Jawaban normal tanpa token.",
                "Harga < 100 dan > 10 tetap aman.",
                "Gunakan format **tebal** dan `kode`."):
        assert llm._clean_text(raw) == raw


def test_cleaning_leaves_real_content():
    raw = '<|inc_tool_call_ml|invoke name="pemeriksaan data"> Sisa cuti Maya tidak tersedia.'
    assert llm._clean_text(raw) == "Sisa cuti Maya tidak tersedia."


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
