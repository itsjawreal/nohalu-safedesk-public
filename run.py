"""Convenience runner: serve UI + API from one process.

Usage (from project root):
    python run.py
Then open http://localhost:8000
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import uvicorn  # noqa: E402
import logging_setup  # noqa: E402

if __name__ == "__main__":
    # Uvicorn menulis log-nya lewat logging; arahkan ke formatter JSON yang
    # sama dengan aplikasi. Tanpa ini, stream log berisi campuran baris JSON
    # dan teks bebas ("INFO:     Started server process ..."), sehingga
    # agregator yang mem-parse per baris gagal pada baris uvicorn.
    logging_setup.setup()
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_config=None,          # jangan pasang handler teks bawaan
        log_level=os.environ.get("NOHALU_LOG_LEVEL", "info"),
    )
