"""
core/version.py
===============
Metadatos de versión y detector de entorno de ejecución para Arandú (Arquitectura Dual).
"""

from __future__ import annotations

import sys

APP_VERSION = "3.2.0"
OFFLINE_PWA_URL = "https://arandu-metodos-offline.vercel.app"


def es_entorno_offline_wasm() -> bool:
    """Detecta si Python se ejecuta dentro del navegador vía WebAssembly (Pyodide/stlite)."""
    return (
        sys.platform == "emscripten"
        or "pyodide" in sys.modules
        or "stlite" in sys.modules
    )

