"""
Paquete core de Arandú: Métodos Computacionales (v3.0).
Lógica matemática pura desacoplada de la interfaz gráfica.
Carga diferida por submódulo (PEP 562) para evitar instanciar SymPy o NumPy en el arranque.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = [
    "catalogo_ejercicios",
    "edo_pvi",
    "fundamentos_errores",
    "integracion",
    "integracion_derivacion",
    "interpolacion",
    "raices",
    "sistemas_autovalores",
    "version",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return import_module(f"core.{name}")
    raise AttributeError(f"module 'core' has no attribute {name!r}")
