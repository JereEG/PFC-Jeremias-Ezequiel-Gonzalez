"""
core/integracion_derivacion.py
==============================
Módulo de cálculo puro para Diferenciación e Integración Numérica Adaptativa
(Arandú: Métodos Computacionales v3.0).

Incluye:
- Diferenciación Numérica por Newton-Gregory Ascendente y Descendente (1° y 2° orden, en u arbitrario).
- Regla del Trapecio Compuesta (notación de cátedra E, P, I) y cota O(h²).
- Reglas de Simpson 1/3, Simpson 3/8 y Simpson Combinado (3/8 + 1/3).
- MOTOR ADAPTATIVO EN VIVO (`integrar_adaptativo_en_vivo`):
  Detecta cambios en Δh, paridad de subintervalos n o edición manual de nodos x_i que
  generen tramos de distinto paso, emite avisos pedagógicos claros y aplica automáticamente
  la regla combinada correspondiente (Simpson 3/8 + Simpson 1/3 o Simpson 1/3 + Trapecio),
  generando las curvas exactas por sub-bloque (lineales, parabólicas y cúbicas) para Plotly.

Diseño: Funciones 100% puras, sin `input()` ni `print()`.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from core._cache_utils import (
    compilar_lambdify_1d,
    evaluar_fn_vectorizado,
    integral_sympy_cached,
)
from core.catalogo_ejercicios import (
    GAUSS_16_PUNTOS as _GAUSS_16,
    INTEGRALES_PRECOMPUTADAS,
    integrar_gauss_legendre_16,
)


COLORES_REGLAS = {
    "Trapecio": "#C86D44",      # Terracota Cálido
    "Simpson 1/3": "#7A9A65",   # Verde Salvia
    "Simpson 3/8": "#D4A359",   # Oro Arandú
}


def validar_equiespaciado(x_values: list[float] | np.ndarray) -> float:
    x = np.array(x_values, dtype=float)
    if len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos.")
    h_arr = np.diff(x)
    h = float(h_arr[0])
    if abs(h) < 1e-14:
        raise ZeroDivisionError("El espaciamiento h entre puntos es cero.")
    if not np.allclose(h_arr, h, rtol=1e-4, atol=1e-6):
        raise ValueError("Los puntos x deben estar equiespaciados.")
    return h


# ==============================================================================
# 1. DIFERENCIACIÓN NUMÉRICA (NEWTON-GREGORY ASCENDENTE Y DESCENDENTE)
# ==============================================================================

_POLY_PARAM_NG: dict[str, dict[int, str]] = {
    "u": {
        1: "u",
        2: "(u² - u) / 2",
        3: "(u³ - 3u² + 2u) / 6",
        4: "(u⁴ - 6u³ + 11u² - 6u) / 24",
    },
    "v": {
        1: "v",
        2: "(v² + v) / 2",
        3: "(v³ + 3v² + 2v) / 6",
        4: "(v⁴ + 6v³ + 11v² + 6v) / 24",
    },
}


def _evaluar_lagrange_en_grilla(
    xs: np.ndarray,
    ys: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Evalúa el polinomio interpolante exacto que pasa por (xs, ys) sobre x_grid."""
    n = len(xs)
    with np.errstate(all="ignore"):
        y_grid = np.zeros_like(x_grid, dtype=float)
        for i in range(n):
            L_i = np.ones_like(x_grid, dtype=float)
            for j in range(n):
                if i != j:
                    L_i *= (x_grid - xs[j]) / (xs[i] - xs[j])
            y_grid += ys[i] * L_i
    return y_grid


def _curva_lagrange_bloque(xs: np.ndarray, ys: np.ndarray, n_pts: int = 35) -> tuple[list[float], list[float]]:
    """Genera los puntos (x_curva, y_curva) del polinomio interpolante exacto del bloque."""
    x_grid = np.linspace(float(xs[0]), float(xs[-1]), n_pts)
    y_grid = _evaluar_lagrange_en_grilla(xs, ys, x_grid)
    return x_grid.tolist(), [float(v) if np.isfinite(v) else None for v in y_grid.tolist()]


def _evaluar_curva_y_punto_derivacion(
    x: np.ndarray,
    y: np.ndarray,
    x_objetivo: float,
    n_pts: int = 120,
) -> tuple[float, dict[str, list[float | None]]]:
    """Evalúa P(x*) y la curva continua del polinomio interpolador para graficar tangentes."""
    y_eval = float(_evaluar_lagrange_en_grilla(x, y, np.array([x_objetivo], dtype=float))[0])
    x_min = min(float(x[0]), float(x_objetivo))
    x_max = max(float(x[-1]), float(x_objetivo))
    x_grid = np.linspace(x_min, x_max, n_pts)
    y_grid = _evaluar_lagrange_en_grilla(x, y, x_grid)
    curva = {
        "x": x_grid.tolist(),
        "y": [float(v) if np.isfinite(v) else None for v in y_grid.tolist()],
    }
    return y_eval, curva


def recomendar_esquema_diferenciacion(
    x_values: list[float] | np.ndarray,
    x_eval: float,
) -> dict[str, str]:
    """Recomienda Gregory Ascendente (Δ) en la mitad inicial o Descendente (∇) en la mitad final."""
    x = np.asarray(x_values, dtype=float)
    x_medio = 0.5 * (float(x[0]) + float(x[-1]))
    if float(x_eval) <= x_medio:
        return {
            "esquema": "ascendente",
            "etiqueta": "Progresivas (Ascendente Δ)",
            "motivo": (
                f"El punto x* = {float(x_eval):g} está en la mitad inicial de la tabla (≤ {x_medio:g}): "
                "conviene usar diferencias progresivas Δᵏ hacia adelante."
            ),
        }
    return {
        "esquema": "descendente",
        "etiqueta": "Regresivas (Descendente ∇)",
        "motivo": (
            f"El punto x* = {float(x_eval):g} está en la mitad final de la tabla (> {x_medio:g}): "
            "conviene usar diferencias regresivas ∇ᵏ hacia atrás."
        ),
    }


def _evaluar_terminos_derivada_ng(
    tabla: np.ndarray,
    idx_punto: int,
    n: int,
    param_sym: str,
    orden_derivada: int,
    coefs_specs: list[tuple[int, float, str]],
    es_ascendente: bool,
) -> tuple[float, list[dict[str, Any]]]:
    """Evalúa los términos distribuidos de la derivada NG Ascendente (u) o Descendente (v)."""
    suma_diff = 0.0
    terminos_detalle: list[dict[str, Any]] = []
    poly_map = _POLY_PARAM_NG[param_sym]
    for k_idx, coef, deriv_expr in coefs_specs:
        disponible = (idx_punto + k_idx) < n if es_ascendente else idx_punto >= k_idx
        if k_idx < n and disponible:
            diff_val = float(tabla[idx_punto, k_idx])
            aporte = coef * diff_val
            suma_diff += aporte
            terminos_detalle.append(
                {
                    "orden_diff": k_idx,
                    "coeficiente": coef,
                    "diferencia": diff_val,
                    "aporte": aporte,
                    "polinomio_param": poly_map[k_idx],
                    "derivada_param_expr": deriv_expr,
                    "orden_derivada": orden_derivada,
                }
            )
    return suma_diff, terminos_detalle


def derivada_newton_gregory_ascendente(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    idx_punto: int = 0,
    orden_derivada: int = 1,
    x_eval_libre: float | None = None,
) -> dict[str, Any]:
    """
    Calcula f'(x) o f''(x) por diferencias progresivas de Newton-Gregory.
    Si se especifica `x_eval_libre`, evalúa la fórmula general en función de u = (x - x[idx_punto]) / h
    (ej. cálculo de f'(12.4) en TP6 con u = 0.4).
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n = len(y)
    if n != len(x):
        raise ValueError("x e y deben tener igual longitud.")
    if orden_derivada not in (1, 2):
        raise ValueError("Solo se admiten orden_derivada = 1 o 2.")
    if idx_punto < 0 or idx_punto >= n:
        raise IndexError(f"Índice {idx_punto} fuera de rango [0, {n-1}].")

    h = validar_equiespaciado(x)
    u = 0.0 if x_eval_libre is None else float((x_eval_libre - x[idx_punto]) / h)
    x_objetivo = float(x[idx_punto]) if x_eval_libre is None else float(x_eval_libre)

    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            tabla[i, j] = tabla[i + 1, j - 1] - tabla[i, j - 1]

    if orden_derivada == 1:
        specs = [
            (1, 1.0, "1"),
            (2, (2.0 * u - 1.0) / 2.0, "(2u - 1) / 2"),
            (3, (3.0 * u**2 - 6.0 * u + 2.0) / 6.0, "(3u² - 6u + 2) / 6"),
            (4, (4.0 * u**3 - 18.0 * u**2 + 22.0 * u - 6.0) / 24.0, "(4u³ - 18u² + 22u - 6) / 24"),
        ]
        suma_diff, terminos_detalle = _evaluar_terminos_derivada_ng(
            tabla, idx_punto, n, "u", 1, specs, es_ascendente=True
        )
        derivada = suma_diff / h
    else:
        if n - idx_punto < 3:
            raise ValueError("Se requieren al menos 3 puntos hacia adelante para la 2da derivada.")
        specs = [
            (2, 1.0, "1"),
            (3, u - 1.0, "u - 1"),
            (4, (6.0 * u**2 - 18.0 * u + 11.0) / 12.0, "(6u² - 18u + 11) / 12"),
        ]
        suma_diff, terminos_detalle = _evaluar_terminos_derivada_ng(
            tabla, idx_punto, n, "u", 2, specs, es_ascendente=True
        )
        derivada = suma_diff / (h**2)

    y_eval, curva_poly = _evaluar_curva_y_punto_derivacion(x, y, x_objetivo)
    return {
        "x_eval": x_objetivo,
        "y_eval": y_eval,
        "u": u,
        "derivada": float(derivada),
        "orden_derivada": orden_derivada,
        "h": h,
        "indice_usado": idx_punto,
        "esquema": "ascendente",
        "recomendacion": recomendar_esquema_diferenciacion(x, x_objetivo),
        "curva_polinomio": curva_poly,
        "tabla_diferencias": tabla.tolist(),
        "terminos_detalle": terminos_detalle,
    }


def derivada_newton_gregory_descendente(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    idx_punto: int = -1,
    orden_derivada: int = 1,
    x_eval_libre: float | None = None,
) -> dict[str, Any]:
    """Calcula f'(x) o f''(x) por diferencias regresivas de Newton-Gregory en v = (x - x[idx_punto]) / h."""
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n = len(y)
    if n != len(x):
        raise ValueError("x e y deben tener igual longitud.")
    if orden_derivada not in (1, 2):
        raise ValueError("Solo se admiten orden_derivada = 1 o 2.")
    if idx_punto < 0:
        idx_punto = n + idx_punto
    if idx_punto < 0 or idx_punto >= n:
        raise IndexError(f"Índice {idx_punto} fuera de rango [0, {n-1}].")
    h = validar_equiespaciado(x)
    v = 0.0 if x_eval_libre is None else float((x_eval_libre - x[idx_punto]) / h)
    x_objetivo = float(x[idx_punto]) if x_eval_libre is None else float(x_eval_libre)

    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            tabla[i, j] = tabla[i, j - 1] - tabla[i - 1, j - 1]

    if orden_derivada == 1:
        specs = [
            (1, 1.0, "1"),
            (2, (2.0 * v + 1.0) / 2.0, "(2v + 1) / 2"),
            (3, (3.0 * v**2 + 6.0 * v + 2.0) / 6.0, "(3v² + 6v + 2) / 6"),
            (4, (4.0 * v**3 + 18.0 * v**2 + 22.0 * v + 6.0) / 24.0, "(4v³ + 18v² + 22v + 6) / 24"),
        ]
        suma_diff, terminos_detalle = _evaluar_terminos_derivada_ng(
            tabla, idx_punto, n, "v", 1, specs, es_ascendente=False
        )
        derivada = suma_diff / h
    else:
        if idx_punto < 2:
            raise ValueError("Se requieren al menos 3 puntos hacia atrás para la 2da derivada retrógrada.")
        specs = [
            (2, 1.0, "1"),
            (3, v + 1.0, "v + 1"),
            (4, (6.0 * v**2 + 18.0 * v + 11.0) / 12.0, "(6v² + 18v + 11) / 12"),
        ]
        suma_diff, terminos_detalle = _evaluar_terminos_derivada_ng(
            tabla, idx_punto, n, "v", 2, specs, es_ascendente=False
        )
        derivada = suma_diff / (h**2)

    y_eval, curva_poly = _evaluar_curva_y_punto_derivacion(x, y, x_objetivo)
    return {
        "x_eval": x_objetivo,
        "y_eval": y_eval,
        "v": v,
        "derivada": float(derivada),
        "orden_derivada": orden_derivada,
        "h": h,
        "indice_usado": idx_punto,
        "esquema": "descendente",
        "recomendacion": recomendar_esquema_diferenciacion(x, x_objetivo),
        "curva_polinomio": curva_poly,
        "tabla_diferencias": tabla.tolist(),
        "terminos_detalle": terminos_detalle,
    }


def calcular_derivada_gregory_completa(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    x_eval: float,
    orden_derivada: int = 1,
    esquema: str = "auto",
) -> dict[str, Any]:
    """Despachador puro de diferenciación por Gregory con selección automática de ancla y sentido."""
    x = np.asarray(x_values, dtype=float)
    n = len(x)
    rec = recomendar_esquema_diferenciacion(x, x_eval)
    esquema_eff = rec["esquema"] if esquema == "auto" else esquema
    if esquema_eff == "ascendente":
        idx_base = max(0, min(n - 2, int(np.searchsorted(x, x_eval) - 1)))
        res = derivada_newton_gregory_ascendente(
            x, y_values, idx_punto=idx_base, orden_derivada=orden_derivada, x_eval_libre=x_eval
        )
    else:
        idx_base = max(1, min(n - 1, int(np.searchsorted(x, x_eval))))
        res = derivada_newton_gregory_descendente(
            x, y_values, idx_punto=idx_base, orden_derivada=orden_derivada, x_eval_libre=x_eval
        )
    res["recomendacion"] = rec
    return res


# ==============================================================================
# 2. REGLAS CLÁSICAS DE INTEGRACIÓN (TRAPECIO, SIMPSON 1/3, 3/8 Y COMBINADO)
# ==============================================================================

def calcular_trapecio_compuesto(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n_puntos = len(y)
    if n_puntos < 2:
        raise ValueError("Se requieren al menos 2 puntos para aplicar Trapecio.")
    h = validar_equiespaciado(x)
    n_intervalos = n_puntos - 1

    E = float(y[0] + y[-1])
    pares_idx = list(range(2, n_puntos - 1, 2))
    impares_idx = list(range(1, n_puntos - 1, 2))
    P = float(np.sum(y[pares_idx])) if pares_idx else 0.0
    I = float(np.sum(y[impares_idx])) if impares_idx else 0.0
    area = (h / 2.0) * (E + 2.0 * P + 2.0 * I)

    return {
        "metodo": "Trapecio",
        "area": float(area),
        "h": h,
        "n_intervalos": n_intervalos,
        "n_puntos": n_puntos,
        "E": E, "P": P, "I": I,
        "formula": f"({h:.4g}/2) · [{E:.4f} + 2·({P:.4f}) + 2·({I:.4f})]",
    }


def calcular_simpson_1_3(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n_puntos = len(y)
    n_intervalos = n_puntos - 1
    if n_puntos < 3:
        raise ValueError("Simpson 1/3 requiere al menos 3 puntos.")
    if n_intervalos % 2 != 0:
        raise ValueError(f"Simpson 1/3 requiere n par de intervalos (recibido n={n_intervalos}).")
    h = validar_equiespaciado(x)

    E = float(y[0] + y[-1])
    pares_idx = list(range(2, n_puntos - 1, 2))
    impares_idx = list(range(1, n_puntos - 1, 2))
    P = float(np.sum(y[pares_idx])) if pares_idx else 0.0
    I = float(np.sum(y[impares_idx])) if impares_idx else 0.0
    area = (h / 3.0) * (E + 4.0 * I + 2.0 * P)

    return {
        "metodo": "Simpson 1/3",
        "area": float(area),
        "h": h,
        "n_intervalos": n_intervalos,
        "n_puntos": n_puntos,
        "E": E, "P": P, "I": I,
        "formula": f"({h:.4g}/3) · [{E:.4f} + 4·({I:.4f}) + 2·({P:.4f})]",
    }


def calcular_simpson_3_8(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n_puntos = len(y)
    n_intervalos = n_puntos - 1
    if n_puntos < 4:
        raise ValueError("Simpson 3/8 requiere al menos 4 puntos.")
    if n_intervalos % 3 != 0:
        raise ValueError(f"Simpson 3/8 requiere n múltiplo de 3 (recibido n={n_intervalos}).")
    h = validar_equiespaciado(x)

    area_total = 0.0
    grupos = []
    for i in range(0, n_intervalos, 3):
        y0, y1, y2, y3 = y[i], y[i + 1], y[i + 2], y[i + 3]
        a_bloque = (3.0 * h / 8.0) * (y0 + 3.0 * y1 + 3.0 * y2 + y3)
        area_total += a_bloque
        grupos.append({
            "grupo": len(grupos) + 1,
            "x_bloque": [float(x[i]), float(x[i + 1]), float(x[i + 2]), float(x[i + 3])],
            "y_bloque": [float(y0), float(y1), float(y2), float(y3)],
            "area_bloque": float(a_bloque),
        })

    return {
        "metodo": "Simpson 3/8",
        "area": float(area_total),
        "h": h,
        "n_intervalos": n_intervalos,
        "n_puntos": n_puntos,
        "grupos": grupos,
        "formula": f"(3·{h:.4g}/8) · Σ(y₀ + 3y₁ + 3y₂ + y₃)",
    }


def calcular_simpson_combinado(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n_intervalos = len(y) - 1
    if n_intervalos % 2 == 0:
        return calcular_simpson_1_3(x, y)
    elif n_intervalos % 3 == 0:
        return calcular_simpson_3_8(x, y)
    elif n_intervalos >= 5 and (n_intervalos - 3) % 2 == 0:
        res_38 = calcular_simpson_3_8(x[:4], y[:4])
        res_13 = calcular_simpson_1_3(x[3:], y[3:])
        return {
            "metodo": "Simpson Combinado (3/8 + 1/3)",
            "area": float(res_38["area"] + res_13["area"]),
            "area_3_8": res_38["area"],
            "area_1_3": res_13["area"],
            "detalles_3_8": res_38,
            "detalles_1_3": res_13,
        }
    else:
        return calcular_trapecio_compuesto(x, y)


def cota_error_trapecio(a: float, b: float, n_intervalos: int, max_d2f: float) -> float:
    if n_intervalos <= 0:
        raise ValueError("n_intervalos debe ser mayor a cero.")
    return float(((b - a) ** 3 / (12.0 * (n_intervalos**2))) * max_d2f)


def cota_error_simpson_1_3(a: float, b: float, n_intervalos: int, max_d4f: float) -> float:
    if n_intervalos <= 0:
        raise ValueError("n_intervalos debe ser mayor a cero.")
    return float(((b - a) ** 5 / (180.0 * (n_intervalos**4))) * max_d4f)


# ==============================================================================
# 3. MOTOR ADAPTATIVO EN VIVO ANTE CAMBIOS DE Δh, PARIDAD O NODOS EDITADOS
# ==============================================================================

def _curva_lagrange_bloque(xs: np.ndarray, ys: np.ndarray, n_pts: int = 35) -> tuple[list[float], list[float]]:
    """Genera los puntos (x_curva, y_curva) del polinomio interpolante exacto del bloque."""
    x_grid = np.linspace(float(xs[0]), float(xs[-1]), n_pts)
    y_grid = _evaluar_lagrange_en_grilla(xs, ys, x_grid)
    return x_grid.tolist(), [float(v) if np.isfinite(v) else None for v in y_grid.tolist()]


def _construir_subbloques_segmento(
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    regla: str,
) -> list[dict[str, Any]]:
    """
    Divide un segmento de paso constante h en bloques elementales gráficos y numéricos:
    - Trapecio: bloques de 1 subintervalo (lineal).
    - Simpson 1/3: bloques de 2 subintervalos (parábola).
    - Simpson 3/8: bloques de 3 subintervalos (cúbica).
    """
    m = len(x_seg) - 1
    h = float(x_seg[1] - x_seg[0])
    bloques = []
    if regla == "Trapecio":
        for i in range(m):
            xs, ys = x_seg[i : i + 2], y_seg[i : i + 2]
            area_b = float((h / 2.0) * (ys[0] + ys[1]))
            bloques.append({
                "regla": "Trapecio",
                "grado": 1,
                "h": h,
                "x_nodos": xs.tolist(),
                "y_nodos": ys.tolist(),
                "area": area_b,
                "formula": f"({h:.4g}/2)·({ys[0]:.4f} + {ys[1]:.4f}) = {area_b:.5f}",
                "color": COLORES_REGLAS["Trapecio"],
                "x_curva": xs.tolist(),
                "y_curva": ys.tolist(),
            })
    elif regla == "Simpson 1/3":
        for i in range(0, m, 2):
            xs, ys = x_seg[i : i + 3], y_seg[i : i + 3]
            area_b = float((h / 3.0) * (ys[0] + 4.0 * ys[1] + ys[2]))
            xc, yc = _curva_lagrange_bloque(xs, ys, 32)
            bloques.append({
                "regla": "Simpson 1/3",
                "grado": 2,
                "h": h,
                "x_nodos": xs.tolist(),
                "y_nodos": ys.tolist(),
                "area": area_b,
                "formula": f"({h:.4g}/3)·({ys[0]:.4f} + 4·{ys[1]:.4f} + {ys[2]:.4f}) = {area_b:.5f}",
                "color": COLORES_REGLAS["Simpson 1/3"],
                "x_curva": xc,
                "y_curva": yc,
            })
    elif regla == "Simpson 3/8":
        for i in range(0, m, 3):
            xs, ys = x_seg[i : i + 4], y_seg[i : i + 4]
            area_b = float((3.0 * h / 8.0) * (ys[0] + 3.0 * ys[1] + 3.0 * ys[2] + ys[3]))
            xc, yc = _curva_lagrange_bloque(xs, ys, 40)
            bloques.append({
                "regla": "Simpson 3/8",
                "grado": 3,
                "h": h,
                "x_nodos": xs.tolist(),
                "y_nodos": ys.tolist(),
                "area": area_b,
                "formula": f"(3·{h:.4g}/8)·({ys[0]:.4f} + 3·{ys[1]:.4f} + 3·{ys[2]:.4f} + {ys[3]:.4f}) = {area_b:.5f}",
                "color": COLORES_REGLAS["Simpson 3/8"],
                "x_curva": xc,
                "y_curva": yc,
            })
    return bloques


def calcular_ponderacion_nodal(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    bloques: list[dict[str, Any]],
    metodo_solicitado: str = "simpson_1_3",
) -> dict[str, Any]:
    """Calcula pesos nodales wᵢ, roles (Extremo/Interior/Frontera) y suma ponderada Σ wᵢ·f(xᵢ)."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    n_pts = len(x)
    n_int = n_pts - 1
    reglas = list(dict.fromkeys(b["regla"] for b in bloques))
    h_unicos = {round(float(b["h"]), 6) for b in bloques}

    acum_coef = [0.0] * n_pts
    es_frontera = [False] * n_pts
    idx_cursor = 0
    for b in bloques:
        m_b = len(b["x_nodos"]) - 1
        h_b = float(b["h"])
        regla_b = b["regla"]
        if regla_b == "Trapecio":
            loc = [(h_b / 2.0) * 1.0, (h_b / 2.0) * 1.0]
        elif regla_b == "Simpson 1/3":
            loc = [(h_b / 3.0) * w for w in (1.0, 4.0, 1.0)]
        else:
            loc = [(3.0 * h_b / 8.0) * w for w in (1.0, 3.0, 3.0, 1.0)]
        for offset, c_loc in enumerate(loc):
            if idx_cursor + offset < n_pts:
                acum_coef[idx_cursor + offset] += c_loc
        if idx_cursor > 0 and idx_cursor < n_pts - 1:
            es_frontera[idx_cursor] = True
        idx_cursor += m_b

    es_puro_uniforme = len(reglas) == 1 and len(h_unicos) == 1
    if es_puro_uniforme:
        regla_u = reglas[0]
        h_u = float(bloques[0]["h"])
        if regla_u == "Simpson 1/3":
            escala = h_u / 3.0
            escala_txt = f"{h_u:.4g} / 3"
            patron = "(1, 4, 2, 4, ..., 2, 4, 1)"
            pesos = [1 if (i == 0 or i == n_int) else (4 if i % 2 == 1 else 2) for i in range(n_pts)]
        elif regla_u == "Simpson 3/8":
            escala = (3.0 * h_u) / 8.0
            escala_txt = f"3·{h_u:.4g} / 8"
            patron = "(1, 3, 3, 2, 3, 3, ..., 2, 3, 3, 1)"
            pesos = [1 if (i == 0 or i == n_int) else (2 if i % 3 == 0 else 3) for i in range(n_pts)]
        else:
            escala = h_u / 2.0
            escala_txt = f"{h_u:.4g} / 2"
            patron = "(1, 2, 2, ..., 2, 1)"
            pesos = [1 if (i == 0 or i == n_int) else 2 for i in range(n_pts)]
    else:
        h_ref = float(bloques[0]["h"]) if bloques else 1.0
        escala = max(h_ref / 24.0, 1e-12)
        escala_txt = f"{h_ref:.4g} / 24"
        patron = "Combinación adaptativa por sub-bloques (" + " + ".join(reglas) + ")"
        pesos = [round(c / escala, 4) for c in acum_coef]
        pesos = [int(round(w)) if abs(w - round(w)) < 1e-6 else w for w in pesos]

    nodos_info = []
    suma_ponderada = 0.0
    for i in range(n_pts):
        w = pesos[i]
        yi = float(y[i])
        wy = float(w) * yi
        suma_ponderada += wy
        if i == 0 or i == n_int:
            rol = "Extremo"
        elif es_frontera[i]:
            rol = "Frontera"
        else:
            rol = "Interior"
        nodos_info.append({
            "index": i,
            "x": float(x[i]),
            "y": yi,
            "weight": w,
            "rol": rol,
            "wy": wy,
            "aporte": float(acum_coef[i] * yi),
        })

    area_res = float(sum(acum_coef[i] * float(y[i]) for i in range(n_pts)))
    return {
        "es_puro_uniforme": es_puro_uniforme,
        "regla_principal": reglas[0] if reglas else metodo_solicitado,
        "patron_texto": patron,
        "escala": float(escala),
        "escala_texto": escala_txt,
        "suma_ponderada": float(suma_ponderada),
        "formula_resumen": f"I ≈ ({escala_txt}) · ({suma_ponderada:.3f}) = {area_res:.6f}",
        "nodos": nodos_info,
    }


def _evaluar_y_desde_f(x_nuevos: np.ndarray, f_str: str) -> np.ndarray:
    """Evalúa una expresión matemática f(x) sobre un arreglo de nodos."""
    clave_f = " ".join(f_str.strip().split())
    if clave_f in INTEGRALES_PRECOMPUTADAS:
        fn_np = INTEGRALES_PRECOMPUTADAS[clave_f]["fn_np"]
    else:
        fn_np = compilar_lambdify_1d(clave_f)
    y_vals = evaluar_fn_vectorizado(fn_np, x_nuevos, limite_abs=None)
    return np.array([v if v is not None else float("nan") for v in y_vals], dtype=float)


def _evaluar_referencia_exacta(
    x: np.ndarray,
    area_total: float,
    f_str: str | None,
) -> dict[str, Any] | None:
    """Calcula la integral analítica (o Gauss-Legendre 16 pts) y la curva de referencia."""
    if not (f_str and f_str.strip()):
        return None
    try:
        clave_f = " ".join(f_str.strip().split())
        a_tot, b_tot = float(x[0]), float(x[-1])
        metodo_ref = "Analítica"
        if clave_f in INTEGRALES_PRECOMPUTADAS:
            pre_int = INTEGRALES_PRECOMPUTADAS[clave_f]
            integral_exacta = float(pre_int["exact_eval"](a_tot, b_tot))
            f_latex_int = pre_int["f_latex"]
            fn_np = pre_int["fn_np"]
            if "exp(-x**2)" in clave_f:
                metodo_ref = "Gauss-Legendre 16 pts"
        else:
            fn_np = compilar_lambdify_1d(clave_f)
            try:
                res_int = integral_sympy_cached(clave_f, a_tot, b_tot)
                integral_exacta = float(res_int["integral_exacta"])
                f_latex_int = res_int["f_latex"]
            except Exception:
                integral_exacta = integrar_gauss_legendre_16(fn_np, a_tot, b_tot)
                f_latex_int = clave_f
                metodo_ref = "Gauss-Legendre 16 pts"

        xg = np.linspace(a_tot, b_tot, 240)
        yg = evaluar_fn_vectorizado(fn_np, xg, limite_abs=1e9)
        err_abs = abs(integral_exacta - area_total)
        err_pct = (err_abs / abs(integral_exacta) * 100.0) if abs(integral_exacta) > 1e-14 else 0.0
        return {
            "integral_exacta": integral_exacta,
            "error_absoluto": err_abs,
            "error_relativo_pct": err_pct,
            "f_latex": f_latex_int,
            "metodo_ref": metodo_ref,
            "curva_exacta": {"x": xg.tolist(), "y": yg},
        }
    except Exception:
        return None


def _validar_nodos_integracion(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valida nodos crecientes y finitos para cuadratura numérica."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 nodos (x, y) válidos.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Todos los nodos (x, y) deben contener valores numéricos finitos.")
    h_arr = np.diff(x)
    if not np.all(h_arr > 1e-12):
        raise ValueError("Los valores de x deben estar estrictamente ordenados de forma creciente.")
    return x, y, h_arr


def _crear_bloque_hibrido(
    regla: str,
    x: np.ndarray,
    y: np.ndarray,
    i_ini: int,
    h_b: float,
    motivo: str,
) -> dict[str, Any]:
    """Construye un bloque elemental (Trapecio, Simpson 1/3 o Simpson 3/8) con su curva y metadatos."""
    if regla == "Simpson 3/8":
        span = 3
        xs_b, ys_b = x[i_ini : i_ini + 4], y[i_ini : i_ini + 4]
        area_b = float((3.0 * h_b / 8.0) * (ys_b[0] + 3.0 * ys_b[1] + 3.0 * ys_b[2] + ys_b[3]))
        xc, yc = _curva_lagrange_bloque(xs_b, ys_b, 40)
        formula = f"(3·{h_b:.4g}/8)·({ys_b[0]:.4f} + 3·{ys_b[1]:.4f} + 3·{ys_b[2]:.4f} + {ys_b[3]:.4f}) = {area_b:.5f}"
        grado = 3
    elif regla == "Simpson 1/3":
        span = 2
        xs_b, ys_b = x[i_ini : i_ini + 3], y[i_ini : i_ini + 3]
        area_b = float((h_b / 3.0) * (ys_b[0] + 4.0 * ys_b[1] + ys_b[2]))
        xc, yc = _curva_lagrange_bloque(xs_b, ys_b, 35)
        formula = f"({h_b:.4g}/3)·({ys_b[0]:.4f} + 4·{ys_b[1]:.4f} + {ys_b[2]:.4f}) = {area_b:.5f}"
        grado = 2
    else:
        span = 1
        xs_b, ys_b = x[i_ini : i_ini + 2], y[i_ini : i_ini + 2]
        area_b = float((h_b / 2.0) * (ys_b[0] + ys_b[1]))
        xc, yc = xs_b.tolist(), ys_b.tolist()
        formula = f"({h_b:.4g}/2)·({ys_b[0]:.4f} + {ys_b[1]:.4f}) = {area_b:.5f}"
        grado = 1

    return {
        "regla": regla,
        "grado": grado,
        "h": float(h_b),
        "n_subintervalos": span,
        "nodos_label": f"x_{i_ini}..x_{i_ini + span}",
        "intervalo": f"[{xs_b[0]:g}, {xs_b[-1]:g}]",
        "x_nodos": xs_b.tolist(),
        "y_nodos": ys_b.tolist(),
        "area": area_b,
        "formula": formula,
        "motivo": motivo,
        "color": COLORES_REGLAS[regla],
        "x_curva": xc,
        "y_curva": yc,
    }


def integrar_simpson_1_3_hibrido(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    f_str: str | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """
    Integra de izquierda a derecha aplicando Regla de Simpson 1/3 sobre cada terna
    de nodos equidistantes (2 subintervalos contiguos con idéntico paso h) y cayendo
    automáticamente a Regla del Trapecio cuando cambia el paso h o resta un tramo impar.
    """
    t_ini = time.perf_counter()
    x, y, h_arr = _validar_nodos_integracion(x_values, y_values)

    n = len(h_arr)
    bloques: list[dict[str, Any]] = []
    avisos: list[str] = []
    i = 0

    while i < n:
        if i + 1 < n and np.isclose(h_arr[i], h_arr[i + 1], rtol=rtol, atol=atol):
            h_b = float(0.5 * (h_arr[i] + h_arr[i + 1]))
            bloques.append(
                _crear_bloque_hibrido(
                    "Simpson 1/3",
                    x,
                    y,
                    i,
                    h_b,
                    f"3 puntos equidistantes (h = {h_b:.4g} constante): parábola de Simpson 1/3",
                )
            )
            i += 2
        else:
            h_b = float(h_arr[i])
            intervalo_str = f"[{x[i]:g}, {x[i + 1]:g}]"
            if i + 1 < n:
                aviso = (
                    f"Cambio de paso h detectado en el tramo {intervalo_str} "
                    f"(h = {h_b:.4g} ≠ siguiente h = {float(h_arr[i + 1]):.4g}): "
                    "al no haber 3 puntos equidistantes, se aplica automáticamente Regla del Trapecio."
                )
            elif i + 1 == n and n >= 2 and not np.isclose(h_arr[i - 1], h_b, rtol=rtol, atol=atol):
                aviso = (
                    f"Cambio de paso h detectado en el tramo {intervalo_str} "
                    f"(h = {h_b:.4g} ≠ anterior h = {float(h_arr[i - 1]):.4g}): "
                    "al no haber 3 puntos equidistantes, se aplica automáticamente Regla del Trapecio."
                )
            else:
                aviso = (
                    f"Tramo final impar en {intervalo_str} (h = {h_b:.4g}): "
                    "al quedar solo 2 puntos sin pareja equidistante, se aplica automáticamente Regla del Trapecio."
                )
            avisos.append(aviso)
            bloques.append(_crear_bloque_hibrido("Trapecio", x, y, i, h_b, aviso))
            i += 1

    area_total = float(sum(b["area"] for b in bloques))
    tramos_simpson_1_3 = sum(1 for b in bloques if b["regla"] == "Simpson 1/3")
    tramos_trapecio = sum(1 for b in bloques if b["regla"] == "Trapecio")
    reglas_globales = list(dict.fromkeys(b["regla"] for b in bloques))
    exacta_info = _evaluar_referencia_exacta(x, area_total, f_str)
    ponderacion = calcular_ponderacion_nodal(x, y, bloques, "simpson_1_3")

    return {
        "area_total": area_total,
        "metodo_solicitado": "simpson_1_3",
        "reglas_aplicadas": " + ".join(reglas_globales),
        "hubo_adaptacion": tramos_trapecio > 0,
        "multitramo": tramos_trapecio > 0,
        "avisos_pedagogicos": avisos,
        "bloques": bloques,
        "tramos_simpson_1_3": tramos_simpson_1_3,
        "tramos_trapecio": tramos_trapecio,
        "subintervalos_simpson_1_3": 2 * tramos_simpson_1_3,
        "subintervalos_trapecio": tramos_trapecio,
        "pasos_h": [float(v) for v in h_arr],
        "n_intervalos_total": n,
        "n_puntos_total": len(x),
        "exacta": exacta_info,
        "ponderacion_nodal": ponderacion,
        "duracion": time.perf_counter() - t_ini,
    }


def integrar_simpson_3_8_hibrido(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    f_str: str | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """
    Integra agrupando tramos contiguos de paso constante h:
    - Prioriza bloques cúbicos de Simpson 3/8 (3 subintervalos / 4 puntos equidistantes).
    - Combina automáticamente con Simpson 1/3 en mallas equidistantes no múltiplos de 3
      (ej. m = 5 -> 3 + 2, m = 4 -> 2 + 2, m = 2 -> 2) preservando orden O(h^4).
    - Cae automáticamente a Regla del Trapecio cuando cambia el paso h en un tramo aislado (m = 1).
    """
    t_ini = time.perf_counter()
    x, y, h_arr = _validar_nodos_integracion(x_values, y_values)

    tramos_indices: list[tuple[int, int]] = []
    ini_idx = 0
    for k in range(1, len(h_arr)):
        if not np.isclose(h_arr[k], h_arr[ini_idx], rtol=rtol, atol=atol):
            tramos_indices.append((ini_idx, k))
            ini_idx = k
    tramos_indices.append((ini_idx, len(x) - 1))

    bloques: list[dict[str, Any]] = []
    avisos: list[str] = []

    for i_ini, i_fin in tramos_indices:
        m = i_fin - i_ini
        h_t = float(np.mean(h_arr[i_ini:i_fin]))
        tramo_str = f"[{x[i_ini]:g}, {x[i_fin]:g}]"

        if m % 3 == 0:
            for pos in range(i_ini, i_fin, 3):
                bloques.append(
                    _crear_bloque_hibrido(
                        "Simpson 3/8",
                        x,
                        y,
                        pos,
                        h_t,
                        f"4 puntos equidistantes (h = {h_t:.4g} constante): cúbica de Simpson 3/8",
                    )
                )
        elif m >= 5 and m % 3 == 2:
            n_38 = m - 2
            avisos.append(
                f"En {tramo_str} hay m = {m} subintervalos equidistantes (h = {h_t:.4g}): "
                f"se combina automáticamente Simpson 3/8 (primeros {n_38} subintervalos) + "
                "Simpson 1/3 (últimos 2 subintervalos) preservando precisión O(h⁴)."
            )
            for pos in range(i_ini, i_ini + n_38, 3):
                bloques.append(
                    _crear_bloque_hibrido(
                        "Simpson 3/8",
                        x,
                        y,
                        pos,
                        h_t,
                        f"Bloque cúbico Simpson 3/8 del esquema combinado (h = {h_t:.4g})",
                    )
                )
            bloques.append(
                _crear_bloque_hibrido(
                    "Simpson 1/3",
                    x,
                    y,
                    i_ini + n_38,
                    h_t,
                    f"Empalme parabólico Simpson 1/3 sobre los 2 subintervalos restantes (h = {h_t:.4g})",
                )
            )
        elif m >= 4 and m % 3 == 1:
            n_38 = m - 4
            if n_38 > 0:
                avisos.append(
                    f"En {tramo_str} hay m = {m} subintervalos equidistantes (h = {h_t:.4g}): "
                    f"se combina automáticamente Simpson 3/8 (primeros {n_38} subintervalos) + "
                    "Simpson 1/3 (últimos 4 subintervalos) preservando precisión O(h⁴)."
                )
                for pos in range(i_ini, i_ini + n_38, 3):
                    bloques.append(
                        _crear_bloque_hibrido(
                            "Simpson 3/8",
                            x,
                            y,
                            pos,
                            h_t,
                            f"Bloque cúbico Simpson 3/8 del esquema combinado (h = {h_t:.4g})",
                        )
                    )
            else:
                avisos.append(
                    f"En {tramo_str} hay m = 4 subintervalos equidistantes (h = {h_t:.4g}): "
                    "al ser múltiplo de 2 y no de 3, se aplica automáticamente Simpson 1/3 (2 parábolas) manteniendo precisión O(h⁴)."
                )
            for pos in range(i_ini + n_38, i_fin, 2):
                bloques.append(
                    _crear_bloque_hibrido(
                        "Simpson 1/3",
                        x,
                        y,
                        pos,
                        h_t,
                        f"Parábola de Simpson 1/3 en el resto par equidistantes (h = {h_t:.4g})",
                    )
                )
        elif m == 2:
            avisos.append(
                f"En {tramo_str} hay m = 2 subintervalos equidistantes (3 puntos, h = {h_t:.4g}): "
                "al no alcanzar 4 puntos para Simpson 3/8, se aplica automáticamente Simpson 1/3."
            )
            bloques.append(
                _crear_bloque_hibrido(
                    "Simpson 1/3",
                    x,
                    y,
                    i_ini,
                    h_t,
                    f"3 puntos equidistantes sin cuarto punto para 3/8 (h = {h_t:.4g}): se resuelve con Simpson 1/3",
                )
            )
        else:
            aviso = (
                f"Cambio de paso h detectado en el tramo {tramo_str} (h = {h_t:.4g}): "
                "al no haber 4 puntos equidistantes para Simpson 3/8 ni 3 para Simpson 1/3, se aplica automáticamente Regla del Trapecio."
            )
            avisos.append(aviso)
            bloques.append(_crear_bloque_hibrido("Trapecio", x, y, i_ini, h_t, aviso))

    area_total = float(sum(b["area"] for b in bloques))
    tramos_simpson_3_8 = sum(1 for b in bloques if b["regla"] == "Simpson 3/8")
    tramos_simpson_1_3 = sum(1 for b in bloques if b["regla"] == "Simpson 1/3")
    tramos_trapecio = sum(1 for b in bloques if b["regla"] == "Trapecio")
    reglas_globales = list(dict.fromkeys(b["regla"] for b in bloques))
    exacta_info = _evaluar_referencia_exacta(x, area_total, f_str)
    ponderacion = calcular_ponderacion_nodal(x, y, bloques, "simpson_3_8")

    return {
        "area_total": area_total,
        "metodo_solicitado": "simpson_3_8",
        "reglas_aplicadas": " + ".join(reglas_globales),
        "hubo_adaptacion": tramos_simpson_1_3 > 0 or tramos_trapecio > 0,
        "multitramo": len(tramos_indices) > 1,
        "avisos_pedagogicos": avisos,
        "bloques": bloques,
        "tramos_simpson_3_8": tramos_simpson_3_8,
        "tramos_simpson_1_3": tramos_simpson_1_3,
        "tramos_trapecio": tramos_trapecio,
        "subintervalos_simpson_3_8": 3 * tramos_simpson_3_8,
        "subintervalos_simpson_1_3": 2 * tramos_simpson_1_3,
        "subintervalos_trapecio": tramos_trapecio,
        "pasos_h": [float(v) for v in h_arr],
        "n_intervalos_total": len(h_arr),
        "n_puntos_total": len(x),
        "exacta": exacta_info,
        "ponderacion_nodal": ponderacion,
        "duracion": time.perf_counter() - t_ini,
    }


def integrar_trapecio_hibrido(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    f_str: str | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """
    Integra mediante Regla de los Trapecios tanto sobre mallas de paso uniforme h
    (reportando acumuladores E, P, I) como sobre mallas de paso variable h_k,
    contrastando el resultado frente a Simpson 1/3 Híbrido sobre los mismos nodos.
    """
    t_ini = time.perf_counter()
    x, y, h_arr = _validar_nodos_integracion(x_values, y_values)
    n = len(h_arr)
    es_uniforme = bool(np.allclose(h_arr, h_arr[0], rtol=rtol, atol=atol))

    bloques: list[dict[str, Any]] = []
    for i in range(n):
        h_i = float(h_arr[i])
        motivo = (
            f"Paso uniforme h = {h_i:.4g}: segmento trapezoidal lineal"
            if es_uniforme
            else f"Paso variable h_{i + 1} = {h_i:.4g}: Trapecio directo sin requerir equidistancia"
        )
        bloques.append(_crear_bloque_hibrido("Trapecio", x, y, i, h_i, motivo))

    area_total = float(sum(b["area"] for b in bloques))
    E = float(y[0] + y[-1])
    if es_uniforme:
        P: float | None = float(np.sum(y[2:-1:2])) if len(y) > 3 else 0.0
        I: float | None = float(np.sum(y[1:-1:2])) if len(y) > 2 else 0.0
    else:
        P = None
        I = None

    avisos: list[str] = []
    if not es_uniforme:
        avisos.append(
            f"Se detectaron pasos variables h_k = {[round(float(v), 4) for v in h_arr]} entre estacas: "
            "Trapecio se adapta nativamente evaluando (hᵢ/2)·(yᵢ + yᵢ₊₁) en cada subintervalo sin requerir ternas equidistantes."
        )

    res_simpson_ref = integrar_simpson_1_3_hibrido(x, y, f_str=None, rtol=rtol, atol=atol)
    exacta_info = _evaluar_referencia_exacta(x, area_total, f_str)
    ponderacion = calcular_ponderacion_nodal(x, y, bloques, "trapecio")

    return {
        "area_total": area_total,
        "metodo_solicitado": "trapecio",
        "reglas_aplicadas": "Trapecio",
        "hubo_adaptacion": not es_uniforme,
        "multitramo": not es_uniforme,
        "es_uniforme": es_uniforme,
        "E": E,
        "P": P,
        "I": I,
        "tramos_trapecio": n,
        "tramos_simpson_posibles": res_simpson_ref["tramos_simpson_1_3"],
        "area_simpson_ref": res_simpson_ref["area_total"],
        "avisos_pedagogicos": avisos,
        "bloques": bloques,
        "pasos_h": [float(v) for v in h_arr],
        "n_intervalos_total": n,
        "n_puntos_total": len(x),
        "exacta": exacta_info,
        "ponderacion_nodal": ponderacion,
        "duracion": time.perf_counter() - t_ini,
    }


def reconstruir_malla_por_pasos(
    x0: float,
    pasos_h: list[float] | np.ndarray,
    f_str: str | None = None,
    y_base: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Reconstruye los nodos (x, y) a partir de un origen x0 y una secuencia de pasos h_k."""
    pasos = [float(h) for h in pasos_h]
    if not pasos:
        raise ValueError("Se requiere al menos un subintervalo en pasos_h.")
    if any((not math.isfinite(h)) or h <= 1e-12 for h in pasos):
        raise ValueError("Todos los pasos h_k deben ser valores positivos finitos.")

    xs: list[float] = [round(float(x0), 6)]
    for h in pasos:
        xs.append(round(xs[-1] + h, 6))

    if f_str and f_str.strip():
        y_arr = _evaluar_y_desde_f(np.asarray(xs, dtype=float), f_str)
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("La función f(x) produjo valores no finitos en alguno de los nodos.")
        ys = [round(float(v), 6) for v in y_arr]
    elif y_base is not None and len(y_base) == len(xs):
        ys = [round(float(v), 6) for v in y_base]
    else:
        raise ValueError("Indicá una función f(x) válida o una lista y_base acorde al número de nodos.")

    return xs, ys


def integrar_adaptativo_en_vivo(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    metodo_solicitado: str = "simpson_1_3",
    estrategia_impar: str = "simpson_3_8",
    f_str: str | None = None,
) -> dict[str, Any]:
    """
    Motor adaptativo inteligente para Integración Numérica ante cambios de Δh, paridad n o nodos x_i:
    1. Agrupa subintervalos contiguos que comparten el mismo paso h.
    2. Si existen varios tramos de distinto h o si la cantidad de subintervalos en un tramo
       viola la restricción del método pedido (ej. n impar en Simpson 1/3 o no múltiplo de 3 en 3/8),
       adapta automáticamente combinando reglas (Simpson 3/8 + Simpson 1/3, o Simpson 1/3 + Trapecio)
       y genera avisos pedagógicos detallando la razón matemática.
    3. Calcula también la integral analítica exacta con SymPy si se suministra `f_str`.
    """
    t_ini = time.perf_counter()
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 nodos (x, y) válidos.")
    if not np.all(np.diff(x) > 1e-12):
        raise ValueError("Los valores de x deben estar estrictamente ordenados de forma creciente.")

    # Segmentar por tramos contiguos de paso h constante
    h_arr = np.diff(x)
    tramos_indices: list[tuple[int, int]] = []
    ini_idx = 0
    for i in range(1, len(h_arr)):
        if not np.isclose(h_arr[i], h_arr[ini_idx], rtol=1e-4, atol=1e-6):
            tramos_indices.append((ini_idx, i))
            ini_idx = i
    tramos_indices.append((ini_idx, len(x) - 1))

    multitramo = len(tramos_indices) > 1
    hubo_adaptacion = multitramo
    avisos: list[str] = []
    bloques_globales: list[dict[str, Any]] = []
    resumen_tramos: list[dict[str, Any]] = []

    if multitramo:
        avisos.append(
            f"Se detectaron {len(tramos_indices)} tramos con distinto paso Δh en la tabla "
            f"(pasos {[round(float(h_arr[a]), 4) for a, _ in tramos_indices]}). "
            "Ninguna regla compuesta pura aplica sobre h variable; el motor segmenta y adapta automáticamente cada tramo."
        )

    for t_idx, (i_ini, i_fin) in enumerate(tramos_indices, start=1):
        xs_t = x[i_ini : i_fin + 1]
        ys_t = y[i_ini : i_fin + 1]
        m = len(xs_t) - 1
        h_t = float(xs_t[1] - xs_t[0])

        bloques_t: list[dict[str, Any]] = []

        if metodo_solicitado == "trapecio":
            bloques_t = _construir_subbloques_segmento(xs_t, ys_t, "Trapecio")

        elif metodo_solicitado in ("simpson_1_3", "simpson_combinado"):
            if m == 1:
                hubo_adaptacion = True
                avisos.append(
                    f"Tramo {t_idx} [{xs_t[0]:g}, {xs_t[-1]:g}] tiene solo m = 1 subintervalo (h={h_t:g}): "
                    "Simpson exige al menos 2 subintervalos; se aplica Regla del Trapecio."
                )
                bloques_t = _construir_subbloques_segmento(xs_t, ys_t, "Trapecio")
            elif m % 2 == 0:
                bloques_t = _construir_subbloques_segmento(xs_t, ys_t, "Simpson 1/3")
            else:
                # m es impar >= 3
                hubo_adaptacion = True
                if estrategia_impar == "simpson_3_8":
                    if metodo_solicitado == "simpson_1_3":
                        avisos.append(
                            f"Simpson 1/3 requiere un número PAR de subintervalos, pero en [{xs_t[0]:g}, {xs_t[-1]:g}] "
                            f"hay m = {m} subintervalos (h={h_t:g}). "
                            "Se aplica automáticamente el esquema combinado: Simpson 3/8 en los primeros 3 subintervalos"
                            + (f" + Simpson 1/3 en los {m - 3} restantes." if m > 3 else ".")
                        )
                    bloques_t.extend(_construir_subbloques_segmento(xs_t[:4], ys_t[:4], "Simpson 3/8"))
                    if m > 3:
                        bloques_t.extend(_construir_subbloques_segmento(xs_t[3:], ys_t[3:], "Simpson 1/3"))
                else:
                    avisos.append(
                        f"Simpson 1/3 requiere m PAR (actual m = {m} en [{xs_t[0]:g}, {xs_t[-1]:g}], h={h_t:g}). "
                        f"Se combina Simpson 1/3 en los primeros {m - 1} subintervalos + Regla del Trapecio en el último subintervalo."
                    )
                    bloques_t.extend(_construir_subbloques_segmento(xs_t[:-1], ys_t[:-1], "Simpson 1/3"))
                    bloques_t.extend(_construir_subbloques_segmento(xs_t[-2:], ys_t[-2:], "Trapecio"))

        elif metodo_solicitado == "simpson_3_8":
            if m % 3 == 0:
                bloques_t = _construir_subbloques_segmento(xs_t, ys_t, "Simpson 3/8")
            else:
                hubo_adaptacion = True
                if m >= 3 and (m - 3) % 2 == 0 and m > 3:
                    avisos.append(
                        f"Simpson 3/8 puro requiere m múltiplo de 3 (actual m = {m}). "
                        f"Se aplica Simpson 3/8 en 3 subintervalos y Simpson 1/3 en los {m - 3} restantes."
                    )
                    bloques_t.extend(_construir_subbloques_segmento(xs_t[:4], ys_t[:4], "Simpson 3/8"))
                    bloques_t.extend(_construir_subbloques_segmento(xs_t[3:], ys_t[3:], "Simpson 1/3"))
                elif m % 2 == 0:
                    avisos.append(
                        f"Simpson 3/8 requiere m múltiplo de 3 (actual m = {m} par). "
                        "Se aplica Simpson 1/3 completo para mantener precisión O(h⁴)."
                    )
                    bloques_t = _construir_subbloques_segmento(xs_t, ys_t, "Simpson 1/3")
                elif m == 1:
                    avisos.append(f"Con m = 1 subintervalo se aplica Regla del Trapecio.")
                    bloques_t = _construir_subbloques_segmento(xs_t, ys_t, "Trapecio")
                else:
                    # m >= 7 y m%3 != 0 y m impar no (m-3)%2? Todo impar >=5 cumple (m-3)%2 == 0!
                    bloques_t = _construir_subbloques_segmento(xs_t, ys_t, "Trapecio")

        area_t = sum(b["area"] for b in bloques_t)
        reglas_t = list(dict.fromkeys(b["regla"] for b in bloques_t))
        resumen_tramos.append({
            "tramo": t_idx,
            "intervalo": f"[{xs_t[0]:g}, {xs_t[-1]:g}]",
            "h": h_t,
            "n_subintervalos": m,
            "reglas_aplicadas": " + ".join(reglas_t),
            "area_tramo": float(area_t),
        })
        bloques_globales.extend(bloques_t)

    area_total = float(sum(b["area"] for b in bloques_globales))
    reglas_globales = list(dict.fromkeys(b["regla"] for b in bloques_globales))

    # Verificación exacta precomputada (cátedra), SymPy cacheado o Gauss-Legendre 16 pts
    exacta_info = _evaluar_referencia_exacta(x, area_total, f_str)

    ponderacion = calcular_ponderacion_nodal(x, y, bloques_globales, metodo_solicitado)

    return {
        "area_total": area_total,
        "metodo_solicitado": metodo_solicitado,
        "reglas_aplicadas": " + ".join(reglas_globales),
        "hubo_adaptacion": hubo_adaptacion,
        "multitramo": multitramo,
        "avisos_pedagogicos": avisos,
        "bloques": bloques_globales,
        "resumen_tramos": resumen_tramos,
        "n_intervalos_total": len(x) - 1,
        "n_puntos_total": len(x),
        "exacta": exacta_info,
        "ponderacion_nodal": ponderacion,
        "duracion": time.perf_counter() - t_ini,
    }


def remuestrear_tabla_integracion(
    x_base: list[float],
    y_base: list[float],
    nuevo_h: float | None = None,
    nuevo_n: int | None = None,
    f_str: str | None = None,
) -> tuple[list[float], list[float], float]:
    """
    Remuestrea la tabla de integración en [a, b] cuando el usuario altera Δh o n en vivo.
    Si el nuevo Δh no divide exactamente a (b - a), genera un último subintervalo residual
    con el paso sobrante, activando naturalmente el motor adaptativo multitramo!
    """
    xb = np.array(x_base, dtype=float)
    yb = np.array(y_base, dtype=float)
    a, b = float(xb[0]), float(xb[-1])

    if nuevo_n is not None and nuevo_n >= 1:
        if int(nuevo_n) > 5000:
            raise ValueError("El número máximo de subintervalos permitido es 5000.")
        x_nuevos = np.linspace(a, b, int(nuevo_n) + 1)
        h_eff = float((b - a) / int(nuevo_n))
    elif nuevo_h is not None and nuevo_h > 0:
        h_val = float(nuevo_h)
        if h_val >= (b - a):
            x_nuevos = np.array([a, b], dtype=float)
        else:
            pts = [a]
            while pts[-1] + h_val < b - 1e-7:
                if len(pts) >= 5000:
                    raise ValueError("El paso h genera demasiados subintervalos (máximo 5000).")
                pts.append(round(pts[-1] + h_val, 8))
            if abs(pts[-1] - b) > 1e-6:
                pts.append(b)
            x_nuevos = np.array(pts, dtype=float)
        h_eff = h_val
    else:
        return list(x_base), list(y_base), float(xb[1] - xb[0])

    if f_str and f_str.strip():
        y_nuevos = _evaluar_y_desde_f(x_nuevos, f_str)
    else:
        with np.errstate(all="ignore"):
            y_nuevos = np.zeros_like(x_nuevos)
            n = len(xb)
            for i in range(n):
                L_i = np.ones_like(x_nuevos)
                for j in range(n):
                    if i != j:
                        L_i *= (x_nuevos - xb[j]) / (xb[i] - xb[j])
                y_nuevos += yb[i] * L_i

    return [round(float(v), 6) for v in x_nuevos], [round(float(v), 6) for v in y_nuevos], h_eff
