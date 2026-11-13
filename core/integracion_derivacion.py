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
import sympy as sp


COLORES_REGLAS = {
    "Trapecio": "#ff7f0e",      # Naranja cálido
    "Simpson 1/3": "#1f77b4",   # Azul sólido
    "Simpson 3/8": "#2ca02c",   # Verde esmeralda
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

    terminos_detalle = []
    if orden_derivada == 1:
        coefs_u = [
            (1, 1.0),
            (2, (2.0 * u - 1.0) / 2.0),
            (3, (3.0 * u**2 - 6.0 * u + 2.0) / 6.0),
            (4, (4.0 * u**3 - 18.0 * u**2 + 22.0 * u - 6.0) / 24.0),
        ]
        suma_diff = 0.0
        for k_idx, coef in coefs_u:
            if k_idx < n and (idx_punto + k_idx) < n:
                diff_val = float(tabla[idx_punto, k_idx])
                aporte = coef * diff_val
                suma_diff += aporte
                terminos_detalle.append({
                    "orden_diff": k_idx, "coeficiente": coef,
                    "diferencia": diff_val, "aporte": aporte,
                })
        derivada = suma_diff / h
    else:
        if n - idx_punto < 3:
            raise ValueError("Se requieren al menos 3 puntos hacia adelante para la 2da derivada.")
        coefs_u2 = [
            (2, 1.0),
            (3, u - 1.0),
            (4, (6.0 * u**2 - 18.0 * u + 11.0) / 12.0),
        ]
        suma_diff = 0.0
        for k_idx, coef in coefs_u2:
            if k_idx < n and (idx_punto + k_idx) < n:
                diff_val = float(tabla[idx_punto, k_idx])
                aporte = coef * diff_val
                suma_diff += aporte
                terminos_detalle.append({
                    "orden_diff": k_idx, "coeficiente": coef,
                    "diferencia": diff_val, "aporte": aporte,
                })
        derivada = suma_diff / (h**2)

    return {
        "x_eval": x_objetivo,
        "u": u,
        "derivada": float(derivada),
        "orden_derivada": orden_derivada,
        "h": h,
        "indice_usado": idx_punto,
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
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n = len(y)
    if idx_punto < 0:
        idx_punto = n + idx_punto
    h = validar_equiespaciado(x)
    v = 0.0 if x_eval_libre is None else float((x_eval_libre - x[idx_punto]) / h)
    x_objetivo = float(x[idx_punto]) if x_eval_libre is None else float(x_eval_libre)

    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            tabla[i, j] = tabla[i, j - 1] - tabla[i - 1, j - 1]

    terminos_detalle = []
    if orden_derivada == 1:
        coefs_v = [
            (1, 1.0),
            (2, (2.0 * v + 1.0) / 2.0),
            (3, (3.0 * v**2 + 6.0 * v + 2.0) / 6.0),
            (4, (4.0 * v**3 + 18.0 * v**2 + 22.0 * v + 6.0) / 24.0),
        ]
        suma_diff = 0.0
        for k_idx, coef in coefs_v:
            if k_idx < n and idx_punto >= k_idx:
                diff_val = float(tabla[idx_punto, k_idx])
                aporte = coef * diff_val
                suma_diff += aporte
                terminos_detalle.append({
                    "orden_diff": k_idx, "coeficiente": coef,
                    "diferencia": diff_val, "aporte": aporte,
                })
        derivada = suma_diff / h
    else:
        if idx_punto < 2:
            raise ValueError("Se requieren al menos 3 puntos hacia atrás para la 2da derivada retrógrada.")
        coefs_v2 = [
            (2, 1.0),
            (3, v + 1.0),
            (4, (6.0 * v**2 + 18.0 * v + 11.0) / 12.0),
        ]
        suma_diff = 0.0
        for k_idx, coef in coefs_v2:
            if k_idx < n and idx_punto >= k_idx:
                diff_val = float(tabla[idx_punto, k_idx])
                aporte = coef * diff_val
                suma_diff += aporte
                terminos_detalle.append({
                    "orden_diff": k_idx, "coeficiente": coef,
                    "diferencia": diff_val, "aporte": aporte,
                })
        derivada = suma_diff / (h**2)

    return {
        "x_eval": x_objetivo,
        "v": v,
        "derivada": float(derivada),
        "orden_derivada": orden_derivada,
        "h": h,
        "indice_usado": idx_punto,
        "tabla_diferencias": tabla.tolist(),
        "terminos_detalle": terminos_detalle,
    }


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
    n = len(xs)
    y_grid = np.zeros_like(x_grid)
    for i in range(n):
        L_i = np.ones_like(x_grid)
        for j in range(n):
            if i != j:
                L_i *= (x_grid - xs[j]) / (xs[i] - xs[j])
        y_grid += ys[i] * L_i
    return x_grid.tolist(), y_grid.tolist()


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

    # Verificación exacta con SymPy si se provee expresión f_str
    exacta_info = None
    if f_str and f_str.strip():
        try:
            x_sym = sp.Symbol("x")
            f_simb = sp.sympify(f_str, locals={"x": x_sym, "e": sp.E, "pi": sp.pi, "ln": sp.log})
            a_tot, b_tot = float(x[0]), float(x[-1])
            integral_exacta = float(sp.N(sp.integrate(f_simb, (x_sym, a_tot, b_tot))))
            err_abs = abs(integral_exacta - area_total)
            err_pct = (err_abs / abs(integral_exacta) * 100.0) if abs(integral_exacta) > 1e-14 else 0.0
            exacta_info = {
                "integral_exacta": integral_exacta,
                "error_absoluto": err_abs,
                "error_relativo_pct": err_pct,
                "f_latex": sp.latex(f_simb),
            }
        except Exception:
            exacta_info = None

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
        x_nuevos = np.linspace(a, b, int(nuevo_n) + 1)
        h_eff = float((b - a) / int(nuevo_n))
    elif nuevo_h is not None and nuevo_h > 0:
        h_val = float(nuevo_h)
        if h_val >= (b - a):
            x_nuevos = np.array([a, b], dtype=float)
        else:
            pts = [a]
            while pts[-1] + h_val < b - 1e-7:
                pts.append(round(pts[-1] + h_val, 8))
            if abs(pts[-1] - b) > 1e-6:
                pts.append(b)
            x_nuevos = np.array(pts, dtype=float)
        h_eff = h_val
    else:
        return list(x_base), list(y_base), float(xb[1] - xb[0])

    if f_str and f_str.strip():
        xsym = sp.Symbol("x")
        expr = sp.sympify(f_str, locals={"x": xsym, "e": sp.E, "pi": sp.pi, "ln": sp.log})
        fn = sp.lambdify(xsym, expr, modules=["numpy", "math"])
        y_nuevos = np.array([float(np.real(fn(float(xv)))) for xv in x_nuevos], dtype=float)
    else:
        # Interpolación polinómica de referencia sobre los puntos base
        y_nuevos = np.zeros_like(x_nuevos)
        n = len(xb)
        for i in range(n):
            L_i = np.ones_like(x_nuevos)
            for j in range(n):
                if i != j:
                    L_i *= (x_nuevos - xb[j]) / (xb[i] - xb[j])
            y_nuevos += yb[i] * L_i

    return [round(float(v), 6) for v in x_nuevos], [round(float(v), 6) for v in y_nuevos], h_eff
