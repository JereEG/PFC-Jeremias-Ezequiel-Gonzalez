"""
core/lab6_integracion.py
========================
Módulo de cálculo puro para Diferenciación e Integración Numérica (Laboratorio 6).

Incluye:
- Diferenciación Numérica por fórmulas de Newton-Gregory:
    * 1° y 2° derivada con diferencias ascendentes (progresivas, al inicio de tabla).
    * 1° y 2° derivada con diferencias descendentes (retrógradas, al final de tabla).
- Regla del Trapecio:
    * Trapecio simple y compuesto: A = (h/2) * (E + 2P + 2I).
    * Cota teórica de error O(h^2): Cota <= ((b - a)^3 / (12*n^2)) * max|f''(x)|.
- Reglas de Simpson:
    * Simpson 1/3 compuesto: A = (h/3) * (E + 2P + 4I) (requiere n par de intervalos).
    * Simpson 3/8 compuesto: A = (3h/8) * sum(y_0 + 3y_1 + 3y_2 + y_3) (n múltiplo de 3).
    * Simpson Combinado: combina 3/8 en los primeros 3 intervalos y 1/3 en el resto.
- Comparación con Integrales Exactas de SymPy y cálculo de cotas analíticas.

Diseño: Funciones 100% puras, sin `input()` ni `print()`, con retornos estructurados.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable
import numpy as np
import sympy as sp


# ==============================================================================
# 1. DIFERENCIACIÓN NUMÉRICA (NEWTON-GREGORY)
# ==============================================================================

def validar_equiespaciado(x_values: list[float] | np.ndarray) -> float:
    """Valida que los puntos x estén equiespaciados y retorna el paso h."""
    x = np.array(x_values, dtype=float)
    if len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos.")
    h_arr = np.diff(x)
    h = float(h_arr[0])
    if abs(h) < 1e-14:
        raise ZeroDivisionError("El espaciamiento h entre puntos es cero.")
    if not np.allclose(h_arr, h, rtol=1e-5):
        raise ValueError("Los puntos x deben estar equiespaciados.")
    return h


def derivada_newton_gregory_ascendente(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    idx_punto: int = 0,
    orden_derivada: int = 1,
) -> dict[str, Any]:
    """
    Calcula la derivada numérica en el punto x[idx_punto] mediante diferencias progresivas:
    - 1ra derivada: f'(x0) ≈ (1/h) * [ Delta f0 - (1/2)*Delta^2 f0 + (1/3)*Delta^3 f0 - (1/4)*Delta^4 f0 ]
    - 2da derivada: f''(x0) ≈ (1/h^2) * [ Delta^2 f0 - Delta^3 f0 + (11/12)*Delta^4 f0 ]
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

    # Matriz de diferencias progresivas
    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            tabla[i, j] = tabla[i + 1, j - 1] - tabla[i, j - 1]

    terminos_detalle = []
    if orden_derivada == 1:
        # Coeficientes: +1, -1/2, +1/3, -1/4
        signos = [1.0, -0.5, 1.0 / 3.0, -0.25]
        suma_diff = 0.0
        for k_idx, coef in enumerate(signos, start=1):
            if k_idx < n and (idx_punto + k_idx) <= n:
                diff_val = float(tabla[idx_punto, k_idx])
                aporte = coef * diff_val
                suma_diff += aporte
                terminos_detalle.append({
                    "orden_diff": k_idx,
                    "coeficiente": coef,
                    "diferencia": diff_val,
                    "aporte": aporte,
                })
        derivada = suma_diff / h
    else:
        # Segunda derivada: [ Delta^2 f0 - Delta^3 f0 + (11/12)*Delta^4 f0 ] / h^2
        if n - idx_punto < 3:
            raise ValueError("Se requieren al menos 3 puntos hacia adelante para la 2da derivada.")
        coefs = [(2, 1.0), (3, -1.0), (4, 11.0 / 12.0)]
        suma_diff = 0.0
        for k_idx, coef in coefs:
            if k_idx < n and (idx_punto + k_idx) <= n:
                diff_val = float(tabla[idx_punto, k_idx])
                aporte = coef * diff_val
                suma_diff += aporte
                terminos_detalle.append({
                    "orden_diff": k_idx,
                    "coeficiente": coef,
                    "diferencia": diff_val,
                    "aporte": aporte,
                })
        derivada = suma_diff / (h**2)

    return {
        "x_eval": float(x[idx_punto]),
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
) -> dict[str, Any]:
    """
    Calcula la derivada numérica en el punto x[idx_punto] mediante diferencias retrógradas:
    - 1ra derivada: f'(xn) ≈ (1/h) * [ nabla fn + (1/2)*nabla^2 fn + (1/3)*nabla^3 fn + (1/4)*nabla^4 fn ]
    - 2da derivada: f''(xn) ≈ (1/h^2) * [ nabla^2 fn + nabla^3 fn + (11/12)*nabla^4 fn ]
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n = len(y)

    if idx_punto < 0:
        idx_punto = n + idx_punto

    h = validar_equiespaciado(x)

    # Matriz de diferencias retrógradas
    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            tabla[i, j] = tabla[i, j - 1] - tabla[i - 1, j - 1]

    terminos_detalle = []
    if orden_derivada == 1:
        # Coeficientes: +1, +1/2, +1/3, +1/4
        coefs = [1.0, 0.5, 1.0 / 3.0, 0.25]
        suma_diff = 0.0
        for k_idx, coef in enumerate(coefs, start=1):
            if k_idx < n and idx_punto >= k_idx:
                diff_val = float(tabla[idx_punto, k_idx])
                aporte = coef * diff_val
                suma_diff += aporte
                terminos_detalle.append({
                    "orden_diff": k_idx,
                    "coeficiente": coef,
                    "diferencia": diff_val,
                    "aporte": aporte,
                })
        derivada = suma_diff / h
    else:
        if idx_punto < 2:
            raise ValueError("Se requieren al menos 3 puntos hacia atrás para la 2da derivada retrógrada.")
        coefs_2da = [(2, 1.0), (3, 1.0), (4, 11.0 / 12.0)]
        suma_diff = 0.0
        for k_idx, coef in coefs_2da:
            if k_idx < n and idx_punto >= k_idx:
                diff_val = float(tabla[idx_punto, k_idx])
                aporte = coef * diff_val
                suma_diff += aporte
                terminos_detalle.append({
                    "orden_diff": k_idx,
                    "coeficiente": coef,
                    "diferencia": diff_val,
                    "aporte": aporte,
                })
        derivada = suma_diff / (h**2)

    return {
        "x_eval": float(x[idx_punto]),
        "derivada": float(derivada),
        "orden_derivada": orden_derivada,
        "h": h,
        "indice_usado": idx_punto,
        "tabla_diferencias": tabla.tolist(),
        "terminos_detalle": terminos_detalle,
    }


# ==============================================================================
# 2. REGLA DEL TRAPECIO
# ==============================================================================

def calcular_trapecio_compuesto(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    """
    Calcula la integral numérica mediante la Regla de los Trapecios Compuesta:
        A = (h / 2) * [ E + 2*P + 2*I ]
    donde:
        E = y[0] + y[-1]  (Extremos)
        P = suma de y[i] para índices pares intermedios (2, 4, 6, ...)
        I = suma de y[i] para índices impares (1, 3, 5, ...)
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n_puntos = len(y)
    if n_puntos < 2:
        raise ValueError("Se requieren al menos 2 puntos para aplicar Trapecio.")

    h = validar_equiespaciado(x)
    n_intervalos = n_puntos - 1

    # Extremos (E)
    E = float(y[0] + y[-1])
    extremos_idx = [0, n_puntos - 1]

    # Pares (P): índices 2, 4, ... < n_puntos - 1
    pares_idx = list(range(2, n_puntos - 1, 2))
    P = float(np.sum(y[pares_idx])) if pares_idx else 0.0

    # Impares (I): índices 1, 3, ... < n_puntos - 1
    impares_idx = list(range(1, n_puntos - 1, 2))
    I = float(np.sum(y[impares_idx])) if impares_idx else 0.0

    area = (h / 2.0) * (E + 2.0 * P + 2.0 * I)

    return {
        "metodo": "Regla del Trapecio Compuesta",
        "area": float(area),
        "h": h,
        "n_intervalos": n_intervalos,
        "n_puntos": n_puntos,
        "E": E,
        "P": P,
        "I": I,
        "formula": f"({h:.5f}/2) * ({E:.5f} + 2*{P:.5f} + 2*{I:.5f})",
        "extremos_idx": extremos_idx,
        "pares_idx": pares_idx,
        "impares_idx": impares_idx,
    }


def cota_error_trapecio(
    a: float,
    b: float,
    n_intervalos: int,
    max_d2f: float,
) -> float:
    """Calcula la cota de error teórica de Trapecio: Error <= ((b - a)^3 / (12 * n^2)) * max|f''(x)|."""
    if n_intervalos <= 0:
        raise ValueError("n_intervalos debe ser mayor a cero.")
    return float(((b - a) ** 3 / (12.0 * (n_intervalos**2))) * max_d2f)


# ==============================================================================
# 3. REGLAS DE SIMPSON (1/3, 3/8 Y COMBINADO)
# ==============================================================================

def calcular_simpson_1_3(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    """
    Calcula la integral mediante la Regla de Simpson 1/3 Compuesta:
        A = (h / 3) * [ E + 2*P + 4*I ]
    Requiere un número par de intervalos (número impar de puntos).
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n_puntos = len(y)
    n_intervalos = n_puntos - 1

    if n_puntos < 3:
        raise ValueError("Simpson 1/3 requiere al menos 3 puntos.")
    if n_intervalos % 2 != 0:
        raise ValueError(f"Simpson 1/3 requiere un número par de intervalos. Se recibieron {n_intervalos} intervalos ({n_puntos} puntos).")

    h = validar_equiespaciado(x)

    E = float(y[0] + y[-1])
    extremos_idx = [0, n_puntos - 1]

    # Pares intermedios: 2, 4, ...
    pares_idx = list(range(2, n_puntos - 1, 2))
    P = float(np.sum(y[pares_idx])) if pares_idx else 0.0

    # Impares intermedios: 1, 3, ...
    impares_idx = list(range(1, n_puntos - 1, 2))
    I = float(np.sum(y[impares_idx])) if impares_idx else 0.0

    area = (h / 3.0) * (E + 2.0 * P + 4.0 * I)

    return {
        "metodo": "Simpson 1/3",
        "area": float(area),
        "h": h,
        "n_intervalos": n_intervalos,
        "n_puntos": n_puntos,
        "E": E,
        "P": P,
        "I": I,
        "formula": f"({h:.5f}/3) * ({E:.5f} + 2*{P:.5f} + 4*{I:.5f})",
        "extremos_idx": extremos_idx,
        "pares_idx": pares_idx,
        "impares_idx": impares_idx,
    }


def calcular_simpson_3_8(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    """
    Calcula la integral mediante la Regla de Simpson 3/8:
        Por bloque de 4 puntos (3 intervalos): A_k = (3h / 8) * (y0 + 3*y1 + 3*y2 + y3).
    Requiere que el número total de intervalos sea múltiplo de 3.
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n_puntos = len(y)
    n_intervalos = n_puntos - 1

    if n_puntos < 4:
        raise ValueError("Simpson 3/8 requiere al menos 4 puntos.")
    if n_intervalos % 3 != 0:
        raise ValueError(f"Simpson 3/8 requiere que el número de intervalos sea múltiplo de 3. Se recibieron {n_intervalos} intervalos.")

    h = validar_equiespaciado(x)

    area_total = 0.0
    grupos = []
    for i in range(0, n_intervalos, 3):
        y0, y1, y2, y3 = y[i], y[i + 1], y[i + 2], y[i + 3]
        a_bloque = (3.0 * h / 8.0) * (y0 + 3.0 * y1 + 3.0 * y2 + y3)
        area_total += a_bloque
        grupos.append({
            "grupo": len(grupos) + 1,
            "indices": [i, i + 1, i + 2, i + 3],
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
    }


def calcular_simpson_combinado(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    """
    Calcula la integral combinando Simpson 1/3 y Simpson 3/8 cuando la cantidad de intervalos
    no es múltiplo de 2 ni de 3 por sí sola (ej. n = 5 intervalos, 6 puntos).
    Aplica Simpson 3/8 a los primeros 3 intervalos y Simpson 1/3 a los restantes.
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n_puntos = len(y)
    n_intervalos = n_puntos - 1

    # Estrategia según paridad de intervalos
    if n_intervalos % 2 == 0:
        return calcular_simpson_1_3(x, y)
    elif n_intervalos % 3 == 0:
        return calcular_simpson_3_8(x, y)
    elif n_intervalos >= 5 and (n_intervalos - 3) % 2 == 0:
        # Partición: 3/8 en [0..3] (4 puntos), 1/3 en [3..end] (resto par)
        res_38 = calcular_simpson_3_8(x[:4], y[:4])
        res_13 = calcular_simpson_1_3(x[3:], y[3:])
        area_comb = res_38["area"] + res_13["area"]
        return {
            "metodo": "Simpson Combinado (3/8 + 1/3)",
            "area": float(area_comb),
            "area_3_8": res_38["area"],
            "area_1_3": res_13["area"],
            "intervalos_3_8": 3,
            "intervalos_1_3": n_intervalos - 3,
            "n_intervalos_total": n_intervalos,
            "detalles_3_8": res_38,
            "detalles_1_3": res_13,
        }
    else:
        raise ValueError(f"No es posible combinar Simpson con {n_intervalos} intervalos.")


def cota_error_simpson_1_3(
    a: float,
    b: float,
    n_intervalos: int,
    max_d4f: float,
) -> float:
    """Calcula la cota de error teórica de Simpson 1/3: Error <= ((b - a)^5 / (180 * n^4)) * max|f^(4)(x)|."""
    if n_intervalos <= 0:
        raise ValueError("n_intervalos debe ser mayor a cero.")
    return float(((b - a) ** 5 / (180.0 * (n_intervalos**4))) * max_d4f)


# ==============================================================================
# 4. INTEGRACIÓN DIRECTA DESDE FUNCIÓN SIMBÓLICA CON VERIFICACIÓN ANALÍTICA
# ==============================================================================

def integrar_funcion_simbolica(
    f_str: str,
    a: float,
    b: float,
    n_intervalos: int,
    metodo: str = "simpson_combinado",
) -> dict[str, Any]:
    """
    Toma una expresión matemática f(x), genera la discretización en [a, b],
    calcula la integral aproximada por el método especificado, y la contrasta contra
    la integral exacta calculada analíticamente con SymPy.
    """
    x_sym = sp.Symbol("x")
    try:
        f_simb = sp.sympify(f_str)
        f_num = sp.lambdify(x_sym, f_simb, "numpy")
    except Exception as e:
        raise ValueError(f"Expresión matemática inválida '{f_str}': {e}")

    x_pts = np.linspace(a, b, n_intervalos + 1)
    y_pts = f_num(x_pts)

    if metodo in ("trapecio", "trapecios"):
        res_aprox = calcular_trapecio_compuesto(x_pts, y_pts)
    elif metodo in ("simpson13", "simpson_1_3"):
        res_aprox = calcular_simpson_1_3(x_pts, y_pts)
    elif metodo in ("simpson38", "simpson_3_8"):
        res_aprox = calcular_simpson_3_8(x_pts, y_pts)
    else:
        res_aprox = calcular_simpson_combinado(x_pts, y_pts)

    # Cálculo analítico exacto con SymPy
    try:
        integral_analitica = float(sp.integrate(f_simb, (x_sym, a, b)).evalf())
    except Exception:
        integral_analitica = None

    if integral_analitica is not None:
        error_abs = abs(integral_analitica - res_aprox["area"])
        error_rel_pct = (error_abs / abs(integral_analitica) * 100.0) if abs(integral_analitica) > 1e-15 else 0.0
    else:
        error_abs = None
        error_rel_pct = None

    return {
        "funcion_str": f_str,
        "a": a,
        "b": b,
        "n_intervalos": n_intervalos,
        "x_puntos": x_pts.tolist(),
        "y_puntos": y_pts.tolist(),
        "resultado_aproximado": res_aprox,
        "integral_exacta": integral_analitica,
        "error_absoluto": error_abs,
        "error_relativo_pct": error_rel_pct,
    }

