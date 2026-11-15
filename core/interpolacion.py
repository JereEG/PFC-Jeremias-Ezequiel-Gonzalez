"""
core/interpolacion.py
=====================
Módulo de cálculo puro para Interpolación Polinomial y Aproximación
(Arandú: Métodos Computacionales v3.0).

Incluye:
- Diagnóstico inteligente de equidistancia de nodos (para detectar alteraciones en x_i).
- Remuestreo dinámico de tablas ante cambios de Δh o número de puntos n en vivo.
- Interpolación Lineal Directa e Inversa.
- Interpolación Cuadrática Inversa (con curva parabólica horizontal para Plotly).
- Polinomio de Lagrange (polinomios base L_i(x) y expresión simbólica SymPy).
- Newton-Gregory Ascendente (Δ^k y0) y Descendente (∇^k yn), con evaluaciones parciales
  paso a paso para slider interactivo de orden k.

Diseño: Funciones 100% puras, sin `input()` ni `print()`.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from core._cache_utils import (
    compilar_lambdify_1d,
    construir_lagrange_latex_cached,
    evaluar_fn_vectorizado,
)
from core.catalogo_ejercicios import (
    EXPRESIONES_PRECOMPUTADAS_RAICES,
    LAGRANGE_PRECOMPUTADOS,
)


def diagnosticar_espaciamiento_nodos(x_puntos: list[float] | np.ndarray) -> dict[str, Any]:
    """
    Analiza si las abscisas x están estrictamente equiespaciadas y ordenadas.
    Si el usuario edita un nodo x_i rompiendo la equidistancia, retorna un diagnóstico
    pedagógico explicando por qué Newton-Gregory pierde validez frente a Lagrange.
    """
    x = np.array(x_puntos, dtype=float)
    if len(x) < 2:
        raise ValueError("Se requieren al menos 2 nodos.")
    if len(np.unique(x)) != len(x):
        return {
            "es_equiespaciado": False,
            "ordenado": False,
            "h_nominal": None,
            "pasos_h": np.diff(x).tolist(),
            "mensaje": "Existen valores x_i duplicados en la tabla. Cada abscisa debe ser única.",
        }

    ordenado = bool(np.all(np.diff(x) > 0))
    pasos = np.diff(x)
    h0 = float(pasos[0])
    es_equiesp = bool(ordenado and abs(h0) > 1e-12 and np.allclose(pasos, h0, rtol=1e-4, atol=1e-6))

    if es_equiesp:
        msg = f"Nodos estrictamente equidistantes con paso uniforme h = {h0:.5g}. Aplican Gregory y Lagrange."
    else:
        msg = (
            f"Espaciamiento irregular detectado (hᵢ = {[round(float(v), 4) for v in pasos]}). "
            "Newton-Gregory exige paso h constante y NO es válido aquí; utilice Polinomio de Lagrange."
        )

    return {
        "es_equiespaciado": es_equiesp,
        "ordenado": ordenado,
        "h_nominal": h0 if es_equiesp else None,
        "pasos_h": [float(p) for p in pasos],
        "mensaje": msg,
    }


def regenerar_tabla_interpolacion(
    x_base: list[float],
    y_base: list[float],
    nuevo_h: float | None = None,
    nuevo_n: int | None = None,
    f_expr: str | None = None,
) -> tuple[list[float], list[float], float]:
    """
    Regenera la tabla (x, y) sobre [x_min, x_max] cuando el usuario cambia Δh o la cantidad
    de nodos n en vivo. Si existe `f_expr`, evalúa la función exacta; en caso contrario,
    utiliza el polinomio interpolador único de los datos originales de referencia.
    """
    xb = np.array(x_base, dtype=float)
    yb = np.array(y_base, dtype=float)
    a, b = float(np.min(xb)), float(np.max(xb))
    if b <= a:
        raise ValueError("El intervalo de la tabla debe cumplir b > a.")

    if nuevo_n is not None and nuevo_n >= 2:
        x_nuevos = np.linspace(a, b, int(nuevo_n))
        h_eff = float((b - a) / (int(nuevo_n) - 1))
    elif nuevo_h is not None and nuevo_h > 0:
        h_val = float(nuevo_h)
        if h_val > (b - a):
            h_val = b - a
        x_lista = [a]
        while x_lista[-1] + h_val <= b + 1e-9:
            x_lista.append(round(x_lista[-1] + h_val, 8))
        if abs(x_lista[-1] - b) > 1e-6 and b - x_lista[-1] > 1e-6:
            # Si no divide exacto, añadimos el extremo b (tramo final no equidistante)
            x_lista.append(b)
        x_nuevos = np.array(x_lista, dtype=float)
        h_eff = h_val
    else:
        return list(x_base), list(y_base), float(xb[1] - xb[0])

    if f_expr and f_expr.strip():
        clave = " ".join(f_expr.strip().split())
        if clave in EXPRESIONES_PRECOMPUTADAS_RAICES:
            fn_np = EXPRESIONES_PRECOMPUTADAS_RAICES[clave]["fn_np"]
        else:
            fn_np = compilar_lambdify_1d(clave)
        y_vals = evaluar_fn_vectorizado(fn_np, x_nuevos, limite_abs=None)
        y_nuevos = np.array([v if v is not None else float("nan") for v in y_vals], dtype=float)
    else:
        y_nuevos = evaluar_curva_lagrange(xb, yb, x_nuevos)

    return [round(float(v), 6) for v in x_nuevos], [round(float(v), 6) for v in y_nuevos], h_eff


def interpolar_lineal_directa(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
) -> dict[str, Any]:
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 pares (x, y).")

    orden = np.argsort(x)
    x, y = x[orden], y[orden]
    n = len(x)
    if x_eval <= x[0]:
        idx = 0
    elif x_eval >= x[-1]:
        idx = n - 2
    else:
        idx = 0
        for i in range(n - 1):
            if x[i] <= x_eval <= x[i + 1]:
                idx = i
                break

    x0, x1 = float(x[idx]), float(x[idx + 1])
    y0, y1 = float(y[idx]), float(y[idx + 1])
    dx = x1 - x0
    if abs(dx) < 1e-15:
        raise ZeroDivisionError("Abscisas duplicadas en el tramo.")

    pendiente = (y1 - y0) / dx
    y_interp = y0 + pendiente * (x_eval - x0)
    return {
        "x_eval": float(x_eval),
        "y_interpolado": float(y_interp),
        "indice_intervalo": int(idx),
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "pendiente": float(pendiente), "h": float(dx), "delta_y": float(y1 - y0),
        "es_extrapolacion": bool(x_eval < x[0] or x_eval > x[-1]),
    }


def interpolar_lineal_inversa(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    y_buscado: float,
) -> dict[str, Any]:
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 pares (x, y).")

    n = len(x)
    idx = 0
    for i in range(n - 1):
        if (y[i] <= y_buscado <= y[i + 1]) or (y[i] >= y_buscado >= y[i + 1]):
            idx = i
            break
    else:
        dists = [min(abs(y[i] - y_buscado), abs(y[i + 1] - y_buscado)) for i in range(n - 1)]
        idx = int(np.argmin(dists))

    x0, x1 = float(x[idx]), float(x[idx + 1])
    y0, y1 = float(y[idx]), float(y[idx + 1])
    h = x1 - x0
    delta_y = y1 - y0
    if abs(delta_y) < 1e-15:
        raise ZeroDivisionError("Valores y idénticos en el intervalo seleccionado.")

    dif_y = y_buscado - y0
    x_interp = x0 + dif_y * (h / delta_y)
    return {
        "y_buscado": float(y_buscado),
        "x_resultado": float(x_interp),
        "indice": int(idx),
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "h": float(h), "delta_y": float(delta_y), "diferencia_y": float(dif_y),
    }


def interpolar_cuadratica_inversa(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    y_buscado: float,
) -> dict[str, Any]:
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if len(x) < 3 or len(y) < 3:
        raise ValueError("Se requieren al menos 3 nodos para interpolación cuadrática inversa.")

    n = len(x)
    idx = 0
    encontrado = False
    for i in range(n - 2):
        y_min, y_max = min(y[i], y[i + 1], y[i + 2]), max(y[i], y[i + 1], y[i + 2])
        if y_min <= y_buscado <= y_max:
            idx = i
            encontrado = True
            break
    if not encontrado:
        idx = 0 if abs(y_buscado - y[0]) < abs(y_buscado - y[-1]) else max(0, n - 3)

    x0, x1, x2 = float(x[idx]), float(x[idx + 1]), float(x[idx + 2])
    y0, y1, y2 = float(y[idx]), float(y[idx + 1]), float(y[idx + 2])
    h = x1 - x0
    if abs(h) < 1e-12:
        raise ZeroDivisionError("Paso nulo h.")
    if abs((x2 - x1) - h) > 1e-4 * abs(h):
        raise ValueError(
            "La fórmula Newton-Gregory de 2° grado usada requiere los 3 nodos consecutivos equiespaciados."
        )

    delta_y0 = y1 - y0
    delta2_y0 = y2 - 2.0 * y1 + y0
    diferencia_y = y_buscado - y0

    a_coef = delta2_y0 / (2.0 * h * h)
    b_coef = (delta_y0 / h) - (delta2_y0 / (2.0 * h))
    c_coef = -diferencia_y

    if abs(a_coef) < 1e-12:
        if abs(b_coef) < 1e-12:
            raise ValueError("Datos constantes: no es posible despejar x.")
        u_sol = -c_coef / b_coef
        x_sel = x0 + u_sol
        return {
            "x_resultado": float(x_sel),
            "es_lineal": True,
            "x0": x0, "x1": x1, "x2": x2, "y0": y0, "y1": y1, "y2": y2,
            "h": h, "y_buscado": float(y_buscado),
            "delta_y0": delta_y0, "delta2_y0": delta2_y0,
            "a_coef": a_coef, "b_coef": b_coef, "c_coef": c_coef,
        }

    discriminante = b_coef**2 - 4.0 * a_coef * c_coef
    if discriminante < 0:
        raise ValueError(f"Discriminante negativo ({discriminante:.4e}): no hay solución real.")

    sqrt_disc = math.sqrt(discriminante)
    u1 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    u2 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
    x_sol1, x_sol2 = x0 + u1, x0 + u2

    x_min, x_max = min(x0, x2), max(x0, x2)
    en1, en2 = (x_min <= x_sol1 <= x_max), (x_min <= x_sol2 <= x_max)
    if en1 and not en2:
        x_sel = x_sol1
    elif en2 and not en1:
        x_sel = x_sol2
    else:
        x_sel = x_sol1 if abs(x_sol1 - x1) <= abs(x_sol2 - x1) else x_sol2

    return {
        "x_resultado": float(x_sel),
        "es_lineal": False,
        "x_sol1": float(x_sol1), "x_sol2": float(x_sol2),
        "u1": float(u1), "u2": float(u2), "discriminante": float(discriminante),
        "a_coef": float(a_coef), "b_coef": float(b_coef), "c_coef": float(c_coef),
        "x0": x0, "x1": x1, "x2": x2, "y0": y0, "y1": y1, "y2": y2,
        "h": float(h), "delta_y0": float(delta_y0), "delta2_y0": float(delta2_y0),
        "y_buscado": float(y_buscado), "indice": int(idx),
    }


def calcular_polinomio_base_lagrange(
    x_puntos: list[float] | np.ndarray,
    i: int,
    x_eval: float,
) -> float:
    x = np.array(x_puntos, dtype=float)
    n = len(x)
    num, den = 1.0, 1.0
    for j in range(n):
        if j != i:
            num *= x_eval - x[j]
            den *= x[i] - x[j]
    return float(num / den)


def interpolar_lagrange(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("Se requieren al menos 2 nodos (x, y).")
    if len(np.unique(x)) != n:
        raise ValueError("Existen abscisas x duplicadas.")

    polinomios_base = []
    terminos_detalle = []
    resultado = 0.0
    for i in range(n):
        L_i = calcular_polinomio_base_lagrange(x, i, x_eval)
        term = float(y[i] * L_i)
        resultado += term
        polinomios_base.append(L_i)
        terminos_detalle.append({
            "i": i, "x_i": float(x[i]), "y_i": float(y[i]),
            "L_i": L_i, "termino": term, "acumulado": float(resultado),
        })

    return {
        "x_eval": float(x_eval),
        "y_interpolado": float(resultado),
        "grado": n - 1,
        "puntos": n,
        "polinomios_base": polinomios_base,
        "terminos_detalle": terminos_detalle,
        "duracion": time.perf_counter() - t_ini,
    }


def evaluar_curva_lagrange(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_grid: np.ndarray,
    indice_base: int | None = None,
) -> np.ndarray:
    """Evalúa vectorizadamente en NumPy P(x) o la función base L_k(x) sobre toda la grilla."""
    x = np.asarray(x_puntos, dtype=float)
    y = np.asarray(y_puntos, dtype=float)
    xg = np.asarray(x_grid, dtype=float)
    n = len(x)
    with np.errstate(all="ignore"):
        if indice_base is not None:
            k = int(indice_base)
            L_k = np.ones_like(xg, dtype=float)
            for j in range(n):
                if j != k:
                    L_k *= (xg - x[j]) / (x[k] - x[j])
            return L_k
        y_total = np.zeros_like(xg, dtype=float)
        for i in range(n):
            L_i = np.ones_like(xg, dtype=float)
            for j in range(n):
                if j != i:
                    L_i *= (xg - x[j]) / (x[i] - x[j])
            y_total += y[i] * L_i
        return y_total


def construir_polinomio_simbolico_lagrange(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
) -> dict[str, Any]:
    x_tup = tuple(round(float(v), 10) for v in x_puntos)
    y_tup = tuple(round(float(v), 10) for v in y_puntos)
    if (x_tup, y_tup) in LAGRANGE_PRECOMPUTADOS:
        pre = LAGRANGE_PRECOMPUTADOS[(x_tup, y_tup)]
        return {
            "polinomio_simbolico": None,
            "polinomio_expandido": None,
            "latex_expandido": pre["latex_expandido"],
            "bases_latex": pre["bases_latex"],
            "bases_simbolicas": [],
        }
    meta = construir_lagrange_latex_cached(x_tup, y_tup)
    return {
        "polinomio_simbolico": None,
        "polinomio_expandido": None,
        "latex_expandido": meta["latex_expandido"],
        "bases_latex": meta["bases_latex"],
        "bases_simbolicas": [],
    }


def calcular_diferencias_ascendentes(y_puntos: list[float] | np.ndarray) -> np.ndarray:
    y = np.array(y_puntos, dtype=float)
    n = len(y)
    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            tabla[i, j] = tabla[i + 1, j - 1] - tabla[i, j - 1]
    return tabla


def interpolar_newton_gregory_ascendente(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
    grado_max: int | None = None,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("Se requieren al menos 2 puntos.")

    diag = diagnosticar_espaciamiento_nodos(x)
    if not diag["es_equiespaciado"]:
        raise ValueError(f"Newton-Gregory Ascendente inválido: {diag['mensaje']}")

    h = float(diag["h_nominal"])
    u = float((x_eval - x[0]) / h)
    tabla_diff = calcular_diferencias_ascendentes(y)

    orden_tope = (n - 1) if grado_max is None else min(int(grado_max), n - 1)
    resultado = float(y[0])
    terminos = [{
        "k": 0, "coef_binomial": 1.0, "diferencia": float(y[0]),
        "termino": float(y[0]), "acumulado": resultado,
    }]
    coef_u = 1.0
    for k in range(1, orden_tope + 1):
        coef_u *= (u - (k - 1)) / k
        diff_k = float(tabla_diff[0, k])
        term = float(coef_u * diff_k)
        resultado += term
        terminos.append({
            "k": k, "coef_binomial": float(coef_u), "diferencia": diff_k,
            "termino": term, "acumulado": float(resultado),
        })

    return {
        "x_eval": float(x_eval),
        "y_interpolado": float(resultado),
        "u": u, "h": h, "x0": float(x[0]), "y0": float(y[0]),
        "tabla_diferencias": tabla_diff.tolist(),
        "terminos": terminos,
        "duracion": time.perf_counter() - t_ini,
    }


def calcular_diferencias_descendentes(y_puntos: list[float] | np.ndarray) -> np.ndarray:
    y = np.array(y_puntos, dtype=float)
    n = len(y)
    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            tabla[i, j] = tabla[i, j - 1] - tabla[i - 1, j - 1]
    return tabla


def interpolar_newton_gregory_descendente(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
    grado_max: int | None = None,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("Se requieren al menos 2 puntos.")

    diag = diagnosticar_espaciamiento_nodos(x)
    if not diag["es_equiespaciado"]:
        raise ValueError(f"Newton-Gregory Descendente inválido: {diag['mensaje']}")

    h = float(diag["h_nominal"])
    v = float((x_eval - x[-1]) / h)
    tabla_diff = calcular_diferencias_descendentes(y)

    orden_tope = (n - 1) if grado_max is None else min(int(grado_max), n - 1)
    resultado = float(y[-1])
    terminos = [{
        "k": 0, "coef_binomial": 1.0, "diferencia": float(y[-1]),
        "termino": float(y[-1]), "acumulado": resultado,
    }]
    coef_v = 1.0
    for k in range(1, orden_tope + 1):
        coef_v *= (v + (k - 1)) / k
        diff_k = float(tabla_diff[n - 1, k])
        term = float(coef_v * diff_k)
        resultado += term
        terminos.append({
            "k": k, "coef_binomial": float(coef_v), "diferencia": diff_k,
            "termino": term, "acumulado": float(resultado),
        })

    return {
        "x_eval": float(x_eval),
        "y_interpolado": float(resultado),
        "v": v, "h": h, "xn": float(x[-1]), "yn": float(y[-1]),
        "tabla_diferencias": tabla_diff.tolist(),
        "terminos": terminos,
        "duracion": time.perf_counter() - t_ini,
    }


def evaluar_curva_gregory_parcial(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_grid: np.ndarray,
    grado: int,
    modo: str = "ascendente",
) -> np.ndarray:
    """Evalúa vectorizadamente el polinomio de Newton-Gregory truncado hasta `grado`."""
    x = np.asarray(x_puntos, dtype=float)
    y = np.asarray(y_puntos, dtype=float)
    xg = np.asarray(x_grid, dtype=float)
    diag = diagnosticar_espaciamiento_nodos(x)
    if not diag["es_equiespaciado"]:
        raise ValueError(f"Newton-Gregory inválido: {diag['mensaje']}")
    h = float(diag["h_nominal"])
    n = len(x)
    max_k = min(max(int(grado), 0), n - 1)
    tabla = calcular_diferencias_ascendentes(y)

    with np.errstate(all="ignore"):
        if modo == "ascendente":
            u = (xg - x[0]) / h
            y_acc = np.full_like(xg, float(tabla[0, 0]), dtype=float)
            coef = np.ones_like(xg, dtype=float)
            for k in range(1, max_k + 1):
                coef = coef * (u - (k - 1)) / k
                y_acc = y_acc + coef * float(tabla[0, k])
            return y_acc

        v = (xg - x[-1]) / h
        y_acc = np.full_like(xg, float(tabla[n - 1, 0]), dtype=float)
        coef = np.ones_like(xg, dtype=float)
        for k in range(1, max_k + 1):
            coef = coef * (v + (k - 1)) / k
            y_acc = y_acc + coef * float(tabla[n - 1 - k, k])
        return y_acc
