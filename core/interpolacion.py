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
- Fórmula de Gauss (Central Progresiva y Central Regresiva), con trazado zig-zag en
  diferencias finitas, recomendación según posición de x* y evaluación parcial k.

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
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Se requieren al menos 2 nodos.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Todas las abscisas x_i deben ser números finitos.")
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
    """Estima y(x*) trazando la recta secante sobre el tramo vecino [x_i, x_{i+1}]."""
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 pares (x, y) de igual longitud.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores de la tabla (x, y) deben ser números finitos.")
    if not np.isfinite(x_eval):
        raise ValueError("El valor x* a evaluar debe ser un número finito.")
    if len(np.unique(x)) != len(x):
        raise ValueError("Existen abscisas x_i duplicadas en la tabla; cada x_i debe ser única.")

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
    if not math.isfinite(y_interp):
        raise ValueError("Desbordamiento numérico al evaluar la interpolación lineal.")
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
    """Despeja x(y*) invirtiendo la recta secante sobre el tramo que encierra y*."""
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 pares (x, y) de igual longitud.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores de la tabla (x, y) deben ser números finitos.")
    if not np.isfinite(y_buscado):
        raise ValueError("El valor y* buscado debe ser un número finito.")
    if len(np.unique(x)) != len(x):
        raise ValueError("Existen abscisas x_i duplicadas en la tabla; cada x_i debe ser única.")

    orden = np.argsort(x)
    x, y = x[orden], y[orden]
    n = len(x)
    idx = 0
    es_extrap = False
    for i in range(n - 1):
        if (y[i] <= y_buscado <= y[i + 1]) or (y[i] >= y_buscado >= y[i + 1]):
            idx = i
            break
    else:
        es_extrap = True
        dists = [min(abs(y[i] - y_buscado), abs(y[i + 1] - y_buscado)) for i in range(n - 1)]
        idx = int(np.argmin(dists))

    x0, x1 = float(x[idx]), float(x[idx + 1])
    y0, y1 = float(y[idx]), float(y[idx + 1])
    h = x1 - x0
    delta_y = y1 - y0
    if abs(delta_y) < 1e-15:
        raise ZeroDivisionError("Valores y idénticos en el intervalo seleccionado (tramo horizontal no invertible).")

    dif_y = y_buscado - y0
    x_interp = x0 + dif_y * (h / delta_y)
    if not math.isfinite(x_interp):
        raise ValueError("Desbordamiento numérico al despejar la interpolación inversa.")
    return {
        "y_buscado": float(y_buscado),
        "x_resultado": float(x_interp),
        "indice": int(idx),
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "h": float(h), "delta_y": float(delta_y), "diferencia_y": float(dif_y),
        "pendiente": float(delta_y / h),
        "es_extrapolacion": es_extrap,
    }


def interpolar_cuadratica_inversa(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    y_buscado: float,
) -> dict[str, Any]:
    """Despeja x(y*) resolviendo la parábola interpolante de grado 2 sobre 3 nodos equidistantes."""
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise ValueError("Se requieren al menos 3 pares (x, y) de igual longitud para interpolación cuadrática inversa.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores de la tabla (x, y) deben ser números finitos.")
    if not np.isfinite(y_buscado):
        raise ValueError("La ordenada objetivo y* debe ser un número finito.")
    if len(np.unique(x)) != len(x):
        raise ValueError("Existen abscisas x_i duplicadas en la tabla; cada x_i debe ser única.")

    orden = np.argsort(x)
    x, y = x[orden], y[orden]
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
        if not math.isfinite(x_sel):
            raise ValueError("Desbordamiento numérico al despejar el caso lineal degenerado.")
        return {
            "x_resultado": float(x_sel),
            "es_lineal": True,
            "es_extrapolacion": not encontrado,
            "x0": x0, "x1": x1, "x2": x2, "y0": y0, "y1": y1, "y2": y2,
            "h": h, "y_buscado": float(y_buscado),
            "delta_y0": delta_y0, "delta2_y0": delta2_y0,
            "a_coef": a_coef, "b_coef": b_coef, "c_coef": c_coef,
            "indice": int(idx),
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

    if not math.isfinite(x_sel):
        raise ValueError("Desbordamiento numérico al calcular las raíces cuadráticas.")

    return {
        "x_resultado": float(x_sel),
        "es_lineal": False,
        "es_extrapolacion": not encontrado,
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
    """Evalúa el i-ésimo polinomio base L_i(x*) de Lagrange en un conjunto de abscisas únicas."""
    x = np.array(x_puntos, dtype=float)
    if not np.all(np.isfinite(x)) or not math.isfinite(float(x_eval)):
        raise ValueError("Las abscisas x_i y el punto x* deben ser números finitos.")
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
    """Construye e interpola el polinomio único de Lagrange sobre nodos libres (x_i, y_i)."""
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 nodos (x, y) de igual longitud.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores (x, y) deben ser números finitos.")
    if not np.isfinite(x_eval):
        raise ValueError("El punto x* a evaluar debe ser un número finito.")
    n = len(x)
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

    if not math.isfinite(resultado):
        raise ValueError("Desbordamiento numérico al evaluar el polinomio de Lagrange.")

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
    """Genera la representación LaTeX expandida del polinomio de Lagrange (usando caché si existe)."""
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 nodos (x, y) de igual longitud.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores (x, y) deben ser números finitos.")
    if len(np.unique(x)) != len(x):
        raise ValueError("Existen abscisas x duplicadas.")

    x_tup = tuple(round(float(v), 10) for v in x)
    y_tup = tuple(round(float(v), 10) for v in y)
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
    """Evalúa el polinomio de Newton-Gregory Ascendente usando diferencias progresivas Δᵏy₀."""
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos de igual longitud.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores (x, y) deben ser números finitos.")
    if not math.isfinite(float(x_eval)):
        raise ValueError("El punto x* a evaluar debe ser un número finito.")
    if grado_max is not None and int(grado_max) < 0:
        raise ValueError("El grado de truncamiento no puede ser negativo.")
    n = len(x)

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

    if not math.isfinite(resultado):
        raise ValueError("Desbordamiento numérico al evaluar Newton-Gregory Ascendente.")

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
    """Evalúa el polinomio de Newton-Gregory Descendente usando diferencias retrógradas ∇ᵏyₙ."""
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos de igual longitud.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores (x, y) deben ser números finitos.")
    if not math.isfinite(float(x_eval)):
        raise ValueError("El punto x* a evaluar debe ser un número finito.")
    if grado_max is not None and int(grado_max) < 0:
        raise ValueError("El grado de truncamiento no puede ser negativo.")
    n = len(x)

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

    if not math.isfinite(resultado):
        raise ValueError("Desbordamiento numérico al evaluar Newton-Gregory Descendente.")

    return {
        "x_eval": float(x_eval),
        "y_interpolado": float(resultado),
        "v": v, "h": h, "xn": float(x[-1]), "yn": float(y[-1]),
        "tabla_diferencias": tabla_diff.tolist(),
        "terminos": terminos,
        "duracion": time.perf_counter() - t_ini,
    }


_MODOS_EQUIDISTANTES = (
    "ascendente",
    "descendente",
    "gauss_progresiva",
    "gauss_regresiva",
)


def _indice_ancla_modo(n: int, modo: str) -> int:
    """Retorna el índice del nodo ancla x_ancla para cada fórmula sobre diferencias finitas."""
    if modo == "ascendente":
        return 0
    if modo == "descendente":
        return n - 1
    if modo == "gauss_progresiva":
        return (n - 1) // 2
    if modo == "gauss_regresiva":
        return n // 2
    raise ValueError(f"Modo equidistante desconocido: {modo!r}")


def _fila_diferencia_modo(n: int, k: int, modo: str) -> int:
    """Retorna la fila i de la tabla ascendente Δ^k y_i usada en el orden k según el método."""
    if modo == "ascendente":
        return 0
    if modo == "descendente":
        return n - 1 - k
    if modo == "gauss_progresiva":
        return ((n - 1) // 2) - (k // 2)
    if modo == "gauss_regresiva":
        return (n // 2) - ((k + 1) // 2)
    raise ValueError(f"Modo equidistante desconocido: {modo!r}")


def _desplazamiento_factor_u(j: int, modo: str) -> int:
    """Retorna el desplazamiento entero d_j tal que el factor j-ésimo (j >= 0) es (u + d_j)/(j + 1)."""
    if modo == "ascendente":
        return -j
    if modo == "descendente":
        return j
    if modo in ("gauss_progresiva", "gauss_regresiva"):
        if j == 0:
            return 0
        magnitud = ((j + 1) // 2) if j % 2 == 1 else (j // 2)
        if modo == "gauss_progresiva":
            return -magnitud if j % 2 == 1 else magnitud
        return magnitud if j % 2 == 1 else -magnitud
    raise ValueError(f"Modo equidistante desconocido: {modo!r}")


def nodos_usados_equidistante(n: int, grado: int, modo: str) -> list[int]:
    """Retorna los índices de los nodos {x_i} que intervienen en P_k(x) al truncar en `grado`."""
    if n < 2:
        return list(range(n))
    k = min(max(int(grado), 0), n - 1)
    fila = _fila_diferencia_modo(n, k, modo)
    return list(range(fila, fila + k + 1))


def recomendar_metodo_equidistante(
    x_puntos: list[float] | np.ndarray,
    x_eval: float,
) -> dict[str, Any]:
    """Sugiere el método óptimo (N-G Asc/Desc o Gauss Prog/Reg) según la posición relativa de x*."""
    x = np.asarray(x_puntos, dtype=float)
    n = len(x)
    if x.ndim != 1 or n < 2:
        raise ValueError("Se requieren al menos 2 nodos para recomendar un método.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Todas las abscisas x deben ser números finitos.")
    if not math.isfinite(float(x_eval)):
        raise ValueError("El punto x* a evaluar debe ser un número finito.")
    span = float(x[-1] - x[0])
    if abs(span) < 1e-15:
        raise ValueError("El rango de abscisas no puede ser nulo.")

    idx_c = (n - 1) // 2
    idx_reg = n // 2
    h = float(x[1] - x[0])
    u_central = float((x_eval - x[idx_c]) / h) if abs(h) > 1e-15 else 0.0
    u_reg = float((x_eval - x[idx_reg]) / h) if abs(h) > 1e-15 else u_central
    t_norm = float((x_eval - x[0]) / span)

    if t_norm < 0.25:
        return {
            "modo": "ascendente",
            "zona": "extremo_inicial",
            "anchor_idx": 0,
            "x_anchor": float(x[0]),
            "u_central": u_central,
            "t_norm": t_norm,
            "motivo": (
                f"x* = {x_eval:.4g} está cerca del inicio de la tabla (t = {t_norm:.2f}). "
                "Newton-Gregory Ascendente anclado en x₀ minimiza el error de truncamiento."
            ),
        }
    if t_norm > 0.75:
        return {
            "modo": "descendente",
            "zona": "extremo_final",
            "anchor_idx": n - 1,
            "x_anchor": float(x[-1]),
            "u_central": u_central,
            "t_norm": t_norm,
            "motivo": (
                f"x* = {x_eval:.4g} está cerca del final de la tabla (t = {t_norm:.2f}). "
                "Newton-Gregory Descendente anclado en xₙ minimiza el error de truncamiento."
            ),
        }
    elige_prog = (u_central >= 0.0) if idx_c == idx_reg else (abs(u_central) <= abs(u_reg))
    if elige_prog:
        return {
            "modo": "gauss_progresiva",
            "zona": "centro_derecha",
            "anchor_idx": idx_c,
            "x_anchor": float(x[idx_c]),
            "u_central": u_central,
            "t_norm": t_norm,
            "motivo": (
                f"x* = {x_eval:.4g} cae en la zona central derecha respecto a x{idx_c} = {x[idx_c]:.4g} "
                f"(u = {u_central:+.3f} ≥ 0). Gauss Progresiva da el primer paso hacia abajo en zig-zag."
            ),
        }
    return {
        "modo": "gauss_regresiva",
        "zona": "centro_izquierda",
        "anchor_idx": idx_reg,
        "x_anchor": float(x[idx_reg]),
        "u_central": u_reg,
        "t_norm": t_norm,
        "motivo": (
            f"x* = {x_eval:.4g} cae en la zona central izquierda respecto a x{idx_reg} = {x[idx_reg]:.4g} "
            f"(u = {u_reg:+.3f} < 0). Gauss Regresiva da el primer paso hacia arriba en zig-zag."
        ),
    }


def _interpolar_gauss_central(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
    modo: str,
    grado_max: int | None = None,
) -> dict[str, Any]:
    """Evalúa Fórmula de Gauss Central (Progresiva o Regresiva) paso a paso sobre malla equidistante."""
    if modo not in ("gauss_progresiva", "gauss_regresiva"):
        raise ValueError(f"Modo Gauss central inválido: {modo!r}")
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos válidos de igual longitud.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores (x, y) deben ser números finitos.")
    if not math.isfinite(float(x_eval)):
        raise ValueError("El punto x* a evaluar debe ser un número finito.")
    if grado_max is not None and int(grado_max) < 0:
        raise ValueError("El grado de truncamiento no puede ser negativo.")
    n = len(x)

    diag = diagnosticar_espaciamiento_nodos(x)
    if not diag["es_equiespaciado"]:
        nombre_m = "Gauss Progresiva" if modo == "gauss_progresiva" else "Gauss Regresiva"
        raise ValueError(f"{nombre_m} inválida: {diag['mensaje']}")

    h = float(diag["h_nominal"])
    anchor_idx = _indice_ancla_modo(n, modo)
    u = float((x_eval - x[anchor_idx]) / h)
    tabla_diff = calcular_diferencias_ascendentes(y)

    orden_tope = (n - 1) if grado_max is None else min(int(grado_max), n - 1)
    y0_c = float(tabla_diff[anchor_idx, 0])
    resultado = y0_c
    terminos: list[dict[str, Any]] = [
        {
            "k": 0,
            "delta_factor": None,
            "coef_binomial": 1.0,
            "diferencia": y0_c,
            "fila_tabla": anchor_idx,
            "termino": y0_c,
            "acumulado": resultado,
        }
    ]
    ruta_celdas: list[dict[str, Any]] = [
        {"k": 0, "fila": anchor_idx, "col": 0, "valor": y0_c}
    ]

    coef_u = 1.0
    for k in range(1, orden_tope + 1):
        delta = _desplazamiento_factor_u(k - 1, modo)
        coef_u *= (u + delta) / k
        fila = _fila_diferencia_modo(n, k, modo)
        diff_k = float(tabla_diff[fila, k])
        term = float(coef_u * diff_k)
        resultado += term
        terminos.append(
            {
                "k": k,
                "delta_factor": int(delta),
                "coef_binomial": float(coef_u),
                "diferencia": diff_k,
                "fila_tabla": int(fila),
                "termino": term,
                "acumulado": float(resultado),
            }
        )
        ruta_celdas.append({"k": k, "fila": int(fila), "col": k, "valor": diff_k})

    if not math.isfinite(resultado):
        raise ValueError("Desbordamiento numérico al evaluar Fórmula de Gauss.")

    return {
        "x_eval": float(x_eval),
        "y_interpolado": float(resultado),
        "u": u,
        "h": h,
        "modo": modo,
        "anchor_idx": int(anchor_idx),
        "x0_central": float(x[anchor_idx]),
        "y0_central": y0_c,
        "ruta_celdas": ruta_celdas,
        "nodos_usados_idx": nodos_usados_equidistante(n, orden_tope, modo),
        "tabla_diferencias": tabla_diff.tolist(),
        "terminos": terminos,
        "duracion": time.perf_counter() - t_ini,
    }


def interpolar_gauss_progresiva(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
    grado_max: int | None = None,
) -> dict[str, Any]:
    """Interpolación mediante Fórmula de Gauss Progresiva (primer paso hacia abajo en zig-zag)."""
    return _interpolar_gauss_central(x_puntos, y_puntos, x_eval, "gauss_progresiva", grado_max=grado_max)


def interpolar_gauss_regresiva(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
    grado_max: int | None = None,
) -> dict[str, Any]:
    """Interpolación mediante Fórmula de Gauss Regresiva (primer paso hacia arriba en zig-zag)."""
    return _interpolar_gauss_central(x_puntos, y_puntos, x_eval, "gauss_regresiva", grado_max=grado_max)


def evaluar_curva_gregory_parcial(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_grid: np.ndarray,
    grado: int,
    modo: str = "ascendente",
) -> np.ndarray:
    """Evalúa vectorizadamente Newton-Gregory o Gauss (Central) truncado hasta `grado`."""
    if modo not in _MODOS_EQUIDISTANTES:
        raise ValueError(f"Modo equidistante inválido: {modo!r}")
    x = np.asarray(x_puntos, dtype=float)
    y = np.asarray(y_puntos, dtype=float)
    xg = np.asarray(x_grid, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 nodos de igual longitud.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Todos los valores (x, y) deben ser números finitos.")
    if int(grado) < 0:
        raise ValueError("El grado de truncamiento no puede ser negativo.")
    diag = diagnosticar_espaciamiento_nodos(x)
    if not diag["es_equiespaciado"]:
        raise ValueError(f"Interpolación equidistante inválida: {diag['mensaje']}")
    h = float(diag["h_nominal"])
    n = len(x)
    max_k = min(int(grado), n - 1)
    tabla = calcular_diferencias_ascendentes(y)
    anchor_idx = _indice_ancla_modo(n, modo)

    with np.errstate(all="ignore"):
        param = (xg - x[anchor_idx]) / h
        y_acc = np.full_like(xg, float(tabla[anchor_idx, 0]), dtype=float)
        coef = np.ones_like(xg, dtype=float)
        for k in range(1, max_k + 1):
            delta = _desplazamiento_factor_u(k - 1, modo)
            coef = coef * (param + delta) / k
            fila = _fila_diferencia_modo(n, k, modo)
            y_acc = y_acc + coef * float(tabla[fila, k])
        return y_acc
