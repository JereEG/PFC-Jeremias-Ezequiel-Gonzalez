"""
core/lab5_interpolacion.py
==========================
Módulo de cálculo puro para Interpolación Polinomial y Aproximación (Laboratorio 5).

Incluye:
- Interpolación Lineal directa e inversa.
- Interpolación Cuadrática Inversa (resolución de u = x - x0 con selección de raíz física).
- Método de Lagrange (polinomios base L_i(x), combinación lineal y expresión simbólica exacta con SymPy).
- Newton-Gregory Ascendente (diferencias progresivas Delta^k y_0, parámetro u = (x - x0)/h).
- Newton-Gregory Descendente (diferencias retrógradas nabla^k y_n, parámetro v = (x - xn)/h).

Diseño: Funciones 100% puras, sin `input()` ni `print()`, con manejo robusto de excepciones
y retorno de historiales detallados estructurados listos para UI.
"""

from __future__ import annotations

import math
import time
from typing import Any
import numpy as np
import sympy as sp


# ==============================================================================
# 1. INTERPOLACIÓN LINEAL DIRECTA E INVERSA
# ==============================================================================

def interpolar_lineal_directa(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
) -> dict[str, Any]:
    """
    Realiza la interpolación lineal directa buscando el intervalo [x_i, x_{i+1}]
    que contiene a x_eval:
        y = y0 + ((y1 - y0) / (x1 - x0)) * (x - x0)
    """
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)

    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 pares de puntos (x, y).")

    # Ordenar por x
    orden = np.argsort(x)
    x = x[orden]
    y = y[orden]

    # Encontrar intervalo
    n = len(x)
    idx = 0
    if x_eval <= x[0]:
        idx = 0
    elif x_eval >= x[-1]:
        idx = n - 2
    else:
        for i in range(n - 1):
            if x[i] <= x_eval <= x[i + 1]:
                idx = i
                break

    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    dx = x1 - x0
    if abs(dx) < 1e-15:
        raise ZeroDivisionError("Puntos con la misma abscisa x detectados.")

    pendiente = (y1 - y0) / dx
    y_interp = y0 + pendiente * (x_eval - x0)

    return {
        "x_eval": x_eval,
        "y_interpolado": float(y_interp),
        "indice_intervalo": int(idx),
        "x0": float(x0),
        "x1": float(x1),
        "y0": float(y0),
        "y1": float(y1),
        "pendiente": float(pendiente),
        "h": float(dx),
        "delta_y": float(y1 - y0),
        "es_extrapolacion": bool(x_eval < x[0] or x_eval > x[-1]),
    }


def interpolar_lineal_inversa(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    y_buscado: float,
) -> dict[str, Any]:
    """
    Calcula el valor de x correspondiente a un y_buscado dado mediante
    interpolación lineal inversa (despejando x en función de y):
        x = x0 + (y_buscado - y0) * (x1 - x0) / (y1 - y0)
    """
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)

    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos 2 pares de puntos (x, y).")

    n = len(x)
    idx = 0
    for i in range(n - 1):
        if (y[i] <= y_buscado <= y[i + 1]) or (y[i] >= y_buscado >= y[i + 1]):
            idx = i
            break
    else:
        # Si no está estrictamente en un intervalo, elegir el más próximo
        dists = [min(abs(y[i] - y_buscado), abs(y[i + 1] - y_buscado)) for i in range(n - 1)]
        idx = int(np.argmin(dists))

    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    h = x1 - x0
    delta_y = y1 - y0

    if abs(delta_y) < 1e-15:
        raise ZeroDivisionError("Los puntos seleccionados tienen idéntico valor de y (división por cero).")

    dif_y = y_buscado - y0
    x_interp = x0 + dif_y * (h / delta_y)

    return {
        "y_buscado": y_buscado,
        "x_resultado": float(x_interp),
        "indice": int(idx),
        "x0": float(x0),
        "x1": float(x1),
        "y0": float(y0),
        "y1": float(y1),
        "h": float(h),
        "delta_y": float(delta_y),
        "diferencia_y": float(dif_y),
    }


# ==============================================================================
# 2. INTERPOLACIÓN CUADRÁTICA INVERSA
# ==============================================================================

def interpolar_cuadratica_inversa(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    y_buscado: float,
) -> dict[str, Any]:
    """
    Calcula x para un y_buscado mediante interpolación cuadrática inversa
    usando 3 puntos consecutivos (x0, x1, x2) con espaciamiento h = x1 - x0:
        y = y0 + (Delta y0 / h)*u + (Delta^2 y0 / (2*h^2))*u*(u - h)
    Reordenando en potencias de u = x - x0:
        a*u^2 + b*u + c = 0
    """
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)

    if len(x) < 3 or len(y) < 3:
        raise ValueError("Se requieren al menos 3 puntos para la interpolación cuadrática inversa.")

    n = len(x)
    # Buscar el trío de puntos consecutivos cuyo rango abarque y_buscado
    idx = 0
    encontrado = False
    for i in range(n - 2):
        y_min = min(y[i], y[i + 1], y[i + 2])
        y_max = max(y[i], y[i + 1], y[i + 2])
        if y_min <= y_buscado <= y_max:
            idx = i
            encontrado = True
            break

    if not encontrado:
        # Elegir el trío más próximo
        idx = 0 if abs(y_buscado - y[0]) < abs(y_buscado - y[-1]) else max(0, n - 3)

    x0, x1, x2 = x[idx], x[idx + 1], x[idx + 2]
    y0, y1, y2 = y[idx], y[idx + 1], y[idx + 2]

    h = x1 - x0
    if abs(h) < 1e-12:
        raise ZeroDivisionError("Espaciamiento nulo h entre puntos x.")

    delta_y0 = y1 - y0
    delta2_y0 = y2 - 2.0 * y1 + y0
    diferencia_y = y_buscado - y0

    # Coeficientes de la ecuación cuadrática en u = x - x0:
    # a = Delta^2 y0 / (2 * h^2)
    # b = Delta y0 / h - Delta^2 y0 / (2 * h)
    # c = - (y_buscado - y0)
    a_coef = delta2_y0 / (2.0 * h * h)
    b_coef = (delta_y0 / h) - (delta2_y0 / (2.0 * h))
    c_coef = -diferencia_y

    if abs(a_coef) < 1e-12:
        # Degenera en lineal
        if abs(b_coef) < 1e-12:
            raise ValueError("Los datos son constantes o incompatibles; no se puede despejar x.")
        u_sol = -c_coef / b_coef
        x_interp = x0 + u_sol
        return {
            "x_resultado": float(x_interp),
            "es_lineal": True,
            "x0": x0, "x1": x1, "x2": x2,
            "y0": y0, "y1": y1, "y2": y2,
            "h": h,
            "y_buscado": y_buscado,
        }

    discriminante = b_coef**2 - 4.0 * a_coef * c_coef
    if discriminante < 0:
        raise ValueError(f"Discriminante negativo ({discriminante:.4e}): no existe solución real en este tramo.")

    sqrt_disc = math.sqrt(discriminante)
    u1 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    u2 = (-b_coef - sqrt_disc) / (2.0 * a_coef)

    x_sol1 = x0 + u1
    x_sol2 = x0 + u2

    # Selección de la raíz dentro del intervalo [min(x0,x2), max(x0,x2)]
    x_min, x_max = min(x0, x2), max(x0, x2)
    en_int1 = (x_min <= x_sol1 <= x_max)
    en_int2 = (x_min <= x_sol2 <= x_max)

    if en_int1 and not en_int2:
        x_sel = x_sol1
    elif en_int2 and not en_int1:
        x_sel = x_sol2
    else:
        # Si ambas o ninguna caen en el intervalo, elegir la más cercana a x1
        x_sel = x_sol1 if abs(x_sol1 - x1) <= abs(x_sol2 - x1) else x_sol2

    return {
        "x_resultado": float(x_sel),
        "x_sol1": float(x_sol1),
        "x_sol2": float(x_sol2),
        "u1": float(u1),
        "u2": float(u2),
        "discriminante": float(discriminante),
        "a_coef": float(a_coef),
        "b_coef": float(b_coef),
        "c_coef": float(c_coef),
        "x0": float(x0), "x1": float(x1), "x2": float(x2),
        "y0": float(y0), "y1": float(y1), "y2": float(y2),
        "h": float(h),
        "delta_y0": float(delta_y0),
        "delta2_y0": float(delta2_y0),
        "y_buscado": float(y_buscado),
        "indice": int(idx),
    }


# ==============================================================================
# 3. POLINOMIO INTERPOLADOR DE LAGRANGE
# ==============================================================================

def calcular_polinomio_base_lagrange(
    x_puntos: list[float] | np.ndarray,
    i: int,
    x_eval: float,
) -> float:
    """Calcula el polinomio base de Lagrange L_i(x) evaluado en x_eval."""
    x = np.array(x_puntos, dtype=float)
    n = len(x)
    num = 1.0
    den = 1.0
    for j in range(n):
        if j != i:
            num *= (x_eval - x[j])
            den *= (x[i] - x[j])
    return float(num / den)


def interpolar_lagrange(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
    x_eval: float,
) -> dict[str, Any]:
    """
    Calcula la interpolación de Lagrange:
        P(x) = sum_{i=0}^{n} y_i * L_i(x)
    No requiere que los puntos estén equiespaciados.
    """
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    n = len(x)

    if n != len(y) or n < 2:
        raise ValueError("Se requieren al menos 2 puntos (x, y).")

    if len(np.unique(x)) != n:
        raise ValueError("Existen abscisas x duplicadas en el conjunto de puntos.")

    polinomios_base = []
    terminos_detalle = []
    resultado = 0.0

    for i in range(n):
        L_i = calcular_polinomio_base_lagrange(x, i, x_eval)
        termino = float(y[i] * L_i)
        resultado += termino
        polinomios_base.append(float(L_i))
        terminos_detalle.append({
            "i": i,
            "x_i": float(x[i]),
            "y_i": float(y[i]),
            "L_i": float(L_i),
            "termino": termino,
        })

    duracion = time.perf_counter() - t_ini

    return {
        "x_eval": x_eval,
        "y_interpolado": float(resultado),
        "grado": n - 1,
        "puntos": n,
        "polinomios_base": polinomios_base,
        "terminos_detalle": terminos_detalle,
        "duracion": duracion,
    }


def construir_polinomio_simbolico_lagrange(
    x_puntos: list[float] | np.ndarray,
    y_puntos: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Construye simbólicamente con SymPy el polinomio interpolador de Lagrange y lo simplifica."""
    x_sym = sp.Symbol("x")
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    n = len(x)

    P_sym = 0
    bases_simbolicas = []
    for i in range(n):
        L_i_sym = 1
        for j in range(n):
            if j != i:
                L_i_sym *= (x_sym - x[j]) / (x[i] - x[j])
        bases_simbolicas.append(sp.simplify(L_i_sym))
        P_sym += y[i] * L_i_sym

    P_expandido = sp.expand(P_sym)

    return {
        "polinomio_simbolico": P_sym,
        "polinomio_expandido": P_expandido,
        "latex_expandido": sp.latex(P_expandido),
        "bases_simbolicas": bases_simbolicas,
    }


# ==============================================================================
# 4. NEWTON-GREGORY ASCENDENTE (DIFERENCIAS PROGRESIVAS)
# ==============================================================================

def calcular_diferencias_ascendentes(y_puntos: list[float] | np.ndarray) -> np.ndarray:
    """Calcula la matriz triangular de diferencias finitas progresivas Delta^k y_i."""
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
) -> dict[str, Any]:
    """
    Interpolación de Newton-Gregory Ascendente:
        P(x) = y0 + u*Delta y0 + (u(u-1)/2!)*Delta^2 y0 + ...
    donde u = (x - x0) / h.
    """
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    n = len(x)

    if n != len(y) or n < 2:
        raise ValueError("Se requieren al menos 2 puntos.")

    h_arr = np.diff(x)
    h = float(h_arr[0])
    if not np.allclose(h_arr, h, rtol=1e-5):
        raise ValueError("Los puntos x deben estar equiespaciados para aplicar Newton-Gregory.")

    u = float((x_eval - x[0]) / h)
    tabla_diff = calcular_diferencias_ascendentes(y)

    resultado = float(y[0])
    terminos = [{"k": 0, "coef_binomial": 1.0, "diferencia": float(y[0]), "termino": float(y[0])}]

    coef_u = 1.0
    for k in range(1, n):
        coef_u *= (u - (k - 1)) / k
        diff_k = float(tabla_diff[0, k])
        term = float(coef_u * diff_k)
        resultado += term
        terminos.append({
            "k": k,
            "coef_binomial": float(coef_u),
            "diferencia": diff_k,
            "termino": term,
        })

    duracion = time.perf_counter() - t_ini

    return {
        "x_eval": x_eval,
        "y_interpolado": float(resultado),
        "u": u,
        "h": h,
        "x0": float(x[0]),
        "y0": float(y[0]),
        "tabla_diferencias": tabla_diff.tolist(),
        "terminos": terminos,
        "duracion": duracion,
    }


# ==============================================================================
# 5. NEWTON-GREGORY DESCENDENTE (DIFERENCIAS RETRÓGRADAS)
# ==============================================================================

def calcular_diferencias_descendentes(y_puntos: list[float] | np.ndarray) -> np.ndarray:
    """Calcula la matriz triangular de diferencias finitas retrógradas nabla^k y_i."""
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
) -> dict[str, Any]:
    """
    Interpolación de Newton-Gregory Descendente (Retrógrada):
        P(x) = yn + v*nabla yn + (v(v+1)/2!)*nabla^2 yn + ...
    donde v = (x - xn) / h.
    """
    t_ini = time.perf_counter()
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    n = len(x)

    if n != len(y) or n < 2:
        raise ValueError("Se requieren al menos 2 puntos.")

    h_arr = np.diff(x)
    h = float(h_arr[0])
    if not np.allclose(h_arr, h, rtol=1e-5):
        raise ValueError("Los puntos x deben estar equiespaciados para aplicar Newton-Gregory.")

    v = float((x_eval - x[-1]) / h)
    tabla_diff = calcular_diferencias_descendentes(y)

    resultado = float(y[-1])
    terminos = [{"k": 0, "coef_binomial": 1.0, "diferencia": float(y[-1]), "termino": float(y[-1])}]

    coef_v = 1.0
    for k in range(1, n):
        coef_v *= (v + (k - 1)) / k
        diff_k = float(tabla_diff[n - 1, k])
        term = float(coef_v * diff_k)
        resultado += term
        terminos.append({
            "k": k,
            "coef_binomial": float(coef_v),
            "diferencia": diff_k,
            "termino": term,
        })

    duracion = time.perf_counter() - t_ini

    return {
        "x_eval": x_eval,
        "y_interpolado": float(resultado),
        "v": v,
        "h": h,
        "xn": float(x[-1]),
        "yn": float(y[-1]),
        "tabla_diferencias": tabla_diff.tolist(),
        "terminos": terminos,
        "duracion": duracion,
    }

