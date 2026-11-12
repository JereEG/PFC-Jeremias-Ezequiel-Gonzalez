"""
core/lab7_edo.py
================
Módulo de cálculo puro para Ecuaciones Diferenciales Ordinarias - Problemas de Valor Inicial (Laboratorio 7).

Incluye:
- Método de Euler Explícito (orden O(h)).
- Método de Euler Modificado / Heun (predictor-corrector de orden O(h^2)).
- Familia de Runge-Kutta de 2° Orden (RK2 Estándar y RK2 Punto Medio).
- Método Clásico de Runge-Kutta de 4° Orden (RK4, orden O(h^4)).
- Método Multipaso Predictor-Corrector de Milne (4° orden, con arranque RK4 y refinamiento iterativo).
- Comparativa lado a lado y contraste contra soluciones analíticas exactas.

Diseño: Funciones 100% puras, sin `input()` ni `print()`, con manejo robusto de excepciones
y retorno de historiales detallados estructurados listos para UI.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable
import numpy as np
import sympy as sp


# ==============================================================================
# 1. PARSEADOR SEGURO DE EXPRESIONES f(x, y)
# ==============================================================================

def compilar_funcion_edo(f_str: str) -> Callable[[float, float], float]:
    """
    Compila de manera segura una expresión textual f(x, y) en una función invocable.
    Soporta funciones de math y operadores estándar.
    """
    x_s, y_s = sp.symbols("x y")
    try:
        expr = sp.sympify(f_str)
        fn_eval = sp.lambdify((x_s, y_s), expr, modules=["numpy", "math"])
    except Exception as e:
        raise ValueError(f"Expresión diferencial f(x, y) = '{f_str}' inválida: {e}")

    def f_wrap(x: float, y: float) -> float:
        try:
            val = fn_eval(float(x), float(y))
            return float(val)
        except Exception as err:
            raise ValueError(f"Error evaluando f({x}, {y}): {err}")

    return f_wrap


# ==============================================================================
# 2. MÉTODO DE EULER EXPLÍCITO Y EULER MODIFICADO (HEUN)
# ==============================================================================

def metodo_euler(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    n_pasos: int,
) -> dict[str, Any]:
    """
    Resuelve el PVI dy/dx = f(x, y), y(x0) = y0 mediante el Método de Euler Explícito:
        y_{n+1} = y_n + h * f(x_n, y_n)
    """
    t_ini = time.perf_counter()
    x_act, y_act = float(x0), float(y0)

    puntos_x = [x_act]
    puntos_y = [y_act]
    historial: list[dict[str, Any]] = []

    for i in range(n_pasos):
        pend = float(f(x_act, y_act))
        x_sig = round(x_act + h, 10)
        y_sig = y_act + h * pend

        historial.append({
            "paso": i + 1,
            "x_n": x_act,
            "y_n": y_act,
            "f_xy": pend,
            "x_sig": x_sig,
            "y_sig": y_sig,
        })

        puntos_x.append(x_sig)
        puntos_y.append(y_sig)
        x_act, y_act = x_sig, y_sig

    duracion = time.perf_counter() - t_ini

    return {
        "metodo": "Euler Explícito",
        "orden": 1,
        "x": puntos_x,
        "y": puntos_y,
        "historial": historial,
        "duracion": duracion,
    }


def metodo_euler_modificado(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    n_pasos: int,
) -> dict[str, Any]:
    """
    Resuelve el PVI mediante el Método de Euler Modificado (Heun):
        k1 = f(x_n, y_n)
        y_{n+1}^{(p)} = y_n + h * k1
        k2 = f(x_n + h, y_{n+1}^{(p)})
        y_{n+1} = y_n + (h / 2) * (k1 + k2)
    """
    t_ini = time.perf_counter()
    x_act, y_act = float(x0), float(y0)

    puntos_x = [x_act]
    puntos_y = [y_act]
    historial: list[dict[str, Any]] = []

    for i in range(n_pasos):
        k1 = float(f(x_act, y_act))
        x_sig = round(x_act + h, 10)
        y_pred = y_act + h * k1
        k2 = float(f(x_sig, y_pred))
        y_sig = y_act + (h / 2.0) * (k1 + k2)

        historial.append({
            "paso": i + 1,
            "x_n": x_act,
            "y_n": y_act,
            "k1": k1,
            "y_predicho": y_pred,
            "k2": k2,
            "x_sig": x_sig,
            "y_sig": y_sig,
        })

        puntos_x.append(x_sig)
        puntos_y.append(y_sig)
        x_act, y_act = x_sig, y_sig

    duracion = time.perf_counter() - t_ini

    return {
        "metodo": "Euler Modificado (Heun)",
        "orden": 2,
        "x": puntos_x,
        "y": puntos_y,
        "historial": historial,
        "duracion": duracion,
    }


# ==============================================================================
# 3. MÉTODOS DE RUNGE-KUTTA (RK2 Y RK4)
# ==============================================================================

def metodo_runge_kutta_2(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    n_pasos: int,
    variante: str = "estandar",
) -> dict[str, Any]:
    """
    Resuelve el PVI mediante Runge-Kutta de 2° orden:
    - variante='estandar':
        k1 = h * f(x_n, y_n)
        k2 = h * f(x_n + h, y_n + k1)
        y_{n+1} = y_n + 0.5 * (k1 + k2)
    - variante='punto_medio':
        k1 = h * f(x_n, y_n)
        k2 = h * f(x_n + h/2, y_n + k1/2)
        y_{n+1} = y_n + k2
    """
    t_ini = time.perf_counter()
    x_act, y_act = float(x0), float(y0)

    puntos_x = [x_act]
    puntos_y = [y_act]
    historial: list[dict[str, Any]] = []

    for i in range(n_pasos):
        x_sig = round(x_act + h, 10)
        k1 = h * float(f(x_act, y_act))

        if variante == "punto_medio":
            k2 = h * float(f(x_act + h / 2.0, y_act + k1 / 2.0))
            y_sig = y_act + k2
            detalle_paso = {
                "paso": i + 1,
                "x_n": x_act,
                "y_n": y_act,
                "k1": k1,
                "k2": k2,
                "x_sig": x_sig,
                "y_sig": y_sig,
            }
        else:
            k2 = h * float(f(x_act + h, y_act + k1))
            y_sig = y_act + 0.5 * (k1 + k2)
            detalle_paso = {
                "paso": i + 1,
                "x_n": x_act,
                "y_n": y_act,
                "k1": k1,
                "k2": k2,
                "x_sig": x_sig,
                "y_sig": y_sig,
            }

        historial.append(detalle_paso)
        puntos_x.append(x_sig)
        puntos_y.append(y_sig)
        x_act, y_act = x_sig, y_sig

    duracion = time.perf_counter() - t_ini

    return {
        "metodo": f"Runge-Kutta 2 ({variante})",
        "orden": 2,
        "variante": variante,
        "x": puntos_x,
        "y": puntos_y,
        "historial": historial,
        "duracion": duracion,
    }


def metodo_runge_kutta_4(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    n_pasos: int,
) -> dict[str, Any]:
    """
    Resuelve el PVI mediante el Método Clásico de Runge-Kutta de 4° Orden (RK4):
        k1 = h * f(x_n, y_n)
        k2 = h * f(x_n + h/2, y_n + k1/2)
        k3 = h * f(x_n + h/2, y_n + k2/2)
        k4 = h * f(x_n + h, y_n + k3)
        y_{n+1} = y_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    t_ini = time.perf_counter()
    x_act, y_act = float(x0), float(y0)

    puntos_x = [x_act]
    puntos_y = [y_act]
    historial: list[dict[str, Any]] = []

    for i in range(n_pasos):
        x_sig = round(x_act + h, 10)
        k1 = h * float(f(x_act, y_act))
        k2 = h * float(f(x_act + h / 2.0, y_act + k1 / 2.0))
        k3 = h * float(f(x_act + h / 2.0, y_act + k2 / 2.0))
        k4 = h * float(f(x_act + h, y_act + k3))

        y_sig = y_act + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        historial.append({
            "paso": i + 1,
            "x_n": x_act,
            "y_n": y_act,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "x_sig": x_sig,
            "y_sig": y_sig,
        })

        puntos_x.append(x_sig)
        puntos_y.append(y_sig)
        x_act, y_act = x_sig, y_sig

    duracion = time.perf_counter() - t_ini

    return {
        "metodo": "Runge-Kutta 4 (RK4)",
        "orden": 4,
        "x": puntos_x,
        "y": puntos_y,
        "historial": historial,
        "duracion": duracion,
    }


# ==============================================================================
# 4. MÉTODO MULTIPASO PREDICTOR-CORRECTOR DE MILNE
# ==============================================================================

def metodo_milne(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    n_pasos: int,
    tolerancia_corrector: float = 1e-6,
    max_iter_corrector: int = 10,
    puntos_arranque: tuple[list[float], list[float]] | None = None,
) -> dict[str, Any]:
    """
    Resuelve el PVI dy/dx = f(x, y) mediante el Método Predictor-Corrector de Milne (4° Orden).
    Requiere 4 valores de arranque: (x0..x3, y0..y3). Si no se suministran, se calculan
    con Runge-Kutta 4 (RK4).

    Fórmulas:
        PREDICTOR:
            y_{n+1}^{(p)} = y_{n-3} + (4h/3) * [ 2*f_n - f_{n-1} + 2*f_{n-2} ]
        CORRECTOR:
            y_{n+1}^{(c)} = y_{n-1} + (h/3) * [ f(x_{n+1}, y_{n+1}) + 4*f_n + f_{n-1} ]
    """
    t_ini = time.perf_counter()

    # 1. Obtención de los 4 puntos de arranque
    if puntos_arranque is not None:
        x_vals = list(puntos_arranque[0])
        y_vals = list(puntos_arranque[1])
        if len(x_vals) < 4 or len(y_vals) < 4:
            raise ValueError("puntos_arranque debe contener al menos 4 puntos.")
    else:
        # Arranque automático con RK4 (3 pasos para tener los índices 0, 1, 2, 3)
        res_rk4 = metodo_runge_kutta_4(f, x0, y0, h, n_pasos=3)
        x_vals = res_rk4["x"][:4]
        y_vals = res_rk4["y"][:4]

    f_vals = [float(f(xi, yi)) for xi, yi in zip(x_vals, y_vals)]
    historial: list[dict[str, Any]] = []

    # Cantidad de pasos que faltan tras los 4 puntos iniciales
    pasos_milne = max(0, n_pasos - 3)

    for i in range(pasos_milne):
        n_idx = len(y_vals) - 1  # Índice del último punto disponible (n)
        x_sig = round(x_vals[n_idx] + h, 10)

        # PASO PREDICTOR:
        # y_{n+1}^{(p)} = y_{n-3} + (4h/3) * [ 2*f_n - f_{n-1} + 2*f_{n-2} ]
        y_pred = y_vals[n_idx - 3] + (4.0 * h / 3.0) * (
            2.0 * f_vals[n_idx] - f_vals[n_idx - 1] + 2.0 * f_vals[n_idx - 2]
        )

        f_pred = float(f(x_sig, y_pred))

        # PASO CORRECTOR ITERATIVO:
        # y_{n+1}^{(c)} = y_{n-1} + (h/3) * [ f(x_{n+1}, y_{n+1}) + 4*f_n + f_{n-1} ]
        y_corr = y_pred
        iter_c = 0
        while iter_c < max_iter_corrector:
            iter_c += 1
            y_ant = y_corr
            f_eval = float(f(x_sig, y_corr))
            y_corr = y_vals[n_idx - 1] + (h / 3.0) * (f_eval + 4.0 * f_vals[n_idx] + f_vals[n_idx - 1])

            if abs(y_corr - y_ant) < tolerancia_corrector:
                break

        f_final = float(f(x_sig, y_corr))
        err_pc = abs(y_corr - y_pred)

        historial.append({
            "paso": i + 4,
            "x_sig": x_sig,
            "y_predicho": y_pred,
            "y_corregido": y_corr,
            "f_predicho": f_pred,
            "f_corregido": f_final,
            "iteraciones_corrector": iter_c,
            "diferencia_pc": err_pc,
            "cota_milne": err_pc / 29.0,
        })

        x_vals.append(x_sig)
        y_vals.append(y_corr)
        f_vals.append(f_final)

    duracion = time.perf_counter() - t_ini

    return {
        "metodo": "Milne Predictor-Corrector (4° orden)",
        "orden": 4,
        "x": x_vals,
        "y": y_vals,
        "arranque_x": x_vals[:4],
        "arranque_y": y_vals[:4],
        "historial": historial,
        "duracion": duracion,
    }


# ==============================================================================
# 5. COMPARATIVA INTEGRAL DE MÉTODOS Y EVALUACIÓN DE ERROR
# ==============================================================================

def comparar_metodos_edo(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    n_pasos: int,
    solucion_exacta_fn: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    """
    Ejecuta en paralelo Euler, Euler Modificado, RK2, RK4 y Milne con los mismos parámetros
    y contrasta contra la solución analítica si se provee.
    """
    res_euler = metodo_euler(f, x0, y0, h, n_pasos)
    res_heun = metodo_euler_modificado(f, x0, y0, h, n_pasos)
    res_rk2 = metodo_runge_kutta_2(f, x0, y0, h, n_pasos, variante="estandar")
    res_rk4 = metodo_runge_kutta_4(f, x0, y0, h, n_pasos)
    res_milne = metodo_milne(f, x0, y0, h, n_pasos)

    xs = res_rk4["x"]

    # Construir tabla comparativa punto a punto
    filas_comparativas = []
    for i, x_val in enumerate(xs):
        fila: dict[str, Any] = {
            "i": i,
            "x": x_val,
            "y_euler": res_euler["y"][i] if i < len(res_euler["y"]) else None,
            "y_heun": res_heun["y"][i] if i < len(res_heun["y"]) else None,
            "y_rk2": res_rk2["y"][i] if i < len(res_rk2["y"]) else None,
            "y_rk4": res_rk4["y"][i] if i < len(res_rk4["y"]) else None,
            "y_milne": res_milne["y"][i] if i < len(res_milne["y"]) else None,
        }

        if solucion_exacta_fn is not None:
            try:
                y_ex = float(solucion_exacta_fn(x_val))
                fila["y_exacto"] = y_ex
                for m_key in ["euler", "heun", "rk2", "rk4", "milne"]:
                    val_aprox = fila[f"y_{m_key}"]
                    if val_aprox is not None:
                        err_abs = abs(y_ex - val_aprox)
                        err_rel = (err_abs / abs(y_ex) * 100.0) if abs(y_ex) > 1e-15 else 0.0
                        fila[f"err_{m_key}"] = err_abs
                        fila[f"err_rel_{m_key}_pct"] = err_rel
            except Exception:
                fila["y_exacto"] = None

        filas_comparativas.append(fila)

    return {
        "x": xs,
        "metodos": {
            "euler": res_euler,
            "heun": res_heun,
            "rk2": res_rk2,
            "rk4": res_rk4,
            "milne": res_milne,
        },
        "tabla_comparativa": filas_comparativas,
    }

