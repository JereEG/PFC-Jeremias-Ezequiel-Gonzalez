"""
Módulo de Ecuaciones Diferenciales Ordinarias (PVI) - Arandú v3.0
Contiene métodos de un paso (Euler, Heun / Euler Modificado, RK2, RK4) y
multipaso Predictor-Corrector (Milne de 4to orden) con soporte para malla dinámica (Δh).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from core._cache_utils import (
    analizar_edo_exacta_cached,
    analizar_edo_f_cached,
    compilar_lambdify_edo_exacta,
    compilar_lambdify_edo_f,
)
from core.catalogo_ejercicios import EDO_PRECOMPUTADAS


def compilar_funcion_edo(f_str: str) -> dict[str, Any]:
    """Compila una expresión f(x, y) para y' = f(x, y) usando caché o precomputación."""
    texto = " ".join((f_str or "").strip().split())
    if not texto:
        raise ValueError("La expresión f(x, y) no puede estar vacía.")

    if texto in EDO_PRECOMPUTADAS["f"]:
        pre = EDO_PRECOMPUTADAS["f"][texto]
        f_latex = pre["f_latex"]
        fn_np = pre["fn_np"]
    else:
        f_latex = analizar_edo_f_cached(texto)
        fn_np = compilar_lambdify_edo_f(texto)

    def fn(x_val: float, y_val: float) -> float:
        with np.errstate(all="ignore"):
            res = float(np.real(fn_np(float(x_val), float(y_val))))
        if not math.isfinite(res):
            raise ValueError(f"f({x_val}, {y_val}) produjo un valor no finito ({res}).")
        return res

    return {
        "x_sym": None,
        "y_sym": None,
        "expr": None,
        "f_latex": f_latex,
        "fn": fn,
        "fn_np": fn_np,
    }


def compilar_exacta_edo(y_str: str) -> dict[str, Any]:
    """Compila una solución analítica exacta y(x) opcional usando caché o precomputación."""
    texto = " ".join((y_str or "").strip().split())
    if not texto:
        raise ValueError("La expresión exacta y(x) está vacía.")

    if texto in EDO_PRECOMPUTADAS["exacta"]:
        pre = EDO_PRECOMPUTADAS["exacta"][texto]
        y_latex = pre["y_latex"]
        fn_np = pre["fn_np"]
    else:
        y_latex = analizar_edo_exacta_cached(texto)
        fn_np = compilar_lambdify_edo_exacta(texto)

    def fn(x_val: float) -> float:
        with np.errstate(all="ignore"):
            return float(np.real(fn_np(float(x_val))))

    return {
        "x_sym": None,
        "expr": None,
        "y_latex": y_latex,
        "fn": fn,
        "fn_np": fn_np,
    }


def _construir_malla_edo(
    x0: float,
    h: float,
    xf: float | None = None,
    n_pasos: int | None = None,
) -> tuple[int, float, float, str | None]:
    """Calcula n_pasos y xf efectivo a partir de h y xf o n_pasos."""
    if h <= 0:
        raise ValueError("El tamaño de paso h debe ser estrictamente positivo.")

    aviso = None
    if n_pasos is not None and n_pasos >= 1:
        n_eff = int(n_pasos)
        xf_real = float(x0 + n_eff * h)
        if xf is not None and abs(xf_real - xf) > 1e-6:
            aviso = (
                f"Con h={h:g} y {n_eff} pasos, el extremo final efectivo es "
                f"xf={xf_real:.6g} (solicitado {xf:g})."
            )
        return n_eff, float(h), xf_real, aviso

    if xf is None:
        raise ValueError("Debe especificarse xf o n_pasos para integrar el PVI.")
    if xf <= x0:
        raise ValueError(f"El extremo final xf ({xf}) debe ser mayor que x0 ({x0}).")

    n_eff = max(1, int(round((xf - x0) / h)))
    xf_real = float(x0 + n_eff * h)
    if abs(xf_real - xf) > 1e-5:
        aviso = (
            f"El intervalo [{x0:g}, {xf:g}] no es múltiplo exacto de h={h:g}. "
            f"Se ajustó a {n_eff} pasos llegando a xf={xf_real:.6g}."
        )
    return n_eff, float(h), xf_real, aviso


def _evaluar_exacta(
    y_exacta_fn: Callable[[float], float] | None,
    x_val: float,
    y_num: float,
) -> tuple[float | None, float | None]:
    if y_exacta_fn is None:
        return None, None
    try:
        ye = float(y_exacta_fn(x_val))
        return ye, abs(ye - y_num)
    except Exception:
        return None, None


def metodo_euler(
    fn: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    xf: float | None = None,
    n_pasos: int | None = None,
    y_exacta_fn: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    """Método de Euler clásico (Orden 1): y_{k+1} = y_k + h * f(x_k, y_k)."""
    n_eff, h_eff, xf_real, aviso = _construir_malla_edo(x0, h, xf, n_pasos)
    xs = [float(x0)]
    ys = [float(y0)]
    tabla: list[dict[str, Any]] = []

    ye0, err0 = _evaluar_exacta(y_exacta_fn, xs[0], ys[0])
    pend0 = float(fn(xs[0], ys[0]))
    tabla.append(
        {
            "k": 0,
            "x_k": round(xs[0], 6),
            "y_k": round(ys[0], 8),
            "f(x_k, y_k)": round(pend0, 8),
            "Delta_y": round(h_eff * pend0, 8),
            "y_exacta": round(ye0, 8) if ye0 is not None else None,
            "error_abs": round(err0, 8) if err0 is not None else None,
        }
    )

    for k in range(n_eff):
        xk, yk = xs[-1], ys[-1]
        pend = float(fn(xk, yk))
        x_sig = x0 + (k + 1) * h_eff
        y_sig = yk + h_eff * pend
        xs.append(x_sig)
        ys.append(y_sig)

        ye, err = _evaluar_exacta(y_exacta_fn, x_sig, y_sig)
        pend_sig = float(fn(x_sig, y_sig)) if k + 1 < n_eff else float(fn(x_sig, y_sig))
        tabla.append(
            {
                "k": k + 1,
                "x_k": round(x_sig, 6),
                "y_k": round(y_sig, 8),
                "f(x_k, y_k)": round(pend_sig, 8),
                "Delta_y": round(h_eff * pend_sig, 8) if k + 1 < n_eff else 0.0,
                "y_exacta": round(ye, 8) if ye is not None else None,
                "error_abs": round(err, 8) if err is not None else None,
            }
        )

    return {
        "metodo": "Euler (Orden 1)",
        "x": xs,
        "y": ys,
        "y_final": ys[-1],
        "xf_real": xf_real,
        "h": h_eff,
        "n_pasos": n_eff,
        "tabla": tabla,
        "avisos": [aviso] if aviso else [],
    }


def metodo_euler_modificado(
    fn: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    xf: float | None = None,
    n_pasos: int | None = None,
    iterar_corrector: bool = True,
    tol: float = 1e-4,
    max_corr: int = 15,
    y_exacta_fn: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    """
    Método de Heun / Euler Modificado (Predictor-Corrector de Orden 2).
    Si iterar_corrector=True, itera el corrector hasta |C^(j) - C^(j-1)| < tol
    siguiendo el esquema de la cátedra.
    """
    n_eff, h_eff, xf_real, aviso = _construir_malla_edo(x0, h, xf, n_pasos)
    xs = [float(x0)]
    ys = [float(y0)]
    tabla: list[dict[str, Any]] = []

    ye0, err0 = _evaluar_exacta(y_exacta_fn, xs[0], ys[0])
    tabla.append(
        {
            "k": 0,
            "x_k": round(xs[0], 6),
            "y_k": round(ys[0], 8),
            "k1_pend": round(float(fn(xs[0], ys[0])), 8),
            "P_euler": None,
            "k2_pend": None,
            "C_final": round(ys[0], 8),
            "iters_corr": 0,
            "y_exacta": round(ye0, 8) if ye0 is not None else None,
            "error_abs": round(err0, 8) if err0 is not None else None,
        }
    )

    for k in range(n_eff):
        xk, yk = xs[-1], ys[-1]
        k1 = float(fn(xk, yk))
        x_sig = x0 + (k + 1) * h_eff
        pred = yk + h_eff * k1
        k2_ini = float(fn(x_sig, pred))
        corr = yk + (h_eff / 2.0) * (k1 + k2_ini)
        iters = 1

        if iterar_corrector:
            for _ in range(1, max_corr):
                k2_nuevo = float(fn(x_sig, corr))
                corr_nuevo = yk + (h_eff / 2.0) * (k1 + k2_nuevo)
                iters += 1
                if abs(corr_nuevo - corr) < tol:
                    corr = corr_nuevo
                    k2_ini = k2_nuevo
                    break
                corr = corr_nuevo
                k2_ini = k2_nuevo

        xs.append(x_sig)
        ys.append(corr)
        ye, err = _evaluar_exacta(y_exacta_fn, x_sig, corr)
        tabla.append(
            {
                "k": k + 1,
                "x_k": round(x_sig, 6),
                "y_k": round(corr, 8),
                "k1_pend": round(k1, 8),
                "P_euler": round(pred, 8),
                "k2_pend": round(k2_ini, 8),
                "C_final": round(corr, 8),
                "iters_corr": iters,
                "y_exacta": round(ye, 8) if ye is not None else None,
                "error_abs": round(err, 8) if err is not None else None,
            }
        )

    return {
        "metodo": "Heun / Euler Modificado (Orden 2)",
        "x": xs,
        "y": ys,
        "y_final": ys[-1],
        "xf_real": xf_real,
        "h": h_eff,
        "n_pasos": n_eff,
        "tabla": tabla,
        "avisos": [aviso] if aviso else [],
    }


def metodo_runge_kutta_2(
    fn: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    xf: float | None = None,
    n_pasos: int | None = None,
    y_exacta_fn: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    """Runge-Kutta de Orden 2 (Heun sin iteración de corrector): k1=h*f(x,y), k2=h*f(x+h, y+k1)."""
    n_eff, h_eff, xf_real, aviso = _construir_malla_edo(x0, h, xf, n_pasos)
    xs = [float(x0)]
    ys = [float(y0)]
    tabla: list[dict[str, Any]] = []

    ye0, err0 = _evaluar_exacta(y_exacta_fn, xs[0], ys[0])
    tabla.append(
        {
            "k": 0,
            "x_k": round(xs[0], 6),
            "y_k": round(ys[0], 8),
            "k1": None,
            "k2": None,
            "y_exacta": round(ye0, 8) if ye0 is not None else None,
            "error_abs": round(err0, 8) if err0 is not None else None,
        }
    )

    for k in range(n_eff):
        xk, yk = xs[-1], ys[-1]
        k1 = h_eff * float(fn(xk, yk))
        x_sig = x0 + (k + 1) * h_eff
        k2 = h_eff * float(fn(x_sig, yk + k1))
        y_sig = yk + 0.5 * (k1 + k2)
        xs.append(x_sig)
        ys.append(y_sig)

        ye, err = _evaluar_exacta(y_exacta_fn, x_sig, y_sig)
        tabla.append(
            {
                "k": k + 1,
                "x_k": round(x_sig, 6),
                "y_k": round(y_sig, 8),
                "k1": round(k1, 8),
                "k2": round(k2, 8),
                "y_exacta": round(ye, 8) if ye is not None else None,
                "error_abs": round(err, 8) if err is not None else None,
            }
        )

    return {
        "metodo": "Runge-Kutta 2 (RK2)",
        "x": xs,
        "y": ys,
        "y_final": ys[-1],
        "xf_real": xf_real,
        "h": h_eff,
        "n_pasos": n_eff,
        "tabla": tabla,
        "avisos": [aviso] if aviso else [],
    }


def metodo_runge_kutta_4(
    fn: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    xf: float | None = None,
    n_pasos: int | None = None,
    y_exacta_fn: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    """Runge-Kutta clásico de Orden 4 (RK4)."""
    n_eff, h_eff, xf_real, aviso = _construir_malla_edo(x0, h, xf, n_pasos)
    xs = [float(x0)]
    ys = [float(y0)]
    tabla: list[dict[str, Any]] = []

    ye0, err0 = _evaluar_exacta(y_exacta_fn, xs[0], ys[0])
    tabla.append(
        {
            "k": 0,
            "x_k": round(xs[0], 6),
            "y_k": round(ys[0], 8),
            "k1": None,
            "k2": None,
            "k3": None,
            "k4": None,
            "y_exacta": round(ye0, 8) if ye0 is not None else None,
            "error_abs": round(err0, 8) if err0 is not None else None,
        }
    )

    for k in range(n_eff):
        xk, yk = xs[-1], ys[-1]
        k1 = h_eff * float(fn(xk, yk))
        k2 = h_eff * float(fn(xk + 0.5 * h_eff, yk + 0.5 * k1))
        k3 = h_eff * float(fn(xk + 0.5 * h_eff, yk + 0.5 * k2))
        x_sig = x0 + (k + 1) * h_eff
        k4 = h_eff * float(fn(x_sig, yk + k3))
        y_sig = yk + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        xs.append(x_sig)
        ys.append(y_sig)
        ye, err = _evaluar_exacta(y_exacta_fn, x_sig, y_sig)
        tabla.append(
            {
                "k": k + 1,
                "x_k": round(x_sig, 6),
                "y_k": round(y_sig, 8),
                "k1": round(k1, 8),
                "k2": round(k2, 8),
                "k3": round(k3, 8),
                "k4": round(k4, 8),
                "y_exacta": round(ye, 8) if ye is not None else None,
                "error_abs": round(err, 8) if err is not None else None,
            }
        )

    return {
        "metodo": "Runge-Kutta 4 (RK4)",
        "x": xs,
        "y": ys,
        "y_final": ys[-1],
        "xf_real": xf_real,
        "h": h_eff,
        "n_pasos": n_eff,
        "tabla": tabla,
        "avisos": [aviso] if aviso else [],
    }


def metodo_milne(
    fn: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    xf: float | None = None,
    n_pasos: int | None = None,
    n_correcciones: int = 2,
    y_exacta_fn: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    """
    Método Predictor-Corrector de Milne (Orden 4).
    Requiere al menos 4 pasos (5 puntos: k=0..4). Arranca k=1,2,3 con RK4 y
    aplica Predictor + Corrector (por defecto 2 pasadas: C y C¹) desde k=4.
    """
    n_eff, h_eff, xf_real, aviso = _construir_malla_edo(x0, h, xf, n_pasos)
    if n_eff < 4:
        raise ValueError(
            f"El método multipaso de Milne necesita al menos 4 pasos (actual: {n_eff} con h={h_eff:g}) "
            "porque requiere 4 puntos iniciales (k=0, 1, 2, 3 generados con RK4). "
            "Reduce el paso Δh o amplía el intervalo xf."
        )

    rk_arranque = metodo_runge_kutta_4(
        fn, x0=x0, y0=y0, h=h_eff, n_pasos=3, y_exacta_fn=y_exacta_fn
    )
    xs = list(rk_arranque["x"])
    ys = list(rk_arranque["y"])
    derivs = [float(fn(xv, yv)) for xv, yv in zip(xs, ys)]

    tabla: list[dict[str, Any]] = []
    for k in range(4):
        ye, err = _evaluar_exacta(y_exacta_fn, xs[k], ys[k])
        tabla.append(
            {
                "k": k,
                "x_k": round(xs[k], 6),
                "origen": "Inicial" if k == 0 else "RK4 (Arranque)",
                "P": None,
                "C": None,
                "C_final": round(ys[k], 8),
                "y'_k": round(derivs[k], 8),
                "diff_P_C": None,
                "y_exacta": round(ye, 8) if ye is not None else None,
                "error_abs": round(err, 8) if err is not None else None,
            }
        )

    for i in range(3, n_eff):
        x_sig = x0 + (i + 1) * h_eff
        # Predictor de Milne: y_{i+1}^P = y_{i-3} + (4h/3)*(2y'_i - y'_{i-1} + 2y'_{i-2})
        p_val = ys[i - 3] + (4.0 * h_eff / 3.0) * (
            2.0 * derivs[i] - derivs[i - 1] + 2.0 * derivs[i - 2]
        )
        yp_curr = float(fn(x_sig, p_val))
        c_primero = None
        c_val = p_val
        for pass_idx in range(max(1, n_correcciones)):
            c_val = ys[i - 1] + (h_eff / 3.0) * (yp_curr + 4.0 * derivs[i] + derivs[i - 1])
            if pass_idx == 0:
                c_primero = c_val
            yp_curr = float(fn(x_sig, c_val))

        xs.append(x_sig)
        ys.append(c_val)
        derivs.append(float(fn(x_sig, c_val)))
        ye, err = _evaluar_exacta(y_exacta_fn, x_sig, c_val)
        tabla.append(
            {
                "k": i + 1,
                "x_k": round(x_sig, 6),
                "origen": "Milne P-C",
                "P": round(p_val, 8),
                "C": round(c_primero, 8) if c_primero is not None else None,
                "C_final": round(c_val, 8),
                "y'_k": round(derivs[-1], 8),
                "diff_P_C": round(abs(c_val - p_val), 8),
                "y_exacta": round(ye, 8) if ye is not None else None,
                "error_abs": round(err, 8) if err is not None else None,
            }
        )

    return {
        "metodo": "Predictor-Corrector de Milne (Orden 4)",
        "x": xs,
        "y": ys,
        "y_final": ys[-1],
        "xf_real": xf_real,
        "h": h_eff,
        "n_pasos": n_eff,
        "tabla": tabla,
        "avisos": [aviso] if aviso else [],
    }


def comparar_metodos_edo(
    fn: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    xf: float,
    y_exacta_fn: Callable[[float], float] | None = None,
    metodos: list[str] | None = None,
) -> dict[str, Any]:
    """Ejecuta múltiples métodos de EDO sobre la misma malla para comparación directa."""
    disponibles = {
        "euler": lambda: metodo_euler(fn, x0, y0, h, xf=xf, y_exacta_fn=y_exacta_fn),
        "heun": lambda: metodo_euler_modificado(
            fn, x0, y0, h, xf=xf, iterar_corrector=True, y_exacta_fn=y_exacta_fn
        ),
        "rk2": lambda: metodo_runge_kutta_2(fn, x0, y0, h, xf=xf, y_exacta_fn=y_exacta_fn),
        "rk4": lambda: metodo_runge_kutta_4(fn, x0, y0, h, xf=xf, y_exacta_fn=y_exacta_fn),
        "milne": lambda: metodo_milne(fn, x0, y0, h, xf=xf, y_exacta_fn=y_exacta_fn),
    }
    seleccion = metodos or ["euler", "heun", "rk2", "rk4", "milne"]

    resultados: dict[str, dict[str, Any]] = {}
    errores_ejecucion: dict[str, str] = {}
    resumen: list[dict[str, Any]] = []

    for clave in seleccion:
        if clave not in disponibles:
            continue
        try:
            res = disponibles[clave]()
            resultados[clave] = res
            err_final = res["tabla"][-1].get("error_abs") if res["tabla"] else None
            resumen.append(
                {
                    "clave": clave,
                    "metodo": res["metodo"],
                    "n_pasos": res["n_pasos"],
                    "y_final": res["y_final"],
                    "error_abs_final": err_final,
                }
            )
        except Exception as exc:
            errores_ejecucion[clave] = str(exc)

    return {
        "resultados": resultados,
        "errores_ejecucion": errores_ejecucion,
        "resumen": resumen,
    }

