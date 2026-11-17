"""
core/raices.py
==============
Módulo de cálculo puro para Raíces de Ecuaciones No Lineales
(Arandú: Métodos Computacionales v3.0).

Incluye:
- Compilación simbólica segura de f(x), f'(x) y f''(x) con SymPy.
- Regla de los Signos de Descartes (polinomios simbólicos).
- Método de Tanteos / Bolzano con paso Δx configurable en vivo.
- Método de Bisección (Intervalo Medio) con cota teórica y geometría de corchetes.
- Interpolación Lineal (Regula Falsi): variante Cátedra UNNE (pivote fijo) y variante Bolzano,
  con puntos de las rectas secantes paso a paso.
- Verificación de Condiciones de Convergencia de Fourier y Método de Newton-Raphson
  con datos completos para trazado de rectas tangentes paso a paso.
- Punto Fijo y Aceleración Δ² de Aitken con trayectoria de escalera/telaraña.

Diseño: Funciones 100% puras, sin `input()` ni `print()`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from core._cache_utils import (
    analizar_g_pf_sympy_cached,
    analizar_raices_sympy_cached,
    compilar_lambdify_pf,
    compilar_lambdify_raices,
)
from core.catalogo_ejercicios import (
    EXPRESIONES_PRECOMPUTADAS_PUNTO_FIJO,
    EXPRESIONES_PRECOMPUTADAS_RAICES,
)


def _wrap_escalar_real(f_call: Any) -> Callable[[float], float]:
    def inner(val: float) -> float:
        with np.errstate(all="ignore"):
            res = f_call(float(val))
            return float(np.real(res))
    return inner


def compilar_funcion_raices(f_str: str) -> dict[str, Any]:
    """
    Compila una expresión matemática f(x) y calcula f'(x) y f''(x).
    Usa diccionario precomputado para expresiones de cátedra y caché SymPy/NumPy para personalizadas.
    """
    clave = " ".join((f_str or "").strip().split())
    if clave in EXPRESIONES_PRECOMPUTADAS_RAICES:
        pre = EXPRESIONES_PRECOMPUTADAS_RAICES[clave]
        return {
            "x_sym": None,
            "f_simb": None,
            "d1_simb": None,
            "d2_simb": None,
            "f_latex": pre["f_latex"],
            "d1_latex": pre["d1_latex"],
            "d2_latex": pre["d2_latex"],
            "descartes": pre["descartes"],
            "fn": _wrap_escalar_real(pre["fn_np"]),
            "d1": _wrap_escalar_real(pre["d1_np"]),
            "d2": _wrap_escalar_real(pre["d2_np"]),
            "fn_np": pre["fn_np"],
            "d1_np": pre["d1_np"],
            "d2_np": pre["d2_np"],
        }

    meta = analizar_raices_sympy_cached(clave)
    fn_raw, d1_raw, d2_raw = compilar_lambdify_raices(clave)
    return {
        "x_sym": None,
        "f_simb": None,
        "d1_simb": None,
        "d2_simb": None,
        "f_latex": meta["f_latex"],
        "d1_latex": meta["d1_latex"],
        "d2_latex": meta["d2_latex"],
        "descartes": meta["descartes"],
        "fn": _wrap_escalar_real(fn_raw),
        "d1": _wrap_escalar_real(d1_raw),
        "d2": _wrap_escalar_real(d2_raw),
        "fn_np": fn_raw,
        "d1_np": d1_raw,
        "d2_np": d2_raw,
    }


def compilar_g_punto_fijo(g_str: str) -> dict[str, Any]:
    """Compila x = g(x) y g'(x) usando precomputación estática o caché vectorizada NumPy."""
    clave = " ".join((g_str or "").strip().split())
    if clave in EXPRESIONES_PRECOMPUTADAS_PUNTO_FIJO:
        pre = EXPRESIONES_PRECOMPUTADAS_PUNTO_FIJO[clave]
        return {
            "g_simb": None,
            "dg_simb": None,
            "g_latex": pre["g_latex"],
            "dg_latex": pre["dg_latex"],
            "gn": _wrap_escalar_real(pre["gn_np"]),
            "dgn": _wrap_escalar_real(pre["dgn_np"]),
            "gn_np": pre["gn_np"],
            "dgn_np": pre["dgn_np"],
        }

    meta = analizar_g_pf_sympy_cached(clave)
    gn_raw, dgn_raw = compilar_lambdify_pf(clave)
    return {
        "g_simb": None,
        "dg_simb": None,
        "g_latex": meta["g_latex"],
        "dg_latex": meta["dg_latex"],
        "gn": _wrap_escalar_real(gn_raw),
        "dgn": _wrap_escalar_real(dgn_raw),
        "gn_np": gn_raw,
        "dgn_np": dgn_raw,
    }


def analizar_descartes(f_simb: Any, x_sym: Any = None) -> dict[str, Any]:
    """Aplica la Regla de los Signos de Descartes (o reutiliza el resultado precomputado)."""
    if isinstance(f_simb, dict):
        if "descartes" in f_simb:
            return f_simb["descartes"]
        if "es_polinomio" in f_simb:
            return f_simb
    if isinstance(f_simb, str):
        clave = " ".join(f_simb.strip().split())
        if clave in EXPRESIONES_PRECOMPUTADAS_RAICES:
            return EXPRESIONES_PRECOMPUTADAS_RAICES[clave]["descartes"]
        return analizar_raices_sympy_cached(clave)["descartes"]

    import sympy as sp

    if x_sym is None:
        x_sym = sp.Symbol("x")
    if not f_simb.is_polynomial(x_sym) or f_simb == 0:
        return {
            "es_polinomio": False,
            "mensaje": "La función no es polinómica: el conteo de cambios de signo (Regla de Descartes) aplica únicamente a polinomios.",
        }

    poly_pos = sp.Poly(f_simb, x_sym)
    grado = poly_pos.degree()

    coefs_pos = [float(c) for _, c in poly_pos.terms() if abs(float(c)) > 1e-14]
    signos_pos = ["+" if c > 0 else "-" for c in coefs_pos]
    cambios_pos = sum(1 for i in range(len(signos_pos) - 1) if signos_pos[i] != signos_pos[i + 1])
    posibles_pos = list(range(cambios_pos, -1, -2))

    f_neg_simb = sp.expand(f_simb.subs(x_sym, -x_sym))
    poly_neg = sp.Poly(f_neg_simb, x_sym)
    coefs_neg = [float(c) for _, c in poly_neg.terms() if abs(float(c)) > 1e-14]
    signos_neg = ["+" if c > 0 else "-" for c in coefs_neg]
    cambios_neg = sum(1 for i in range(len(signos_neg) - 1) if signos_neg[i] != signos_neg[i + 1])
    posibles_neg = list(range(cambios_neg, -1, -2))

    return {
        "es_polinomio": True,
        "grado": grado,
        "f_pos_latex": sp.latex(f_simb),
        "f_neg_latex": sp.latex(f_neg_simb),
        "coefs_pos": coefs_pos,
        "signos_pos": signos_pos,
        "variaciones_pos": cambios_pos,
        "posibles_raices_pos": posibles_pos,
        "coefs_neg": coefs_neg,
        "signos_neg": signos_neg,
        "variaciones_neg": cambios_neg,
        "posibles_raices_neg": posibles_neg,
        "max_raices_positivas": max(posibles_pos) if posibles_pos else 0,
        "max_raices_negativas": max(posibles_neg) if posibles_neg else 0,
    }


def barrido_tanteo(
    fn: Callable[[float], float],
    x_ini: float = 0.0,
    dx: float = 0.5,
    max_pasos: int = 25,
    detener_en_primero: bool = False,
) -> dict[str, Any]:
    """
    Evalúa f(x) con paso dx detectando todos los cambios de signo (Teorema de Bolzano).
    """
    if abs(dx) < 1e-12:
        raise ValueError("El paso de tanteo Δx no puede ser cero.")

    filas: list[dict[str, Any]] = []
    intervalos_hallados: list[tuple[float, float]] = []

    x_ant = float(x_ini)
    f_ant = float(fn(x_ant))
    signo_ant = "+" if f_ant > 0 else ("-" if f_ant < 0 else "0 (Raíz)")
    filas.append({
        "paso": 0,
        "x_k": x_ant,
        "f_xk": f_ant,
        "signo": signo_ant,
        "cambio_signo": abs(f_ant) < 1e-12,
        "observacion": "Inicio del tanteo" if abs(f_ant) >= 1e-12 else "🎯 Raíz exacta en inicio",
    })
    if abs(f_ant) < 1e-12:
        intervalos_hallados.append((x_ant, x_ant))

    for k in range(1, max_pasos + 1):
        x_act = round(float(x_ini + k * dx), 8)
        try:
            f_act = float(fn(x_act))
        except Exception:
            filas.append({
                "paso": k, "x_k": x_act, "f_xk": float("nan"),
                "signo": "Indefinido", "cambio_signo": False,
                "observacion": "Fuera de dominio / asíntota",
            })
            break

        signo_act = "+" if f_act > 0 else ("-" if f_act < 0 else "0 (Raíz)")
        cambio = False
        if abs(f_act) < 1e-12:
            intervalos_hallados.append((x_act, x_act))
            obs = f"🎯 Raíz exacta en x = {x_act:g}"
            cambio = True
        elif f_ant * f_act < 0:
            a_int, b_int = min(x_ant, x_act), max(x_ant, x_act)
            intervalos_hallados.append((a_int, b_int))
            obs = f"✅ Cambio de signo en ({a_int:g} ; {b_int:g})"
            cambio = True
        else:
            obs = "Sin cambio de signo"

        filas.append({
            "paso": k, "x_k": x_act, "f_xk": f_act,
            "signo": signo_act, "cambio_signo": cambio, "observacion": obs,
        })
        if cambio and detener_en_primero:
            break
        x_ant, f_ant = x_act, f_act

    return {
        "intervalo": intervalos_hallados[0] if intervalos_hallados else None,
        "intervalos": intervalos_hallados,
        "historial": filas,
        "encontrado": len(intervalos_hallados) > 0,
    }


def _validar_parametros_intervalo(
    fn: Callable[[float], float],
    x_a: float,
    x_b: float,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    """Valida extremos, tolerancia, iteraciones y condición de Bolzano sin reordenar los argumentos."""
    if not (math.isfinite(float(x_a)) and math.isfinite(float(x_b))):
        raise ValueError("Los extremos del intervalo deben ser valores reales finitos.")
    if float(tol) <= 0:
        raise ValueError("La tolerancia E debe ser estrictamente positiva.")
    if int(max_iter) < 1:
        raise ValueError("La cantidad máxima de iteraciones debe ser al menos 1.")
    if abs(float(x_b) - float(x_a)) <= 1e-15:
        raise ValueError("Los extremos a y b deben ser distintos para definir un intervalo válido.")

    fa, fb = float(fn(float(x_a))), float(fn(float(x_b)))
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError("f(a) o f(b) produjo un valor no finito.")
    if fa * fb >= 0:
        raise ValueError(
            f"No se cumple Bolzano en [{min(x_a, x_b):g}, {max(x_a, x_b):g}]: "
            f"f({x_a:g})={fa:g}, f({x_b:g})={fb:g}."
        )
    return fa, fb


def metodo_biseccion(
    fn: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Método de Bisección (Intervalo Medio) con geometría paso a paso."""
    t_ini = time.perf_counter()
    a_act = float(min(a, b))
    b_act = float(max(a, b))
    fa, fb = _validar_parametros_intervalo(fn, a_act, b_act, tol, max_iter)
    amplitud = b_act - a_act
    evals_f = 2

    n_teorico_exacto = (math.log10(amplitud) - math.log10(tol)) / math.log10(2.0)
    n_teorico_entero = max(1, int(math.ceil(n_teorico_exacto)))

    historial: list[dict[str, Any]] = []
    x_ant: float | None = None
    err: float | None = None

    for k in range(1, int(max_iter) + 1):
        xk = 0.5 * (a_act + b_act)
        fxk = float(fn(xk))
        evals_f += 1
        err = abs(xk - x_ant) if x_ant is not None else 0.5 * (b_act - a_act)

        if fa * fxk < 0:
            nuevo_a, nuevo_b, next_fa, next_fb = a_act, xk, fa, fxk
            decision = f"f({a_act:g})·f({xk:g}) < 0 ⇒ [{a_act:g} ; {xk:g}]"
        else:
            nuevo_a, nuevo_b, next_fa, next_fb = xk, b_act, fxk, fb
            decision = f"f({xk:g})·f({b_act:g}) < 0 ⇒ [{xk:g} ; {b_act:g}]"

        cumple = (x_ant is not None and err < tol) or abs(fxk) < 1e-14
        historial.append({
            "iteracion": k,
            "a": a_act,
            "b": b_act,
            "x_k": xk,
            "f_a": fa,
            "f_b": fb,
            "f_xk": fxk,
            "error": err if x_ant is not None else None,
            "cumple_tol": cumple,
            "nuevo_a": nuevo_a,
            "nuevo_b": nuevo_b,
            "decision": decision,
        })

        if cumple:
            return {
                "raiz": xk,
                "iteraciones": k,
                "evals_f": evals_f,
                "error_final": err if err is not None else 0.0,
                "f_raiz": fxk,
                "convergente": True,
                "n_teorico_exacto": n_teorico_exacto,
                "n_teorico_entero": n_teorico_entero,
                "duracion": time.perf_counter() - t_ini,
                "historial": historial,
            }
        a_act, b_act, fa, fb, x_ant = nuevo_a, nuevo_b, next_fa, next_fb, xk

    xk_fin = x_ant if x_ant is not None else a_act
    return {
        "raiz": xk_fin,
        "iteraciones": int(max_iter),
        "evals_f": evals_f,
        "error_final": err if err is not None else amplitud,
        "f_raiz": float(fn(xk_fin)),
        "convergente": False,
        "n_teorico_exacto": n_teorico_exacto,
        "n_teorico_entero": n_teorico_entero,
        "duracion": time.perf_counter() - t_ini,
        "historial": historial,
    }


def metodo_interpolacion_lineal_catedra(
    fn: Callable[[float], float],
    x0_fijo: float,
    x1_ini: float,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> dict[str, Any]:
    """Interpolación Lineal (Regula Falsi) con Extremo Fijo x0 (Cátedra UNNE)."""
    t_ini = time.perf_counter()
    fx0, fxi = _validar_parametros_intervalo(fn, x0_fijo, x1_ini, tol, max_iter)
    xi = float(x1_ini)
    evals_f = 2

    historial: list[dict[str, Any]] = []
    err = abs(xi - x0_fijo)

    for k in range(1, int(max_iter) + 1):
        denom = fxi - fx0
        if abs(denom) < 1e-15:
            return {
                "raiz": xi, "iteraciones": k - 1, "evals_f": evals_f, "error_final": err, "f_raiz": fxi,
                "convergente": False, "mensaje": "División por cero: f(x_i) - f(x_0) = 0",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        num = float(x0_fijo) * fxi - xi * fx0
        x_sig = num / denom
        fx_sig = float(fn(x_sig))
        evals_f += 1
        err = abs(x_sig - xi)
        cumple = (k > 1 and err < tol) or abs(fx_sig) < 1e-14

        historial.append({
            "iteracion": k,
            "subindice": f"x_{k + 1}",
            "x0_fijo": float(x0_fijo),
            "f_x0": fx0,
            "x_i": xi,
            "f_xi": fxi,
            "x_sig": x_sig,
            "f_xsig": fx_sig,
            "error": err if k > 1 else None,
            "cumple_tol": cumple,
        })

        if cumple:
            return {
                "raiz": x_sig, "iteraciones": k, "evals_f": evals_f, "error_final": err, "f_raiz": fx_sig,
                "convergente": True, "mensaje": "Convergencia exitosa alcanzada.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }
        xi, fxi = x_sig, fx_sig

    return {
        "raiz": xi, "iteraciones": int(max_iter), "evals_f": evals_f, "error_final": err, "f_raiz": fxi,
        "convergente": False, "mensaje": "Se alcanzó el número máximo de iteraciones.",
        "duracion": time.perf_counter() - t_ini, "historial": historial,
    }


def metodo_interpolacion_lineal_bolzano(
    fn: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Regula Falsi Clásica actualizando ambos extremos del intervalo según el signo."""
    t_ini = time.perf_counter()
    a_act, b_act = float(min(a, b)), float(max(a, b))
    fa, fb = _validar_parametros_intervalo(fn, a_act, b_act, tol, max_iter)
    evals_f = 2

    historial: list[dict[str, Any]] = []
    x_ant: float | None = None
    err = abs(b_act - a_act)

    for k in range(1, int(max_iter) + 1):
        denom = fb - fa
        if abs(denom) < 1e-15:
            break
        x_sig = (a_act * fb - b_act * fa) / denom
        fx_sig = float(fn(x_sig))
        evals_f += 1
        err = abs(x_sig - x_ant) if x_ant is not None else abs(b_act - a_act)
        cumple = (x_ant is not None and err < tol) or abs(fx_sig) < 1e-14

        historial.append({
            "iteracion": k,
            "a": a_act,
            "b": b_act,
            "f_a": fa,
            "f_b": fb,
            "x_k": x_sig,
            "f_xk": fx_sig,
            "error": err if x_ant is not None else None,
            "cumple_tol": cumple,
        })

        if cumple:
            return {
                "raiz": x_sig, "iteraciones": k, "evals_f": evals_f, "error_final": err, "f_raiz": fx_sig,
                "convergente": True, "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        if fa * fx_sig < 0:
            b_act, fb = x_sig, fx_sig
        else:
            a_act, fa = x_sig, fx_sig
        x_ant = x_sig

    xk_fin = x_ant if x_ant is not None else a_act
    return {
        "raiz": xk_fin, "iteraciones": int(max_iter), "evals_f": evals_f, "error_final": err,
        "f_raiz": float(fn(xk_fin)), "convergente": False,
        "duracion": time.perf_counter() - t_ini, "historial": historial,
    }


def verificar_condiciones_fourier(
    fn: Callable[[float], float],
    d1: Callable[[float], float],
    d2: Callable[[float], float],
    a: float,
    b: float,
) -> dict[str, Any]:
    """Verifica Bolzano, monotonía y Condición de Fourier f(x0)·f''(x0) > 0."""
    a_f, b_f = float(a), float(b)
    if not (math.isfinite(a_f) and math.isfinite(b_f)):
        raise ValueError("Los extremos a y b para verificar Fourier deben ser finitos.")
    if abs(b_f - a_f) <= 1e-15:
        raise ValueError("Los extremos a y b deben ser distintos para verificar el intervalo.")

    fa, fb = float(fn(a_f)), float(fn(b_f))
    d1a, d1b = float(d1(a_f)), float(d1(b_f))
    d2a, d2b = float(d2(a_f)), float(d2(b_f))
    if not all(math.isfinite(v) for v in (fa, fb, d1a, d1b, d2a, d2b)):
        raise ValueError("f, f' o f'' produjo un valor no finito en los extremos del intervalo.")

    cumple_bolzano = fa * fb < 0
    monotona_bordes = (d1a * d1b > 0) and abs(d1a) > 1e-12 and abs(d1b) > 1e-12
    fourier_a = fa * d2a
    fourier_b = fb * d2b

    if fourier_a > 0:
        x0_sugerido = a_f
        fourier_cumple = True
        detalle = f"f({a_f:g})·f''({a_f:g}) = {fourier_a:+.3e} > 0 ⇒ Punto de arranque seguro x₀ = {a_f:g}"
    elif fourier_b > 0:
        x0_sugerido = b_f
        fourier_cumple = True
        detalle = f"f({b_f:g})·f''({b_f:g}) = {fourier_b:+.3e} > 0 ⇒ Punto de arranque seguro x₀ = {b_f:g}"
    else:
        x0_sugerido = a_f if abs(fa) < abs(fb) else b_f
        fourier_cumple = False
        detalle = f"Ningún borde cumple estrictamente f·f'' > 0. Se sugiere el más cercano al eje: x₀ = {x0_sugerido:g}"

    return {
        "cumple_bolzano": cumple_bolzano,
        "monotona_bordes": monotona_bordes,
        "fourier_cumple": fourier_cumple,
        "x0_sugerido": x0_sugerido,
        "fourier_a": fourier_a,
        "fourier_b": fourier_b,
        "detalle_fourier": detalle,
    }


def metodo_newton_raphson(
    fn: Callable[[float], float],
    d1: Callable[[float], float],
    x0: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> dict[str, Any]:
    """Newton-Raphson con registro completo de tangentes paso a paso."""
    t_ini = time.perf_counter()
    x_val = float(x0)
    tol_f = float(tol)
    max_it = int(max_iter)
    if not math.isfinite(x_val):
        raise ValueError("La semilla inicial x₀ debe ser un número real finito.")
    if tol_f <= 0:
        raise ValueError("La tolerancia E debe ser estrictamente positiva.")
    if max_it < 1:
        raise ValueError("La cantidad máxima de iteraciones debe ser al menos 1.")

    historial: list[dict[str, Any]] = []
    err = float("inf")
    evals_f = 0

    for k in range(1, max_it + 1):
        try:
            fx = float(fn(x_val))
            dfx = float(d1(x_val))
            evals_f += 1
        except Exception:
            fx, dfx = float("nan"), float("nan")

        if not (math.isfinite(fx) and math.isfinite(dfx)):
            return {
                "raiz": x_val,
                "iteraciones": len(historial),
                "evals_f": evals_f,
                "error_final": err if math.isfinite(err) else 0.0,
                "f_raiz": 0.0 if not math.isfinite(fx) else fx,
                "convergente": False,
                "mensaje": f"Evaluación no finita o fuera de dominio en iteración {k} (x = {x_val:g}).",
                "duracion": time.perf_counter() - t_ini,
                "historial": historial,
            }

        if abs(dfx) < 1e-14:
            historial.append({
                "iteracion": k, "x_k": x_val, "f_xk": fx, "df_xk": dfx,
                "x_sig": None, "error": None, "cumple_tol": False,
            })
            return {
                "raiz": x_val, "iteraciones": k, "evals_f": evals_f, "error_final": abs(fx), "f_raiz": fx,
                "convergente": False,
                "mensaje": f"Derivada casi nula en iteración {k} (|f'(x)| < 1e-14): tangente horizontal.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        x_sig = x_val - (fx / dfx)
        if not math.isfinite(x_sig):
            return {
                "raiz": x_val, "iteraciones": k - 1, "evals_f": evals_f, "error_final": err, "f_raiz": fx,
                "convergente": False, "mensaje": "El salto secante/tangente produjo un valor no finito.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        err = abs(x_sig - x_val)
        cumple = err < tol_f or abs(fx) < 1e-14

        historial.append({
            "iteracion": k,
            "x_k": x_val,
            "f_xk": fx,
            "df_xk": dfx,
            "x_sig": x_sig,
            "error": err,
            "cumple_tol": cumple,
        })

        if cumple:
            try:
                f_fin = float(fn(x_sig))
                evals_f += 1
            except Exception:
                f_fin = fx
            return {
                "raiz": x_sig, "iteraciones": k, "evals_f": evals_f, "error_final": err,
                "f_raiz": f_fin if math.isfinite(f_fin) else fx,
                "convergente": True, "mensaje": "Convergencia cuadrática exitosa.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        if abs(x_sig) > 1e10:
            return {
                "raiz": x_val, "iteraciones": k, "evals_f": evals_f, "error_final": err, "f_raiz": fx,
                "convergente": False, "mensaje": "El método divergió fuera de rango numérico.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }
        x_val = x_sig

    try:
        f_fin = float(fn(x_val))
        evals_f += 1
    except Exception:
        f_fin = float("nan")
    return {
        "raiz": x_val, "iteraciones": max_it, "evals_f": evals_f, "error_final": err,
        "f_raiz": f_fin, "convergente": False, "mensaje": "Se alcanzó el máximo de iteraciones sin converger.",
        "duracion": time.perf_counter() - t_ini, "historial": historial,
    }


def _eval_phi_seguro(phi_fn: Callable[[float], float], val: float) -> float | None:
    """Evalúa phi_fn(val) garantizando finitud y cota numérica."""
    if not math.isfinite(val):
        return None
    try:
        res = float(phi_fn(val))
    except Exception:
        return None
    if not math.isfinite(res) or abs(res) > 1e8:
        return None
    return res


def metodo_punto_fijo_aitken(
    phi_fn: Callable[[float], float],
    x0: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> dict[str, Any]:
    """Ejecuta Punto Fijo simple y Aceleración Δ² de Aitken con protección ante divergencia."""
    x0_f = float(x0)
    tol_f = float(tol)
    max_it = int(max_iter)
    if not math.isfinite(x0_f):
        raise ValueError("La semilla inicial x₀ debe ser un número real finito.")
    if tol_f <= 0:
        raise ValueError("La tolerancia E debe ser estrictamente positiva.")
    if max_it < 1:
        raise ValueError("La cantidad máxima de iteraciones debe ser al menos 1.")

    t0 = time.perf_counter()
    x_act = x0_f
    hist_pf: list[dict[str, Any]] = []
    convergente_pf = False

    for k in range(1, max_it + 1):
        x_sig = _eval_phi_seguro(phi_fn, x_act)
        if x_sig is None:
            break
        err = abs(x_sig - x_act)
        cumple = err < tol_f
        hist_pf.append({
            "iteracion": k, "x_ant": x_act, "x_sig": x_sig, "error": err, "cumple_tol": cumple,
        })
        x_act = x_sig
        if cumple:
            convergente_pf = True
            break
    t_pf = time.perf_counter() - t0

    t1 = time.perf_counter()
    x_base = x0_f
    hist_aitken: list[dict[str, Any]] = []
    convergente_aitken = False

    for ciclo in range(1, max_it + 1):
        x1 = _eval_phi_seguro(phi_fn, x_base)
        if x1 is None:
            break
        x2 = _eval_phi_seguro(phi_fn, x1)
        if x2 is None:
            break
        denom = x2 - 2.0 * x1 + x_base
        s_aitken = x2 if abs(denom) < 1e-15 else x2 - ((x2 - x1) ** 2) / denom
        if not math.isfinite(s_aitken) or abs(s_aitken) > 1e8:
            break
        x_post = _eval_phi_seguro(phi_fn, s_aitken)
        if x_post is None:
            break
        err_aitken = abs(x_post - s_aitken)
        cumple_a = err_aitken < tol_f

        hist_aitken.append({
            "ciclo": ciclo, "x0": x_base, "x1": x1, "x2": x2,
            "numerador": (x2 - x1) ** 2, "denominador": denom,
            "s_acelerado": s_aitken, "x_post": x_post,
            "error": err_aitken, "cumple_tol": cumple_a,
        })
        if cumple_a:
            convergente_aitken = True
            break
        x_base = x_post
    t_aitken = time.perf_counter() - t1

    return {
        "punto_fijo": {
            "raiz": x_act,
            "iteraciones": len(hist_pf),
            "error_final": hist_pf[-1]["error"] if hist_pf else float("inf"),
            "convergente": convergente_pf,
            "tiempo": t_pf,
            "historial": hist_pf,
        },
        "aitken": {
            "raiz": hist_aitken[-1]["s_acelerado"] if hist_aitken else x0_f,
            "ciclos": len(hist_aitken),
            "error_final": hist_aitken[-1]["error"] if hist_aitken else float("inf"),
            "convergente": convergente_aitken,
            "tiempo": t_aitken,
            "historial": hist_aitken,
        },
    }
