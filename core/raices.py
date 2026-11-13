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
import sympy as sp


def compilar_funcion_raices(f_str: str) -> dict[str, Any]:
    """
    Compila una expresión matemática f(x) y calcula automáticamente f'(x) y f''(x) con SymPy.
    Retorna funciones numéricas invocables y sus representaciones LaTeX.
    """
    x_sym = sp.Symbol("x")
    entorno = {"x": x_sym, "e": sp.E, "pi": sp.pi, "ln": sp.log}
    try:
        f_simb = sp.sympify(f_str, locals=entorno)
        d1_simb = sp.diff(f_simb, x_sym)
        d2_simb = sp.diff(d1_simb, x_sym)
        fn_raw = sp.lambdify(x_sym, f_simb, modules=["numpy", "math"])
        d1_raw = sp.lambdify(x_sym, d1_simb, modules=["numpy", "math"])
        d2_raw = sp.lambdify(x_sym, d2_simb, modules=["numpy", "math"])
    except Exception as e:
        raise ValueError(f"Expresión matemática inválida '{f_str}': {e}")

    def _wrap(f_call: Any) -> Callable[[float], float]:
        def inner(val: float) -> float:
            res = f_call(float(val))
            return float(np.real(res))
        return inner

    return {
        "x_sym": x_sym,
        "f_simb": f_simb,
        "d1_simb": d1_simb,
        "d2_simb": d2_simb,
        "f_latex": sp.latex(f_simb),
        "d1_latex": sp.latex(d1_simb),
        "d2_latex": sp.latex(d2_simb),
        "fn": _wrap(fn_raw),
        "d1": _wrap(d1_raw),
        "d2": _wrap(d2_raw),
    }


def compilar_g_punto_fijo(g_str: str) -> dict[str, Any]:
    """Compila x = g(x) y g'(x) para verificar contractividad |g'(x)| < 1."""
    x_sym = sp.Symbol("x")
    entorno = {"x": x_sym, "e": sp.E, "pi": sp.pi, "ln": sp.log}
    try:
        g_simb = sp.sympify(g_str, locals=entorno)
        dg_simb = sp.diff(g_simb, x_sym)
        gn_raw = sp.lambdify(x_sym, g_simb, modules=["numpy", "math"])
        dg_raw = sp.lambdify(x_sym, dg_simb, modules=["numpy", "math"])
    except Exception as e:
        raise ValueError(f"Expresión g(x) inválida '{g_str}': {e}")

    return {
        "g_simb": g_simb,
        "dg_simb": dg_simb,
        "g_latex": sp.latex(g_simb),
        "dg_latex": sp.latex(dg_simb),
        "gn": lambda v: float(np.real(gn_raw(float(v)))),
        "dgn": lambda v: float(np.real(dg_raw(float(v)))),
    }


def analizar_descartes(f_simb: sp.Expr, x_sym: sp.Symbol) -> dict[str, Any]:
    """Aplica la Regla de los Signos de Descartes a un polinomio simbólico."""
    if not f_simb.is_polynomial(x_sym) or f_simb == 0:
        return {
            "es_polinomio": False,
            "mensaje": "La función es trascendente o no polinómica: la Regla de Descartes aplica únicamente a polinomios.",
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
    fa, fb = float(fn(a_act)), float(fn(b_act))

    if fa * fb >= 0:
        raise ValueError(
            f"No se cumple Bolzano en [{a_act:g}, {b_act:g}]: f({a_act:g})={fa:g}, f({b_act:g})={fb:g}."
        )

    amplitud = b_act - a_act
    n_teorico_exacto = (math.log10(amplitud) - math.log10(tol)) / math.log10(2.0)
    n_teorico_entero = max(1, int(math.ceil(n_teorico_exacto)))

    historial: list[dict[str, Any]] = []
    x_ant: float | None = None
    err: float | None = None

    for k in range(1, max_iter + 1):
        fa = float(fn(a_act))
        fb = float(fn(b_act))
        xk = 0.5 * (a_act + b_act)
        fxk = float(fn(xk))
        err = abs(xk - x_ant) if x_ant is not None else 0.5 * (b_act - a_act)

        if fa * fxk < 0:
            nuevo_a, nuevo_b = a_act, xk
            decision = f"f({a_act:g})·f({xk:g}) < 0 ⇒ [{a_act:g} ; {xk:g}]"
        else:
            nuevo_a, nuevo_b = xk, b_act
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
                "error_final": err if err is not None else 0.0,
                "f_raiz": fxk,
                "convergente": True,
                "n_teorico_exacto": n_teorico_exacto,
                "n_teorico_entero": n_teorico_entero,
                "duracion": time.perf_counter() - t_ini,
                "historial": historial,
            }
        a_act, b_act, x_ant = nuevo_a, nuevo_b, xk

    xk_fin = x_ant if x_ant is not None else a_act
    return {
        "raiz": xk_fin,
        "iteraciones": max_iter,
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
    fx0 = float(fn(x0_fijo))
    xi = float(x1_ini)
    fxi = float(fn(xi))

    if fx0 * fxi >= 0:
        raise ValueError(
            f"f(x0) y f(x1) deben tener signos opuestos por Bolzano: f({x0_fijo:g})={fx0:g}, f({x1_ini:g})={fxi:g}."
        )

    historial: list[dict[str, Any]] = []
    err = abs(xi - x0_fijo)

    for k in range(1, max_iter + 1):
        fxi = float(fn(xi))
        denom = fxi - fx0
        if abs(denom) < 1e-15:
            return {
                "raiz": xi, "iteraciones": k - 1, "error_final": err, "f_raiz": fxi,
                "convergente": False, "mensaje": "División por cero: f(x_i) - f(x_0) = 0",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        num = x0_fijo * fxi - xi * fx0
        x_sig = num / denom
        fx_sig = float(fn(x_sig))
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
                "raiz": x_sig, "iteraciones": k, "error_final": err, "f_raiz": fx_sig,
                "convergente": True, "mensaje": "Convergencia exitosa alcanzada.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }
        xi = x_sig

    return {
        "raiz": xi, "iteraciones": max_iter, "error_final": err, "f_raiz": float(fn(xi)),
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
    fa, fb = float(fn(a_act)), float(fn(b_act))

    if fa * fb >= 0:
        raise ValueError(f"No se cumple Bolzano en [{a_act:g}, {b_act:g}].")

    historial: list[dict[str, Any]] = []
    x_ant: float | None = None
    err = abs(b_act - a_act)

    for k in range(1, max_iter + 1):
        fa, fb = float(fn(a_act)), float(fn(b_act))
        denom = fb - fa
        if abs(denom) < 1e-15:
            break
        x_sig = (a_act * fb - b_act * fa) / denom
        fx_sig = float(fn(x_sig))
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
                "raiz": x_sig, "iteraciones": k, "error_final": err, "f_raiz": fx_sig,
                "convergente": True, "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        if fa * fx_sig < 0:
            b_act = x_sig
        else:
            a_act = x_sig
        x_ant = x_sig

    xk_fin = x_ant if x_ant is not None else a_act
    return {
        "raiz": xk_fin, "iteraciones": max_iter, "error_final": err,
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
    fa, fb = float(fn(a)), float(fn(b))
    d1a, d1b = float(d1(a)), float(d1(b))
    d2a, d2b = float(d2(a)), float(d2(b))

    cumple_bolzano = fa * fb < 0
    monotona_bordes = (d1a * d1b > 0) and abs(d1a) > 1e-12 and abs(d1b) > 1e-12
    fourier_a = fa * d2a
    fourier_b = fb * d2b

    if fourier_a > 0:
        x0_sugerido = float(a)
        fourier_cumple = True
        detalle = f"f({a:g})·f''({a:g}) = {fourier_a:+.3e} > 0 ⇒ Punto de arranque seguro x₀ = {a:g}"
    elif fourier_b > 0:
        x0_sugerido = float(b)
        fourier_cumple = True
        detalle = f"f({b:g})·f''({b:g}) = {fourier_b:+.3e} > 0 ⇒ Punto de arranque seguro x₀ = {b:g}"
    else:
        x0_sugerido = float(a if abs(fa) < abs(fb) else b)
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
    historial: list[dict[str, Any]] = []
    err = float("inf")

    for k in range(1, max_iter + 1):
        fx = float(fn(x_val))
        dfx = float(d1(x_val))

        if abs(dfx) < 1e-14:
            historial.append({
                "iteracion": k, "x_k": x_val, "f_xk": fx, "df_xk": dfx,
                "x_sig": None, "error": None, "cumple_tol": False,
            })
            return {
                "raiz": x_val, "iteraciones": k, "error_final": abs(fx), "f_raiz": fx,
                "convergente": False,
                "mensaje": f"Derivada casi nula en iteración {k} (|f'(x)| < 1e-14): tangente horizontal.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        x_sig = x_val - (fx / dfx)
        err = abs(x_sig - x_val)
        cumple = err < tol or abs(fx) < 1e-14

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
            return {
                "raiz": x_sig, "iteraciones": k, "error_final": err, "f_raiz": float(fn(x_sig)),
                "convergente": True, "mensaje": "Convergencia cuadrática exitosa.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }

        if abs(x_sig) > 1e10 or math.isnan(x_sig):
            return {
                "raiz": x_val, "iteraciones": k, "error_final": err, "f_raiz": fx,
                "convergente": False, "mensaje": "El método divergió fuera de rango numérico.",
                "duracion": time.perf_counter() - t_ini, "historial": historial,
            }
        x_val = x_sig

    return {
        "raiz": x_val, "iteraciones": max_iter, "error_final": err, "f_raiz": float(fn(x_val)),
        "convergente": False, "mensaje": "Se alcanzó el máximo de iteraciones sin converger.",
        "duracion": time.perf_counter() - t_ini, "historial": historial,
    }


def metodo_punto_fijo_aitken(
    phi_fn: Callable[[float], float],
    x0: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> dict[str, Any]:
    """Ejecuta Punto Fijo simple y Aceleración Δ² de Aitken con protección ante divergencia."""
    t0 = time.perf_counter()
    x_act = float(x0)
    hist_pf: list[dict[str, Any]] = []
    convergente_pf = False

    for k in range(1, max_iter + 1):
        try:
            x_sig = float(phi_fn(x_act))
        except Exception:
            break
        if math.isnan(x_sig) or math.isinf(x_sig) or abs(x_sig) > 1e8:
            break
        err = abs(x_sig - x_act)
        cumple = err < tol
        hist_pf.append({
            "iteracion": k, "x_ant": x_act, "x_sig": x_sig, "error": err, "cumple_tol": cumple,
        })
        x_act = x_sig
        if cumple:
            convergente_pf = True
            break
    t_pf = time.perf_counter() - t0

    t1 = time.perf_counter()
    x_base = float(x0)
    hist_aitken: list[dict[str, Any]] = []
    convergente_aitken = False

    for ciclo in range(1, max_iter + 1):
        try:
            x1 = float(phi_fn(x_base))
            x2 = float(phi_fn(x1))
        except Exception:
            break
        if any(math.isnan(v) or math.isinf(v) or abs(v) > 1e8 for v in (x1, x2)):
            break
        denom = x2 - 2.0 * x1 + x_base
        s_aitken = x2 if abs(denom) < 1e-15 else x2 - ((x2 - x1) ** 2) / denom
        try:
            x_post = float(phi_fn(s_aitken))
        except Exception:
            break
        err_aitken = abs(x_post - s_aitken)
        cumple_a = err_aitken < tol

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
            "raiz": hist_aitken[-1]["s_acelerado"] if hist_aitken else x0,
            "ciclos": len(hist_aitken),
            "error_final": hist_aitken[-1]["error"] if hist_aitken else float("inf"),
            "convergente": convergente_aitken,
            "tiempo": t_aitken,
            "historial": hist_aitken,
        },
    }
