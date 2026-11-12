"""
core/lab3_raices.py
===================
Módulo de cálculo puro para la Búsqueda de Raíces de Ecuaciones No Lineales (Laboratorio 3).

Incluye:
- Regla de los Signos de Descartes (polinomios simbólicos y cotas de raíces).
- Método de Tanteos (separación y aislamiento de intervalos por Teorema de Bolzano).
- Método de Bisección (Intervalo Medio) con cálculo de iteraciones teóricas mínimas.
- Método de Interpolación Lineal (Regula Falsi):
    * Variante con Extremo Fijo (según consignas oficiales de cátedra).
    * Variante Clásica de Bolzano (actualización de corchetes dinámicos).
- Verificación de Condiciones de Convergencia de Fourier:
    * Bolzano, Monotonía (f' != 0) y Condición de Fourier (f(x0)*f''(x0) > 0).
- Método de Newton-Raphson (Tangentes) con control de derivadas nulas.
- Método de Iteración (Punto Fijo) y Aceleración Delta^2 de Aitken.

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
# 1. ANÁLISIS PREVIO: REGLA DE DESCARTES Y TANTEOS (BOLZANO)
# ==============================================================================

def analizar_descartes(f_simb: sp.Expr, x_sym: sp.Symbol) -> dict[str, Any]:
    """
    Aplica la Regla de los Signos de Descartes para determinar el número posible
    de raíces reales positivas y negativas de un polinomio.

    Parámetros:
    -----------
    f_simb : sp.Expr
        Expresión polinómica de SymPy.
    x_sym : sp.Symbol
        Símbolo de la variable independiente (típicamente sp.Symbol('x')).

    Retorna:
    --------
    dict con grado, coeficientes, variaciones de signo y listas de posibles raíces.
    """
    if not f_simb.is_polynomial(x_sym) or f_simb == 0:
        raise ValueError("La expresión debe ser un polinomio no nulo para aplicar la Regla de Descartes.")

    poly_pos = sp.Poly(f_simb, x_sym)
    grado = poly_pos.degree()

    # Variaciones de signo para x > 0
    coefs_pos = [float(c) for _, c in poly_pos.terms() if abs(float(c)) > 1e-14]
    signos_pos = ["+" if c > 0 else "-" for c in coefs_pos]
    cambios_pos = sum(1 for i in range(len(signos_pos) - 1) if signos_pos[i] != signos_pos[i + 1])
    posibles_pos = list(range(cambios_pos, -1, -2))

    # P(-x) para variaciones de signo con x < 0
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
) -> dict[str, Any]:
    """
    Aplica el Método de Tanteos evaluando f(x_k) paso a paso a partir de x_ini con paso dx
    hasta detectar un cambio de signo (Teorema de Bolzano: f(a)*f(b) < 0) o una raíz exacta.

    Retorna:
    --------
    dict con historial paso a paso y el intervalo aislado (a, b) si se encontró.
    """
    if abs(dx) < 1e-12:
        raise ValueError("El paso de tanteo dx no puede ser cero.")

    filas: list[dict[str, Any]] = []
    intervalo_hallado: tuple[float, float] | None = None

    x_ant = float(x_ini)
    try:
        f_ant = float(fn(x_ant))
    except Exception as e:
        raise ValueError(f"Error al evaluar la función en el punto inicial x0 = {x_ant}: {e}")

    signo_ant = "+" if f_ant > 0 else ("-" if f_ant < 0 else "0 (Raíz)")
    filas.append({
        "paso": 0,
        "x_k": x_ant,
        "f_xk": f_ant,
        "signo": signo_ant,
        "observacion": f"Inicio del tanteo en x₀ = {x_ant:g}",
    })

    if abs(f_ant) < 1e-12:
        intervalo_hallado = (x_ant, x_ant)
        filas[0]["observacion"] = f"🎯 Raíz exacta en punto de partida x = {x_ant:g}"
        return {
            "intervalo": intervalo_hallado,
            "historial": filas,
            "encontrado": True,
        }

    for k in range(1, max_pasos + 1):
        x_act = round(float(x_ini + k * dx), 8)
        try:
            f_act = float(fn(x_act))
        except Exception:
            filas.append({
                "paso": k,
                "x_k": x_act,
                "f_xk": float("nan"),
                "signo": "Indefinido",
                "observacion": "Error de evaluación (posible asíntota o fuera de dominio)",
            })
            break

        signo_act = "+" if f_act > 0 else ("-" if f_act < 0 else "0 (Raíz)")

        if abs(f_act) < 1e-12:
            intervalo_hallado = (x_act, x_act)
            obs = f"🎯 Raíz exacta encontrada en x = {x_act:g}"
            filas.append({"paso": k, "x_k": x_act, "f_xk": f_act, "signo": signo_act, "observacion": obs})
            break
        elif f_ant * f_act < 0:
            a_int = min(x_ant, x_act)
            b_int = max(x_ant, x_act)
            intervalo_hallado = (a_int, b_int)
            obs = f"✅ Cambio de signo detectado: f({x_ant:g})={f_ant:+.3e} y f({x_act:g})={f_act:+.3e} ⇒ Raíz en ({a_int:g} ; {b_int:g})"
            filas.append({"paso": k, "x_k": x_act, "f_xk": f_act, "signo": signo_act, "observacion": obs})
            break
        else:
            obs = "Mismo signo, continúa el tanteo"
            filas.append({"paso": k, "x_k": x_act, "f_xk": f_act, "signo": signo_act, "observacion": obs})

        x_ant = x_act
        f_ant = f_act

    return {
        "intervalo": intervalo_hallado,
        "historial": filas,
        "encontrado": intervalo_hallado is not None,
    }


# ==============================================================================
# 2. MÉTODO DE BISECCIÓN (INTERVALO MEDIO)
# ==============================================================================

def metodo_biseccion(
    fn: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> dict[str, Any]:
    """
    Ejecuta el Método de Bisección sobre el intervalo [a, b].
    Calcula la cota mínima teórica de iteraciones según el criterio:
        n_teorico = ceil((log10(b - a) - log10(tol)) / log10(2))

    Retorna un diccionario completo con la raíz hallada, historial paso a paso y métricas.
    """
    t_ini = time.perf_counter()
    a_act = float(min(a, b))
    b_act = float(max(a, b))

    fa = float(fn(a_act))
    fb = float(fn(b_act))

    if fa * fb >= 0:
        raise ValueError(
            f"No se cumple el Teorema de Bolzano en [{a_act:g}, {b_act:g}]: "
            f"f({a_act:g}) = {fa:g} y f({b_act:g}) = {fb:g} tienen el mismo signo."
        )

    # Cálculo teórico de iteraciones
    amplitud = b_act - a_act
    n_teorico_exacto = (math.log10(amplitud) - math.log10(tol)) / math.log10(2.0)
    n_teorico_entero = int(math.ceil(n_teorico_exacto))

    historial: list[dict[str, Any]] = []
    x_ant: float | None = None
    convergente = False

    for k in range(1, max_iter + 1):
        fa = float(fn(a_act))
        fb = float(fn(b_act))
        xk = 0.5 * (a_act + b_act)
        fxk = float(fn(xk))
        err = abs(xk - x_ant) if x_ant is not None else None

        if fa * fxk < 0:
            nuevo_a, nuevo_b = a_act, xk
            decision = f"f({a_act:g})·f({xk:g}) < 0 ⇒ Raíz en [{a_act:g} ; {xk:g}]"
        else:
            nuevo_a, nuevo_b = xk, b_act
            decision = f"f({xk:g})·f({b_act:g}) < 0 ⇒ Raíz en [{xk:g} ; {b_act:g}]"

        cumple = (err is not None and err < tol) or abs(fxk) < 1e-14

        historial.append({
            "iteracion": k,
            "a": a_act,
            "b": b_act,
            "x_k": xk,
            "f_a": fa,
            "f_b": fb,
            "f_xk": fxk,
            "error": err,
            "cumple_tol": cumple,
            "nuevo_a": nuevo_a,
            "nuevo_b": nuevo_b,
            "decision": decision,
        })

        if cumple:
            convergente = True
            duracion = time.perf_counter() - t_ini
            return {
                "raiz": xk,
                "iteraciones": k,
                "error_final": err if err is not None else 0.0,
                "f_raiz": fxk,
                "convergente": True,
                "n_teorico_exacto": n_teorico_exacto,
                "n_teorico_entero": n_teorico_entero,
                "duracion": duracion,
                "historial": historial,
            }

        a_act = nuevo_a
        b_act = nuevo_b
        x_ant = xk

    duracion = time.perf_counter() - t_ini
    return {
        "raiz": x_ant if x_ant is not None else a_act,
        "iteraciones": max_iter,
        "error_final": err if err is not None else amplitud,
        "f_raiz": float(fn(x_ant if x_ant is not None else a_act)),
        "convergente": False,
        "n_teorico_exacto": n_teorico_exacto,
        "n_teorico_entero": n_teorico_entero,
        "duracion": duracion,
        "historial": historial,
    }


# ==============================================================================
# 3. MÉTODO DE INTERPOLACIÓN LINEAL (REGULA FALSI)
# ==============================================================================

def metodo_interpolacion_lineal_catedra(
    fn: Callable[[float], float],
    x0_fijo: float,
    x1_ini: float,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> dict[str, Any]:
    """
    Ejecuta el Método de Interpolación Lineal con Extremo Fijo x0 tal como se evalúa
    en la cátedra de Métodos Computacionales (PDF 2 oficial de Interpolación Lineal):
        x_{i+1} = (x_0 * f(x_i) - x_i * f(x_0)) / (f(x_i) - f(x_0))
    """
    t_ini = time.perf_counter()
    fx0 = float(fn(x0_fijo))
    xi = float(x1_ini)
    fxi = float(fn(xi))

    if fx0 * fxi >= 0:
        raise ValueError(
            f"f(x0) y f(x1) deben tener signos opuestos por Bolzano. "
            f"f({x0_fijo:g}) = {fx0:g}, f({x1_ini:g}) = {fxi:g}"
        )

    historial: list[dict[str, Any]] = []

    for idx in range(2, max_iter + 2):
        fxi = float(fn(xi))
        denom = fxi - fx0

        if abs(denom) < 1e-15:
            duracion = time.perf_counter() - t_ini
            return {
                "raiz": xi,
                "iteraciones": idx - 2,
                "error_final": 0.0,
                "convergente": False,
                "mensaje": "División por cero: f(x_i) - f(x_0) = 0",
                "duracion": duracion,
                "historial": historial,
            }

        num = x0_fijo * fxi - xi * fx0
        x_sig = num / denom
        fx_sig = float(fn(x_sig))
        err = abs(x_sig - xi)
        cumple = (idx > 2 and err < tol) or abs(fx_sig) < 1e-14

        historial.append({
            "subindice": f"x_{idx}",
            "x0_fijo": x0_fijo,
            "f_x0": fx0,
            "x_i": xi,
            "f_xi": fxi,
            "numerador": num,
            "denominador": denom,
            "x_sig": x_sig,
            "f_xsig": fx_sig,
            "error": err if idx > 2 else None,
            "cumple_tol": cumple,
        })

        if cumple:
            duracion = time.perf_counter() - t_ini
            return {
                "raiz": x_sig,
                "iteraciones": idx - 1,
                "error_final": err,
                "f_raiz": fx_sig,
                "convergente": True,
                "mensaje": "Convergencia exitosa alcanzada.",
                "duracion": duracion,
                "historial": historial,
            }

        xi = x_sig

    duracion = time.perf_counter() - t_ini
    return {
        "raiz": xi,
        "iteraciones": max_iter,
        "error_final": err,
        "f_raiz": float(fn(xi)),
        "convergente": False,
        "mensaje": "Se alcanzó el número máximo de iteraciones.",
        "duracion": duracion,
        "historial": historial,
    }


def metodo_interpolacion_lineal_bolzano(
    fn: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> dict[str, Any]:
    """
    Ejecuta el Método Clásico de Regula Falsi actualizando dinámicamente los extremos [a, b]
    mediante el signo de f(x) * f(a) para garantizar el encierro continuo de la raíz.
    """
    t_ini = time.perf_counter()
    fa = float(fn(a))
    fb = float(fn(b))

    if fa * fb >= 0:
        raise ValueError(f"No se cumple Bolzano en [{a:g}, {b:g}]: f(a)*f(b) >= 0.")

    historial: list[dict[str, Any]] = []
    x_ant = a
    a_act, b_act = a, b

    for k in range(1, max_iter + 1):
        fa = float(fn(a_act))
        fb = float(fn(b_act))
        denom = fb - fa
        if abs(denom) < 1e-15:
            break

        x_sig = (a_act * fb - b_act * fa) / denom
        fx_sig = float(fn(x_sig))
        err = abs(x_sig - x_ant) if k > 1 else abs(b_act - a_act)
        cumple = err < tol or abs(fx_sig) < 1e-14

        historial.append({
            "iteracion": k,
            "a": a_act,
            "b": b_act,
            "x_k": x_sig,
            "f_xk": fx_sig,
            "error": err,
            "cumple_tol": cumple,
        })

        if cumple:
            duracion = time.perf_counter() - t_ini
            return {
                "raiz": x_sig,
                "iteraciones": k,
                "error_final": err,
                "f_raiz": fx_sig,
                "convergente": True,
                "duracion": duracion,
                "historial": historial,
            }

        if fa * fx_sig < 0:
            b_act = x_sig
        else:
            a_act = x_sig
        x_ant = x_sig

    duracion = time.perf_counter() - t_ini
    return {
        "raiz": x_ant,
        "iteraciones": max_iter,
        "error_final": err,
        "f_raiz": float(fn(x_ant)),
        "convergente": False,
        "duracion": duracion,
        "historial": historial,
    }


# ==============================================================================
# 4. CONDICIÓN DE FOURIER Y MÉTODO DE NEWTON-RAPHSON (TANGENTES)
# ==============================================================================

def verificar_condiciones_fourier(
    fn: Callable[[float], float],
    d1: Callable[[float], float],
    d2: Callable[[float], float],
    a: float,
    b: float,
) -> dict[str, Any]:
    """
    Verifica las tres condiciones de convergencia de Newton-Raphson en [a, b]:
    1. Bolzano: f(a) * f(b) < 0.
    2. Monotonía: f'(a) y f'(b) tienen el mismo signo y no son nulas en los extremos.
    3. Condición de Fourier: f(x0) * f''(x0) > 0.
       Elige el extremo que cumple esta condición como punto de arranque x0 seguro.
    """
    fa = float(fn(a))
    fb = float(fn(b))
    d1a, d1b = float(d1(a)), float(d1(b))
    d2a, d2b = float(d2(a)), float(d2(b))

    cumple_bolzano = (fa * fb < 0)
    monotona_bordes = (d1a * d1b > 0) and abs(d1a) > 1e-12 and abs(d1b) > 1e-12

    fourier_a = fa * d2a
    fourier_b = fb * d2b

    if fourier_a > 0:
        x0_sugerido = a
        fourier_cumple = True
        detalle_fourier = f"f({a:g})·f''({a:g}) = ({fa:+.3e})·({d2a:+.3e}) = {fourier_a:+.3e} > 0 ⇒ x₀ = {a:g}"
    elif fourier_b > 0:
        x0_sugerido = b
        fourier_cumple = True
        detalle_fourier = f"f({b:g})·f''({b:g}) = ({fb:+.3e})·({d2b:+.3e}) = {fourier_b:+.3e} > 0 ⇒ x₀ = {b:g}"
    else:
        # En caso de no cumplir estrictamente en los bordes, se elige el de menor valor absoluto de f
        x0_sugerido = a if abs(fa) < abs(fb) else b
        fourier_cumple = False
        detalle_fourier = f"Ningún borde cumple f·f''>0 estrictamente. Se elige el más próximo al eje x: x₀ = {x0_sugerido:g}"

    return {
        "cumple_bolzano": cumple_bolzano,
        "monotona_bordes": monotona_bordes,
        "fourier_cumple": fourier_cumple,
        "x0_sugerido": x0_sugerido,
        "fourier_a": fourier_a,
        "fourier_b": fourier_b,
        "detalle_fourier": detalle_fourier,
    }


def metodo_newton_raphson(
    fn: Callable[[float], float],
    d1: Callable[[float], float],
    x0: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> dict[str, Any]:
    """
    Aplica el Método de Newton-Raphson:
        x_{k+1} = x_k - f(x_k) / f'(x_k)
    Atrapa y maneja divisiones por derivada nula sin terminar en fallo del sistema.
    """
    t_ini = time.perf_counter()
    x_val = float(x0)
    historial: list[dict[str, Any]] = []

    for k in range(1, max_iter + 1):
        fx = float(fn(x_val))
        dfx = float(d1(x_val))

        if abs(dfx) < 1e-14:
            duracion = time.perf_counter() - t_ini
            historial.append({
                "iteracion": k,
                "x_k": x_val,
                "f_xk": fx,
                "df_xk": dfx,
                "x_sig": None,
                "error": None,
                "cumple_tol": False,
            })
            return {
                "raiz": x_val,
                "iteraciones": k,
                "error_final": abs(fx),
                "f_raiz": fx,
                "convergente": False,
                "mensaje": f"Derivada casi nula (|f'(x)| < 1e-14) en iteración {k}. El método se detuvo para evitar división por cero.",
                "duracion": duracion,
                "historial": historial,
            }

        x_sig = x_val - (fx / dfx)
        err = abs(x_sig - x_val)
        cumple = (err < tol) or (abs(fx) < tol)

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
            duracion = time.perf_counter() - t_ini
            return {
                "raiz": x_sig,
                "iteraciones": k,
                "error_final": err,
                "f_raiz": float(fn(x_sig)),
                "convergente": True,
                "mensaje": "Convergencia exitosa alcanzada.",
                "duracion": duracion,
                "historial": historial,
            }

        x_val = x_sig

    duracion = time.perf_counter() - t_ini
    return {
        "raiz": x_val,
        "iteraciones": max_iter,
        "error_final": err,
        "f_raiz": float(fn(x_val)),
        "convergente": False,
        "mensaje": "Se alcanzó el número máximo de iteraciones.",
        "duracion": duracion,
        "historial": historial,
    }


# ==============================================================================
# 5. PUNTO FIJO Y ACELERACIÓN DE AITKEN
# ==============================================================================

def metodo_punto_fijo_aitken(
    phi_fn: Callable[[float], float],
    x0: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> dict[str, Any]:
    """
    Ejecuta el Método de Iteración de Punto Fijo (x_{n+1} = phi(x_n)) y simultáneamente
    la Aceleración Delta^2 de Aitken según el PDF oficial de cátedra:
        S = x_2 - (x_2 - x_1)^2 / (x_2 - 2*x_1 + x_0)
    Compara ambos métodos en velocidad, pasos y precisión alcanzada.
    """
    # 1. Iteración Simple
    t0 = time.perf_counter()
    x_act = float(x0)
    hist_pf: list[dict[str, Any]] = []
    convergente_pf = False

    for k in range(1, max_iter + 1):
        x_sig = float(phi_fn(x_act))
        err = abs(x_sig - x_act)
        cumple = err < tol
        hist_pf.append({
            "iteracion": k,
            "x_ant": x_act,
            "x_sig": x_sig,
            "error": err,
            "cumple_tol": cumple,
        })
        x_act = x_sig
        if cumple:
            convergente_pf = True
            break
    t_pf = time.perf_counter() - t0

    # 2. Aceleración de Aitken
    t1 = time.perf_counter()
    x_base = float(x0)
    hist_aitken: list[dict[str, Any]] = []
    convergente_aitken = False

    for ciclo in range(1, max_iter + 1):
        x1 = float(phi_fn(x_base))
        x2 = float(phi_fn(x1))
        denom = x2 - 2.0 * x1 + x_base

        if abs(denom) < 1e-15:
            s_aitken = x2
        else:
            s_aitken = x2 - ((x2 - x1) ** 2) / denom

        x_post = float(phi_fn(s_aitken))
        err_aitken = abs(x_post - s_aitken)
        cumple_a = err_aitken < tol

        hist_aitken.append({
            "ciclo": ciclo,
            "x0": x_base,
            "x1": x1,
            "x2": x2,
            "numerador": (x2 - x1) ** 2,
            "denominador": denom,
            "s_acelerado": s_aitken,
            "x_post": x_post,
            "error": err_aitken,
            "cumple_tol": cumple_a,
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
            "error_final": hist_pf[-1]["error"] if hist_pf else 0.0,
            "convergente": convergente_pf,
            "tiempo": t_pf,
            "historial": hist_pf,
        },
        "aitken": {
            "raiz": hist_aitken[-1]["s_acelerado"] if hist_aitken else x0,
            "ciclos": len(hist_aitken),
            "error_final": hist_aitken[-1]["error"] if hist_aitken else 0.0,
            "convergente": convergente_aitken,
            "tiempo": t_aitken,
            "historial": hist_aitken,
        },
    }

