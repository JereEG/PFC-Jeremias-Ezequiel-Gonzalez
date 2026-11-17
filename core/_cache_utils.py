"""
core/_cache_utils.py
====================
Utilidades de compilación cacheada (SymPy -> NumPy), vectorización segura
y sanitización de datos numéricos para Plotly y Streamlit.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable, Iterable

import numpy as np
import streamlit as st

_MAX_LEN_EXPR = 220
_MAX_ANIDAMIENTO_EXPR = 12
_MAX_PASOS_AST = 120
_MAX_EXPONENTE_LITERAL = 50.0
_MAX_GRADO_POLINOMIO = 30
_FUNCIONES_SEGURAS = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "exp", "log", "ln", "sqrt", "abs", "Abs",
})
_CONSTANTES_SEGURAS = frozenset({"e", "pi", "E"})
_RE_IDENTIFICADORES = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_RE_ATRIBUTO = re.compile(r"\.[A-Za-z_]")


def _normalizar_expr_str(expr_str: str) -> str:
    return " ".join((expr_str or "").strip().split())


def _validar_nodo_ast(nodo: ast.AST, variables: frozenset[str], contador: list[int], en_potencia: bool = False) -> None:
    contador[0] += 1
    if contador[0] > _MAX_PASOS_AST:
        raise ValueError("La expresión es demasiado compleja.")

    if isinstance(nodo, ast.Expression):
        _validar_nodo_ast(nodo.body, variables, contador)
    elif isinstance(nodo, ast.Constant):
        if not isinstance(nodo.value, (int, float)) or isinstance(nodo.value, bool):
            raise ValueError("Solo se permiten constantes numéricas reales.")
        if abs(float(nodo.value)) > 1e12:
            raise ValueError("Constante numérica fuera de rango permitido.")
    elif isinstance(nodo, ast.Name):
        if nodo.id not in variables and nodo.id not in _CONSTANTES_SEGURAS:
            raise ValueError(f"Símbolo no permitido: '{nodo.id}'.")
    elif isinstance(nodo, ast.UnaryOp):
        if not isinstance(nodo.op, (ast.UAdd, ast.USub)):
            raise ValueError("Operador unario no permitido.")
        _validar_nodo_ast(nodo.operand, variables, contador)
    elif isinstance(nodo, ast.BinOp):
        if not isinstance(nodo.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise ValueError("Operador binario no permitido.")
        es_pow = isinstance(nodo.op, ast.Pow)
        if es_pow and en_potencia:
            raise ValueError("No se permiten torres de potencias anidadas.")
        _validar_nodo_ast(nodo.left, variables, contador, en_potencia=es_pow)
        _validar_nodo_ast(nodo.right, variables, contador, en_potencia=es_pow)
        if es_pow and isinstance(nodo.right, ast.Constant) and abs(float(nodo.right.value)) > _MAX_EXPONENTE_LITERAL:
            raise ValueError(f"Exponente literal fuera de rango (|k| <= {int(_MAX_EXPONENTE_LITERAL)}).")
        if es_pow and isinstance(nodo.right, ast.UnaryOp) and isinstance(nodo.right.operand, ast.Constant):
            if abs(float(nodo.right.operand.value)) > _MAX_EXPONENTE_LITERAL:
                raise ValueError(f"Exponente literal fuera de rango (|k| <= {int(_MAX_EXPONENTE_LITERAL)}).")
    elif isinstance(nodo, ast.Call):
        if not isinstance(nodo.func, ast.Name) or nodo.func.id not in _FUNCIONES_SEGURAS:
            raise ValueError("Función matemática no permitida.")
        if nodo.keywords or len(nodo.args) != 1:
            raise ValueError("Cada función matemática admite exactamente 1 argumento posicional.")
        _validar_nodo_ast(nodo.args[0], variables, contador)
    else:
        raise ValueError("Construcción sintáctica no permitida en expresiones matemáticas.")


def _validar_expresion_segura(expr_str: str, variables: frozenset[str] = frozenset({"x"})) -> str:
    """Valida estrictamente expresiones matemáticas contra inyección de código y DoS."""
    texto = _normalizar_expr_str(expr_str)
    if not texto:
        raise ValueError("La expresión matemática no puede estar vacía.")
    if len(texto) > _MAX_LEN_EXPR:
        raise ValueError(f"La expresión supera el máximo permitido de {_MAX_LEN_EXPR} caracteres.")
    if "__" in texto or _RE_ATRIBUTO.search(texto) or any(ch in texto for ch in "[]{};:=\"'\\`@~"):
        raise ValueError("La expresión contiene caracteres o accesos no permitidos.")

    profundidad = 0
    for ch in texto:
        if ch == "(":
            profundidad += 1
            if profundidad > _MAX_ANIDAMIENTO_EXPR:
                raise ValueError("Demasiados paréntesis anidados en la expresión.")
        elif ch == ")":
            profundidad -= 1
            if profundidad < 0:
                raise ValueError("Paréntesis desbalanceados en la expresión.")
    if profundidad != 0:
        raise ValueError("Paréntesis desbalanceados en la expresión.")

    permitidos = variables | _CONSTANTES_SEGURAS | _FUNCIONES_SEGURAS
    for ident in _RE_IDENTIFICADORES.findall(texto):
        if ident not in permitidos:
            raise ValueError(f"Identificador no permitido en la expresión: '{ident}'.")

    texto_ast = texto.replace("^", "**")
    texto_ast = re.sub(r"(\d)([A-Za-z(])", r"\1*\2", texto_ast)
    texto_ast = re.sub(r"\)\s*([A-Za-z0-9(])", r")*\1", texto_ast)
    try:
        arbol = ast.parse(texto_ast, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Sintaxis matemática inválida: {exc.msg}") from exc
    _validar_nodo_ast(arbol, variables, [0])
    return texto


def limpiar_valores_plotly(
    valores: Iterable[Any],
    limite_abs: float | None = None,
) -> list[float | None]:
    """Convierte valores no finitos o complejos en None para evitar quiebres en Plotly."""
    salida: list[float | None] = []
    for val in valores:
        if val is None:
            salida.append(None)
            continue
        try:
            if isinstance(val, complex) or np.iscomplexobj(val):
                if abs(float(np.imag(val))) > 1e-8:
                    salida.append(None)
                    continue
                num = float(np.real(val))
            else:
                num = float(val)
        except Exception:
            salida.append(None)
            continue
        if not math.isfinite(num):
            salida.append(None)
        elif limite_abs is not None and abs(num) >= limite_abs:
            salida.append(None)
        else:
            salida.append(num)
    return salida


def evaluar_fn_vectorizado(
    fn_np: Callable[[Any], Any],
    x_grid: np.ndarray | list[float],
    limite_abs: float | None = 1e6,
) -> list[float | None]:
    """Evalúa vectorizadamente en NumPy ignorando warnings de dominio y limpiando NaN/Inf."""
    x_arr = np.asarray(x_grid, dtype=float)
    if x_arr.size == 0:
        return []
    try:
        with np.errstate(all="ignore"):
            raw = fn_np(x_arr)
            arr = np.asarray(raw)
            if arr.ndim == 0:
                arr = np.broadcast_to(arr, x_arr.shape)
            if np.iscomplexobj(arr):
                real_arr = np.real(arr).astype(float, copy=False)
                valid_mask = np.isfinite(real_arr) & (np.abs(np.imag(arr)) <= 1e-8)
            else:
                real_arr = arr.astype(float, copy=False)
                valid_mask = np.isfinite(real_arr)
            if limite_abs is not None:
                valid_mask = valid_mask & (np.abs(real_arr) < limite_abs)
            return [float(v) if ok else None for v, ok in zip(real_arr.tolist(), valid_mask.tolist())]
    except Exception:
        salida: list[float | None] = []
        with np.errstate(all="ignore"):
            for xv in x_arr.tolist():
                try:
                    raw_val = fn_np(float(xv))
                    if isinstance(raw_val, complex) or np.iscomplexobj(raw_val):
                        if abs(float(np.imag(raw_val))) > 1e-8:
                            salida.append(None)
                            continue
                        num = float(np.real(raw_val))
                    else:
                        num = float(raw_val)
                    if math.isfinite(num) and (limite_abs is None or abs(num) < limite_abs):
                        salida.append(num)
                    else:
                        salida.append(None)
                except Exception:
                    salida.append(None)
        return salida


def muestrear_indices_acotados(n_total: int, max_puntos: int = 300) -> list[int]:
    """Devuelve índices equiespaciados manteniendo extremos cuando n_total > max_puntos."""
    if n_total <= max_puntos:
        return list(range(n_total))
    return sorted(set(np.linspace(0, n_total - 1, max_puntos, dtype=int).tolist()))


def _sympy_entorno_1d(sp: Any, x_sym: Any) -> dict[str, Any]:
    return {"x": x_sym, "e": sp.E, "pi": sp.pi, "ln": sp.log}


@st.cache_data(max_entries=128, show_spinner=False)
def analizar_raices_sympy_cached(f_str: str) -> dict[str, Any]:
    """Parsea f(x), calcula f'(x), f''(x), LaTeX y Descartes en caché para expresiones personalizadas."""
    import sympy as sp

    texto = _validar_expresion_segura(f_str)
    x_sym = sp.Symbol("x")
    try:
        f_simb = sp.sympify(texto, locals=_sympy_entorno_1d(sp, x_sym))
        d1_simb = sp.diff(f_simb, x_sym)
        d2_simb = sp.diff(d1_simb, x_sym)
    except Exception as e:
        raise ValueError(f"Expresión matemática inválida '{f_str}': {e}") from e

    if not f_simb.is_polynomial(x_sym) or f_simb == 0:
        descartes: dict[str, Any] = {
            "es_polinomio": False,
            "mensaje": "La función no es polinómica: la regla de conteo de signos (Descartes) aplica únicamente a polinomios.",
        }
    else:
        poly_pos = sp.Poly(f_simb, x_sym)
        grado = int(poly_pos.degree())
        if grado > _MAX_GRADO_POLINOMIO:
            raise ValueError(f"El grado del polinomio ({grado}) supera el máximo permitido ({_MAX_GRADO_POLINOMIO}).")
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

        descartes = {
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

    return {
        "f_latex": sp.latex(f_simb),
        "d1_latex": sp.latex(d1_simb),
        "d2_latex": sp.latex(d2_simb),
        "descartes": descartes,
    }


@st.cache_resource(max_entries=128, show_spinner=False)
def compilar_lambdify_raices(f_str: str) -> tuple[Any, Any, Any]:
    """Compila con sp.lambdify(..., 'numpy') las derivadas f, f' y f'' y retiene en memoria."""
    import sympy as sp

    texto = _validar_expresion_segura(f_str)
    x_sym = sp.Symbol("x")
    try:
        f_simb = sp.sympify(texto, locals=_sympy_entorno_1d(sp, x_sym))
        d1_simb = sp.diff(f_simb, x_sym)
        d2_simb = sp.diff(d1_simb, x_sym)
        fn_np = sp.lambdify(x_sym, f_simb, "numpy")
        d1_np = sp.lambdify(x_sym, d1_simb, "numpy")
        d2_np = sp.lambdify(x_sym, d2_simb, "numpy")
    except Exception as e:
        raise ValueError(f"Expresión matemática inválida '{f_str}': {e}") from e
    return fn_np, d1_np, d2_np


@st.cache_data(max_entries=128, show_spinner=False)
def analizar_g_pf_sympy_cached(g_str: str) -> dict[str, str]:
    import sympy as sp

    texto = _validar_expresion_segura(g_str)
    x_sym = sp.Symbol("x")
    try:
        g_simb = sp.sympify(texto, locals=_sympy_entorno_1d(sp, x_sym))
        dg_simb = sp.diff(g_simb, x_sym)
    except Exception as e:
        raise ValueError(f"Expresión g(x) inválida '{g_str}': {e}") from e
    return {"g_latex": sp.latex(g_simb), "dg_latex": sp.latex(dg_simb)}


@st.cache_resource(max_entries=128, show_spinner=False)
def compilar_lambdify_pf(g_str: str) -> tuple[Any, Any]:
    import sympy as sp

    texto = _validar_expresion_segura(g_str)
    x_sym = sp.Symbol("x")
    try:
        g_simb = sp.sympify(texto, locals=_sympy_entorno_1d(sp, x_sym))
        dg_simb = sp.diff(g_simb, x_sym)
        return sp.lambdify(x_sym, g_simb, "numpy"), sp.lambdify(x_sym, dg_simb, "numpy")
    except Exception as e:
        raise ValueError(f"Expresión g(x) inválida '{g_str}': {e}") from e


@st.cache_resource(max_entries=128, show_spinner=False)
def compilar_lambdify_1d(f_str: str) -> Any:
    import sympy as sp

    texto = _validar_expresion_segura(f_str)
    x_sym = sp.Symbol("x")
    try:
        expr = sp.sympify(texto, locals=_sympy_entorno_1d(sp, x_sym))
        return sp.lambdify(x_sym, expr, "numpy")
    except Exception as e:
        raise ValueError(f"Expresión matemática inválida '{f_str}': {e}") from e


@st.cache_data(max_entries=128, show_spinner=False)
def evaluar_grilla_1d_cached(
    f_str: str,
    x_min: float,
    x_max: float,
    n_puntos: int = 220,
    limite_abs: float | None = 1e6,
) -> tuple[list[float], list[float | None]]:
    """Evalúa y cachea una grilla vectorizada de 220 puntos para cualquier expresión 1D."""
    from core.catalogo_ejercicios import EXPRESIONES_PRECOMPUTADAS_RAICES, EXPRESIONES_PRECOMPUTADAS_PUNTO_FIJO

    clave = _normalizar_expr_str(f_str)
    if clave in EXPRESIONES_PRECOMPUTADAS_RAICES:
        fn_np = EXPRESIONES_PRECOMPUTADAS_RAICES[clave]["fn_np"]
    elif clave in EXPRESIONES_PRECOMPUTADAS_PUNTO_FIJO:
        fn_np = EXPRESIONES_PRECOMPUTADAS_PUNTO_FIJO[clave]["gn_np"]
    else:
        fn_np = compilar_lambdify_1d(clave)

    x_grid = np.linspace(float(x_min), float(x_max), int(n_puntos))
    y_clean = evaluar_fn_vectorizado(fn_np, x_grid, limite_abs=limite_abs)
    return x_grid.tolist(), y_clean


def _parsear_expr_edo(sp: Any, texto: str, incluir_y: bool = True) -> tuple[Any, Any]:
    """Parsea una expresión EDO previamente validada sin exponer builtins."""
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    x_sym = sp.Symbol("x")
    variables: Any = x_sym
    entorno: dict[str, Any] = {
        "x": x_sym, "e": sp.E, "pi": sp.pi,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
    }
    if incluir_y:
        y_sym = sp.Symbol("y")
        entorno["y"] = y_sym
        variables = (x_sym, y_sym)
    globales_sympy = {
        "__builtins__": {},
        "Symbol": sp.Symbol,
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
    }
    transf = standard_transformations + (implicit_multiplication_application, convert_xor)
    expr = parse_expr(texto, local_dict=entorno, global_dict=globales_sympy, transformations=transf)
    return variables, expr


@st.cache_data(max_entries=128, show_spinner=False)
def analizar_edo_f_cached(f_str: str) -> str:
    import sympy as sp

    texto = _validar_expresion_segura(f_str, variables=frozenset({"x", "y"}))
    try:
        _, expr = _parsear_expr_edo(sp, texto, incluir_y=True)
        return sp.latex(expr)
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar f(x, y) = '{f_str}': {exc}") from exc


@st.cache_resource(max_entries=128, show_spinner=False)
def compilar_lambdify_edo_f(f_str: str) -> Any:
    import sympy as sp

    texto = _validar_expresion_segura(f_str, variables=frozenset({"x", "y"}))
    try:
        variables, expr = _parsear_expr_edo(sp, texto, incluir_y=True)
        return sp.lambdify(variables, expr, "numpy")
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar f(x, y) = '{f_str}': {exc}") from exc


@st.cache_data(max_entries=128, show_spinner=False)
def analizar_edo_exacta_cached(y_str: str) -> str:
    import sympy as sp

    texto = _validar_expresion_segura(y_str)
    try:
        _, expr = _parsear_expr_edo(sp, texto, incluir_y=False)
        return sp.latex(expr)
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar y_exacta(x) = '{y_str}': {exc}") from exc


@st.cache_resource(max_entries=128, show_spinner=False)
def compilar_lambdify_edo_exacta(y_str: str) -> Any:
    import sympy as sp

    texto = _validar_expresion_segura(y_str)
    try:
        x_sym, expr = _parsear_expr_edo(sp, texto, incluir_y=False)
        return sp.lambdify(x_sym, expr, "numpy")
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar y_exacta(x) = '{y_str}': {exc}") from exc


@st.cache_data(max_entries=128, show_spinner=False)
def integral_sympy_cached(f_str: str, a_tot: float, b_tot: float) -> dict[str, Any]:
    import sympy as sp

    texto = _validar_expresion_segura(f_str)
    x_sym = sp.Symbol("x")
    f_simb = sp.sympify(texto, locals=_sympy_entorno_1d(sp, x_sym))
    integral_exacta = float(sp.N(sp.integrate(f_simb, (x_sym, float(a_tot), float(b_tot)))))
    return {
        "integral_exacta": integral_exacta,
        "f_latex": sp.latex(f_simb),
    }


@st.cache_data(max_entries=128, show_spinner=False)
def construir_lagrange_latex_cached(
    x_tup: tuple[float, ...],
    y_tup: tuple[float, ...],
) -> dict[str, Any]:
    import sympy as sp

    x_sym = sp.Symbol("x")
    n = len(x_tup)
    P_sym = sp.Integer(0)
    bases_latex: list[str] = []
    for i in range(n):
        L_i_sym = sp.Integer(1)
        for j in range(n):
            if j != i:
                L_i_sym *= (x_sym - x_tup[j]) / (x_tup[i] - x_tup[j])
        bases_latex.append(sp.latex(sp.simplify(L_i_sym)))
        P_sym += y_tup[i] * L_i_sym
    P_expandido = sp.expand(P_sym)
    return {
        "latex_expandido": sp.latex(P_expandido),
        "bases_latex": bases_latex,
    }

