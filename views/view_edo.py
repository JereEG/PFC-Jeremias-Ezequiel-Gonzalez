"""
Vista de Ecuaciones Diferenciales Ordinarias (PVI) - Arandú.
Euler, Heun (Corrector Iterativo), Runge-Kutta (RK2 / RK4) y Milne (Orden 4).
Separación Static-First entre teoría de cátedra y simulación aislada con @st.fragment.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core._cache_utils import evaluar_fn_vectorizado
from core.edo_pvi import (
    compilar_exacta_edo,
    compilar_funcion_edo,
    metodo_euler,
    metodo_euler_modificado,
    metodo_milne,
    metodo_runge_kutta_2,
    metodo_runge_kutta_4,
)
from views.theme import PALETA_ARANDU
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    muestrear_para_grafico,
    registrar_contexto_ia,
    render_bloque_1_teoria_y_ejemplo,
    render_controles_edicion_practica,
    render_dataframe_acotado,
    render_excepcion_validacion,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
)


def _inputs_pvi(clave_state: str, solo_lectura: bool) -> tuple[str, float, float, float, float, str]:
    c1, c2 = st.columns(2)
    f_str = c1.text_input(
        "EDO $y' = f(x, y)$:",
        value=str(st.session_state[f"{clave_state}_f_str"]),
        disabled=solo_lectura,
        help="Expresión en variables x, y con operadores Python (*, **). Ej: 6*x**2 / y",
        key=f"ui_{clave_state}_f",
    )
    y_ex_str = c2.text_input(
        "Solución exacta $y(x)$ (opcional):",
        value=str(st.session_state[f"{clave_state}_y_exacta_str"]),
        disabled=solo_lectura,
        help="Expresión analítica en variable x para medir error real. Ej: 2*x*sqrt(x)",
        key=f"ui_{clave_state}_yex",
    )

    c3, c4, c5, c6 = st.columns(4)
    x0 = c3.number_input(
        "$x_0$:",
        value=float(st.session_state[f"{clave_state}_x0"]),
        format="%.4f",
        disabled=solo_lectura,
        help="Abscisa inicial (real, distinta de x_final).",
        key=f"ui_{clave_state}_x0",
    )
    y0 = c4.number_input(
        "$y_0 = y(x_0)$:",
        value=float(st.session_state[f"{clave_state}_y0"]),
        format="%.4f",
        disabled=solo_lectura,
        help="Condición inicial y(x_0) dentro del dominio de f(x, y).",
        key=f"ui_{clave_state}_y0",
    )
    xf = c5.number_input(
        r"$x_{\text{final}}$:",
        value=float(st.session_state[f"{clave_state}_xf"]),
        format="%.4f",
        disabled=solo_lectura,
        help="Extremo de integración final; en Milne debe cubrir al menos 4 pasos h.",
        key=f"ui_{clave_state}_xf",
    )
    h = c6.number_input(
        r"Paso $\Delta h$:",
        value=float(st.session_state[f"{clave_state}_h"]),
        min_value=0.005,
        max_value=2.0,
        step=0.05,
        format="%.4f",
        disabled=solo_lectura,
        help="Incremento de malla Δh en rango [0.005, 2.0].",
        key=f"ui_{clave_state}_h",
    )

    if not solo_lectura:
        st.session_state[f"{clave_state}_f_str"] = f_str
        st.session_state[f"{clave_state}_y_exacta_str"] = y_ex_str
        st.session_state[f"{clave_state}_x0"] = x0
        st.session_state[f"{clave_state}_y0"] = y0
        st.session_state[f"{clave_state}_xf"] = xf
        st.session_state[f"{clave_state}_h"] = h

    return f_str, x0, y0, xf, h, y_ex_str


def _compilar_pvi(f_str: str, y_ex_str: str) -> tuple[dict, object | None, object | None]:
    comp_f = compilar_funcion_edo(f_str)
    y_ex_fn = None
    y_ex_fn_np = None
    if y_ex_str.strip():
        try:
            comp_ex = compilar_exacta_edo(y_ex_str)
            y_ex_fn = comp_ex["fn"]
            y_ex_fn_np = comp_ex.get("fn_np")
        except Exception as exc:
            st.warning(f"Solución exacta ignorada: {exc}")
    return comp_f, y_ex_fn, y_ex_fn_np


def _plot_edo_paso_a_paso(
    titulo: str,
    x0: float,
    xf_real: float,
    series: list[tuple[str, list[float], list[float], str]],
    y_ex_fn,
    y_ex_fn_np,
    k_vis: int,
    chart_key: str,
) -> None:
    fig = crear_figura_base_plotly(titulo, "x", "y(x)")
    if y_ex_fn_np is not None or y_ex_fn is not None:
        xg = np.linspace(x0, xf_real, 220)
        try:
            if y_ex_fn_np is not None:
                yg = evaluar_fn_vectorizado(y_ex_fn_np, xg, limite_abs=1e6)
            else:
                yg = evaluar_fn_vectorizado(np.vectorize(y_ex_fn), xg, limite_abs=1e6)
            fig.add_trace(
                go.Scatter(
                    x=xg.tolist(),
                    y=yg,
                    mode="lines",
                    name="Solución Exacta",
                    line=dict(color=PALETA_ARANDU["oro"], width=2.8, dash="dash"),
                )
            )
        except Exception:
            pass

    estilos_dash = ("solid", "dot", "dashdot")
    simbolos = ("circle", "diamond", "square")
    for idx, (nombre, xs, ys, color) in enumerate(series):
        pares = muestrear_para_grafico(list(zip(xs[: k_vis + 1], ys[: k_vis + 1])), max_puntos=300)
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pares],
                y=[p[1] for p in pares],
                mode="lines+markers",
                name=nombre,
                line=dict(color=color, width=2.4, dash=estilos_dash[idx % len(estilos_dash)]),
                marker=dict(size=7, symbol=simbolos[idx % len(simbolos)]),
            )
        )
    render_plotly_arandu(fig, key=chart_key)


@st.fragment
def _frag_euler_heun(defaults: dict) -> None:
    modo = render_controles_edicion_practica(clave_state="edo_eh", defaults_dict=defaults)
    solo_lectura = modo == "catedra"
    f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_eh", solo_lectura)
    iterar_corr = st.checkbox(
        "Iterar corrector de Heun hasta convergencia (bucle do-while estilo Cátedra)",
        value=True,
        key="ui_eh_corr",
    )

    try:
        comp_f, y_ex_fn, y_ex_fn_np = _compilar_pvi(f_str, y_ex_str)
        res_eu = metodo_euler(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
        res_he = metodo_euler_modificado(
            comp_f["fn"], x0, y0, h, xf=xf, iterar_corrector=iterar_corr, y_exacta_fn=y_ex_fn
        )
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="EDO f(x, y), condiciones iniciales o paso Δh",
            sugerencia_defecto="Verificá sintaxis Python de f(x,y), evitá división por cero en (x_0, y_0) y asegurate de que x_final != x_0.",
            ejemplo_defecto="f(x, y) = 6*x**2 / y, x0 = 4.0, y0 = 16.0, xf = 4.4, h = 0.1",
        )
        return

    for av in res_he["avisos"]:
        st.info(av)

    st.markdown("### Bloque 3 — Resultados Euler vs Heun en Vivo")
    m1, m2, m3 = st.columns(3)
    m1.metric("Euler y(x_f)", f"{res_eu['y_final']:.6f}")
    m2.metric("Heun y(x_f)", f"{res_he['y_final']:.6f}")
    m3.metric("Pasos (n)", res_he["n_pasos"])
    registrar_contexto_ia(
        "Resultados PVI Euler vs Heun",
        f"- **EDO:** `{f_str}` | **$x_0 = {x0:g}, y_0 = {y0:g}, h = {h:g}, x_f = {res_he['xf_real']:g}$** (`n = {res_he['n_pasos']}` pasos)\n"
        f"- **Euler:** `y(x_f) = {res_eu['y_final']:.6f}` | **Heun:** `y(x_f) = {res_he['y_final']:.6f}`",
        clave="edo_eh_res",
    )

    t1, t2 = st.tabs(["Tabla Euler", "Tabla Heun (Euler Modificado)"])
    with t1:
        render_dataframe_acotado(res_eu["tabla"], max_filas=300, key_prefix="tbl_edo_euler")
    with t2:
        render_dataframe_acotado(res_he["tabla"], max_filas=300, key_prefix="tbl_edo_heun")

    st.markdown("### Bloque 4 — Trayectoria Numérica Paso a Paso")
    k_vis = render_slider_paso_k("Ver evolución hasta el paso $k$:", res_he["n_pasos"], "vis_eh_k")
    _plot_edo_paso_a_paso(
        f"Euler vs Heun hasta paso k = {k_vis}",
        x0,
        res_he["xf_real"],
        [("Euler (O1)", res_eu["x"], res_eu["y"], PALETA_ARANDU["terracota"]), ("Heun (O2)", res_he["x"], res_he["y"], PALETA_ARANDU["salvia"])],
        y_ex_fn,
        y_ex_fn_np,
        k_vis,
        chart_key="plot_edo_eh",
    )


def _tab_euler_heun() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("euler_heun")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.4, "y_exacta_str": "2*x*sqrt(x)"},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_euler_heun(defaults)


@st.fragment
def _frag_runge_kutta(defaults: dict) -> None:
    modo = render_controles_edicion_practica(clave_state="edo_rk", defaults_dict=defaults)
    solo_lectura = modo == "catedra"
    f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_rk", solo_lectura)

    try:
        comp_f, y_ex_fn, y_ex_fn_np = _compilar_pvi(f_str, y_ex_str)
        res_rk2 = metodo_runge_kutta_2(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
        res_rk4 = metodo_runge_kutta_4(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="EDO f(x, y), condiciones iniciales o paso Δh",
            sugerencia_defecto="Verificá sintaxis Python de f(x,y), evitá división por cero en (x_0, y_0) y asegurate de que x_final != x_0.",
            ejemplo_defecto="f(x, y) = 6*x**2 / y, x0 = 4.0, y0 = 16.0, xf = 4.4, h = 0.1",
        )
        return

    for av in res_rk4["avisos"]:
        st.info(av)

    st.markdown("### Bloque 3 — Pendientes $k_i$ de RK2 y RK4 en Vivo")
    m1, m2, m3 = st.columns(3)
    m1.metric("RK2 y(x_f)", f"{res_rk2['y_final']:.6f}")
    m2.metric("RK4 y(x_f)", f"{res_rk4['y_final']:.6f}")
    m3.metric("Pasos (n)", res_rk4["n_pasos"])
    registrar_contexto_ia(
        "Resultados PVI RK2 vs RK4",
        f"- **EDO:** `{f_str}` | **$x_0 = {x0:g}, y_0 = {y0:g}, h = {h:g}, x_f = {res_rk4['xf_real']:g}$** (`n = {res_rk4['n_pasos']}` pasos)\n"
        f"- **RK2:** `y(x_f) = {res_rk2['y_final']:.6f}` | **RK4:** `y(x_f) = {res_rk4['y_final']:.6f}`",
        clave="edo_rk_res",
    )

    t1, t2 = st.tabs(["Tabla RK4 (k1, k2, k3, k4)", "Tabla RK2 (k1, k2)"])
    with t1:
        render_dataframe_acotado(res_rk4["tabla"], max_filas=300, key_prefix="tbl_edo_rk4")
    with t2:
        render_dataframe_acotado(res_rk2["tabla"], max_filas=300, key_prefix="tbl_edo_rk2")

    st.markdown("### Bloque 4 — Curvas RK2 y RK4 Paso a Paso")
    k_vis = render_slider_paso_k("Ver evolución hasta el paso $k$:", res_rk4["n_pasos"], "vis_rk_k")
    _plot_edo_paso_a_paso(
        f"RK2 vs RK4 hasta paso k = {k_vis}",
        x0,
        res_rk4["xf_real"],
        [("RK2 (O2)", res_rk2["x"], res_rk2["y"], PALETA_ARANDU["terracota"]), ("RK4 (O4)", res_rk4["x"], res_rk4["y"], PALETA_ARANDU["salvia"])],
        y_ex_fn,
        y_ex_fn_np,
        k_vis,
        chart_key="plot_edo_rk",
    )


def _tab_runge_kutta() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("runge_kutta")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.3, "y_exacta_str": "2*x*sqrt(x)"},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_runge_kutta(defaults)


@st.fragment
def _frag_milne(defaults: dict) -> None:
    modo = render_controles_edicion_practica(clave_state="edo_mil", defaults_dict=defaults)
    solo_lectura = modo == "catedra"
    f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_mil", solo_lectura)

    st.markdown(r"### Bloque 3 — Arranque RK4 ($k \le 3$) + Predictor-Corrector de Milne")
    try:
        comp_f, y_ex_fn, y_ex_fn_np = _compilar_pvi(f_str, y_ex_str)
        res = metodo_milne(comp_f["fn"], x0, y0, h, xf=xf, n_correcciones=2, y_exacta_fn=y_ex_fn)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Intervalo de Milne [x_0, x_final] o f(x, y)",
            sugerencia_defecto="Milne requiere al menos 4 pasos completos desde x_0 (|x_final - x_0| >= 4*Δh). Reducí Δh o ampliá x_final.",
            ejemplo_defecto="x0 = 4.0, xf = 4.4, h = 0.1 (4 pasos)",
        )
        return

    for av in res["avisos"]:
        st.info(av)

    m1, m2, m3 = st.columns(3)
    m1.metric("Milne y(x_f)", f"{res['y_final']:.6f}")
    m2.metric("Pasos totales", res["n_pasos"])
    err_fin = res["tabla"][-1].get("error_abs")
    m3.metric("Error absoluto final", f"{err_fin:.3e}" if err_fin is not None else "-")
    err_mil = f"{err_fin:.3e}" if err_fin is not None else "No provista"
    registrar_contexto_ia(
        "Resultados PVI Milne Predictor-Corrector",
        f"- **EDO:** `{f_str}` | **$x_0 = {x0:g}, y_0 = {y0:g}, h = {h:g}, x_f = {res['xf_real']:g}$**\n"
        f"- **Milne $y(x_f)$:** `{res['y_final']:.6f}` | **Pasos:** `{res['n_pasos']}` | **Error abs. final:** `{err_mil}`",
        clave="edo_mil_res",
    )

    render_dataframe_acotado(res["tabla"], max_filas=300, key_prefix="tbl_edo_milne")

    st.markdown("### Bloque 4 — Evolución Arranque RK4 vs Pasos Multipaso de Milne")
    k_vis = render_slider_paso_k("Ver evolución hasta el paso $k$:", res["n_pasos"], "vis_mil_k")
    _plot_edo_paso_a_paso(
        f"Milne (Arranque RK4 k≤3, P-C k≥4) hasta paso k = {k_vis}",
        x0,
        res["xf_real"],
        [("Milne P-C (O4)", res["x"], res["y"], PALETA_ARANDU["salvia"])],
        y_ex_fn,
        y_ex_fn_np,
        k_vis,
        chart_key="plot_edo_milne",
    )


def _tab_milne() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("milne")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.4, "y_exacta_str": "2*x*sqrt(x)"},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_milne(defaults)


def vista_edo() -> None:
    inject_arandu_css()
    st.title("Ecuaciones Diferenciales Ordinarias (PVI)")
    st.markdown(
        r"Resolución numérica paso a paso sobre malla $\Delta h$ dinámica y contraste directo contra solución analítica."
    )
    activo = render_selector_fichas_metodos(
        "edo",
        (
            {
                "numero": "1",
                "clave_svg": "euler_heun",
                "titulo": "Euler y Heun",
                "subtitulo": "Paso simple · Corrector Heun",
                "resumen": "Euler proyecta una pendiente simple y Heun promedia entrada y salida mejorando el orden.",
                "pista": "Tangente frente a corrector Heun",
            },
            {
                "numero": "2",
                "clave_svg": "runge_kutta",
                "titulo": "Runge-Kutta (RK2 / RK4)",
                "subtitulo": "Autoiniciable · Cuatro pendientes",
                "resumen": "Integra ecuaciones diferenciales desde el primer paso promediando múltiples pendientes ponderadas de referencia.",
                "pista": "Autoiniciable · Orden O(h⁴)",
            },
            {
                "numero": "3",
                "clave_svg": "milne",
                "titulo": "Predictor-Corrector de Milne",
                "subtitulo": "Multipaso · Memoria de 4 puntos",
                "resumen": "Arranca con cuatro pasos iniciales de RK4 y corrige iterativamente cada nuevo punto y_{n+1}.",
                "pista": "Memoria 4 pasos · Orden O(h⁴)",
            },
        ),
    )
    (_tab_euler_heun, _tab_runge_kutta, _tab_milne)[activo]()


vista_edo()

