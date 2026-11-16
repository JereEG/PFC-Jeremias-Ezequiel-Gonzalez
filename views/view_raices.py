"""
Vista de Raíces de Ecuaciones No Lineales - Arandú.
Bisección (+ Descartes/Tanteo), Interpolación Lineal (Regula Falsi),
Newton-Raphson (Fourier + tangentes paso a paso) y Punto Fijo + Aitken Δ².
Arquitectura Static-First con aislamiento por @st.fragment.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core._cache_utils import evaluar_fn_vectorizado
from core.raices import (
    analizar_descartes,
    barrido_tanteo,
    compilar_funcion_raices,
    compilar_g_punto_fijo,
    metodo_biseccion,
    metodo_interpolacion_lineal_bolzano,
    metodo_interpolacion_lineal_catedra,
    metodo_newton_raphson,
    metodo_punto_fijo_aitken,
    verificar_condiciones_fourier,
)
from views.theme import PALETA_ARANDU, hex_a_rgba
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    muestrear_para_grafico,
    registrar_contexto_ia,
    render_bloque_1_teoria_y_ejemplo,
    render_controles_edicion_practica,
    render_dataframe_acotado,
    render_deduccion_paso_a_paso_viva,
    render_excepcion_validacion,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
)


def _muestrear_curva(comp: dict, fn_key: str, np_key: str, x_min: float, x_max: float, puntos: int = 220) -> tuple[list[float], list[float | None]]:
    xs = np.linspace(x_min, x_max, puntos)
    fn_np = comp.get(np_key)
    if fn_np is not None:
        ys = evaluar_fn_vectorizado(fn_np, xs, limite_abs=1e6)
    else:
        ys = evaluar_fn_vectorizado(np.vectorize(comp[fn_key]), xs, limite_abs=1e6)
    return xs.tolist(), ys


@st.fragment
def _frag_biseccion(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="raiz_bis",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica la expresión $f(x)$, extremos $[a, b]$ o tolerancia en vivo.",
    )
    solo_lectura = modo == "catedra"

    f_str = st.text_input(
        "Ecuación $f(x) = 0$:",
        value=str(st.session_state["raiz_bis_f_str"]),
        disabled=solo_lectura,
        help="Expresión en variable x usando * y ** (ej: -2.75*x**3 + 18*x**2 - 21*x - 12).",
        key="ui_bis_f",
    )
    c1, c2, c3, c4 = st.columns(4)
    a_val = c1.number_input("Extremo $a$:", value=float(st.session_state["raiz_bis_a"]), format="%.4f", disabled=solo_lectura, help="Límite inferior del intervalo [a, b].", key="ui_bis_a")
    b_val = c2.number_input("Extremo $b$:", value=float(st.session_state["raiz_bis_b"]), format="%.4f", disabled=solo_lectura, help="Límite superior del intervalo [a, b] con signo opuesto a f(a).", key="ui_bis_b")
    tol_val = c3.number_input("Tolerancia $E$:", value=float(st.session_state["raiz_bis_tol"]), min_value=1e-10, format="%.6f", disabled=solo_lectura, help="Error máximo permitido (> 1e-10, unidades de x).", key="ui_bis_tol")
    max_it = c4.number_input("Iteraciones máx:", value=int(st.session_state["raiz_bis_max_iter"]), min_value=1, max_value=200, disabled=solo_lectura, help="Tope de ciclos permitido entre 1 y 200.", key="ui_bis_max")

    if not solo_lectura:
        st.session_state["raiz_bis_f_str"] = f_str
        st.session_state["raiz_bis_a"] = a_val
        st.session_state["raiz_bis_b"] = b_val
        st.session_state["raiz_bis_tol"] = tol_val
        st.session_state["raiz_bis_max_iter"] = int(max_it)

    try:
        comp = compilar_funcion_raices(f_str)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Ecuación f(x) = 0",
            sugerencia_defecto="Ingresá una expresión válida en variable x usando operadores explícitos * y **.",
            ejemplo_defecto="-2.75*x**3 + 18*x**2 - 21*x - 12",
        )
        return

    with st.expander("Análisis Previo: Conteo de Signos (Regla de Descartes) & Barrido por Tanteos"):
        desc = comp.get("descartes") or analizar_descartes(f_str)
        if desc["es_polinomio"]:
            st.markdown(
                f"Grado del polinomio: **{desc['grado']}** | Raíces reales positivas posibles: **{desc['posibles_raices_pos']}** | "
                f"Raíces reales negativas posibles: **{desc['posibles_raices_neg']}** "
                "(cota combinatoria contando variaciones de signo)."
            )
        else:
            st.info(desc["mensaje"])
        bt = barrido_tanteo(comp["fn"], x_ini=min(a_val, b_val) - 2.0, dx=0.5, max_pasos=16)
        st.markdown(rf"Intervalos con cambio de signo detectados en barrido $\Delta x = 0.5$: **{bt['intervalos']}**")

    st.markdown("### Bloque 3 — Ejecución en Vivo (Bisección)")
    try:
        res = metodo_biseccion(comp["fn"], a_val, b_val, tol=tol_val, max_iter=int(max_it))
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Intervalo [a, b] de Bisección",
            sugerencia_defecto="Elegí extremos a y b donde f(a)*f(b) < 0 (usá el barrido por tanteos arriba).",
            ejemplo_defecto="a = 2.0, b = 2.5 para f(x) = x**3 - 2*x - 5",
        )
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raíz aproximada", f"{res['raiz']:.6f}")
    m2.metric("Iteraciones", f"{res['iteraciones']} (teórico: {res['n_teorico_entero']})")
    m3.metric("Error final |Δx|", f"{res['error_final']:.3e}")
    m4.metric("f(raíz)", f"{res['f_raiz']:.3e}")
    registrar_contexto_ia(
        "Resultado Bisección",
        f"- **Raíz:** `{res['raiz']:.6f}` | **Iteraciones:** `{res['iteraciones']}` (teórico `{res['n_teorico_entero']}`)\n"
        f"- **Error final:** `{res['error_final']:.3e}` | **$f(\\text{{raíz}})$:** `{res['f_raiz']:.3e}`",
        clave="raiz_bis_res",
    )

    hist = res["historial"]
    pasos_bis_vivos = [
        (
            f"**Verificación de Bolzano y cota teórica:** En $[{min(a_val, b_val):g}, {max(a_val, b_val):g}]$, "
            f"$f(a)\\cdot f(b) = ({hist[0]['f_a']:.5g})\\cdot({hist[0]['f_b']:.5g}) < 0$. "
            f"La cota a priori exige $n \\ge \\frac{{\\ln({abs(b_val - a_val):g}) - \\ln({tol_val:g})}}{{\\ln 2}} = {res['n_teorico_exacto']:.4f} "
            f"\\implies \\lceil n \\rceil = {res['n_teorico_entero']}$ bisecciones."
        )
    ]
    for r_b in hist[:6]:
        err_t = f" | $|\\Delta x| = {r_b['error']:.3e}$" if r_b["error"] is not None else ""
        pasos_bis_vivos.append(
            f"**Iteración $k = {r_b['iteracion']}$:** Punto medio $x_{{{r_b['iteracion']}}} = \\frac{{{r_b['a']:.6g} + {r_b['b']:.6g}}}{{2}} = {r_b['x_k']:.6f}$, "
            f"$f(x_k) = {r_b['f_xk']:.5g} \\implies$ `{r_b['decision']}`{err_t}."
        )
    if len(hist) > 6:
        pasos_bis_vivos.append(
            f"**Cierre en $k = {res['iteraciones']}$:** Se alcanza $x^* = {res['raiz']:.6f}$ con $|\\Delta x| = {res['error_final']:.3e} < {tol_val:g}$."
        )
    render_deduccion_paso_a_paso_viva(
        f"Bisección en [{min(a_val, b_val):g}, {max(a_val, b_val):g}]",
        pasos_bis_vivos,
        clave_ia="raiz_bis_viva",
    )
    cols_bis = ["iteracion", "a", "b", "x_k", "f_a", "f_xk", "f_b", "error", "cumple_tol"]
    render_dataframe_acotado(
        [{k: fila[k] for k in cols_bis} for fila in hist],
        max_filas=300,
        key_prefix="tbl_raiz_bis",
    )

    st.markdown("### Bloque 4 — Geometría Paso a Paso (Encierro de Intervalos)")
    k_vis = render_slider_paso_k("Ver hasta la iteración $k$:", len(hist), key="vis_bis_k")
    fila_k = hist[k_vis - 1]
    pad = max(0.5, 0.35 * abs(b_val - a_val))
    xs, ys = _muestrear_curva(comp, "fn", "fn_np", min(a_val, b_val) - pad, max(a_val, b_val) + pad)

    fig = crear_figura_base_plotly(f"Bisección — Iteración {k_vis}: [{fila_k['a']:.4f}, {fila_k['b']:.4f}]", "x", "f(x)")
    fig.add_vrect(
        x0=fila_k["a"],
        x1=fila_k["b"],
        fillcolor=hex_a_rgba(PALETA_ARANDU["salvia"], 0.22),
        line_color=PALETA_ARANDU["salvia_borde"],
        line_width=1,
        annotation_text=f"Intervalo k={k_vis}",
    )
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.5)))
    sub_hist = muestrear_para_grafico(hist[:k_vis], max_puntos=150)
    fig.add_trace(
        go.Scatter(
            x=[r["x_k"] for r in sub_hist],
            y=[r["f_xk"] for r in sub_hist],
            mode="markers",
            name="Puntos medios x_k",
            marker=dict(size=8, color=PALETA_ARANDU["salvia"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[fila_k["x_k"]],
            y=[fila_k["f_xk"]],
            mode="markers",
            name=f"Actual x_{k_vis}",
            marker=dict(size=13, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_raiz_bis")


def _tab_biseccion() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("biseccion")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "-2.75*x**3 + 18*x**2 - 21*x - 12", "a": 2.0, "b": 2.5, "tol": 0.01, "max_iter": 50},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_biseccion(defaults)


@st.fragment
def _frag_interpolacion_lineal(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="raiz_rf",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica $f(x)$, los extremos iniciales o la variante de actualización del intervalo.",
    )
    solo_lectura = modo == "catedra"

    f_str = st.text_input("Ecuación $f(x) = 0$:", value=str(st.session_state["raiz_rf_f_str"]), disabled=solo_lectura, key="ui_rf_f")
    c1, c2, c3, c4 = st.columns(4)
    a_val = c1.number_input("$x_0$ / extremo $a$:", value=float(st.session_state["raiz_rf_a"]), format="%.4f", disabled=solo_lectura, key="ui_rf_a")
    b_val = c2.number_input("$x_1$ / extremo $b$:", value=float(st.session_state["raiz_rf_b"]), format="%.4f", disabled=solo_lectura, key="ui_rf_b")
    tol_val = c3.number_input("Tolerancia $E$:", value=float(st.session_state["raiz_rf_tol"]), min_value=1e-10, format="%.6f", disabled=solo_lectura, key="ui_rf_tol")
    variante = c4.selectbox(
        "Variante de reemplazo:",
        ["Cátedra (pivote fijo)", "Regula Falsi (Bolzano)"],
        index=0 if str(st.session_state["raiz_rf_variante"]).startswith("Cátedra") else 1,
        key="ui_rf_var",
    )
    st.caption(
        r"La variante estándar mantiene un cambio de signo permanente ($f(a)\cdot f(b) < 0$, criterio de Bolzano) "
        r"al actualizar uno de los dos extremos según dónde caiga la cuerda."
    )

    if not solo_lectura:
        st.session_state["raiz_rf_f_str"] = f_str
        st.session_state["raiz_rf_a"] = a_val
        st.session_state["raiz_rf_b"] = b_val
        st.session_state["raiz_rf_tol"] = tol_val
        st.session_state["raiz_rf_variante"] = variante

    try:
        comp = compilar_funcion_raices(f_str)
        st.markdown("### Bloque 3 — Cuerdas en Vivo")
        if variante.startswith("Cátedra"):
            res = metodo_interpolacion_lineal_catedra(comp["fn"], x0_fijo=a_val, x1_ini=b_val, tol=tol_val)
        else:
            res = metodo_interpolacion_lineal_bolzano(comp["fn"], a=min(a_val, b_val), b=max(a_val, b_val), tol=tol_val)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Intervalo [a, b] o ecuación f(x) (Regula Falsi)",
            sugerencia_defecto="Revisá que f(a)*f(b) < 0 y la función sea continua en [a, b].",
            ejemplo_defecto="f(x) = x**3 - 2*x - 5, a = 2.0, b = 2.5",
        )
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Raíz aproximada", f"{res['raiz']:.6f}")
    m2.metric("Iteraciones", res["iteraciones"])
    m3.metric("Error final", f"{res['error_final']:.3e}")
    registrar_contexto_ia(
        "Resultado Interpolación Lineal / Regula Falsi",
        f"- **Variante:** {variante}\n- **Raíz:** `{res['raiz']:.6f}` | **Iteraciones:** `{res['iteraciones']}` | **Error:** `{res['error_final']:.3e}`",
        clave="raiz_rf_res",
    )

    hist = res["historial"]
    pasos_rf_vivos = [
        f"**Planteo de la cuerda ({variante}):** Partimos desde $x_0 = {a_val:g}$ y $x_1 = {b_val:g}$ buscando intersecciones secantes hasta $|x_{{k+1}} - x_k| < {tol_val:g}$."
    ]
    for r_rf in hist[:6]:
        if variante.startswith("Cátedra"):
            pasos_rf_vivos.append(
                f"**Paso $k = {r_rf['iteracion']}$:** Cuerda entre $(x_0={r_rf['x0_fijo']:g}, {r_rf['f_x0']:.5g})$ y $(x_i={r_rf['x_i']:.6f}, {r_rf['f_xi']:.5g}) \\implies "
                f"x_{{\\text{{sig}}}} = \\frac{{({r_rf['x0_fijo']:g})({r_rf['f_xi']:.5g}) - ({r_rf['x_i']:.6f})({r_rf['f_x0']:.5g})}}{{{r_rf['f_xi']:.5g} - ({r_rf['f_x0']:.5g})}} = {r_rf['x_sig']:.6f}$ ($f = {r_rf['f_xsig']:.5g}$)."
            )
        else:
            pasos_rf_vivos.append(
                f"**Paso $k = {r_rf['iteracion']}$:** Intervalo $[{r_rf['a']:.6f}, {r_rf['b']:.6f}] \\implies x_k = {r_rf['x_k']:.6f}$ ($f(x_k) = {r_rf['f_xk']:.5g}$); "
                "se conserva el subintervalo con cambio de signo."
            )
    if len(hist) > 6:
        pasos_rf_vivos.append(
            f"**Cierre en iteración $k = {res['iteraciones']}$:** Raíz estimada $r = {res['raiz']:.6f}$ con error final `{res['error_final']:.3e}`."
        )
    render_deduccion_paso_a_paso_viva(
        f"Interpolación Lineal ({variante})",
        pasos_rf_vivos,
        clave_ia="raiz_rf_viva",
    )
    render_dataframe_acotado(hist, max_filas=300, key_prefix="tbl_raiz_rf")

    st.markdown("### Bloque 4 — Secantes / Cuerdas Paso a Paso")
    k_vis = render_slider_paso_k("Ver hasta la iteración $k$:", len(hist), key="vis_rf_k")
    pad = max(0.4, 0.35 * abs(b_val - a_val))
    xs, ys = _muestrear_curva(comp, "fn", "fn_np", min(a_val, b_val) - pad, max(a_val, b_val) + pad)
    fig = crear_figura_base_plotly(f"Interpolación Lineal — Paso {k_vis}", "x", "f(x)")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.5)))

    fila_k = hist[k_vis - 1]
    if variante.startswith("Cátedra"):
        p0, f0 = fila_k["x0_fijo"], fila_k["f_x0"]
        p1, f1 = fila_k["x_i"], fila_k["f_xi"]
        xr = fila_k["x_sig"]
    else:
        p0, f0 = fila_k["a"], fila_k["f_a"]
        p1, f1 = fila_k["b"], fila_k["f_b"]
        xr = fila_k["x_k"]
    fig.add_trace(
        go.Scatter(
            x=[p0, p1],
            y=[f0, f1],
            mode="lines+markers",
            name=f"Cuerda k={k_vis}",
            line=dict(color=PALETA_ARANDU["salvia"], dash="dash", width=2.2),
            marker=dict(size=8, color=PALETA_ARANDU["salvia"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xr],
            y=[0.0],
            mode="markers",
            name="Corte eje X",
            marker=dict(size=13, color=PALETA_ARANDU["marfil_nodo"], symbol="diamond", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_raiz_rf")


def _tab_interpolacion_lineal() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("interpolacion_lineal_raices")
    practica = meta.get("practica_profesora", {})
    defaults = {
        "f_str": "2*x**4 - 6*x**2 + 7*x - 2",
        "a": -2.5,
        "b": -2.0,
        "tol": 0.001,
        "variante": "Cátedra (pivote fijo)",
        **practica.get("datos_iniciales", {}),
    }
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_interpolacion_lineal(defaults)


@st.fragment
def _frag_newton_raphson(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="raiz_nr",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica $f(x)$, los extremos de verificación de arranque $[a, b]$ o la semilla $x_0$.",
    )
    solo_lectura = modo == "catedra"

    f_str = st.text_input("Ecuación $f(x) = 0$:", value=str(st.session_state["raiz_nr_f_str"]), disabled=solo_lectura, key="ui_nr_f")
    c1, c2, c3, c4 = st.columns(4)
    a_val = c1.number_input("Extremo $a$ (validación):", value=float(st.session_state["raiz_nr_a"]), format="%.4f", disabled=solo_lectura, key="ui_nr_a")
    b_val = c2.number_input("Extremo $b$ (validación):", value=float(st.session_state["raiz_nr_b"]), format="%.4f", disabled=solo_lectura, key="ui_nr_b")
    x0_val = c3.number_input("Semilla $x_0$:", value=float(st.session_state["raiz_nr_x0"]), format="%.4f", disabled=solo_lectura, key="ui_nr_x0")
    tol_val = c4.number_input("Tolerancia $E$:", value=float(st.session_state["raiz_nr_tol"]), min_value=1e-12, format="%.6f", disabled=solo_lectura, key="ui_nr_tol")

    if not solo_lectura:
        st.session_state["raiz_nr_f_str"] = f_str
        st.session_state["raiz_nr_a"] = a_val
        st.session_state["raiz_nr_b"] = b_val
        st.session_state["raiz_nr_x0"] = x0_val
        st.session_state["raiz_nr_tol"] = tol_val

    try:
        comp = compilar_funcion_raices(f_str)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Ecuación f(x) para Newton-Raphson",
            sugerencia_defecto="Ingresá una función derivable f(x) con sintaxis Python válida (*, **).",
            ejemplo_defecto="x**3 - 2*x - 5",
        )
        return

    st.latex(rf"f'(x) = {comp['d1_latex']}, \qquad f''(x) = {comp['d2_latex']}")
    fourier = verificar_condiciones_fourier(comp["fn"], comp["d1"], comp["d2"], min(a_val, b_val), max(a_val, b_val))
    st.info(
        rf"**Validación de arranque seguro en $[{min(a_val, b_val):g}, {max(a_val, b_val):g}]$ (Condiciones de Fourier):** "
        f"{'Sugerido $x_0 = ' + str(fourier['x0_sugerido']) + '$' if fourier['fourier_cumple'] else 'Sin garantía estricta en extremos (el salto inicial podría alejarse de la raíz)'} "
        f"— {fourier['detalle_fourier']}"
    )

    st.markdown("### Bloque 3 — Iteraciones de Newton-Raphson en Vivo")
    res = metodo_newton_raphson(comp["fn"], comp["d1"], x0_val, tol=tol_val, max_iter=int(st.session_state["raiz_nr_max_iter"]))
    if not res["convergente"] and res.get("mensaje"):
        st.warning(res["mensaje"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Raíz aproximada", f"{res['raiz']:.7f}")
    m2.metric("Iteraciones", res["iteraciones"])
    m3.metric("Error final", f"{res['error_final']:.3e}")
    registrar_contexto_ia(
        "Derivadas, Fourier y Resultado Newton-Raphson",
        f"- **Derivadas:** $$f'(x) = {comp['d1_latex']}, \\qquad f''(x) = {comp['d2_latex']}$$\n"
        f"- **Validación Fourier:** {fourier['detalle_fourier']}\n"
        f"- **Raíz:** `{res['raiz']:.7f}` | **Iteraciones:** `{res['iteraciones']}` | **Error:** `{res['error_final']:.3e}`",
        clave="raiz_nr_res",
    )

    hist = res["historial"]
    pasos_nr_vivos = [
        (
            f"**Derivadas analíticas y chequeo de Fourier:** $f'(x) = {comp['d1_latex']}$ y $f''(x) = {comp['d2_latex']}$. "
            f"{fourier['detalle_fourier']}. Iniciamos con semilla $x_0 = {x0_val:g}$."
        )
    ]
    for r_nr in hist[:6]:
        if r_nr["x_sig"] is not None:
            pasos_nr_vivos.append(
                f"**Iteración $k = {r_nr['iteracion']}$:** $x_{{{r_nr['iteracion']}}} = {r_nr['x_k']:.6g} - \\frac{{{r_nr['f_xk']:.6g}}}{{{r_nr['df_xk']:.6g}}} = {r_nr['x_sig']:.7f}$ "
                f"($|\\Delta x| = {r_nr['error']:.3e}$)."
            )
        else:
            pasos_nr_vivos.append(
                f"**Iteración $k = {r_nr['iteracion']}$:** Detenido por tangente casi horizontal ($f'(x_k) = {r_nr['df_xk']:.3e}$)."
            )
    if len(hist) > 6:
        pasos_nr_vivos.append(
            f"**Cierre en $k = {res['iteraciones']}$:** Raíz final $x^* = {res['raiz']:.7f}$ con error `{res['error_final']:.3e}`."
        )
    render_deduccion_paso_a_paso_viva(
        f"Newton-Raphson (x0={x0_val:g})",
        pasos_nr_vivos,
        clave_ia="raiz_nr_viva",
    )
    render_dataframe_acotado(hist, max_filas=300, key_prefix="tbl_raiz_nr")

    st.markdown("### Bloque 4 — Rectas Tangentes Paso a Paso")
    k_vis = render_slider_paso_k("Ver hasta la iteración $k$:", max(1, len(hist)), key="vis_nr_k")
    puntos_x = [row["x_k"] for row in hist[:k_vis]] + [row["x_sig"] for row in hist[:k_vis] if row["x_sig"] is not None]
    xmin, xmax = min(puntos_x) - 0.5, max(puntos_x) + 0.5
    xs, ys = _muestrear_curva(comp, "fn", "fn_np", xmin, xmax)

    fig = crear_figura_base_plotly(f"Newton-Raphson — Tangentes hasta k = {k_vis}", "x", "f(x)")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.5)))
    for idx, row in enumerate(hist[:k_vis], start=1):
        if row["x_sig"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=[row["x_k"], row["x_k"], row["x_sig"]],
                    y=[0.0, row["f_xk"], 0.0],
                    mode="lines+markers",
                    name=f"Tangente k={idx}",
                    line=dict(color=PALETA_ARANDU["terracota"], width=2.0, dash="dash"),
                    marker=dict(size=8, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.6)),
                    opacity=1.0 if idx == k_vis else 0.45,
                )
            )
    render_plotly_arandu(fig, key="plot_raiz_nr")


def _tab_newton_raphson() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("newton_raphson")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "x**3 - 2*x - 5", "a": 2.0, "b": 2.5, "x0": 2.5, "tol": 1e-4, "max_iter": 50},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_newton_raphson(defaults)


@st.fragment
def _frag_punto_fijo_aitken(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="raiz_pf",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica $f(x)$, el despeje $x = g(x)$ o la semilla $x_0$ para evaluar estabilidad.",
    )
    solo_lectura = modo == "catedra"

    c1, c2 = st.columns(2)
    f_str = c1.text_input("Ecuación $f(x) = 0$:", value=str(st.session_state["raiz_pf_f_str"]), disabled=solo_lectura, key="ui_pf_f")
    g_str = c2.text_input("Despeje $x = g(x)$:", value=str(st.session_state["raiz_pf_g_str"]), disabled=solo_lectura, key="ui_pf_g")
    c3, c4 = st.columns(2)
    x0_val = c3.number_input("Semilla $x_0$:", value=float(st.session_state["raiz_pf_x0"]), format="%.4f", disabled=solo_lectura, key="ui_pf_x0")
    tol_val = c4.number_input("Tolerancia $E$:", value=float(st.session_state["raiz_pf_tol"]), min_value=1e-12, format="%.6f", disabled=solo_lectura, key="ui_pf_tol")

    if not solo_lectura:
        st.session_state["raiz_pf_f_str"] = f_str
        st.session_state["raiz_pf_g_str"] = g_str
        st.session_state["raiz_pf_x0"] = x0_val
        st.session_state["raiz_pf_tol"] = tol_val

    try:
        comp_g = compilar_g_punto_fijo(g_str)
        deriv_x0 = abs(comp_g["dgn"](x0_val))
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Despeje x = g(x) o semilla x0",
            sugerencia_defecto="Escribí g(x) con variable x usando sintaxis Python y evitá puntos x0 donde g(x0) o g'(x0) se indefina.",
            ejemplo_defecto="g(x) = (2*x + 5)**(1/3), x0 = 2.0",
        )
        return

    st.latex(rf"g'(x) = {comp_g['dg_latex']}, \qquad |g'(x_0)| = {deriv_x0:.4f}")
    if deriv_x0 < 1.0:
        st.success(
            r"**$|g'(x_0)| < 1$:** La pendiente actúa como factor multiplicativo reductor en cada iteración "
            r"del bucle (**esquema contractivo**), garantizando convergencia local hacia la raíz."
        )
    else:
        st.warning(
            r"**$|g'(x_0)| \ge 1$:** El factor amplifica el error en cada vuelta del bucle y la secuencia simple puede diverger."
        )

    st.markdown(r"### Bloque 3 — Punto Fijo vs Aceleración $\Delta^2$ de Aitken")
    res = metodo_punto_fijo_aitken(comp_g["gn"], x0_val, tol=tol_val, max_iter=int(st.session_state["raiz_pf_max_iter"]))
    pf, ait = res["punto_fijo"], res["aitken"]
    m1, m2 = st.columns(2)
    m1.metric("Punto Fijo simple", f"{pf['raiz']:.6f}", f"{pf['iteraciones']} iters | Err {pf['error_final']:.2e}")
    m2.metric("Aceleración Δ² de Aitken", f"{ait['raiz']:.6f}", f"{ait['ciclos']} ciclos | Err {ait['error_final']:.2e}")
    registrar_contexto_ia(
        "Derivada g'(x), Punto Fijo y Aitken Δ²",
        f"- **Derivada:** $$g'(x) = {comp_g['dg_latex']}, \\qquad |g'(x_0)| = {deriv_x0:.4f}$$\n"
        f"- **Punto Fijo simple:** raíz `{pf['raiz']:.6f}` (`{pf['iteraciones']}` iters, error `{pf['error_final']:.2e}`)\n"
        f"- **Aitken $\\Delta^2$:** raíz `{ait['raiz']:.6f}` (`{ait['ciclos']}` ciclos, error `{ait['error_final']:.2e}`)",
        clave="raiz_pf_res",
    )

    pasos_pf_vivos = [
        (
            f"**Chequeo de condición contractiva en $x_0 = {x0_val:g}$:** Derivando $g(x)$ obtenemos $g'(x) = {comp_g['dg_latex']}$, "
            f"con $|g'(x_0)| = {deriv_x0:.4f}$ ({'menor que 1: esquema contractivo convergente' if deriv_x0 < 1.0 else 'mayor o igual a 1: riesgo de divergencia'})."
        )
    ]
    for r_ait in ait["historial"][:5]:
        pasos_pf_vivos.append(
            f"**Ciclo Aitken $k = {r_ait['ciclo']}$:** Partiendo de $x_0 = {r_ait['x0']:.6f}$, dos pasos simples dan "
            f"$x_1 = {r_ait['x1']:.6f}$ y $x_2 = {r_ait['x2']:.6f}$. El salto extrapolado vale "
            f"$S = {r_ait['x2']:.6f} - \\frac{{({r_ait['x2']:.6f} - {r_ait['x1']:.6f})^2}}{{{r_ait['denominador']:.6g}}} = {r_ait['s_acelerado']:.6f}$ "
            f"($|g(S) - S| = {r_ait['error']:.3e}$)."
        )
    pasos_pf_vivos.append(
        f"**Comparación de eficiencia:** Punto Fijo simple necesitó `{pf['iteraciones']}` iteraciones (raíz `{pf['raiz']:.6f}`), "
        f"mientras Aceleración $\\Delta^2$ de Aitken cerró en `{ait['ciclos']}` ciclos (raíz `{ait['raiz']:.6f}`)."
    )
    render_deduccion_paso_a_paso_viva(
        f"Punto Fijo y Aceleración Δ² de Aitken (x0={x0_val:g})",
        pasos_pf_vivos,
        clave_ia="raiz_pf_viva",
    )

    t_pf, t_ait = st.tabs(["Tabla Punto Fijo", "Tabla Aitken Δ²"])
    with t_pf:
        render_dataframe_acotado(pf["historial"], max_filas=300, key_prefix="tbl_raiz_pf")
    with t_ait:
        render_dataframe_acotado(ait["historial"], max_filas=300, key_prefix="tbl_raiz_ait")

    if pf["historial"]:
        st.markdown("### Bloque 4 — Diagrama Telaraña (Cobweb) Paso a Paso")
        k_vis = render_slider_paso_k(
            "Ver hasta iteración de Punto Fijo $k$:",
            len(pf["historial"]),
            key="vis_pf_k",
            default_val=min(len(pf["historial"]), 8),
        )
        sub = pf["historial"][:k_vis]
        pts = [r["x_ant"] for r in sub] + [r["x_sig"] for r in sub]
        xmin, xmax = min(pts) - 0.3, max(pts) + 0.3
        xs, ys = _muestrear_curva(comp_g, "gn", "gn_np", xmin, xmax)

        fig = crear_figura_base_plotly(f"Punto Fijo (y = g(x) vs y = x) hasta k = {k_vis}", "x", "y")
        fig.add_trace(go.Scatter(x=[xmin, xmax], y=[xmin, xmax], mode="lines", name="y = x", line=dict(color=PALETA_ARANDU["texto_atenuado"], dash="dash")))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="y = g(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.5)))

        cob_x, cob_y = [sub[0]["x_ant"]], [sub[0]["x_ant"]]
        for r in sub:
            cob_x.extend([r["x_ant"], r["x_sig"]])
            cob_y.extend([r["x_sig"], r["x_sig"]])
        fig.add_trace(
            go.Scatter(
                x=cob_x,
                y=cob_y,
                mode="lines+markers",
                name="Trayectoria Telaraña",
                line=dict(color=PALETA_ARANDU["terracota"], width=2.2),
                marker=dict(size=7, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.4)),
            )
        )
        render_plotly_arandu(fig, key="plot_raiz_pf")


def _tab_punto_fijo_aitken() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("punto_fijo_aitken")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "exp(x) - x**2 - 5", "g_str": "log(x**2 + 5)", "x0": 2.5, "tol": 1e-4, "max_iter": 50},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_punto_fijo_aitken(defaults)


def vista_raices() -> None:
    inject_arandu_css()
    st.title("Raíces de Ecuaciones No Lineales")
    st.markdown(
        r"Localización inicial, encierro con cambio de signo garantizado (**Bolzano**) y algoritmos abiertos "
        r"desde convergencia lineal hasta orden cuadrático."
    )
    activo = render_selector_fichas_metodos(
        "raices",
        (
            {
                "numero": "1",
                "clave_svg": "biseccion",
                "titulo": "Bisección (y Descartes/Tanteo)",
                "subtitulo": "Bolzano · Encierro",
                "resumen": "Divide [a, b] por la mitad asegurando la raíz siempre que f(a)·f(b) < 0.",
                "pista": "Orden 1 · Encierro seguro",
            },
            {
                "numero": "2",
                "clave_svg": "regula_falsi",
                "titulo": "Interpolación Lineal (Regula Falsi)",
                "subtitulo": "Secante · Bolzano",
                "resumen": "Traza una recta secante dentro del intervalo para acercarse a la raíz sin soltarla.",
                "pista": "Cuerda ponderada · Bolzano",
            },
            {
                "numero": "3",
                "clave_svg": "newton_raphson",
                "titulo": "Newton-Raphson (Fourier)",
                "subtitulo": "Tangente · Criterio de Fourier",
                "resumen": "Proyecta la recta tangente f'(x) bajo el criterio de Fourier para converger más rápido.",
                "pista": "Orden 2 cuadrático · f'(x) ≠ 0",
            },
            {
                "numero": "4",
                "clave_svg": "punto_fijo",
                "titulo": "Punto Fijo & Aitken Δ²",
                "subtitulo": "Recurrencia · Aceleración Aitken",
                "resumen": "Itera x = g(x) cuando |g'(x)| < 1 y acelera la sucesión mediante Aitken Δ².",
                "pista": "Telaraña g(x) · Aceleración Δ²",
            },
        ),
    )
    (
        _tab_biseccion,
        _tab_interpolacion_lineal,
        _tab_newton_raphson,
        _tab_punto_fijo_aitken,
    )[activo]()


vista_raices()
