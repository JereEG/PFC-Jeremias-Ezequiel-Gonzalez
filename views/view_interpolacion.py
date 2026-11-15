"""
Vista de Interpolación Polinomial - Arandú.
Interpolación Lineal (Directa/Inversa), Cuadrática Inversa, Polinomio de Lagrange,
Newton-Gregory (Ascendente/Descendente) y Fórmula de Gauss (Progresiva/Regresiva)
con detector de equidistancia en vivo.
Arquitectura Static-First sin dependencias de SymPy en tiempo de renderizado.
"""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.interpolacion import (
    calcular_diferencias_ascendentes,
    construir_polinomio_simbolico_lagrange,
    diagnosticar_espaciamiento_nodos,
    evaluar_curva_gregory_parcial,
    evaluar_curva_lagrange,
    interpolar_cuadratica_inversa,
    interpolar_gauss_progresiva,
    interpolar_gauss_regresiva,
    interpolar_lagrange,
    interpolar_lineal_directa,
    interpolar_lineal_inversa,
    interpolar_newton_gregory_ascendente,
    interpolar_newton_gregory_descendente,
    nodos_usados_equidistante,
    recomendar_metodo_equidistante,
    regenerar_tabla_interpolacion,
)
from views.theme import PALETA_ARANDU, hex_a_rgba
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    limpiar_valores_plotly,
    render_bloque_1_teoria_y_ejemplo,
    render_controles_edicion_practica,
    render_data_editor_arandu,
    render_dataframe_acotado,
    render_excepcion_validacion,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
    render_subtitulo_controles_vivo,
    rerun_fragmento_seguro,
)


def _editor_tabla_xy(
    clave_state: str,
    solo_lectura: bool,
    col_x_label: str = "x",
    col_y_label: str = "y",
) -> tuple[list[float], list[float]]:
    xs = st.session_state[f"{clave_state}_x"]
    ys = st.session_state[f"{clave_state}_y"]
    df_in = pd.DataFrame({col_x_label: xs, col_y_label: ys})
    df_edit = render_data_editor_arandu(
        df_in,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic" if not solo_lectura else "fixed",
        disabled=solo_lectura,
        key=f"{clave_state}__editor_df",
    )
    if not solo_lectura:
        df_clean = df_edit.dropna()
        if len(df_clean) >= 2:
            xs = df_clean[col_x_label].astype(float).tolist()
            ys = df_clean[col_y_label].astype(float).tolist()
            st.session_state[f"{clave_state}_x"] = xs
            st.session_state[f"{clave_state}_y"] = ys
    return xs, ys


@st.fragment
def _frag_lineal(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="int_lin",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica los pares $(x_i, y_i)$ de la tabla o alterna entre interpolación directa e inversa.",
    )
    solo_lectura = modo == "catedra"
    xs, ys = _editor_tabla_xy("int_lin", solo_lectura)

    c1, c2 = st.columns(2)
    sentido = c1.radio(
        "Tipo de interpolación lineal:",
        ["Inversa: buscar x dado y*", "Directa: buscar y dado x*"],
        horizontal=True,
        key="ui_lin_sent",
    )
    if sentido.startswith("Inversa"):
        y_busc = c2.number_input("Valor $y^*$ buscado:", value=float(st.session_state["int_lin_y_buscado"]), format="%.4f", disabled=solo_lectura, key="ui_lin_yb")
        if not solo_lectura:
            st.session_state["int_lin_y_buscado"] = y_busc
    else:
        x_eval = c2.number_input("Valor $x^*$ a evaluar:", value=float(st.session_state["int_lin_x_eval"]), format="%.4f", disabled=solo_lectura, key="ui_lin_xe")
        if not solo_lectura:
            st.session_state["int_lin_x_eval"] = x_eval

    st.markdown("### Bloque 3 — Resultado Lineal en Vivo")
    try:
        if sentido.startswith("Inversa"):
            res = interpolar_lineal_inversa(xs, ys, y_busc)
            px, py = res["x_resultado"], res["y_buscado"]
            st.metric("x estimado por interpolación lineal inversa", f"{px:.6f}")
        else:
            res = interpolar_lineal_directa(xs, ys, x_eval)
            px, py = res["x_eval"], res["y_interpolado"]
            st.metric("y estimado por interpolación lineal directa", f"{py:.6f}")
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla (x, y) o valor buscado (Lineal)",
            sugerencia_defecto="Ingresá nodos x crecientes sin repetir y un valor objetivo dentro del intervalo de la tabla.",
            ejemplo_defecto="x = [0.0, 5.0, 10.0], x* = 2.5",
        )
        return

    st.markdown("### Bloque 4 — Recta Secante Local Paso a Paso")
    fig = crear_figura_base_plotly("Interpolación Lineal sobre Segmento Activo", "x", "y")
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="Tabla de puntos",
            line=dict(color=PALETA_ARANDU["texto_atenuado"], dash="dot"),
            marker=dict(size=8, color=PALETA_ARANDU["oro"]),
        )
    )
    fig.add_trace(go.Scatter(x=[res["x0"], res["x1"]], y=[res["y0"], res["y1"]], mode="lines+markers", name="Segmento secante", line=dict(color=PALETA_ARANDU["salvia"], width=3)))
    fig.add_trace(
        go.Scatter(
            x=[px],
            y=[py],
            mode="markers",
            name="Punto interpolado",
            marker=dict(size=13, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_int_lin")


def _tab_lineal() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("interpolacion_lineal")
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    defaults = {
        "x": raw_def.get("x", [0.0, 5.0, 10.0, 15.0]),
        "y": raw_def.get("y", [1.792, 1.519, 1.308, 1.140]),
        "x_eval": float(raw_def.get("x_objetivo", raw_def.get("x_eval", 2.5))),
        "y_buscado": float(raw_def.get("y_objetivo", raw_def.get("y_buscado", 1.702))),
    }
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_lineal(defaults)


@st.fragment
def _frag_cuadratica_inversa(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="int_cuad",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica los puntos de la tabla o la ordenada objetivo $y^*$.",
    )
    solo_lectura = modo == "catedra"
    xs, ys = _editor_tabla_xy("int_cuad", solo_lectura)

    y_busc = st.number_input("Ordenada objetivo $y^*$:", value=float(st.session_state["int_cuad_y_buscado"]), format="%.4f", disabled=solo_lectura, key="ui_cuad_yb")
    if not solo_lectura:
        st.session_state["int_cuad_y_buscado"] = y_busc

    st.markdown("### Bloque 3 — Inversión de Parábola en Vivo")
    try:
        res = interpolar_cuadratica_inversa(xs, ys, y_busc)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla (x, y) u ordenada y* (Cuadrática Inversa)",
            sugerencia_defecto="Asegurá al menos 3 puntos con ordenadas y distintas para evitar división por cero.",
            ejemplo_defecto="x = [0.0, 5.0, 10.0], y = [1.792, 1.519, 1.308]",
        )
        return

    m1, m2 = st.columns(2)
    m1.metric("x estimado (cuadrática inversa)", f"{res['x_resultado']:.6f}")
    m2.metric("Nodos usados", f"({res['x0']}, {res['x1']}, {res['x2']})")

    st.markdown("### Bloque 4 — Parábola Local vs Recta Horizontal $y = y^*$")
    x_tri = [res["x0"], res["x1"], res["x2"]]
    y_tri = [res["y0"], res["y1"], res["y2"]]
    coefs = np.polyfit(x_tri, y_tri, 2)
    x_grid = np.linspace(min(xs), max(xs), 220)
    with np.errstate(all="ignore"):
        y_parab = limpiar_valores_plotly(np.polyval(coefs, x_grid), limite_abs=1e9)

    fig = crear_figura_base_plotly("Interpolación Cuadrática Inversa", "x", "y")
    fig.add_hline(y=y_busc, line_dash="dash", line_color=PALETA_ARANDU["terracota"], annotation_text=f"y* = {y_busc}")
    fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_parab, mode="lines", name="Parábola P₂(x)", line=dict(color=PALETA_ARANDU["salvia"], width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos tabla", marker=dict(size=9, color=PALETA_ARANDU["oro"])))
    fig.add_trace(
        go.Scatter(
            x=[res["x_resultado"]],
            y=[y_busc],
            mode="markers",
            name="Raíz seleccionada",
            marker=dict(size=14, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_int_cuad")


def _tab_cuadratica_inversa() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("cuadratica_inversa")
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    defaults = {
        "x": raw_def.get("x", [0.0, 5.0, 10.0, 15.0]),
        "y": raw_def.get("y", [1.792, 1.519, 1.308, 1.140]),
        "y_buscado": float(raw_def.get("y_objetivo", raw_def.get("y_buscado", 1.702))),
    }
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_cuadratica_inversa(defaults)


@st.fragment
def _frag_lagrange(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="int_lag",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica nodos libres $(x_i, y_i)$ o prueba remuestrear a $N$ nodos para observar el fenómeno de Runge.",
    )
    solo_lectura = modo == "catedra"
    xs, ys = _editor_tabla_xy("int_lag", solo_lectura, col_x_label="x (o T)", col_y_label="y (o P)")

    c1, c2 = st.columns(2)
    x_eval = c1.number_input("Punto $x^*$ a interpolar:", value=float(st.session_state["int_lag_x_eval"]), format="%.4f", disabled=solo_lectura, key="ui_lag_xe")
    if not solo_lectura:
        st.session_state["int_lag_x_eval"] = x_eval

    if not solo_lectura:
        with c2:
            nuevo_n = st.slider("Remuestrear a $N$ nodos equidistantes (opcional):", 2, 10, len(xs), key="ui_lag_n")
            if st.button("Aplicar N nodos", key="btn_lag_remuestrear"):
                x_new, y_new, _ = regenerar_tabla_interpolacion(defaults["x"], defaults["y"], nuevo_n=nuevo_n)
                st.session_state["int_lag_x"] = x_new
                st.session_state["int_lag_y"] = y_new
                if "int_lag__editor_df" in st.session_state:
                    del st.session_state["int_lag__editor_df"]
                rerun_fragmento_seguro()

    st.caption(
        "**Alerta sobre grado alto (Fenómeno de Runge):** Existe un único polinomio de grado $n-1$ que pasa por todos los nodos; "
        "sin embargo, aumentar demasiado $N$ sobre nodos equidistantes introduce oscilaciones pronunciadas cerca de los extremos."
    )

    st.markdown("### Bloque 3 — Bases $L_i(x)$ y Polinomio Combinado")
    try:
        res = interpolar_lagrange(xs, ys, x_eval)
        simb = construir_polinomio_simbolico_lagrange(xs, ys)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos (x, y) (Lagrange)",
            sugerencia_defecto="Usá al menos 2 nodos con abscisas x distintas sin duplicar.",
            ejemplo_defecto="x = [-1.0, 0.0, 2.0, 3.0]",
        )
        return

    m1, m2 = st.columns(2)
    m1.metric(f"P({x_eval:g}) estimado", f"{res['y_interpolado']:.6f}")
    m2.metric("Grado del polinomio", res["grado"])
    st.latex(rf"P_{{{res['grado']}}}(x) = {simb['latex_expandido']}")
    render_dataframe_acotado(res["terminos_detalle"], max_filas=300, key_prefix="tbl_int_lag")

    st.markdown("### Bloque 4 — Construcción Geométrica por Suma Parcial de Bases")
    k_vis = render_slider_paso_k("Sumar hasta término $i = k$:", len(xs) - 1, key="vis_lag_k", min_val=0)
    x_grid = np.linspace(min(min(xs), x_eval) - 2, max(max(xs), x_eval) + 2, 220)
    y_full = limpiar_valores_plotly(evaluar_curva_lagrange(xs, ys, x_grid), limite_abs=1e9)

    y_parcial_arr = np.zeros_like(x_grid, dtype=float)
    for idx in range(k_vis + 1):
        y_parcial_arr += float(ys[idx]) * evaluar_curva_lagrange(xs, ys, x_grid, indice_base=idx)
    y_parcial = limpiar_valores_plotly(y_parcial_arr, limite_abs=1e9)

    fig = crear_figura_base_plotly(f"Polinomio de Lagrange — Suma parcial i = 0..{k_vis}", "x", "P(x)")
    fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_full, mode="lines", name="P(x) completo", line=dict(color=PALETA_ARANDU["texto_atenuado"], dash="dash")))
    fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_parcial, mode="lines", name=f"Suma hasta i={k_vis}", line=dict(color=PALETA_ARANDU["salvia"], width=2.8)))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos", marker=dict(size=9, color=PALETA_ARANDU["oro"])))
    fig.add_trace(
        go.Scatter(
            x=[x_eval],
            y=[res["y_interpolado"]],
            mode="markers",
            name="P(x*)",
            marker=dict(size=13, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_int_lag")


def _tab_lagrange() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("lagrange")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"x": [100.0, 120.0, 200.0, 220.0], "y": [1.01, 3.61, 15.5, 23.2], "x_eval": 180.0},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_lagrange(defaults)


_SUPERS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_SUBS = str.maketrans("0123456789-n", "₀₁₂₃₄₅₆₇₈₉₋ₙ")

_PRESETS_NEWTON_GREGORY: dict[str, dict] = {
    "tp5_ej1_asc": {
        "titulo": "TP5 Ej. 1a — Ascendente en x = 0.35",
        "enunciado": "Con la tabla equiespaciada de la cátedra (h = 0.3), construir la tabla de diferencias finitas y evaluar P(0.35) mediante Newton-Gregory Ascendente.",
        "referencia": r"Ancla $x_0 = 0.2$, $h = 0.3$, $u = 0.5 \implies P_4(0.35) = 1.214734375$.",
        "x": [0.2, 0.5, 0.8, 1.1, 1.4],
        "y": [0.258, 2.535, 4.902, 6.722, 10.704],
        "x_eval": 0.35,
        "exacta": None,
        "exacta_tex": None,
    },
    "tp5_ej1_desc": {
        "titulo": "TP5 Ej. 1b — Descendente en x = 1.37",
        "enunciado": "Con la misma tabla del TP5 Ej. 1 (h = 0.3), aproximar f(1.37) usando Newton-Gregory Descendente por proximidad al extremo final xn = 1.4.",
        "referencia": r"Ancla $x_n = 1.4$, $h = 0.3$, $v = -0.1 \implies P_4(1.37) = 10.06216678$.",
        "x": [0.2, 0.5, 0.8, 1.1, 1.4],
        "y": [0.258, 2.535, 4.902, 6.722, 10.704],
        "x_eval": 1.37,
        "exacta": None,
        "exacta_tex": None,
    },
    "tp5_viscosidad": {
        "titulo": "TP5 Ej. 2b — Viscosidad del Agua (T = 1.53 °C)",
        "enunciado": "Tabla experimental de viscosidad dinámica del agua con paso h = 5 °C. Estimar la viscosidad en T = 1.53 °C cerca del comienzo de la tabla.",
        "referencia": r"Ancla $x_0 = 0$, $h = 5$, $u = 0.306 \implies \mu(1.53) \approx 1.701980$ (grado 3).",
        "x": [0.0, 5.0, 10.0, 15.0],
        "y": [1.792, 1.519, 1.308, 1.140],
        "x_eval": 1.53,
        "exacta": None,
        "exacta_tex": None,
    },
    "u5_teoria_cuad": {
        "titulo": "U5 Teoría — Parábola Exacta f(x) = x² (Δ³y = 0)",
        "enunciado": "Demostración teórica de la cátedra: al muestrear un polinomio de grado 2 con h = 1, las diferencias de orden k ≥ 3 se anulan idénticamente.",
        "referencia": r"Exacta desde grado $k = 2$: $f(1.5) = 2.250000$ con $\Delta^3 y_i = 0$.",
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 1.0, 4.0, 9.0, 16.0],
        "x_eval": 1.5,
        "exacta": lambda arr: np.square(arr),
        "exacta_tex": "f(x) = x^2",
    },
    "tp6_tabla_h1": {
        "titulo": "TP6 Ej. — Tabla h = 1 (x = 12..17, x* = 12.4)",
        "enunciado": "Tabla auxiliar de seis nodos con paso unitario h = 1 de los trabajos prácticos de la cátedra, evaluada cerca del origen x₀ = 12.",
        "referencia": r"Ancla $x_0 = 12$, $h = 1$, $u = 0.4 \implies P_5(12.4) \approx 0.956555$.",
        "x": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "y": [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151],
        "x_eval": 12.4,
        "exacta": None,
        "exacta_tex": None,
    },
}

_MODOS_VISUALES_NG = (
    "Comparar Ambos Métodos",
    "Solo Ascendente (Δ)",
    "Solo Descendente (∇)",
)


def _superindice(valor: int | str) -> str:
    return str(valor).translate(_SUPERS)


def _subindice(valor: int | str) -> str:
    return str(valor).translate(_SUBS)


def _al_cambiar_preset_ng() -> None:
    """Carga el ejercicio oficial seleccionado y limpia widgets dependientes."""
    preset_id = st.session_state.get("ui_ng_preset", "tp5_ej1_asc")
    preset = _PRESETS_NEWTON_GREGORY[preset_id]
    st.session_state["int_ng_preset"] = preset_id
    st.session_state["int_ng_x"] = list(preset["x"])
    st.session_state["int_ng_y"] = list(preset["y"])
    st.session_state["int_ng_x_eval"] = float(preset["x_eval"])
    for clave in (
        "int_ng__editor_df",
        "ui_ng_xe",
        "ui_ng_xe_slider",
        "ui_ng_dh",
        "ui_ng_x0",
        "vis_ng_k",
        "ui_ng_insp_k",
        "ui_ng_insp_i",
    ):
        st.session_state.pop(clave, None)


def _al_cambiar_slider_xe_ng() -> None:
    """Sincroniza el slider de x* hacia el input numérico."""
    val = float(st.session_state["ui_ng_xe_slider"])
    st.session_state["int_ng_x_eval"] = val
    st.session_state["ui_ng_xe"] = val


def _al_cambiar_input_xe_ng() -> None:
    """Sincroniza el input numérico de x* hacia el estado interno y el slider."""
    val = float(st.session_state["ui_ng_xe"])
    st.session_state["int_ng_x_eval"] = val
    xs = st.session_state.get("int_ng_x", [0.2, 1.4])
    x_min, x_max = min(xs), max(xs)
    span = max(abs(x_max - x_min), 1.0)
    pad = round(max(0.35, 0.15 * span), 4)
    st.session_state["ui_ng_xe_slider"] = float(np.clip(val, x_min - pad, x_max + pad))


def _render_formulas_completas_gregory() -> None:
    """Muestra las fórmulas completas de Newton-Gregory Ascendente, Descendente y error."""
    with st.expander(
        "Fórmulas Oficiales Completas: Diferencias Finitas, Polinomios y Cota de Error",
        expanded=True,
    ):
        st.markdown("**1. Recurrencia de Diferencias Finitas ($h = x_{i+1} - x_i$ constante):**")
        st.latex(
            r"\Delta^0 y_i = y_i, \qquad "
            r"\Delta^k y_i = \Delta^{k-1} y_{i+1} - \Delta^{k-1} y_i, \qquad "
            r"\nabla^k y_n = \Delta^k y_{n-k}"
        )
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(r"**2A. Newton-Gregory Ascendente / Progresiva ($\Delta$, inicio $x_0$)**")
            st.latex(r"u = \frac{x - x_0}{h} \quad (u \ge 0 \text{ dentro de } [x_0, x_n])")
            st.latex(
                r"P_k^{\Delta}(x) = y_0 + \frac{\Delta y_0}{1!\,h}(x - x_0) "
                r"+ \frac{\Delta^2 y_0}{2!\,h^2}(x - x_0)(x - x_1) "
                r"+ \dots + \frac{\Delta^k y_0}{k!\,h^k}\prod_{j=0}^{k-1}(x - x_j)"
            )
            st.latex(
                r"P_k^{\Delta}(u) = y_0 + u\,\Delta y_0 + \frac{u(u-1)}{2!}\Delta^2 y_0 "
                r"+ \dots + \frac{u(u-1)\cdots(u-k+1)}{k!}\Delta^k y_0"
            )
        with col_f2:
            st.markdown(r"**2B. Newton-Gregory Descendente / Regresiva ($\nabla$, final $x_n$)**")
            st.latex(r"v = \frac{x - x_n}{h} \quad (v \le 0 \text{ dentro de } [x_0, x_n])")
            st.latex(
                r"P_k^{\nabla}(x) = y_n + \frac{\nabla y_n}{1!\,h}(x - x_n) "
                r"+ \frac{\nabla^2 y_n}{2!\,h^2}(x - x_n)(x - x_{n-1}) "
                r"+ \dots + \frac{\nabla^k y_n}{k!\,h^k}\prod_{j=0}^{k-1}(x - x_{n-j})"
            )
            st.latex(
                r"P_k^{\nabla}(v) = y_n + v\,\nabla y_n + \frac{v(v+1)}{2!}\nabla^2 y_n "
                r"+ \dots + \frac{v(v+1)\cdots(v+k-1)}{k!}\nabla^k y_n"
            )
        st.markdown("**3. Error de Truncamiento y Teorema de Unicidad:**")
        st.latex(
            r"R_k^{\Delta}(x) = \frac{f^{(k+1)}(\xi)}{(k+1)!}\prod_{j=0}^{k}(x - x_j) "
            r"\approx \binom{u}{k+1}\Delta^{k+1}y_0, \qquad "
            r"R_k^{\nabla}(x) = \frac{f^{(k+1)}(\eta)}{(k+1)!}\prod_{j=0}^{k}(x - x_{n-j}) "
            r"\approx \binom{v+k}{k+1}\nabla^{k+1}y_n"
        )
        st.caption(
            "Si k = n - 1 (grado completo usando todos los nodos), ambos polinomios son idénticos por "
            "unicidad del polinomio interpolante. Al truncar (k < n - 1), Ascendente resulta más exacto "
            "cerca de x₀ y Descendente cerca de xₙ."
        )


def _construir_html_banner_recomendacion_ng(
    xs: list[float],
    x_eval: float,
    k_vis: int,
    grado_max: int,
) -> str:
    """Genera el banner de recomendación Ascendente vs Descendente según x*."""
    x0, xn = xs[0], xs[-1]
    mid = 0.5 * (x0 + xn)
    if x_eval < x0:
        clase = "arandu-ng-banner-asc"
        recomendado = "Conviene Newton-Gregory Ascendente (Δ)"
        motivo = f"x* = {x_eval:g} extrapola a la izquierda de x₀ = {x0:g}; la diagonal superior minimiza |x - xⱼ|."
    elif x_eval > xn:
        clase = "arandu-ng-banner-desc"
        recomendado = "Conviene Newton-Gregory Descendente (∇)"
        motivo = f"x* = {x_eval:g} extrapola a la derecha de xₙ = {xn:g}; la diagonal inferior anclada en xₙ minimiza el residuo."
    elif x_eval <= mid:
        clase = "arandu-ng-banner-asc"
        recomendado = "Conviene Newton-Gregory Ascendente (Δ)"
        motivo = f"x* = {x_eval:g} está en la mitad inicial de la tabla (≤ {mid:g}), más próximo al origen x₀ = {x0:g}."
    else:
        clase = "arandu-ng-banner-desc"
        recomendado = "Conviene Newton-Gregory Descendente (∇)"
        motivo = f"x* = {x_eval:g} está en la mitad final de la tabla (> {mid:g}), más próximo al cierre xₙ = {xn:g}."

    nota_grado = (
        f"Con grado completo k = {grado_max}, ambos polinomios coinciden algebraicamente."
        if k_vis >= grado_max
        else f"Polinomio truncado en k = {k_vis} (< {grado_max}): las dos fórmulas difieren y conviene la recomendada."
    )
    return (
        f"<div class='arandu-ng-banner {clase}'>"
        f"<strong>{html.escape(recomendado)}:</strong> {html.escape(motivo)} "
        f"<span class='arandu-ng-pill-muted'>{html.escape(nota_grado)}</span>"
        f"</div>"
    )


def _construir_html_resumen_gregory(
    res_asc: dict,
    res_desc: dict,
    x_eval: float,
    k_vis: int,
    grado_max: int,
    exacta_fn,
) -> str:
    """Construye las 3 tarjetas de resultado comparativo Ascendente vs Descendente."""
    y_asc = float(res_asc["y_interpolado"])
    y_desc = float(res_desc["y_interpolado"])
    diff = abs(y_asc - y_desc)

    if exacta_fn is not None:
        y_ex = float(exacta_fn(np.array([x_eval]))[0])
        sub_comp = (
            f"Valor exacto f({x_eval:g}) = {y_ex:.6f} | "
            f"|Err Δ| = {abs(y_ex - y_asc):.2e} | |Err ∇| = {abs(y_ex - y_desc):.2e}"
        )
    elif k_vis >= grado_max:
        sub_comp = "Grado completo: coincidencia exacta por teorema de unicidad."
    else:
        sub_comp = f"Truncado en k = {k_vis}: divergencia respecto al grado completo {grado_max}."

    return (
        "<div class='arandu-ng-summary-grid'>"
        "<div class='arandu-ng-summary-card arandu-ng-summary-asc'>"
        "<div class='arandu-ng-summary-head'>"
        "<span>Ascendente (Δ)</span>"
        f"<span class='arandu-ng-pill-asc'>Ancla x₀ = {res_asc['x0']:g}</span>"
        "</div>"
        f"<div class='arandu-ng-summary-val'>{y_asc:.6f}</div>"
        f"<div class='arandu-ng-summary-sub'>u = {res_asc['u']:.4f} | h = {res_asc['h']:g} | k = {k_vis}</div>"
        "</div>"
        "<div class='arandu-ng-summary-card arandu-ng-summary-desc'>"
        "<div class='arandu-ng-summary-head'>"
        "<span>Descendente (∇)</span>"
        f"<span class='arandu-ng-pill-desc'>Ancla xₙ = {res_desc['xn']:g}</span>"
        "</div>"
        f"<div class='arandu-ng-summary-val'>{y_desc:.6f}</div>"
        f"<div class='arandu-ng-summary-sub'>v = {res_desc['v']:.4f} | h = {res_desc['h']:g} | k = {k_vis}</div>"
        "</div>"
        "<div class='arandu-ng-summary-card'>"
        "<div class='arandu-ng-summary-head'>"
        "<span>Comparación |Δ - ∇|</span>"
        f"<span class='arandu-ng-pill-muted'>Grado {k_vis} / {grado_max}</span>"
        "</div>"
        f"<div class='arandu-ng-summary-val'>{diff:.6f}</div>"
        f"<div class='arandu-ng-summary-sub'>{html.escape(sub_comp)}</div>"
        "</div>"
        "</div>"
    )


def _construir_html_tabla_escalonada_gregory(
    xs: list[float],
    tabla: np.ndarray,
    k_vis: int,
    modo_visual: str,
) -> str:
    """Renderiza la matriz escalonada de diferencias finitas para Newton-Gregory."""
    n = len(xs)
    show_asc = modo_visual != "Solo Descendente (∇)"
    show_desc = modo_visual != "Solo Ascendente (Δ)"

    rutas: dict[tuple[int, int], tuple[str, str, str]] = {}
    for k in range(n):
        act_asc = show_asc
        act_desc = show_desc
        i_asc = 0
        i_desc = n - 1 - k
        if act_asc and act_desc and i_asc == i_desc:
            rutas[(i_asc, k)] = (
                "arandu-ng-cell-both",
                "arandu-ng-badge-desc",
                f"<span class='arandu-sr-only'>Ruta ascendente y descendente: </span>Δ{_superindice(k)}y₀ = ∇{_superindice(k)}yₙ",
            )
        else:
            if act_asc:
                rutas[(i_asc, k)] = (
                    "arandu-ng-cell-asc",
                    "arandu-ng-badge-asc",
                    f"<span class='arandu-sr-only'>Ruta ascendente: </span>Δ{_superindice(k)}y₀",
                )
            if act_desc:
                rutas[(i_desc, k)] = (
                    "arandu-ng-cell-desc",
                    "arandu-ng-badge-desc",
                    f"<span class='arandu-sr-only'>Ruta descendente: </span>∇{_superindice(k)}yₙ",
                )
    return _construir_html_tabla_diferencias_comun(
        xs,
        tabla,
        k_vis,
        rutas=rutas,
        anchor_gauss=None,
        aria_caption="Tabla escalonada de diferencias finitas ascendentes y descendentes de Newton-Gregory.",
    )


def _construir_html_tabla_diferencias_comun(
    xs: list[float],
    tabla: np.ndarray,
    k_vis: int,
    rutas: dict[tuple[int, int], tuple[str, str, str]],
    anchor_gauss: int | None,
    aria_caption: str,
) -> str:
    """Construye la tabla HTML escalonada de 2n - 1 filas para Gregory y Gauss."""
    n = len(xs)
    total_rows = 2 * n - 1

    ths = [
        "<th scope='col' class='arandu-ng-diff-th'>i</th>",
        "<th scope='col' class='arandu-ng-diff-th'>xᵢ</th>",
        "<th scope='col' class='arandu-ng-diff-th'>yᵢ (k=0)</th>",
    ]
    for k in range(1, n):
        etiqueta = f"Δ{_superindice(k)}y" if k > 1 else "Δy"
        ths.append(f"<th scope='col' class='arandu-ng-diff-th'>{etiqueta} (k={k})</th>")

    filas_html = []
    for r in range(total_rows):
        celdas = []
        if r % 2 == 0:
            idx = r // 2
            tags = []
            if idx == 0:
                tags.append("x₀")
            if anchor_gauss is not None and idx == anchor_gauss:
                tags.append("Centro")
            if idx == n - 1:
                tags.append("xₙ")
            tag = f" ({' · '.join(tags)})" if tags else ""
            celdas.append(f"<th scope='row' class='arandu-ng-diff-idx'>{idx}</th>")
            celdas.append(f"<td class='arandu-ng-diff-idx'>{xs[idx]:.4g}{tag}</td>")
        else:
            celdas.append("<td class='arandu-ng-diff-empty'></td>")
            celdas.append("<td class='arandu-ng-diff-empty'></td>")

        for k in range(n):
            if r >= k and (r - k) % 2 == 0 and (r - k) // 2 < n - k:
                i = (r - k) // 2
                val = float(tabla[i, k])
                clases = ["arandu-ng-cell"]
                if anchor_gauss is not None and k == 0 and i == anchor_gauss:
                    clases.append("arandu-ng-cell-anchor")

                badge = ""
                if (i, k) in rutas:
                    cell_cls, badge_cls, badge_inner = rutas[(i, k)]
                    clases.append(cell_cls)
                    badge = f"<span class='{badge_cls}'>{badge_inner}</span>"
                    if k > k_vis:
                        clases.append("arandu-ng-cell-trunc")

                trunc_sr = (
                    f"<span class='arandu-sr-only'> (orden superior al grado actual {k_vis})</span>"
                    if k > k_vis and (i, k) in rutas
                    else ""
                )

                if k == 0:
                    tip = f"y{_subindice(i)} = f({xs[i]:.4g}) = {val:.6g}"
                else:
                    sup_prev = _superindice(k - 1) if k > 1 else ""
                    tip = (
                        f"Δ{_superindice(k)}y{_subindice(i)} = "
                        f"Δ{sup_prev}y{_subindice(i + 1)} - Δ{sup_prev}y{_subindice(i)} = "
                        f"{tabla[i + 1, k - 1]:.6g} - {tabla[i, k - 1]:.6g} = {val:.6g}"
                    )

                celdas.append(
                    f"<td class='{' '.join(clases)}' title='{html.escape(tip)}'>"
                    f"{badge}{val:.5f}{trunc_sr}</td>"
                )
            else:
                celdas.append("<td class='arandu-ng-diff-empty'></td>")
        filas_html.append(f"<tr>{''.join(celdas)}</tr>")

    return (
        "<div class='arandu-ng-diff-wrap' role='region' "
        f"aria-label='{html.escape(aria_caption)}' tabindex='0'>"
        "<table class='arandu-ng-diff-table' "
        f"aria-label='{html.escape(aria_caption)}'>"
        f"<caption class='arandu-sr-only'>{html.escape(aria_caption)}</caption>"
        f"<thead><tr>{''.join(ths)}</tr></thead>"
        f"<tbody>{''.join(filas_html)}</tbody>"
        "</table></div>"
    )


_META_MODOS_EQ: dict[str, dict[str, str]] = {
    "ascendente": {
        "suf": "asc",
        "simbolo": "Δ",
        "simbolo_tex": r"\Delta",
        "param": "u",
        "ancla_sym": "x₀",
        "titulo": "Polinomio Ascendente (Newton-Gregory Δ)",
        "cell_cls": "arandu-ng-cell-asc",
        "badge_cls": "arandu-ng-badge-asc",
    },
    "descendente": {
        "suf": "desc",
        "simbolo": "∇",
        "simbolo_tex": r"\nabla",
        "param": "v",
        "ancla_sym": "xₙ",
        "titulo": "Polinomio Descendente (Newton-Gregory ∇)",
        "cell_cls": "arandu-ng-cell-desc",
        "badge_cls": "arandu-ng-badge-desc",
    },
    "gauss_progresiva": {
        "suf": "gprog",
        "simbolo": "Gp",
        "simbolo_tex": r"\mathrm{Gp}",
        "param": "u",
        "ancla_sym": "x_c",
        "titulo": "Fórmula de Gauss Progresiva (central ↓ primero)",
        "cell_cls": "arandu-ng-cell-gprog",
        "badge_cls": "arandu-ng-badge-gprog",
    },
    "gauss_regresiva": {
        "suf": "greg",
        "simbolo": "Gr",
        "simbolo_tex": r"\mathrm{Gr}",
        "param": "u",
        "ancla_sym": "x_c",
        "titulo": "Fórmula de Gauss Regresiva (central ↑ primero)",
        "cell_cls": "arandu-ng-cell-greg",
        "badge_cls": "arandu-ng-badge-greg",
    },
}


def _formatear_factor_tex(param_val: float, delta: int) -> str:
    if delta == 0:
        return rf"\left({param_val:.4f}\right)"
    signo = "+" if delta > 0 else "-"
    return rf"\left({param_val:.4f}{signo}{abs(delta)}\right)"


def _construir_desglose_formula_lado(
    res: dict,
    modo: str,
    k_vis: int,
    atenuado: bool = False,
) -> tuple[str, str]:
    """Construye la tarjeta HTML paso a paso y el bloque LaTeX del polinomio."""
    meta = _META_MODOS_EQ[modo]
    suf = meta["suf"]
    simbolo = meta["simbolo"]
    simbolo_tex = meta["simbolo_tex"]
    param_nom = meta["param"]
    ancla_sym = meta["ancla_sym"]
    if modo == "ascendente":
        param_val = float(res["u"])
        ancla_val = float(res["x0"])
        y_sym = "y₀"
    elif modo == "descendente":
        param_val = float(res["v"])
        ancla_val = float(res["xn"])
        y_sym = "yₙ"
    else:
        param_val = float(res["u"])
        ancla_val = float(res["x0_central"])
        y_sym = f"y{_subindice(res['anchor_idx'])}"

    terminos_k = res["terminos"][: k_vis + 1]
    filas = []
    partes_tex_u = []
    partes_tex_val = []
    factores_acum_label: list[str] = []
    factores_acum_tex: list[str] = []

    for row in terminos_k:
        k = int(row["k"])
        dif = float(row["diferencia"])
        contrib = float(row["termino"])
        if k == 0:
            etiqueta = y_sym
            partes_tex_u.append(f"{dif:.5g}")
            partes_tex_val.append(f"{contrib:.6f}")
        else:
            if modo in ("gauss_progresiva", "gauss_regresiva"):
                delta = int(row["delta_factor"])
                fila_tab = int(row["fila_tabla"])
                diff_lbl = f"Δ{_superindice(k)}y{_subindice(fila_tab)}"
            elif modo == "ascendente":
                delta = -(k - 1)
                diff_lbl = f"Δ{_superindice(k)}y₀"
            else:
                delta = k - 1
                diff_lbl = f"∇{_superindice(k)}yₙ"

            if delta == 0:
                factores_acum_label.append(param_nom)
            else:
                sig = "+" if delta > 0 else "-"
                factores_acum_label.append(f"({param_nom}{sig}{abs(delta)})")
            factores_acum_tex.append(_formatear_factor_tex(param_val, delta))

            prod_lbl = "".join(factores_acum_label)
            etiqueta = f"[{prod_lbl}/{k}!] · {diff_lbl}" if k > 1 else f"{prod_lbl} · {diff_lbl}"
            num_tex = "".join(factores_acum_tex)
            if k == 1:
                partes_tex_u.append(rf"{num_tex}({dif:.5g})")
            else:
                partes_tex_u.append(rf"\frac{{{num_tex}}}{{{math.factorial(k)}}}({dif:.5g})")
            partes_tex_val.append(f"{contrib:+.6f}")

        signo = "+" if k > 0 and contrib >= 0 else ""
        filas.append(
            "<div class='arandu-ng-step-row'>"
            f"<span>k={k}: {html.escape(etiqueta)}</span>"
            f"<span class='arandu-ng-step-val-{suf}'>{signo}{contrib:.6f}</span>"
            "</div>"
        )

    clase_mut = " arandu-ng-formula-muted" if atenuado else ""
    aviso_atenuado = (
        "<div class='arandu-sr-only'>Enfoque secundario para la posición actual de x estrella.</div>"
        if atenuado
        else ""
    )
    val_k_vis = float(terminos_k[-1]["acumulado"])
    html_card = (
        f"<div class='arandu-ng-formula-card arandu-ng-formula-card-{suf}{clase_mut}'>"
        f"{aviso_atenuado}"
        "<div class='arandu-ng-formula-head'>"
        "<div>"
        f"<div class='arandu-ng-summary-head'>{html.escape(meta['titulo'])}</div>"
        f"<div class='arandu-ng-summary-sub'>Anclada en {ancla_sym} = {ancla_val:g} | {param_nom} = {param_val:.4f}</div>"
        "</div>"
        f"<div class='arandu-ng-formula-icon-{suf}' aria-hidden='true'>{html.escape(simbolo)}</div>"
        "</div>"
        f"{''.join(filas)}"
        f"<div class='arandu-ng-step-total-{suf}'>"
        f"<span>P_{k_vis}({res['x_eval']:g}) =</span>"
        f"<span>{val_k_vis:.6f}</span>"
        "</div>"
        "</div>"
    )
    tex_pol = (
        rf"P_{{{k_vis}}}^{{{simbolo_tex}}}({res['x_eval']:g}) = "
        + " + ".join(partes_tex_u)
        + rf" = {' '.join(partes_tex_val)} = {val_k_vis:.6f}"
    )
    return html_card, tex_pol


def _render_inspector_celda_diferencia(xs: list[float], tabla: np.ndarray) -> None:
    """Muestra la resta exacta Δ^k y_i = Δ^(k-1)y_(i+1) - Δ^(k-1)y_i para cualquier celda."""
    n = len(xs)
    ordenes = list(range(n))
    st.session_state["ui_ng_insp_k"] = min(int(st.session_state.get("ui_ng_insp_k", min(1, n - 1))), n - 1)
    ci1, ci2, ci3 = st.columns([1.0, 1.0, 3.0])
    k_sel = ci1.selectbox("Inspeccionar orden k:", ordenes, key="ui_ng_insp_k")
    max_i = n - 1 - k_sel
    st.session_state["ui_ng_insp_i"] = min(int(st.session_state.get("ui_ng_insp_i", 0)), max_i)
    i_sel = ci2.selectbox("Fila base i:", list(range(max_i + 1)), key="ui_ng_insp_i")

    val = float(tabla[i_sel, k_sel])
    pos = (
        "Diagonal Ascendente y Descendente"
        if (i_sel == 0 and i_sel == max_i)
        else (
            "Diagonal Ascendente (Δ)"
            if i_sel == 0
            else ("Diagonal Descendente (∇)" if i_sel == max_i else "Diferencia interior")
        )
    )
    if k_sel == 0:
        detalle = f"y{_subindice(i_sel)} = f({xs[i_sel]:g}) = {val:.6f}"
    else:
        a = float(tabla[i_sel + 1, k_sel - 1])
        b = float(tabla[i_sel, k_sel - 1])
        sup_prev = _superindice(k_sel - 1) if k_sel > 1 else ""
        detalle = (
            f"Δ{_superindice(k_sel)}y{_subindice(i_sel)} = "
            f"Δ{sup_prev}y{_subindice(i_sel + 1)} - Δ{sup_prev}y{_subindice(i_sel)} = "
            f"{a:.6f} - ({b:.6f}) = {val:.6f}"
        )
    ci3.markdown(
        f"<div class='arandu-ng-summary-card' style='margin-top:0.2rem;padding:0.65rem 0.9rem;'>"
        f"<div class='arandu-ng-summary-head'><span>Inspector de Celda ({html.escape(pos)})</span></div>"
        f"<div class='arandu-ng-summary-sub' style='color:var(--arandu-text);margin-top:0.25rem;'>{html.escape(detalle)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_grafica_gregory_comparativa(
    xs: list[float],
    ys: list[float],
    x_eval: float,
    k_vis: int,
    grado_max: int,
    modo_visual: str,
    res_asc: dict,
    res_desc: dict,
    exacta_fn,
    exacta_tex: str | None,
) -> None:
    """Genera el gráfico Plotly interactivo comparando Ascendente, Descendente y grado completo."""
    x_min = min(min(xs), x_eval)
    x_max = max(max(xs), x_eval)
    span = max(abs(x_max - x_min), 1.0)
    x_grid = np.linspace(x_min - 0.12 * span, x_max + 0.12 * span, 240)

    show_asc = modo_visual != "Solo Descendente (∇)"
    show_desc = modo_visual != "Solo Ascendente (Δ)"

    fig = crear_figura_base_plotly(
        f"Comparación Newton-Gregory — Grado k = {k_vis} (máx. {grado_max})",
        "x",
        "y",
    )
    fig.add_vrect(
        x0=xs[0],
        x1=xs[-1],
        fillcolor=hex_a_rgba(PALETA_ARANDU["oro"], 0.06),
        line_width=0,
    )
    for xv in xs:
        fig.add_vline(
            x=xv,
            line_dash="dot",
            line_color=hex_a_rgba(PALETA_ARANDU["texto_atenuado"], 0.25),
            line_width=1,
        )

    if exacta_fn is not None:
        y_ex_grid = limpiar_valores_plotly(exacta_fn(x_grid), limite_abs=1e9)
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_ex_grid,
                mode="lines",
                name=f"Exacta {exacta_tex or 'f(x)'}",
                line=dict(color=PALETA_ARANDU["terracota"], width=2, dash="dot"),
            )
        )

    if k_vis < grado_max:
        y_full = limpiar_valores_plotly(
            evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=grado_max, modo="ascendente"),
            limite_abs=1e9,
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_full,
                mode="lines",
                name=f"Polinomio completo (grado {grado_max})",
                line=dict(color=hex_a_rgba(PALETA_ARANDU["texto_atenuado"], 0.65), width=1.6, dash="dot"),
            )
        )

    if show_asc:
        y_asc_grid = limpiar_valores_plotly(
            evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=k_vis, modo="ascendente"),
            limite_abs=1e9,
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_asc_grid,
                mode="lines",
                name=f"Ascendente Δ (k={k_vis})",
                line=dict(color=PALETA_ARANDU["salvia"], width=3),
            )
        )

    if show_desc:
        y_desc_grid = limpiar_valores_plotly(
            evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=k_vis, modo="descendente"),
            limite_abs=1e9,
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_desc_grid,
                mode="lines",
                name=f"Descendente ∇ (k={k_vis})",
                line=dict(
                    color=PALETA_ARANDU["oro"],
                    width=2.8,
                    dash="dash" if show_asc else "solid",
                ),
            )
        )

    if len(xs) > 2:
        fig.add_trace(
            go.Scatter(
                x=xs[1:-1],
                y=ys[1:-1],
                mode="markers",
                name="Nodos internos",
                marker=dict(size=8, color=PALETA_ARANDU["texto_atenuado"]),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[xs[0]],
            y=[ys[0]],
            mode="markers",
            name="Ancla x₀ (Ascendente)",
            marker=dict(size=12, color=PALETA_ARANDU["salvia"], line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.6)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xs[-1]],
            y=[ys[-1]],
            mode="markers",
            name="Ancla xₙ (Descendente)",
            marker=dict(size=12, symbol="triangle-up", color=PALETA_ARANDU["oro"], line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.6)),
        )
    )

    fig.add_vline(
        x=x_eval,
        line_dash="dash",
        line_color=PALETA_ARANDU["texto_principal"],
        line_width=1.4,
    )

    if show_asc:
        fig.add_trace(
            go.Scatter(
                x=[x_eval],
                y=[res_asc["y_interpolado"]],
                mode="markers",
                name="P_asc(x*)",
                marker=dict(size=13, color=PALETA_ARANDU["salvia"], symbol="star", line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.5)),
            )
        )
    if show_desc:
        fig.add_trace(
            go.Scatter(
                x=[x_eval],
                y=[res_desc["y_interpolado"]],
                mode="markers",
                name="P_desc(x*)",
                marker=dict(size=12, color=PALETA_ARANDU["oro"], symbol="diamond", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.5)),
            )
        )

    render_plotly_arandu(fig, key="plot_int_ng")


def _aplicar_regeneracion_malla_eq(
    prefijo: str,
    slider_xe_key: str,
    xs: list[float],
    ys: list[float],
    nuevo_x0: float,
    nuevo_h: float,
    preset: dict,
) -> None:
    """Regenera nodos equidistantes con el x0 y paso h indicados sin perder coherencia."""
    if abs(nuevo_x0 - xs[0]) < 1e-9 and preset.get("exacta") is None:
        x_new, y_new, _ = regenerar_tabla_interpolacion(xs, ys, nuevo_h=nuevo_h)
    else:
        x_new = [round(nuevo_x0 + i * nuevo_h, 6) for i in range(len(xs))]
        if preset.get("exacta") is not None:
            y_new = [round(float(v), 6) for v in preset["exacta"](np.array(x_new))]
        else:
            y_new = [round(float(interpolar_lagrange(xs, ys, xv)["y_interpolado"]), 6) for xv in x_new]
    st.session_state[f"{prefijo}_x"] = x_new
    st.session_state[f"{prefijo}_y"] = y_new
    st.session_state.pop(f"{prefijo}__editor_df", None)
    st.session_state.pop(slider_xe_key, None)


def _aplicar_regeneracion_malla_ng(xs: list[float], ys: list[float], nuevo_x0: float, nuevo_h: float, preset: dict) -> None:
    """Regenera nodos equidistantes en el panel de Newton-Gregory."""
    _aplicar_regeneracion_malla_eq("int_ng", "ui_ng_xe_slider", xs, ys, nuevo_x0, nuevo_h, preset)


@st.fragment
def _frag_newton_gregory(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="int_ng",
        defaults_dict=defaults,
        mensaje_edicion=r"Edita nodos directamente, agrega/quita puntos o cambia $x_0$ y $\Delta h$; si rompes la equidistancia se activa Lagrange.",
    )
    solo_lectura = modo == "catedra"

    if "ui_ng_preset" not in st.session_state:
        st.session_state["int_ng_preset"] = "tp5_ej1_asc"
    preset_id = st.selectbox(
        "Ejercicio oficial de la cátedra / Caso de estudio:",
        options=list(_PRESETS_NEWTON_GREGORY.keys()),
        format_func=lambda k: _PRESETS_NEWTON_GREGORY[k]["titulo"],
        key="ui_ng_preset",
        on_change=_al_cambiar_preset_ng,
    )
    preset = _PRESETS_NEWTON_GREGORY[preset_id]
    render_practica_catedra_info(
        titulo_practica=preset["titulo"],
        enunciado_practica=preset["enunciado"],
        solucion_referencia=preset["referencia"],
    )

    col_ctrl, col_tbl = st.columns([1.45, 1.15])
    with col_tbl:
        st.markdown("#### Nodos de la Tabla")
        xs, ys = _editor_tabla_xy("int_ng", solo_lectura)
        diag_prev = diagnosticar_espaciamiento_nodos(xs)
        if not solo_lectura:
            h_base = float(diag_prev["h_nominal"]) if diag_prev["es_equiespaciado"] and diag_prev["h_nominal"] else 0.3
            st.session_state.setdefault("ui_ng_x0", float(xs[0]))
            st.session_state.setdefault("ui_ng_dh", float(np.clip(h_base, 0.05, 10.0)))
            st.session_state["ui_ng_dh"] = float(np.clip(float(st.session_state["ui_ng_dh"]), 0.05, 10.0))
            c_m1, c_m2 = st.columns(2)
            nuevo_x0 = c_m1.number_input("Inicio x₀:", step=0.1, format="%.4f", key="ui_ng_x0")
            nuevo_h = c_m2.number_input(
                r"Paso $\Delta h$:",
                min_value=0.05,
                max_value=10.0,
                step=0.05,
                format="%.4f",
                key="ui_ng_dh",
            )
            cb1, cb2, cb3 = st.columns([1.3, 1.0, 1.0])
            if cb1.button("Regenerar malla Δh", key="btn_ng_dh", use_container_width=True):
                _aplicar_regeneracion_malla_ng(xs, ys, float(nuevo_x0), float(nuevo_h), preset)
                rerun_fragmento_seguro()
            if cb2.button("+ Nodo", key="btn_ng_add_node", disabled=len(xs) >= 10, use_container_width=True):
                x_next = round(xs[-1] + h_base, 6)
                if preset.get("exacta") is not None:
                    y_next = round(float(preset["exacta"](np.array([x_next]))[0]), 6)
                else:
                    grado_fit = min(2, len(xs) - 1)
                    coef = np.polyfit(xs[-grado_fit - 1 :], ys[-grado_fit - 1 :], grado_fit)
                    y_next = round(float(np.polyval(coef, x_next)), 6)
                st.session_state["int_ng_x"] = list(xs) + [x_next]
                st.session_state["int_ng_y"] = list(ys) + [y_next]
                st.session_state.pop("int_ng__editor_df", None)
                rerun_fragmento_seguro()
            if cb3.button("- Nodo", key="btn_ng_rem_node", disabled=len(xs) <= 2, use_container_width=True):
                st.session_state["int_ng_x"] = list(xs[:-1])
                st.session_state["int_ng_y"] = list(ys[:-1])
                st.session_state.pop("int_ng__editor_df", None)
                rerun_fragmento_seguro()

    with col_ctrl:
        st.markdown("#### Punto Objetivo $x^*$, Grado y Enfoque")
        render_subtitulo_controles_vivo()
        x_min, x_max = min(xs), max(xs)
        span = max(abs(x_max - x_min), 1.0)
        pad = round(max(0.35, 0.15 * span), 4)
        s_min, s_max = round(x_min - pad, 4), round(x_max + pad, 4)
        step_s = max(round(span / 160.0, 4), 0.001)

        if "ui_ng_xe" not in st.session_state:
            st.session_state["ui_ng_xe"] = float(st.session_state["int_ng_x_eval"])
        if "ui_ng_xe_slider" not in st.session_state:
            st.session_state["ui_ng_xe_slider"] = float(np.clip(st.session_state["int_ng_x_eval"], s_min, s_max))
        else:
            st.session_state["ui_ng_xe_slider"] = float(np.clip(st.session_state["ui_ng_xe_slider"], s_min, s_max))

        cx1, cx2 = st.columns([2.0, 1.0])
        cx1.slider(
            "Deslizar x* (incluye extrapolación leve):",
            min_value=float(s_min),
            max_value=float(s_max),
            step=float(step_s),
            key="ui_ng_xe_slider",
            on_change=_al_cambiar_slider_xe_ng,
        )
        cx2.number_input(
            "Valor exacto x*:",
            format="%.4f",
            key="ui_ng_xe",
            on_change=_al_cambiar_input_xe_ng,
        )
        x_eval = float(st.session_state["int_ng_x_eval"])

        grado_max = max(1, len(xs) - 1)
        if grado_max > 1:
            if "vis_ng_k" not in st.session_state:
                st.session_state["vis_ng_k"] = grado_max
            else:
                st.session_state["vis_ng_k"] = int(np.clip(int(st.session_state["vis_ng_k"]), 1, grado_max))
            k_vis = st.slider(
                "Grado polinómico de truncamiento (k):",
                min_value=1,
                max_value=grado_max,
                key="vis_ng_k",
            )
        else:
            k_vis = 1
            st.caption("Con 2 nodos el máximo grado disponible es k = 1.")

        modo_visual = st.radio(
            "Enfoque visual:",
            options=list(_MODOS_VISUALES_NG),
            horizontal=True,
            key="ui_ng_var",
        )

    st.markdown("### Bloque 3 — Diagnóstico de Equidistancia y Comparación Visual")
    diag = diagnosticar_espaciamiento_nodos(xs)
    if not diag["es_equiespaciado"]:
        st.warning(f"**{diag['mensaje']}** Se calcula automáticamente con **Lagrange** como respaldo válido para espaciado irregular.")
        try:
            res_lag = interpolar_lagrange(xs, ys, x_eval)
            st.metric("P(x*) por Lagrange (Respaldo válido)", f"{res_lag['y_interpolado']:.6f}")
            x_grid = np.linspace(min(min(xs), x_eval) - 0.1, max(max(xs), x_eval) + 0.1, 220)
            y_lag_grid = limpiar_valores_plotly(evaluar_curva_lagrange(xs, ys, x_grid), limite_abs=1e9)
            fig = crear_figura_base_plotly("Nodos No Equidistantes → Respaldo Polinomio de Lagrange", "x", "y")
            fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_lag_grid, mode="lines", name="Lagrange P(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.5)))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos editados", marker=dict(size=9, color=PALETA_ARANDU["terracota"])))
            fig.add_trace(
                go.Scatter(
                    x=[x_eval],
                    y=[res_lag["y_interpolado"]],
                    mode="markers",
                    name="P(x*)",
                    marker=dict(size=13, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
                )
            )
            render_plotly_arandu(fig, key="plot_int_ng_fallback")
        except ValueError as exc:
            render_excepcion_validacion(
                exc,
                campo="Tabla de nodos (respaldo Lagrange)",
                sugerencia_defecto="Eliminá nodos con valores x repetidos e ingresá al menos 2 filas finitas.",
                ejemplo_defecto="x = [0.0, 0.25, 0.5, 0.75]",
            )
        return

    st.success(diag["mensaje"])
    st.markdown(_construir_html_banner_recomendacion_ng(xs, x_eval, k_vis, grado_max), unsafe_allow_html=True)

    res_asc = interpolar_newton_gregory_ascendente(xs, ys, x_eval, grado_max=k_vis)
    res_desc = interpolar_newton_gregory_descendente(xs, ys, x_eval, grado_max=k_vis)
    st.markdown(
        _construir_html_resumen_gregory(res_asc, res_desc, x_eval, k_vis, grado_max, preset.get("exacta")),
        unsafe_allow_html=True,
    )
    _render_grafica_gregory_comparativa(
        xs=xs,
        ys=ys,
        x_eval=x_eval,
        k_vis=k_vis,
        grado_max=grado_max,
        modo_visual=modo_visual,
        res_asc=res_asc,
        res_desc=res_desc,
        exacta_fn=preset.get("exacta"),
        exacta_tex=preset.get("exacta_tex"),
    )

    st.markdown("### Bloque 4 — Tabla Escalonada de Diferencias Finitas")
    st.caption(
        "Cada celda superior es el extremo derecho menos el izquierdo (Δᵏyᵢ = Δᵏ⁻¹yᵢ₊₁ - Δᵏ⁻¹yᵢ). "
        "Salvia marca la diagonal Ascendente (Δᵏy₀); Oro marca la diagonal Descendente (∇ᵏyₙ). "
        "Las columnas atenuadas superan el grado truncado k."
    )
    tabla_arr = calcular_diferencias_ascendentes(ys)
    st.markdown(
        _construir_html_tabla_escalonada_gregory(xs, tabla_arr, k_vis, modo_visual),
        unsafe_allow_html=True,
    )
    _render_inspector_celda_diferencia(xs, tabla_arr)

    st.markdown("### Bloque 5 — Desglose Paso a Paso de Ambos Polinomios")
    col_asc, col_desc = st.columns(2)
    html_asc, tex_asc = _construir_desglose_formula_lado(
        res_asc, "ascendente", k_vis, atenuado=(modo_visual == "Solo Descendente (∇)")
    )
    html_desc, tex_desc = _construir_desglose_formula_lado(
        res_desc, "descendente", k_vis, atenuado=(modo_visual == "Solo Ascendente (Δ)")
    )
    with col_asc:
        st.markdown(html_asc, unsafe_allow_html=True)
        with st.expander("Expresión algebraica sustituida Ascendente (LaTeX)"):
            st.latex(tex_asc)
    with col_desc:
        st.markdown(html_desc, unsafe_allow_html=True)
        with st.expander("Expresión algebraica sustituida Descendente (LaTeX)"):
            st.latex(tex_desc)

    if k_vis < grado_max:
        u_val = float(res_asc["u"])
        v_val = float(res_desc["v"])
        next_asc = float(tabla_arr[0, k_vis + 1])
        next_desc = float(tabla_arr[len(xs) - 1 - (k_vis + 1), k_vis + 1])
        comb_asc = math.prod(u_val - j for j in range(k_vis + 1)) / math.factorial(k_vis + 1)
        comb_desc = math.prod(v_val + j for j in range(k_vis + 1)) / math.factorial(k_vis + 1)
        err_est_asc = abs(comb_asc * next_asc)
        err_est_desc = abs(comb_desc * next_desc)
        st.info(
            f"Estimación de residuo por primer término omitido (orden k+1 = {k_vis + 1}): "
            f"|R_asc| ≈ {err_est_asc:.6f} vs |R_desc| ≈ {err_est_desc:.6f}."
        )


def _tab_newton_gregory() -> None:
    meta_asc = render_bloque_1_teoria_y_ejemplo("newton_gregory_asc")
    _render_formulas_completas_gregory()
    practica = meta_asc.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "x": [0.2, 0.5, 0.8, 1.1, 1.4],
            "y": [0.258, 2.535, 4.902, 6.722, 10.704],
            "x_eval": 0.35,
        },
    )
    st.divider()
    _frag_newton_gregory(defaults)


_PRESETS_GAUSS: dict[str, dict] = {
    "tp5_gauss_centro": {
        "titulo": "TP5 Central — Tabla h = 0.3 en x* = 0.90 (Ancla x₂ = 0.8)",
        "enunciado": (
            "Sobre la tabla equiespaciada del TP5 (h = 0.3), centrar la Fórmula de Gauss en el nodo "
            "medio x₂ = 0.8 y comparar el truncamiento k frente a Newton-Gregory Ascendente y Descendente."
        ),
        "referencia": r"Ancla $x_2 = 0.8$, $u = \frac{0.90 - 0.8}{0.3} = \frac{1}{3} \implies P_4(0.90) = 5.50451440$.",
        "x": [0.2, 0.5, 0.8, 1.1, 1.4],
        "y": [0.258, 2.535, 4.902, 6.722, 10.704],
        "x_eval": 0.90,
        "exacta": None,
        "exacta_tex": None,
    },
    "html_sin_lin": {
        "titulo": "f(x) = sen(x) + 0.3x  [0, 1, 2, 3, 4] (Visualizador Cátedra)",
        "enunciado": (
            "Función suave oscilante sobre 5 nodos con paso h = 1. Permite contrastar en vivo el error "
            "de truncamiento (k < 4) en los extremos frente a la zona central [-0.5 <= u <= 0.5]."
        ),
        "referencia": r"En $x^* = 2.35$ ($u = +0.35$ respecto a $x_2 = 2$), Gauss Progresiva logra menor error en $k = 2$.",
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [float(np.sin(x) + 0.3 * x) for x in (0.0, 1.0, 2.0, 3.0, 4.0)],
        "x_eval": 2.35,
        "exacta": lambda arr: np.sin(arr) + 0.3 * arr,
        "exacta_tex": r"f(x) = \operatorname{sen}(x) + 0.3x",
    },
    "html_exp": {
        "titulo": "f(x) = e^(0.6x)  [0, 1, 2, 3, 4] (Visualizador Cátedra)",
        "enunciado": (
            "Crecimiento exponencial sobre 5 nodos uniformes (h = 1). Ideal para observar cómo "
            "Gauss concentra sus diferencias en la banda media de la tabla."
        ),
        "referencia": r"Valor exacto en $x^* = 2.35$: $e^{1.41} \approx 4.095955$.",
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [float(np.exp(0.6 * x)) for x in (0.0, 1.0, 2.0, 3.0, 4.0)],
        "x_eval": 2.35,
        "exacta": lambda arr: np.exp(0.6 * arr),
        "exacta_tex": r"f(x) = e^{0.6x}",
    },
    "html_runge": {
        "titulo": "Fenómeno de Runge: f(x) = 1 / (1 + x²)  [-2, -1, 0, 1, 2]",
        "enunciado": (
            "Curva de campana clásica centrada en x = 0. Demuestra por qué interpolar cerca del "
            "origen anclado en el nodo central supera al borde en grados bajos."
        ),
        "referencia": r"En $x^* = -0.35$ ($u = -0.35$ desde $x_2 = 0$), Gauss Regresiva aprovecha la simetría central.",
        "x": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "y": [float(1.0 / (1.0 + x * x)) for x in (-2.0, -1.0, 0.0, 1.0, 2.0)],
        "x_eval": -0.35,
        "exacta": lambda arr: 1.0 / (1.0 + np.square(arr)),
        "exacta_tex": r"f(x) = \frac{1}{1 + x^2}",
    },
    "html_cubica": {
        "titulo": "Polinomio cúbico: 0.25x³ - 0.5x² + 0.5x + 1  [-2..2]",
        "enunciado": (
            "Muestra la anulación exacta de diferencias finitas a partir de Δ⁴y = 0 y comprueba "
            "que desde k = 3 los cuatro métodos coinciden sin error."
        ),
        "referencia": r"Para $k \ge 3$ los 4 métodos entregan el valor exacto; en $k = 2$ Gauss gana en la región central.",
        "x": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "y": [float(0.25 * (x**3) - 0.5 * (x**2) + 0.5 * x + 1.0) for x in (-2.0, -1.0, 0.0, 1.0, 2.0)],
        "x_eval": 0.35,
        "exacta": lambda arr: 0.25 * np.power(arr, 3) - 0.5 * np.square(arr) + 0.5 * arr + 1.0,
        "exacta_tex": r"f(x) = 0.25x^3 - 0.5x^2 + 0.5x + 1",
    },
}

_OPCIONES_METODO_GAUSS: tuple[str, ...] = (
    "gauss_progresiva",
    "gauss_regresiva",
    "ascendente",
    "descendente",
)

_NOMBRES_METODO_GAUSS: dict[str, str] = {
    "gauss_progresiva": "Gauss Progresiva (central, ↓ primero)",
    "gauss_regresiva": "Gauss Regresiva (central, ↑ primero)",
    "ascendente": "Newton-Gregory Ascendente (extremo x₀)",
    "descendente": "Newton-Gregory Descendente (extremo xₙ)",
}


def _sincronizar_xe_gauss(valor: float, xs: list[float]) -> None:
    """Sincroniza x* entre session_state, input numérico y slider de Gauss."""
    val = float(valor)
    st.session_state["int_gauss_x_eval"] = val
    st.session_state["ui_gauss_xe"] = val
    x_min, x_max = min(xs), max(xs)
    span = max(abs(x_max - x_min), 1.0)
    pad = round(max(0.35, 0.15 * span), 4)
    st.session_state["ui_gauss_xe_slider"] = float(np.clip(val, x_min - pad, x_max + pad))


def _al_cambiar_preset_gauss() -> None:
    preset_id = st.session_state.get("ui_gauss_preset", "tp5_gauss_centro")
    preset = _PRESETS_GAUSS[preset_id]
    st.session_state["int_gauss_preset"] = preset_id
    st.session_state["int_gauss_x"] = list(preset["x"])
    st.session_state["int_gauss_y"] = list(preset["y"])
    st.session_state["int_gauss_x_eval"] = float(preset["x_eval"])
    st.session_state["vis_gauss_k"] = min(2, max(len(preset["x"]) - 1, 0))
    for clave in (
        "int_gauss__editor_df",
        "ui_gauss_xe",
        "ui_gauss_xe_slider",
        "ui_gauss_dh",
        "ui_gauss_x0",
    ):
        st.session_state.pop(clave, None)


def _al_cambiar_slider_xe_gauss() -> None:
    val = float(st.session_state["ui_gauss_xe_slider"])
    st.session_state["int_gauss_x_eval"] = val
    st.session_state["ui_gauss_xe"] = val


def _al_cambiar_input_xe_gauss() -> None:
    xs = st.session_state.get("int_gauss_x", [0.2, 1.4])
    _sincronizar_xe_gauss(float(st.session_state["ui_gauss_xe"]), xs)


def _al_aplicar_optimo_gauss(modo_opt: str) -> None:
    st.session_state["ui_gauss_metodo"] = modo_opt


def _render_formulas_completas_gauss() -> None:
    """Muestra las fórmulas oficiales de la Cátedra (Hoja 2026) para Gauss Progresiva y Regresiva."""
    with st.expander(
        "Fórmulas Oficiales Completas: Fórmula de Gauss Central (Hoja de Cátedra 2026)",
        expanded=True,
    ):
        st.markdown("**1. Cambio de Variable Centrado en el Nodo Central $x_0^{\\mathrm{c}}$:**")
        st.latex(
            r"u = \frac{x - x_0}{h} \quad (-0.5 \le u \le 0.5 \text{ en el intervalo central}), \qquad "
            r"\Delta^k f(x_i) = \Delta^{k-1} f(x_{i+1}) - \Delta^{k-1} f(x_i)"
        )
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"**2A. Gauss Progresiva (primer paso abajo $\Delta f(x_0)$, $0 \le u \le 0.5$)**")
            st.latex(
                r"P_n(x) = f(x_0) + \Delta f(x_0)\, u + \frac{\Delta^2 f(x_0 - h)}{2!}(u - 1)u "
                r"+ \frac{\Delta^3 f(x_0 - h)}{3!}(u - 1)u(u + 1)"
            )
            st.latex(
                r"+ \frac{\Delta^4 f(x_0 - 2h)}{4!}(u - 2)(u - 1)u(u + 1) "
                r"+ \frac{\Delta^5 f(x_0 - 2h)}{5!}(u - 2)(u - 1)u(u + 1)(u + 2) + \dots"
            )
        with col_g2:
            st.markdown(r"**2B. Gauss Regresiva (primer paso arriba $\Delta f(x_0 - h)$, $-0.5 \le u < 0$)**")
            st.latex(
                r"P_n^{\mathrm{reg}}(x) = f(x_0) + \Delta f(x_0 - h)\, u + \frac{\Delta^2 f(x_0 - h)}{2!}u(u + 1) "
                r"+ \frac{\Delta^3 f(x_0 - 2h)}{3!}(u - 1)u(u + 1)"
            )
            st.latex(
                r"+ \frac{\Delta^4 f(x_0 - 2h)}{4!}(u + 2)(u - 1)u(u + 1) "
                r"+ \frac{\Delta^5 f(x_0 - 3h)}{5!}(u - 2)(u + 2)(u - 1)u(u + 1) + \dots"
            )
        st.caption(
            "Regla de oro para exámenes: extremo inicial -> Newton-Gregory Ascendente; extremo final -> "
            "Newton-Gregory Descendente; zona central -> Fórmula de Gauss (Progresiva si u >= 0, Regresiva si u < 0)."
        )


def _construir_html_banner_recomendacion_gauss(
    rec: dict,
    modo_activo: str,
    k_vis: int,
    grado_max: int,
) -> str:
    """Genera el banner de recomendación óptima según x* y estado del truncamiento."""
    meta_rec = _META_MODOS_EQ[rec["modo"]]
    clase = f"arandu-ng-banner-{meta_rec['suf']}"
    coincide = (
        "Método óptimo activo en la tabla."
        if modo_activo == rec["modo"]
        else f"Actualmente visualizando {_NOMBRES_METODO_GAUSS[modo_activo]}."
    )
    nota_k = (
        f"En grado completo k = {grado_max}, los 4 métodos coinciden."
        if k_vis >= grado_max
        else f"Truncamiento k = {k_vis} (< {grado_max}): el zig-zag central reduce el error en el medio."
    )
    return (
        f"<div class='arandu-ng-banner {clase}'>"
        f"<strong>Óptimo sugerido: {html.escape(_NOMBRES_METODO_GAUSS[rec['modo']])}.</strong> "
        f"{html.escape(rec['motivo'])} "
        f"<span class='arandu-ng-pill-muted'>{html.escape(coincide)} {html.escape(nota_k)}</span>"
        "</div>"
    )


def _construir_html_tabla_gauss_zigzag(
    xs: list[float],
    tabla: np.ndarray,
    k_vis: int,
    modo_activo: str,
    res_activo: dict,
) -> tuple[str, str]:
    """Renderiza la matriz escalonada de diferencias finitas destacando el trazado zig-zag de Gauss."""
    n = len(xs)
    meta = _META_MODOS_EQ[modo_activo]
    anchor_gauss = (n - 1) // 2 if modo_activo in ("ascendente", "descendente") else int(res_activo["anchor_idx"])
    rutas: dict[tuple[int, int], tuple[str, str, str]] = {}
    pasos_tira: list[str] = []

    if modo_activo in ("gauss_progresiva", "gauss_regresiva"):
        celdas_ruta = res_activo["ruta_celdas"]
        etq_corta = "Gp" if modo_activo == "gauss_progresiva" else "Gr"
        for item in celdas_ruta:
            k = int(item["k"])
            fila = int(item["fila"])
            if k == 0:
                etq_badge = "T0: Ancla"
                flecha = "●"
            elif modo_activo == "gauss_progresiva":
                flecha = "↴" if k % 2 == 1 else "↱"
                etq_badge = f"T{k} {flecha} {etq_corta}"
            else:
                flecha = "↰" if k % 2 == 1 else "↳"
                etq_badge = f"T{k} {flecha} {etq_corta}"
            rutas[(fila, k)] = (
                meta["cell_cls"],
                meta["badge_cls"],
                f"<span class='arandu-sr-only'>{meta['titulo']} paso {k}: </span>{etq_badge}",
            )
            if k <= k_vis:
                sym = f"y{_subindice(fila)}" if k == 0 else f"Δ{_superindice(k)}y{_subindice(fila)}"
                pasos_tira.append(f"{flecha} T{k}: {sym} = {float(item['valor']):.4f}")
    else:
        for k in range(n):
            fila = 0 if modo_activo == "ascendente" else (n - 1 - k)
            flecha = "↘" if modo_activo == "ascendente" else "↗"
            sim_op = "Δ" if modo_activo == "ascendente" else "∇"
            etq_badge = f"T{k} {flecha} {sim_op}{_superindice(k)}" if k > 0 else "T0: Ancla"
            rutas[(fila, k)] = (
                meta["cell_cls"],
                meta["badge_cls"],
                f"<span class='arandu-sr-only'>{meta['titulo']} paso {k}: </span>{etq_badge}",
            )
            if k <= k_vis:
                sym = (
                    f"{'y₀' if modo_activo == 'ascendente' else 'yₙ'}"
                    if k == 0
                    else f"{sim_op}{_superindice(k)}{'y₀' if modo_activo == 'ascendente' else 'yₙ'}"
                )
                pasos_tira.append(f"{flecha} T{k}: {sym} = {float(tabla[fila, k]):.4f}")

    tabla_html = _construir_html_tabla_diferencias_comun(
        xs,
        tabla,
        k_vis,
        rutas=rutas,
        anchor_gauss=anchor_gauss,
        aria_caption=f"Tabla escalonada de diferencias finitas mostrando ruta de {_NOMBRES_METODO_GAUSS[modo_activo]}.",
    )
    tira_html = (
        "<div class='arandu-gauss-path-strip'>"
        f"<strong>Trazado activo hasta grado k = {k_vis}:</strong> "
        + html.escape("  ->  ".join(pasos_tira))
        + "</div>"
    )
    return tabla_html, tira_html


def _construir_html_comparativa_4metodos(
    resultados: dict[str, dict],
    k_vis: int,
    grado_max: int,
    modo_activo: str,
    modo_optimo: str,
    y_ref: float | None,
    ref_es_exacta: bool,
) -> str:
    """Construye la tabla comparativa simultánea de los 4 métodos en el grado actual k."""
    col_err = "Error Abs. |f(x*) - P_k|" if ref_es_exacta else "Desvío Truncamiento |Pₙ₋₁ - P_k|"
    metricas: dict[str, tuple[float, float | None]] = {}
    errores_validos: list[float] = []
    for clave in _OPCIONES_METODO_GAUSS:
        r = resultados[clave]
        val_k = float(r["terminos"][k_vis]["acumulado"])
        err = abs(y_ref - val_k) if y_ref is not None else None
        metricas[clave] = (val_k, err)
        if err is not None:
            errores_validos.append(err)

    min_err = min(errores_validos) if errores_validos else None
    empate_total = k_vis >= grado_max or (
        min_err is not None and errores_validos and (max(errores_validos) - min_err) < 1e-10
    )

    filas = []
    for clave in _OPCIONES_METODO_GAUSS:
        r = resultados[clave]
        val_k, err = metricas[clave]
        if clave == "ascendente":
            ancla_txt = f"x₀ = {r['x0']:g}"
            param_txt = f"u = {r['u']:+.4f}"
        elif clave == "descendente":
            ancla_txt = f"xₙ = {r['xn']:g}"
            param_txt = f"v = {r['v']:+.4f}"
        else:
            ancla_txt = f"x{r['anchor_idx']} = {r['x0_central']:g}"
            param_txt = f"u = {r['u']:+.4f}"

        etiquetas = []
        if clave == modo_activo:
            etiquetas.append("Activo")
        if clave == modo_optimo:
            etiquetas.append("Óptimo x*")
        if not empate_total and err is not None and min_err is not None and abs(err - min_err) <= 1e-12:
            etiquetas.append("Menor error")

        badges = "".join(f"<span class='arandu-gauss-comp-best'>{html.escape(e)}</span>" for e in etiquetas)
        err_str = f"{err:.6f}" if err is not None else "-"
        row_cls = " class='arandu-gauss-comp-row-active'" if clave == modo_activo else ""
        filas.append(
            f"<tr{row_cls}>"
            f"<td>{html.escape(_NOMBRES_METODO_GAUSS[clave])}{badges}</td>"
            f"<td>{html.escape(ancla_txt)}</td>"
            f"<td>{html.escape(param_txt)}</td>"
            f"<td>{val_k:.6f}</td>"
            f"<td>{err_str}</td>"
            "</tr>"
        )

    return (
        "<div class='arandu-gauss-comp-wrap' role='region' "
        "aria-label='Comparativa simultánea de los 4 métodos sobre diferencias finitas' tabindex='0'>"
        "<table class='arandu-gauss-comp-table'>"
        "<caption class='arandu-sr-only'>Comparación numérica simultánea entre Newton-Gregory Ascendente, Descendente, Gauss Progresiva y Gauss Regresiva.</caption>"
        f"<thead><tr><th>Método</th><th>Ancla</th><th>Parámetro</th><th>P_{k_vis}(x*)</th><th>{html.escape(col_err)}</th></tr></thead>"
        f"<tbody>{''.join(filas)}</tbody>"
        "</table></div>"
    )


def _render_grafica_gauss_comparativa(
    xs: list[float],
    ys: list[float],
    x_eval: float,
    k_vis: int,
    modo_activo: str,
    resultados: dict[str, dict],
    exacta_fn,
    ver_superposicion: bool,
) -> None:
    """Renderiza el gráfico Plotly interactivo destacando nodos usados vs omitidos y los 4 métodos."""
    n = len(xs)
    pad = max((max(xs) - min(xs)) * 0.15, 0.5)
    x_min = min(min(xs), x_eval) - pad * 0.4
    x_max = max(max(xs), x_eval) + pad * 0.4
    x_grid = np.linspace(x_min, x_max, 240)

    fig = crear_figura_base_plotly(
        f"Curva de Interpolación (k = {k_vis}) — Ruta {_NOMBRES_METODO_GAUSS[modo_activo]}",
        "x",
        "P(x)",
        altura=440,
    )

    if exacta_fn is not None:
        try:
            with np.errstate(all="ignore"):
                y_ex = limpiar_valores_plotly(np.asarray(exacta_fn(x_grid), dtype=float), limite_abs=1e9)
            fig.add_trace(
                go.Scatter(
                    x=x_grid.tolist(),
                    y=y_ex,
                    mode="lines",
                    name="Función exacta f(x)",
                    line=dict(color=PALETA_ARANDU["texto_atenuado"], width=2.0, dash="dot"),
                )
            )
        except Exception:
            pass

    estilos = {
        "gauss_progresiva": (PALETA_ARANDU["terracota"], "solid"),
        "gauss_regresiva": (PALETA_ARANDU["texto_principal"], "dashdot"),
        "ascendente": (PALETA_ARANDU["salvia"], "dash"),
        "descendente": (PALETA_ARANDU["oro"], "dot"),
    }

    modos_trazar = list(_OPCIONES_METODO_GAUSS) if ver_superposicion else [modo_activo]
    for m_cur in modos_trazar:
        if m_cur == modo_activo:
            continue
        col_m, dash_m = estilos[m_cur]
        yg_m = limpiar_valores_plotly(
            evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=k_vis, modo=m_cur),
            limite_abs=1e9,
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=yg_m,
                mode="lines",
                name=f"{_NOMBRES_METODO_GAUSS[m_cur]} (k={k_vis})",
                line=dict(color=col_m, width=1.9, dash=dash_m),
                opacity=0.72,
            )
        )

    col_act, _ = estilos[modo_activo]
    yg_act = limpiar_valores_plotly(
        evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=k_vis, modo=modo_activo),
        limite_abs=1e9,
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid.tolist(),
            y=yg_act,
            mode="lines",
            name=f"Activo: {_NOMBRES_METODO_GAUSS[modo_activo]} (k={k_vis})",
            line=dict(color=col_act, width=3.4, dash="solid"),
        )
    )

    usados = set(nodos_usados_equidistante(n, k_vis, modo_activo))
    xs_usados = [xs[i] for i in range(n) if i in usados]
    ys_usados = [ys[i] for i in range(n) if i in usados]
    xs_omitidos = [xs[i] for i in range(n) if i not in usados]
    ys_omitidos = [ys[i] for i in range(n) if i not in usados]

    if xs_omitidos:
        fig.add_trace(
            go.Scatter(
                x=xs_omitidos,
                y=ys_omitidos,
                mode="markers",
                name="Nodos no usados en k actual",
                marker=dict(
                    size=9,
                    symbol="circle-open",
                    color=PALETA_ARANDU["texto_atenuado"],
                    line=dict(color=PALETA_ARANDU["texto_atenuado"], width=2),
                ),
            )
        )
    if xs_usados:
        fig.add_trace(
            go.Scatter(
                x=xs_usados,
                y=ys_usados,
                mode="markers",
                name=f"Nodos usados por P_{k_vis}",
                marker=dict(
                    size=11,
                    color=PALETA_ARANDU["oro"],
                    line=dict(color=PALETA_ARANDU["bg_lienzo"], width=1.6),
                ),
            )
        )

    idx_c = (n - 1) // 2 if modo_activo in ("ascendente", "descendente") else int(resultados[modo_activo]["anchor_idx"])
    fig.add_trace(
        go.Scatter(
            x=[xs[idx_c]],
            y=[ys[idx_c]],
            mode="markers",
            name="Ancla central Gauss",
            marker=dict(
                size=13,
                symbol="diamond",
                color=PALETA_ARANDU["terracota"],
                line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.8),
            ),
        )
    )

    fig.add_vline(x=x_eval, line_dash="dash", line_color=PALETA_ARANDU["texto_principal"], line_width=1.4)
    val_eval = float(resultados[modo_activo]["terminos"][k_vis]["acumulado"])
    fig.add_trace(
        go.Scatter(
            x=[x_eval],
            y=[val_eval],
            mode="markers",
            name=f"P_{k_vis}(x*) = {val_eval:.5f}",
            marker=dict(
                size=14,
                symbol="star",
                color=col_act,
                line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.5),
            ),
        )
    )

    if exacta_fn is not None:
        try:
            y_ex_pt = float(exacta_fn(np.array([x_eval], dtype=float))[0])
            if np.isfinite(y_ex_pt):
                fig.add_trace(
                    go.Scatter(
                        x=[x_eval],
                        y=[y_ex_pt],
                        mode="markers",
                        name=f"Valor real f(x*) = {y_ex_pt:.5f}",
                        marker=dict(
                            size=10,
                            symbol="cross",
                            color=PALETA_ARANDU["salvia"],
                            line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.2),
                        ),
                    )
                )
        except Exception:
            pass

    render_plotly_arandu(fig, key="plot_int_gauss")


@st.fragment
def _frag_formula_gauss(defaults: dict) -> None:
    preset_id = st.session_state.get("int_gauss_preset", "tp5_gauss_centro")
    if preset_id not in _PRESETS_GAUSS:
        preset_id = "tp5_gauss_centro"
        st.session_state["int_gauss_preset"] = preset_id
    preset = _PRESETS_GAUSS[preset_id]

    if "int_gauss_x" not in st.session_state:
        st.session_state["int_gauss_x"] = list(defaults.get("x", preset["x"]))
    if "int_gauss_y" not in st.session_state:
        st.session_state["int_gauss_y"] = list(defaults.get("y", preset["y"]))
    if "int_gauss_x_eval" not in st.session_state:
        st.session_state["int_gauss_x_eval"] = float(defaults.get("x_eval", preset["x_eval"]))
    if "ui_gauss_metodo" not in st.session_state:
        st.session_state["ui_gauss_metodo"] = "gauss_progresiva"

    render_subtitulo_controles_vivo()

    c_pre, c_met = st.columns([1.2, 1.3])
    c_pre.selectbox(
        "Escenario / Función de prueba:",
        list(_PRESETS_GAUSS.keys()),
        index=list(_PRESETS_GAUSS.keys()).index(preset_id),
        format_func=lambda k: _PRESETS_GAUSS[k]["titulo"],
        key="ui_gauss_preset",
        on_change=_al_cambiar_preset_gauss,
    )
    modo_activo = c_met.selectbox(
        "Método activo para trazar en tabla y gráfica:",
        list(_OPCIONES_METODO_GAUSS),
        format_func=lambda k: _NOMBRES_METODO_GAUSS[k],
        key="ui_gauss_metodo",
    )

    xs_estado = list(st.session_state["int_gauss_x"])
    ys_estado = list(st.session_state["int_gauss_y"])
    defaults_escenario = {
        "x": list(preset["x"]),
        "y": list(preset["y"]),
        "x_eval": float(preset["x_eval"]),
    }

    modo_edicion = render_controles_edicion_practica(
        clave_state="int_gauss",
        defaults_dict=defaults_escenario,
        mensaje_edicion=(
            "Mové x* en vivo, ajustá el grado de truncamiento k o editá la tabla manteniendo paso h uniforme."
        ),
    )
    xs_estado = list(st.session_state["int_gauss_x"])
    ys_estado = list(st.session_state["int_gauss_y"])
    x_eval_ini = float(st.session_state["int_gauss_x_eval"])
    if "ui_gauss_xe" in st.session_state and abs(float(st.session_state["ui_gauss_xe"]) - x_eval_ini) > 1e-9:
        _sincronizar_xe_gauss(x_eval_ini, xs_estado)

    col_cfg, col_main = st.columns([1.05, 1.55])

    with col_cfg:
        if modo_edicion == "manual":
            with st.container(border=True):
                st.caption("Modificá el paso Δh manteniendo equidistancia o editá nodos en la tabla:")
                n_cur = len(xs_estado)
                h_cur = (xs_estado[1] - xs_estado[0]) if n_cur >= 2 else 0.3
                cx0, cdh, cbtn = st.columns([1, 1, 1])
                nuevo_x0 = cx0.number_input("x₀ inicial", value=float(xs_estado[0]), format="%.4f", key="ui_gauss_x0")
                nuevo_h = cdh.number_input("Paso Δh", min_value=0.05, value=float(max(abs(h_cur), 0.05)), step=0.1, format="%.4f", key="ui_gauss_dh")
                if cbtn.button("Regenerar malla", key="btn_gauss_dh", use_container_width=True):
                    _aplicar_regeneracion_malla_eq(
                        "int_gauss",
                        "ui_gauss_xe_slider",
                        xs_estado,
                        ys_estado,
                        float(nuevo_x0),
                        float(nuevo_h),
                        preset,
                    )
                    rerun_fragmento_seguro()

        xs, ys = _editor_tabla_xy("int_gauss", solo_lectura=(modo_edicion == "catedra"))

    if len(xs) < 2:
        render_excepcion_validacion(
            ValueError("Se necesitan al menos 2 nodos (xᵢ, yᵢ)."),
            campo="Tabla de nodos Gauss",
            sugerencia_defecto="Ingresá entre 3 y 7 nodos con paso h uniforme.",
            ejemplo_defecto="x = [0.2, 0.5, 0.8, 1.1, 1.4]",
        )
        return

    diag = diagnosticar_espaciamiento_nodos(xs)
    if not diag["es_equiespaciado"]:
        render_excepcion_validacion(
            ValueError(f"Malla no equidistante: {diag['mensaje']}"),
            campo="Columna x de la tabla",
            sugerencia_defecto="Gauss y Newton-Gregory exigen paso h uniforme. Usá Lagrange si los nodos son irregulares o regenerá h.",
            ejemplo_defecto="h = 0.3 constante -> x = [0.2, 0.5, 0.8, 1.1, 1.4]",
        )
        return

    h_val = float(diag["h_nominal"])
    n_pts = len(xs)
    grado_max = n_pts - 1
    x_min, x_max = min(xs), max(xs)
    span = max(abs(x_max - x_min), 1.0)
    pad = round(max(0.35, 0.15 * span), 4)
    s_min, s_max = round(x_min - pad, 4), round(x_max + pad, 4)
    step_s = max(round(span / 160.0, 4), 0.001)

    with col_cfg:
        st.markdown("**Posición de $x^*$ y Grado de Truncamiento $k$**")
        x_eval_cur = float(st.session_state.get("int_gauss_x_eval", float(preset["x_eval"])))
        idx_c_prog = (n_pts - 1) // 2
        idx_c_reg = n_pts // 2

        qb1, qb2 = st.columns(2)
        qb1.button(
            "Inicio (cerca x₀)",
            key="btn_gauss_p_ini",
            use_container_width=True,
            on_click=_sincronizar_xe_gauss,
            args=(round(xs[0] + 0.35 * h_val, 6), xs),
        )
        qb2.button(
            "Centro-Izq (u = -0.35)",
            key="btn_gauss_p_cizq",
            use_container_width=True,
            on_click=_sincronizar_xe_gauss,
            args=(round(xs[idx_c_reg] - 0.35 * h_val, 6), xs),
        )
        qb3, qb4 = st.columns(2)
        qb3.button(
            "Centro-Der (u = +0.35)",
            key="btn_gauss_p_cder",
            use_container_width=True,
            on_click=_sincronizar_xe_gauss,
            args=(round(xs[idx_c_prog] + 0.35 * h_val, 6), xs),
        )
        qb4.button(
            "Final (cerca xₙ)",
            key="btn_gauss_p_fin",
            use_container_width=True,
            on_click=_sincronizar_xe_gauss,
            args=(round(xs[-1] - 0.35 * h_val, 6), xs),
        )

        st.session_state.setdefault("ui_gauss_xe", float(x_eval_cur))
        if "ui_gauss_xe_slider" not in st.session_state:
            st.session_state["ui_gauss_xe_slider"] = float(np.clip(x_eval_cur, s_min, s_max))
        else:
            st.session_state["ui_gauss_xe_slider"] = float(np.clip(st.session_state["ui_gauss_xe_slider"], s_min, s_max))
        st.slider(
            "Deslizar x* a lo largo de la tabla:",
            min_value=float(s_min),
            max_value=float(s_max),
            step=float(step_s),
            key="ui_gauss_xe_slider",
            on_change=_al_cambiar_slider_xe_gauss,
        )
        st.number_input(
            "Valor exacto de x*:",
            format="%.4f",
            key="ui_gauss_xe",
            on_change=_al_cambiar_input_xe_gauss,
        )

        x_eval = float(st.session_state["int_gauss_x_eval"])
        k_default = min(2, grado_max)
        if "vis_gauss_k" in st.session_state:
            st.session_state["vis_gauss_k"] = min(max(int(st.session_state["vis_gauss_k"]), 0), grado_max)
        else:
            st.session_state["vis_gauss_k"] = k_default
        k_vis = int(
            st.slider(
                "Grado de truncamiento k:",
                min_value=0,
                max_value=grado_max,
                key="vis_gauss_k",
            )
        )
        ver_superposicion = st.checkbox(
            "Superponer los 4 métodos truncados en el gráfico",
            value=True,
            key="ui_gauss_overlay",
        )

    rec = recomendar_metodo_equidistante(xs, x_eval)
    c_ban, c_opt = st.columns([2.6, 1.0])
    c_ban.markdown(
        _construir_html_banner_recomendacion_gauss(rec, modo_activo, k_vis, grado_max),
        unsafe_allow_html=True,
    )
    c_opt.button(
        "Aplicar método óptimo según x*",
        key="btn_gauss_optimo",
        use_container_width=True,
        disabled=(modo_activo == rec["modo"]),
        on_click=_al_aplicar_optimo_gauss,
        args=(rec["modo"],),
    )

    resultados = {
        "ascendente": interpolar_newton_gregory_ascendente(xs, ys, x_eval),
        "descendente": interpolar_newton_gregory_descendente(xs, ys, x_eval),
        "gauss_progresiva": interpolar_gauss_progresiva(xs, ys, x_eval),
        "gauss_regresiva": interpolar_gauss_regresiva(xs, ys, x_eval),
    }

    exacta_fn = preset.get("exacta")
    if exacta_fn is not None and modo_edicion != "catedra":
        try:
            ys_check = np.asarray(exacta_fn(np.asarray(xs, dtype=float)), dtype=float)
            if not np.allclose(ys_check, np.asarray(ys, dtype=float), rtol=1e-4, atol=1e-4):
                exacta_fn = None
        except Exception:
            exacta_fn = None

    y_ref: float | None = None
    ref_es_exacta = False
    if exacta_fn is not None:
        try:
            cand = float(exacta_fn(np.array([x_eval], dtype=float))[0])
            if np.isfinite(cand):
                y_ref = cand
                ref_es_exacta = True
        except Exception:
            pass
    if y_ref is None:
        y_ref = float(resultados[modo_activo]["y_interpolado"])

    res_activo = resultados[modo_activo]
    tabla_arr = np.asarray(res_activo["tabla_diferencias"], dtype=float)
    if modo_activo == "descendente":
        tabla_asc_arr = calcular_diferencias_ascendentes(ys)
    else:
        tabla_asc_arr = tabla_arr

    with col_main:
        st.markdown(f"**Comparativa Simultánea en Grado $k = {k_vis}$ (vs. Grado Completo $n - 1 = {grado_max}$)**")
        st.markdown(
            _construir_html_comparativa_4metodos(
                resultados,
                k_vis,
                grado_max,
                modo_activo,
                rec["modo"],
                y_ref,
                ref_es_exacta,
            ),
            unsafe_allow_html=True,
        )
        _render_grafica_gauss_comparativa(
            xs,
            ys,
            x_eval,
            k_vis,
            modo_activo,
            resultados,
            exacta_fn,
            ver_superposicion,
        )

    st.markdown("**Tabla de Diferencias Finitas Escalonada y Trazado en Zig-Zag**")
    tabla_html, tira_html = _construir_html_tabla_gauss_zigzag(
        xs, tabla_asc_arr, k_vis, modo_activo, res_activo
    )
    st.markdown(tabla_html, unsafe_allow_html=True)
    st.markdown(tira_html, unsafe_allow_html=True)
    _render_inspector_celda_diferencia(xs, tabla_asc_arr)

    st.markdown("**Descomposición Término a Término del Polinomio Seleccionado**")
    html_card, tex_pol = _construir_desglose_formula_lado(res_activo, modo_activo, k_vis, atenuado=False)
    st.markdown(html_card, unsafe_allow_html=True)
    with st.expander("Expresión algebraica sustituida del método activo (LaTeX)"):
        st.latex(tex_pol)


def _tab_formula_gauss() -> None:
    meta_g = render_bloque_1_teoria_y_ejemplo("formula_gauss")
    _render_formulas_completas_gauss()
    practica = meta_g.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "x": [0.2, 0.5, 0.8, 1.1, 1.4],
            "y": [0.258, 2.535, 4.902, 6.722, 10.704],
            "x_eval": 0.90,
        },
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_formula_gauss(defaults)


def vista_interpolacion() -> None:
    inject_arandu_css()
    st.title("Interpolación Polinomial")
    st.markdown(
        r"Construcción de polinomios sobre tablas discretas $(x_i, y_i)$ con detección automática "
        r"de pérdida de equidistancia ($\Delta h$) y alerta del fenómeno de Runge."
    )
    activo = render_selector_fichas_metodos(
        "interpolacion",
        (
            {
                "numero": "1",
                "clave_svg": "interp_lineal",
                "titulo": "Lineal (Directa / Inversa)",
                "subtitulo": "Grado 1 · Directa e inversa",
                "resumen": "Une dos puntos consecutivos mediante una recta para evaluar ordenadas o despejar valores x*.",
                "pista": "2 nodos · Directa / Inversa",
            },
            {
                "numero": "2",
                "clave_svg": "cuadratica_inversa",
                "titulo": "Cuadrática Inversa",
                "subtitulo": "Parábola · Inversión x(y)",
                "resumen": "Ajusta una parábola sobre tres nodos cercanos para despejar la abscisa x* dado un y*.",
                "pista": "3 nodos · Inversión x = P(y)",
            },
            {
                "numero": "3",
                "clave_svg": "lagrange",
                "titulo": "Lagrange (Nodos Libres)",
                "subtitulo": "Bases L_i · Nodos libres",
                "resumen": "Combina polinomios base de Lagrange sobre tablas cuyos nodos tienen paso irregular variable.",
                "pista": "Paso libre Δh · Cualquier malla",
            },
            {
                "numero": "4",
                "clave_svg": "newton_gregory",
                "titulo": "Newton-Gregory (Asc / Desc)",
                "subtitulo": "Diferencias · Paso Δh constante",
                "resumen": "Acelera el cálculo usando una tabla de diferencias finitas cuando el paso Δh es uniforme.",
                "pista": "Paso Δh constante · Ascendente / Descendente",
            },
            {
                "numero": "5",
                "clave_svg": "formula_gauss",
                "titulo": "Fórmula de Gauss (Central)",
                "subtitulo": "Diferencias · Paso Δh constante",
                "resumen": "Anclada en el nodo central, traza un zig-zag óptimo para interpolar lejos de los extremos.",
                "pista": "Paso Δh constante · Progresiva / Regresiva",
            },
        ),
    )
    (
        _tab_lineal,
        _tab_cuadratica_inversa,
        _tab_lagrange,
        _tab_newton_gregory,
        _tab_formula_gauss,
    )[activo]()


vista_interpolacion()
