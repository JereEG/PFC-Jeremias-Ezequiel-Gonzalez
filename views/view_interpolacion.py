"""
Vista de Interpolación Polinomial - Arandú.
Interpolación Lineal (Directa/Inversa), Cuadrática Inversa, Polinomio de Lagrange
y Newton-Gregory Ascendente/Descendente con detector de equidistancia en vivo.
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
    interpolar_lagrange,
    interpolar_lineal_directa,
    interpolar_lineal_inversa,
    interpolar_newton_gregory_ascendente,
    interpolar_newton_gregory_descendente,
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
    """Renderiza la matriz escalonada de diferencias finitas (2n - 1 filas) en HTML."""
    n = len(xs)
    total_rows = 2 * n - 1
    show_asc = modo_visual != "Solo Descendente (∇)"
    show_desc = modo_visual != "Solo Ascendente (Δ)"

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
            tag = " (x₀)" if idx == 0 else (" (xₙ)" if idx == n - 1 else "")
            celdas.append(f"<th scope='row' class='arandu-ng-diff-idx'>{idx}</th>")
            celdas.append(f"<td class='arandu-ng-diff-idx'>{xs[idx]:.4g}{tag}</td>")
        else:
            celdas.append("<td class='arandu-ng-diff-empty'></td>")
            celdas.append("<td class='arandu-ng-diff-empty'></td>")

        for k in range(n):
            if r >= k and (r - k) % 2 == 0 and (r - k) // 2 < n - k:
                i = (r - k) // 2
                val = float(tabla[i, k])
                is_asc = i == 0
                is_desc = i == n - 1 - k
                act_asc = is_asc and show_asc
                act_desc = is_desc and show_desc

                clases = ["arandu-ng-cell"]
                if act_asc and act_desc:
                    clases.append("arandu-ng-cell-both")
                elif act_asc:
                    clases.append("arandu-ng-cell-asc")
                elif act_desc:
                    clases.append("arandu-ng-cell-desc")
                if k > k_vis:
                    clases.append("arandu-ng-cell-trunc")

                badge = ""
                if act_asc and act_desc:
                    badge = (
                        f"<span class='arandu-ng-badge-desc'>"
                        f"<span class='arandu-sr-only'>Ruta ascendente y descendente: </span>"
                        f"Δ{_superindice(k)}y₀ = ∇{_superindice(k)}yₙ</span>"
                    )
                elif act_asc:
                    badge = (
                        f"<span class='arandu-ng-badge-asc'>"
                        f"<span class='arandu-sr-only'>Ruta ascendente: </span>"
                        f"Δ{_superindice(k)}y₀</span>"
                    )
                elif act_desc:
                    badge = (
                        f"<span class='arandu-ng-badge-desc'>"
                        f"<span class='arandu-sr-only'>Ruta descendente: </span>"
                        f"∇{_superindice(k)}yₙ</span>"
                    )

                trunc_sr = (
                    f"<span class='arandu-sr-only'> (orden superior al grado actual {k_vis})</span>"
                    if k > k_vis
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
        "aria-label='Tabla escalonada de diferencias finitas de Newton-Gregory' tabindex='0'>"
        "<table class='arandu-ng-diff-table' "
        "aria-label='Diferencias finitas escalonadas de Newton-Gregory'>"
        "<caption class='arandu-sr-only'>"
        "Tabla escalonada de diferencias finitas ascendentes y descendentes de Newton-Gregory."
        "</caption>"
        f"<thead><tr>{''.join(ths)}</tr></thead>"
        f"<tbody>{''.join(filas_html)}</tbody>"
        "</table></div>"
    )


def _construir_desglose_formula_lado(
    res: dict,
    modo: str,
    k_vis: int,
    atenuado: bool,
) -> tuple[str, str]:
    """Construye la tarjeta HTML paso a paso y el bloque LaTeX del polinomio."""
    es_asc = modo == "ascendente"
    suf = "asc" if es_asc else "desc"
    simbolo = "Δ" if es_asc else "∇"
    simbolo_tex = r"\Delta" if es_asc else r"\nabla"
    param_nom = "u" if es_asc else "v"
    param_val = float(res["u"] if es_asc else res["v"])
    ancla_sym = "x₀" if es_asc else "xₙ"
    ancla_val = float(res["x0"] if es_asc else res["xn"])
    terminos_k = res["terminos"][: k_vis + 1]

    filas = []
    partes_tex_u = []
    partes_tex_val = []
    for row in terminos_k:
        k = int(row["k"])
        dif = float(row["diferencia"])
        contrib = float(row["termino"])
        if k == 0:
            etiqueta = f"y{'₀' if es_asc else 'ₙ'}"
            partes_tex_u.append(f"{dif:.5g}")
            partes_tex_val.append(f"{contrib:.6f}")
        elif k == 1:
            etiqueta = f"{param_nom} · {simbolo}y{'₀' if es_asc else 'ₙ'}"
            partes_tex_u.append(rf"\left({param_val:.4f}\right)({dif:.5g})")
            partes_tex_val.append(f"{contrib:+.6f}")
        else:
            factores = []
            for j in range(1, k):
                sig = "-" if es_asc else "+"
                factores.append(f"({param_nom}{sig}{j})")
            etiqueta = f"[{param_nom}{''.join(factores)}/{k}!] · {simbolo}{_superindice(k)}y{'₀' if es_asc else 'ₙ'}"
            fact_tex = "".join(
                rf"\left({param_val:.4f}{'-' if es_asc else '+'}{j}\right)" for j in range(1, k)
            )
            partes_tex_u.append(
                rf"\frac{{{param_val:.4f}{fact_tex}}}{{{math.factorial(k)}}}({dif:.5g})"
            )
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
    titulo = "Polinomio Ascendente (Progresivo)" if es_asc else "Polinomio Descendente (Regresivo)"
    html_card = (
        f"<div class='arandu-ng-formula-card arandu-ng-formula-card-{suf}{clase_mut}'>"
        f"{aviso_atenuado}"
        "<div class='arandu-ng-formula-head'>"
        "<div>"
        f"<div class='arandu-ng-summary-head'>{html.escape(titulo)}</div>"
        f"<div class='arandu-ng-summary-sub'>Anclada en {ancla_sym} = {ancla_val:g} | {param_nom} = {param_val:.4f}</div>"
        "</div>"
        f"<div class='arandu-ng-formula-icon-{suf}' aria-hidden='true'>{simbolo}</div>"
        "</div>"
        f"{''.join(filas)}"
        f"<div class='arandu-ng-step-total-{suf}'>"
        f"<span>P_{k_vis}({res['x_eval']:g}) =</span>"
        f"<span>{res['y_interpolado']:.6f}</span>"
        "</div>"
        "</div>"
    )
    tex_pol = (
        rf"P_{{{k_vis}}}^{{{simbolo_tex}}}({res['x_eval']:g}) = "
        + " + ".join(partes_tex_u)
        + rf" = {' '.join(partes_tex_val)} = {res['y_interpolado']:.6f}"
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


def _aplicar_regeneracion_malla_ng(xs: list[float], ys: list[float], nuevo_x0: float, nuevo_h: float, preset: dict) -> None:
    """Regenera nodos equidistantes con el x0 y paso h indicados sin perder coherencia."""
    if abs(nuevo_x0 - xs[0]) < 1e-9 and preset.get("exacta") is None:
        x_new, y_new, _ = regenerar_tabla_interpolacion(xs, ys, nuevo_h=nuevo_h)
    else:
        x_new = [round(nuevo_x0 + i * nuevo_h, 6) for i in range(len(xs))]
        if preset.get("exacta") is not None:
            y_new = [round(float(v), 6) for v in preset["exacta"](np.array(x_new))]
        else:
            y_new = [round(float(interpolar_lagrange(xs, ys, xv)["y_interpolado"]), 6) for xv in x_new]
    st.session_state["int_ng_x"] = x_new
    st.session_state["int_ng_y"] = y_new
    st.session_state.pop("int_ng__editor_df", None)
    st.session_state.pop("ui_ng_xe_slider", None)


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
        ),
    )
    (
        _tab_lineal,
        _tab_cuadratica_inversa,
        _tab_lagrange,
        _tab_newton_gregory,
    )[activo]()


vista_interpolacion()
