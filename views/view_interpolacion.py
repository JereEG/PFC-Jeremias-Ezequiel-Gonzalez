"""
Vista de Interpolación Polinomial - Arandú.
Interpolación Lineal (Directa/Inversa), Cuadrática Inversa, Polinomio de Lagrange
y Newton-Gregory Ascendente/Descendente con detector de equidistancia en vivo.
Arquitectura Static-First sin dependencias de SymPy en tiempo de renderizado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.interpolacion import (
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
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    limpiar_valores_plotly,
    render_bloque_1_teoria_y_ejemplo,
    render_controles_edicion_practica,
    render_dataframe_acotado,
    render_mini_catalogo_vista,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_slider_paso_k,
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
    df_edit = st.data_editor(
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
        mensaje_edicion=r"✏️ Modifica los pares $(x_i, y_i)$ de la tabla o alterna entre interpolación directa e inversa.",
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

    st.markdown("### ⚙️ Bloque 3 — Resultado Lineal en Vivo")
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
        st.error(str(exc))
        return

    st.markdown("### 📐 Bloque 4 — Recta Secante Local Paso a Paso")
    fig = crear_figura_base_plotly("Interpolación Lineal sobre Segmento Activo", "x", "y")
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="Tabla de puntos",
            line=dict(color="#9E9585", dash="dot"),
            marker=dict(size=8, color="#D4A359"),
        )
    )
    fig.add_trace(go.Scatter(x=[res["x0"], res["x1"]], y=[res["y0"], res["y1"]], mode="lines+markers", name="Segmento secante", line=dict(color="#7A9A65", width=3)))
    fig.add_trace(
        go.Scatter(
            x=[px],
            y=[py],
            mode="markers",
            name="Punto interpolado",
            marker=dict(size=13, color="#FFFDF9", symbol="star", line=dict(color="#B5673F", width=1.8)),
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
        mensaje_edicion=r"✏️ Modifica los puntos de la tabla o la ordenada objetivo $y^*$.",
    )
    solo_lectura = modo == "catedra"
    xs, ys = _editor_tabla_xy("int_cuad", solo_lectura)

    y_busc = st.number_input("Ordenada objetivo $y^*$:", value=float(st.session_state["int_cuad_y_buscado"]), format="%.4f", disabled=solo_lectura, key="ui_cuad_yb")
    if not solo_lectura:
        st.session_state["int_cuad_y_buscado"] = y_busc

    st.markdown("### ⚙️ Bloque 3 — Inversión de Parábola en Vivo")
    try:
        res = interpolar_cuadratica_inversa(xs, ys, y_busc)
    except Exception as exc:
        st.error(str(exc))
        return

    m1, m2 = st.columns(2)
    m1.metric("x estimado (cuadrática inversa)", f"{res['x_resultado']:.6f}")
    m2.metric("Nodos usados", f"({res['x0']}, {res['x1']}, {res['x2']})")

    st.markdown("### 📐 Bloque 4 — Parábola Local vs Recta Horizontal $y = y^*$")
    x_tri = [res["x0"], res["x1"], res["x2"]]
    y_tri = [res["y0"], res["y1"], res["y2"]]
    coefs = np.polyfit(x_tri, y_tri, 2)
    x_grid = np.linspace(min(xs), max(xs), 220)
    with np.errstate(all="ignore"):
        y_parab = limpiar_valores_plotly(np.polyval(coefs, x_grid), limite_abs=1e9)

    fig = crear_figura_base_plotly("Interpolación Cuadrática Inversa", "x", "y")
    fig.add_hline(y=y_busc, line_dash="dash", line_color="#C86D44", annotation_text=f"y* = {y_busc}")
    fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_parab, mode="lines", name="Parábola P₂(x)", line=dict(color="#7A9A65", width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos tabla", marker=dict(size=9, color="#D4A359")))
    fig.add_trace(
        go.Scatter(
            x=[res["x_resultado"]],
            y=[y_busc],
            mode="markers",
            name="Raíz seleccionada",
            marker=dict(size=14, color="#FFFDF9", symbol="star", line=dict(color="#B5673F", width=1.8)),
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
        mensaje_edicion=r"✏️ Modifica nodos libres $(x_i, y_i)$ o prueba remuestrear a $N$ nodos para observar el fenómeno de Runge.",
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
        "⚠️ **Alerta sobre grado alto (Fenómeno de Runge):** Existe un único polinomio de grado $n-1$ que pasa por todos los nodos; "
        "sin embargo, aumentar demasiado $N$ sobre nodos equidistantes introduce oscilaciones pronunciadas cerca de los extremos."
    )

    st.markdown("### ⚙️ Bloque 3 — Bases $L_i(x)$ y Polinomio Combinado")
    try:
        res = interpolar_lagrange(xs, ys, x_eval)
        simb = construir_polinomio_simbolico_lagrange(xs, ys)
    except Exception as exc:
        st.error(str(exc))
        return

    m1, m2 = st.columns(2)
    m1.metric(f"P({x_eval:g}) estimado", f"{res['y_interpolado']:.6f}")
    m2.metric("Grado del polinomio", res["grado"])
    st.latex(rf"P_{{{res['grado']}}}(x) = {simb['latex_expandido']}")
    render_dataframe_acotado(res["terminos_detalle"], max_filas=300, key_prefix="tbl_int_lag")

    st.markdown("### 📐 Bloque 4 — Construcción Geométrica por Suma Parcial de Bases")
    k_vis = render_slider_paso_k("Sumar hasta término $i = k$:", len(xs) - 1, key="vis_lag_k", min_val=0)
    x_grid = np.linspace(min(min(xs), x_eval) - 2, max(max(xs), x_eval) + 2, 220)
    y_full = limpiar_valores_plotly(evaluar_curva_lagrange(xs, ys, x_grid), limite_abs=1e9)

    y_parcial_arr = np.zeros_like(x_grid, dtype=float)
    for idx in range(k_vis + 1):
        y_parcial_arr += float(ys[idx]) * evaluar_curva_lagrange(xs, ys, x_grid, indice_base=idx)
    y_parcial = limpiar_valores_plotly(y_parcial_arr, limite_abs=1e9)

    fig = crear_figura_base_plotly(f"Polinomio de Lagrange — Suma parcial i = 0..{k_vis}", "x", "P(x)")
    fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_full, mode="lines", name="P(x) completo", line=dict(color="#9E9585", dash="dash")))
    fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_parcial, mode="lines", name=f"Suma hasta i={k_vis}", line=dict(color="#7A9A65", width=2.8)))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos", marker=dict(size=9, color="#D4A359")))
    fig.add_trace(
        go.Scatter(
            x=[x_eval],
            y=[res["y_interpolado"]],
            mode="markers",
            name="P(x*)",
            marker=dict(size=13, color="#FFFDF9", symbol="star", line=dict(color="#B5673F", width=1.8)),
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


@st.fragment
def _frag_newton_gregory(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="int_ng",
        defaults_dict=defaults,
        mensaje_edicion=r"✏️ Edita nodos o regenera la malla con otro paso $\Delta h$; si rompes la equidistancia se activa Lagrange como respaldo.",
    )
    solo_lectura = modo == "catedra"
    xs, ys = _editor_tabla_xy("int_ng", solo_lectura)

    c1, c2, c3 = st.columns([1.4, 1.3, 1.5])
    variante = c1.radio("Variante Gregory:", ["Ascendente (inicio)", "Descendente (final)"], horizontal=True, key="ui_ng_var")
    x_eval = c2.number_input("Punto $x^*$:", value=float(st.session_state["int_ng_x_eval"]), format="%.4f", disabled=solo_lectura, key="ui_ng_xe")
    if not solo_lectura:
        st.session_state["int_ng_x_eval"] = x_eval

    if not solo_lectura:
        with c3:
            nuevo_h = st.number_input(r"Cambiar $\Delta h$ en vivo:", value=0.3, min_value=0.05, max_value=1.0, step=0.05, key="ui_ng_dh")
            if st.button("Regenerar malla Δh", key="btn_ng_dh"):
                x_new, y_new, _ = regenerar_tabla_interpolacion(defaults["x"], defaults["y"], nuevo_h=nuevo_h)
                st.session_state["int_ng_x"] = x_new
                st.session_state["int_ng_y"] = y_new
                if "int_ng__editor_df" in st.session_state:
                    del st.session_state["int_ng__editor_df"]
                rerun_fragmento_seguro()

    st.markdown("### ⚙️ Bloque 3 — Diagnóstico de Equidistancia y Tabla de Diferencias")
    diag = diagnosticar_espaciamiento_nodos(xs)
    if not diag["es_equiespaciado"]:
        st.warning(f"⚠️ **{diag['mensaje']}** Se calcula automáticamente con **Lagrange** como respaldo válido para espaciado irregular.")
        res_lag = interpolar_lagrange(xs, ys, x_eval)
        st.metric("P(x*) por Lagrange (Respaldo válido)", f"{res_lag['y_interpolado']:.6f}")
        x_grid = np.linspace(min(xs) - 0.1, max(xs) + 0.1, 220)
        y_lag_grid = limpiar_valores_plotly(evaluar_curva_lagrange(xs, ys, x_grid), limite_abs=1e9)
        fig = crear_figura_base_plotly("Nodos No Equidistantes → Respaldo Polinomio de Lagrange", "x", "y")
        fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_lag_grid, mode="lines", name="Lagrange P(x)", line=dict(color="#D4A359", width=2.5)))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos editados", marker=dict(size=9, color="#C86D44")))
        fig.add_trace(
            go.Scatter(
                x=[x_eval],
                y=[res_lag["y_interpolado"]],
                mode="markers",
                name="P(x*)",
                marker=dict(size=13, color="#FFFDF9", symbol="star", line=dict(color="#B5673F", width=1.8)),
            )
        )
        render_plotly_arandu(fig, key="plot_int_ng_fallback")
        return

    st.success(f"✅ {diag['mensaje']}")
    es_asc = variante.startswith("Ascendente")
    if es_asc:
        res = interpolar_newton_gregory_ascendente(xs, ys, x_eval)
        param_label, param_val = "u", res["u"]
        modo_str = "ascendente"
    else:
        res = interpolar_newton_gregory_descendente(xs, ys, x_eval)
        param_label, param_val = "v", res["v"]
        modo_str = "descendente"

    m1, m2, m3 = st.columns(3)
    m1.metric(f"P({x_eval:g})", f"{res['y_interpolado']:.6f}")
    m2.metric(f"Parámetro {param_label}", f"{param_val:.6f}")
    m3.metric("Paso h", f"{res['h']:.6g}")

    with st.expander("📊 Ver Tabla Completa de Diferencias Finitas"):
        render_dataframe_acotado(res["tabla_diferencias"], max_filas=300, key_prefix="tbl_int_ng_diff", hide_index=False)
    render_dataframe_acotado(res["terminos"], max_filas=300, key_prefix="tbl_int_ng_term")

    st.markdown("### 📐 Bloque 4 — Curva Polinómica por Grado $k$ (Paso a Paso)")
    grado_max = len(xs) - 1
    k_vis = render_slider_paso_k("Grado polinómico $k$ de Newton-Gregory:", grado_max, key="vis_ng_k")
    x_grid = np.linspace(min(min(xs), x_eval) - 0.15, max(max(xs), x_eval) + 0.15, 220)
    y_full = limpiar_valores_plotly(evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=grado_max, modo=modo_str), limite_abs=1e9)
    y_k = limpiar_valores_plotly(evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=k_vis, modo=modo_str), limite_abs=1e9)

    fig = crear_figura_base_plotly(f"Newton-Gregory ({modo_str.title()}) — Grado k = {k_vis}", "x", "y")
    fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_full, mode="lines", name=f"Grado completo ({grado_max})", line=dict(color="#9E9585", dash="dash")))
    fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_k, mode="lines", name=f"Grado k = {k_vis}", line=dict(color="#7A9A65", width=2.8)))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos tabla", marker=dict(size=9, color="#D4A359")))
    fig.add_trace(
        go.Scatter(
            x=[x_eval],
            y=[res["y_interpolado"]],
            mode="markers",
            name="P(x*)",
            marker=dict(size=13, color="#FFFDF9", symbol="star", line=dict(color="#B5673F", width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_int_ng")


def _tab_newton_gregory() -> None:
    meta_asc = render_bloque_1_teoria_y_ejemplo("newton_gregory_asc")
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
    render_practica_catedra_info(
        titulo_practica="TP5 Ej. 3: Tabla de Diferencias y Newton-Gregory Ascendente / Descendente",
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=r"Ascendente en $x = 0.35 \approx 1.2147$ | Descendente en $x = 1.37 \approx 10.0622$.",
    )
    _frag_newton_gregory(defaults)


def vista_interpolacion() -> None:
    inject_arandu_css()
    st.title("📈 Interpolación Polinomial")
    st.markdown(
        r"Construcción de polinomios sobre tablas discretas $(x_i, y_i)$ con detección automática "
        r"de pérdida de equidistancia ($\Delta h$) y alerta del fenómeno de Runge."
    )
    render_mini_catalogo_vista(
        [
            ("1. Lineal", "Une dos puntos consecutivos con una recta directa o inversa."),
            ("2. Cuadrática Inversa", "Ajusta una parábola en 3 nodos para despejar x* dado y*."),
            ("3. Lagrange", "Combina bases polinomiales sobre nodos con paso irregular."),
            ("4. Newton-Gregory", "Acelera con diferencias finitas Δy cuando Δh es constante."),
        ]
    )

    tabs = st.tabs(
        [
            "1️⃣ Lineal (Directa / Inversa)",
            "2️⃣ Cuadrática Inversa",
            "3️⃣ Lagrange (Nodos Libres)",
            "4️⃣ Newton-Gregory (Asc / Desc)",
        ]
    )
    with tabs[0]:
        _tab_lineal()
    with tabs[1]:
        _tab_cuadratica_inversa()
    with tabs[2]:
        _tab_lagrange()
    with tabs[3]:
        _tab_newton_gregory()


vista_interpolacion()
