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
    html_tabla_a_markdown,
    inject_arandu_css,
    limpiar_valores_plotly,
    registrar_contexto_ia,
    render_bloque_1_teoria_y_ejemplo,
    render_controles_edicion_practica,
    render_data_editor_arandu,
    render_dataframe_acotado,
    render_deduccion_paso_a_paso_viva,
    render_excepcion_validacion,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
    render_subtitulo_controles_vivo,
    render_toggle_modo_detallado,
    rerun_fragmento_seguro,
)


_BATERIA_X = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0]
_BATERIA_Y = [10.0, 35.0, 58.0, 76.0, 91.0, 98.0]


def _aplicar_preset_interpolacion(valores: dict[str, object], claves_ui: tuple[str, ...]) -> None:
    """Actualiza parámetros en session_state y limpia claves de widgets UI asociados."""
    for clave, valor in valores.items():
        st.session_state[clave] = valor
    for clave_ui in claves_ui:
        st.session_state.pop(clave_ui, None)


def _parsear_lista_interpolacion(texto: str, etiqueta: str) -> list[float]:
    """Convierte 'x0, x1, ...' en lista de floats finitos de longitud >= 2."""
    limpio = (texto or "").replace(";", ",").replace("[", "").replace("]", "")
    partes = [float(p.strip()) for p in limpio.split(",") if p.strip()]
    if len(partes) < 2:
        raise ValueError(f"La lista de {etiqueta} debe contener al menos 2 valores numéricos separados por coma.")
    if not all(math.isfinite(p) for p in partes):
        raise ValueError(f"Todos los valores de {etiqueta} deben ser números reales finitos.")
    return partes


def _construir_tabla_tramos_lineal(
    xs: list[float],
    ys: list[float],
    idx_activo: int,
    etiqueta_x: str,
    etiqueta_y: str,
) -> list[dict]:
    """Formatea los subintervalos consecutivos de la tabla con encabezados 100% en español."""
    pares = sorted(zip(xs, ys), key=lambda t: t[0])
    filas = []
    for i in range(len(pares) - 1):
        x0, y0 = float(pares[i][0]), float(pares[i][1])
        x1, y1 = float(pares[i + 1][0]), float(pares[i + 1][1])
        h_seg = x1 - x0
        dy_seg = y1 - y0
        pend = dy_seg / h_seg if abs(h_seg) > 1e-15 else None
        filas.append(
            {
                "Tramo": f"Tramo {i + 1}",
                f"Inicio ({etiqueta_x}, {etiqueta_y})": f"({x0:g}, {y0:g})",
                f"Fin ({etiqueta_x}, {etiqueta_y})": f"({x1:g}, {y1:g})",
                "Paso h": round(h_seg, 6),
                "Diferencia Δy": round(dy_seg, 6),
                "Pendiente Δy/h": round(pend, 6) if pend is not None else None,
                "Selección": "★ Activo" if i == idx_activo else "Vecino",
            }
        )
    return filas


def _editor_tabla_xy(
    clave_state: str,
    solo_lectura: bool,
    col_x_label: str = "Abscisa xᵢ",
    col_y_label: str = "Ordenada yᵢ",
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
def _frag_lineal(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_bateria", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Carga Rápida de Batería USB-PD (Minutos t ↔ Carga %)"
            if op == "real_bateria"
            else "Tabla Libre (x, y) y Cátedra TP5 Viscosidad"
        ),
        horizontal=True,
        key="ui_lin_contexto",
    )
    es_caso_real = contexto == "real_bateria"

    error_parseo: Exception | None = None
    if es_caso_real:
        st.caption(
            r"Mediciones USB-PD: $t = [0, 10, 20, 30, 45, 60]\;\text{min}$ y carga $= [10, 35, 58, 76, 91, 98]\%$. "
            r"Estimá cuánta carga hay a los $t^*$ minutos (**Directa**) o en qué minuto llegás al $y^*$% (**Inversa**)."
        )
        if "int_lin_real_sent" not in st.session_state:
            st.session_state["int_lin_real_sent"] = "Directa: Minutos t* → Carga %"
        if "int_lin_real_t" not in st.session_state:
            st.session_state["int_lin_real_t"] = 15.0
        if "int_lin_real_pct" not in st.session_state:
            st.session_state["int_lin_real_pct"] = 80.0

        claves_ui_real = ("ui_lin_real_sent", "ui_lin_real_t", "ui_lin_real_pct")
        b1, b2, b3 = st.columns(3)
        if b1.button("Carga Directa (t* = 15 min → 46.5%)", key="btn_lin_bat_dir", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_lin_real_sent": "Directa: Minutos t* → Carga %", "int_lin_real_t": 15.0},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button("Meta Inversa (80% → 34.0 min)", key="btn_lin_bat_inv", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_lin_real_sent": "Inversa: Carga %* → Minutos t", "int_lin_real_pct": 80.0},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button("Extrapolación (t* = 70 min → aviso)", key="btn_lin_bat_ext", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_lin_real_sent": "Directa: Minutos t* → Carga %", "int_lin_real_t": 70.0},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2 = st.columns([1.2, 1.0])
        sentido = str(
            c1.radio(
                "Sentido del cálculo:",
                ["Directa: Minutos t* → Carga %", "Inversa: Carga %* → Minutos t"],
                index=0 if str(st.session_state["int_lin_real_sent"]).startswith("Directa") else 1,
                horizontal=True,
                key="ui_lin_real_sent",
            )
        )
        st.session_state["int_lin_real_sent"] = sentido
        es_inversa = sentido.startswith("Inversa")
        if es_inversa:
            y_busc = float(
                c2.slider(
                    "Porcentaje de carga objetivo y* (%):",
                    min_value=5.0,
                    max_value=110.0,
                    value=float(st.session_state["int_lin_real_pct"]),
                    step=1.0,
                    key="ui_lin_real_pct",
                )
            )
            st.session_state["int_lin_real_pct"] = y_busc
            x_eval = float(st.session_state["int_lin_real_t"])
        else:
            x_eval = float(
                c2.slider(
                    "Minuto a consultar t* (min):",
                    min_value=-5.0,
                    max_value=80.0,
                    value=float(st.session_state["int_lin_real_t"]),
                    step=1.0,
                    key="ui_lin_real_t",
                )
            )
            st.session_state["int_lin_real_t"] = x_eval
            y_busc = float(st.session_state["int_lin_real_pct"])

        xs, ys = list(_BATERIA_X), list(_BATERIA_Y)
        etiqueta_x, etiqueta_y = "t (min)", "Carga (%)"
    else:
        if "int_lin_x" not in st.session_state:
            st.session_state["int_lin_x"] = list(defaults["x"])
        if "int_lin_y" not in st.session_state:
            st.session_state["int_lin_y"] = list(defaults["y"])
        if "int_lin_x_eval" not in st.session_state:
            st.session_state["int_lin_x_eval"] = float(defaults["x_eval"])
        if "int_lin_y_buscado" not in st.session_state:
            st.session_state["int_lin_y_buscado"] = float(defaults["y_buscado"])
        if "int_lin_sent" not in st.session_state:
            st.session_state["int_lin_sent"] = "Inversa: buscar x dado y*"

        claves_ui_lib = ("ui_lin_xs_txt", "ui_lin_ys_txt", "int_lin__editor_df", "ui_lin_sent", "ui_lin_yb", "ui_lin_xe")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP5 Viscosidad (y* = 1.702 → 1.648 °C)", key="btn_lin_tp5", use_container_width=True):
            _aplicar_preset_interpolacion(
                {
                    "int_lin_x": list(defaults["x"]),
                    "int_lin_y": list(defaults["y"]),
                    "int_lin_x_eval": float(defaults["x_eval"]),
                    "int_lin_y_buscado": float(defaults["y_buscado"]),
                    "int_lin_sent": "Inversa: buscar x dado y*",
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Carga Batería USB-PD (80% → 34.0 min)", key="btn_lin_bat_lib", use_container_width=True):
            _aplicar_preset_interpolacion(
                {
                    "int_lin_x": list(_BATERIA_X),
                    "int_lin_y": list(_BATERIA_Y),
                    "int_lin_x_eval": 15.0,
                    "int_lin_y_buscado": 80.0,
                    "int_lin_sent": "Inversa: buscar x dado y*",
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        if modo_detallado:
            xs, ys = _editor_tabla_xy("int_lin", solo_lectura=False, col_x_label="Abscisa xᵢ", col_y_label="Ordenada yᵢ")
        else:
            cx, cy = st.columns(2)
            xs_cur = st.session_state["int_lin_x"]
            ys_cur = st.session_state["int_lin_y"]
            xs_txt = cx.text_input(
                "Abscisas x (separadas por coma):",
                value=", ".join(f"{v:g}" for v in xs_cur),
                key="ui_lin_xs_txt",
            )
            ys_txt = cy.text_input(
                "Ordenadas y (separadas por coma):",
                value=", ".join(f"{v:g}" for v in ys_cur),
                key="ui_lin_ys_txt",
            )
            try:
                xs_p = _parsear_lista_interpolacion(xs_txt, "abscisas x")
                ys_p = _parsear_lista_interpolacion(ys_txt, "ordenadas y")
                if len(xs_p) != len(ys_p):
                    raise ValueError(f"Cantidad desigual: {len(xs_p)} abscisas x y {len(ys_p)} ordenadas y.")
                xs, ys = xs_p, ys_p
                st.session_state["int_lin_x"] = xs
                st.session_state["int_lin_y"] = ys
            except Exception as exc:
                xs, ys = xs_cur, ys_cur
                error_parseo = exc

        c1, c2 = st.columns([1.2, 1.0])
        sentido = str(
            c1.radio(
                "Tipo de interpolación lineal:",
                ["Inversa: buscar x dado y*", "Directa: buscar y dado x*"],
                index=0 if str(st.session_state["int_lin_sent"]).startswith("Inversa") else 1,
                horizontal=True,
                key="ui_lin_sent",
            )
        )
        st.session_state["int_lin_sent"] = sentido
        es_inversa = sentido.startswith("Inversa")
        if es_inversa:
            y_busc = float(
                c2.number_input(
                    "Valor objetivo y* buscado:",
                    value=float(st.session_state["int_lin_y_buscado"]),
                    format="%.4f",
                    key="ui_lin_yb",
                )
            )
            st.session_state["int_lin_y_buscado"] = y_busc
            x_eval = float(st.session_state["int_lin_x_eval"])
        else:
            x_eval = float(
                c2.number_input(
                    "Valor x* a evaluar:",
                    value=float(st.session_state["int_lin_x_eval"]),
                    format="%.4f",
                    key="ui_lin_xe",
                )
            )
            st.session_state["int_lin_x_eval"] = x_eval
            y_busc = float(st.session_state["int_lin_y_buscado"])

        etiqueta_x, etiqueta_y = "x", "y"

    try:
        if error_parseo is not None:
            raise error_parseo
        if es_inversa:
            res = interpolar_lineal_inversa(xs, ys, y_busc)
            px, py = res["x_resultado"], res["y_buscado"]
            idx_activo = res["indice"]
        else:
            res = interpolar_lineal_directa(xs, ys, x_eval)
            px, py = res["x_eval"], res["y_interpolado"]
            idx_activo = res["indice_intervalo"]
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla (x, y) o valor objetivo (Lineal)",
            sugerencia_defecto="Ingresá al menos 2 pares (x, y) finitos con abscisas xᵢ únicas y tramo no horizontal para inversa.",
            ejemplo_defecto="x = 0, 5, 10, 15 | y = 1.792, 1.519, 1.308, 1.140",
        )
        return

    if res.get("es_extrapolacion"):
        st.warning(
            "El valor objetivo cae fuera del rango tabulado: la fórmula realiza **extrapolación lineal**, "
            "proyectando la recta más allá de los datos medidos."
        )

    pend = float(res["pendiente"])
    m1, m2, m3 = st.columns(3)
    if es_inversa:
        lbl_m1 = "Tiempo estimado t(y*)" if es_caso_real else "Abscisa estimada x(y*)"
        val_m1 = f"{px:.2f} min" if es_caso_real else f"{px:.6f}"
    else:
        lbl_m1 = "Carga estimada y(t*)" if es_caso_real else "Ordenada estimada P₁(x*)"
        val_m1 = f"{py:.2f}%" if es_caso_real else f"{py:.6f}"
    m1.metric(
        lbl_m1,
        val_m1,
        "⚠ Extrapolación fuera de rango" if res.get("es_extrapolacion") else "✓ Interpolación en tramo interno",
        delta_color="off",
    )
    m2.metric(
        "Tramo vecino activo",
        f"[{res['x0']:g}, {res['x1']:g}]",
        f"h = {res['h']:g} | Δy = {res['delta_y']:.4g}",
        delta_color="off",
    )
    m3.metric(
        "Pendiente secante Δy/h",
        f"{pend:+.4f} %/min" if es_caso_real else f"{pend:+.6f}",
        "Creciente ↗" if pend > 0 else "Decreciente ↘",
        delta_color="off",
    )

    registrar_contexto_ia(
        f"Resultado Interpolación Lineal ({'Inversa' if es_inversa else 'Directa'})",
        f"- **Escenario:** {'Batería USB-PD' if es_caso_real else 'Libre / TP5 Viscosidad'}\n"
        f"- **Objetivo:** `{'y*' if es_inversa else 'x*'} = {py if es_inversa else px:g}` | "
        f"**Resultado:** `{'x' if es_inversa else 'y'} = {px if es_inversa else py:.6f}`\n"
        f"- **Tramo activo:** `[{res['x0']:g}, {res['x1']:g}]` | **Pendiente:** `{pend:.6f}`",
        clave="int_lin_res",
    )

    k_vis = (
        render_slider_paso_k("Paso de construcción geométrica k:", 3, key="vis_lin_k")
        if modo_detallado
        else 3
    )

    pares_ord = sorted(zip(xs, ys), key=lambda t: t[0])
    x_ord = [float(p[0]) for p in pares_ord]
    y_ord = [float(p[1]) for p in pares_ord]

    fig = crear_figura_base_plotly(
        "Recta Secante sobre Tramo Vecino" if not es_caso_real else "Curva de Carga USB-PD y Recta Secante Activa",
        etiqueta_x,
        etiqueta_y,
        altura=330,
    )
    fig.add_trace(
        go.Scatter(
            x=x_ord,
            y=y_ord,
            mode="lines+markers",
            name="Mediciones tabuladas",
            line=dict(color=PALETA_ARANDU["texto_atenuado"], dash="dot", width=1.8),
            marker=dict(size=8, color=PALETA_ARANDU["oro"]),
        )
    )
    if len(x_ord) >= 3 and modo_detallado:
        fig.add_trace(
            go.Scatter(
                x=[x_ord[0], x_ord[-1]],
                y=[y_ord[0], y_ord[-1]],
                mode="lines",
                name="Cuerda global (extremos)",
                line=dict(color=PALETA_ARANDU["terracota_oscuro"], dash="dashdot", width=1.2),
            )
        )
    if k_vis >= 2:
        x_seg = [res["x0"], res["x1"]]
        y_seg = [res["y0"], res["y1"]]
        if res.get("es_extrapolacion"):
            xs_ext = sorted([res["x0"], res["x1"], px])
            x_seg = [xs_ext[0], xs_ext[-1]]
            y_seg = [res["y0"] + pend * (xv - res["x0"]) for xv in x_seg]
        fig.add_trace(
            go.Scatter(
                x=x_seg,
                y=limpiar_valores_plotly(y_seg),
                mode="lines+markers",
                name="Secante activa",
                line=dict(color=PALETA_ARANDU["salvia"], width=3.0),
                marker=dict(size=10, color=PALETA_ARANDU["salvia"]),
            )
        )
    if k_vis >= 3:
        fig.add_vline(x=px, line_dash="dot", line_color=PALETA_ARANDU["texto_atenuado"], line_width=1.1)
        fig.add_hline(y=py, line_dash="dot", line_color=PALETA_ARANDU["texto_atenuado"], line_width=1.1)
        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[py],
                mode="markers",
                name="Punto estimado",
                marker=dict(
                    size=14,
                    color=PALETA_ARANDU["marfil_nodo"],
                    symbol="star",
                    line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8),
                ),
            )
        )
    render_plotly_arandu(fig, key="plot_int_lin")

    if modo_detallado:
        if es_inversa:
            pasos_lin_vivos = [
                f"**Selección del tramo activo:** Entre $(x_0={res['x0']:g}, y_0={res['y0']:g})$ y $(x_1={res['x1']:g}, y_1={res['y1']:g})$ queda encerrada la ordenada objetivo $y^* = {py:g}$.",
                f"**Cálculo del paso y diferencias:** $h = x_1 - x_0 = {res['h']:g}$, $\\Delta y_0 = y_1 - y_0 = {res['delta_y']:.6g}$ y $y^* - y_0 = {res['diferencia_y']:.6g}$.",
                f"**Sustitución en fórmula lineal inversa:** $x = {res['x0']:g} + \\frac{{({res['diferencia_y']:.6g})({res['h']:g})}}{{{res['delta_y']:.6g}}} = {px:.6f}$.",
            ]
        else:
            pasos_lin_vivos = [
                f"**Selección del tramo activo:** Para $x^* = {px:g}$ se utiliza el segmento entre $(x_0={res['x0']:g}, y_0={res['y0']:g})$ y $(x_1={res['x1']:g}, y_1={res['y1']:g})$.",
                f"**Pendiente local secante:** $h = {res['h']:g}$, $\\Delta y_0 = {res['delta_y']:.6g} \\implies m = \\frac{{\\Delta y_0}}{{h}} = {pend:.6g}$.",
                f"**Sustitución en $P_1(x^*)$:** $P_1({px:g}) = {res['y0']:g} + ({pend:.6g})({px:g} - {res['x0']:g}) = {py:.6f}$.",
            ]
        render_deduccion_paso_a_paso_viva(
            f"Interpolación Lineal {'Inversa' if es_inversa else 'Directa'} ({'y*' if es_inversa else 'x*'}={py if es_inversa else px:g})",
            pasos_lin_vivos,
            clave_ia="int_lin_viva",
            expanded=True,
        )
        render_dataframe_acotado(
            _construir_tabla_tramos_lineal(xs, ys, idx_activo, etiqueta_x, etiqueta_y),
            max_filas=100,
            key_prefix="tbl_int_lin",
            hide_index=True,
        )


def _tab_lineal() -> None:
    modo_detallado = render_toggle_modo_detallado("interpolacion_lineal")
    meta = render_bloque_1_teoria_y_ejemplo("interpolacion_lineal", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    defaults = {
        "x": raw_def.get("x", [0.0, 5.0, 10.0, 15.0]),
        "y": raw_def.get("y", [1.792, 1.519, 1.308, 1.140]),
        "x_eval": float(raw_def.get("x_objetivo", raw_def.get("x_eval", 2.5))),
        "y_buscado": float(raw_def.get("y_objetivo", raw_def.get("y_buscado", 1.702))),
    }
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_lineal(defaults, modo_detallado=modo_detallado)


_FRENADO_X = [40.0, 60.0, 80.0, 100.0, 120.0]
_FRENADO_Y = [18.0, 36.0, 60.0, 90.0, 126.0]


def _construir_tabla_cuadratica_inversa(res: dict, res_lin: dict | None) -> list[dict]:
    """Formatea los coeficientes y raíces candidatas con encabezados 100% en español."""
    filas = [
        {"Elemento": "Trío equidistante (x₀, x₁, x₂)", "Valor": f"({res['x0']:g}, {res['x1']:g}, {res['x2']:g})", "Interpretación": f"Paso uniforme h = {res['h']:g}"},
        {"Elemento": "Diferencia primera Δy₀", "Valor": f"{res['delta_y0']:.6g}", "Interpretación": "Pendiente inicial y₁ − y₀"},
        {"Elemento": "Curvatura Δ²y₀", "Valor": f"{res['delta2_y0']:.6g}", "Interpretación": "Corrección parabólica de 2° orden"},
        {"Elemento": "Coeficientes (a, b, c)", "Valor": f"a={res['a_coef']:.5g}, b={res['b_coef']:.5g}, c={res['c_coef']:.5g}", "Interpretación": "Ecuación a·u² + b·u + c = 0 (u = x − x₀)"},
    ]
    if not res.get("es_lineal"):
        x_sel = float(res["x_resultado"])
        x1, x2 = float(res["x_sol1"]), float(res["x_sol2"])
        filas.append({"Elemento": "Discriminante b² − 4ac", "Valor": f"{res['discriminante']:.6g}", "Interpretación": "Δ ≥ 0 garantiza raíces reales"})
        filas.append({"Elemento": "Candidata 1 (x₁)", "Valor": f"{x1:.6f}", "Interpretación": "★ Seleccionada (dentro del rango)" if abs(x1 - x_sel) < 1e-9 else "Descartada (rama externa)"})
        filas.append({"Elemento": "Candidata 2 (x₂)", "Valor": f"{x2:.6f}", "Interpretación": "★ Seleccionada (dentro del rango)" if abs(x2 - x_sel) < 1e-9 else "Descartada (rama externa)"})
    if res_lin is not None:
        filas.append({"Elemento": "Comparación Lineal Inversa", "Valor": f"{res_lin['x_resultado']:.6f}", "Interpretación": f"Diferencia de curvatura Δx = {res['x_resultado'] - res_lin['x_resultado']:+.6f}"})
    return filas


@st.fragment
def _frag_cuadratica_inversa(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_frenado", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Peritaje de Frenado Vehicular (Velocidad ↔ Distancia)"
            if op == "real_frenado"
            else "Tabla Equidistante Libre y Cátedra TP5 Ej. 2b"
        ),
        horizontal=True,
        key="ui_cuad_contexto",
    )
    es_caso_real = contexto == "real_frenado"

    error_parseo: Exception | None = None
    if es_caso_real:
        st.caption(
            r"Ensayo vehicular ($h = 20\;\text{km/h}$): $v = [40, 60, 80, 100, 120]\;\text{km/h}$ y frenado $= [18, 36, 60, 90, 126]\;\text{m}$ ($\Delta^2 d = +6\;\text{m}$). "
            r"Calculá la velocidad ante una huella $d^*$ usando la parábola exacta."
        )
        if "int_cuad_real_d" not in st.session_state:
            st.session_state["int_cuad_real_d"] = 48.0

        claves_ui_real = ("ui_cuad_real_d",)
        b1, b2, b3 = st.columns(3)
        if b1.button("Huella 48 m (→ 70.62 km/h)", key="btn_cuad_fr_48", use_container_width=True):
            _aplicar_preset_interpolacion({"int_cuad_real_d": 48.0}, claves_ui_real)
            rerun_fragmento_seguro()
        if b2.button("Frenado largo 110 m (→ 111.52 km/h)", key="btn_cuad_fr_110", use_container_width=True):
            _aplicar_preset_interpolacion({"int_cuad_real_d": 110.0}, claves_ui_real)
            rerun_fragmento_seguro()
        if b3.button("Corte imposible (-2 m → Δ < 0)", key="btn_cuad_fr_imp", use_container_width=True):
            _aplicar_preset_interpolacion({"int_cuad_real_d": -2.0}, claves_ui_real)
            rerun_fragmento_seguro()

        y_busc = float(
            st.slider(
                "Distancia de frenado medida d* (m):",
                min_value=-5.0,
                max_value=140.0,
                value=float(st.session_state["int_cuad_real_d"]),
                step=1.0,
                key="ui_cuad_real_d",
            )
        )
        st.session_state["int_cuad_real_d"] = y_busc
        xs, ys = list(_FRENADO_X), list(_FRENADO_Y)
        etiqueta_x, etiqueta_y = "Velocidad v (km/h)", "Distancia d (m)"
    else:
        if "int_cuad_x" not in st.session_state:
            st.session_state["int_cuad_x"] = list(defaults["x"])
        if "int_cuad_y" not in st.session_state:
            st.session_state["int_cuad_y"] = list(defaults["y"])
        if "int_cuad_y_buscado" not in st.session_state:
            st.session_state["int_cuad_y_buscado"] = float(defaults["y_buscado"])

        claves_ui_lib = ("ui_cuad_xs_txt", "ui_cuad_ys_txt", "int_cuad__editor_df", "ui_cuad_yb")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP5 Ej. 2b (Viscosidad y* = 1.702 → 1.528 °C)", key="btn_cuad_tp5", use_container_width=True):
            _aplicar_preset_interpolacion(
                {
                    "int_cuad_x": list(defaults["x"]),
                    "int_cuad_y": list(defaults["y"]),
                    "int_cuad_y_buscado": float(defaults["y_buscado"]),
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Frenado Vehicular (48 m → 70.62 km/h)", key="btn_cuad_fr_lib", use_container_width=True):
            _aplicar_preset_interpolacion(
                {
                    "int_cuad_x": list(_FRENADO_X),
                    "int_cuad_y": list(_FRENADO_Y),
                    "int_cuad_y_buscado": 48.0,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        if modo_detallado:
            xs, ys = _editor_tabla_xy("int_cuad", solo_lectura=False, col_x_label="Abscisa xᵢ", col_y_label="Ordenada yᵢ")
        else:
            cx, cy = st.columns(2)
            xs_cur = st.session_state["int_cuad_x"]
            ys_cur = st.session_state["int_cuad_y"]
            xs_txt = cx.text_input(
                "Abscisas equidistantes x (separadas por coma):",
                value=", ".join(f"{v:g}" for v in xs_cur),
                key="ui_cuad_xs_txt",
            )
            ys_txt = cy.text_input(
                "Ordenadas y (separadas por coma):",
                value=", ".join(f"{v:g}" for v in ys_cur),
                key="ui_cuad_ys_txt",
            )
            try:
                xs_p = _parsear_lista_interpolacion(xs_txt, "abscisas x")
                ys_p = _parsear_lista_interpolacion(ys_txt, "ordenadas y")
                if len(xs_p) != len(ys_p):
                    raise ValueError(f"Cantidad desigual: {len(xs_p)} abscisas x y {len(ys_p)} ordenadas y.")
                xs, ys = xs_p, ys_p
                st.session_state["int_cuad_x"] = xs
                st.session_state["int_cuad_y"] = ys
            except Exception as exc:
                xs, ys = xs_cur, ys_cur
                error_parseo = exc

        y_busc = float(
            st.number_input(
                "Ordenada objetivo y*:",
                value=float(st.session_state["int_cuad_y_buscado"]),
                format="%.4f",
                key="ui_cuad_yb",
            )
        )
        st.session_state["int_cuad_y_buscado"] = y_busc
        etiqueta_x, etiqueta_y = "x", "y"

    try:
        if error_parseo is not None:
            raise error_parseo
        res = interpolar_cuadratica_inversa(xs, ys, y_busc)
        try:
            res_lin = interpolar_lineal_inversa(xs, ys, y_busc)
        except Exception:
            res_lin = None
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla (x, y) u ordenada y* (Cuadrática Inversa)",
            sugerencia_defecto="Asegurá al menos 3 nodos equidistantes únicos y un valor y* alcanzable por la parábola (Δ = b² − 4ac ≥ 0).",
            ejemplo_defecto="x = 0, 5, 10, 15 | y = 1.792, 1.519, 1.308, 1.140",
        )
        return

    if res.get("es_extrapolacion"):
        st.warning("La ordenada objetivo cae fuera del rango tabulado: la parábola extrapoló más allá de los nodos medidos.")

    x_est = float(res["x_resultado"])
    if not res.get("es_lineal"):
        rama_ext = res["x_sol2"] if abs(res["x_sol1"] - x_est) < 1e-9 else res["x_sol1"]
        sub_m1 = f"Descarta rama {rama_ext:.2f}"
    else:
        sub_m1 = "Caso lineal degenerado (Δ²y₀ ≈ 0)"

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Velocidad estimada v(d*)" if es_caso_real else "Abscisa estimada x(y*)",
        f"{x_est:.2f} km/h" if es_caso_real else f"{x_est:.6f}",
        sub_m1,
        delta_color="off",
    )
    m2.metric(
        "Trío equidistante (x₀, x₁, x₂)",
        f"({res['x0']:g}, {res['x1']:g}, {res['x2']:g})",
        f"h = {res['h']:g} | Δ²y₀ = {res['delta2_y0']:.4g}",
        delta_color="off",
    )
    if res_lin is not None:
        diff_lin = x_est - float(res_lin["x_resultado"])
        val_m3 = f"{diff_lin:+.2f} km/h" if es_caso_real else f"{diff_lin:+.6f}"
        sub_m3 = f"Lineal daba {res_lin['x_resultado']:.2f} km/h" if es_caso_real else f"Lineal daba {res_lin['x_resultado']:.6f}"
    else:
        val_m3 = "Sin lineal"
        sub_m3 = "Tramo lineal horizontal"
    m3.metric("Corrección vs Lineal", val_m3, sub_m3, delta_color="off")

    registrar_contexto_ia(
        "Resultado Cuadrática Inversa",
        f"- **Escenario:** {'Peritaje Frenado' if es_caso_real else 'Libre / TP5 Ej. 2b'} | **y*:** `{y_busc:g}`\n"
        f"- **Resultado x*:** `{x_est:.6f}` | **Trío:** `({res['x0']:g}, {res['x1']:g}, {res['x2']:g})`",
        clave="int_cuad_res",
    )

    k_vis = (
        render_slider_paso_k("Paso de análisis parabólico k:", 3, key="vis_cuad_k")
        if modo_detallado
        else 3
    )

    pares_ord = sorted(zip(xs, ys), key=lambda t: t[0])
    x_ord = [float(p[0]) for p in pares_ord]
    y_ord = [float(p[1]) for p in pares_ord]
    x_tri = [res["x0"], res["x1"], res["x2"]]
    y_tri = [res["y0"], res["y1"], res["y2"]]
    coefs = np.polyfit(x_tri, y_tri, 2)
    x_grid_lo = min(x_ord[0], x_est)
    x_grid_hi = max(x_ord[-1], x_est)
    x_grid = np.linspace(x_grid_lo, x_grid_hi, 220)
    with np.errstate(all="ignore"):
        y_parab = limpiar_valores_plotly(np.polyval(coefs, x_grid), limite_abs=1e9)

    fig = crear_figura_base_plotly(
        "Parábola de Frenado vs Objetivo d*" if es_caso_real else "Parábola Cuadrática Inversa P₂(x) vs y = y*",
        etiqueta_x,
        etiqueta_y,
        altura=330,
    )
    fig.add_hline(y=y_busc, line_dash="dash", line_color=PALETA_ARANDU["terracota"], annotation_text=f"y* = {y_busc:g}")
    fig.add_trace(
        go.Scatter(
            x=x_ord,
            y=y_ord,
            mode="markers",
            name="Nodos tabulados",
            marker=dict(size=9, color=PALETA_ARANDU["oro"]),
        )
    )
    if k_vis >= 2:
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_parab,
                mode="lines",
                name="Parábola P₂(x)",
                line=dict(color=PALETA_ARANDU["salvia"], width=2.8),
            )
        )
        if res_lin is not None:
            fig.add_trace(
                go.Scatter(
                    x=[res_lin["x0"], res_lin["x1"]],
                    y=[res_lin["y0"], res_lin["y1"]],
                    mode="lines",
                    name="Secante lineal (comparativa)",
                    line=dict(color=PALETA_ARANDU["texto_atenuado"], dash="dot", width=1.8),
                )
            )
    if k_vis >= 3:
        fig.add_vline(x=x_est, line_dash="dot", line_color=PALETA_ARANDU["texto_atenuado"], line_width=1.1)
        fig.add_trace(
            go.Scatter(
                x=[x_est],
                y=[y_busc],
                mode="markers",
                name="Raíz seleccionada",
                marker=dict(
                    size=14,
                    color=PALETA_ARANDU["marfil_nodo"],
                    symbol="star",
                    line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8),
                ),
            )
        )
    render_plotly_arandu(fig, key="plot_int_cuad")

    if modo_detallado:
        pasos_cuad_vivos = [
            f"**Nodos equidistantes seleccionados ($h = {res['h']:g}$):** $(x_0={res['x0']:g}, y_0={res['y0']:g})$, $(x_1={res['x1']:g}, y_1={res['y1']:g})$, $(x_2={res['x2']:g}, y_2={res['y2']:g})$.",
            f"**Diferencias progresivas hasta orden 2:** $\\Delta y_0 = {res['delta_y0']:.6g}$ y curvatura $\\Delta^2 y_0 = {res['delta2_y0']:.6g}$.",
            f"**Ecuación cuadrática en $u = x - x_0$ para $y^* = {y_busc:g}$:** ${res['a_coef']:.6g}\\,u^2 + ({res['b_coef']:.6g})\\,u + ({res['c_coef']:.6g}) = 0$.",
        ]
        if not res.get("es_lineal"):
            pasos_cuad_vivos.append(
                f"**Resolvente cuadrática y selección de rama:** Discriminante $\\Delta = {res['discriminante']:.6g}$; "
                f"candidatos $x_1 = {res['x_sol1']:.6f}$ y $x_2 = {res['x_sol2']:.6f} \\implies$ se selecciona la raíz dentro del rango: $x = {x_est:.6f}$."
            )
        else:
            pasos_cuad_vivos.append(
                f"**Caso degenerado lineal ($\\Delta^2 y_0 \\approx 0$):** Se despeja directamente $x = {x_est:.6f}$."
            )
        render_deduccion_paso_a_paso_viva(
            f"Cuadrática Inversa (y*={y_busc:g})",
            pasos_cuad_vivos,
            clave_ia="int_cuad_viva",
            expanded=True,
        )
        render_dataframe_acotado(
            _construir_tabla_cuadratica_inversa(res, res_lin),
            max_filas=50,
            key_prefix="tbl_int_cuad",
            hide_index=True,
        )


def _tab_cuadratica_inversa() -> None:
    modo_detallado = render_toggle_modo_detallado("cuadratica_inversa")
    meta = render_bloque_1_teoria_y_ejemplo("cuadratica_inversa", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    defaults = {
        "x": raw_def.get("x", [0.0, 5.0, 10.0, 15.0]),
        "y": raw_def.get("y", [1.792, 1.519, 1.308, 1.140]),
        "y_buscado": float(raw_def.get("y_objetivo", raw_def.get("y_buscado", 1.702))),
    }
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_cuadratica_inversa(defaults, modo_detallado=modo_detallado)


_TELE_X = [0.0, 2.0, 5.0, 10.0]
_TELE_Y = [18.0, 32.0, 58.0, 72.0]


def _construir_tabla_lagrange(terminos_detalle: list[dict]) -> list[dict]:
    """Formatea los pesos base L_i(x*) y aportes de Lagrange con encabezados 100% en español."""
    return [
        {
            "Índice i": int(td["i"]),
            "Nodo xᵢ": round(float(td["x_i"]), 6),
            "Ordenada yᵢ": round(float(td["y_i"]), 6),
            "Base Lᵢ(x*)": round(float(td["L_i"]), 6),
            "Aporte yᵢ·Lᵢ(x*)": round(float(td["termino"]), 6),
            "Suma acumulada": round(float(td["acumulado"]), 6),
        }
        for td in terminos_detalle
    ]


@st.fragment
def _frag_lagrange(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_telemetria", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Telemetría Irregular de Sensor (y Fenómeno de Runge)"
            if op == "real_telemetria"
            else "Nodos Libres, Remuestreo y Cátedra TP5 Ej. 2"
        ),
        horizontal=True,
        key="ui_lag_contexto",
    )
    es_caso_real = contexto == "real_telemetria"

    error_parseo: Exception | None = None
    sub_real = "irreg"
    if es_caso_real:
        if "int_lag_real_sub" not in st.session_state:
            st.session_state["int_lag_real_sub"] = "irreg"
        if "int_lag_real_t" not in st.session_state:
            st.session_state["int_lag_real_t"] = 7.0
        if "int_lag_real_xr" not in st.session_state:
            st.session_state["int_lag_real_xr"] = 0.85
        if "int_lag_real_nr" not in st.session_state:
            st.session_state["int_lag_real_nr"] = 9

        claves_ui_real = ("ui_lag_real_t", "ui_lag_real_xr", "ui_lag_real_nr")
        b1, b2, b3 = st.columns(3)
        if b1.button("Malla irregular (t* = 7 h → 71.20 °C)", key="btn_lag_real_irreg", use_container_width=True):
            _aplicar_preset_interpolacion({"int_lag_real_sub": "irreg", "int_lag_real_t": 7.0}, claves_ui_real)
            rerun_fragmento_seguro()
        if b2.button("Inicio de jornada (t* = 1 h → 24.24 °C)", key="btn_lag_real_borde", use_container_width=True):
            _aplicar_preset_interpolacion({"int_lag_real_sub": "irreg", "int_lag_real_t": 1.0}, claves_ui_real)
            rerun_fragmento_seguro()
        if b3.button("Trampa Runge (9 nodos campana, x* = 0.85)", key="btn_lag_real_runge", use_container_width=True):
            _aplicar_preset_interpolacion({"int_lag_real_sub": "runge", "int_lag_real_nr": 9, "int_lag_real_xr": 0.85}, claves_ui_real)
            rerun_fragmento_seguro()

        sub_real = str(st.session_state["int_lag_real_sub"])
        if sub_real == "runge":
            st.caption(
                r"Muestra equiespaciada de la campana $f(x) = \frac{1}{1+25x^2}$ en $[-1, 1]$: "
                r"al subir el grado $N$ con paso uniforme, el polinomio oscila violentamente en los extremos (**Fenómeno de Runge**)."
            )
            c1, c2 = st.columns(2)
            n_runge = int(c1.slider("Cantidad de nodos equidistantes N:", 3, 11, int(st.session_state["int_lag_real_nr"]), step=2, key="ui_lag_real_nr"))
            x_eval = float(c2.slider("Punto x* cercano al borde:", -0.95, 0.95, float(st.session_state["int_lag_real_xr"]), step=0.05, key="ui_lag_real_xr"))
            st.session_state["int_lag_real_nr"] = n_runge
            st.session_state["int_lag_real_xr"] = x_eval
            xs, ys, _ = regenerar_tabla_interpolacion([-1.0, 1.0], [1.0 / 26.0, 1.0 / 26.0], nuevo_n=n_runge, f_expr="1 / (1 + 25 * x**2)")
            etiqueta_x, etiqueta_y = "Posición x", "Señal f(x)"
        else:
            st.caption(
                r"Sensor IoT asíncrono en horas $t = [0, 2, 5, 10]\;\text{h}$ ($\Delta h = [2, 3, 5]\;\text{h}$ irregular): "
                r"Lagrange construye $P_3(t)$ exacto sin exigir paso constante."
            )
            x_eval = float(
                st.slider(
                    "Hora t* a estimar (h):",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(st.session_state["int_lag_real_t"]),
                    step=0.5,
                    key="ui_lag_real_t",
                )
            )
            st.session_state["int_lag_real_t"] = x_eval
            xs, ys = list(_TELE_X), list(_TELE_Y)
            etiqueta_x, etiqueta_y = "Hora t (h)", "Temperatura (°C)"
    else:
        if "int_lag_x" not in st.session_state:
            st.session_state["int_lag_x"] = list(defaults["x"])
        if "int_lag_y" not in st.session_state:
            st.session_state["int_lag_y"] = list(defaults["y"])
        if "int_lag_x_eval" not in st.session_state:
            st.session_state["int_lag_x_eval"] = float(defaults["x_eval"])

        claves_ui_lib = ("ui_lag_xs_txt", "ui_lag_ys_txt", "int_lag__editor_df", "ui_lag_xe", "ui_lag_n")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP5 Ej. 2 (Presión Vapor T* = 180 → 10.562 Pa)", key="btn_lag_tp5", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_lag_x": list(defaults["x"]), "int_lag_y": list(defaults["y"]), "int_lag_x_eval": float(defaults["x_eval"])},
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Telemetría Irregular (t* = 7 h → 71.20 °C)", key="btn_lag_tele_lib", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_lag_x": list(_TELE_X), "int_lag_y": list(_TELE_Y), "int_lag_x_eval": 7.0},
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        if modo_detallado:
            xs, ys = _editor_tabla_xy("int_lag", solo_lectura=False, col_x_label="Abscisa xᵢ", col_y_label="Ordenada yᵢ")
        else:
            cx, cy = st.columns(2)
            xs_cur = st.session_state["int_lag_x"]
            ys_cur = st.session_state["int_lag_y"]
            xs_txt = cx.text_input("Abscisas libres x (separadas por coma):", value=", ".join(f"{v:g}" for v in xs_cur), key="ui_lag_xs_txt")
            ys_txt = cy.text_input("Ordenadas y (separadas por coma):", value=", ".join(f"{v:g}" for v in ys_cur), key="ui_lag_ys_txt")
            try:
                xs_p = _parsear_lista_interpolacion(xs_txt, "abscisas x")
                ys_p = _parsear_lista_interpolacion(ys_txt, "ordenadas y")
                if len(xs_p) != len(ys_p):
                    raise ValueError(f"Cantidad desigual: {len(xs_p)} abscisas x y {len(ys_p)} ordenadas y.")
                xs, ys = xs_p, ys_p
                st.session_state["int_lag_x"] = xs
                st.session_state["int_lag_y"] = ys
            except Exception as exc:
                xs, ys = xs_cur, ys_cur
                error_parseo = exc

        c1, c2 = st.columns([1.0, 1.35])
        x_eval = float(c1.number_input("Punto x* a interpolar:", value=float(st.session_state["int_lag_x_eval"]), format="%.4f", key="ui_lag_xe"))
        st.session_state["int_lag_x_eval"] = x_eval
        with c2:
            c2_a, c2_b = st.columns([1.6, 1.0])
            nuevo_n = c2_a.slider("Remuestrear a N nodos equidistantes:", 2, 10, min(10, max(2, len(xs))), key="ui_lag_n")
            if c2_b.button("Aplicar N nodos", key="btn_lag_remuestrear", use_container_width=True):
                x_new, y_new, _ = regenerar_tabla_interpolacion(defaults["x"], defaults["y"], nuevo_n=nuevo_n)
                _aplicar_preset_interpolacion({"int_lag_x": x_new, "int_lag_y": y_new}, ("ui_lag_xs_txt", "ui_lag_ys_txt", "int_lag__editor_df"))
                rerun_fragmento_seguro()
        etiqueta_x, etiqueta_y = "x", "P(x)"

    try:
        if error_parseo is not None:
            raise error_parseo
        pares_diag = sorted(zip(xs, ys), key=lambda t: t[0])
        diag = diagnosticar_espaciamiento_nodos([p[0] for p in pares_diag])
        res = interpolar_lagrange(xs, ys, x_eval)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos (x, y) o punto x* (Lagrange)",
            sugerencia_defecto="Ingresá al menos 2 nodos con abscisas xᵢ únicas y valores reales finitos.",
            ejemplo_defecto="x = 0, 2, 5, 10 | y = 18, 32, 58, 72",
        )
        return

    if (es_caso_real and sub_real == "runge" and len(xs) >= 7) or len(xs) >= 7:
        st.warning(
            "**Alerta sobre Fenómeno de Runge:** Con "
            f"$N = {len(xs)}$ nodos equidistantes, el polinomio de grado ${res['grado']}$ desarrolla oscilaciones pronunciadas cerca de los extremos."
        )

    y_est = float(res["y_interpolado"])
    pesos = res["polinomios_base"]
    idx_max_w = int(np.argmax(np.abs(pesos)))
    w_max = float(pesos[idx_max_w])

    m1, m2, m3 = st.columns(3)
    if es_caso_real and sub_real == "runge":
        exact_r = 1.0 / (1.0 + 25.0 * (x_eval**2))
        m1.metric(f"Estimación P_{res['grado']}({x_eval:g})", f"{y_est:.4f}", f"Exacta = {exact_r:.4f} (error {y_est - exact_r:+.3f})", delta_color="off")
    elif es_caso_real:
        m1.metric(f"Temperatura P₃({x_eval:g} h)", f"{y_est:.2f} °C", f"Grado {res['grado']} ({res['puntos']} lecturas)", delta_color="off")
    else:
        m1.metric(f"Estimación P_{res['grado']}({x_eval:g})", f"{y_est:.6f}", f"Grado {res['grado']} ({res['puntos']} nodos)", delta_color="off")

    m2.metric(
        "Malla de abscisas",
        "Irregular ✓" if not diag["es_equiespaciado"] else f"Uniforme (h = {diag['h_nominal']:.3g})",
        "Paso variable → Lagrange" if not diag["es_equiespaciado"] else ("Riesgo Runge con grado alto" if len(xs) >= 7 else "Coincide con Newton-Gregory"),
        delta_color="off",
    )
    m3.metric(
        "Suma de bases ΣLᵢ(x*)",
        f"{sum(pesos):.6f}",
        f"Peso dominante: L_{idx_max_w} = {w_max:+.3f}",
        delta_color="off",
    )

    simb = construir_polinomio_simbolico_lagrange(xs, ys) if modo_detallado else None
    registrar_contexto_ia(
        "Polinomio y Evaluación (Lagrange)",
        f"- **Escenario:** {'Telemetría / Runge' if es_caso_real else 'Libre / TP5 Ej. 2'}\n"
        f"- **$x^*$:** `{x_eval:g}` | **$P({x_eval:g})$:** `{y_est:.6f}` (grado `{res['grado']}`)",
        clave="int_lag_res",
    )

    k_vis = (
        render_slider_paso_k("Sumar hasta término i = k:", len(xs) - 1, key="vis_lag_k", min_val=0)
        if modo_detallado
        else len(xs) - 1
    )

    x_lo = min(min(xs), x_eval)
    x_hi = max(max(xs), x_eval)
    span = max(1e-3, x_hi - x_lo)
    pad = 0.0 if (es_caso_real and sub_real == "runge") else 0.04 * span
    x_grid = np.linspace(x_lo - pad, x_hi + pad, 240)
    y_full = limpiar_valores_plotly(evaluar_curva_lagrange(xs, ys, x_grid), limite_abs=1e9)

    y_parcial_arr = np.zeros_like(x_grid, dtype=float)
    for idx in range(k_vis + 1):
        y_parcial_arr += float(ys[idx]) * evaluar_curva_lagrange(xs, ys, x_grid, indice_base=idx)
    y_parcial = limpiar_valores_plotly(y_parcial_arr, limite_abs=1e9)

    fig = crear_figura_base_plotly(
        "Campana Real vs Oscilación Polinómica de Runge"
        if (es_caso_real and sub_real == "runge")
        else (f"Polinomio de Lagrange — Suma parcial i = 0..{k_vis}" if modo_detallado else f"Curva Interpoladora de Lagrange P_{res['grado']}(x)"),
        etiqueta_x,
        etiqueta_y,
        altura=335,
    )
    if es_caso_real and sub_real == "runge":
        y_true_curve = 1.0 / (1.0 + 25.0 * (x_grid**2))
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_true_curve.tolist(),
                mode="lines",
                name="Función real 1/(1+25x²)",
                line=dict(color=PALETA_ARANDU["terracota"], dash="dash", width=2.0),
            )
        )
    elif modo_detallado:
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_full,
                mode="lines",
                name="P(x) completo",
                line=dict(color=PALETA_ARANDU["texto_atenuado"], dash="dash"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_grid.tolist(),
            y=y_parcial,
            mode="lines",
            name=f"Suma hasta i={k_vis}" if modo_detallado else f"Lagrange P_{res['grado']}(x)",
            line=dict(color=PALETA_ARANDU["salvia"], width=2.8),
        )
    )
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos medidos", marker=dict(size=9, color=PALETA_ARANDU["oro"])))
    fig.add_trace(
        go.Scatter(
            x=[x_eval],
            y=[y_est],
            mode="markers",
            name="P(x*)",
            marker=dict(size=14, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_int_lag")

    if modo_detallado:
        if simb is not None:
            st.latex(rf"P_{{{res['grado']}}}(x) = {simb['latex_expandido']}")
        pasos_lag_vivos = [
            f"**Diagnóstico de malla:** {diag['mensaje']}",
            f"**Planteo de Lagrange ($n+1 = {res['puntos']}$ nodos, grado ${res['grado']}$):** $P_{{{res['grado']}}}(x^*) = \\sum_{{i=0}}^{{{res['grado']}}} y_i L_i({x_eval:g})$.",
        ]
        for td in res["terminos_detalle"][:6]:
            pasos_lag_vivos.append(
                f"**Base $i = {td['i']}$ ($x_{td['i']} = {td['x_i']:g}, y_{td['i']} = {td['y_i']:g}$):** "
                f"$L_{{{td['i']}}}({x_eval:g}) = {td['L_i']:.6f} \\implies y_{{{td['i']}}} L_{{{td['i']}}} = {td['termino']:.6f}$ "
                f"(acumulado $= {td['acumulado']:.6f}$)."
            )
        if simb is not None:
            pasos_lag_vivos.append(
                f"**Polinomio expandido resultante:** $P_{{{res['grado']}}}(x) = {simb['latex_expandido']}$, entregando $P({x_eval:g}) = {y_est:.6f}$."
            )
        render_deduccion_paso_a_paso_viva(
            f"Polinomio de Lagrange (x*={x_eval:g})",
            pasos_lag_vivos,
            clave_ia="int_lag_viva",
            expanded=True,
        )
        render_dataframe_acotado(
            _construir_tabla_lagrange(res["terminos_detalle"]),
            max_filas=300,
            key_prefix="tbl_int_lag",
            hide_index=True,
        )


def _tab_lagrange() -> None:
    modo_detallado = render_toggle_modo_detallado("lagrange")
    meta = render_bloque_1_teoria_y_ejemplo("lagrange", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"x": [100.0, 120.0, 200.0, 220.0], "y": [1.01, 3.61, 15.5, 23.2], "x_eval": 180.0},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_lagrange(defaults, modo_detallado=modo_detallado)


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
        "referencia": r"Ancla $x_0 = 0$, $h = 5$, $u = 0.306 \implies \mu_3(1.53) \approx 1.700740$ (grado 2: $1.701879$).",
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
        "referencia": r"Ancla $x_0 = 12$, $h = 1$, $u = 0.4 \implies P_5(12.4) \approx 0.958531$.",
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
        "ui_ng_xs_txt",
        "ui_ng_ys_txt",
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
    etiqueta_x: str = "x",
    etiqueta_y: str = "y",
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
        etiqueta_x,
        etiqueta_y,
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


_TRAFICO_X = [0.0, 2.0, 4.0, 6.0, 8.0]
_TRAFICO_X_DESIG = [0.0, 2.0, 4.7, 6.0, 8.0]
_TRAFICO_Y = [100.0, 125.0, 165.0, 225.0, 315.0]


@st.fragment
def _frag_newton_gregory(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_trafico", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Monitoreo Horario de Servidores (h = 2 h)"
            if op == "real_trafico"
            else "Tabla Libre y Prácticas de Cátedra (TP5 / U5 / TP6)"
        ),
        horizontal=True,
        key="ui_ng_contexto",
    )
    es_caso_real = contexto == "real_trafico"

    error_parseo: Exception | None = None
    exacta_fn = None
    exacta_tex = None
    if es_caso_real:
        if "int_ng_real_t" not in st.session_state:
            st.session_state["int_ng_real_t"] = 1.0
        if "int_ng_real_k" not in st.session_state:
            st.session_state["int_ng_real_k"] = 2
        if "int_ng_real_var" not in st.session_state:
            st.session_state["int_ng_real_var"] = _MODOS_VISUALES_NG[0]
        if "int_ng_real_desig" not in st.session_state:
            st.session_state["int_ng_real_desig"] = False

        claves_ui_real = ("ui_ng_real_t", "ui_ng_real_k", "ui_ng_real_var")
        st.caption(
            r"Serie cada $2\;\text{h}$: $t = [0, 2, 4, 6, 8]\;\text{h}$, $y = [100, 125, 165, 225, 315]\;\text{req/s}$. "
            r"Con grado truncado $k=2$, **Ascendente ($\Delta$)** acierta al inicio y **Descendente ($\nabla$)** acierta al final."
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("Inicio truncado k=2 (t* = 1 h → Δ gana)", key="btn_ng_real_ini", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_ng_real_t": 1.0, "int_ng_real_k": 2, "int_ng_real_var": _MODOS_VISUALES_NG[0], "int_ng_real_desig": False},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button("Final truncado k=2 (t* = 7 h → ∇ gana)", key="btn_ng_real_fin", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_ng_real_t": 7.0, "int_ng_real_k": 2, "int_ng_real_var": _MODOS_VISUALES_NG[0], "int_ng_real_desig": False},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button("Romper paso (t₂=4.7 → pasa a Lagrange)", key="btn_ng_real_desig", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_ng_real_t": 1.0, "int_ng_real_k": 4, "int_ng_real_var": _MODOS_VISUALES_NG[0], "int_ng_real_desig": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns([1.15, 0.95, 1.3])
        x_eval = float(c1.slider("Hora a estimar t* (h):", 0.0, 8.0, float(st.session_state["int_ng_real_t"]), step=0.25, key="ui_ng_real_t"))
        k_vis = int(c2.slider("Grado truncamiento k:", 1, 4, int(st.session_state["int_ng_real_k"]), key="ui_ng_real_k"))
        modo_visual = str(
            c3.radio(
                "Enfoque visual:",
                options=list(_MODOS_VISUALES_NG),
                index=_MODOS_VISUALES_NG.index(str(st.session_state["int_ng_real_var"])) if str(st.session_state["int_ng_real_var"]) in _MODOS_VISUALES_NG else 0,
                horizontal=True,
                key="ui_ng_real_var",
            )
        )
        st.session_state["int_ng_real_t"] = x_eval
        st.session_state["int_ng_real_k"] = k_vis
        st.session_state["int_ng_real_var"] = modo_visual

        xs = list(_TRAFICO_X_DESIG if st.session_state["int_ng_real_desig"] else _TRAFICO_X)
        ys = list(_TRAFICO_Y)
        grado_max = len(xs) - 1
        etiqueta_x, etiqueta_y = "Hora t (h)", "Carga (req/s)"
    else:
        if "int_ng_preset" not in st.session_state:
            st.session_state["int_ng_preset"] = "tp5_ej1_asc"
        if "int_ng_x" not in st.session_state:
            st.session_state["int_ng_x"] = list(defaults["x"])
        if "int_ng_y" not in st.session_state:
            st.session_state["int_ng_y"] = list(defaults["y"])
        if "int_ng_x_eval" not in st.session_state:
            st.session_state["int_ng_x_eval"] = float(defaults["x_eval"])

        preset_id = st.selectbox(
            "Ejercicio oficial de la cátedra / Caso de estudio:",
            options=list(_PRESETS_NEWTON_GREGORY.keys()),
            format_func=lambda k: _PRESETS_NEWTON_GREGORY[k]["titulo"],
            key="ui_ng_preset",
            on_change=_al_cambiar_preset_ng,
        )
        preset = _PRESETS_NEWTON_GREGORY[preset_id]
        exacta_fn = preset.get("exacta")
        exacta_tex = preset.get("exacta_tex")
        if modo_detallado:
            render_practica_catedra_info(
                titulo_practica=preset["titulo"],
                enunciado_practica=preset["enunciado"],
                solucion_referencia=preset["referencia"],
            )

        if modo_detallado:
            col_ctrl, col_tbl = st.columns([1.45, 1.15])
            with col_tbl:
                st.markdown("#### Nodos de la Tabla")
                xs, ys = _editor_tabla_xy("int_ng", False, col_x_label="Abscisa xᵢ", col_y_label="Ordenada yᵢ")
                try:
                    diag_prev = diagnosticar_espaciamiento_nodos(xs)
                    h_base = float(diag_prev["h_nominal"]) if diag_prev["es_equiespaciado"] and diag_prev["h_nominal"] else 0.3
                except Exception:
                    h_base = 0.3
                st.session_state.setdefault("ui_ng_x0", float(xs[0]))
                st.session_state.setdefault("ui_ng_dh", float(np.clip(h_base, 0.05, 10.0)))
                st.session_state["ui_ng_dh"] = float(np.clip(float(st.session_state["ui_ng_dh"]), 0.05, 10.0))
                c_m1, c_m2 = st.columns(2)
                nuevo_x0 = c_m1.number_input("Inicio x₀:", step=0.1, format="%.4f", key="ui_ng_x0")
                nuevo_h = c_m2.number_input("Paso Δh:", min_value=0.05, max_value=10.0, step=0.05, format="%.4f", key="ui_ng_dh")
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
            ctrl_ctx = col_ctrl
        else:
            cx, cy = st.columns(2)
            xs_cur = st.session_state["int_ng_x"]
            ys_cur = st.session_state["int_ng_y"]
            xs_txt = cx.text_input("Abscisas equidistantes x (separadas por coma):", value=", ".join(f"{v:g}" for v in xs_cur), key="ui_ng_xs_txt")
            ys_txt = cy.text_input("Ordenadas y (separadas por coma):", value=", ".join(f"{v:g}" for v in ys_cur), key="ui_ng_ys_txt")
            try:
                xs_p = _parsear_lista_interpolacion(xs_txt, "abscisas x")
                ys_p = _parsear_lista_interpolacion(ys_txt, "ordenadas y")
                if len(xs_p) != len(ys_p):
                    raise ValueError(f"Cantidad desigual: {len(xs_p)} abscisas x y {len(ys_p)} ordenadas y.")
                xs, ys = xs_p, ys_p
                st.session_state["int_ng_x"] = xs
                st.session_state["int_ng_y"] = ys
            except Exception as exc:
                xs, ys = xs_cur, ys_cur
                error_parseo = exc
            ctrl_ctx = st.container()

        with ctrl_ctx:
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
                "Deslizar x*:",
                min_value=float(s_min),
                max_value=float(s_max),
                step=float(step_s),
                key="ui_ng_xe_slider",
                on_change=_al_cambiar_slider_xe_ng,
            )
            cx2.number_input("Valor exacto x*:", format="%.4f", key="ui_ng_xe", on_change=_al_cambiar_input_xe_ng)
            x_eval = float(st.session_state["int_ng_x_eval"])

            grado_max = max(1, len(xs) - 1)
            c_gr, c_md = st.columns([1.0, 1.35])
            if grado_max > 1:
                if "vis_ng_k" not in st.session_state:
                    st.session_state["vis_ng_k"] = grado_max
                else:
                    st.session_state["vis_ng_k"] = int(np.clip(int(st.session_state["vis_ng_k"]), 1, grado_max))
                k_vis = int(c_gr.slider("Grado polinómico k:", min_value=1, max_value=grado_max, key="vis_ng_k"))
            else:
                k_vis = 1
                c_gr.caption("Con 2 nodos el grado máximo es k = 1.")

            modo_visual = c_md.radio(
                "Enfoque visual:",
                options=list(_MODOS_VISUALES_NG),
                horizontal=True,
                key="ui_ng_var",
            )
        etiqueta_x, etiqueta_y = "x", "y"

    try:
        if error_parseo is not None:
            raise error_parseo
        diag = diagnosticar_espaciamiento_nodos(xs)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos (x, y) (Newton-Gregory)",
            sugerencia_defecto="Ingresá al menos 2 nodos reales finitos sin valores repetidos ni longitudes desiguales.",
            ejemplo_defecto="x = 0.2, 0.5, 0.8, 1.1, 1.4",
        )
        return

    if not diag["es_equiespaciado"]:
        st.warning(f"**{diag['mensaje']}** Se activa automáticamente **Lagrange** como respaldo válido para paso irregular.")
        try:
            res_lag = interpolar_lagrange(xs, ys, x_eval)
            ml1, ml2, ml3 = st.columns(3)
            val_lag = f"{res_lag['y_interpolado']:.2f} req/s" if es_caso_real else f"{res_lag['y_interpolado']:.6f}"
            ml1.metric("Respaldo Lagrange P(x*)", val_lag, "Válido en malla irregular", delta_color="off")
            ml2.metric("Paso entre nodos", "Irregular ⚠", "Newton-Gregory invalidado", delta_color="off")
            ml3.metric("Grado del respaldo", f"Grado {res_lag['grado']}", f"{res_lag['puntos']} nodos usados", delta_color="off")

            x_grid = np.linspace(min(min(xs), x_eval) - 0.1, max(max(xs), x_eval) + 0.1, 220)
            y_lag_grid = limpiar_valores_plotly(evaluar_curva_lagrange(xs, ys, x_grid), limite_abs=1e9)
            fig = crear_figura_base_plotly("Nodos No Equidistantes → Respaldo Polinomio de Lagrange", etiqueta_x, etiqueta_y, altura=330)
            fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_lag_grid, mode="lines", name="Lagrange P(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.6)))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos actuales", marker=dict(size=9, color=PALETA_ARANDU["terracota"])))
            fig.add_trace(
                go.Scatter(
                    x=[x_eval],
                    y=[res_lag["y_interpolado"]],
                    mode="markers",
                    name="P(x*)",
                    marker=dict(size=14, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
                )
            )
            render_plotly_arandu(fig, key="plot_int_ng_fallback")
        except Exception as exc:
            render_excepcion_validacion(
                exc,
                campo="Tabla de nodos (respaldo Lagrange)",
                sugerencia_defecto="Eliminá abscisas x repetidas e ingresá al menos 2 pares finitos.",
                ejemplo_defecto="x = 0.0, 0.25, 0.5, 0.75",
            )
        return

    try:
        res_asc = interpolar_newton_gregory_ascendente(xs, ys, x_eval, grado_max=k_vis)
        res_desc = interpolar_newton_gregory_descendente(xs, ys, x_eval, grado_max=k_vis)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Evaluación Newton-Gregory",
            sugerencia_defecto="Verificá que todos los valores de (x, y) y x* sean reales finitos.",
            ejemplo_defecto="x* = 1.0",
        )
        return

    st.markdown(_construir_html_banner_recomendacion_ng(xs, x_eval, k_vis, grado_max), unsafe_allow_html=True)

    y_asc = float(res_asc["y_interpolado"])
    y_desc = float(res_desc["y_interpolado"])
    diff = abs(y_asc - y_desc)
    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"Ascendente Δ (k={k_vis})",
        f"{y_asc:.2f} req/s" if es_caso_real else f"{y_asc:.6f}",
        f"Ancla x₀={res_asc['x0']:g} | u={res_asc['u']:.3f}",
        delta_color="off",
    )
    m2.metric(
        f"Descendente ∇ (k={k_vis})",
        f"{y_desc:.2f} req/s" if es_caso_real else f"{y_desc:.6f}",
        f"Ancla xₙ={res_desc['xn']:g} | v={res_desc['v']:.3f}",
        delta_color="off",
    )
    if exacta_fn is not None:
        y_ex = float(exacta_fn(np.array([x_eval]))[0])
        sub_m3 = f"Exacta={y_ex:.4f} | Err Δ={abs(y_ex - y_asc):.2e}"
    elif k_vis >= grado_max:
        sub_m3 = "Unicidad exacta ✓ (k = máx)"
    else:
        sub_m3 = f"Truncado k={k_vis} (< {grado_max})"
    m3.metric("Diferencia |Δ − ∇|", f"{diff:.4f}" if es_caso_real else f"{diff:.6f}", sub_m3, delta_color="off")

    _render_grafica_gregory_comparativa(
        xs=xs,
        ys=ys,
        x_eval=x_eval,
        k_vis=k_vis,
        grado_max=grado_max,
        modo_visual=modo_visual,
        res_asc=res_asc,
        res_desc=res_desc,
        exacta_fn=exacta_fn,
        exacta_tex=exacta_tex,
        etiqueta_x=etiqueta_x,
        etiqueta_y=etiqueta_y,
    )

    tabla_arr = calcular_diferencias_ascendentes(ys)
    tabla_ng_html = _construir_html_tabla_escalonada_gregory(xs, tabla_arr, k_vis, modo_visual)
    html_asc, tex_asc = _construir_desglose_formula_lado(res_asc, "ascendente", k_vis, atenuado=(modo_visual == "Solo Descendente (∇)"))
    html_desc, tex_desc = _construir_desglose_formula_lado(res_desc, "descendente", k_vis, atenuado=(modo_visual == "Solo Ascendente (Δ)"))

    resumen_ng_md = [
        f"- **Diagnóstico de malla:** {diag['mensaje']}",
        f"- **$x^*$:** `{x_eval:g}` | **Grado truncado `k`:** `{k_vis}` (máximo `{grado_max}`)",
        f"- **Ascendente ($u = {float(res_asc['u']):.4f}$):** `P_{k_vis}({x_eval:g}) = {y_asc:.6f}`",
        f"- **Descendente ($v = {float(res_desc['v']):.4f}$):** `P_{k_vis}({x_eval:g}) = {y_desc:.6f}`",
        "**Tabla escalonada de diferencias finitas:**",
        html_tabla_a_markdown(tabla_ng_html),
        "**Polinomios sustituidos:**",
        f"$$\n{tex_asc}\n$$",
        f"$$\n{tex_desc}\n$$",
    ]
    registrar_contexto_ia("Newton-Gregory — Resultados y Diferencias Finitas", "\n".join(resumen_ng_md), clave="int_ng_res")

    if modo_detallado:
        st.markdown(_construir_html_resumen_gregory(res_asc, res_desc, x_eval, k_vis, grado_max, exacta_fn), unsafe_allow_html=True)
        st.markdown("#### Tabla Escalonada de Diferencias Finitas")
        st.caption(
            "Cada celda superior es Δᵏyᵢ = Δᵏ⁻¹yᵢ₊₁ − Δᵏ⁻¹yᵢ. "
            "Salvia marca la diagonal Ascendente (Δᵏy₀); Oro marca la diagonal Descendente (∇ᵏyₙ)."
        )
        st.markdown(tabla_ng_html, unsafe_allow_html=True)
        _render_inspector_celda_diferencia(xs, tabla_arr)

        st.markdown("#### Desglose Paso a Paso de Ambos Polinomios")
        col_asc, col_desc = st.columns(2)
        with col_asc:
            st.markdown(html_asc, unsafe_allow_html=True)
            with st.expander("Expresión algebraica sustituida Ascendente (LaTeX)"):
                st.latex(tex_asc)
        with col_desc:
            st.markdown(html_desc, unsafe_allow_html=True)
            with st.expander("Expresión algebraica sustituida Descendente (LaTeX)"):
                st.latex(tex_desc)

        pasos_ng_vivos = [
            f"**Equidistancia y parámetros ($h = {res_asc['h']:g}$):** $u = \\frac{{{x_eval:g} - {res_asc['x0']:g}}}{{{res_asc['h']:g}}} = {res_asc['u']:.6f}$ y $v = \\frac{{{x_eval:g} - {res_desc['xn']:g}}}{{{res_desc['h']:g}}} = {res_desc['v']:.6f}$.",
        ]
        for ta, td in zip(res_asc["terminos"], res_desc["terminos"]):
            pasos_ng_vivos.append(
                f"**Grado $k = {ta['k']}$:** "
                f"Ascendente `{ta['coef_binomial']:.6f}` $\\times \\Delta^{{{ta['k']}}}y_0$ (`{ta['diferencia']:.6g}`) $= {ta['termino']:.6f}$ (acum. `{ta['acumulado']:.6f}`) | "
                f"Descendente `{td['coef_binomial']:.6f}` $\\times \\nabla^{{{td['k']}}}y_n$ (`{td['diferencia']:.6g}`) $= {td['termino']:.6f}$ (acum. `{td['acumulado']:.6f}`)."
            )
        pasos_ng_vivos.append(
            f"**Conclusión en grado $k = {k_vis}$:** $P_{{{k_vis}}}^{{\\Delta}}({x_eval:g}) = {y_asc:.6f}$ y $P_{{{k_vis}}}^{{\\nabla}}({x_eval:g}) = {y_desc:.6f}$."
        )
        render_deduccion_paso_a_paso_viva(
            f"Newton-Gregory Ascendente y Descendente (k={k_vis})",
            pasos_ng_vivos,
            clave_ia="int_ng_viva",
            expanded=True,
        )

        if k_vis < grado_max:
            u_val = float(res_asc["u"])
            v_val = float(res_desc["v"])
            next_asc = float(tabla_arr[0, k_vis + 1])
            next_desc = float(tabla_arr[len(xs) - 1 - (k_vis + 1), k_vis + 1])
            comb_asc = math.prod(u_val - j for j in range(k_vis + 1)) / math.factorial(k_vis + 1)
            comb_desc = math.prod(v_val + j for j in range(k_vis + 1)) / math.factorial(k_vis + 1)
            st.info(
                f"Estimación de residuo por primer término omitido (orden k+1 = {k_vis + 1}): "
                f"|R_asc| ≈ {abs(comb_asc * next_asc):.6f} vs |R_desc| ≈ {abs(comb_desc * next_desc):.6f}."
            )


def _tab_newton_gregory() -> None:
    modo_detallado = render_toggle_modo_detallado("newton_gregory")
    meta_asc = render_bloque_1_teoria_y_ejemplo("newton_gregory_asc", modo_detallado=modo_detallado)
    if modo_detallado:
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
    _frag_newton_gregory(defaults, modo_detallado=modo_detallado)


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

_NOMBRES_CORTOS_GAUSS: dict[str, str] = {
    "gauss_progresiva": "Gauss Progresiva",
    "gauss_regresiva": "Gauss Regresiva",
    "ascendente": "N-G Ascendente (Δ)",
    "descendente": "N-G Descendente (∇)",
}

_SOLAR_X = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
_SOLAR_X_DESIG = [6.0, 8.0, 10.7, 12.0, 14.0, 16.0, 18.0]
_SOLAR_Y = [6.42, 30.94, 74.71, 103.17, 80.68, 34.25, 5.56]


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
        "ui_gauss_xs_txt",
        "ui_gauss_ys_txt",
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


def _al_aplicar_optimo_gauss(modo_opt: str, widget_key: str, state_key: str | None = None) -> None:
    st.session_state[widget_key] = modo_opt
    if state_key:
        st.session_state[state_key] = modo_opt


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
    etiqueta_x: str = "x",
    etiqueta_y: str = "P(x)",
) -> None:
    """Renderiza el gráfico Plotly interactivo destacando nodos usados vs omitidos y los 4 métodos."""
    n = len(xs)
    pad = max((max(xs) - min(xs)) * 0.15, 0.5)
    x_min = min(min(xs), x_eval) - pad * 0.4
    x_max = max(max(xs), x_eval) + pad * 0.4
    x_grid = np.linspace(x_min, x_max, 240)

    fig = crear_figura_base_plotly(
        f"Curva de Interpolación (k = {k_vis}) — Ruta {_NOMBRES_METODO_GAUSS[modo_activo]}",
        etiqueta_x,
        etiqueta_y,
        altura=430,
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
                name=f"{_NOMBRES_CORTOS_GAUSS[m_cur]} (k={k_vis})",
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
            name=f"Activo: {_NOMBRES_CORTOS_GAUSS[modo_activo]} (k={k_vis})",
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
def _frag_formula_gauss(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_solar", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Generación Solar Horaria (h = 2 h)"
            if op == "real_solar"
            else "Tabla Libre y Prácticas de Cátedra (TP5 / Runge / Exponencial)"
        ),
        horizontal=True,
        key="ui_gauss_contexto",
    )
    es_caso_real = contexto == "real_solar"

    error_parseo: Exception | None = None
    exacta_fn = None

    if es_caso_real:
        if "int_gauss_real_t" not in st.session_state:
            st.session_state["int_gauss_real_t"] = 12.7
        if "int_gauss_real_k" not in st.session_state:
            st.session_state["int_gauss_real_k"] = 2
        if "int_gauss_real_modo" not in st.session_state:
            st.session_state["int_gauss_real_modo"] = "gauss_progresiva"
        if "int_gauss_real_desig" not in st.session_state:
            st.session_state["int_gauss_real_desig"] = False

        claves_ui_real = ("ui_gauss_real_t", "ui_gauss_real_k", "ui_gauss_real_modo")
        st.caption(
            r"Planta solar cada $2\;\text{h}$: $t = [6, 8, 10, 12, 14, 16, 18]\;\text{h}$, "
            r"$y = [6.42, 30.94, 74.71, 103.17, 80.68, 34.25, 5.56]\;\text{kW}$ (ancla central $t_3 = 12\;\text{h}$)."
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("Centro-Der k=2 (t* = 12.7 h → Gauss gana)", key="btn_gauss_real_der", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_gauss_real_t": 12.7, "int_gauss_real_k": 2, "int_gauss_real_modo": "gauss_progresiva", "int_gauss_real_desig": False},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button("Centro-Izq k=3 (t* = 11.3 h → Gauss Reg. gana)", key="btn_gauss_real_izq", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_gauss_real_t": 11.3, "int_gauss_real_k": 3, "int_gauss_real_modo": "gauss_regresiva", "int_gauss_real_desig": False},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button("Romper paso (t₂=10.7 → pasa a Lagrange)", key="btn_gauss_real_desig", use_container_width=True):
            _aplicar_preset_interpolacion(
                {"int_gauss_real_t": 12.7, "int_gauss_real_k": 6, "int_gauss_real_modo": "gauss_progresiva", "int_gauss_real_desig": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns([1.15, 0.9, 1.35])
        x_eval = float(c1.slider("Hora a estimar t* (h):", 6.0, 18.0, float(st.session_state["int_gauss_real_t"]), step=0.1, key="ui_gauss_real_t"))
        k_vis = int(c2.slider("Grado truncamiento k:", 1, 6, int(st.session_state["int_gauss_real_k"]), key="ui_gauss_real_k"))
        modo_cur = str(st.session_state["int_gauss_real_modo"])
        idx_modo = _OPCIONES_METODO_GAUSS.index(modo_cur) if modo_cur in _OPCIONES_METODO_GAUSS else 0
        modo_activo = str(
            c3.selectbox(
                "Ruta trazada:",
                list(_OPCIONES_METODO_GAUSS),
                index=idx_modo,
                format_func=lambda k: _NOMBRES_METODO_GAUSS[k],
                key="ui_gauss_real_modo",
            )
        )
        ver_superposicion = st.checkbox("Superponer los 4 métodos truncados en el gráfico", value=True, key="ui_gauss_real_overlay")
        st.session_state["int_gauss_real_t"] = x_eval
        st.session_state["int_gauss_real_k"] = k_vis
        st.session_state["int_gauss_real_modo"] = modo_activo

        xs = list(_SOLAR_X_DESIG if st.session_state["int_gauss_real_desig"] else _SOLAR_X)
        ys = list(_SOLAR_Y)
        grado_max = len(xs) - 1
        preset = {"titulo": "Generación Solar Horaria (6 h a 18 h)", "exacta": None}
        etiqueta_x, etiqueta_y = "Hora t (h)", "Potencia (kW)"
        widget_opt_key, state_opt_key = "ui_gauss_real_modo", "int_gauss_real_modo"
    else:
        preset_id = st.session_state.get("int_gauss_preset", "tp5_gauss_centro")
        if preset_id not in _PRESETS_GAUSS:
            preset_id = "tp5_gauss_centro"
            st.session_state["int_gauss_preset"] = preset_id
        preset = _PRESETS_GAUSS[preset_id]
        exacta_fn = preset.get("exacta")

        if "int_gauss_x" not in st.session_state:
            st.session_state["int_gauss_x"] = list(defaults.get("x", preset["x"]))
        if "int_gauss_y" not in st.session_state:
            st.session_state["int_gauss_y"] = list(defaults.get("y", preset["y"]))
        if "int_gauss_x_eval" not in st.session_state:
            st.session_state["int_gauss_x_eval"] = float(defaults.get("x_eval", preset["x_eval"]))
        if "ui_gauss_metodo" not in st.session_state:
            st.session_state["ui_gauss_metodo"] = "gauss_progresiva"

        c_pre, c_met = st.columns([1.2, 1.3])
        c_pre.selectbox(
            "Escenario / Función de prueba:",
            list(_PRESETS_GAUSS.keys()),
            index=list(_PRESETS_GAUSS.keys()).index(preset_id),
            format_func=lambda k: _PRESETS_GAUSS[k]["titulo"],
            key="ui_gauss_preset",
            on_change=_al_cambiar_preset_gauss,
        )
        modo_activo = str(
            c_met.selectbox(
                "Método activo para trazar en tabla y gráfica:",
                list(_OPCIONES_METODO_GAUSS),
                format_func=lambda k: _NOMBRES_METODO_GAUSS[k],
                key="ui_gauss_metodo",
            )
        )
        if modo_detallado:
            render_practica_catedra_info(
                titulo_practica=preset["titulo"],
                enunciado_practica=preset["enunciado"],
                solucion_referencia=preset["referencia"],
            )

        if modo_detallado:
            col_ctrl, col_tbl = st.columns([1.4, 1.15])
            with col_tbl:
                st.markdown("#### Nodos de la Tabla")
                xs, ys = _editor_tabla_xy("int_gauss", False, col_x_label="Abscisa xᵢ", col_y_label="Ordenada yᵢ")
                try:
                    diag_prev = diagnosticar_espaciamiento_nodos(xs)
                    h_base = float(diag_prev["h_nominal"]) if diag_prev["es_equiespaciado"] and diag_prev["h_nominal"] else 0.3
                except Exception:
                    h_base = 0.3
                st.session_state.setdefault("ui_gauss_x0", float(xs[0]))
                st.session_state.setdefault("ui_gauss_dh", float(np.clip(abs(h_base), 0.05, 10.0)))
                st.session_state["ui_gauss_dh"] = float(np.clip(float(st.session_state["ui_gauss_dh"]), 0.05, 10.0))
                cx0, cdh = st.columns(2)
                nuevo_x0 = cx0.number_input("Inicio x₀:", step=0.1, format="%.4f", key="ui_gauss_x0")
                nuevo_h = cdh.number_input("Paso Δh:", min_value=0.05, max_value=10.0, step=0.05, format="%.4f", key="ui_gauss_dh")
                cb1, cb2, cb3 = st.columns([1.3, 1.0, 1.0])
                if cb1.button("Regenerar malla Δh", key="btn_gauss_dh", use_container_width=True):
                    _aplicar_regeneracion_malla_eq("int_gauss", "ui_gauss_xe_slider", xs, ys, float(nuevo_x0), float(nuevo_h), preset)
                    rerun_fragmento_seguro()
                if cb2.button("+ Nodo", key="btn_gauss_add_node", disabled=len(xs) >= 10, use_container_width=True):
                    x_next = round(xs[-1] + h_base, 6)
                    if preset.get("exacta") is not None:
                        y_next = round(float(preset["exacta"](np.array([x_next]))[0]), 6)
                    else:
                        grado_fit = min(2, len(xs) - 1)
                        coef = np.polyfit(xs[-grado_fit - 1 :], ys[-grado_fit - 1 :], grado_fit)
                        y_next = round(float(np.polyval(coef, x_next)), 6)
                    st.session_state["int_gauss_x"] = list(xs) + [x_next]
                    st.session_state["int_gauss_y"] = list(ys) + [y_next]
                    st.session_state.pop("int_gauss__editor_df", None)
                    rerun_fragmento_seguro()
                if cb3.button("- Nodo", key="btn_gauss_rem_node", disabled=len(xs) <= 2, use_container_width=True):
                    st.session_state["int_gauss_x"] = list(xs[:-1])
                    st.session_state["int_gauss_y"] = list(ys[:-1])
                    st.session_state.pop("int_gauss__editor_df", None)
                    rerun_fragmento_seguro()
            ctrl_ctx = col_ctrl
        else:
            cx, cy = st.columns(2)
            xs_cur = list(st.session_state["int_gauss_x"])
            ys_cur = list(st.session_state["int_gauss_y"])
            xs_txt = cx.text_input("Abscisas equidistantes x (separadas por coma):", value=", ".join(f"{v:g}" for v in xs_cur), key="ui_gauss_xs_txt")
            ys_txt = cy.text_input("Ordenadas y (separadas por coma):", value=", ".join(f"{v:g}" for v in ys_cur), key="ui_gauss_ys_txt")
            try:
                xs_p = _parsear_lista_interpolacion(xs_txt, "abscisas x")
                ys_p = _parsear_lista_interpolacion(ys_txt, "ordenadas y")
                if len(xs_p) != len(ys_p):
                    raise ValueError(f"Cantidad desigual: {len(xs_p)} abscisas x y {len(ys_p)} ordenadas y.")
                xs, ys = xs_p, ys_p
                st.session_state["int_gauss_x"] = xs
                st.session_state["int_gauss_y"] = ys
            except Exception as exc:
                xs, ys = xs_cur, ys_cur
                error_parseo = exc
            ctrl_ctx = st.container()

        with ctrl_ctx:
            x_eval_ini = float(st.session_state["int_gauss_x_eval"])
            if "ui_gauss_xe" in st.session_state and abs(float(st.session_state["ui_gauss_xe"]) - x_eval_ini) > 1e-9:
                _sincronizar_xe_gauss(x_eval_ini, xs)

            n_pts = len(xs)
            grado_max = max(1, n_pts - 1)
            h_btn = float(xs[1] - xs[0]) if n_pts >= 2 and abs(xs[1] - xs[0]) > 1e-12 else 0.3
            idx_c_prog = (n_pts - 1) // 2
            idx_c_reg = n_pts // 2

            if modo_detallado:
                qb1, qb2 = st.columns(2)
                qb3, qb4 = st.columns(2)
            else:
                qb1, qb2, qb3, qb4 = st.columns(4)
            qb1.button("Inicio (cerca x₀)", key="btn_gauss_p_ini", use_container_width=True, on_click=_sincronizar_xe_gauss, args=(round(xs[0] + 0.35 * h_btn, 6), xs))
            qb2.button("Centro-Izq (u = -0.35)", key="btn_gauss_p_cizq", use_container_width=True, on_click=_sincronizar_xe_gauss, args=(round(xs[idx_c_reg] - 0.35 * h_btn, 6), xs))
            qb3.button("Centro-Der (u = +0.35)", key="btn_gauss_p_cder", use_container_width=True, on_click=_sincronizar_xe_gauss, args=(round(xs[idx_c_prog] + 0.35 * h_btn, 6), xs))
            qb4.button("Final (cerca xₙ)", key="btn_gauss_p_fin", use_container_width=True, on_click=_sincronizar_xe_gauss, args=(round(xs[-1] - 0.35 * h_btn, 6), xs))

            x_min, x_max = min(xs), max(xs)
            span = max(abs(x_max - x_min), 1.0)
            pad = round(max(0.35, 0.15 * span), 4)
            s_min, s_max = round(x_min - pad, 4), round(x_max + pad, 4)
            step_s = max(round(span / 160.0, 4), 0.001)
            x_eval_cur = float(st.session_state.get("int_gauss_x_eval", float(preset["x_eval"])))

            st.session_state.setdefault("ui_gauss_xe", float(x_eval_cur))
            if "ui_gauss_xe_slider" not in st.session_state:
                st.session_state["ui_gauss_xe_slider"] = float(np.clip(x_eval_cur, s_min, s_max))
            else:
                st.session_state["ui_gauss_xe_slider"] = float(np.clip(st.session_state["ui_gauss_xe_slider"], s_min, s_max))

            cx1, cx2 = st.columns([2.0, 1.0])
            cx1.slider("Deslizar x* a lo largo de la tabla:", min_value=float(s_min), max_value=float(s_max), step=float(step_s), key="ui_gauss_xe_slider", on_change=_al_cambiar_slider_xe_gauss)
            cx2.number_input("Valor exacto de x*:", format="%.4f", key="ui_gauss_xe", on_change=_al_cambiar_input_xe_gauss)
            x_eval = float(st.session_state["int_gauss_x_eval"])

            k_default = min(2, grado_max)
            if "vis_gauss_k" in st.session_state:
                st.session_state["vis_gauss_k"] = min(max(int(st.session_state["vis_gauss_k"]), 0), grado_max)
            else:
                st.session_state["vis_gauss_k"] = k_default

            ck1, ck2 = st.columns([1.2, 1.1])
            k_vis = int(ck1.slider("Grado de truncamiento k:", min_value=0, max_value=grado_max, key="vis_gauss_k"))
            ver_superposicion = bool(ck2.checkbox("Superponer los 4 métodos en el gráfico", value=True, key="ui_gauss_overlay"))

        etiqueta_x, etiqueta_y = "x", "P(x)"
        widget_opt_key, state_opt_key = "ui_gauss_metodo", None

    try:
        if error_parseo is not None:
            raise error_parseo
        diag = diagnosticar_espaciamiento_nodos(xs)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos Gauss",
            sugerencia_defecto="Ingresá al menos 2 nodos reales finitos sin valores repetidos ni longitudes desiguales.",
            ejemplo_defecto="x = 0.2, 0.5, 0.8, 1.1, 1.4",
        )
        return

    if not diag["es_equiespaciado"]:
        st.warning(f"**{diag['mensaje']}** Se activa automáticamente **Lagrange** como respaldo válido para paso irregular.")
        try:
            res_lag = interpolar_lagrange(xs, ys, x_eval)
            ml1, ml2, ml3 = st.columns(3)
            val_lag = f"{res_lag['y_interpolado']:.2f} kW" if es_caso_real else f"{res_lag['y_interpolado']:.6f}"
            ml1.metric("Respaldo Lagrange P(x*)", val_lag, "Válido en malla irregular", delta_color="off")
            ml2.metric("Paso entre nodos", "Irregular ⚠", "Gauss / Gregory invalidados", delta_color="off")
            ml3.metric("Grado del respaldo", f"Grado {res_lag['grado']}", f"{res_lag['puntos']} nodos usados", delta_color="off")

            x_grid = np.linspace(min(min(xs), x_eval) - 0.1, max(max(xs), x_eval) + 0.1, 220)
            y_lag_grid = limpiar_valores_plotly(evaluar_curva_lagrange(xs, ys, x_grid), limite_abs=1e9)
            fig = crear_figura_base_plotly("Nodos No Equidistantes → Respaldo Polinomio de Lagrange", etiqueta_x, etiqueta_y, altura=330)
            fig.add_trace(go.Scatter(x=x_grid.tolist(), y=y_lag_grid, mode="lines", name="Lagrange P(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.6)))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos actuales", marker=dict(size=9, color=PALETA_ARANDU["terracota"])))
            fig.add_trace(
                go.Scatter(
                    x=[x_eval],
                    y=[res_lag["y_interpolado"]],
                    mode="markers",
                    name="P(x*)",
                    marker=dict(size=14, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
                )
            )
            render_plotly_arandu(fig, key="plot_int_gauss_fallback")
        except Exception as exc:
            render_excepcion_validacion(
                exc,
                campo="Tabla de nodos (respaldo Lagrange)",
                sugerencia_defecto="Eliminá abscisas x repetidas e ingresá al menos 2 pares finitos.",
                ejemplo_defecto="x = 0.2, 0.5, 0.8, 1.1, 1.4",
            )
        return

    try:
        rec = recomendar_metodo_equidistante(xs, x_eval)
        resultados = {
            "ascendente": interpolar_newton_gregory_ascendente(xs, ys, x_eval),
            "descendente": interpolar_newton_gregory_descendente(xs, ys, x_eval),
            "gauss_progresiva": interpolar_gauss_progresiva(xs, ys, x_eval),
            "gauss_regresiva": interpolar_gauss_regresiva(xs, ys, x_eval),
        }
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Evaluación Fórmula de Gauss",
            sugerencia_defecto="Verificá que (x, y) y x* sean valores reales finitos sobre malla equiespaciada.",
            ejemplo_defecto="x* = 0.90",
        )
        return

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
        args=(rec["modo"], widget_opt_key, state_opt_key),
    )

    if exacta_fn is not None and not es_caso_real and modo_detallado:
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
    val_act = float(res_activo["terminos"][k_vis]["acumulado"])
    u_lbl = "v" if modo_activo == "descendente" else "u"
    u_val = float(res_activo["v"] if modo_activo == "descendente" else res_activo["u"])
    x_anc = float(res_activo.get("x0_central", res_activo.get("x0", res_activo.get("xn", xs[0]))))

    def _fmt_g(v: float) -> str:
        return f"{v:.2f} kW" if es_caso_real else f"{v:.6f}"

    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"{_NOMBRES_CORTOS_GAUSS[modo_activo]} (k={k_vis})",
        _fmt_g(val_act),
        f"Ancla {x_anc:g} | {u_lbl}={u_val:+.3f}",
        delta_color="off",
    )
    m2.metric(
        "Exacta f(x*)" if ref_es_exacta else f"Curva completa (k={grado_max})",
        _fmt_g(y_ref),
        f"Desvío activo: {_fmt_g(abs(y_ref - val_act))}" if (k_vis < grado_max or ref_es_exacta) else "Unicidad exacta ✓",
        delta_color="off",
    )
    if k_vis >= grado_max:
        spread = max(float(resultados[c]["y_interpolado"]) for c in _OPCIONES_METODO_GAUSS) - min(
            float(resultados[c]["y_interpolado"]) for c in _OPCIONES_METODO_GAUSS
        )
        m3.metric("Diferencia entre métodos", f"{spread:.4f}" if es_caso_real else f"{spread:.6f}", "Los 4 coinciden ✓ (k = máx)", delta_color="off")
    elif modo_activo != rec["modo"]:
        val_rec = float(resultados[rec["modo"]]["terminos"][k_vis]["acumulado"])
        m3.metric(
            f"Óptimo ({_NOMBRES_CORTOS_GAUSS[rec['modo']]})",
            _fmt_g(val_rec),
            f"Desvío óptimo: {_fmt_g(abs(y_ref - val_rec))}",
            delta_color="off",
        )
    else:
        alt_modo = max(("ascendente", "descendente"), key=lambda c: abs(y_ref - float(resultados[c]["terminos"][k_vis]["acumulado"])))
        val_alt = float(resultados[alt_modo]["terminos"][k_vis]["acumulado"])
        m3.metric(
            f"Contraste ({_NOMBRES_CORTOS_GAUSS[alt_modo]})",
            _fmt_g(val_alt),
            f"Desvío borde: {_fmt_g(abs(y_ref - val_alt))}",
            delta_color="off",
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
        etiqueta_x=etiqueta_x,
        etiqueta_y=etiqueta_y,
    )

    tabla_arr = np.asarray(res_activo["tabla_diferencias"], dtype=float)
    tabla_asc_arr = calcular_diferencias_ascendentes(ys) if modo_activo == "descendente" else tabla_arr
    comp_html = _construir_html_comparativa_4metodos(
        resultados,
        k_vis,
        grado_max,
        modo_activo,
        rec["modo"],
        y_ref,
        ref_es_exacta,
    )
    tabla_html, tira_html = _construir_html_tabla_gauss_zigzag(xs, tabla_asc_arr, k_vis, modo_activo, res_activo)
    html_card, tex_pol = _construir_desglose_formula_lado(res_activo, modo_activo, k_vis, atenuado=False)

    tira_limpia = html_tabla_a_markdown(tira_html)
    resumen_gauss_md = [
        f"- **Escenario:** {preset['titulo']}",
        f"- **Método activo:** {_NOMBRES_METODO_GAUSS[modo_activo]}",
        f"- **Recomendación óptima para $x^* = {x_eval:g}$:** {_NOMBRES_METODO_GAUSS[rec['modo']]} ({rec['motivo']})",
        f"- **Grado de truncamiento:** `k = {k_vis}` (máximo `n - 1 = {grado_max}`)",
        f"- **{tira_limpia}**",
        "**Comparativa de los 4 métodos:**",
        html_tabla_a_markdown(comp_html),
        "**Tabla escalonada de diferencias finitas:**",
        html_tabla_a_markdown(tabla_html),
        "**Expresión sustituida del método activo (LaTeX):**",
        f"$$\n{tex_pol}\n$$",
    ]
    registrar_contexto_ia("Fórmula de Gauss — Comparativa, Zig-Zag y Desarrollo", "\n".join(resumen_gauss_md), clave="int_gauss_res")

    if modo_detallado:
        st.markdown(f"**Comparativa Simultánea en Grado $k = {k_vis}$ (vs. Grado Completo $n - 1 = {grado_max}$)**")
        st.markdown(comp_html, unsafe_allow_html=True)

        st.markdown("**Tabla de Diferencias Finitas Escalonada y Trazado en Zig-Zag**")
        st.markdown(tabla_html, unsafe_allow_html=True)
        st.markdown(tira_html, unsafe_allow_html=True)
        _render_inspector_celda_diferencia(xs, tabla_asc_arr)

        st.markdown("**Descomposición Término a Término del Polinomio Seleccionado**")
        st.markdown(html_card, unsafe_allow_html=True)
        with st.expander("Expresión algebraica sustituida del método activo (LaTeX)"):
            st.latex(tex_pol)

        pasos_gauss_vivos = [
            f"**Anclaje y parámetro adimensional ({_NOMBRES_METODO_GAUSS[modo_activo]}):** Con paso $h = {res_activo['h']:g}$ y nodo ancla $x_a = {x_anc:g}$, obtenemos $\\frac{{{x_eval:g} - {x_anc:g}}}{{{res_activo['h']:g}}} = {u_val:.6f}$. {rec['motivo']}",
        ]
        for tg in res_activo["terminos"][: k_vis + 1]:
            df_txt = f" (desplazamiento `{tg['delta_factor']}`)" if tg.get("delta_factor") is not None else ""
            pasos_gauss_vivos.append(
                f"**Orden $k = {tg['k']}$:** Coeficiente `{tg['coef_binomial']:.6f}`{df_txt} $\\times$ diferencia `{tg['diferencia']:.6g}` "
                f"$= {tg['termino']:.6f} \\implies P_{{{tg['k']}}}({x_eval:g}) = {tg['acumulado']:.6f}$."
            )
        render_deduccion_paso_a_paso_viva(
            f"{_NOMBRES_METODO_GAUSS[modo_activo]} en Zig-Zag (k={k_vis})",
            pasos_gauss_vivos,
            clave_ia="int_gauss_viva",
            expanded=True,
        )


def _tab_formula_gauss() -> None:
    modo_detallado = render_toggle_modo_detallado("formula_gauss")
    meta_g = render_bloque_1_teoria_y_ejemplo("formula_gauss", modo_detallado=modo_detallado)
    if modo_detallado:
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
    _frag_formula_gauss(defaults, modo_detallado=modo_detallado)


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
