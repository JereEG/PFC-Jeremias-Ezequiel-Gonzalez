"""
Vista de Derivación e Integración Numérica - Arandú.
Incluye Diferenciación (Gregory + punto libre u=0.4) y Motor Adaptativo en Vivo
para Trapecio, Simpson 1/3 y Simpson 3/8 ante cambios de Δh o paridad n.
Arquitectura Static-First con aislamiento mediante @st.fragment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.integracion_derivacion import (
    derivada_newton_gregory_ascendente,
    derivada_newton_gregory_descendente,
    integrar_adaptativo_en_vivo,
    remuestrear_tabla_integracion,
)
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    limpiar_valores_plotly,
    render_bloque_1_teoria_y_ejemplo,
    render_controles_edicion_practica,
    render_dataframe_acotado,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
    rerun_fragmento_seguro,
)


@st.fragment
def _frag_diferenciacion(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="int_deriv",
        defaults_dict=defaults,
        mensaje_edicion=r"Edita la tabla $(x_i, y_i)$, alterna el sentido de diferencias o cambia $x^*$ (incluso fuera de nodos tabulados).",
    )
    solo_lectura = modo == "catedra"

    df_in = pd.DataFrame({"x": st.session_state["int_deriv_x"], "y": st.session_state["int_deriv_y"]})
    df_edit = st.data_editor(
        df_in,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic" if not solo_lectura else "fixed",
        disabled=solo_lectura,
        key="int_deriv__editor_df",
    )
    if not solo_lectura:
        df_c = df_edit.dropna()
        if len(df_c) >= 3:
            st.session_state["int_deriv_x"] = df_c["x"].astype(float).tolist()
            st.session_state["int_deriv_y"] = df_c["y"].astype(float).tolist()
    xs = st.session_state["int_deriv_x"]
    ys = st.session_state["int_deriv_y"]

    c1, c2, c3 = st.columns(3)
    esquema = c1.radio("Diferencias:", ["Progresivas (Ascendente)", "Regresivas (Descendente)"], horizontal=True, key="ui_der_esq")
    orden = c2.radio("Orden:", [1, 2], format_func=lambda o: "1ra Derivada f'(x)" if o == 1 else "2da Derivada f''(x)", horizontal=True, key="ui_der_ord")
    x_eval = c3.number_input("Punto $x^*$ a derivar (puede ser no tabular):", value=float(st.session_state["int_deriv_x_eval"]), format="%.4f", disabled=solo_lectura, key="ui_der_xe")
    if not solo_lectura:
        st.session_state["int_deriv_x_eval"] = x_eval

    st.markdown("### Bloque 3 — Cálculo de Derivada en Vivo")
    try:
        if esquema.startswith("Progresivas"):
            idx_base = max(0, min(len(xs) - 2, int(np.searchsorted(xs, x_eval) - 1)))
            res = derivada_newton_gregory_ascendente(xs, ys, idx_punto=idx_base, orden_derivada=orden, x_eval_libre=x_eval)
            param = res["u"]
            param_name = "u"
        else:
            idx_base = max(1, min(len(xs) - 1, int(np.searchsorted(xs, x_eval))))
            res = derivada_newton_gregory_descendente(xs, ys, idx_punto=idx_base, orden_derivada=orden, x_eval_libre=x_eval)
            param = res["v"]
            param_name = "v"
    except Exception as exc:
        st.error(str(exc))
        return

    m1, m2, m3 = st.columns(3)
    m1.metric(f"f{'´' if orden == 1 else '´´'}({x_eval:g})", f"{res['derivada']:.6f}")
    m2.metric(f"Parámetro {param_name}", f"{param:.4f}")
    m3.metric("Paso h", f"{res['h']:.4g}")

    render_dataframe_acotado(res["terminos_detalle"], max_filas=300, key_prefix="tbl_int_deriv")

    st.markdown("### Bloque 4 — Pendiente Tangente en $x^*$")
    y_en_x = float(np.interp(x_eval, xs, ys))
    fig = crear_figura_base_plotly(f"Derivada Numérica en x* = {x_eval:g}", "x", "y")
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="Datos f(x)",
            line=dict(color="#D4A359", width=2.5),
            marker=dict(size=7, color="#D4A359"),
        )
    )
    if orden == 1:
        dx = res["h"] * 1.2
        fig.add_trace(
            go.Scatter(
                x=[x_eval - dx, x_eval + dx],
                y=[y_en_x - res["derivada"] * dx, y_en_x + res["derivada"] * dx],
                mode="lines",
                name=f"Recta tangente (m={res['derivada']:.4f})",
                line=dict(color="#C86D44", width=2.5, dash="dash"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[x_eval],
            y=[y_en_x],
            mode="markers",
            name="Punto x*",
            marker=dict(size=13, color="#FFFDF9", symbol="star", line=dict(color="#B5673F", width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_int_deriv")


def _tab_diferenciacion() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("diferenciacion_numerica")
    practica = meta.get("practica_profesora", {})
    defaults = {
        "x": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "y": [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151],
        "x_eval": 12.4,
        **practica.get("datos_iniciales", {}),
    }
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_diferenciacion(defaults)


@st.fragment
def _frag_integracion(clave_state: str, metodo_solicitado: str, defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state=clave_state,
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica los puntos tabulados o ajusta el paso $\Delta h$ y la cantidad de subintervalos $n$.",
    )
    solo_lectura = modo == "catedra"

    col_tbl, col_ctrl = st.columns([1.4, 1.3])
    with col_tbl:
        df_in = pd.DataFrame({"x": st.session_state[f"{clave_state}_x"], "y": st.session_state[f"{clave_state}_y"]})
        df_edit = st.data_editor(
            df_in,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic" if not solo_lectura else "fixed",
            disabled=solo_lectura,
            key=f"{clave_state}__editor_df",
        )
        if not solo_lectura:
            df_c = df_edit.dropna()
            if len(df_c) >= 2:
                st.session_state[f"{clave_state}_x"] = df_c["x"].astype(float).tolist()
                st.session_state[f"{clave_state}_y"] = df_c["y"].astype(float).tolist()

    with col_ctrl:
        f_str = st.text_input(
            "Función $f(x)$ de referencia (opcional para remuestreo y error exacto):",
            value=str(st.session_state[f"{clave_state}_f_str"]),
            disabled=solo_lectura,
            key=f"ui_{clave_state}_fstr",
        )
        if not solo_lectura:
            st.session_state[f"{clave_state}_f_str"] = f_str

        xs_act = st.session_state[f"{clave_state}_x"]
        h_actual = float(xs_act[1] - xs_act[0]) if len(xs_act) >= 2 else 1.0
        n_actual = max(1, len(xs_act) - 1)

        st.markdown(r"**Ajuste de Paso $\Delta h$ y Subintervalos $n$ en Vivo:**")
        nuevo_h = st.number_input(
            r"Paso $\Delta h$ deseado (prueba cambiarlo para observar adaptación de paridad):",
            value=float(round(h_actual, 4)),
            min_value=0.1,
            step=0.5,
            disabled=solo_lectura,
            key=f"ui_{clave_state}_dh",
        )
        nuevo_n = st.slider(
            "O fijar cantidad de subintervalos $n$:",
            min_value=2,
            max_value=24,
            value=n_actual,
            disabled=solo_lectura,
            key=f"ui_{clave_state}_n",
        )
        estrategia_impar = st.selectbox(
            "Estrategia si hay paridad incompatible:",
            options=["simpson_3_8", "trapecio"],
            format_func=lambda s: "Combinar Simpson 3/8 + 1/3" if s == "simpson_3_8" else "Combinar Simpson 1/3 + Trapecio",
            key=f"ui_{clave_state}_est",
        )

        b1, b2 = st.columns(2)
        if not solo_lectura and b1.button("Aplicar Δh", key=f"btn_dh_{clave_state}", use_container_width=True):
            xn, yn, _ = remuestrear_tabla_integracion(defaults["x"], defaults["y"], nuevo_h=nuevo_h, f_str=f_str or None)
            st.session_state[f"{clave_state}_x"] = xn
            st.session_state[f"{clave_state}_y"] = yn
            if f"{clave_state}__editor_df" in st.session_state:
                del st.session_state[f"{clave_state}__editor_df"]
            rerun_fragmento_seguro()
        if not solo_lectura and b2.button("Aplicar n subinterv.", key=f"btn_n_{clave_state}", use_container_width=True):
            xn, yn, _ = remuestrear_tabla_integracion(defaults["x"], defaults["y"], nuevo_n=nuevo_n, f_str=f_str or None)
            st.session_state[f"{clave_state}_x"] = xn
            st.session_state[f"{clave_state}_y"] = yn
            if f"{clave_state}__editor_df" in st.session_state:
                del st.session_state[f"{clave_state}__editor_df"]
            rerun_fragmento_seguro()

    xs = st.session_state[f"{clave_state}_x"]
    ys = st.session_state[f"{clave_state}_y"]

    st.markdown("### Bloque 3 — Integración Adaptativa y Diagnóstico en Vivo")
    try:
        res = integrar_adaptativo_en_vivo(
            xs,
            ys,
            metodo_solicitado=metodo_solicitado,
            estrategia_impar=estrategia_impar,
            f_str=f_str.strip() or None,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    for aviso in res["avisos_pedagogicos"]:
        if res["hubo_adaptacion"] or res["multitramo"]:
            st.warning(aviso)
        else:
            st.success(aviso)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Área Total Estimada", f"{res['area_total']:.6f}")
    m2.metric("Reglas Aplicadas", res["reglas_aplicadas"])
    m3.metric("Subintervalos (n)", res["n_intervalos_total"])
    if res["exacta"]:
        m4.metric("Integral Exacta", f"{res['exacta']['integral_exacta']:.6f}", f"Ea {res['exacta']['error_absoluto']:.2e}")
    else:
        m4.metric("Exacta", "No provista")

    filas_bloques = [
        {
            "Bloque": i + 1,
            "Regla": b["regla"],
            "Intervalo": f"[{b['x_nodos'][0]:g}, {b['x_nodos'][-1]:g}]",
            "h": round(b["h"], 6),
            "Fórmula": b["formula"],
            "Área": round(b["area"], 6),
        }
        for i, b in enumerate(res["bloques"])
    ]
    render_dataframe_acotado(filas_bloques, max_filas=300, key_prefix=f"tbl_{clave_state}_blq")

    st.markdown("### Bloque 4 — Áreas Sombreadas por Sub-bloque Paso a Paso")
    bloques = res["bloques"]
    k_vis = render_slider_paso_k(
        "Ver hasta el bloque elemental $k$:",
        len(bloques),
        key=f"vis_{clave_state}_k",
    )

    fig = crear_figura_base_plotly(
        f"Reglas Aplicadas: {res['reglas_aplicadas']} ({k_vis}/{len(bloques)} bloques)",
        "x",
        "f(x)",
    )
    fill_por_regla = {
        "Trapecio": "rgba(200, 109, 68, 0.25)",
        "Simpson 1/3": "rgba(122, 154, 101, 0.25)",
        "Simpson 3/8": "rgba(212, 163, 89, 0.25)",
    }
    for idx, b in enumerate(bloques[:k_vis], start=1):
        fig.add_trace(
            go.Scatter(
                x=b["x_curva"],
                y=limpiar_valores_plotly(b["y_curva"], limite_abs=1e9),
                mode="lines",
                fill="tozeroy",
                fillcolor=fill_por_regla.get(b["regla"], "rgba(122, 154, 101, 0.25)"),
                name=f"B{idx}: {b['regla']} (A={b['area']:.3f})",
                line=dict(color=b["color"], width=2.4),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Nodos tabla",
            marker=dict(size=9, color="#FFFDF9", line=dict(color="#B5673F", width=1.8)),
        )
    )
    render_plotly_arandu(fig, key=f"plot_{clave_state}")


def _render_integracion_tab(clave_catalogo: str, clave_state: str, metodo_solicitado: str) -> None:
    meta = render_bloque_1_teoria_y_ejemplo(clave_catalogo)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "x": [0.5, 5.5, 10.5, 15.5, 20.5, 25.5],
            "y": [0.05, 6.05, 22.05, 48.05, 84.05, 130.05],
            "f_str": "x**2 / 5",
        },
    )
    if "f_str" not in defaults:
        defaults = {**defaults, "f_str": "x**2 / 5"}
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_integracion(clave_state, metodo_solicitado, defaults)


def vista_integracion() -> None:
    inject_arandu_css()
    st.title("Integración y Derivación Numérica")
    st.markdown(
        r"Diferenciación numérica mediante Gregory y cuadratura compuesta ($O(h^2)$ a $O(h^4)$) "
        r"con adaptación automática ante cambios de $\Delta h$ o paridad $n$."
    )
    activo = render_selector_fichas_metodos(
        "integracion",
        (
            {
                "numero": "1",
                "clave_svg": "diferenciacion",
                "titulo": "Diferenciación (Gregory)",
                "subtitulo": "GREGORY · DERIVADAS",
                "resumen": "Estima f'(x*) y f''(x*) tanto en puntos tabulares como con paso libre u.",
                "pista": "Diferencias · 1ra y 2da derivada",
            },
            {
                "numero": "2",
                "clave_svg": "trapecios",
                "titulo": "Regla de Trapecios",
                "subtitulo": "POLIGONAL · ORDEN O(h²)",
                "resumen": "Aproxima el área uniendo pares consecutivos con segmentos rectos.",
                "pista": "Poligonal recta · Cualquier n",
            },
            {
                "numero": "3",
                "clave_svg": "simpson_13",
                "titulo": "Regla de Simpson 1/3",
                "subtitulo": "PARÁBOLAS · ORDEN O(h⁴)",
                "resumen": "Ajusta arcos parabólicos cada dos subintervalos para máxima precisión.",
                "pista": "Parábolas · Requiere n par",
            },
            {
                "numero": "4",
                "clave_svg": "simpson_38",
                "titulo": "Simpson 3/8 y Combinado",
                "subtitulo": "CÚBICAS · PARIDAD MIXTA",
                "resumen": "Aplica arcos cúbicos o combina esquemas automáticamente si n es impar.",
                "pista": "Cúbicas 3/8 · Paridad adaptable",
            },
        ),
    )
    if activo == 0:
        _tab_diferenciacion()
    elif activo == 1:
        _render_integracion_tab("trapecio", "int_trap", "trapecio")
    elif activo == 2:
        _render_integracion_tab("simpson_1_3", "int_s13", "simpson_1_3")
    else:
        _render_integracion_tab("simpson_3_8_combinado", "int_s38", "simpson_3_8")


vista_integracion()
