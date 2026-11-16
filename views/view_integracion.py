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

from core.integracion import integrar_paso_variable
from core.integracion_derivacion import (
    derivada_newton_gregory_ascendente,
    derivada_newton_gregory_descendente,
    integrar_adaptativo_en_vivo,
    remuestrear_tabla_integracion,
)
from views.theme import PALETA_ARANDU
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


@st.fragment
def _frag_diferenciacion(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="int_deriv",
        defaults_dict=defaults,
        mensaje_edicion=r"Edita la tabla $(x_i, y_i)$, alterna el sentido de diferencias o cambia $x^*$ (incluso fuera de nodos tabulados).",
    )
    solo_lectura = modo == "catedra"

    df_in = pd.DataFrame({"x": st.session_state["int_deriv_x"], "y": st.session_state["int_deriv_y"]})
    df_edit = render_data_editor_arandu(
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
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos o punto x* de derivación",
            sugerencia_defecto="Ingresá nodos x crecientes equidistantes (h > 0) y un punto x* dentro o próximo a la tabla.",
            ejemplo_defecto="x = [0.0, 0.2, 0.4, 0.6], x* = 0.2",
        )
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
            line=dict(color=PALETA_ARANDU["oro"], width=2.5),
            marker=dict(size=7, color=PALETA_ARANDU["oro"]),
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
                line=dict(color=PALETA_ARANDU["terracota"], width=2.5, dash="dash"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[x_eval],
            y=[y_en_x],
            mode="markers",
            name="Punto x*",
            marker=dict(size=13, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
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
        df_edit = render_data_editor_arandu(
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
        render_subtitulo_controles_vivo()
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
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos o f(x) de integración",
            sugerencia_defecto="Asegurá al menos 2 nodos crecientes sin repetir en x y sintaxis Python válida si activaste f(x).",
            ejemplo_defecto="x = [0.0, 0.25, 0.5, 0.75, 1.0], f(x) = exp(x)",
        )
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
        "Trapecio": PALETA_ARANDU["terracota_fill"],
        "Simpson 1/3": PALETA_ARANDU["salvia_fill"],
        "Simpson 3/8": PALETA_ARANDU["oro_fill"],
    }
    color_por_regla = {
        "Trapecio": PALETA_ARANDU["terracota"],
        "Simpson 1/3": PALETA_ARANDU["salvia"],
        "Simpson 3/8": PALETA_ARANDU["oro"],
    }
    dash_por_regla = {
        "Trapecio": "dash",
        "Simpson 1/3": "solid",
        "Simpson 3/8": "dot",
    }
    for idx, b in enumerate(bloques[:k_vis], start=1):
        fig.add_trace(
            go.Scatter(
                x=b["x_curva"],
                y=limpiar_valores_plotly(b["y_curva"], limite_abs=1e9),
                mode="lines",
                fill="tozeroy",
                fillcolor=fill_por_regla.get(b["regla"], PALETA_ARANDU["salvia_fill"]),
                name=f"B{idx}: {b['regla']} (A={b['area']:.3f})",
                line=dict(
                    color=color_por_regla.get(b["regla"], PALETA_ARANDU["salvia"]),
                    width=2.4,
                    dash=dash_por_regla.get(b["regla"], "solid"),
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Nodos tabla",
            marker=dict(size=9, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
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


_LAB_SUBINDICES = ("₁", "₂", "₃", "₄", "₅", "₆")
_LAB_NODOS = ("x₀", "x₁", "x₂", "x₃", "x₄", "x₅", "x₆")


def _f_lab_paso_variable(x: float | np.ndarray) -> float | np.ndarray:
    return np.sin(x) + 0.5 * x + 1.5


def _exacta_lab_paso_variable(a: float, b: float) -> float:
    def _primitiva(t: float) -> float:
        return float(-np.cos(t) + 0.25 * (t**2) + 1.5 * t)

    return _primitiva(b) - _primitiva(a)


@st.fragment
def _frag_laboratorio_paso_variable() -> None:
    st.markdown("## Laboratorio de Paso Variable (Δh): ¿Cuándo usar Trapecios o Simpson?")
    st.markdown(
        r"Modifica independientemente las longitudes de los 6 subintervalos $\Delta h_1 \dots \Delta h_6$ "
        r"sobre $f(x) = \sin(x) + 0.5x + 1.5$ desde $x_0 = 0$. "
        r"El motor prioriza **Simpson** ($1/3$ o $3/8$) en bloques contiguos de paso uniforme y cae en **Trapecio** ante cada variación de $\Delta h$."
    )

    for idx in range(1, 7):
        if f"lab_dh_{idx}" not in st.session_state:
            st.session_state[f"lab_dh_{idx}"] = 0.5

    if st.button("Igualar todos los pasos (Δh = 0.5)", key="btn_reset_lab_dh"):
        for idx in range(1, 7):
            st.session_state[f"lab_dh_{idx}"] = 0.5
        rerun_fragmento_seguro()

    render_subtitulo_controles_vivo()
    fila_1 = st.columns(3)
    for offset, col in enumerate(fila_1):
        idx = offset + 1
        col.slider(
            f"Δh{_LAB_SUBINDICES[idx - 1]}",
            min_value=0.2,
            max_value=1.5,
            step=0.1,
            format="%.1f",
            key=f"lab_dh_{idx}",
        )

    fila_2 = st.columns(3)
    for offset, col in enumerate(fila_2):
        idx = offset + 4
        col.slider(
            f"Δh{_LAB_SUBINDICES[idx - 1]}",
            min_value=0.2,
            max_value=1.5,
            step=0.1,
            format="%.1f",
            key=f"lab_dh_{idx}",
        )

    pasos_h = [float(st.session_state[f"lab_dh_{idx}"]) for idx in range(1, 7)]
    res_lab = integrar_paso_variable(
        _f_lab_paso_variable,
        x0=0.0,
        pasos_h=pasos_h,
        integral_exacta_fn=_exacta_lab_paso_variable,
    )

    fig_lab = crear_figura_base_plotly(
        "Cuadratura Adaptativa por Tramos ante Variaciones de Δh",
        "x",
        "f(x)",
    )

    for tramo in res_lab["tramos"]:
        es_simpson = tramo["metodo"].startswith("Simpson")
        color_linea = PALETA_ARANDU["salvia"] if es_simpson else PALETA_ARANDU["terracota"]
        color_relleno = PALETA_ARANDU["salvia_fill"] if es_simpson else PALETA_ARANDU["terracota_fill"]
        estilo_linea = "solid" if es_simpson else "dash"
        fig_lab.add_trace(
            go.Scatter(
                x=tramo["x_curva"],
                y=limpiar_valores_plotly(tramo["y_curva"], limite_abs=1e9),
                mode="lines",
                fill="tozeroy",
                fillcolor=color_relleno,
                name=f"Tramo {tramo['tramo_id']}: {tramo['metodo']} ({tramo['nodos']})",
                line=dict(color=color_linea, width=2.4, dash=estilo_linea),
            )
        )

    stem_x: list[float | None] = []
    stem_y: list[float | None] = []
    for nx, ny in zip(res_lab["nodos_x"], res_lab["nodos_y"]):
        stem_x.extend([nx, nx, None])
        stem_y.extend([0.0, ny, None])
    fig_lab.add_trace(
        go.Scatter(
            x=stem_x,
            y=stem_y,
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color=PALETA_ARANDU["texto_principal"], width=1.3, dash="dot"),
        )
    )

    fig_lab.add_trace(
        go.Scatter(
            x=res_lab["nodos_x"],
            y=res_lab["nodos_y"],
            mode="markers",
            name="Nodos xᵢ",
            text=_LAB_NODOS,
            hovertemplate="%{text}: x=%{x:.2f}, f(x)=%{y:.4f}<extra></extra>",
            marker=dict(
                size=9,
                color=PALETA_ARANDU["texto_principal"],
                line=dict(color=PALETA_ARANDU["bg_lienzo"], width=1.6),
            ),
        )
    )

    xs_exacta = np.linspace(res_lab["x0"], res_lab["xf"], 220)
    ys_exacta = _f_lab_paso_variable(xs_exacta)
    fig_lab.add_trace(
        go.Scatter(
            x=xs_exacta,
            y=limpiar_valores_plotly(ys_exacta, limite_abs=1e9),
            mode="lines",
            name="Curva exacta f(x)",
            line=dict(color=PALETA_ARANDU["oro"], width=2.8),
        )
    )

    render_plotly_arandu(fig_lab, key="plot_int_lab_paso_variable")

    m1, m2, m3 = st.columns(3)
    m1.metric("Integral Exacta", f"{res_lab['integral_exacta']:.6f}")
    m2.metric("Aproximación Híbrida Actual", f"{res_lab['integral_aprox']:.6f}")
    m3.metric("Error Relativo Porcentual", f"{res_lab['error_relativo_pct']:.4f} %")

    filas_diag = [
        {
            "Tramo": tramo["tramo_id"],
            "Nodos": tramo["nodos"],
            "Intervalo": f"[{tramo['inicio']:.2f}, {tramo['fin']:.2f}]",
            "Paso Δh": f"{tramo['ancho_h']:.2f}",
            "Regla Activada": tramo["metodo"],
            "Área Parcial": round(tramo["area"], 6),
            "Justificación Matemática": tramo["motivo"],
        }
        for tramo in res_lab["tramos"]
    ]
    render_dataframe_acotado(filas_diag, max_filas=300, key_prefix="df_lab_paso_variable")


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
                "subtitulo": "Gregory · Derivadas numéricas",
                "resumen": "Estima derivadas primera y segunda sobre tablas equiespaciadas tanto en nodos como puntos libres.",
                "pista": "Diferencias · 1ra y 2da derivada",
            },
            {
                "numero": "2",
                "clave_svg": "trapecios",
                "titulo": "Regla de Trapecios",
                "subtitulo": "Poligonal · Orden O(h²)",
                "resumen": "Aproxima el área bajo la curva uniendo pares de nodos mediante segmentos rectos sucesivos.",
                "pista": "Poligonal recta · Cualquier n",
            },
            {
                "numero": "3",
                "clave_svg": "simpson_13",
                "titulo": "Regla de Simpson 1/3",
                "subtitulo": "Parábolas · Orden O(h⁴)",
                "resumen": "Ajusta arcos parabólicos cada dos subintervalos para integrar curvas suaves con alta precisión.",
                "pista": "Parábolas · Requiere n par",
            },
            {
                "numero": "4",
                "clave_svg": "simpson_38",
                "titulo": "Simpson 3/8 y Combinado",
                "subtitulo": "Cúbicas · Paridad mixta",
                "resumen": "Aplica arcos cúbicos de cuatro nodos o combina esquemas automáticamente si n es impar.",
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

    st.divider()
    _frag_laboratorio_paso_variable()


vista_integracion()
