"""
Vista de Derivación e Integración Numérica - Arandú.
Incluye Diferenciación (Gregory + punto libre u=0.4) y Motor Adaptativo en Vivo
para Trapecio, Simpson 1/3 y Simpson 3/8 ante cambios de Δh o paridad n.
Arquitectura Static-First con aislamiento mediante @st.fragment.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.catalogo_ejercicios import INTEGRALES_PRECOMPUTADAS
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
    rerun_fragmento_seguro,
)


_ESCENARIOS_INTEGRACION: dict[str, tuple[dict[str, Any], ...]] = {
    "trapecio": (
        {
            "id": "tp6_ej1",
            "nombre": "TP6 Ej. 1 (Cátedra): y = x²/5 en [0.5, 25.5] (h=5, n=5)",
            "f_str": "x**2 / 5",
            "x": [0.5, 5.5, 10.5, 15.5, 20.5, 25.5],
            "y": [0.05, 6.05, 22.05, 48.05, 84.05, 130.05],
            "nota": "Ejercicio oficial TP6 Ej. 1: trapecios compuestos sobre tabla con h = 5.",
        },
        {
            "id": "trap_sin",
            "nombre": "Visualizador: f(x) = sin(x) + 1.5 en [0, 4] (n=6)",
            "f_str": "sin(x) + 1.5",
            "a": 0.0,
            "b": 4.0,
            "n": 6,
            "nota": "Función trigonométrica suave para observar cómo los segmentos lineales subestiman crestas y sobreestiman valles.",
        },
        {
            "id": "trap_gauss",
            "nombre": "Visualizador: Campana 3·e^(-x²) en [-2, 2] (n=8)",
            "f_str": "3*exp(-x**2)",
            "a": -2.0,
            "b": 2.0,
            "n": 8,
            "nota": "Campana gaussiana sin primitiva elemental cerrada; comparada frente a Gauss-Legendre de 16 puntos.",
        },
    ),
    "simpson_1_3": (
        {
            "id": "tp6_ej2",
            "nombre": "TP6 Ej. 2 (Cátedra): y = x²/5 en [0.5, 20.5] (n=4, h=5)",
            "f_str": "x**2 / 5",
            "x": [0.5, 5.5, 10.5, 15.5, 20.5],
            "y": [0.05, 6.05, 22.05, 48.05, 84.05],
            "nota": "Ejercicio oficial TP6 Ej. 2: Simpson 1/3 puro con n = 4 subintervalos pares (error nulo en grado <= 3).",
        },
        {
            "id": "s13_sin",
            "nombre": "Visualizador: f(x) = sin(x) + 1.5 en [0, 4] (n=4)",
            "f_str": "sin(x) + 1.5",
            "a": 0.0,
            "b": 4.0,
            "n": 4,
            "nota": "Función trigonométrica suave ideal para observar cómo las parábolas se adaptan a crestas y valles.",
        },
        {
            "id": "s13_cub",
            "nombre": "Visualizador: f(x) = 0.2x³ - x² + 3 en [0, 4] (n=4)",
            "f_str": "0.2*x**3 - x**2 + 3",
            "a": 0.0,
            "b": 4.0,
            "n": 4,
            "nota": "Polinomio cúbico (grado 3). Simpson 1/3 integra polinomios de hasta grado 3 con error nulo.",
        },
        {
            "id": "s13_gauss",
            "nombre": "Visualizador: f(x) = 3·e^(-x²) en [-2, 2] (n=6)",
            "f_str": "3*exp(-x**2)",
            "a": -2.0,
            "b": 2.0,
            "n": 6,
            "nota": "Campana de Gauss sin antiderivada elemental cerrada; comparada frente a cuadratura de alta precisión.",
        },
        {
            "id": "s13_parab",
            "nombre": "Visualizador: f(x) = 0.5x² + 0.5 en [0, 4] (n=4)",
            "f_str": "0.5*x**2 + 0.5",
            "a": 0.0,
            "b": 4.0,
            "n": 4,
            "nota": "Parábola cuadrática pura. Al coincidir exactamente con el modelo interpolador, el error es siempre nulo.",
        },
        {
            "id": "s13_osc",
            "nombre": "Visualizador: f(x) = cos(2x) + 0.3x + 1.5 en [0, 4] (n=6)",
            "f_str": "cos(2*x) + 0.3*x + 1.5",
            "a": 0.0,
            "b": 4.0,
            "n": 6,
            "nota": "Función oscilatoria combinada con tendencia lineal. Aumentá n para capturar las ondas.",
        },
    ),
    "simpson_3_8": (
        {
            "id": "tp6_ej3",
            "nombre": "TP6 Ej. 3 (Cátedra): y = x²/5 en [0.5, 25.5] (n=5, Combinado)",
            "f_str": "x**2 / 5",
            "x": [0.5, 5.5, 10.5, 15.5, 20.5, 25.5],
            "y": [0.05, 6.05, 22.05, 48.05, 84.05, 130.05],
            "nota": "Ejercicio oficial TP6 Ej. 3: n = 5 impar resuelto con Simpson 3/8 [0.5, 15.5] + Simpson 1/3 [15.5, 25.5].",
        },
        {
            "id": "s38_cub",
            "nombre": "Visualizador: f(x) = x³ - 2x + 3 en [-1, 2] (n=6)",
            "f_str": "x**3 - 2*x + 3",
            "a": -1.0,
            "b": 2.0,
            "n": 6,
            "nota": "Polinomio cúbico (grado 3). Simpson 3/8 ajusta cúbicas exactas, logrando error nulo incluso con n = 3.",
        },
        {
            "id": "s38_sin",
            "nombre": "Visualizador: f(x) = sin(x) + 1.5 en [0, 4.5] (n=6)",
            "f_str": "sin(x) + 1.5",
            "a": 0.0,
            "b": 4.5,
            "n": 6,
            "nota": "Función trigonométrica suave. Cada bloque cúbico abraza puntos de inflexión y curvaturas.",
        },
        {
            "id": "s38_exp",
            "nombre": "Visualizador: f(x) = e^(0.5x) en [0, 3] (n=6)",
            "f_str": "exp(0.5*x)",
            "a": 0.0,
            "b": 3.0,
            "n": 6,
            "nota": "Crecimiento exponencial continuo. Al aumentar los bloques cúbicos, el error disminuye con orden O(h^4).",
        },
        {
            "id": "s38_gauss",
            "nombre": "Visualizador: f(x) = 3·e^(-x²) en [-1.5, 1.5] (n=6)",
            "f_str": "3*exp(-x**2)",
            "a": -1.5,
            "b": 1.5,
            "n": 6,
            "nota": "Campana de Gauss sin antiderivada elemental cerrada; comparada frente a cuadratura de Gauss-Legendre 16 pts.",
        },
    ),
}


def _materializar_escenario_integracion(esc: dict[str, Any]) -> tuple[list[float], list[float]]:
    if "x" in esc and "y" in esc:
        return list(esc["x"]), list(esc["y"])
    a_v = float(esc["a"])
    b_v = float(esc["b"])
    n_v = int(esc["n"])
    xs_v = np.linspace(a_v, b_v, n_v + 1)
    fn_np = INTEGRALES_PRECOMPUTADAS[esc["f_str"]]["fn_np"]
    ys_v = np.asarray(fn_np(xs_v), dtype=float)
    return [round(float(v), 6) for v in xs_v], [round(float(v), 6) for v in ys_v]


def _al_cambiar_preset_integracion(clave_state: str, metodo_solicitado: str) -> None:
    escenarios = _ESCENARIOS_INTEGRACION.get(metodo_solicitado, ())
    id_sel = st.session_state.get(f"ui_{clave_state}_preset")
    esc = next((e for e in escenarios if e["id"] == id_sel), None)
    if esc is None:
        return
    xn, yn = _materializar_escenario_integracion(esc)
    st.session_state[f"{clave_state}_x"] = xn
    st.session_state[f"{clave_state}_y"] = yn
    st.session_state[f"{clave_state}_f_str"] = esc["f_str"]
    st.session_state[f"{clave_state}_nota"] = esc["nota"]
    for w_suf in ("__editor_df", "fstr", "dh", "n", "lim_a", "lim_b"):
        st.session_state.pop(f"{clave_state}{w_suf}" if w_suf.startswith("__") else f"ui_{clave_state}_{w_suf}", None)


def _construir_html_ponderacion_nodal(pond: dict[str, Any], clave_state: str) -> str:
    nodos = pond.get("nodos", [])
    regla = pond.get("regla_principal", "")
    w_int_label = "4" if "1/3" in regla else ("3" if "3/8" in regla else "2")
    pills = []
    for nd in nodos:
        rol = nd["rol"]
        cls_rol = (
            "arandu-simpson-pill-extremo"
            if rol == "Extremo"
            else ("arandu-simpson-pill-frontera" if rol == "Frontera" else "arandu-simpson-pill-interior")
        )
        pills.append(
            f'<div class="arandu-simpson-pill {cls_rol}" '
            f'title="Nodo x{nd["index"]}={nd["x"]:.4f} | f={nd["y"]:.5f} | Peso={nd["weight"]} ({rol})">'
            f'<span class="arandu-simpson-pill-x">x<sub>{nd["index"]}</sub>={nd["x"]:.2f}</span>'
            f'<span class="arandu-simpson-pill-w">w = {nd["weight"]}</span>'
            f'<span class="arandu-simpson-pill-meta">{html.escape(rol)} · f={nd["y"]:.2f}</span>'
            "</div>"
        )
    titulo_vec = "Fórmula y Vector de Coeficientes de Ponderación"
    patron_esc = html.escape(str(pond.get("patron_texto", "")))
    resumen_esc = html.escape(str(pond.get("formula_resumen", "")))
    return (
        '<div class="arandu-simpson-weights-card" role="region" aria-label="Vector de coeficientes de ponderación nodal">'
        '<div class="arandu-simpson-weights-head">'
        f'<div><strong>{titulo_vec}</strong><br/>'
        f'<span style="font-size:0.86rem;color:var(--arandu-muted);">Patrón: <code>{patron_esc}</code></span></div>'
        f'<div class="arandu-simpson-formula-badge">{resumen_esc}</div>'
        "</div>"
        f'<div class="arandu-simpson-weights-grid">{"".join(pills)}</div>'
        '<div class="arandu-simpson-role-legend">'
        '<span><strong>Extremo (w=1):</strong> bordes del intervalo [a, b]</span>'
        f'<span><strong>Interior (w={w_int_label}):</strong> nodos internos de cada bloque</span>'
        '<span><strong>Frontera (w=2):</strong> nodos compartidos entre bloques adyacentes</span>'
        "</div>"
        "</div>"
    )


def _construir_md_tabla_nodal(pond: dict[str, Any]) -> str:
    lineas = [
        "| i | xᵢ | f(xᵢ) | Peso wᵢ | Rol | wᵢ · f(xᵢ) | Aporte al área |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for nd in pond.get("nodos", []):
        lineas.append(
            f"| {nd['index']} | {nd['x']:.4f} | {nd['y']:.5f} | {nd['weight']} | {nd['rol']} | {nd['wy']:.5f} | {nd['aporte']:.6f} |"
        )
    return "\n".join(lineas)


_ESCENARIOS_DIFERENCIACION: tuple[dict[str, Any], ...] = (
    {
        "id": "tp5_catedra",
        "nombre": "TP5 Ej. Cátedra: f'(12.4) desde tabla trigonométrica (h=1, u=0.4)",
        "x": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "y": [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151],
        "x_eval": 12.4,
        "nota": "Ejercicio oficial de cátedra: diferencia progresiva evaluada en el punto no tabular x* = 12.4.",
    },
    {
        "id": "parcial2_2026",
        "nombre": "Modelo 2do Parcial 2026: f(x) = ln(x), estimar f'(1.2) con Gregory Ascendente",
        "x": [1.0, 1.5, 2.0, 2.5, 3.0],
        "y": [0.0, 0.40547, 0.69315, 0.91629, 1.09861],
        "x_eval": 1.2,
        "nota": "Tema evaluado en 2do Parcial 2026: distribuir las potencias de u = (1.2 - 1.0)/0.5 = 0.4 antes de derivar.",
    },
    {
        "id": "cubica_exacta",
        "nombre": "Verificación Cúbica Exacta: f(x) = 0.5x³ - 1.5x² + x + 2 en x* = 1.2",
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 2.0, 2.0, 5.0, 14.0],
        "x_eval": 1.2,
        "nota": "En polinomios cúbicos, Δ⁴y = 0 y Gregory hasta 3er orden arroja exactitud analítica: f'(1.2) = -0.44 y f''(1.2) = 0.60.",
    },
)


def _al_cambiar_preset_diferenciacion(defaults: dict) -> None:
    id_sel = st.session_state.get("ui_der_preset", "tp5_catedra")
    esc = next((e for e in _ESCENARIOS_DIFERENCIACION if e["id"] == id_sel), None)
    if esc is None:
        return
    if id_sel == "tp5_catedra":
        st.session_state["int_deriv_x"] = list(defaults["x"])
        st.session_state["int_deriv_y"] = list(defaults["y"])
        st.session_state["int_deriv_x_eval"] = float(defaults["x_eval"])
    else:
        st.session_state["int_deriv_x"] = list(esc["x"])
        st.session_state["int_deriv_y"] = list(esc["y"])
        st.session_state["int_deriv_x_eval"] = float(esc["x_eval"])
    st.session_state.pop("int_deriv__editor_df", None)
    st.session_state.pop("ui_der_xe", None)


@st.fragment
def _frag_diferenciacion(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="int_deriv",
        defaults_dict=defaults,
        mensaje_edicion=r"Edita la tabla $(x_i, y_i)$, alterna el sentido de diferencias o cambia $x^*$ (incluso fuera de nodos tabulados).",
    )
    ids_der = [e["id"] for e in _ESCENARIOS_DIFERENCIACION]
    nom_der = {e["id"]: e["nombre"] for e in _ESCENARIOS_DIFERENCIACION}
    if "ui_der_preset" not in st.session_state or st.session_state["ui_der_preset"] not in ids_der:
        st.session_state["ui_der_preset"] = ids_der[0]
    der_id = st.selectbox(
        "Escenario de diferenciación:",
        options=ids_der,
        format_func=lambda k_d: nom_der[k_d],
        key="ui_der_preset",
        on_change=_al_cambiar_preset_diferenciacion,
        args=(defaults,),
    )
    esc_der = next(e for e in _ESCENARIOS_DIFERENCIACION if e["id"] == der_id)
    st.caption(esc_der["nota"])
    solo_lectura = (modo == "catedra") and (der_id == ids_der[0])

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
    registrar_contexto_ia(
        "Derivada Numérica (Newton-Gregory)",
        f"- **Escenario:** {esc_der['nombre']} | **Orden:** `{orden}` | **$x^*$:** `{x_eval:g}` | **{param_name}:** `{param:.4f}` | **Paso `h`:** `{res['h']:.4g}`\n"
        f"- **Derivada estimada:** `{res['derivada']:.6f}`",
        clave="ing_deriv_res",
    )
    simb_diff = r"\Delta" if esquema.startswith("Progresivas") else r"\nabla"
    pasos_der_vivos = [
        (
            f"**Ancla y parámetro adimensional:** Con paso uniforme $h = {res['h']:.4g}$ e índice base "
            f"$x_{{\\text{{base}}}} = {xs[res['indice_usado']]:g}$, calculamos "
            f"${param_name} = \\frac{{{x_eval:g} - {xs[res['indice_usado']]:g}}}{{{res['h']:.4g}}} = {param:.6f}$."
        )
    ]
    suma_aportes = 0.0
    for td in res["terminos_detalle"]:
        suma_aportes += td["aporte"]
        pasos_der_vivos.append(
            f"**Diferencia de orden {td['orden_diff']} (${simb_diff}^{{{td['orden_diff']}}} y$):** "
            f"Polinomio distribuido `{td['polinomio_param']}` $\\implies$ derivada en ${param_name}$ = "
            f"`{td['derivada_param_expr']}` $= {td['coeficiente']:.6f}$ $\\times$ `{td['diferencia']:.6g}` "
            f"$= {td['aporte']:.6f}$."
        )
    denom_h = f"h = {res['h']:.4g}" if orden == 1 else f"h^2 = {res['h']**2:.4g}"
    pasos_der_vivos.append(
        f"**División final por ${denom_h}$:** $\\frac{{{suma_aportes:.6f}}}{{{res['h'] if orden == 1 else res['h']**2:.4g}}} = {res['derivada']:.6f}$."
    )
    render_deduccion_paso_a_paso_viva(
        f"Diferenciación ({esquema}, orden {orden}, x*={x_eval:g})",
        pasos_der_vivos,
        clave_ia="ing_deriv_viva",
    )

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


_ESCENARIOS_SECANTE_TANGENTE: tuple[dict[str, Any], ...] = (
    {
        "id": "sec_parab",
        "nombre": "Clásico: f(x) = 0.5·x² (f'(x) = x, P en x₁ = 1.0)",
        "expr": "0.5·x²",
        "deriv_expr": "x",
        "fn": lambda x: 0.5 * np.power(x, 2),
        "dfn": lambda x: np.asarray(x, dtype=float),
        "x1": 1.0,
        "x2": 3.0,
        "xmin": -1.0,
        "xmax": 4.5,
    },
    {
        "id": "sec_sin",
        "nombre": "Onda Senoidal: f(x) = sin(x) + 1.5 (P en x₁ = 1.0)",
        "expr": "sin(x) + 1.5",
        "deriv_expr": "cos(x)",
        "fn": lambda x: np.sin(x) + 1.5,
        "dfn": lambda x: np.cos(x),
        "x1": 1.0,
        "x2": 2.6,
        "xmin": -0.5,
        "xmax": 4.5,
    },
    {
        "id": "sec_cub",
        "nombre": "Polinomio Cúbico: f(x) = 0.2x³ - x² + 3 (P en x₁ = 1.5)",
        "expr": "0.2x³ - x² + 3",
        "deriv_expr": "0.6x² - 2x",
        "fn": lambda x: 0.2 * np.power(x, 3) - np.power(x, 2) + 3.0,
        "dfn": lambda x: 0.6 * np.power(x, 2) - 2.0 * np.asarray(x, dtype=float),
        "x1": 1.5,
        "x2": 3.2,
        "xmin": -0.5,
        "xmax": 4.2,
    },
    {
        "id": "sec_exp",
        "nombre": "Crecimiento Exponencial: f(x) = e^(0.5x) (P en x₁ = 0.8)",
        "expr": "e^(0.5x)",
        "deriv_expr": "0.5·e^(0.5x)",
        "fn": lambda x: np.exp(0.5 * np.asarray(x, dtype=float)),
        "dfn": lambda x: 0.5 * np.exp(0.5 * np.asarray(x, dtype=float)),
        "x1": 0.8,
        "x2": 2.5,
        "xmin": -0.5,
        "xmax": 3.5,
    },
)


def _al_cambiar_preset_sectan() -> None:
    id_sel = st.session_state.get("ui_sectan_preset", "sec_parab")
    esc = next((e for e in _ESCENARIOS_SECANTE_TANGENTE if e["id"] == id_sel), _ESCENARIOS_SECANTE_TANGENTE[0])
    st.session_state["ui_sectan_x1"] = float(esc["x1"])
    st.session_state["ui_sectan_x2"] = float(esc["x2"])


@st.fragment
def _frag_secante_tangente() -> None:
    st.markdown("## Simulador Geométrico: Recta Secante y Recta Tangente")
    st.markdown(
        r"Cómo el límite de las pendientes secantes entre $P(x_1, f(x_1))$ y un punto móvil $Q(x_2, f(x_2))$ "
        r"define la derivada instantánea $f'(x_1) = \lim_{\Delta x \to 0} \frac{f(x_1+\Delta x)-f(x_1)}{\Delta x}$."
    )
    ids_st = [e["id"] for e in _ESCENARIOS_SECANTE_TANGENTE]
    mapa_st = {e["id"]: e["nombre"] for e in _ESCENARIOS_SECANTE_TANGENTE}
    if "ui_sectan_preset" not in st.session_state or st.session_state["ui_sectan_preset"] not in ids_st:
        st.session_state["ui_sectan_preset"] = ids_st[0]

    esc_id = st.selectbox(
        "Función de estudio:",
        options=ids_st,
        format_func=lambda k_s: mapa_st[k_s],
        key="ui_sectan_preset",
        on_change=_al_cambiar_preset_sectan,
    )
    esc = next(e for e in _ESCENARIOS_SECANTE_TANGENTE if e["id"] == esc_id)

    if "ui_sectan_x1" not in st.session_state:
        st.session_state["ui_sectan_x1"] = float(esc["x1"])
    if "ui_sectan_x2" not in st.session_state:
        st.session_state["ui_sectan_x2"] = float(esc["x2"])

    c1, c2 = st.columns([1.0, 2.0])
    x1_val = c1.number_input(
        "Punto fijo P (x₁):",
        min_value=float(esc["xmin"] + 0.1),
        max_value=float(esc["xmax"] - 0.1),
        step=0.1,
        format="%.2f",
        help="Coordenada fija P(x₁, f(x₁)) donde se evalúa la recta tangente.",
        key="ui_sectan_x1",
    )
    x2_raw = c2.slider(
        "Desplazar punto móvil Q (x₂) hacia x₁:",
        min_value=float(esc["xmin"]),
        max_value=float(esc["xmax"]),
        step=0.02,
        format="%.2f",
        help="Acercá x₂ hacia x₁ para observar cómo la secante converge geométricamente hacia la tangente.",
        key="ui_sectan_x2",
    )

    x2_eff = float(x2_raw)
    if abs(x2_eff - x1_val) < 0.015:
        x2_eff = x1_val + 0.02 if x1_val + 0.02 <= esc["xmax"] else x1_val - 0.02

    y1 = float(esc["fn"](x1_val))
    y2 = float(esc["fn"](x2_eff))
    dx = x2_eff - x1_val
    dy = y2 - y1
    m_sec = dy / dx
    m_tan = float(esc["dfn"](x1_val))
    diff_m = abs(m_sec - m_tan)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pendiente Secante (m_sec)", f"{m_sec:.5f}")
    m2.metric("Pendiente Tangente (m_tan)", f"{m_tan:.5f}")
    m3.metric("Diferencia |m_sec - m_tan|", f"{diff_m:.5f}")
    m4.metric("Incremento Δx = x₂ - x₁", f"{dx:+.3f}")

    xs_grid = np.linspace(esc["xmin"], esc["xmax"], 220)
    ys_grid = esc["fn"](xs_grid)
    ys_tan = m_tan * (xs_grid - x1_val) + y1
    ys_sec = m_sec * (xs_grid - x1_val) + y1

    fig_sec = crear_figura_base_plotly(
        f"Convergencia Secante → Tangente: f(x) = {esc['expr']} (x₁={x1_val:.2f}, x₂={x2_eff:.2f})",
        "x",
        "f(x)",
    )
    fig_sec.add_trace(
        go.Scatter(
            x=xs_grid.tolist(),
            y=limpiar_valores_plotly(ys_grid, limite_abs=1e6),
            mode="lines",
            name=f"f(x) = {esc['expr']}",
            line=dict(color=PALETA_ARANDU["salvia"], width=2.8),
        )
    )
    fig_sec.add_trace(
        go.Scatter(
            x=xs_grid.tolist(),
            y=limpiar_valores_plotly(ys_tan, limite_abs=1e6),
            mode="lines",
            name=f"Tangente en P (m={m_tan:.4f})",
            line=dict(color=PALETA_ARANDU["oro"], width=2.3, dash="dash"),
        )
    )
    fig_sec.add_trace(
        go.Scatter(
            x=xs_grid.tolist(),
            y=limpiar_valores_plotly(ys_sec, limite_abs=1e6),
            mode="lines",
            name=f"Secante P-Q (m={m_sec:.4f})",
            line=dict(color=PALETA_ARANDU["terracota"], width=2.5),
        )
    )
    fig_sec.add_trace(
        go.Scatter(
            x=[x1_val, x2_eff, x2_eff],
            y=[y1, y1, y2],
            mode="lines",
            name="Triángulo Δx / Δy",
            line=dict(color=PALETA_ARANDU["texto_atenuado"], width=1.6, dash="dot"),
        )
    )
    fig_sec.add_trace(
        go.Scatter(
            x=[x1_val],
            y=[y1],
            mode="markers",
            name=f"P({x1_val:.2f}, {y1:.3f})",
            marker=dict(
                size=13,
                symbol="star",
                color=PALETA_ARANDU["oro"],
                line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.6),
            ),
        )
    )
    fig_sec.add_trace(
        go.Scatter(
            x=[x2_eff],
            y=[y2],
            mode="markers",
            name=f"Q({x2_eff:.2f}, {y2:.3f})",
            marker=dict(
                size=10,
                symbol="circle",
                color=PALETA_ARANDU["terracota"],
                line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.6),
            ),
        )
    )
    render_plotly_arandu(fig_sec, key="plot_int_sectan")

    pasos_st = [
        f"**Función y derivada analítica:** $f(x) = {esc['expr']}$ con derivada $f'(x) = {esc['deriv_expr']}$.",
        f"**Puntos de corte de la secante:** $P = ({x1_val:.2f}, {y1:.5f})$ y $Q = ({x2_eff:.2f}, {y2:.5f})$ ($\\Delta x = {dx:+.4f}$, $\\Delta y = {dy:+.5f}$).",
        f"**Cociente incremental (pendiente secante):** $m_{{\\text{{sec}}}} = \\frac{{{y2:.5f} - ({y1:.5f})}}{{{x2_eff:.2f} - ({x1_val:.2f})}} = {m_sec:.5f}$ (tasa de cambio promedio).",
        f"**Límite instantáneo (pendiente tangente):** En $x_1 = {x1_val:.2f}$, $m_{{\\text{{tan}}}} = f'({x1_val:.2f}) = {m_tan:.5f}$; la discrepancia actual es $|\\Delta m| = {diff_m:.5f}$.",
    ]
    registrar_contexto_ia(
        "Simulador Geométrico Recta Secante y Tangente",
        f"- **Función:** `{esc['expr']}` | **$P$:** `({x1_val:.2f}, {y1:.5f})` | **$Q$:** `({x2_eff:.2f}, {y2:.5f})`\n"
        f"- **$m_{{\\text{{sec}}}}$:** `{m_sec:.5f}` | **$m_{{\\text{{tan}}}}$:** `{m_tan:.5f}` | **$|\\Delta m|$:** `{diff_m:.5f}`",
        clave="ing_sectan_res",
    )
    render_deduccion_paso_a_paso_viva(
        f"Convergencia Secante → Tangente ({esc['expr']})",
        pasos_st,
        clave_ia="ing_sectan_viva",
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("**La Recta Secante (*secare*)**")
        st.caption("Corta la curva en dos puntos P y Q. Representa la tasa de cambio promedio en [x₁, x₂].")
        st.latex(r"m_{\text{sec}} = \frac{f(x_2) - f(x_1)}{x_2 - x_1}")
    with k2:
        st.markdown("**La Recta Tangente (*tangere*)**")
        st.caption("Posición límite a la que converge la secante cuando x₂ tiende a x₁ (tasa instantánea).")
        st.latex(r"m_{\text{tan}} = f'(x_1) = \lim_{x_2 \to x_1} m_{\text{sec}}")
    with k3:
        st.markdown("**Función Trigonométrica `sec(θ)`**")
        st.caption("En trigonometría es la recíproca del coseno: mide la secante desde el centro unitario hasta la tangente vertical.")
        st.latex(r"\sec(\theta) = \frac{1}{\cos(\theta)}")


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
    st.divider()
    _frag_secante_tangente()


@st.fragment
def _frag_integracion(clave_state: str, metodo_solicitado: str, defaults: dict) -> None:
    escenarios = _ESCENARIOS_INTEGRACION.get(metodo_solicitado, ())
    modo = render_controles_edicion_practica(
        clave_state=clave_state,
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica los puntos tabulados, cambia de función o ajusta límites $[a, b]$, paso $\Delta h$ y subintervalos $n$.",
    )
    if escenarios:
        ids_esc = [e["id"] for e in escenarios]
        mapa_nom = {e["id"]: e["nombre"] for e in escenarios}
        preset_key = f"ui_{clave_state}_preset"
        if preset_key not in st.session_state or st.session_state[preset_key] not in ids_esc:
            st.session_state[preset_key] = ids_esc[0]
        esc_id = st.selectbox(
            "Escenario o función de estudio:",
            options=ids_esc,
            format_func=lambda k_esc: mapa_nom[k_esc],
            key=preset_key,
            on_change=_al_cambiar_preset_integracion,
            args=(clave_state, metodo_solicitado),
        )
        esc_actual = next((e for e in escenarios if e["id"] == esc_id), escenarios[0])
        solo_lectura = (modo == "catedra") and (esc_id == ids_esc[0])
        st.caption(esc_actual["nota"])
    else:
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
        ys_act = st.session_state[f"{clave_state}_y"]
        a_act = float(xs_act[0])
        b_act = float(xs_act[-1])
        h_actual = float(xs_act[1] - xs_act[0]) if len(xs_act) >= 2 else 1.0
        n_actual = max(1, len(xs_act) - 1)

        if metodo_solicitado == "simpson_1_3":
            info_bloques = f"h = {h_actual:.4f} | {n_actual // 2} parábola(s)" if n_actual % 2 == 0 else f"h = {h_actual:.4f} | n={n_actual} (adaptativo)"
        elif metodo_solicitado == "simpson_3_8":
            info_bloques = f"h = {h_actual:.4f} | {n_actual // 3} bloque(s) cúbico(s)" if n_actual % 3 == 0 else f"h = {h_actual:.4f} | n={n_actual} (combinado)"
        else:
            info_bloques = f"h = {h_actual:.4f} | {n_actual} trapecio(s)"

        st.markdown(rf"**Ajuste de Límites $[a, b]$, Paso $\Delta h$ y Subintervalos $n$:** (`{info_bloques}`)")
        render_subtitulo_controles_vivo()

        c_a, c_b = st.columns(2)
        lim_a = c_a.number_input(
            "Límite inferior $a$:",
            value=float(round(a_act, 4)),
            step=0.25,
            format="%.4f",
            disabled=solo_lectura,
            key=f"ui_{clave_state}_lim_a",
        )
        lim_b = c_b.number_input(
            "Límite superior $b$:",
            value=float(round(b_act, 4)),
            step=0.25,
            format="%.4f",
            disabled=solo_lectura,
            key=f"ui_{clave_state}_lim_b",
        )

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
            max_value=30,
            value=min(30, max(2, n_actual)),
            disabled=solo_lectura,
            key=f"ui_{clave_state}_n",
        )
        estrategia_impar = st.selectbox(
            "Estrategia si hay paridad incompatible:",
            options=["simpson_3_8", "trapecio"],
            format_func=lambda s: "Combinar Simpson 3/8 + 1/3" if s == "simpson_3_8" else "Combinar Simpson 1/3 + Trapecio",
            key=f"ui_{clave_state}_est",
        )

        b1, b2, b3 = st.columns(3)
        if not solo_lectura and b1.button("Aplicar Δh", key=f"btn_dh_{clave_state}", use_container_width=True):
            xn, yn, _ = remuestrear_tabla_integracion(xs_act, ys_act, nuevo_h=nuevo_h, f_str=f_str or None)
            st.session_state[f"{clave_state}_x"] = xn
            st.session_state[f"{clave_state}_y"] = yn
            if f"{clave_state}__editor_df" in st.session_state:
                del st.session_state[f"{clave_state}__editor_df"]
            rerun_fragmento_seguro()
        if not solo_lectura and b2.button("Aplicar n subinterv.", key=f"btn_n_{clave_state}", use_container_width=True):
            xn, yn, _ = remuestrear_tabla_integracion(xs_act, ys_act, nuevo_n=nuevo_n, f_str=f_str or None)
            st.session_state[f"{clave_state}_x"] = xn
            st.session_state[f"{clave_state}_y"] = yn
            if f"{clave_state}__editor_df" in st.session_state:
                del st.session_state[f"{clave_state}__editor_df"]
            rerun_fragmento_seguro()
        if not solo_lectura and b3.button("Aplicar [a, b]", key=f"btn_ab_{clave_state}", use_container_width=True):
            if lim_b <= lim_a + 1e-6:
                st.warning("El límite superior b debe ser estrictamente mayor que a.")
            else:
                xn, yn, _ = remuestrear_tabla_integracion([lim_a, lim_b], [0.0, 0.0], nuevo_n=nuevo_n, f_str=f_str or None) if (f_str and f_str.strip()) else remuestrear_tabla_integracion([lim_a, lim_b], [ys_act[0], ys_act[-1]], nuevo_n=nuevo_n, f_str=None)
                st.session_state[f"{clave_state}_x"] = xn
                st.session_state[f"{clave_state}_y"] = yn
                if f"{clave_state}__editor_df" in st.session_state:
                    del st.session_state[f"{clave_state}__editor_df"]
                rerun_fragmento_seguro()

        c_cap1, c_cap2 = st.columns(2)
        mostrar_polinomios = c_cap1.checkbox(
            "Polinomios sombreados",
            value=True,
            key=f"ui_{clave_state}_cap_poly",
        )
        mostrar_nodos = c_cap2.checkbox(
            "Nodos diferenciados (wᵢ)",
            value=True,
            key=f"ui_{clave_state}_cap_nodes",
        )

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
    m1.metric("Área Total Estimada", f"{res['area_total']:.6f}", f"{res['reglas_aplicadas']} (n={res['n_intervalos_total']})")
    if res["exacta"]:
        ex_inf = res["exacta"]
        m2.metric("Integral Exacta", f"{ex_inf['integral_exacta']:.6f}", ex_inf.get("metodo_ref", "Analítica"))
        m3.metric("Error Absoluto |E|", f"{ex_inf['error_absoluto']:.3e}" if ex_inf["error_absoluto"] >= 1e-11 else "0.000000")
        m4.metric("Error Relativo (%)", f"{ex_inf['error_relativo_pct']:.4f}%" if ex_inf["error_relativo_pct"] >= 1e-9 else "0.0000%")
    else:
        m2.metric("Integral Exacta", "No provista")
        m3.metric("Error Absoluto |E|", "-")
        m4.metric("Error Relativo (%)", "-")

    pond = res.get("ponderacion_nodal")
    exacta_md = (
        f"- **Integral exacta ({res['exacta'].get('metodo_ref', 'Analítica')}):** `{res['exacta']['integral_exacta']:.6f}` | "
        f"**Error abs.:** `{res['exacta']['error_absoluto']:.3e}` | **Error rel.:** `{res['exacta']['error_relativo_pct']:.4f}%`"
        if res["exacta"]
        else ""
    )
    pond_md = ""
    if pond:
        pond_md = f"\n- **Fórmula de ponderación nodal:** `{pond['formula_resumen']}` (Patrón `{pond['patron_texto']}`)\n" + _construir_md_tabla_nodal(pond)
    registrar_contexto_ia(
        f"Resultado Cuadratura ({clave_state})",
        f"- **Área estimada:** `{res['area_total']:.6f}` | **Reglas aplicadas:** {res['reglas_aplicadas']} | **Subintervalos:** `{res['n_intervalos_total']}`\n{exacta_md}{pond_md}".strip(),
        clave=f"{clave_state}_res",
    )

    if pond:
        st.markdown(_construir_html_ponderacion_nodal(pond, clave_state), unsafe_allow_html=True)
        with st.expander("Ver tabla detallada de evaluación nodal (xᵢ, f(xᵢ), wᵢ, wᵢ·f(xᵢ))", expanded=False):
            filas_nodal = [
                {
                    "i": nd["index"],
                    "xᵢ": round(nd["x"], 4),
                    "f(xᵢ)": round(nd["y"], 5),
                    "Peso wᵢ": nd["weight"],
                    "Rol": nd["rol"],
                    "wᵢ · f(xᵢ)": round(nd["wy"], 5),
                    "Aporte al área": round(nd["aporte"], 6),
                }
                for nd in pond["nodos"]
            ]
            render_dataframe_acotado(filas_nodal, max_filas=300, key_prefix=f"tbl_{clave_state}_nodal")

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
    pasos_int_vivos = [
        f"**Segmentación por paso $\\Delta h$ y paridad:** Se evaluaron `{res['n_intervalos_total']}` subintervalos aplicando `{res['reglas_aplicadas']}`."
    ]
    for i_b, b in enumerate(res["bloques"][:8], start=1):
        pasos_int_vivos.append(
            f"**Bloque {i_b} ({b['regla']} en $[{b['x_nodos'][0]:g}, {b['x_nodos'][-1]:g}]$):** `{b['formula']}` $\\implies$ Área $= {b['area']:.6f}$."
        )
    if len(res["bloques"]) > 8:
        pasos_int_vivos.append(f"_(Se resumen los primeros 8 bloques de {len(res['bloques'])} totales; suma acumulada completa abajo)_.")
    pasos_int_vivos.append(
        f"**Suma integral total:** $\\sum \\text{{Áreas}}_k = {res['area_total']:.6f}$"
        + (f" frente a la exacta `{res['exacta']['integral_exacta']:.6f}` ($E_a = {res['exacta']['error_absoluto']:.3e}$)." if res["exacta"] else ".")
    )
    render_deduccion_paso_a_paso_viva(
        f"Cuadratura ({res['reglas_aplicadas']})",
        pasos_int_vivos,
        clave_ia=f"{clave_state}_viva",
    )
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
    reglas_distintas = len({b["regla"] for b in bloques})
    if mostrar_polinomios:
        for idx, b in enumerate(bloques[:k_vis], start=1):
            if reglas_distintas == 1 and b["regla"] in ("Simpson 1/3", "Simpson 3/8"):
                es_par_b = idx % 2 == 0
                fill_b = PALETA_ARANDU["oro_fill"] if es_par_b else PALETA_ARANDU["salvia_fill"]
                col_b = PALETA_ARANDU["oro"] if es_par_b else PALETA_ARANDU["salvia"]
                etq_bloque = f"B{idx} ({'par' if es_par_b else 'impar'}): {b['regla']} (A={b['area']:.3f})"
            else:
                fill_b = fill_por_regla.get(b["regla"], PALETA_ARANDU["salvia_fill"])
                col_b = color_por_regla.get(b["regla"], PALETA_ARANDU["salvia"])
                etq_bloque = f"B{idx}: {b['regla']} (A={b['area']:.3f})"
            fig.add_trace(
                go.Scatter(
                    x=b["x_curva"],
                    y=limpiar_valores_plotly(b["y_curva"], limite_abs=1e9),
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=fill_b,
                    name=etq_bloque,
                    line=dict(
                        color=col_b,
                        width=2.4,
                        dash=dash_por_regla.get(b["regla"], "solid"),
                    ),
                )
            )
            bx0, bxf = float(b["x_nodos"][0]), float(b["x_nodos"][-1])
            by0, byf = float(b["y_nodos"][0]), float(b["y_nodos"][-1])
            fig.add_trace(
                go.Scatter(
                    x=[bx0, bx0, None, bxf, bxf],
                    y=[0.0, by0, None, 0.0, byf],
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(color=col_b, width=1.1, dash="dot"),
                )
            )

    if res["exacta"] and res["exacta"].get("curva_exacta"):
        curva_ex = res["exacta"]["curva_exacta"]
        fig.add_trace(
            go.Scatter(
                x=curva_ex["x"],
                y=limpiar_valores_plotly(curva_ex["y"], limite_abs=1e9),
                mode="lines",
                name="Curva exacta f(x)",
                line=dict(color=PALETA_ARANDU["texto_principal"], width=2.6),
            )
        )

    if mostrar_nodos and pond:
        grupos_nodos = (
            ("Extremo (w=1)", "Extremo", PALETA_ARANDU["marfil_nodo"], "square"),
            ("Interior", "Interior", PALETA_ARANDU["salvia"], "circle"),
            ("Frontera (w=2)", "Frontera", PALETA_ARANDU["oro"], "diamond"),
        )
        ver_etq = len(xs) <= 15
        for nombre_grp, rol_filtro, col_grp, simb_grp in grupos_nodos:
            sub_nd = [nd for nd in pond["nodos"] if nd["rol"] == rol_filtro]
            if not sub_nd:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[nd["x"] for nd in sub_nd],
                    y=[nd["y"] for nd in sub_nd],
                    mode="markers+text" if ver_etq else "markers",
                    name=nombre_grp,
                    text=[f"x{nd['index']}" for nd in sub_nd] if ver_etq else None,
                    textposition="top center",
                    customdata=[[nd["weight"], nd["rol"]] for nd in sub_nd],
                    hovertemplate="x=%{x:.4f}<br>f(x)=%{y:.5f}<br>Peso w=%{customdata[0]} (%{customdata[1]})<extra></extra>",
                    marker=dict(
                        size=10,
                        color=col_grp,
                        symbol=simb_grp,
                        line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.6),
                    ),
                )
            )
    elif mostrar_nodos:
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
    pasos_lab_vivos = [
        f"**Partición de pasos en vivo:** $\\Delta h = {[round(v, 2) for v in pasos_h]}$ sobre $f(x) = \\sin(x) + 0.5x + 1.5$ en $[{res_lab['x0']:.2f}, {res_lab['xf']:.2f}]$."
    ]
    for tramo in res_lab["tramos"]:
        pasos_lab_vivos.append(
            f"**Tramo {tramo['tramo_id']} ({tramo['nodos']}, $[{tramo['inicio']:.2f}, {tramo['fin']:.2f}]$):** `{tramo['metodo']}` ($h={tramo['ancho_h']:.2f}$) $\\implies$ Área $= {tramo['area']:.6f}$. {tramo['motivo']}"
        )
    pasos_lab_vivos.append(
        f"**Total híbrido vs Exacta:** Suma $= {res_lab['integral_aprox']:.6f}$ frente a exacta ${res_lab['integral_exacta']:.6f}$ ($E_r = {res_lab['error_relativo_pct']:.4f}\\%)$."
    )
    render_deduccion_paso_a_paso_viva(
        "Laboratorio Paso Variable (Δh)",
        pasos_lab_vivos,
        clave_ia="int_lab_pv_viva",
    )
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
