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
    calcular_derivada_gregory_completa,
    derivada_newton_gregory_ascendente,
    derivada_newton_gregory_descendente,
    integrar_adaptativo_en_vivo,
    integrar_simpson_1_3_hibrido,
    integrar_simpson_3_8_hibrido,
    integrar_trapecio_hibrido,
    recomendar_esquema_diferenciacion,
    reconstruir_malla_por_pasos,
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
    render_toggle_modo_detallado,
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


_RADAR_T_DEFAULT = [0.0, 1.0, 2.0, 3.0, 4.0]
_RADAR_S_DEFAULT = [2.0, 5.0, 8.0, 14.0, 26.0]


def _aplicar_preset_radar_deriv(t_eval: float, esquema: str, orden: int = 1) -> None:
    st.session_state["int_deriv_real_t"] = float(t_eval)
    st.session_state["int_deriv_real_esq"] = esquema
    st.session_state["int_deriv_real_ord"] = int(orden)
    for k_ui in ("ui_der_real_t", "ui_der_real_esq", "ui_der_real_ord"):
        st.session_state.pop(k_ui, None)


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
def _frag_diferenciacion(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_radar", "catedra_libre"],
        format_func=lambda op: (
            "Caso Real: Telemetría Radar s(t) → Velocidad v(t) y Aceleración a(t)"
            if op == "real_radar"
            else "Tablas de Cátedra (TP6 / 2do Parcial 2026) y Edición Manual"
        ),
        horizontal=True,
        key="ui_int_deriv_contexto",
    )
    es_caso_real = contexto == "real_radar"

    if es_caso_real:
        st.caption(
            r"Muestras radar de posición $s(t)$ en $t = [0, 1, 2, 3, 4]\;\text{s}$ ($s = [2, 5, 8, 14, 26]\;\text{m}$, "
            r"trayectoria $s(t) = 0.5t^3 - 1.5t^2 + 4t + 2$). Estimá velocidad $v(t^*)$ y aceleración $a(t^*)$."
        )
        if "int_deriv_real_t" not in st.session_state:
            st.session_state["int_deriv_real_t"] = 1.2
        if "int_deriv_real_esq" not in st.session_state:
            st.session_state["int_deriv_real_esq"] = "ascendente"
        if "int_deriv_real_ord" not in st.session_state:
            st.session_state["int_deriv_real_ord"] = 1

        b1, b2, b3 = st.columns(3)
        if b1.button(
            "Inicio de maniobra (t* = 0.5 s → Gregory Ascendente desde t₀ = 0 s)",
            key="btn_der_real_ini",
            use_container_width=True,
        ):
            _aplicar_preset_radar_deriv(0.5, "ascendente", 1)
            rerun_fragmento_seguro()
        if b2.button(
            "Punto no tabular intermedio (t* = 1.2 s, u = 0.2 → Velocidad y Aceleración exactas)",
            key="btn_der_real_med",
            use_container_width=True,
        ):
            _aplicar_preset_radar_deriv(1.2, "ascendente", 1)
            rerun_fragmento_seguro()
        if b3.button(
            "Tiempo real al final (t* = 3.6 s → Gregory Descendente desde t₄ = 4 s)",
            key="btn_der_real_fin",
            use_container_width=True,
        ):
            _aplicar_preset_radar_deriv(3.6, "descendente", 1)
            rerun_fragmento_seguro()

        xs = list(_RADAR_T_DEFAULT)
        ys = list(_RADAR_S_DEFAULT)
        c1, c2, c3 = st.columns(3)
        esquema_cod = c1.radio(
            "Sentido de diferencias:",
            options=["ascendente", "descendente"],
            index=0 if st.session_state["int_deriv_real_esq"] == "ascendente" else 1,
            format_func=lambda s: "Progresivas (Ascendente Δ)" if s == "ascendente" else "Regresivas (Descendente ∇)",
            horizontal=True,
            key="ui_der_real_esq",
        )
        orden = int(
            c2.radio(
                "Orden de derivada:",
                options=[1, 2],
                index=0 if int(st.session_state["int_deriv_real_ord"]) == 1 else 1,
                format_func=lambda o: "1ra Derivada: Velocidad v(t)" if o == 1 else "2da Derivada: Aceleración a(t)",
                horizontal=True,
                key="ui_der_real_ord",
            )
        )
        x_eval = float(
            c3.number_input(
                "Instante de evaluación t* (s):",
                min_value=-1.0,
                max_value=5.0,
                value=float(st.session_state["int_deriv_real_t"]),
                step=0.1,
                format="%.2f",
                key="ui_der_real_t",
            )
        )
        st.session_state["int_deriv_real_esq"] = esquema_cod
        st.session_state["int_deriv_real_ord"] = orden
        st.session_state["int_deriv_real_t"] = x_eval
    else:
        if "int_deriv_x" not in st.session_state:
            st.session_state["int_deriv_x"] = list(defaults["x"])
            st.session_state["int_deriv_y"] = list(defaults["y"])
            st.session_state["int_deriv_x_eval"] = float(defaults["x_eval"])

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

        df_in = pd.DataFrame({"x": st.session_state["int_deriv_x"], "y": st.session_state["int_deriv_y"]})
        df_edit = render_data_editor_arandu(
            df_in,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "x": st.column_config.NumberColumn("Abscisa xᵢ", format="%.4f"),
                "y": st.column_config.NumberColumn("Ordenada yᵢ", format="%.5f"),
            },
            key="int_deriv__editor_df",
        )
        df_c = df_edit.dropna()
        if len(df_c) >= 3:
            st.session_state["int_deriv_x"] = df_c["x"].astype(float).tolist()
            st.session_state["int_deriv_y"] = df_c["y"].astype(float).tolist()
        xs = list(st.session_state["int_deriv_x"])
        ys = list(st.session_state["int_deriv_y"])

        c1, c2, c3 = st.columns(3)
        esquema_cod = c1.radio(
            "Sentido de diferencias:",
            options=["ascendente", "descendente"],
            format_func=lambda s: "Progresivas (Ascendente Δ)" if s == "ascendente" else "Regresivas (Descendente ∇)",
            horizontal=True,
            key="ui_der_esq",
        )
        orden = int(
            c2.radio(
                "Orden de derivada:",
                options=[1, 2],
                format_func=lambda o: "1ra Derivada f'(x)" if o == 1 else "2da Derivada f''(x)",
                horizontal=True,
                key="ui_der_ord",
            )
        )
        x_eval = float(
            c3.number_input(
                "Punto x* a derivar (puede ser no tabular):",
                value=float(st.session_state["int_deriv_x_eval"]),
                format="%.4f",
                key="ui_der_xe",
            )
        )
        st.session_state["int_deriv_x_eval"] = x_eval

    try:
        rec = recomendar_esquema_diferenciacion(xs, x_eval)
        res = calcular_derivada_gregory_completa(xs, ys, x_eval, orden_derivada=orden, esquema=esquema_cod)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos o punto x* de derivación",
            sugerencia_defecto="Ingresá nodos x crecientes equidistantes (h > 0) y elegí Ascendente al inicio o Descendente al final de la tabla.",
            ejemplo_defecto="x = [0.0, 1.0, 2.0, 3.0, 4.0], x* = 1.2",
        )
        return

    if rec["esquema"] != esquema_cod:
        st.info(f"Sugerencia didáctica: {rec['motivo']}")

    param_name = "u" if esquema_cod == "ascendente" else "v"
    param_val = float(res["u"] if esquema_cod == "ascendente" else res["v"])

    m1, m2, m3 = st.columns(3)
    if es_caso_real:
        exacta_val = (1.5 * (x_eval**2) - 3.0 * x_eval + 4.0) if orden == 1 else (3.0 * x_eval - 3.0)
        unidad_d = "m/s" if orden == 1 else "m/s²"
        etq_kpi = f"Velocidad Estimada v({x_eval:g})" if orden == 1 else f"Aceleración Estimada a({x_eval:g})"
        m1.metric(
            etq_kpi,
            f"{res['derivada']:.4f} {unidad_d}",
            f"Exacta: {exacta_val:.4f} {unidad_d} · Err {abs(exacta_val - res['derivada']):.2e}",
            delta_color="off",
        )
        ancla_txt = f"Ancla t_{res['indice_usado']} = {xs[res['indice_usado']]:g} s"
        paso_val_txt = f"h = {res['h']:g} s"
    else:
        etq_kpi = f"Derivada Estimada f{'´' if orden == 1 else '´´'}({x_eval:g})"
        m1.metric(
            etq_kpi,
            f"{res['derivada']:.6f}",
            f"P({x_eval:g}) = {res['y_eval']:.5f}",
            delta_color="off",
        )
        ancla_txt = f"Ancla x_{res['indice_usado']} = {xs[res['indice_usado']]:g}"
        paso_val_txt = f"h = {res['h']:g}"

    m2.metric("Nodo Ancla y Parámetro (u / v)", f"{param_name} = {param_val:.4f}", ancla_txt, delta_color="off")
    m3.metric(
        "Paso Uniforme h y Orden",
        paso_val_txt,
        f"{len(res['terminos_detalle'])} término(s) de diferencias",
        delta_color="off",
    )

    y_en_x = float(res["y_eval"])
    curva_poly = res["curva_polinomio"]
    x_lbl = "Tiempo t (s)" if es_caso_real else "x"
    y_lbl = "Posición s(t) [m]" if es_caso_real else "f(x)"
    fig = crear_figura_base_plotly(
        f"Diferenciación por Gregory ({'Ascendente Δ' if esquema_cod == 'ascendente' else 'Descendente ∇'}) en {x_eval:g}",
        x_lbl,
        y_lbl,
    )
    fig.add_trace(
        go.Scatter(
            x=curva_poly["x"],
            y=limpiar_valores_plotly(curva_poly["y"], limite_abs=1e9),
            mode="lines",
            name="Polinomio interpolador",
            line=dict(color=PALETA_ARANDU["salvia"], width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Lecturas radar" if es_caso_real else "Nodos medidos",
            marker=dict(size=8, color=PALETA_ARANDU["oro"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.4)),
        )
    )
    dx = res["h"] * 1.15
    x_loc = np.linspace(x_eval - dx, x_eval + dx, 40)
    if orden == 1:
        y_loc = y_en_x + res["derivada"] * (x_loc - x_eval)
        fig.add_trace(
            go.Scatter(
                x=x_loc.tolist(),
                y=y_loc.tolist(),
                mode="lines",
                name=f"Recta tangente (pendiente = {res['derivada']:.4f})",
                line=dict(color=PALETA_ARANDU["terracota"], width=2.6, dash="dash"),
            )
        )
    else:
        try:
            r_p1 = calcular_derivada_gregory_completa(xs, ys, x_eval, orden_derivada=1, esquema=esquema_cod)
            m_loc = float(r_p1["derivada"])
            y_loc = y_en_x + m_loc * (x_loc - x_eval) + 0.5 * res["derivada"] * ((x_loc - x_eval) ** 2)
            fig.add_trace(
                go.Scatter(
                    x=x_loc.tolist(),
                    y=y_loc.tolist(),
                    mode="lines",
                    name=f"Curvatura local (f'' = {res['derivada']:.4f})",
                    line=dict(color=PALETA_ARANDU["terracota"], width=2.5, dash="dash"),
                )
            )
        except Exception:
            pass

    fig.add_trace(
        go.Scatter(
            x=[x_eval],
            y=[y_en_x],
            mode="markers",
            name=f"Evaluación ({x_eval:g}, {y_en_x:.3f})",
            marker=dict(
                size=13,
                color=PALETA_ARANDU["marfil_nodo"],
                symbol="star",
                line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8),
            ),
        )
    )
    render_plotly_arandu(fig, key="plot_int_deriv")

    registrar_contexto_ia(
        "Derivada Numérica (Newton-Gregory)",
        f"- **Contexto:** {'Telemetría Radar s(t)' if es_caso_real else 'Tabla de Cátedra / Libre'} | "
        f"**Sentido:** `{esquema_cod}` | **Orden:** `{orden}` | **Punto:** `{x_eval:g}` | **{param_name}:** `{param_val:.4f}` | **Paso `h`:** `{res['h']:g}`\n"
        f"- **Resultado:** `{res['derivada']:.6f}`",
        clave="ing_deriv_res",
    )

    if modo_detallado:
        simb_diff = r"\Delta" if esquema_cod == "ascendente" else r"\nabla"
        pasos_der_vivos = [
            (
                f"**Ancla y parámetro adimensional:** Con paso uniforme $h = {res['h']:g}$ e índice base "
                f"$x_{{{res['indice_usado']}}} = {xs[res['indice_usado']]:g}$, calculamos "
                f"${param_name} = \\frac{{{x_eval:g} - {xs[res['indice_usado']]:g}}}{{{res['h']:g}}} = {param_val:.6f}$."
            )
        ]
        suma_aportes = 0.0
        for td in res["terminos_detalle"]:
            suma_aportes += td["aporte"]
            pasos_der_vivos.append(
                f"**Diferencia de orden {td['orden_diff']} (${simb_diff}^{{{td['orden_diff']}}} y$):** "
                f"Factor `{td['polinomio_param']}` $\\implies$ derivada `{td['derivada_param_expr']}` $= {td['coeficiente']:.6f}$ "
                f"$\\times$ `{td['diferencia']:.6g}` $= {td['aporte']:.6f}$."
            )
        denom_h = f"h = {res['h']:g}" if orden == 1 else f"h^2 = {res['h']**2:g}"
        pasos_der_vivos.append(
            f"**División final por ${denom_h}$:** $\\frac{{{suma_aportes:.6f}}}{{{res['h'] if orden == 1 else res['h']**2:g}}} = {res['derivada']:.6f}$."
        )
        render_deduccion_paso_a_paso_viva(
            f"Diferenciación ({esquema_cod.capitalize()}, orden {orden}, x*={x_eval:g})",
            pasos_der_vivos,
            clave_ia="ing_deriv_viva",
            expanded=True,
        )

        st.markdown("#### Tabla de Términos de Diferencias Finitas")
        filas_terminos = [
            {
                "Orden k": td["orden_diff"],
                "Polinomio en u/v": td["polinomio_param"],
                "Derivada del Factor": td["derivada_param_expr"],
                "Coeficiente Evaluado": round(float(td["coeficiente"]), 6),
                "Diferencia Finita (Δᵏy / ∇ᵏy)": round(float(td["diferencia"]), 6),
                "Aporte al Numerador": round(float(td["aporte"]), 6),
            }
            for td in res["terminos_detalle"]
        ]
        render_dataframe_acotado(filas_terminos, max_filas=300, key_prefix="tbl_int_deriv")


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
    modo_detallado = render_toggle_modo_detallado("diferenciacion")
    meta = render_bloque_1_teoria_y_ejemplo("diferenciacion_numerica", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = {
        "x": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "y": [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151],
        "x_eval": 12.4,
        **practica.get("datos_iniciales", {}),
    }
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_diferenciacion(defaults, modo_detallado=modo_detallado)
    if modo_detallado:
        st.divider()
        with st.expander("Simulador Geométrico: Recta Secante y Recta Tangente (Interpretación del límite)", expanded=False):
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


_FUNC_TELEMETRIA_POTENCIA = "2.5 + 1.5*sin(x) + 0.15*x**2"
_FUNC_TELEMETRIA_CAUDAL = "0.1*x**3 - 0.6*x**2 + 1.5*x + 4.0"
_FUNC_TOPOGRAFIA_LOTE = "sin(x) + 0.5*x + 1.5"
_SUBINDICES_S13 = ("₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈")


def _aplicar_preset_pasos(clave_state: str, pasos_nuevos: list[float]) -> None:
    """Fija una secuencia de pasos Δt_k para un caso real de cuadratura híbrida."""
    st.session_state[f"{clave_state}_real_n"] = len(pasos_nuevos)
    st.session_state[f"{clave_state}_real_pasos"] = list(pasos_nuevos)
    st.session_state.pop(f"ui_{clave_state}_real_n", None)
    for k in range(1, 9):
        st.session_state.pop(f"ui_{clave_state}_real_h_{k}", None)


def _al_cambiar_preset_libre_hibrido(clave_state: str, metodo_clave: str) -> None:
    """Sincroniza escenario seleccionado en el modo de función libre / TP6."""
    escenarios = _ESCENARIOS_INTEGRACION[metodo_clave]
    id_sel = st.session_state.get(f"ui_{clave_state}_libre_preset")
    esc = next((e for e in escenarios if e["id"] == id_sel), None)
    if esc is None:
        return
    xn, yn = _materializar_escenario_integracion(esc)
    pasos = [round(float(xn[i + 1] - xn[i]), 6) for i in range(len(xn) - 1)]
    st.session_state[f"{clave_state}_libre_x"] = xn
    st.session_state[f"{clave_state}_libre_y"] = yn
    st.session_state[f"{clave_state}_libre_f_str"] = esc["f_str"]
    st.session_state[f"{clave_state}_libre_nota"] = esc["nota"]
    st.session_state[f"{clave_state}_libre_pasos"] = pasos
    for k in range(1, 9):
        st.session_state.pop(f"ui_{clave_state}_libre_h_{k}", None)
    st.session_state.pop(f"{clave_state}_libre__editor_df", None)
    for suf_ui in ("fstr", "a", "b", "n", "x0", "np"):
        st.session_state.pop(f"ui_{clave_state}_libre_{suf_ui}", None)


def _sincronizar_pasos_lista(lista_base: list[float], n_objetivo: int, valor_relleno: float = 1.0) -> list[float]:
    if len(lista_base) == n_objetivo:
        return list(lista_base)
    if len(lista_base) > n_objetivo:
        return list(lista_base[:n_objetivo])
    return list(lista_base) + [valor_relleno] * (n_objetivo - len(lista_base))


def _render_inputs_pasos_h(
    pasos_actuales: list[float],
    prefijo_key: str,
    etiqueta_prefijo: str,
    min_h: float = 0.1,
    max_h: float = 10.0,
    step_h: float = 0.1,
) -> list[float]:
    """Renderiza inputs numéricos por cada subintervalo para editar pasos h_k en vivo."""
    n = len(pasos_actuales)
    nuevos: list[float] = []
    cols_por_fila = min(4, n) if n > 0 else 1
    for ini in range(0, n, cols_por_fila):
        fin = min(ini + cols_por_fila, n)
        cols = st.columns(fin - ini)
        for offset, idx in enumerate(range(ini, fin)):
            sub_txt = _SUBINDICES_S13[idx] if idx < len(_SUBINDICES_S13) else f"_{idx + 1}"
            w_key = f"{prefijo_key}_{idx + 1}"
            if w_key not in st.session_state:
                st.session_state[w_key] = float(pasos_actuales[idx])
            val = cols[offset].number_input(
                f"{etiqueta_prefijo}{sub_txt}:",
                min_value=float(min_h),
                max_value=float(max_h),
                step=float(step_h),
                format="%.2f",
                key=w_key,
            )
            nuevos.append(float(val))
    return nuevos


def _render_malla_libre_hibrida(
    clave_state: str,
    metodo_clave: str,
    defaults: dict,
    ejemplo_defecto: str,
) -> tuple[list[float], list[float], str | None] | None:
    """Renderiza el selector de escenario libre y edición por malla uniforme, pasos h_k o tabla."""
    escenarios = _ESCENARIOS_INTEGRACION[metodo_clave]
    ids_esc = [e["id"] for e in escenarios]
    mapa_nom = {e["id"]: e["nombre"] for e in escenarios}
    if f"{clave_state}_libre_x" not in st.session_state:
        st.session_state[f"{clave_state}_libre_x"] = list(defaults["x"])
        st.session_state[f"{clave_state}_libre_y"] = list(defaults["y"])
        st.session_state[f"{clave_state}_libre_f_str"] = str(defaults.get("f_str", "x**2 / 5"))
        st.session_state[f"{clave_state}_libre_nota"] = escenarios[0]["nota"]
        st.session_state[f"{clave_state}_libre_pasos"] = [
            round(float(defaults["x"][i + 1] - defaults["x"][i]), 6) for i in range(len(defaults["x"]) - 1)
        ]

    preset_key = f"ui_{clave_state}_libre_preset"
    if preset_key not in st.session_state or st.session_state[preset_key] not in ids_esc:
        st.session_state[preset_key] = ids_esc[0]

    esc_id = st.selectbox(
        "Escenario o función de estudio:",
        options=ids_esc,
        format_func=lambda k_e: mapa_nom[k_e],
        key=preset_key,
        on_change=_al_cambiar_preset_libre_hibrido,
        args=(clave_state, metodo_clave),
    )
    esc_actual = next((e for e in escenarios if e["id"] == esc_id), escenarios[0])
    st.caption(st.session_state.get(f"{clave_state}_libre_nota", esc_actual["nota"]))

    c_f, c_modo_malla = st.columns([1.2, 1.5])
    with c_f:
        f_libre = st.text_input(
            "Función $f(x)$ (opcional si editás nodos manuales):",
            value=str(st.session_state[f"{clave_state}_libre_f_str"]),
            key=f"ui_{clave_state}_libre_fstr",
        )
        st.session_state[f"{clave_state}_libre_f_str"] = f_libre
    with c_modo_malla:
        modo_malla = st.radio(
            "Definición de la malla:",
            options=["uniforme", "pasos", "tabla"],
            format_func=lambda m: (
                "Malla uniforme [a, b]"
                if m == "uniforme"
                else ("Editar pasos hₖ por tramo" if m == "pasos" else "Tabla editable de nodos")
            ),
            horizontal=True,
            key=f"ui_{clave_state}_libre_malla",
        )

    xs_cur = list(st.session_state[f"{clave_state}_libre_x"])
    ys_cur = list(st.session_state[f"{clave_state}_libre_y"])
    f_limpia = f_libre.strip() or None

    try:
        if modo_malla == "uniforme":
            ca, cb, cn = st.columns([1.0, 1.0, 1.3])
            a_in = ca.number_input("Límite inferior a:", value=float(xs_cur[0]), step=0.5, format="%.2f", key=f"ui_{clave_state}_libre_a")
            b_in = cb.number_input("Límite superior b:", value=float(xs_cur[-1]), step=0.5, format="%.2f", key=f"ui_{clave_state}_libre_b")
            n_in = cn.slider(
                "Subintervalos n:",
                min_value=2,
                max_value=16,
                value=min(16, max(2, len(xs_cur) - 1)),
                key=f"ui_{clave_state}_libre_n",
            )
            if b_in <= a_in + 1e-6:
                st.warning("El límite superior b debe ser mayor que a.")
                return None
            mismos_extremos = (
                f_limpia is None
                and abs(float(a_in) - float(xs_cur[0])) < 1e-6
                and abs(float(b_in) - float(xs_cur[-1])) < 1e-6
            )
            xs, ys, _ = remuestrear_tabla_integracion(
                xs_cur if mismos_extremos else [float(a_in), float(b_in)],
                ys_cur if mismos_extremos else [float(ys_cur[0]), float(ys_cur[-1])],
                nuevo_n=int(n_in),
                f_str=f_limpia,
            )
            st.session_state[f"{clave_state}_libre_x"] = xs
            st.session_state[f"{clave_state}_libre_y"] = ys
            st.session_state[f"{clave_state}_libre_pasos"] = [
                round(float(xs[i + 1] - xs[i]), 6) for i in range(len(xs) - 1)
            ]
        elif modo_malla == "pasos":
            c_x0, c_np = st.columns([1.0, 1.5])
            x0_in = c_x0.number_input("Origen x₀:", value=float(xs_cur[0]), step=0.5, format="%.2f", key=f"ui_{clave_state}_libre_x0")
            n_pasos = int(
                c_np.slider(
                    "Cantidad de subintervalos n:",
                    min_value=2,
                    max_value=8,
                    value=min(8, max(2, len(xs_cur) - 1)),
                    key=f"ui_{clave_state}_libre_np",
                )
            )
            h_def = round(float(xs_cur[1] - xs_cur[0]), 4) if len(xs_cur) >= 2 else 1.0
            pasos_sync = _sincronizar_pasos_lista(st.session_state.get(f"{clave_state}_libre_pasos", [h_def]), n_pasos, max(0.2, h_def))
            pasos_edit = _render_inputs_pasos_h(
                pasos_sync,
                prefijo_key=f"ui_{clave_state}_libre_h",
                etiqueta_prefijo="h",
                min_h=0.1,
                max_h=20.0,
                step_h=0.5,
            )
            st.session_state[f"{clave_state}_libre_pasos"] = pasos_edit
            xs, ys = reconstruir_malla_por_pasos(
                float(x0_in),
                pasos_edit,
                f_str=f_limpia,
                y_base=ys_cur if not f_limpia else None,
            )
            st.session_state[f"{clave_state}_libre_x"] = xs
            st.session_state[f"{clave_state}_libre_y"] = ys
        else:
            df_in = pd.DataFrame({"x": xs_cur, "y": ys_cur})
            df_edit = render_data_editor_arandu(
                df_in,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "x": st.column_config.NumberColumn("Abscisa xᵢ", format="%.4f"),
                    "y": st.column_config.NumberColumn("Ordenada yᵢ", format="%.4f"),
                },
                key=f"{clave_state}_libre__editor_df",
            )
            df_limpia = df_edit.dropna()
            if len(df_limpia) < 2:
                st.warning("Ingresá al menos 2 nodos válidos en la tabla.")
                return None
            xs = [float(v) for v in df_limpia["x"].tolist()]
            ys = [float(v) for v in df_limpia["y"].tolist()]
            st.session_state[f"{clave_state}_libre_x"] = xs
            st.session_state[f"{clave_state}_libre_y"] = ys
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Configuración de malla o función f(x)",
            sugerencia_defecto="Verificá que los pasos h_k sean positivos y la expresión de f(x) use sintaxis Python válida.",
            ejemplo_defecto=ejemplo_defecto,
        )
        return None

    return xs, ys, f_limpia


def _plot_cuadratura_hibrida(
    res: dict[str, Any],
    xs: list[float],
    ys: list[float],
    titulo_plot: str,
    x_lbl: str,
    y_lbl: str,
    nombre_exacta: str,
    nombre_nodos: str,
    plot_key: str,
    color_exacta: str,
) -> None:
    """Grafica diferenciando Simpson 3/8 (cúbica), Simpson 1/3 (parábola) y Trapecio (recta)."""
    fig = crear_figura_base_plotly(titulo_plot, x_lbl, y_lbl)
    estilos_regla = {
        "Simpson 3/8": (PALETA_ARANDU["oro"], PALETA_ARANDU["oro_fill"], "solid", "Cúbica"),
        "Simpson 1/3": (PALETA_ARANDU["salvia"], PALETA_ARANDU["salvia_fill"], "solid", "Parábola"),
        "Trapecio": (PALETA_ARANDU["terracota"], PALETA_ARANDU["terracota_fill"], "dash", "Respaldo"),
    }

    for idx, b in enumerate(res["bloques"], start=1):
        regla_b = b["regla"]
        color_b, fill_b, dash_b, tipo_b = estilos_regla.get(
            regla_b, (PALETA_ARANDU["salvia"], PALETA_ARANDU["salvia_fill"], "solid", "")
        )
        etiqueta = f"Tramo {idx}: {regla_b} ({tipo_b}, h={b['h']:g})"
        fig.add_trace(
            go.Scatter(
                x=b["x_curva"],
                y=limpiar_valores_plotly(b["y_curva"], limite_abs=1e9),
                mode="lines",
                fill="tozeroy",
                fillcolor=fill_b,
                name=etiqueta,
                line=dict(color=color_b, width=2.6, dash=dash_b),
            )
        )

    stem_x: list[float | None] = []
    stem_y: list[float | None] = []
    for nx, ny in zip(xs, ys):
        stem_x.extend([nx, nx, None])
        stem_y.extend([0.0, ny, None])
    fig.add_trace(
        go.Scatter(
            x=stem_x,
            y=stem_y,
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color=PALETA_ARANDU["texto_atenuado"], width=1.1, dash="dot"),
        )
    )

    if res.get("exacta") and res["exacta"].get("curva_exacta"):
        curva_ex = res["exacta"]["curva_exacta"]
        fig.add_trace(
            go.Scatter(
                x=curva_ex["x"],
                y=limpiar_valores_plotly(curva_ex["y"], limite_abs=1e9),
                mode="lines",
                name=nombre_exacta,
                line=dict(color=color_exacta, width=2.4),
            )
        )

    pond = res.get("ponderacion_nodal") or {}
    nodos_pond = pond.get("nodos", [])
    ver_etq = len(xs) <= 12
    custom_nodos = [[nd["weight"], nd["rol"]] for nd in nodos_pond] if len(nodos_pond) == len(xs) else None
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text" if ver_etq else "markers",
            name=nombre_nodos,
            text=[f"x{i}" for i in range(len(xs))] if ver_etq else None,
            textposition="top center",
            customdata=custom_nodos,
            hovertemplate=(
                "x=%{x:.4f}<br>y=%{y:.4f}<br>Peso w=%{customdata[0]} (%{customdata[1]})<extra></extra>"
                if custom_nodos is not None
                else "x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>"
            ),
            marker=dict(
                size=9,
                color=PALETA_ARANDU["marfil_nodo"],
                line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.6),
            ),
        )
    )

    render_plotly_arandu(fig, key=plot_key)


def _render_detalle_cuadratura_hibrida(
    res: dict[str, Any],
    clave_state: str,
    titulo_deduccion: str,
    unidad_txt: str,
) -> None:
    """Despliega tarjetas de pesos, deducción viva y tablas con encabezados en español."""
    pond = res.get("ponderacion_nodal")
    if pond:
        st.markdown(_construir_html_ponderacion_nodal(pond, clave_state), unsafe_allow_html=True)

    pasos_vivos = [
        f"**Malla analizada ({res['n_intervalos_total']} subintervalos):** pasos $h = {[round(v, 4) for v in res['pasos_h']]}$.",
    ]
    if res.get("es_uniforme") and res.get("P") is not None and res.get("I") is not None:
        pasos_vivos.append(
            f"**Acumuladores de cátedra ($E, P, I$):** $E = {res['E']:.4f}$, $P = {res['P']:.4f}$, $I = {res['I']:.4f}$."
        )
    for idx_b, b in enumerate(res["bloques"], start=1):
        pasos_vivos.append(
            f"**Tramo {idx_b} — {b['regla']} en {b['intervalo']}:** {b['motivo']} "
            f"`{b['formula']}` $\\implies$ aporte $= {b['area']:.6f}{unidad_txt}$."
        )
    pasos_vivos.append(f"**Suma total:** `{res['area_total']:.6f}{unidad_txt}` ({res['reglas_aplicadas']}).")
    render_deduccion_paso_a_paso_viva(
        titulo_deduccion,
        pasos_vivos,
        clave_ia=f"{clave_state}_viva",
        expanded=True,
    )

    st.markdown("#### Tabla Completa de Tramos Integrados")
    filas_tramos = [
        {
            "Tramo": idx_b,
            "Regla Aplicada": b["regla"],
            "Nodos": b["nodos_label"],
            "Intervalo": b["intervalo"],
            "Paso h": round(b["h"], 6),
            "Fórmula Evaluada": b["formula"],
            "Resultado Parcial": round(b["area"], 6),
            "Criterio Geométrico": b["motivo"],
        }
        for idx_b, b in enumerate(res["bloques"], start=1)
    ]
    render_dataframe_acotado(filas_tramos, max_filas=300, key_prefix=f"tbl_{clave_state}_tramos")

    if pond:
        st.markdown("#### Tabla de Evaluación Nodal")
        filas_nodos = [
            {
                "Índice i": nd["index"],
                "Nodo xᵢ": round(nd["x"], 4),
                "Valor f(xᵢ)": round(nd["y"], 5),
                "Peso Nodal wᵢ": nd["weight"],
                "Rol": nd["rol"],
                "Producto wᵢ·f(xᵢ)": round(nd["wy"], 5),
                "Aporte al Total": round(nd["aporte"], 6),
            }
            for nd in pond["nodos"]
        ]
        render_dataframe_acotado(filas_nodos, max_filas=300, key_prefix=f"tbl_{clave_state}_nodos")


@st.fragment
def _frag_simpson_1_3(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_telemetria", "funcion_libre"],
        format_func=lambda op: (
            "Caso Real: Telemetría de Potencia P(t) → Energía (kWh)"
            if op == "real_telemetria"
            else "Función Matemática Libre f(x) / Cátedra TP6"
        ),
        horizontal=True,
        key="ui_int_s13_contexto",
    )
    es_caso_real = contexto == "real_telemetria"

    if es_caso_real:
        st.caption(
            r"Medidor inteligente desde $t_0 = 0\text{ h}$ sobre $P(t) = 2.5 + 1.5\sin(t) + 0.15t^2\;\text{[kW]}$. "
            r"Modificá los intervalos de muestreo $\Delta t_k$ para provocar un cambio de paso y observar la caída automática a Trapecio."
        )
        if "int_s13_real_pasos" not in st.session_state:
            st.session_state["int_s13_real_pasos"] = [1.0, 1.0, 1.0, 1.0]
        if "int_s13_real_n" not in st.session_state:
            st.session_state["int_s13_real_n"] = len(st.session_state["int_s13_real_pasos"])

        b_unif, b_jit, b_imp = st.columns(3)
        if b_unif.button(
            "Muestreo uniforme (Δt = 1.0 h constante → 100% Simpson 1/3)",
            key="btn_s13_real_unif",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_s13", [1.0, 1.0, 1.0, 1.0])
            rerun_fragmento_seguro()
        if b_jit.button(
            "Jitter de red / Cambio de paso (Δt = [1.0, 1.0, 1.5, 0.5] h → Simpson 1/3 + Trapecio)",
            key="btn_s13_real_jit",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_s13", [1.0, 1.0, 1.5, 0.5])
            rerun_fragmento_seguro()
        if b_imp.button(
            "Muestreo impar (3 subintervalos de Δt = 1.0 h → 1 parábola + 1 Trapecio)",
            key="btn_s13_real_imp",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_s13", [1.0, 1.0, 1.0])
            rerun_fragmento_seguro()

        n_real = int(
            st.slider(
                "Cantidad de subintervalos de muestreo ($n$):",
                min_value=2,
                max_value=6,
                value=int(st.session_state["int_s13_real_n"]),
                key="ui_int_s13_real_n",
            )
        )
        st.session_state["int_s13_real_n"] = n_real
        pasos_sync = _sincronizar_pasos_lista(st.session_state["int_s13_real_pasos"], n_real, 1.0)
        pasos_edit = _render_inputs_pasos_h(
            pasos_sync,
            prefijo_key="ui_int_s13_real_h",
            etiqueta_prefijo="Δt",
            min_h=0.1,
            max_h=5.0,
            step_h=0.1,
        )
        st.session_state["int_s13_real_pasos"] = pasos_edit
        f_str_activo = _FUNC_TELEMETRIA_POTENCIA
        try:
            xs, ys = reconstruir_malla_por_pasos(0.0, pasos_edit, f_str=f_str_activo)
        except Exception as exc:
            render_excepcion_validacion(
                exc,
                campo="Intervalos de muestreo Δt",
                sugerencia_defecto="Ingresá intervalos Δt_k positivos (por ejemplo 1.0, 1.0, 1.5, 0.5).",
                ejemplo_defecto="Δt = [1.0, 1.0, 1.5, 0.5]",
            )
            return
    else:
        malla_libre = _render_malla_libre_hibrida(
            "int_s13",
            "simpson_1_3",
            defaults,
            "f(x) = x**2 / 5, pasos h = [5.0, 5.0, 5.0, 5.0]",
        )
        if malla_libre is None:
            return
        xs, ys, f_str_activo = malla_libre

    try:
        res = integrar_simpson_1_3_hibrido(xs, ys, f_str=f_str_activo)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos (x, y)",
            sugerencia_defecto="Asegurá al menos 2 nodos con valores x estrictamente crecientes y finitos.",
            ejemplo_defecto="x = [0.0, 1.0, 2.0, 3.5, 4.0]",
        )
        return

    if res["tramos_trapecio"] > 0:
        for aviso in res["avisos_pedagogicos"]:
            st.warning(aviso)
    else:
        st.success(
            "Todos los tramos mantienen 3 puntos equidistantes: se aplica **Simpson 1/3** puro (orden $\\mathcal{O}(h^4)$)."
        )

    k1, k2, k3 = st.columns(3)
    if es_caso_real:
        delta_e = (
            f"Exacta: {res['exacta']['integral_exacta']:.4f} kWh · Err {res['exacta']['error_relativo_pct']:.3f}%"
            if res.get("exacta")
            else f"{res['n_intervalos_total']} intervalos"
        )
        k1.metric("Resultado Total (Energía)", f"{res['area_total']:.4f} kWh", delta_e, delta_color="off")
    else:
        delta_i = (
            f"Exacta: {res['exacta']['integral_exacta']:.6f} · Err {res['exacta']['error_relativo_pct']:.4f}%"
            if res.get("exacta")
            else f"{res['n_intervalos_total']} intervalos"
        )
        k1.metric("Resultado Total (Integral)", f"{res['area_total']:.6f}", delta_i, delta_color="off")

    k2.metric(
        "Tramos con Simpson 1/3",
        str(res["tramos_simpson_1_3"]),
        f"{res['subintervalos_simpson_1_3']} subintervalos (3 pts equidistantes c/u)",
        delta_color="off",
    )
    k3.metric(
        "Tramos resueltos por Trapecio",
        str(res["tramos_trapecio"]),
        (
            "0 (sin caída)"
            if res["tramos_trapecio"] == 0
            else f"{res['subintervalos_trapecio']} subintervalo(s) por cambio de h / paridad"
        ),
        delta_color="off",
    )

    _plot_cuadratura_hibrida(
        res,
        xs,
        ys,
        titulo_plot=(
            f"Energía Acumulada E = ∫ P(t) dt ({res['reglas_aplicadas']})"
            if es_caso_real
            else f"Regla de Simpson 1/3 Híbrida ({res['reglas_aplicadas']})"
        ),
        x_lbl="Tiempo t (horas)" if es_caso_real else "x",
        y_lbl="Potencia P(t) [kW]" if es_caso_real else "f(x)",
        nombre_exacta="Curva exacta P(t)" if es_caso_real else "Curva exacta f(x)",
        nombre_nodos="Muestras medidas" if es_caso_real else "Nodos (xᵢ, yᵢ)",
        plot_key="plot_int_s13_hibrido",
        color_exacta=PALETA_ARANDU["oro"],
    )

    unidad_txt = " kWh" if es_caso_real else ""
    exacta_ia = (
        f" | **Exacta:** `{res['exacta']['integral_exacta']:.6f}{unidad_txt}` (Error rel: `{res['exacta']['error_relativo_pct']:.4f}%`)"
        if res.get("exacta")
        else ""
    )
    registrar_contexto_ia(
        "Resultado Simpson 1/3 Híbrido",
        f"- **Contexto:** {'Telemetría de Potencia P(t)' if es_caso_real else 'Función Libre / TP6'}\n"
        f"- **Resultado total:** `{res['area_total']:.6f}{unidad_txt}`{exacta_ia}\n"
        f"- **Tramos Simpson 1/3:** `{res['tramos_simpson_1_3']}` | **Tramos Trapecio:** `{res['tramos_trapecio']}`",
        clave="int_s13_res",
    )

    if modo_detallado:
        _render_detalle_cuadratura_hibrida(
            res,
            "int_s13",
            f"Simpson 1/3 Híbrido ({res['reglas_aplicadas']})",
            unidad_txt,
        )


def _render_simpson_1_3_tab() -> None:
    modo_detallado = render_toggle_modo_detallado("simpson_1_3")
    meta = render_bloque_1_teoria_y_ejemplo("simpson_1_3", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "x": [0.5, 5.5, 10.5, 15.5, 20.5],
            "y": [0.05, 6.05, 22.05, 48.05, 84.05],
            "f_str": "x**2 / 5",
        },
    )
    if "f_str" not in defaults:
        defaults = {**defaults, "f_str": "x**2 / 5"}
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_simpson_1_3(defaults, modo_detallado=modo_detallado)


@st.fragment
def _frag_simpson_3_8(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_caudal", "funcion_libre"],
        format_func=lambda op: (
            "Caso Real: Telemetría de Caudal Q(t) → Volumen de Agua (m³)"
            if op == "real_caudal"
            else "Función Matemática Libre f(x) / Cátedra TP6"
        ),
        horizontal=True,
        key="ui_int_s38_contexto",
    )
    es_caso_real = contexto == "real_caudal"

    if es_caso_real:
        st.caption(
            r"Caudalímetro desde $t_0 = 0\text{ h}$ sobre $Q(t) = 0.1t^3 - 0.6t^2 + 1.5t + 4.0\;\text{[m}^3/\text{h]}$. "
            r"Con $n = 5$ lecturas equidistantes combina **Simpson 3/8 + 1/3** preservando orden $\mathcal{O}(h^4)$; modificá $\Delta t_k$ para activar el respaldo por **Trapecio**."
        )
        if "int_s38_real_pasos" not in st.session_state:
            st.session_state["int_s38_real_pasos"] = [1.0, 1.0, 1.0, 1.0, 1.0]
        if "int_s38_real_n" not in st.session_state:
            st.session_state["int_s38_real_n"] = len(st.session_state["int_s38_real_pasos"])

        b_comb, b_puro, b_jit = st.columns(3)
        if b_comb.button(
            "Turno de 5 h (n=5, Δt=1.0 h → Combinado 3/8 + 1/3 exacto)",
            key="btn_s38_real_comb",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_s38", [1.0, 1.0, 1.0, 1.0, 1.0])
            rerun_fragmento_seguro()
        if b_puro.button(
            "Múltiplo de 3 (n=3, Δt=1.0 h → 100% Simpson 3/8 puro)",
            key="btn_s38_real_puro",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_s38", [1.0, 1.0, 1.0])
            rerun_fragmento_seguro()
        if b_jit.button(
            "Cambio de muestreo (Δt = [1.0, 1.0, 1.0, 1.5, 0.5] h → 3/8 + Caída a Trapecio)",
            key="btn_s38_real_jit",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_s38", [1.0, 1.0, 1.0, 1.5, 0.5])
            rerun_fragmento_seguro()

        n_real = int(
            st.slider(
                "Cantidad de subintervalos de muestreo ($n$):",
                min_value=3,
                max_value=8,
                value=int(st.session_state["int_s38_real_n"]),
                key="ui_int_s38_real_n",
            )
        )
        st.session_state["int_s38_real_n"] = n_real
        pasos_sync = _sincronizar_pasos_lista(st.session_state["int_s38_real_pasos"], n_real, 1.0)
        pasos_edit = _render_inputs_pasos_h(
            pasos_sync,
            prefijo_key="ui_int_s38_real_h",
            etiqueta_prefijo="Δt",
            min_h=0.1,
            max_h=5.0,
            step_h=0.1,
        )
        st.session_state["int_s38_real_pasos"] = pasos_edit
        f_str_activo = _FUNC_TELEMETRIA_CAUDAL
        try:
            xs, ys = reconstruir_malla_por_pasos(0.0, pasos_edit, f_str=f_str_activo)
        except Exception as exc:
            render_excepcion_validacion(
                exc,
                campo="Intervalos de muestreo Δt",
                sugerencia_defecto="Ingresá intervalos Δt_k positivos (por ejemplo 1.0, 1.0, 1.0, 1.0, 1.0).",
                ejemplo_defecto="Δt = [1.0, 1.0, 1.0, 1.5, 0.5]",
            )
            return
    else:
        malla_libre = _render_malla_libre_hibrida(
            "int_s38",
            "simpson_3_8",
            defaults,
            "f(x) = x**2 / 5, pasos h = [5.0, 5.0, 5.0, 5.0, 5.0]",
        )
        if malla_libre is None:
            return
        xs, ys, f_str_activo = malla_libre

    try:
        res = integrar_simpson_3_8_hibrido(xs, ys, f_str=f_str_activo)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos (x, y)",
            sugerencia_defecto="Asegurá al menos 2 nodos con valores x estrictamente crecientes y finitos.",
            ejemplo_defecto="x = [0.0, 1.0, 2.0, 3.0, 4.5, 5.0]",
        )
        return

    if not res["avisos_pedagogicos"]:
        st.success(
            "Todos los tramos forman bloques de 4 puntos equidistantes: se aplica **Simpson 3/8** puro (orden $\\mathcal{O}(h^4)$)."
        )
    else:
        for aviso in res["avisos_pedagogicos"]:
            if "Trapecio" in aviso:
                st.warning(aviso)
            else:
                st.info(aviso)

    k1, k2, k3 = st.columns(3)
    if es_caso_real:
        delta_v = (
            f"Exacta: {res['exacta']['integral_exacta']:.4f} m³ · Err {res['exacta']['error_relativo_pct']:.3f}%"
            if res.get("exacta")
            else f"{res['n_intervalos_total']} intervalos"
        )
        k1.metric("Resultado Total (Volumen)", f"{res['area_total']:.4f} m³", delta_v, delta_color="off")
    else:
        delta_i = (
            f"Exacta: {res['exacta']['integral_exacta']:.6f} · Err {res['exacta']['error_relativo_pct']:.4f}%"
            if res.get("exacta")
            else f"{res['n_intervalos_total']} intervalos"
        )
        k1.metric("Resultado Total (Integral)", f"{res['area_total']:.6f}", delta_i, delta_color="off")

    k2.metric(
        "Tramos con Simpson 3/8",
        str(res["tramos_simpson_3_8"]),
        f"{res['subintervalos_simpson_3_8']} subintervalos (4 pts equidistantes c/u)",
        delta_color="off",
    )
    if res["tramos_simpson_1_3"] == 0 and res["tramos_trapecio"] == 0:
        delta_empalme = "Sin empalme ni caída"
    else:
        delta_empalme = (
            f"{res['subintervalos_simpson_1_3']} subinterv. 1/3 · {res['subintervalos_trapecio']} Trapecio"
        )
    k3.metric(
        "Empalme Simpson 1/3 y Trapecio",
        f"{res['tramos_simpson_1_3']} Simpson 1/3 · {res['tramos_trapecio']} Trapecio",
        delta_empalme,
        delta_color="off",
    )

    _plot_cuadratura_hibrida(
        res,
        xs,
        ys,
        titulo_plot=(
            f"Volumen Acumulado V = ∫ Q(t) dt ({res['reglas_aplicadas']})"
            if es_caso_real
            else f"Simpson 3/8 y Esquema Combinado ({res['reglas_aplicadas']})"
        ),
        x_lbl="Tiempo t (horas)" if es_caso_real else "x",
        y_lbl="Caudal Q(t) [m³/h]" if es_caso_real else "f(x)",
        nombre_exacta="Curva exacta Q(t)" if es_caso_real else "Curva exacta f(x)",
        nombre_nodos="Lecturas medidas" if es_caso_real else "Nodos (xᵢ, yᵢ)",
        plot_key="plot_int_s38_hibrido",
        color_exacta=PALETA_ARANDU["texto_principal"],
    )

    unidad_txt = " m³" if es_caso_real else ""
    exacta_ia = (
        f" | **Exacta:** `{res['exacta']['integral_exacta']:.6f}{unidad_txt}` (Error rel: `{res['exacta']['error_relativo_pct']:.4f}%`)"
        if res.get("exacta")
        else ""
    )
    registrar_contexto_ia(
        "Resultado Simpson 3/8 Híbrido",
        f"- **Contexto:** {'Telemetría de Caudal Q(t)' if es_caso_real else 'Función Libre / TP6'}\n"
        f"- **Resultado total:** `{res['area_total']:.6f}{unidad_txt}`{exacta_ia}\n"
        f"- **Tramos Simpson 3/8:** `{res['tramos_simpson_3_8']}` | **Simpson 1/3:** `{res['tramos_simpson_1_3']}` | **Trapecio:** `{res['tramos_trapecio']}`",
        clave="int_s38_res",
    )

    if modo_detallado:
        _render_detalle_cuadratura_hibrida(
            res,
            "int_s38",
            f"Simpson 3/8 y Combinado ({res['reglas_aplicadas']})",
            unidad_txt,
        )


def _render_simpson_3_8_tab() -> None:
    modo_detallado = render_toggle_modo_detallado("simpson_3_8")
    meta = render_bloque_1_teoria_y_ejemplo("simpson_3_8_combinado", modo_detallado=modo_detallado)
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
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_simpson_3_8(defaults, modo_detallado=modo_detallado)


@st.fragment
def _frag_trapecio(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_topografia", "funcion_libre"],
        format_func=lambda op: (
            "Caso Real: Agrimensura de Lote Ribereño y(x) → Área (m²)"
            if op == "real_topografia"
            else "Función Matemática Libre f(x) / Cátedra TP6"
        ),
        horizontal=True,
        key="ui_int_trap_contexto",
    )
    es_caso_real = contexto == "real_topografia"

    if es_caso_real:
        st.caption(
            r"Relevamiento topográfico desde estaca $x_0 = 0\text{ m}$ con ancho ribereño $y(x) = \sin(x) + 0.5x + 1.5\;\text{[m]}$. "
            r"Probá estacas uniformes o espaciamientos variables $\Delta x_k$ sin perder validez."
        )
        if "int_trap_real_pasos" not in st.session_state:
            st.session_state["int_trap_real_pasos"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        if "int_trap_real_n" not in st.session_state:
            st.session_state["int_trap_real_n"] = len(st.session_state["int_trap_real_pasos"])

        b_unif, b_obst, b_gruesa = st.columns(3)
        if b_unif.button(
            "Estacas uniformes (6 tramos de Δx = 0.5 m → Regla Compuesta E, P, I)",
            key="btn_trap_real_unif",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_trap", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            rerun_fragmento_seguro()
        if b_obst.button(
            "Terreno con obstáculos (Δx = [0.5, 0.5, 1.2, 0.8, 0.5, 0.5] m → Paso Variable)",
            key="btn_trap_real_obst",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_trap", [0.5, 0.5, 1.2, 0.8, 0.5, 0.5])
            rerun_fragmento_seguro()
        if b_gruesa.button(
            "Malla gruesa (3 tramos de Δx = 1.0 m → Mayor error de curvatura O(h²))",
            key="btn_trap_real_gruesa",
            use_container_width=True,
        ):
            _aplicar_preset_pasos("int_trap", [1.0, 1.0, 1.0])
            rerun_fragmento_seguro()

        n_real = int(
            st.slider(
                "Cantidad de tramos entre estacas ($n$):",
                min_value=2,
                max_value=8,
                value=int(st.session_state["int_trap_real_n"]),
                key="ui_int_trap_real_n",
            )
        )
        st.session_state["int_trap_real_n"] = n_real
        pasos_sync = _sincronizar_pasos_lista(st.session_state["int_trap_real_pasos"], n_real, 0.5)
        pasos_edit = _render_inputs_pasos_h(
            pasos_sync,
            prefijo_key="ui_int_trap_real_h",
            etiqueta_prefijo="Δx",
            min_h=0.1,
            max_h=5.0,
            step_h=0.1,
        )
        st.session_state["int_trap_real_pasos"] = pasos_edit
        f_str_activo = _FUNC_TOPOGRAFIA_LOTE
        try:
            xs, ys = reconstruir_malla_por_pasos(0.0, pasos_edit, f_str=f_str_activo)
        except Exception as exc:
            render_excepcion_validacion(
                exc,
                campo="Distancias entre estacas Δx",
                sugerencia_defecto="Ingresá distancias Δx_k positivas (por ejemplo 0.5, 0.5, 1.2, 0.8, 0.5, 0.5).",
                ejemplo_defecto="Δx = [0.5, 0.5, 1.2, 0.8, 0.5, 0.5]",
            )
            return
    else:
        malla_libre = _render_malla_libre_hibrida(
            "int_trap",
            "trapecio",
            defaults,
            "f(x) = x**2 / 5, pasos h = [5.0, 5.0, 5.0, 5.0, 5.0]",
        )
        if malla_libre is None:
            return
        xs, ys, f_str_activo = malla_libre

    try:
        res = integrar_trapecio_hibrido(xs, ys, f_str=f_str_activo)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Tabla de nodos (x, y)",
            sugerencia_defecto="Asegurá al menos 2 nodos con valores x estrictamente crecientes y finitos.",
            ejemplo_defecto="x = [0.5, 5.5, 10.5, 15.5, 20.5, 25.5]",
        )
        return

    if res["es_uniforme"]:
        st.success(
            f"Malla uniforme ($h = {res['pasos_h'][0]:g}$): aplica directamente la regla compuesta $\\frac{{h}}{{2}}[E + 2P + 2I]$."
        )
    else:
        for aviso in res["avisos_pedagogicos"]:
            st.info(aviso)

    k1, k2, k3 = st.columns(3)
    if es_caso_real:
        delta_s = (
            f"Exacta: {res['exacta']['integral_exacta']:.4f} m² · Err {res['exacta']['error_relativo_pct']:.3f}%"
            if res.get("exacta")
            else f"{res['n_intervalos_total']} tramos"
        )
        k1.metric("Resultado Total (Superficie)", f"{res['area_total']:.4f} m²", delta_s, delta_color="off")
        simpson_val_txt = f"{res['area_simpson_ref']:.4f} m²"
    else:
        delta_i = (
            f"Exacta: {res['exacta']['integral_exacta']:.6f} · Err {res['exacta']['error_relativo_pct']:.4f}%"
            if res.get("exacta")
            else f"{res['n_intervalos_total']} tramos"
        )
        k1.metric("Resultado Total (Integral)", f"{res['area_total']:.6f}", delta_i, delta_color="off")
        simpson_val_txt = f"{res['area_simpson_ref']:.6f}"

    k2.metric(
        "Tramos Trapezoidales Activos",
        f"{res['tramos_trapecio']} trapecios",
        "Paso h constante (fórmula E, P, I)" if res["es_uniforme"] else "Paso hₖ variable por tramo",
        delta_color="off",
    )
    k3.metric(
        "Contraste vs. Simpson 1/3 Híbrido",
        simpson_val_txt,
        f"{res['tramos_simpson_posibles']} parábola(s) posibles en esta malla",
        delta_color="off",
    )

    _plot_cuadratura_hibrida(
        res,
        xs,
        ys,
        titulo_plot=(
            f"Superficie Ribereña por Trapecios ({'h constante' if res['es_uniforme'] else 'h variable'})"
            if es_caso_real
            else f"Regla de los Trapecios ({'h constante' if res['es_uniforme'] else 'h variable'})"
        ),
        x_lbl="Distancia x (m)" if es_caso_real else "x",
        y_lbl="Ancho y(x) [m]" if es_caso_real else "f(x)",
        nombre_exacta="Ribera exacta y(x)" if es_caso_real else "Curva exacta f(x)",
        nombre_nodos="Estacas medidas" if es_caso_real else "Nodos (xᵢ, yᵢ)",
        plot_key="plot_int_trap_hibrido",
        color_exacta=PALETA_ARANDU["oro"],
    )

    unidad_txt = " m²" if es_caso_real else ""
    exacta_ia = (
        f" | **Exacta:** `{res['exacta']['integral_exacta']:.6f}{unidad_txt}` (Error rel: `{res['exacta']['error_relativo_pct']:.4f}%`)"
        if res.get("exacta")
        else ""
    )
    registrar_contexto_ia(
        "Resultado Regla de Trapecios",
        f"- **Contexto:** {'Agrimensura de Lote Ribereño' if es_caso_real else 'Función Libre / TP6'}\n"
        f"- **Resultado total:** `{res['area_total']:.6f}{unidad_txt}`{exacta_ia}\n"
        f"- **Trapecios activos:** `{res['tramos_trapecio']}` | **Contraste Simpson 1/3 Híbrido:** `{res['area_simpson_ref']:.6f}{unidad_txt}`",
        clave="int_trap_res",
    )

    if modo_detallado:
        _render_detalle_cuadratura_hibrida(
            res,
            "int_trap",
            f"Regla de los Trapecios ({'Uniforme' if res['es_uniforme'] else 'Paso Variable'})",
            unidad_txt,
        )


def _render_trapecio_tab() -> None:
    modo_detallado = render_toggle_modo_detallado("trapecio")
    meta = render_bloque_1_teoria_y_ejemplo("trapecio", modo_detallado=modo_detallado)
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
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_trapecio(defaults, modo_detallado=modo_detallado)


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
        _render_trapecio_tab()
    elif activo == 2:
        _render_simpson_1_3_tab()
    else:
        _render_simpson_3_8_tab()


vista_integracion()
