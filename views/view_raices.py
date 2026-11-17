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
    render_toggle_modo_detallado,
    rerun_fragmento_seguro,
)


def _muestrear_curva(comp: dict, fn_key: str, np_key: str, x_min: float, x_max: float, puntos: int = 220) -> tuple[list[float], list[float | None]]:
    xs = np.linspace(x_min, x_max, puntos)
    fn_np = comp.get(np_key)
    if fn_np is not None:
        ys = evaluar_fn_vectorizado(fn_np, xs, limite_abs=1e6)
    else:
        ys = evaluar_fn_vectorizado(np.vectorize(comp[fn_key]), xs, limite_abs=1e6)
    return xs.tolist(), ys


_F_TANQUE_BISEC = "x**3 - 9*x**2 + 30"
_F_VAN_TIR = "-100 + 35/(1+x) + 45/(1+x)**2 + 50/(1+x)**3"
_F_DIODO_NR = "0.1*(exp(2*x) - 1) + x - 5"
_G_ALOHA_PF = "exp(-0.85*x)"


def _aplicar_preset_raices(valores: dict[str, object], claves_ui: tuple[str, ...]) -> None:
    """Actualiza parámetros de simulación en session_state y limpia las claves de widgets UI."""
    for clave, valor in valores.items():
        st.session_state[clave] = valor
    for clave_ui in claves_ui:
        st.session_state.pop(clave_ui, None)


def _construir_filas_tanteo(historial: list[dict]) -> list[dict]:
    """Formatea la tabla del barrido por tanteos con encabezados limpios en español."""
    return [
        {
            "Paso": r["paso"],
            "Nodo xₖ": round(float(r["x_k"]), 4),
            "f(xₖ)": round(float(r["f_xk"]), 6) if np.isfinite(r["f_xk"]) else None,
            "Signo": r["signo"],
            "¿Cambio de Signo?": "Sí" if r["cambio_signo"] else "No",
            "Observación": r["observacion"],
        }
        for r in historial
    ]


def _construir_filas_biseccion(historial: list[dict]) -> list[dict]:
    """Formatea la tabla de iteraciones de Bisección con encabezados limpios en español."""
    return [
        {
            "Iteración k": r["iteracion"],
            "Extremo a": round(float(r["a"]), 6),
            "Extremo b": round(float(r["b"]), 6),
            "Punto Medio xₖ": round(float(r["x_k"]), 6),
            "f(a)": round(float(r["f_a"]), 6),
            "f(xₖ)": round(float(r["f_xk"]), 6),
            "f(b)": round(float(r["f_b"]), 6),
            "Error |Δx|": round(float(r["error"]), 6) if r["error"] is not None else None,
            "¿Cumple Tolerancia?": "Sí" if r["cumple_tol"] else "No",
        }
        for r in historial
    ]


@st.fragment
def _frag_biseccion(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_tanque", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Nivel en Tanque Esférico h³ − 9h² + 30 = 0 → Altura h (m)"
            if op == "real_tanque"
            else "Ecuación Libre f(x) = 0 y Cátedra TP3"
        ),
        horizontal=True,
        key="ui_bis_contexto",
    )
    es_caso_real = contexto == "real_tanque"

    if es_caso_real:
        st.caption(
            r"Depósito esférico ($R = 3\;\text{m}$, llenado $V = 10\pi\;\text{m}^3$): buscá la altura física $h \in [0, 6]\;\text{m}$ "
            r"resolviendo $h^3 - 9h^2 + 30 = 0$ mediante encierro binario garantizado."
        )
        if "raiz_bis_real_a" not in st.session_state:
            st.session_state["raiz_bis_real_a"] = 2.0
        if "raiz_bis_real_b" not in st.session_state:
            st.session_state["raiz_bis_real_b"] = 2.5
        if "raiz_bis_real_tol" not in st.session_state:
            st.session_state["raiz_bis_real_tol"] = 0.001

        claves_ui_real = ("ui_bis_real_a", "ui_bis_real_b", "ui_bis_real_tol")
        b1, b2, b3 = st.columns(3)
        if b1.button(
            "Estándar (h ∈ [2.0, 2.5] m, E=0.001 m → 9 pasos)",
            key="btn_bis_real_std",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {"raiz_bis_real_a": 2.0, "raiz_bis_real_b": 2.5, "raiz_bis_real_tol": 0.001},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button(
            "Tolerancia gruesa (h ∈ [2.0, 2.5] m, E=0.01 m → 6 pasos)",
            key="btn_bis_real_gruesa",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {"raiz_bis_real_a": 2.0, "raiz_bis_real_b": 2.5, "raiz_bis_real_tol": 0.01},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button(
            "Tanque completo (h ∈ [0.0, 6.0] m, E=0.001 m → 13 pasos)",
            key="btn_bis_real_amplio",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {"raiz_bis_real_a": 0.0, "raiz_bis_real_b": 6.0, "raiz_bis_real_tol": 0.001},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns(3)
        a_val = float(
            c1.number_input(
                "Cota inferior hₐ (m):",
                min_value=0.0,
                max_value=6.0,
                value=float(st.session_state["raiz_bis_real_a"]),
                step=0.25,
                format="%.3f",
                key="ui_bis_real_a",
            )
        )
        b_val = float(
            c2.number_input(
                "Cota superior h_b (m):",
                min_value=0.0,
                max_value=6.0,
                value=float(st.session_state["raiz_bis_real_b"]),
                step=0.25,
                format="%.3f",
                key="ui_bis_real_b",
            )
        )
        tol_val = float(
            c3.number_input(
                "Tolerancia E (m):",
                min_value=1e-7,
                max_value=0.5,
                value=float(st.session_state["raiz_bis_real_tol"]),
                step=0.001,
                format="%.5f",
                key="ui_bis_real_tol",
            )
        )
        st.session_state["raiz_bis_real_a"] = a_val
        st.session_state["raiz_bis_real_b"] = b_val
        st.session_state["raiz_bis_real_tol"] = tol_val
        f_str = _F_TANQUE_BISEC
        max_it = 100
    else:
        if "raiz_bis_f_str" not in st.session_state:
            st.session_state["raiz_bis_f_str"] = str(defaults.get("f_str", "-2.75*x**3 + 18*x**2 - 21*x - 12"))
        if "raiz_bis_a" not in st.session_state:
            st.session_state["raiz_bis_a"] = float(defaults.get("a", 2.0))
        if "raiz_bis_b" not in st.session_state:
            st.session_state["raiz_bis_b"] = float(defaults.get("b", 2.5))
        if "raiz_bis_tol" not in st.session_state:
            st.session_state["raiz_bis_tol"] = float(defaults.get("tol", 0.01))
        if "raiz_bis_max_iter" not in st.session_state:
            st.session_state["raiz_bis_max_iter"] = int(defaults.get("max_iter", 50))

        claves_ui_lib = ("ui_bis_f", "ui_bis_a", "ui_bis_b", "ui_bis_tol", "ui_bis_max")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP3 Ej. 1 (polinomio cúbico)", key="btn_bis_rest_tp3", use_container_width=True):
            _aplicar_preset_raices(
                {
                    "raiz_bis_f_str": str(defaults.get("f_str", "-2.75*x**3 + 18*x**2 - 21*x - 12")),
                    "raiz_bis_a": float(defaults.get("a", 2.0)),
                    "raiz_bis_b": float(defaults.get("b", 2.5)),
                    "raiz_bis_tol": float(defaults.get("tol", 0.01)),
                    "raiz_bis_max_iter": int(defaults.get("max_iter", 50)),
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Cargar trascendente: exp(-x) - x = 0 en [0, 1]", key="btn_bis_trans", use_container_width=True):
            _aplicar_preset_raices(
                {
                    "raiz_bis_f_str": "exp(-x) - x",
                    "raiz_bis_a": 0.0,
                    "raiz_bis_b": 1.0,
                    "raiz_bis_tol": 0.001,
                    "raiz_bis_max_iter": 50,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        f_str = st.text_input(
            "Ecuación f(x) = 0:",
            value=str(st.session_state["raiz_bis_f_str"]),
            help="Expresión en variable x usando * y ** (ej: -2.75*x**3 + 18*x**2 - 21*x - 12).",
            key="ui_bis_f",
        )
        c1, c2, c3, c4 = st.columns(4)
        a_val = float(c1.number_input("Extremo a:", value=float(st.session_state["raiz_bis_a"]), format="%.4f", key="ui_bis_a"))
        b_val = float(c2.number_input("Extremo b:", value=float(st.session_state["raiz_bis_b"]), format="%.4f", key="ui_bis_b"))
        tol_val = float(
            c3.number_input("Tolerancia E:", value=float(st.session_state["raiz_bis_tol"]), min_value=1e-10, format="%.6f", key="ui_bis_tol")
        )
        max_it = int(
            c4.number_input("Iteraciones máx:", value=int(st.session_state["raiz_bis_max_iter"]), min_value=1, max_value=200, key="ui_bis_max")
        )
        st.session_state["raiz_bis_f_str"] = f_str
        st.session_state["raiz_bis_a"] = a_val
        st.session_state["raiz_bis_b"] = b_val
        st.session_state["raiz_bis_tol"] = tol_val
        st.session_state["raiz_bis_max_iter"] = max_it

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

    try:
        res = metodo_biseccion(comp["fn"], a_val, b_val, tol=tol_val, max_iter=int(max_it))
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Intervalo [a, b] o tolerancia de Bisección",
            sugerencia_defecto="Elegí extremos distintos a y b donde f(a)·f(b) < 0 y tolerancia E > 0.",
            ejemplo_defecto="a = 2.0, b = 2.5, E = 0.001",
        )
        return

    unidad_txt = " m" if es_caso_real else ""
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Raíz Aproximada",
        f"{res['raiz']:.6f}{unidad_txt}",
        f"f(raíz) = {res['f_raiz']:.3e}",
        delta_color="off",
    )
    m2.metric(
        "Iteraciones (Real vs Cota)",
        f"{res['iteraciones']} pasos",
        f"Teórico ⌈n⌉ = {res['n_teorico_entero']} ({res['evals_f']} evals)",
        delta_color="off",
    )
    m3.metric(
        "Error Final |Δx|",
        f"{res['error_final']:.3e}{unidad_txt}",
        f"{'Convergió' if res['convergente'] else 'Tope máx. iter.'} (E = {tol_val:g})",
        delta_color="off",
    )

    hist = res["historial"]
    k_vis = (
        render_slider_paso_k("Ver encierro hasta la iteración $k$:", len(hist), key="vis_bis_k")
        if modo_detallado
        else len(hist)
    )
    fila_k = hist[k_vis - 1]
    pad = max(0.5, 0.35 * abs(b_val - a_val))
    xs, ys = _muestrear_curva(comp, "fn", "fn_np", min(a_val, b_val) - pad, max(a_val, b_val) + pad)

    fig = crear_figura_base_plotly(
        (
            f"Calibración de Nivel h(V) — Bisección ({res['iteraciones']} pasos)"
            if es_caso_real and not modo_detallado
            else f"Bisección — Iteración {k_vis}: [{fila_k['a']:.4f}, {fila_k['b']:.4f}]"
        ),
        "Altura h (m)" if es_caso_real else "x",
        "f(h)" if es_caso_real else "f(x)",
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(160, 160, 160, 0.65)")
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
            name="Puntos medios xₖ",
            marker=dict(size=8, color=PALETA_ARANDU["salvia"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[fila_k["x_k"]],
            y=[fila_k["f_xk"]],
            mode="markers",
            name=f"Actual x_{k_vis}",
            marker=dict(
                size=13,
                color=PALETA_ARANDU["marfil_nodo"],
                symbol="star",
                line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8),
            ),
        )
    )
    render_plotly_arandu(fig, key="plot_raiz_bis")

    registrar_contexto_ia(
        "Resultado Bisección",
        f"- **Escenario:** {'Tanque Esférico IoT' if es_caso_real else 'Libre / TP3'} | **f(x):** `{f_str}`\n"
        f"- **Raíz:** `{res['raiz']:.6f}{unidad_txt}` | **Iteraciones:** `{res['iteraciones']}` (teórico `{res['n_teorico_entero']}`)\n"
        f"- **Error final:** `{res['error_final']:.3e}` | **f(raíz):** `{res['f_raiz']:.3e}`",
        clave="raiz_bis_res",
    )

    if modo_detallado:
        with st.expander("Análisis Previo: Regla de Descartes y Barrido por Tanteos (Δx = 0.5)", expanded=False):
            desc = comp.get("descartes") or analizar_descartes(f_str)
            if desc["es_polinomio"]:
                st.markdown(
                    f"Grado del polinomio: **{desc['grado']}** | Raíces reales positivas posibles: **{desc['posibles_raices_pos']}** | "
                    f"Raíces reales negativas posibles: **{desc['posibles_raices_neg']}** "
                    "(cota combinatoria contando variaciones de signo)."
                )
            else:
                st.info(desc["mensaje"])
            bt = barrido_tanteo(comp["fn"], x_ini=min(a_val, b_val) - 1.5, dx=0.5, max_pasos=14)
            st.markdown(rf"Intervalos con cambio de signo detectados ($\Delta x = 0.5$): **{bt['intervalos']}**")
            render_dataframe_acotado(_construir_filas_tanteo(bt["historial"]), max_filas=50, key_prefix="tbl_bis_tanteo")

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
            expanded=True,
        )
        render_dataframe_acotado(_construir_filas_biseccion(hist), max_filas=300, key_prefix="tbl_raiz_bis")


def _tab_biseccion() -> None:
    modo_detallado = render_toggle_modo_detallado("biseccion")
    meta = render_bloque_1_teoria_y_ejemplo("biseccion", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "-2.75*x**3 + 18*x**2 - 21*x - 12", "a": 2.0, "b": 2.5, "tol": 0.01, "max_iter": 50},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_biseccion(defaults, modo_detallado=modo_detallado)


def _construir_filas_regula_falsi(historial: list[dict], es_catedra: bool) -> list[dict]:
    """Formatea la tabla de Interpolación Lineal (Cátedra o Bolzano) en español."""
    if es_catedra:
        return [
            {
                "Iteración k": r["iteracion"],
                "Subíndice": r["subindice"],
                "Pivote Fijo x₀": round(float(r["x0_fijo"]), 6),
                "Extremo Móvil xᵢ": round(float(r["x_i"]), 6),
                "f(x₀)": round(float(r["f_x0"]), 6),
                "f(xᵢ)": round(float(r["f_xi"]), 6),
                "Corte Secante": round(float(r["x_sig"]), 6),
                "f(Corte)": round(float(r["f_xsig"]), 6),
                "Error |Δx|": round(float(r["error"]), 6) if r["error"] is not None else None,
                "¿Cumple Tolerancia?": "Sí" if r["cumple_tol"] else "No",
            }
            for r in historial
        ]
    return [
        {
            "Iteración k": r["iteracion"],
            "Extremo a": round(float(r["a"]), 6),
            "Extremo b": round(float(r["b"]), 6),
            "f(a)": round(float(r["f_a"]), 6),
            "f(b)": round(float(r["f_b"]), 6),
            "Corte Secante xₖ": round(float(r["x_k"]), 6),
            "f(xₖ)": round(float(r["f_xk"]), 6),
            "Error |Δx|": round(float(r["error"]), 6) if r["error"] is not None else None,
            "¿Cumple Tolerancia?": "Sí" if r["cumple_tol"] else "No",
        }
        for r in historial
    ]


@st.fragment
def _frag_interpolacion_lineal(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_tir", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Tasa Interna de Retorno VAN(r) = 0 → TIR"
            if op == "real_tir"
            else "Ecuación Libre f(x) = 0 y Cátedra TP3 Ej. 2"
        ),
        horizontal=True,
        key="ui_rf_contexto",
    )
    es_caso_real = contexto == "real_tir"

    if es_caso_real:
        st.caption(
            r"Flujos de proyecto Cloud `[-100, +35, +45, +50]`: hallá la tasa $r$ con $\text{VAN}(r) = 0$ usando cuerdas secantes."
        )
        if "raiz_rf_real_a" not in st.session_state:
            st.session_state["raiz_rf_real_a"] = 0.05
        if "raiz_rf_real_b" not in st.session_state:
            st.session_state["raiz_rf_real_b"] = 0.30
        if "raiz_rf_real_tol" not in st.session_state:
            st.session_state["raiz_rf_real_tol"] = 0.0001
        if "raiz_rf_real_var" not in st.session_state:
            st.session_state["raiz_rf_real_var"] = "Cátedra (pivote fijo)"

        claves_ui_real = ("ui_rf_real_a", "ui_rf_real_b", "ui_rf_real_tol", "ui_rf_real_var")
        b1, b2, b3 = st.columns(3)
        if b1.button(
            "Cátedra pivote 5% (5 pasos vs 12 Bisección)",
            key="btn_rf_tir_cat",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {
                    "raiz_rf_real_a": 0.05,
                    "raiz_rf_real_b": 0.30,
                    "raiz_rf_real_tol": 0.0001,
                    "raiz_rf_real_var": "Cátedra (pivote fijo)",
                },
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button(
            "Regula Falsi Bolzano (5 pasos en [5%, 30%])",
            key="btn_rf_tir_bol",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {
                    "raiz_rf_real_a": 0.05,
                    "raiz_rf_real_b": 0.30,
                    "raiz_rf_real_tol": 0.0001,
                    "raiz_rf_real_var": "Regula Falsi (Bolzano)",
                },
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button(
            "Pivote opuesto 30% (7 pasos por concavidad)",
            key="btn_rf_tir_op",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {
                    "raiz_rf_real_a": 0.30,
                    "raiz_rf_real_b": 0.05,
                    "raiz_rf_real_tol": 0.0001,
                    "raiz_rf_real_var": "Cátedra (pivote fijo)",
                },
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3, c4 = st.columns(4)
        a_val = float(
            c1.number_input(
                "Tasa r₀ / extremo a:",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state["raiz_rf_real_a"]),
                step=0.01,
                format="%.4f",
                key="ui_rf_real_a",
            )
        )
        b_val = float(
            c2.number_input(
                "Tasa r₁ / extremo b:",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state["raiz_rf_real_b"]),
                step=0.01,
                format="%.4f",
                key="ui_rf_real_b",
            )
        )
        tol_val = float(
            c3.number_input(
                "Tolerancia E:",
                min_value=1e-7,
                max_value=0.1,
                value=float(st.session_state["raiz_rf_real_tol"]),
                step=0.0001,
                format="%.5f",
                key="ui_rf_real_tol",
            )
        )
        variante = str(
            c4.selectbox(
                "Variante de actualización:",
                ["Cátedra (pivote fijo)", "Regula Falsi (Bolzano)"],
                index=0 if str(st.session_state["raiz_rf_real_var"]).startswith("Cátedra") else 1,
                key="ui_rf_real_var",
            )
        )
        st.session_state["raiz_rf_real_a"] = a_val
        st.session_state["raiz_rf_real_b"] = b_val
        st.session_state["raiz_rf_real_tol"] = tol_val
        st.session_state["raiz_rf_real_var"] = variante
        f_str = _F_VAN_TIR
    else:
        if "raiz_rf_f_str" not in st.session_state:
            st.session_state["raiz_rf_f_str"] = str(defaults.get("f_str", "2*x**4 - 6*x**2 + 7*x - 2"))
        if "raiz_rf_a" not in st.session_state:
            st.session_state["raiz_rf_a"] = float(defaults.get("a", -2.5))
        if "raiz_rf_b" not in st.session_state:
            st.session_state["raiz_rf_b"] = float(defaults.get("b", -2.0))
        if "raiz_rf_tol" not in st.session_state:
            st.session_state["raiz_rf_tol"] = float(defaults.get("tol", 0.001))
        if "raiz_rf_variante" not in st.session_state:
            st.session_state["raiz_rf_variante"] = str(defaults.get("variante", "Cátedra (pivote fijo)"))

        claves_ui_lib = ("ui_rf_f", "ui_rf_a", "ui_rf_b", "ui_rf_tol", "ui_rf_var")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP3 Ej. 2 (2x⁴ − 6x² + 7x − 2)", key="btn_rf_rest_tp3", use_container_width=True):
            _aplicar_preset_raices(
                {
                    "raiz_rf_f_str": str(defaults.get("f_str", "2*x**4 - 6*x**2 + 7*x - 2")),
                    "raiz_rf_a": float(defaults.get("a", -2.5)),
                    "raiz_rf_b": float(defaults.get("b", -2.0)),
                    "raiz_rf_tol": float(defaults.get("tol", 0.001)),
                    "raiz_rf_variante": "Cátedra (pivote fijo)",
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Estancamiento extremo: x⁸ − 0.05 en [0.5, 1.0] (21 vs 9 Bisección)", key="btn_rf_estanco", use_container_width=True):
            _aplicar_preset_raices(
                {
                    "raiz_rf_f_str": "x**8 - 0.05",
                    "raiz_rf_a": 0.5,
                    "raiz_rf_b": 1.0,
                    "raiz_rf_tol": 0.001,
                    "raiz_rf_variante": "Regula Falsi (Bolzano)",
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        f_str = st.text_input("Ecuación f(x) = 0:", value=str(st.session_state["raiz_rf_f_str"]), key="ui_rf_f")
        c1, c2, c3, c4 = st.columns(4)
        a_val = float(c1.number_input("x₀ / extremo a:", value=float(st.session_state["raiz_rf_a"]), format="%.4f", key="ui_rf_a"))
        b_val = float(c2.number_input("x₁ / extremo b:", value=float(st.session_state["raiz_rf_b"]), format="%.4f", key="ui_rf_b"))
        tol_val = float(
            c3.number_input("Tolerancia E:", value=float(st.session_state["raiz_rf_tol"]), min_value=1e-10, format="%.6f", key="ui_rf_tol")
        )
        variante = str(
            c4.selectbox(
                "Variante de reemplazo:",
                ["Cátedra (pivote fijo)", "Regula Falsi (Bolzano)"],
                index=0 if str(st.session_state["raiz_rf_variante"]).startswith("Cátedra") else 1,
                key="ui_rf_var",
            )
        )
        st.session_state["raiz_rf_f_str"] = f_str
        st.session_state["raiz_rf_a"] = a_val
        st.session_state["raiz_rf_b"] = b_val
        st.session_state["raiz_rf_tol"] = tol_val
        st.session_state["raiz_rf_variante"] = variante

    es_catedra = variante.startswith("Cátedra")
    try:
        comp = compilar_funcion_raices(f_str)
        if es_catedra:
            res = metodo_interpolacion_lineal_catedra(comp["fn"], x0_fijo=a_val, x1_ini=b_val, tol=tol_val, max_iter=100)
        else:
            res = metodo_interpolacion_lineal_bolzano(comp["fn"], a=min(a_val, b_val), b=max(a_val, b_val), tol=tol_val, max_iter=100)
        res_bis = metodo_biseccion(comp["fn"], a=min(a_val, b_val), b=max(a_val, b_val), tol=tol_val, max_iter=100)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Intervalo [a, b] o ecuación f(x) (Regula Falsi)",
            sugerencia_defecto="Revisá que f(a)·f(b) < 0, extremos distintos, tolerancia E > 0 y función continua.",
            ejemplo_defecto="a = 0.05, b = 0.30, E = 0.0001",
        )
        return

    m1, m2, m3 = st.columns(3)
    delta_r = (
        f"TIR = {res['raiz'] * 100.0:.2f}% | VAN = {res['f_raiz']:.2e}"
        if es_caso_real
        else f"f(raíz) = {res['f_raiz']:.3e}"
    )
    m1.metric("Raíz Aproximada", f"{res['raiz']:.6f}", delta_r, delta_color="off")
    m2.metric(
        "Iteraciones (vs Bisección)",
        f"{res['iteraciones']} pasos",
        f"Bisección: {res_bis['iteraciones']} pasos ({res['evals_f']} evals)",
        delta_color="off",
    )
    m3.metric(
        "Error Final |Δx|",
        f"{res['error_final']:.3e}",
        f"{'Convergió' if res['convergente'] else 'No convergió'} (E = {tol_val:g})",
        delta_color="off",
    )

    hist = res["historial"]
    k_vis = (
        render_slider_paso_k("Ver hasta la iteración $k$:", len(hist), key="vis_rf_k")
        if modo_detallado and len(hist) > 0
        else len(hist)
    )
    pad = max(0.05 if es_caso_real else 0.4, 0.35 * abs(b_val - a_val))
    xs, ys = _muestrear_curva(comp, "fn", "fn_np", max(-0.8, min(a_val, b_val) - pad) if es_caso_real else min(a_val, b_val) - pad, max(a_val, b_val) + pad)
    fig = crear_figura_base_plotly(
        (
            f"Cálculo de TIR — Cuerdas Secantes ({res['iteraciones']} pasos)"
            if es_caso_real and not modo_detallado
            else f"Interpolación Lineal — Paso {k_vis}"
        ),
        "Tasa de descuento r" if es_caso_real else "x",
        "VAN(r)" if es_caso_real else "f(x)",
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(160, 160, 160, 0.65)")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.5)))

    if hist and k_vis >= 1:
        fila_k = hist[k_vis - 1]
        if es_catedra:
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
                marker=dict(
                    size=13,
                    color=PALETA_ARANDU["marfil_nodo"],
                    symbol="diamond",
                    line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8),
                ),
            )
        )
    render_plotly_arandu(fig, key="plot_raiz_rf")

    registrar_contexto_ia(
        "Resultado Interpolación Lineal / Regula Falsi",
        f"- **Escenario:** {'TIR VAN(r)=0' if es_caso_real else 'Libre / TP3 Ej. 2'} | **Variante:** {variante}\n"
        f"- **Raíz:** `{res['raiz']:.6f}` | **Iteraciones:** `{res['iteraciones']}` (Bisección `{res_bis['iteraciones']}`) | **Error:** `{res['error_final']:.3e}`",
        clave="raiz_rf_res",
    )

    if modo_detallado:
        pasos_rf_vivos = [
            f"**Planteo de la cuerda ({variante}):** Partimos desde $x_0 = {a_val:g}$ y $x_1 = {b_val:g}$ buscando intersecciones secantes hasta $|x_{{k+1}} - x_k| < {tol_val:g}$."
        ]
        for r_rf in hist[:6]:
            if es_catedra:
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
            expanded=True,
        )
        render_dataframe_acotado(
            _construir_filas_regula_falsi(hist, es_catedra=es_catedra),
            max_filas=300,
            key_prefix="tbl_raiz_rf",
        )


def _tab_interpolacion_lineal() -> None:
    modo_detallado = render_toggle_modo_detallado("interpolacion_lineal_raices")
    meta = render_bloque_1_teoria_y_ejemplo("interpolacion_lineal_raices", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = {
        "f_str": "2*x**4 - 6*x**2 + 7*x - 2",
        "a": -2.5,
        "b": -2.0,
        "tol": 0.001,
        "variante": "Cátedra (pivote fijo)",
        **practica.get("datos_iniciales", {}),
    }
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_interpolacion_lineal(defaults, modo_detallado=modo_detallado)


def _construir_filas_newton(historial: list[dict]) -> list[dict]:
    """Formatea la tabla de Newton-Raphson con encabezados limpios en español."""
    return [
        {
            "Iteración k": r["iteracion"],
            "Valor Actual xₖ": round(float(r["x_k"]), 7),
            "f(xₖ)": round(float(r["f_xk"]), 6),
            "f'(xₖ)": round(float(r["df_xk"]), 6),
            "Siguiente xₖ₊₁": round(float(r["x_sig"]), 7) if r["x_sig"] is not None else None,
            "Error |Δx|": round(float(r["error"]), 7) if r["error"] is not None else None,
            "¿Cumple Tolerancia?": "Sí" if r["cumple_tol"] else "No",
        }
        for r in historial
    ]


@st.fragment
def _frag_newton_raphson(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_diodo", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Circuito Diodo LED / IoT 0.1(e²ˣ − 1) + x − 5 = 0 → Tensión V_D (V)"
            if op == "real_diodo"
            else "Ecuación Libre f(x) = 0 y Cátedra TP3 Ej. 3"
        ),
        horizontal=True,
        key="ui_nr_contexto",
    )
    es_caso_real = contexto == "real_diodo"

    if es_caso_real:
        st.caption(
            r"Fuente de $5\;\text{V}$ con resistencia serie y diodo de Shockley ($V_D = x$): compará la convergencia cuadrática desde "
            r"el arranque seguro de Fourier ($x_0 = 2.5\;\text{V}$) frente a una semilla sin garantía ($x_0 = 0\;\text{V}$)."
        )
        if "raiz_nr_real_a" not in st.session_state:
            st.session_state["raiz_nr_real_a"] = 0.0
        if "raiz_nr_real_b" not in st.session_state:
            st.session_state["raiz_nr_real_b"] = 2.5
        if "raiz_nr_real_x0" not in st.session_state:
            st.session_state["raiz_nr_real_x0"] = 2.5
        if "raiz_nr_real_tol" not in st.session_state:
            st.session_state["raiz_nr_real_tol"] = 1e-5

        claves_ui_real = ("ui_nr_real_a", "ui_nr_real_b", "ui_nr_real_x0", "ui_nr_real_tol")
        b1, b2, b3 = st.columns(3)
        if b1.button(
            "Arranque Fourier (x₀ = 2.5 V → 6 pasos)",
            key="btn_nr_real_fourier",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {"raiz_nr_real_a": 0.0, "raiz_nr_real_b": 2.5, "raiz_nr_real_x0": 2.5, "raiz_nr_real_tol": 1e-5},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button(
            "Sin garantía Fourier (x₀ = 0.0 V → disparo 4.17 V, 10 pasos)",
            key="btn_nr_real_mal_x0",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {"raiz_nr_real_a": 0.0, "raiz_nr_real_b": 2.5, "raiz_nr_real_x0": 0.0, "raiz_nr_real_tol": 1e-5},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button(
            "Semilla cercana (x₀ = 1.8 V → 3 pasos)",
            key="btn_nr_real_cerca",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {"raiz_nr_real_a": 0.0, "raiz_nr_real_b": 2.5, "raiz_nr_real_x0": 1.8, "raiz_nr_real_tol": 1e-5},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3, c4 = st.columns(4)
        a_val = float(
            c1.number_input(
                "Extremo a (Fourier):",
                min_value=-0.5,
                max_value=3.5,
                value=float(st.session_state["raiz_nr_real_a"]),
                step=0.25,
                format="%.2f",
                key="ui_nr_real_a",
            )
        )
        b_val = float(
            c2.number_input(
                "Extremo b (Fourier):",
                min_value=-0.5,
                max_value=3.5,
                value=float(st.session_state["raiz_nr_real_b"]),
                step=0.25,
                format="%.2f",
                key="ui_nr_real_b",
            )
        )
        x0_val = float(
            c3.number_input(
                "Semilla inicial x₀ (V):",
                min_value=-0.5,
                max_value=3.5,
                value=float(st.session_state["raiz_nr_real_x0"]),
                step=0.1,
                format="%.4f",
                key="ui_nr_real_x0",
            )
        )
        tol_val = float(
            c4.number_input(
                "Tolerancia E (V):",
                min_value=1e-10,
                max_value=0.1,
                value=float(st.session_state["raiz_nr_real_tol"]),
                format="%.6f",
                key="ui_nr_real_tol",
            )
        )
        st.session_state["raiz_nr_real_a"] = a_val
        st.session_state["raiz_nr_real_b"] = b_val
        st.session_state["raiz_nr_real_x0"] = x0_val
        st.session_state["raiz_nr_real_tol"] = tol_val
        f_str = _F_DIODO_NR
        max_it = 50
    else:
        if "raiz_nr_f_str" not in st.session_state:
            st.session_state["raiz_nr_f_str"] = str(defaults.get("f_str", "x**3 - 2*x - 5"))
        if "raiz_nr_a" not in st.session_state:
            st.session_state["raiz_nr_a"] = float(defaults.get("a", 2.0))
        if "raiz_nr_b" not in st.session_state:
            st.session_state["raiz_nr_b"] = float(defaults.get("b", 2.5))
        if "raiz_nr_x0" not in st.session_state:
            st.session_state["raiz_nr_x0"] = float(defaults.get("x0", 2.5))
        if "raiz_nr_tol" not in st.session_state:
            st.session_state["raiz_nr_tol"] = float(defaults.get("tol", 1e-4))
        if "raiz_nr_max_iter" not in st.session_state:
            st.session_state["raiz_nr_max_iter"] = int(defaults.get("max_iter", 50))

        claves_ui_lib = ("ui_nr_f", "ui_nr_a", "ui_nr_b", "ui_nr_x0", "ui_nr_tol")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP3 Ej. 3 (x³ − 2x − 5 en [2, 2.5])", key="btn_nr_rest_tp3", use_container_width=True):
            _aplicar_preset_raices(
                {
                    "raiz_nr_f_str": str(defaults.get("f_str", "x**3 - 2*x - 5")),
                    "raiz_nr_a": float(defaults.get("a", 2.0)),
                    "raiz_nr_b": float(defaults.get("b", 2.5)),
                    "raiz_nr_x0": float(defaults.get("x0", 2.5)),
                    "raiz_nr_tol": float(defaults.get("tol", 1e-4)),
                    "raiz_nr_max_iter": int(defaults.get("max_iter", 50)),
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Trampa de ciclo infinito: x³ − 2x + 2 con x₀ = 0", key="btn_nr_ciclo", use_container_width=True):
            _aplicar_preset_raices(
                {
                    "raiz_nr_f_str": "x**3 - 2*x + 2",
                    "raiz_nr_a": -2.0,
                    "raiz_nr_b": 0.5,
                    "raiz_nr_x0": 0.0,
                    "raiz_nr_tol": 1e-4,
                    "raiz_nr_max_iter": 30,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        f_str = st.text_input("Ecuación f(x) = 0:", value=str(st.session_state["raiz_nr_f_str"]), key="ui_nr_f")
        c1, c2, c3, c4 = st.columns(4)
        a_val = float(c1.number_input("Extremo a (Fourier):", value=float(st.session_state["raiz_nr_a"]), format="%.4f", key="ui_nr_a"))
        b_val = float(c2.number_input("Extremo b (Fourier):", value=float(st.session_state["raiz_nr_b"]), format="%.4f", key="ui_nr_b"))
        x0_val = float(c3.number_input("Semilla x₀:", value=float(st.session_state["raiz_nr_x0"]), format="%.4f", key="ui_nr_x0"))
        tol_val = float(c4.number_input("Tolerancia E:", value=float(st.session_state["raiz_nr_tol"]), min_value=1e-12, format="%.6f", key="ui_nr_tol"))

        st.session_state["raiz_nr_f_str"] = f_str
        st.session_state["raiz_nr_a"] = a_val
        st.session_state["raiz_nr_b"] = b_val
        st.session_state["raiz_nr_x0"] = x0_val
        st.session_state["raiz_nr_tol"] = tol_val
        max_it = int(st.session_state["raiz_nr_max_iter"])

    try:
        comp = compilar_funcion_raices(f_str)
        fourier = verificar_condiciones_fourier(comp["fn"], comp["d1"], comp["d2"], min(a_val, b_val), max(a_val, b_val))
        res = metodo_newton_raphson(comp["fn"], comp["d1"], x0_val, tol=tol_val, max_iter=max_it)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Ecuación f(x), intervalo Fourier [a, b] o semilla x₀ (Newton-Raphson)",
            sugerencia_defecto="Ingresá una función derivable f(x) con sintaxis Python, extremos distintos a y b, x₀ finito y E > 0.",
            ejemplo_defecto="f(x) = x**3 - 2*x - 5, a = 2.0, b = 2.5, x₀ = 2.5",
        )
        return

    if not res["convergente"] and res.get("mensaje"):
        st.warning(res["mensaje"])

    unidad_txt = " V" if es_caso_real else ""
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Raíz Aproximada",
        f"{res['raiz']:.6f}{unidad_txt}",
        f"{'f(V_D)' if es_caso_real else 'f(raíz)'} = {res['f_raiz']:.3e}",
        delta_color="off",
    )
    m2.metric(
        "Iteraciones (Arranque Fourier)",
        f"{res['iteraciones']} pasos",
        f"Fourier sugiere x₀ = {fourier['x0_sugerido']:g} ({res['evals_f']} evals)",
        delta_color="off",
    )
    m3.metric(
        "Error Final |Δx|",
        f"{res['error_final']:.3e}{unidad_txt}",
        f"{'Convergió' if res['convergente'] else 'No convergió'} (E = {tol_val:g})",
        delta_color="off",
    )

    hist = res["historial"]
    k_vis = (
        render_slider_paso_k("Ver tangentes hasta la iteración $k$:", max(1, len(hist)), key="vis_nr_k")
        if modo_detallado and len(hist) > 0
        else len(hist)
    )
    span_base = max(1.0, abs(b_val - a_val))
    x_lim_min = min(a_val, b_val, x0_val) - 2.5 * span_base
    x_lim_max = max(a_val, b_val, x0_val) + 2.5 * span_base
    puntos_x = [min(a_val, b_val), max(a_val, b_val), x0_val] + [
        p
        for row in hist[:k_vis]
        for p in (row["x_k"], row["x_sig"])
        if p is not None and x_lim_min <= p <= x_lim_max
    ]
    pad = max(0.35 if es_caso_real else 0.5, 0.18 * (max(puntos_x) - min(puntos_x)))
    xmin = max(-0.4, min(puntos_x) - pad) if es_caso_real else min(puntos_x) - pad
    xmax = max(puntos_x) + pad
    xs, ys = _muestrear_curva(comp, "fn", "fn_np", xmin, xmax)

    fig = crear_figura_base_plotly(
        (
            f"Punto de Operación V_D — Rectas Tangentes ({res['iteraciones']} pasos)"
            if es_caso_real and not modo_detallado
            else f"Newton-Raphson — Tangentes hasta k = {k_vis}"
        ),
        "Tensión diodo V_D (V)" if es_caso_real else "x",
        "f(V_D)" if es_caso_real else "f(x)",
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(160, 160, 160, 0.65)")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color=PALETA_ARANDU["oro"], width=2.5)))
    for idx, row in enumerate(hist[:k_vis], start=1):
        if row["x_sig"] is not None and x_lim_min <= row["x_k"] <= x_lim_max and x_lim_min <= row["x_sig"] <= x_lim_max:
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
    if es_caso_real:
        ys_fin = [y for y in ys if y is not None and np.isfinite(y)]
        y_a_val, y_b_val = float(comp["fn"](min(a_val, b_val))), float(comp["fn"](max(a_val, b_val)))
        y_ref_max = max(abs(y_a_val), abs(y_b_val), 8.0)
        if ys_fin and max(abs(v) for v in ys_fin) > 4.0 * y_ref_max:
            fig.update_yaxes(range=[-1.4 * y_ref_max, 1.4 * y_ref_max])
    render_plotly_arandu(fig, key="plot_raiz_nr")

    registrar_contexto_ia(
        "Derivadas, Fourier y Resultado Newton-Raphson",
        f"- **Escenario:** {'Diodo IoT' if es_caso_real else 'Libre / TP3 Ej. 3'} | **f(x):** `{f_str}`\n"
        f"- **Derivadas:** $$f'(x) = {comp['d1_latex']}, \\qquad f''(x) = {comp['d2_latex']}$$\n"
        f"- **Validación Fourier:** {fourier['detalle_fourier']}\n"
        f"- **Raíz:** `{res['raiz']:.6f}{unidad_txt}` | **Iteraciones:** `{res['iteraciones']}` | **Error:** `{res['error_final']:.3e}`",
        clave="raiz_nr_res",
    )

    if modo_detallado:
        st.latex(rf"f'(x) = {comp['d1_latex']}, \qquad f''(x) = {comp['d2_latex']}")
        st.info(
            rf"**Validación de arranque seguro en $[{min(a_val, b_val):g}, {max(a_val, b_val):g}]$ (Condiciones de Fourier):** "
            f"{'Sugerido $x_0 = ' + str(fourier['x0_sugerido']) + '$' if fourier['fourier_cumple'] else 'Sin garantía estricta en extremos'} "
            f"— {fourier['detalle_fourier']}"
        )
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
            f"Newton-Raphson (x₀={x0_val:g})",
            pasos_nr_vivos,
            clave_ia="raiz_nr_viva",
            expanded=True,
        )
        render_dataframe_acotado(_construir_filas_newton(hist), max_filas=300, key_prefix="tbl_raiz_nr")


def _tab_newton_raphson() -> None:
    modo_detallado = render_toggle_modo_detallado("newton_raphson")
    meta = render_bloque_1_teoria_y_ejemplo("newton_raphson", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "x**3 - 2*x - 5", "a": 2.0, "b": 2.5, "x0": 2.5, "tol": 1e-4, "max_iter": 50},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_newton_raphson(defaults, modo_detallado=modo_detallado)


def _construir_filas_punto_fijo(historial: list[dict]) -> list[dict]:
    """Formatea la tabla de iteraciones de Punto Fijo simple con encabezados limpios en español."""
    return [
        {
            "Iteración k": r["iteracion"],
            "xₖ₋₁": round(float(r["x_ant"]), 6),
            "xₖ = g(xₖ₋₁)": round(float(r["x_sig"]), 6),
            "Error |Δx|": round(float(r["error"]), 6),
            "¿Cumple Tolerancia?": "Sí" if r["cumple_tol"] else "No",
        }
        for r in historial
    ]


def _construir_filas_aitken(historial: list[dict]) -> list[dict]:
    """Formatea la tabla de Aceleración Δ² de Aitken con encabezados limpios en español."""
    return [
        {
            "Ciclo k": r["ciclo"],
            "x₀": round(float(r["x0"]), 6),
            "x₁ = g(x₀)": round(float(r["x1"]), 6),
            "x₂ = g(x₁)": round(float(r["x2"]), 6),
            "Δ² x₀": round(float(r["denominador"]), 6),
            "Salto Acelerado S": round(float(r["s_acelerado"]), 6),
            "g(S)": round(float(r["x_post"]), 6),
            "Error |g(S)−S|": round(float(r["error"]), 6),
            "¿Cumple Tolerancia?": "Sí" if r["cumple_tol"] else "No",
        }
        for r in historial
    ]


@st.fragment
def _frag_punto_fijo_aitken(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_aloha", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Equilibrio de Red Aloha p = exp(−0.85·p) → Éxito p"
            if op == "real_aloha"
            else "Despeje Libre x = g(x) y Cátedra TP3 Ej. 4"
        ),
        horizontal=True,
        key="ui_pf_contexto",
    )
    es_caso_real = contexto == "real_aloha"

    if es_caso_real:
        st.caption(
            r"Probabilidad de transmisión exitosa $p = e^{-0.85\,p}$: compará el despeje contractivo en espiral ($e^{-0.85x}$: "
            r"15 pasos PF vs 2 ciclos Aitken) frente al despeje algebraico divergente ($-\ln(x)/0.85$)."
        )
        if "raiz_pf_real_g" not in st.session_state:
            st.session_state["raiz_pf_real_g"] = _G_ALOHA_PF
        if "raiz_pf_real_x0" not in st.session_state:
            st.session_state["raiz_pf_real_x0"] = 0.2
        if "raiz_pf_real_tol" not in st.session_state:
            st.session_state["raiz_pf_real_tol"] = 1e-4

        claves_ui_real = ("ui_pf_real_g", "ui_pf_real_x0", "ui_pf_real_tol")
        b1, b2 = st.columns(2)
        if b1.button(
            "Despeje contractivo: exp(−0.85x), x₀ = 0.2 (PF 15 vs Aitken 2)",
            key="btn_pf_aloha_conv",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {"raiz_pf_real_g": _G_ALOHA_PF, "raiz_pf_real_x0": 0.2, "raiz_pf_real_tol": 1e-4},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button(
            "Despeje divergente: −ln(x)/0.85, x₀ = 0.55 (|g'| = 2.14 ≥ 1)",
            key="btn_pf_aloha_div",
            use_container_width=True,
        ):
            _aplicar_preset_raices(
                {"raiz_pf_real_g": "-log(x)/0.85", "raiz_pf_real_x0": 0.55, "raiz_pf_real_tol": 1e-4},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns(3)
        opciones_g = [_G_ALOHA_PF, "-log(x)/0.85"]
        g_idx = 0 if str(st.session_state["raiz_pf_real_g"]) == _G_ALOHA_PF else 1
        g_str = str(
            c1.selectbox(
                "Despeje x = g(x):",
                opciones_g,
                index=g_idx,
                key="ui_pf_real_g",
            )
        )
        x0_val = float(
            c2.number_input(
                "Semilla inicial p₀:",
                value=float(st.session_state["raiz_pf_real_x0"]),
                step=0.05,
                format="%.4f",
                key="ui_pf_real_x0",
            )
        )
        tol_val = float(
            c3.number_input(
                "Tolerancia E:",
                min_value=1e-10,
                max_value=0.1,
                value=float(st.session_state["raiz_pf_real_tol"]),
                format="%.6f",
                key="ui_pf_real_tol",
            )
        )
        st.session_state["raiz_pf_real_g"] = g_str
        st.session_state["raiz_pf_real_x0"] = x0_val
        st.session_state["raiz_pf_real_tol"] = tol_val
        f_str = "exp(-0.85*x) - x"
        max_it = 50
    else:
        if "raiz_pf_f_str" not in st.session_state:
            st.session_state["raiz_pf_f_str"] = str(defaults.get("f_str", "exp(x) - x**2 - 5"))
        if "raiz_pf_g_str" not in st.session_state:
            st.session_state["raiz_pf_g_str"] = str(defaults.get("g_str", "log(x**2 + 5)"))
        if "raiz_pf_x0" not in st.session_state:
            st.session_state["raiz_pf_x0"] = float(defaults.get("x0", 2.5))
        if "raiz_pf_tol" not in st.session_state:
            st.session_state["raiz_pf_tol"] = float(defaults.get("tol", 1e-4))
        if "raiz_pf_max_iter" not in st.session_state:
            st.session_state["raiz_pf_max_iter"] = int(defaults.get("max_iter", 50))

        claves_ui_lib = ("ui_pf_f", "ui_pf_g", "ui_pf_x0", "ui_pf_tol")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP3 Ej. 4 (ln(x² + 5), x₀ = 2.5)", key="btn_pf_rest_tp3", use_container_width=True):
            _aplicar_preset_raices(
                {
                    "raiz_pf_f_str": str(defaults.get("f_str", "exp(x) - x**2 - 5")),
                    "raiz_pf_g_str": str(defaults.get("g_str", "log(x**2 + 5)")),
                    "raiz_pf_x0": float(defaults.get("x0", 2.5)),
                    "raiz_pf_tol": float(defaults.get("tol", 1e-4)),
                    "raiz_pf_max_iter": int(defaults.get("max_iter", 50)),
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Telaraña lenta: 2/(1 + 0.7x²), x₀ = 0.5 (PF > 50 vs Aitken 3)", key="btn_pf_lento", use_container_width=True):
            _aplicar_preset_raices(
                {
                    "raiz_pf_f_str": "2/(1 + 0.7*x**2) - x",
                    "raiz_pf_g_str": "2 / (1 + 0.7*x**2)",
                    "raiz_pf_x0": 0.5,
                    "raiz_pf_tol": 1e-4,
                    "raiz_pf_max_iter": 50,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        c1, c2 = st.columns(2)
        f_str = str(c1.text_input("Ecuación f(x) = 0 (referencia):", value=str(st.session_state["raiz_pf_f_str"]), key="ui_pf_f"))
        g_str = str(c2.text_input("Despeje x = g(x):", value=str(st.session_state["raiz_pf_g_str"]), key="ui_pf_g"))
        c3, c4 = st.columns(2)
        x0_val = float(c3.number_input("Semilla x₀:", value=float(st.session_state["raiz_pf_x0"]), format="%.4f", key="ui_pf_x0"))
        tol_val = float(c4.number_input("Tolerancia E:", value=float(st.session_state["raiz_pf_tol"]), min_value=1e-12, format="%.6f", key="ui_pf_tol"))

        st.session_state["raiz_pf_f_str"] = f_str
        st.session_state["raiz_pf_g_str"] = g_str
        st.session_state["raiz_pf_x0"] = x0_val
        st.session_state["raiz_pf_tol"] = tol_val
        max_it = int(st.session_state["raiz_pf_max_iter"])

    try:
        comp_g = compilar_g_punto_fijo(g_str)
        gx0 = float(comp_g["gn"](x0_val))
        deriv_x0 = abs(float(comp_g["dgn"](x0_val)))
        if not (np.isfinite(gx0) and np.isfinite(deriv_x0)):
            raise ValueError(f"g(x₀) o g'(x₀) no está definido en x₀ = {x0_val:g}.")
        res = metodo_punto_fijo_aitken(comp_g["gn"], x0_val, tol=tol_val, max_iter=max_it)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Despeje x = g(x) o semilla x₀ (Punto Fijo)",
            sugerencia_defecto="Escribí g(x) con variable x usando sintaxis Python y verificá que g(x₀) y g'(x₀) existan.",
            ejemplo_defecto="g(x) = log(x**2 + 5), x₀ = 2.5",
        )
        return

    pf, ait = res["punto_fijo"], res["aitken"]
    if deriv_x0 >= 1.0 or not pf["convergente"]:
        st.warning(
            rf"**$|g'(x_0)| = {deriv_x0:.4f}$:** "
            + ("No se cumple contractividad local; la iteración simple puede oscilar o diverger." if deriv_x0 >= 1.0 else "Punto Fijo simple no alcanzó la tolerancia en el límite de iteraciones.")
        )

    err_pf_txt = f"{pf['error_final']:.2e}" if np.isfinite(pf["error_final"]) else "Divergió"
    err_ait_txt = f"{ait['error_final']:.2e}" if np.isfinite(ait["error_final"]) else "Divergió"
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Punto Fijo Simple",
        f"{pf['raiz']:.6f}" if pf["convergente"] else "No convergió",
        f"{pf['iteraciones']} pasos | Err {err_pf_txt}",
        delta_color="off",
    )
    m2.metric(
        "Aceleración Δ² Aitken",
        f"{ait['raiz']:.6f}" if ait["convergente"] else "No convergió",
        f"{ait['ciclos']} ciclos | Err {err_ait_txt}",
        delta_color="off",
    )
    m3.metric(
        "Pendiente |g'(x₀)|",
        f"{deriv_x0:.4f}",
        "Contractivo (|g'| < 1)" if deriv_x0 < 1.0 else "Expansivo (|g'| ≥ 1)",
        delta_color="off",
    )

    if pf["historial"]:
        k_vis = (
            render_slider_paso_k(
                "Ver telaraña hasta iteración $k$:",
                len(pf["historial"]),
                key="vis_pf_k",
                default_val=min(len(pf["historial"]), 8),
            )
            if modo_detallado
            else len(pf["historial"])
        )
        sub = pf["historial"][:k_vis]
        pts = [r["x_ant"] for r in sub] + [r["x_sig"] for r in sub]
        xmin, xmax = min(pts) - 0.3, max(pts) + 0.3
        xs, ys = _muestrear_curva(comp_g, "gn", "gn_np", xmin, xmax)

        fig = crear_figura_base_plotly(
            (
                f"Equilibrio de Red — Telaraña y Saltos Aitken (PF: {pf['iteraciones']} vs Aitken: {ait['ciclos']} ciclos)"
                if es_caso_real and not modo_detallado
                else f"Punto Fijo (y = g(x) vs y = x) hasta k = {k_vis}"
            ),
            "x",
            "y = g(x)",
        )
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
        if ait["historial"]:
            s_vals = [h["s_acelerado"] for h in ait["historial"] if xmin - 0.5 <= h["s_acelerado"] <= xmax + 0.5]
            if s_vals:
                fig.add_trace(
                    go.Scatter(
                        x=s_vals,
                        y=s_vals,
                        mode="markers",
                        name="Saltos Aitken Δ²",
                        marker=dict(size=12, color=PALETA_ARANDU["salvia"], symbol="star", line=dict(color=PALETA_ARANDU["oro"], width=1.5)),
                    )
                )
        render_plotly_arandu(fig, key="plot_raiz_pf")

    registrar_contexto_ia(
        "Derivada g'(x), Punto Fijo y Aitken Δ²",
        f"- **Escenario:** {'Red Aloha' if es_caso_real else 'Libre / TP3 Ej. 4'} | **g(x):** `{g_str}`\n"
        f"- **Derivada:** $$g'(x) = {comp_g['dg_latex']}, \\qquad |g'(x_0)| = {deriv_x0:.4f}$$\n"
        f"- **Punto Fijo simple:** raíz `{pf['raiz']:.6f}` (`{pf['iteraciones']}` iters, error `{err_pf_txt}`)\n"
        f"- **Aitken $\\Delta^2$:** raíz `{ait['raiz']:.6f}` (`{ait['ciclos']}` ciclos, error `{err_ait_txt}`)",
        clave="raiz_pf_res",
    )

    if modo_detallado:
        st.latex(rf"g'(x) = {comp_g['dg_latex']}, \qquad |g'(x_0)| = {deriv_x0:.4f}")
        if deriv_x0 < 1.0:
            st.success(
                r"**$|g'(x_0)| < 1$:** La pendiente reduce el error en cada paso (**esquema contractivo**)."
            )
        pasos_pf_vivos = [
            (
                f"**Chequeo de condición contractiva en $x_0 = {x0_val:g}$:** Derivando $g(x)$ obtenemos $g'(x) = {comp_g['dg_latex']}$, "
                f"con $|g'(x_0)| = {deriv_x0:.4f}$ ({'menor que 1: esquema contractivo convergente' if deriv_x0 < 1.0 else 'mayor o igual a 1: riesgo de divergencia simple'})."
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
            f"**Comparación de eficiencia:** Punto Fijo simple necesitó `{pf['iteraciones']}` iteraciones (`{'convergió' if pf['convergente'] else 'sin converger'}`), "
            f"mientras Aceleración $\\Delta^2$ de Aitken cerró en `{ait['ciclos']}` ciclos (`{ait['raiz']:.6f}`)."
        )
        render_deduccion_paso_a_paso_viva(
            f"Punto Fijo y Aceleración Δ² de Aitken (x₀={x0_val:g})",
            pasos_pf_vivos,
            clave_ia="raiz_pf_viva",
            expanded=True,
        )

        t_pf, t_ait = st.tabs(["Tabla Punto Fijo", "Tabla Aitken Δ²"])
        with t_pf:
            render_dataframe_acotado(_construir_filas_punto_fijo(pf["historial"]), max_filas=300, key_prefix="tbl_raiz_pf")
        with t_ait:
            render_dataframe_acotado(_construir_filas_aitken(ait["historial"]), max_filas=300, key_prefix="tbl_raiz_ait")


def _tab_punto_fijo_aitken() -> None:
    modo_detallado = render_toggle_modo_detallado("punto_fijo_aitken")
    meta = render_bloque_1_teoria_y_ejemplo("punto_fijo_aitken", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "exp(x) - x**2 - 5", "g_str": "log(x**2 + 5)", "x0": 2.5, "tol": 1e-4, "max_iter": 50},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_punto_fijo_aitken(defaults, modo_detallado=modo_detallado)


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
