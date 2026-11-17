"""
Vista de Ecuaciones Diferenciales Ordinarias (PVI) - Arandú.
Euler, Heun (Corrector Iterativo), Runge-Kutta (RK2 / RK4) y Milne (Orden 4).
Separación Static-First entre teoría de cátedra y simulación aislada con @st.fragment.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core._cache_utils import evaluar_fn_vectorizado
from core.edo_pvi import (
    compilar_exacta_edo,
    compilar_funcion_edo,
    metodo_euler,
    metodo_euler_modificado,
    metodo_milne,
    metodo_runge_kutta_2,
    metodo_runge_kutta_4,
)
from views.theme import PALETA_ARANDU
from views.ui_shared import (
    crear_figura_base_plotly,
    inicializar_estado_practica,
    inject_arandu_css,
    muestrear_para_grafico,
    registrar_contexto_ia,
    render_bloque_1_teoria_y_ejemplo,
    render_dataframe_acotado,
    render_deduccion_paso_a_paso_viva,
    render_excepcion_validacion,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
    render_toggle_modo_detallado,
    rerun_fragmento_seguro,
    restaurar_estado_practica,
)


def _inputs_pvi(clave_state: str, solo_lectura: bool) -> tuple[str, float, float, float, float, str]:
    c1, c2 = st.columns(2)
    f_str = c1.text_input(
        "EDO $y' = f(x, y)$:",
        value=str(st.session_state[f"{clave_state}_f_str"]),
        disabled=solo_lectura,
        help="Expresión en variables x, y con operadores Python (*, **). Ej: 6*x**2 / y",
        key=f"ui_{clave_state}_f",
    )
    y_ex_str = c2.text_input(
        "Solución exacta $y(x)$ (opcional):",
        value=str(st.session_state[f"{clave_state}_y_exacta_str"]),
        disabled=solo_lectura,
        help="Expresión analítica en variable x para medir error real. Ej: 2*x*sqrt(x)",
        key=f"ui_{clave_state}_yex",
    )

    c3, c4, c5, c6 = st.columns(4)
    x0 = c3.number_input(
        "$x_0$:",
        value=float(st.session_state[f"{clave_state}_x0"]),
        format="%.4f",
        disabled=solo_lectura,
        help="Abscisa inicial (real, distinta de x_final).",
        key=f"ui_{clave_state}_x0",
    )
    y0 = c4.number_input(
        "$y_0 = y(x_0)$:",
        value=float(st.session_state[f"{clave_state}_y0"]),
        format="%.4f",
        disabled=solo_lectura,
        help="Condición inicial y(x_0) dentro del dominio de f(x, y).",
        key=f"ui_{clave_state}_y0",
    )
    xf = c5.number_input(
        r"$x_{\text{final}}$:",
        value=float(st.session_state[f"{clave_state}_xf"]),
        format="%.4f",
        disabled=solo_lectura,
        help="Extremo de integración final; en Milne debe cubrir al menos 4 pasos h.",
        key=f"ui_{clave_state}_xf",
    )
    h = c6.number_input(
        r"Paso $\Delta h$:",
        value=float(st.session_state[f"{clave_state}_h"]),
        min_value=0.005,
        max_value=2.0,
        step=0.05,
        format="%.4f",
        disabled=solo_lectura,
        help="Incremento de malla Δh en rango [0.005, 2.0].",
        key=f"ui_{clave_state}_h",
    )

    if not solo_lectura:
        st.session_state[f"{clave_state}_f_str"] = f_str
        st.session_state[f"{clave_state}_y_exacta_str"] = y_ex_str
        st.session_state[f"{clave_state}_x0"] = x0
        st.session_state[f"{clave_state}_y0"] = y0
        st.session_state[f"{clave_state}_xf"] = xf
        st.session_state[f"{clave_state}_h"] = h

    return f_str, x0, y0, xf, h, y_ex_str


def _compilar_pvi(f_str: str, y_ex_str: str) -> tuple[dict, object | None, object | None]:
    comp_f = compilar_funcion_edo(f_str)
    y_ex_fn = None
    y_ex_fn_np = None
    if y_ex_str.strip():
        try:
            comp_ex = compilar_exacta_edo(y_ex_str)
            y_ex_fn = comp_ex["fn"]
            y_ex_fn_np = comp_ex.get("fn_np")
        except Exception as exc:
            st.warning(f"Solución exacta ignorada: {exc}")
    return comp_f, y_ex_fn, y_ex_fn_np


def _plot_edo_paso_a_paso(
    titulo: str,
    x0: float,
    xf_real: float,
    series: list[tuple[str, list[float], list[float], str]],
    y_ex_fn,
    y_ex_fn_np,
    k_vis: int,
    chart_key: str,
    x_label: str = "x",
    y_label: str = "y(x)",
) -> None:
    fig = crear_figura_base_plotly(titulo, x_label, y_label)
    if y_ex_fn_np is not None or y_ex_fn is not None:
        xg = np.linspace(x0, xf_real, 220)
        try:
            if y_ex_fn_np is not None:
                yg = evaluar_fn_vectorizado(y_ex_fn_np, xg, limite_abs=1e6)
            else:
                yg = evaluar_fn_vectorizado(np.vectorize(y_ex_fn), xg, limite_abs=1e6)
            fig.add_trace(
                go.Scatter(
                    x=xg.tolist(),
                    y=yg,
                    mode="lines",
                    name="Solución Exacta",
                    line=dict(color=PALETA_ARANDU["oro"], width=2.8, dash="dash"),
                )
            )
        except Exception:
            pass

    estilos_dash = ("solid", "dot", "dashdot")
    simbolos = ("circle", "diamond", "square")
    for idx, (nombre, xs, ys, color) in enumerate(series):
        pares = muestrear_para_grafico(list(zip(xs[: k_vis + 1], ys[: k_vis + 1])), max_puntos=300)
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pares],
                y=[p[1] for p in pares],
                mode="lines+markers",
                name=nombre,
                line=dict(color=color, width=2.4, dash=estilos_dash[idx % len(estilos_dash)]),
                marker=dict(size=7, symbol=simbolos[idx % len(simbolos)]),
            )
        )
    render_plotly_arandu(fig, key=chart_key)


_EDO_TERMICA_F = "-0.5*(y - 25)"
_EDO_TERMICA_EXACTA = "25 + 60*exp(-0.5*x)"
_EDO_LOGISTICA_F = "1.2*y*(1 - y/100)"
_EDO_LOGISTICA_EXACTA = "100 / (1 + 19*exp(-1.2*x))"
_EDO_RC_F = "-0.8*(y - 1)"
_EDO_RC_EXACTA = "1 + 9*exp(-0.8*x)"


def _aplicar_preset_edo(valores_estado: dict[str, float], claves_ui: tuple[str, ...]) -> None:
    """Actualiza parámetros de un escenario real en session_state limpiando las claves UI."""
    for clave, valor in valores_estado.items():
        st.session_state[clave] = float(valor)
    for clave_ui in claves_ui:
        st.session_state.pop(clave_ui, None)


def _delta_error_edo(res: dict, fmt_err: str, unidad_txt: str) -> str:
    """Formatea el delta de error final respecto a la solución analítica."""
    if res.get("error_final") is not None:
        return f"Err: {res['error_final']:{fmt_err}}{unidad_txt} ({res['error_rel_final_pct']:.3f}%)"
    return "Sin exacta de referencia"


def _render_metric_exacta_malla(col: object, res: dict, fmt_v: str, unidad_txt: str) -> None:
    """Dibuja el KPI de solución exacta final y tamaño de malla."""
    ye_final = res["tabla"][-1].get("y_exacta")
    if ye_final is not None:
        col.metric(
            "Solución Exacta y Malla",
            f"{float(ye_final):{fmt_v}}{unidad_txt}",
            f"{res['n_pasos']} pasos de h = {res['h']:g}",
            delta_color="off",
        )
    else:
        col.metric("Solución Exacta y Malla", f"{res['n_pasos']} pasos", f"h = {res['h']:g}", delta_color="off")


@st.fragment
def _frag_euler_heun(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_termico", "edo_libre"],
        format_func=lambda op: (
            "Caso Real: Disipación Térmica de CPU dT/dt = -0.5(T - 25) → Temperatura (°C)"
            if op == "real_termico"
            else "Función / EDO Libre f(x, y) y Cátedra TP7"
        ),
        horizontal=True,
        key="ui_edo_eh_contexto",
    )
    es_caso_real = contexto == "real_termico"

    if es_caso_real:
        st.caption(
            r"Enfriamiento de un procesador desde $T(0) = 85^\circ\text{C}$ con temperatura ambiente $T_{\text{amb}} = 25^\circ\text{C}$: "
            r"$\frac{dT}{dt} = -0.5(T - 25)$. Compará la deriva lineal de **Euler** frente a la corrección de **Heun**."
        )
        if "edo_eh_real_t0" not in st.session_state:
            st.session_state["edo_eh_real_t0"] = 85.0
        if "edo_eh_real_xf" not in st.session_state:
            st.session_state["edo_eh_real_xf"] = 4.0
        if "edo_eh_real_h" not in st.session_state:
            st.session_state["edo_eh_real_h"] = 0.5

        b1, b2, b3 = st.columns(3)
        if b1.button(
            "Paso moderado (Δt = 0.5 s en [0, 4] s → Heun corrige la deriva de Euler)",
            key="btn_eh_real_mod",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_eh_real_h": 0.5, "edo_eh_real_xf": 4.0},
                ("ui_eh_real_h", "ui_eh_real_xf"),
            )
            rerun_fragmento_seguro()
        if b2.button(
            "Paso grande (Δt = 1.0 s en [0, 4] s → Fuerte desvío lineal de Euler O(h))",
            key="btn_eh_real_grande",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_eh_real_h": 1.0, "edo_eh_real_xf": 4.0},
                ("ui_eh_real_h", "ui_eh_real_xf"),
            )
            rerun_fragmento_seguro()
        if b3.button(
            "Paso fino (Δt = 0.2 s en [0, 4] s → Convergencia de ambos métodos)",
            key="btn_eh_real_fino",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_eh_real_h": 0.2, "edo_eh_real_xf": 4.0},
                ("ui_eh_real_h", "ui_eh_real_xf"),
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns(3)
        y0 = float(
            c1.number_input(
                "Temperatura inicial T₀ (°C):",
                min_value=25.0,
                max_value=150.0,
                value=float(st.session_state["edo_eh_real_t0"]),
                step=5.0,
                format="%.1f",
                key="ui_eh_real_t0",
            )
        )
        xf = float(
            c2.number_input(
                "Tiempo final t_f (s):",
                min_value=0.5,
                max_value=15.0,
                value=float(st.session_state["edo_eh_real_xf"]),
                step=0.5,
                format="%.1f",
                key="ui_eh_real_xf",
            )
        )
        h = float(
            c3.number_input(
                "Paso temporal Δt = h (s):",
                min_value=0.05,
                max_value=2.0,
                value=float(st.session_state["edo_eh_real_h"]),
                step=0.05,
                format="%.2f",
                key="ui_eh_real_h",
            )
        )
        st.session_state["edo_eh_real_t0"] = y0
        st.session_state["edo_eh_real_xf"] = xf
        st.session_state["edo_eh_real_h"] = h
        x0 = 0.0
        f_str = _EDO_TERMICA_F
        y_ex_str = _EDO_TERMICA_EXACTA if abs(y0 - 85.0) < 1e-6 else f"25 + ({y0 - 25.0:g})*exp(-0.5*x)"
    else:
        inicializar_estado_practica("edo_eh", defaults)
        c_info, c_rst = st.columns([2.4, 1.1])
        c_info.caption("Editá la EDO $y' = f(x, y)$, condiciones iniciales $(x_0, y_0)$, paso $\\Delta h$ o restaurá el ejercicio TP7 Ej. 1.")
        if c_rst.button("Restaurar TP7 Ej. 1", key="btn_eh_rest_tp7", use_container_width=True):
            restaurar_estado_practica("edo_eh", defaults)
            rerun_fragmento_seguro()
        f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_eh", solo_lectura=False)

    iterar_corr = st.checkbox(
        "Iterar corrector de Heun hasta convergencia (|C⁽ᵐ⁾ - C⁽ᵐ⁻¹⁾| < 10⁻⁴)",
        value=True,
        key="ui_eh_corr",
    )

    try:
        comp_f, y_ex_fn, y_ex_fn_np = _compilar_pvi(f_str, y_ex_str)
        res_eu = metodo_euler(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
        res_he = metodo_euler_modificado(
            comp_f["fn"], x0, y0, h, xf=xf, iterar_corrector=iterar_corr, y_exacta_fn=y_ex_fn
        )
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="EDO f(x, y), condiciones iniciales o paso Δh",
            sugerencia_defecto="Verificá sintaxis Python de f(x,y), evitá división por cero en (x_0, y_0) y asegurate de que x_final > x_0.",
            ejemplo_defecto="f(x, y) = 6*x**2 / y, x0 = 4.0, y0 = 16.0, xf = 4.4, h = 0.1",
        )
        return

    for av in res_he["avisos"]:
        st.info(av)

    unidad_txt = " °C" if es_caso_real else ""
    fmt_v = ".4f" if es_caso_real else ".6f"
    fmt_err = ".4f" if es_caso_real else ".5f"

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Euler Clásico (Orden O(h))",
        f"{res_eu['y_final']:{fmt_v}}{unidad_txt}",
        _delta_error_edo(res_eu, fmt_err, unidad_txt),
        delta_color="off",
    )
    m2.metric(
        "Heun / Euler Modificado (Orden O(h²))",
        f"{res_he['y_final']:{fmt_v}}{unidad_txt}",
        _delta_error_edo(res_he, fmt_err, unidad_txt),
        delta_color="off",
    )
    _render_metric_exacta_malla(m3, res_he, fmt_v, unidad_txt)

    k_vis = (
        render_slider_paso_k("Ver evolución hasta el paso $k$:", res_he["n_pasos"], "vis_eh_k")
        if modo_detallado
        else res_he["n_pasos"]
    )
    _plot_edo_paso_a_paso(
        (
            f"Enfriamiento T(t): Euler vs Heun (Δt = {res_he['h']:g} s)"
            if es_caso_real
            else f"Euler vs Heun hasta paso k = {k_vis}"
        ),
        x0,
        res_he["xf_real"],
        [
            ("Euler Clásico (O(h))", res_eu["x"], res_eu["y"], PALETA_ARANDU["terracota"]),
            ("Heun / Euler Modificado (O(h²))", res_he["x"], res_he["y"], PALETA_ARANDU["salvia"]),
        ],
        y_ex_fn,
        y_ex_fn_np,
        k_vis,
        chart_key="plot_edo_eh",
        x_label="Tiempo t (s)" if es_caso_real else "x",
        y_label="Temperatura T(t) [°C]" if es_caso_real else "y(x)",
    )

    registrar_contexto_ia(
        "Resultados PVI Euler vs Heun",
        f"- **Contexto:** {'Disipación Térmica CPU' if es_caso_real else 'EDO Libre / TP7'} | **EDO:** `{f_str}` | "
        f"**$x_0 = {x0:g}, y_0 = {y0:g}, h = {res_he['h']:g}, x_f = {res_he['xf_real']:g}$** (`n = {res_he['n_pasos']}` pasos)\n"
        f"- **Euler:** `{res_eu['y_final']:{fmt_v}}{unidad_txt}` | **Heun:** `{res_he['y_final']:{fmt_v}}{unidad_txt}`",
        clave="edo_eh_res",
    )

    if modo_detallado:
        pasos_eh_vivos = [
            f"**Condición inicial y malla:** EDO $y' = {comp_f['f_latex']}$, $y({x0:g}) = {y0:g}$, paso $h = {res_he['h']:g}$ hasta $x_f = {res_he['xf_real']:g}$ ({res_he['n_pasos']} pasos)."
        ]
        for r_eu, r_he in zip(res_eu["tabla"][1:6], res_he["tabla"][1:6]):
            pasos_eh_vivos.append(
                f"**Paso $k = {r_he['k']}$ ($x_{{{r_he['k']}}} = {r_he['x_k']:g}$):** "
                f"Euler predice $y = {r_eu['y_k']:.6f}$; "
                f"Heun parte con $k_1 = {r_he['k1_pend']:.5f}$, $P = {r_he['P_euler']:.6f}$, $k_2 = {r_he['k2_pend']:.5f}$ y corrige ({r_he['iters_corr']} it.) a $C = {r_he['C_final']:.6f}$."
            )
        if res_he["n_pasos"] > 5:
            pasos_eh_vivos.append(
                f"**Cierre en $x_f = {res_he['xf_real']:g}$:** Euler $= {res_eu['y_final']:.6f}$ | Heun $= {res_he['y_final']:.6f}$."
            )
        render_deduccion_paso_a_paso_viva(
            f"Euler vs Heun (h={res_he['h']:g})",
            pasos_eh_vivos,
            clave_ia="edo_eh_viva",
            expanded=True,
        )

        filas_euler = [
            {
                "Paso k": r["k"],
                "Nodo xₖ": round(float(r["x_k"]), 4),
                "Euler yₖ": round(float(r["y_k"]), 6),
                "Pendiente f(xₖ, yₖ)": round(float(r["f(x_k, y_k)"]), 6),
                "Incremento h·f": round(float(r["Delta_y"]), 6),
                "Valor Exacto": round(float(r["y_exacta"]), 6) if r["y_exacta"] is not None else None,
                "Error Absoluto |E|": round(float(r["error_abs"]), 6) if r["error_abs"] is not None else None,
            }
            for r in res_eu["tabla"]
        ]
        filas_heun = [
            {
                "Paso k": r["k"],
                "Nodo xₖ": round(float(r["x_k"]), 4),
                "Predictor P (Euler)": round(float(r["P_euler"]), 6) if r["P_euler"] is not None else None,
                "Pendiente Inicial k₁": round(float(r["k1_pend"]), 6) if r["k1_pend"] is not None else None,
                "Pendiente Destino k₂": round(float(r["k2_pend"]), 6) if r["k2_pend"] is not None else None,
                "Corrector Final C": round(float(r["C_final"]), 6),
                "Iteraciones Corrector": int(r["iters_corr"]),
                "Valor Exacto": round(float(r["y_exacta"]), 6) if r["y_exacta"] is not None else None,
                "Error Absoluto |E|": round(float(r["error_abs"]), 6) if r["error_abs"] is not None else None,
            }
            for r in res_he["tabla"]
        ]
        t1, t2 = st.tabs(["Tabla Euler", "Tabla Heun (Euler Modificado)"])
        with t1:
            render_dataframe_acotado(filas_euler, max_filas=300, key_prefix="tbl_edo_euler")
        with t2:
            render_dataframe_acotado(filas_heun, max_filas=300, key_prefix="tbl_edo_heun")


def _tab_euler_heun() -> None:
    modo_detallado = render_toggle_modo_detallado("euler_heun")
    meta = render_bloque_1_teoria_y_ejemplo("euler_heun", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.4, "y_exacta_str": "2*x*sqrt(x)"},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_euler_heun(defaults, modo_detallado=modo_detallado)


def _construir_filas_rk(tabla: list[dict], orden: int) -> list[dict]:
    """Formatea la tabla de Runge-Kutta (RK2 o RK4) con encabezados limpios en español."""
    filas: list[dict] = []
    etiqueta_y = f"RK{orden} yₖ"
    for r in tabla:
        fila: dict[str, object] = {
            "Paso k": r["k"],
            "Nodo xₖ": round(float(r["x_k"]), 4),
            etiqueta_y: round(float(r["y_k"]), 6),
            "Pendiente k₁": round(float(r["k1"]), 6) if r["k1"] is not None else None,
            "Pendiente k₂": round(float(r["k2"]), 6) if r["k2"] is not None else None,
        }
        if orden == 4:
            fila["Pendiente k₃"] = round(float(r["k3"]), 6) if r["k3"] is not None else None
            fila["Pendiente k₄"] = round(float(r["k4"]), 6) if r["k4"] is not None else None
        fila["Valor Exacto"] = round(float(r["y_exacta"]), 6) if r["y_exacta"] is not None else None
        fila["Error Absoluto |E|"] = round(float(r["error_abs"]), 6) if r["error_abs"] is not None else None
        filas.append(fila)
    return filas


@st.fragment
def _frag_runge_kutta(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_logistico", "edo_libre"],
        format_func=lambda op: (
            "Caso Real: Escalado Cloud dU/dt = 1.2·U·(1 - U/100) → Usuarios (miles)"
            if op == "real_logistico"
            else "Función / EDO Libre f(x, y) y Cátedra TP7"
        ),
        horizontal=True,
        key="ui_edo_rk_contexto",
    )
    es_caso_real = contexto == "real_logistico"

    if es_caso_real:
        st.caption(
            r"Escalado de tráfico web desde $U(0) = 5$ mil usuarios concurrentes hacia capacidad $K = 100$ mil: "
            r"$\frac{dU}{dt} = 1.2\,U\,(1 - \frac{U}{100})$. Compará la deriva de **RK2** ($\mathcal{O}(h^2)$) frente a **RK4 Clásico** ($\mathcal{O}(h^4)$)."
        )
        if "edo_rk_real_u0" not in st.session_state:
            st.session_state["edo_rk_real_u0"] = 5.0
        if "edo_rk_real_xf" not in st.session_state:
            st.session_state["edo_rk_real_xf"] = 6.0
        if "edo_rk_real_h" not in st.session_state:
            st.session_state["edo_rk_real_h"] = 1.0

        claves_ui_rk = ("ui_rk_real_u0", "ui_rk_real_xf", "ui_rk_real_h")
        b1, b2, b3 = st.columns(3)
        if b1.button(
            "Paso grueso (Δt = 1.0 h en [0, 6] h → RK4 preciso, RK2 se desvía)",
            key="btn_rk_real_mod",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_rk_real_u0": 5.0, "edo_rk_real_xf": 6.0, "edo_rk_real_h": 1.0},
                claves_ui_rk,
            )
            rerun_fragmento_seguro()
        if b2.button(
            "Paso extremo (Δt = 1.5 h en [0, 6] h → Fuerte error de RK2 en inflexión)",
            key="btn_rk_real_extremo",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_rk_real_u0": 5.0, "edo_rk_real_xf": 6.0, "edo_rk_real_h": 1.5},
                claves_ui_rk,
            )
            rerun_fragmento_seguro()
        if b3.button(
            "Paso fino (Δt = 0.5 h en [0, 6] h → Referencia convergente)",
            key="btn_rk_real_fino",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_rk_real_u0": 5.0, "edo_rk_real_xf": 6.0, "edo_rk_real_h": 0.5},
                claves_ui_rk,
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns(3)
        y0 = float(
            c1.number_input(
                "Usuarios iniciales U₀ (miles):",
                min_value=1.0,
                max_value=99.0,
                value=float(st.session_state["edo_rk_real_u0"]),
                step=1.0,
                format="%.1f",
                key="ui_rk_real_u0",
            )
        )
        xf = float(
            c2.number_input(
                "Horizonte t_f (h):",
                min_value=1.0,
                max_value=12.0,
                value=float(st.session_state["edo_rk_real_xf"]),
                step=0.5,
                format="%.1f",
                key="ui_rk_real_xf",
            )
        )
        h = float(
            c3.number_input(
                "Paso temporal Δt = h (h):",
                min_value=0.1,
                max_value=2.0,
                value=float(st.session_state["edo_rk_real_h"]),
                step=0.1,
                format="%.2f",
                key="ui_rk_real_h",
            )
        )
        st.session_state["edo_rk_real_u0"] = y0
        st.session_state["edo_rk_real_xf"] = xf
        st.session_state["edo_rk_real_h"] = h
        x0 = 0.0
        f_str = _EDO_LOGISTICA_F
        c_ratio = (100.0 - y0) / y0
        y_ex_str = _EDO_LOGISTICA_EXACTA if abs(y0 - 5.0) < 1e-6 else f"100 / (1 + ({c_ratio:.10g})*exp(-1.2*x))"
    else:
        inicializar_estado_practica("edo_rk", defaults)
        c_info, c_rst = st.columns([2.4, 1.1])
        c_info.caption(
            "Editá la EDO $y' = f(x, y)$, condiciones iniciales $(x_0, y_0)$, paso $\\Delta h$ o restaurá el ejercicio TP7 Ej. 2."
        )
        if c_rst.button("Restaurar TP7 Ej. 2", key="btn_rk_rest_tp7", use_container_width=True):
            restaurar_estado_practica("edo_rk", defaults)
            rerun_fragmento_seguro()
        f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_rk", solo_lectura=False)

    try:
        comp_f, y_ex_fn, y_ex_fn_np = _compilar_pvi(f_str, y_ex_str)
        res_rk2 = metodo_runge_kutta_2(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
        res_rk4 = metodo_runge_kutta_4(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="EDO f(x, y), condiciones iniciales o paso Δh",
            sugerencia_defecto="Verificá sintaxis Python de f(x,y), evitá división por cero en (x_0, y_0) y asegurate de que x_final != x_0.",
            ejemplo_defecto="f(x, y) = 6*x**2 / y, x0 = 4.0, y0 = 16.0, xf = 4.3, h = 0.1",
        )
        return

    for av in res_rk4["avisos"]:
        st.info(av)

    unidad_txt = " k" if es_caso_real else ""
    fmt_v = ".4f" if es_caso_real else ".6f"
    fmt_err = ".4f" if es_caso_real else ".5f"

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "RK2 (Orden O(h²))",
        f"{res_rk2['y_final']:{fmt_v}}{unidad_txt}",
        _delta_error_edo(res_rk2, fmt_err, unidad_txt),
        delta_color="off",
    )
    m2.metric(
        "RK4 Clásico (Orden O(h⁴))",
        f"{res_rk4['y_final']:{fmt_v}}{unidad_txt}",
        _delta_error_edo(res_rk4, fmt_err, unidad_txt),
        delta_color="off",
    )
    _render_metric_exacta_malla(m3, res_rk4, fmt_v, unidad_txt)

    k_vis = (
        render_slider_paso_k("Ver evolución hasta el paso $k$:", res_rk4["n_pasos"], "vis_rk_k")
        if modo_detallado
        else res_rk4["n_pasos"]
    )
    _plot_edo_paso_a_paso(
        (
            f"Escalado Logístico U(t): RK2 vs RK4 (Δt = {res_rk4['h']:g} h)"
            if es_caso_real
            else f"RK2 vs RK4 hasta paso k = {k_vis}"
        ),
        x0,
        res_rk4["xf_real"],
        [
            ("RK2 (Orden O(h²))", res_rk2["x"], res_rk2["y"], PALETA_ARANDU["terracota"]),
            ("RK4 Clásico (Orden O(h⁴))", res_rk4["x"], res_rk4["y"], PALETA_ARANDU["salvia"]),
        ],
        y_ex_fn,
        y_ex_fn_np,
        k_vis,
        chart_key="plot_edo_rk",
        x_label="Tiempo t (h)" if es_caso_real else "x",
        y_label="Usuarios concurrentes U(t) [miles]" if es_caso_real else "y(x)",
    )

    registrar_contexto_ia(
        "Resultados PVI RK2 vs RK4",
        f"- **Contexto:** {'Escalado Cloud Logístico' if es_caso_real else 'EDO Libre / TP7'} | **EDO:** `{f_str}` | "
        f"**$x_0 = {x0:g}, y_0 = {y0:g}, h = {res_rk4['h']:g}, x_f = {res_rk4['xf_real']:g}$** (`n = {res_rk4['n_pasos']}` pasos)\n"
        f"- **RK2:** `{res_rk2['y_final']:{fmt_v}}{unidad_txt}` | **RK4:** `{res_rk4['y_final']:{fmt_v}}{unidad_txt}`",
        clave="edo_rk_res",
    )

    if modo_detallado:
        pasos_rk_vivos = [
            f"**Condición inicial y malla:** $y' = {comp_f['f_latex']}$, $y({x0:g}) = {y0:g}$, paso $h = {res_rk4['h']:g}$ ({res_rk4['n_pasos']} pasos hasta $x_f = {res_rk4['xf_real']:g}$)."
        ]
        for r2, r4 in zip(res_rk2["tabla"][1:6], res_rk4["tabla"][1:6]):
            pasos_rk_vivos.append(
                f"**Paso $k = {r4['k']}$ ($x_{{{r4['k']}}} = {r4['x_k']:g}$):** "
                f"RK2 ($k_1={r2['k1']:.5f}, k_2={r2['k2']:.5f}$) $\\implies y = {r2['y_k']:.6f}$; "
                f"RK4 ($k_1={r4['k1']:.5f}, k_2={r4['k2']:.5f}, k_3={r4['k3']:.5f}, k_4={r4['k4']:.5f}$) $\\implies y = {r4['y_k']:.6f}$."
            )
        if res_rk4["n_pasos"] > 5:
            pasos_rk_vivos.append(
                f"**Cierre en $x_f = {res_rk4['xf_real']:g}$:** RK2 $= {res_rk2['y_final']:.6f}$ | RK4 $= {res_rk4['y_final']:.6f}$."
            )
        render_deduccion_paso_a_paso_viva(
            f"Runge-Kutta RK2 vs RK4 (h={res_rk4['h']:g})",
            pasos_rk_vivos,
            clave_ia="edo_rk_viva",
            expanded=True,
        )

        t1, t2 = st.tabs(["Tabla RK4 Clásico (k₁..k₄)", "Tabla RK2 (k₁, k₂)"])
        with t1:
            render_dataframe_acotado(
                _construir_filas_rk(res_rk4["tabla"], orden=4),
                max_filas=300,
                key_prefix="tbl_edo_rk4",
            )
        with t2:
            render_dataframe_acotado(
                _construir_filas_rk(res_rk2["tabla"], orden=2),
                max_filas=300,
                key_prefix="tbl_edo_rk2",
            )


def _tab_runge_kutta() -> None:
    modo_detallado = render_toggle_modo_detallado("runge_kutta")
    meta = render_bloque_1_teoria_y_ejemplo("runge_kutta", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.3, "y_exacta_str": "2*x*sqrt(x)"},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_runge_kutta(defaults, modo_detallado=modo_detallado)


def _construir_filas_milne(tabla: list[dict]) -> list[dict]:
    """Formatea la tabla del método multipaso de Milne con encabezados limpios en español."""
    filas: list[dict] = []
    for r in tabla:
        diff_pc = r.get("diff_P_C")
        filas.append(
            {
                "Paso k": r["k"],
                "Nodo xₖ": round(float(r["x_k"]), 4),
                "Fase / Origen": r["origen"],
                "Predictor P": round(float(r["P"]), 6) if r["P"] is not None else None,
                "Corrector 1 (C)": round(float(r["C"]), 6) if r["C"] is not None else None,
                "Valor Final (C¹/yₖ)": round(float(r["C_final"]), 6),
                "Pendiente y'ₖ": round(float(r["y'_k"]), 6),
                "|C − P|": round(float(diff_pc), 6) if diff_pc is not None else None,
                "Error Est. |C−P|/29": round(float(diff_pc) / 29.0, 6) if diff_pc is not None else None,
                "Valor Exacto": round(float(r["y_exacta"]), 6) if r["y_exacta"] is not None else None,
                "Error Absoluto |E|": round(float(r["error_abs"]), 6) if r["error_abs"] is not None else None,
            }
        )
    return filas


@st.fragment
def _frag_milne(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_rc", "edo_libre"],
        format_func=lambda op: (
            "Caso Real: Filtro RC dV/dt = -0.8(V - 1) → Tensión V(t) (V)"
            if op == "real_rc"
            else "Función / EDO Libre f(x, y) y Cátedra TP7"
        ),
        horizontal=True,
        key="ui_edo_mil_contexto",
    )
    es_caso_real = contexto == "real_rc"

    if es_caso_real:
        st.caption(
            r"Descarga de un capacitor de filtro desde $V(0) = 10\;\text{V}$ hacia $1\;\text{V}$: "
            r"$\frac{dV}{dt} = -0.8(V - 1)$. Arranca con $4$ puntos de **RK4** ($k \le 3$) y avanza con **Milne P-C** ($k \ge 4$)."
        )
        if "edo_mil_real_v0" not in st.session_state:
            st.session_state["edo_mil_real_v0"] = 10.0
        if "edo_mil_real_xf" not in st.session_state:
            st.session_state["edo_mil_real_xf"] = 10.0
        if "edo_mil_real_h" not in st.session_state:
            st.session_state["edo_mil_real_h"] = 0.25

        claves_ui_mil = ("ui_mil_real_v0", "ui_mil_real_xf", "ui_mil_real_h")
        b1, b2, b3 = st.columns(3)
        if b1.button(
            "Malla fina (Δt = 0.25 s, 40 pasos → Alta precisión y ahorro de f)",
            key="btn_mil_real_fino",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_mil_real_v0": 10.0, "edo_mil_real_xf": 10.0, "edo_mil_real_h": 0.25},
                claves_ui_mil,
            )
            rerun_fragmento_seguro()
        if b2.button(
            "Malla intermedia (Δt = 0.50 s, 20 pasos → P-C rápido en [0, 10] s)",
            key="btn_mil_real_med",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_mil_real_v0": 10.0, "edo_mil_real_xf": 10.0, "edo_mil_real_h": 0.50},
                claves_ui_mil,
            )
            rerun_fragmento_seguro()
        if b3.button(
            "Paso crítico (Δt = 1.25 s, 8 pasos → Oscilación parásita de Milne)",
            key="btn_mil_real_critico",
            use_container_width=True,
        ):
            _aplicar_preset_edo(
                {"edo_mil_real_v0": 10.0, "edo_mil_real_xf": 10.0, "edo_mil_real_h": 1.25},
                claves_ui_mil,
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns(3)
        y0 = float(
            c1.number_input(
                "Tensión inicial V₀ (V):",
                min_value=2.0,
                max_value=24.0,
                value=float(st.session_state["edo_mil_real_v0"]),
                step=1.0,
                format="%.1f",
                key="ui_mil_real_v0",
            )
        )
        xf = float(
            c2.number_input(
                "Tiempo final t_f (s):",
                min_value=2.0,
                max_value=15.0,
                value=float(st.session_state["edo_mil_real_xf"]),
                step=0.5,
                format="%.1f",
                key="ui_mil_real_xf",
            )
        )
        h = float(
            c3.number_input(
                "Paso temporal Δt = h (s):",
                min_value=0.1,
                max_value=2.5,
                value=float(st.session_state["edo_mil_real_h"]),
                step=0.05,
                format="%.2f",
                key="ui_mil_real_h",
            )
        )
        st.session_state["edo_mil_real_v0"] = y0
        st.session_state["edo_mil_real_xf"] = xf
        st.session_state["edo_mil_real_h"] = h
        x0 = 0.0
        f_str = _EDO_RC_F
        y_ex_str = _EDO_RC_EXACTA if abs(y0 - 10.0) < 1e-6 else f"1 + ({y0 - 1.0:.10g})*exp(-0.8*x)"
    else:
        inicializar_estado_practica("edo_mil", defaults)
        c_info, c_rst = st.columns([2.4, 1.1])
        c_info.caption(
            "Editá la EDO $y' = f(x, y)$, condiciones iniciales $(x_0, y_0)$, paso $\\Delta h$ (mínimo $4$ pasos) o restaurá TP7 Ej. 3."
        )
        if c_rst.button("Restaurar TP7 Ej. 3", key="btn_mil_rest_tp7", use_container_width=True):
            restaurar_estado_practica("edo_mil", defaults)
            rerun_fragmento_seguro()
        f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_mil", solo_lectura=False)

    doble_corr = st.checkbox(
        "Aplicar 2da corrección C¹ de cátedra (si se desactiva: 1 pasada P-C = 2 evals/paso vs 4 de RK4)",
        value=True,
        key="ui_mil_doble_corr",
    )
    n_corr = 2 if doble_corr else 1

    try:
        comp_f, y_ex_fn, y_ex_fn_np = _compilar_pvi(f_str, y_ex_str)
        res = metodo_milne(comp_f["fn"], x0, y0, h, xf=xf, n_correcciones=n_corr, y_exacta_fn=y_ex_fn)
        res_rk4 = metodo_runge_kutta_4(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Intervalo de Milne [x_0, x_final] o f(x, y)",
            sugerencia_defecto="Milne requiere al menos 4 pasos completos desde x_0 (|x_final - x_0| >= 4*Δh). Reducí Δh o ampliá x_final.",
            ejemplo_defecto="x0 = 4.0, xf = 4.4, h = 0.1 (4 pasos)",
        )
        return

    for av in res["avisos"]:
        st.info(av)

    unidad_txt = " V" if es_caso_real else ""
    fmt_v = ".4f" if es_caso_real else ".6f"
    fmt_err = ".4f" if es_caso_real else ".5f"

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Milne P-C (Orden O(h⁴))",
        f"{res['y_final']:{fmt_v}}{unidad_txt}",
        _delta_error_edo(res, fmt_err, unidad_txt),
        delta_color="off",
    )
    evals_mil = res["evals_f"]
    evals_rk = res["evals_rk4_equiv"]
    if evals_mil < evals_rk:
        ahorro = evals_rk - evals_mil
        unidad_ev = "eval" if ahorro == 1 else "evals"
        delta_costo = f"Ahorro {ahorro} {unidad_ev} ({ahorro / evals_rk * 100.0:.0f}% vs {evals_rk} RK4)"
    else:
        delta_costo = f"RK4: {evals_rk} evals (arranque domina en {res['n_pasos']} pasos)"
    m2.metric("Costo de Evaluaciones f", f"{evals_mil} evals", delta_costo, delta_color="off")

    ye_final = res["tabla"][-1].get("y_exacta")
    err_est = res.get("err_est_local_final")
    delta_est = (
        f"Error est. |C−P|/29 ≈ {err_est:.2e}"
        if err_est is not None
        else f"{res['n_pasos']} pasos de h = {res['h']:g}"
    )
    val_m3 = f"{float(ye_final):{fmt_v}}{unidad_txt}" if ye_final is not None else f"{res['n_pasos']} pasos"
    m3.metric("Exacta y Est. Local", val_m3, delta_est, delta_color="off")

    k_vis = (
        render_slider_paso_k("Ver evolución hasta el paso $k$:", res["n_pasos"], "vis_mil_k")
        if modo_detallado
        else res["n_pasos"]
    )
    _plot_edo_paso_a_paso(
        (
            f"Descarga RC V(t): Milne P-C vs RK4 (Δt = {res['h']:g} s)"
            if es_caso_real
            else f"Milne (Arranque RK4 k≤3, P-C k≥4) hasta paso k = {k_vis}"
        ),
        x0,
        res["xf_real"],
        [
            ("Milne P-C (Orden O(h⁴))", res["x"], res["y"], PALETA_ARANDU["salvia"]),
            ("Referencia RK4 (Orden O(h⁴))", res_rk4["x"], res_rk4["y"], PALETA_ARANDU["terracota"]),
        ],
        y_ex_fn,
        y_ex_fn_np,
        k_vis,
        chart_key="plot_edo_milne",
        x_label="Tiempo t (s)" if es_caso_real else "x",
        y_label="Tensión V(t) [V]" if es_caso_real else "y(x)",
    )

    err_fin = res["tabla"][-1].get("error_abs")
    err_mil = f"{err_fin:.3e}" if err_fin is not None else "No provista"
    registrar_contexto_ia(
        "Resultados PVI Milne Predictor-Corrector",
        f"- **Contexto:** {'Filtro RC' if es_caso_real else 'EDO Libre / TP7'} | **EDO:** `{f_str}` | "
        f"**$x_0 = {x0:g}, y_0 = {y0:g}, h = {res['h']:g}, x_f = {res['xf_real']:g}$**\n"
        f"- **Milne $y(x_f)$:** `{res['y_final']:{fmt_v}}{unidad_txt}` | **Evals f:** `{evals_mil}` vs RK4 `{evals_rk}` | **Error abs.:** `{err_mil}`",
        clave="edo_mil_res",
    )

    if modo_detallado:
        arr_pts = ", ".join(f"y({r['x_k']:g})={r['C_final']:.6f}" for r in res["tabla"][:4])
        pasos_mil_vivos = [
            f"**Arranque RK4 ($k = 0 \\dots 3$, $h = {res['h']:g}$):** Se calculan los cuatro puntos iniciales obligatorios: `{arr_pts}`."
        ]
        for rm in res["tabla"][4:10]:
            corr_txt = (
                f"1ra corrección $C = {rm['C']:.6f}$, 2da corrección $C^1 = {rm['C_final']:.6f}$"
                if n_corr >= 2
                else f"Corrector $C = {rm['C_final']:.6f}$"
            )
            pasos_mil_vivos.append(
                f"**Paso $k = {rm['k']}$ ($x_{{{rm['k']}}} = {rm['x_k']:g}$):** Predictor $P = {rm['P']:.6f}$, "
                f"{corr_txt} ($|C - P| = {rm['diff_P_C']:.3e}$, error local est. $\\approx |C-P|/29 = {rm['diff_P_C']/29.0:.3e}$)."
            )
        if len(res["tabla"]) > 10:
            pasos_mil_vivos.append(
                f"**Cierre en $x_f = {res['xf_real']:g}$:** $y(x_f) = {res['y_final']:.6f}$ (error abs.: `{err_mil}`)."
            )
        render_deduccion_paso_a_paso_viva(
            f"Milne Predictor-Corrector (h={res['h']:g})",
            pasos_mil_vivos,
            clave_ia="edo_mil_viva",
            expanded=True,
        )
        render_dataframe_acotado(_construir_filas_milne(res["tabla"]), max_filas=300, key_prefix="tbl_edo_milne")


def _tab_milne() -> None:
    modo_detallado = render_toggle_modo_detallado("milne")
    meta = render_bloque_1_teoria_y_ejemplo("milne", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.4, "y_exacta_str": "2*x*sqrt(x)"},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_milne(defaults, modo_detallado=modo_detallado)


def vista_edo() -> None:
    inject_arandu_css()
    st.title("Ecuaciones Diferenciales Ordinarias (PVI)")
    st.markdown(
        r"Resolución numérica paso a paso sobre malla $\Delta h$ dinámica y contraste directo contra solución analítica."
    )
    activo = render_selector_fichas_metodos(
        "edo",
        (
            {
                "numero": "1",
                "clave_svg": "euler_heun",
                "titulo": "Euler y Heun",
                "subtitulo": "Paso simple · Corrector Heun",
                "resumen": "Euler proyecta una pendiente simple y Heun promedia entrada y salida mejorando el orden.",
                "pista": "Tangente frente a corrector Heun",
            },
            {
                "numero": "2",
                "clave_svg": "runge_kutta",
                "titulo": "Runge-Kutta (RK2 / RK4)",
                "subtitulo": "Autoiniciable · Cuatro pendientes",
                "resumen": "Integra ecuaciones diferenciales desde el primer paso promediando múltiples pendientes ponderadas de referencia.",
                "pista": "Autoiniciable · Orden O(h⁴)",
            },
            {
                "numero": "3",
                "clave_svg": "milne",
                "titulo": "Predictor-Corrector de Milne",
                "subtitulo": "Multipaso · Memoria de 4 puntos",
                "resumen": "Arranca con cuatro pasos iniciales de RK4 y corrige iterativamente cada nuevo punto y_{n+1}.",
                "pista": "Memoria 4 pasos · Orden O(h⁴)",
            },
        ),
    )
    (_tab_euler_heun, _tab_runge_kutta, _tab_milne)[activo]()


vista_edo()

