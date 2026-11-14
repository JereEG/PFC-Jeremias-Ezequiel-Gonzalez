"""
Vista de Ecuaciones Diferenciales Ordinarias (PVI) - Arandú v3.0.
Euler, Heun (Corrector Iterativo), Runge-Kutta (RK2 / RK4) y Milne (Orden 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.edo_pvi import (
    compilar_exacta_edo,
    compilar_funcion_edo,
    metodo_euler,
    metodo_euler_modificado,
    metodo_milne,
    metodo_runge_kutta_2,
    metodo_runge_kutta_4,
)
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    render_banner_edicion_profesora,
    render_bloque_1_teoria_y_ejemplo,
)


def _inputs_pvi(clave_state: str, solo_lectura: bool) -> tuple[str, float, float, float, float, str]:
    c1, c2 = st.columns(2)
    f_str = c1.text_input("EDO y' = f(x, y):", value=str(st.session_state[f"{clave_state}_f_str"]), disabled=solo_lectura, key=f"ui_{clave_state}_f")
    y_ex_str = c2.text_input("Solución exacta y(x) (opcional):", value=str(st.session_state[f"{clave_state}_y_exacta_str"]), disabled=solo_lectura, key=f"ui_{clave_state}_yex")

    c3, c4, c5, c6 = st.columns(4)
    x0 = c3.number_input("x₀:", value=float(st.session_state[f"{clave_state}_x0"]), format="%.4f", disabled=solo_lectura, key=f"ui_{clave_state}_x0")
    y0 = c4.number_input("y₀ = y(x₀):", value=float(st.session_state[f"{clave_state}_y0"]), format="%.4f", disabled=solo_lectura, key=f"ui_{clave_state}_y0")
    xf = c5.number_input("x_final:", value=float(st.session_state[f"{clave_state}_xf"]), format="%.4f", disabled=solo_lectura, key=f"ui_{clave_state}_xf")
    h = c6.number_input("Paso Δh:", value=float(st.session_state[f"{clave_state}_h"]), min_value=0.005, max_value=2.0, step=0.05, format="%.4f", disabled=solo_lectura, key=f"ui_{clave_state}_h")

    if not solo_lectura:
        st.session_state[f"{clave_state}_f_str"] = f_str
        st.session_state[f"{clave_state}_y_exacta_str"] = y_ex_str
        st.session_state[f"{clave_state}_x0"] = x0
        st.session_state[f"{clave_state}_y0"] = y0
        st.session_state[f"{clave_state}_xf"] = xf
        st.session_state[f"{clave_state}_h"] = h

    return f_str, x0, y0, xf, h, y_ex_str


def _compilar_pvi(f_str: str, y_ex_str: str):
    comp_f = compilar_funcion_edo(f_str)
    y_ex_fn = None
    if y_ex_str.strip():
        try:
            y_ex_fn = compilar_exacta_edo(y_ex_str)["fn"]
        except Exception as exc:
            st.warning(f"Solución exacta ignorada: {exc}")
    return comp_f, y_ex_fn


def _plot_edo_paso_a_paso(
    titulo: str,
    x0: float,
    xf_real: float,
    series: list[tuple[str, list[float], list[float], str]],
    y_ex_fn,
    k_vis: int,
) -> None:
    fig = crear_figura_base_plotly(titulo, "x", "y(x)")
    if y_ex_fn is not None:
        xg = np.linspace(x0, xf_real, 180)
        try:
            yg = [y_ex_fn(float(v)) for v in xg]
            fig.add_trace(go.Scatter(x=xg.tolist(), y=yg, mode="lines", name="Solución Exacta", line=dict(color="#D4A359", width=2.8, dash="dash")))
        except Exception:
            pass

    for nombre, xs, ys, color in series:
        sub_x = xs[: k_vis + 1]
        sub_y = ys[: k_vis + 1]
        fig.add_trace(go.Scatter(x=sub_x, y=sub_y, mode="lines+markers", name=nombre, line=dict(color=color, width=2.4), marker=dict(size=8)))
    st.plotly_chart(fig, use_container_width=True)


def _tab_euler_heun() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("euler_heun")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.4, "y_exacta_str": "2*x*sqrt(x)"},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="edo_eh",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"
    f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_eh", solo_lectura)
    iterar_corr = st.checkbox("Iterar corrector de Heun hasta convergencia (estilo Cátedra)", value=True, key="ui_eh_corr")

    try:
        comp_f, y_ex_fn = _compilar_pvi(f_str, y_ex_str)
        res_eu = metodo_euler(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
        res_he = metodo_euler_modificado(comp_f["fn"], x0, y0, h, xf=xf, iterar_corrector=iterar_corr, y_exacta_fn=y_ex_fn)
    except Exception as exc:
        st.error(str(exc))
        return

    for av in res_he["avisos"]:
        st.info(av)

    st.markdown("### ⚙️ Bloque 3 — Resultados Euler vs Heun en Vivo")
    m1, m2, m3 = st.columns(3)
    m1.metric("Euler y(x_f)", f"{res_eu['y_final']:.6f}")
    m2.metric("Heun y(x_f)", f"{res_he['y_final']:.6f}")
    m3.metric("Pasos (n)", res_he["n_pasos"])

    t1, t2 = st.tabs(["Tabla Euler", "Tabla Heun (Euler Modificado)"])
    t1.dataframe(pd.DataFrame(res_eu["tabla"]), use_container_width=True, hide_index=True)
    t2.dataframe(pd.DataFrame(res_he["tabla"]), use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Trayectoria Numérica Paso a Paso")
    k_vis = st.slider("Ver hasta el paso k:", 1, res_he["n_pasos"], res_he["n_pasos"], key="vis_eh_k")
    _plot_edo_paso_a_paso(
        f"Euler vs Heun hasta paso k = {k_vis}",
        x0,
        res_he["xf_real"],
        [("Euler (O1)", res_eu["x"], res_eu["y"], "#C86D44"), ("Heun (O2)", res_he["x"], res_he["y"], "#7A9A65")],
        y_ex_fn,
        k_vis,
    )


def _tab_runge_kutta() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("runge_kutta")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.3, "y_exacta_str": "2*x*sqrt(x)"},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="edo_rk",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"
    f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_rk", solo_lectura)

    try:
        comp_f, y_ex_fn = _compilar_pvi(f_str, y_ex_str)
        res_rk2 = metodo_runge_kutta_2(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
        res_rk4 = metodo_runge_kutta_4(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn)
    except Exception as exc:
        st.error(str(exc))
        return

    for av in res_rk4["avisos"]:
        st.info(av)

    st.markdown("### ⚙️ Bloque 3 — Pendientes kᵢ de RK2 y RK4 en Vivo")
    m1, m2, m3 = st.columns(3)
    m1.metric("RK2 y(x_f)", f"{res_rk2['y_final']:.6f}")
    m2.metric("RK4 y(x_f)", f"{res_rk4['y_final']:.6f}")
    m3.metric("Pasos (n)", res_rk4["n_pasos"])

    t1, t2 = st.tabs(["Tabla RK4 (k1, k2, k3, k4)", "Tabla RK2 (k1, k2)"])
    t1.dataframe(pd.DataFrame(res_rk4["tabla"]), use_container_width=True, hide_index=True)
    t2.dataframe(pd.DataFrame(res_rk2["tabla"]), use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Curvas RK2 y RK4 Paso a Paso")
    k_vis = st.slider("Ver hasta el paso k:", 1, res_rk4["n_pasos"], res_rk4["n_pasos"], key="vis_rk_k")
    _plot_edo_paso_a_paso(
        f"RK2 vs RK4 hasta paso k = {k_vis}",
        x0,
        res_rk4["xf_real"],
        [("RK2 (O2)", res_rk2["x"], res_rk2["y"], "#C86D44"), ("RK4 (O4)", res_rk4["x"], res_rk4["y"], "#7A9A65")],
        y_ex_fn,
        k_vis,
    )


def _tab_milne() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("milne")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "6*x**2 / y", "x0": 4.0, "y0": 16.0, "h": 0.1, "xf": 4.4, "y_exacta_str": "2*x*sqrt(x)"},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="edo_mil",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"
    f_str, x0, y0, xf, h, y_ex_str = _inputs_pvi("edo_mil", solo_lectura)

    st.markdown("### ⚙️ Bloque 3 — Arranque RK4 (k=1..3) + Predictor-Corrector de Milne")
    try:
        comp_f, y_ex_fn = _compilar_pvi(f_str, y_ex_str)
        res = metodo_milne(comp_f["fn"], x0, y0, h, xf=xf, n_correcciones=2, y_exacta_fn=y_ex_fn)
    except Exception as exc:
        st.warning(str(exc))
        return

    for av in res["avisos"]:
        st.info(av)

    m1, m2, m3 = st.columns(3)
    m1.metric("Milne y(x_f)", f"{res['y_final']:.6f}")
    m2.metric("Pasos totales", res["n_pasos"])
    err_fin = res["tabla"][-1].get("error_abs")
    m3.metric("Error absoluto final", f"{err_fin:.3e}" if err_fin is not None else "-")

    st.dataframe(pd.DataFrame(res["tabla"]), use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Evolución Arranque RK4 vs Pasos Multipaso de Milne")
    k_vis = st.slider("Ver hasta el paso k:", 1, res["n_pasos"], res["n_pasos"], key="vis_mil_k")
    _plot_edo_paso_a_paso(
        f"Milne (Arranque RK4 k≤3, P-C k≥4) hasta paso k = {k_vis}",
        x0,
        res["xf_real"],
        [("Milne P-C (O4)", res["x"], res["y"], "#7A9A65")],
        y_ex_fn,
        k_vis,
    )


def vista_edo() -> None:
    inject_arandu_css()
    st.title("⏱️ Ecuaciones Diferenciales Ordinarias (PVI)")
    st.caption("Resolución numérica con tamaño de paso Δh dinámico y comparación contra solución analítica.")

    tabs = st.tabs(
        [
            "1️⃣ Euler y Heun",
            "2️⃣ Runge-Kutta (RK2 / RK4)",
            "3️⃣ Predictor-Corrector de Milne",
        ]
    )
    with tabs[0]:
        _tab_euler_heun()
    with tabs[1]:
        _tab_runge_kutta()
    with tabs[2]:
        _tab_milne()


vista_edo()

