"""
Vista de Raíces de Ecuaciones No Lineales - Arandú v3.0.
Bisección (+ Descartes/Tanteo), Interpolación Lineal (Regula Falsi),
Newton-Raphson (Fourier + tangentes paso a paso) y Punto Fijo + Aitken Δ².
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
from views import (
    crear_figura_base_plotly,
    inject_arandu_css,
    render_banner_edicion_profesora,
    render_bloque_1_teoria_y_ejemplo,
    render_slider_paso_k,
)


def _muestrear_curva(fn, x_min: float, x_max: float, puntos: int = 240) -> tuple[list[float], list[float | None]]:
    xs = np.linspace(x_min, x_max, puntos)
    ys: list[float | None] = []
    for xv in xs:
        try:
            val = float(fn(float(xv)))
            ys.append(val if np.isfinite(val) and abs(val) < 1e6 else None)
        except Exception:
            ys.append(None)
    return xs.tolist(), ys


def _tab_biseccion() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("biseccion")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "-2.75*x**3 + 18*x**2 - 21*x - 12", "a": 2.0, "b": 2.5, "tol": 0.01, "max_iter": 50},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="raiz_bis",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
        mensaje_edicion="✏️ Ecuación e Intervalo Interactivos: Modifica f(x), extremos [a, b] o tolerancia en vivo.",
    )
    solo_lectura = modo == "catedra"

    f_str = st.text_input("Ecuación f(x) = 0:", value=str(st.session_state["raiz_bis_f_str"]), disabled=solo_lectura, key="ui_bis_f")
    c1, c2, c3, c4 = st.columns(4)
    a_val = c1.number_input("Extremo a:", value=float(st.session_state["raiz_bis_a"]), format="%.4f", disabled=solo_lectura, key="ui_bis_a")
    b_val = c2.number_input("Extremo b:", value=float(st.session_state["raiz_bis_b"]), format="%.4f", disabled=solo_lectura, key="ui_bis_b")
    tol_val = c3.number_input("Tolerancia E:", value=float(st.session_state["raiz_bis_tol"]), min_value=1e-10, format="%.6f", disabled=solo_lectura, key="ui_bis_tol")
    max_it = c4.number_input("Iteraciones máx:", value=int(st.session_state["raiz_bis_max_iter"]), min_value=1, max_value=200, disabled=solo_lectura, key="ui_bis_max")

    if not solo_lectura:
        st.session_state["raiz_bis_f_str"] = f_str
        st.session_state["raiz_bis_a"] = a_val
        st.session_state["raiz_bis_b"] = b_val
        st.session_state["raiz_bis_tol"] = tol_val
        st.session_state["raiz_bis_max_iter"] = int(max_it)

    try:
        comp = compilar_funcion_raices(f_str)
    except Exception as exc:
        st.error(str(exc))
        return

    with st.expander("🔍 Análisis Previo: Regla de Descartes & Barrido por Tanteos"):
        desc = analizar_descartes(comp["f_simb"], comp["x_sym"])
        if desc["es_polinomio"]:
            st.write(
                f"Grado: **{desc['grado']}** | Posibles raíces positivas: **{desc['posibles_raices_pos']}** | "
                f"Posibles raíces negativas: **{desc['posibles_raices_neg']}**"
            )
        else:
            st.info(desc["mensaje"])
        bt = barrido_tanteo(comp["fn"], x_ini=min(a_val, b_val) - 2.0, dx=0.5, max_pasos=16)
        st.write(f"Intervalos detectados en barrido Δx=0.5: **{bt['intervalos']}**")

    st.markdown("### ⚙️ Bloque 3 — Ejecución en Vivo (Bisección)")
    try:
        res = metodo_biseccion(comp["fn"], a_val, b_val, tol=tol_val, max_iter=int(max_it))
    except Exception as exc:
        st.error(str(exc))
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raíz aproximada", f"{res['raiz']:.6f}")
    m2.metric("Iteraciones", f"{res['iteraciones']} (teórico: {res['n_teorico_entero']})")
    m3.metric("Error final |Δx|", f"{res['error_final']:.3e}")
    m4.metric("f(raíz)", f"{res['f_raiz']:.3e}")

    df = pd.DataFrame(res["historial"])
    st.dataframe(df[["iteracion", "a", "b", "x_k", "f_a", "f_xk", "f_b", "error", "cumple_tol"]], use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Geometría Paso a Paso (Encierro de Intervalos)")
    k_vis = render_slider_paso_k("Ver hasta la iteración k:", len(df), key="vis_bis_k")
    fila_k = res["historial"][k_vis - 1]
    pad = max(0.5, 0.35 * abs(b_val - a_val))
    xs, ys = _muestrear_curva(comp["fn"], min(a_val, b_val) - pad, max(a_val, b_val) + pad)

    fig = crear_figura_base_plotly(f"Bisección — Iteración {k_vis}: [{fila_k['a']:.4f}, {fila_k['b']:.4f}]", "x", "f(x)")
    fig.add_vrect(x0=fila_k["a"], x1=fila_k["b"], fillcolor="rgba(59,130,246,0.14)", line_width=1, annotation_text=f"Intervalo k={k_vis}")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color="#2563eb", width=2.5)))
    df_sub = df.iloc[:k_vis]
    fig.add_trace(go.Scatter(x=df_sub["x_k"], y=df_sub["f_xk"], mode="markers", name="Puntos medios x_k", marker=dict(size=8, color="#f59e0b")))
    fig.add_trace(go.Scatter(x=[fila_k["x_k"]], y=[fila_k["f_xk"]], mode="markers", name=f"Actual x_{k_vis}", marker=dict(size=12, color="#10b981", symbol="star")))
    st.plotly_chart(fig, use_container_width=True)


def _tab_interpolacion_lineal() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("interpolacion_lineal_raices")
    practica = meta.get("practica_profesora", {})
    defaults = {
        "f_str": "2*x**4 - 6*x**2 + 7*x - 2",
        "a": -2.5,
        "b": -2.0,
        "tol": 0.001,
        "variante": "Cátedra (pivote fijo)",
        **practica.get("datos_iniciales", {}),
    }
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="raiz_rf",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"

    f_str = st.text_input("Ecuación f(x) = 0:", value=str(st.session_state["raiz_rf_f_str"]), disabled=solo_lectura, key="ui_rf_f")
    c1, c2, c3, c4 = st.columns(4)
    a_val = c1.number_input("x0 / a:", value=float(st.session_state["raiz_rf_a"]), format="%.4f", disabled=solo_lectura, key="ui_rf_a")
    b_val = c2.number_input("x1 / b:", value=float(st.session_state["raiz_rf_b"]), format="%.4f", disabled=solo_lectura, key="ui_rf_b")
    tol_val = c3.number_input("Tolerancia E:", value=float(st.session_state["raiz_rf_tol"]), min_value=1e-10, format="%.6f", disabled=solo_lectura, key="ui_rf_tol")
    variante = c4.selectbox(
        "Variante:",
        ["Cátedra (pivote fijo)", "Regula Falsi (Bolzano)"],
        index=0 if st.session_state["raiz_rf_variante"].startswith("Cátedra") else 1,
        key="ui_rf_var",
    )

    if not solo_lectura:
        st.session_state["raiz_rf_f_str"] = f_str
        st.session_state["raiz_rf_a"] = a_val
        st.session_state["raiz_rf_b"] = b_val
        st.session_state["raiz_rf_tol"] = tol_val
        st.session_state["raiz_rf_variante"] = variante

    try:
        comp = compilar_funcion_raices(f_str)
        st.markdown("### ⚙️ Bloque 3 — Cuerdas en Vivo")
        if variante.startswith("Cátedra"):
            res = metodo_interpolacion_lineal_catedra(comp["fn"], x0_fijo=a_val, x1_ini=b_val, tol=tol_val)
        else:
            res = metodo_interpolacion_lineal_bolzano(comp["fn"], a=min(a_val, b_val), b=max(a_val, b_val), tol=tol_val)
    except Exception as exc:
        st.error(str(exc))
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Raíz aproximada", f"{res['raiz']:.6f}")
    m2.metric("Iteraciones", res["iteraciones"])
    m3.metric("Error final", f"{res['error_final']:.3e}")

    df = pd.DataFrame(res["historial"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Secantes / Cuerdas Paso a Paso")
    k_vis = render_slider_paso_k("Ver hasta la iteración k:", len(df), key="vis_rf_k")
    pad = max(0.4, 0.35 * abs(b_val - a_val))
    xs, ys = _muestrear_curva(comp["fn"], min(a_val, b_val) - pad, max(a_val, b_val) + pad)
    fig = crear_figura_base_plotly(f"Interpolación Lineal — Paso {k_vis}", "x", "f(x)")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color="#2563eb", width=2.5)))

    fila_k = res["historial"][k_vis - 1]
    if variante.startswith("Cátedra"):
        p0, f0 = fila_k["x0_fijo"], fila_k["f_x0"]
        p1, f1 = fila_k["x_i"], fila_k["f_xi"]
        xr = fila_k["x_sig"]
    else:
        p0, f0 = fila_k["a"], fila_k["f_a"]
        p1, f1 = fila_k["b"], fila_k["f_b"]
        xr = fila_k["x_k"]
    fig.add_trace(go.Scatter(x=[p0, p1], y=[f0, f1], mode="lines+markers", name=f"Cuerda k={k_vis}", line=dict(color="#f59e0b", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=[xr], y=[0.0], mode="markers", name="Corte eje X", marker=dict(size=12, color="#10b981", symbol="diamond")))
    st.plotly_chart(fig, use_container_width=True)


def _tab_newton_raphson() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("newton_raphson")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "x**3 - 2*x - 5", "a": 2.0, "b": 2.5, "x0": 2.5, "tol": 1e-4, "max_iter": 50},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="raiz_nr",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"

    f_str = st.text_input("Ecuación f(x) = 0:", value=str(st.session_state["raiz_nr_f_str"]), disabled=solo_lectura, key="ui_nr_f")
    c1, c2, c3, c4 = st.columns(4)
    a_val = c1.number_input("Fourier extremo a:", value=float(st.session_state["raiz_nr_a"]), format="%.4f", disabled=solo_lectura, key="ui_nr_a")
    b_val = c2.number_input("Fourier extremo b:", value=float(st.session_state["raiz_nr_b"]), format="%.4f", disabled=solo_lectura, key="ui_nr_b")
    x0_val = c3.number_input("Semilla x0:", value=float(st.session_state["raiz_nr_x0"]), format="%.4f", disabled=solo_lectura, key="ui_nr_x0")
    tol_val = c4.number_input("Tolerancia E:", value=float(st.session_state["raiz_nr_tol"]), min_value=1e-12, format="%.6f", disabled=solo_lectura, key="ui_nr_tol")

    if not solo_lectura:
        st.session_state["raiz_nr_f_str"] = f_str
        st.session_state["raiz_nr_a"] = a_val
        st.session_state["raiz_nr_b"] = b_val
        st.session_state["raiz_nr_x0"] = x0_val
        st.session_state["raiz_nr_tol"] = tol_val

    try:
        comp = compilar_funcion_raices(f_str)
    except Exception as exc:
        st.error(str(exc))
        return

    st.latex(rf"f'(x) = {comp['d1_latex']}, \qquad f''(x) = {comp['d2_latex']}")
    fourier = verificar_condiciones_fourier(comp["fn"], comp["d1"], comp["d2"], min(a_val, b_val), max(a_val, b_val))
    st.info(
        f"**Condiciones de Fourier en [{min(a_val, b_val):g}, {max(a_val, b_val):g}]:** "
        f"{'✅ Sugerido x0 = ' + str(fourier['x0_sugerido']) if fourier['fourier_cumple'] else '⚠️ Sin sugerencia estricta en extremos'} "
        f"({fourier['detalle_fourier']})"
    )

    st.markdown("### ⚙️ Bloque 3 — Iteraciones de Newton-Raphson en Vivo")
    res = metodo_newton_raphson(comp["fn"], comp["d1"], x0_val, tol=tol_val, max_iter=int(st.session_state["raiz_nr_max_iter"]))
    if not res["convergente"] and res.get("mensaje"):
        st.warning(res["mensaje"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Raíz aproximada", f"{res['raiz']:.7f}")
    m2.metric("Iteraciones", res["iteraciones"])
    m3.metric("Error final", f"{res['error_final']:.3e}")

    df = pd.DataFrame(res["historial"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Rectas Tangentes Paso a Paso")
    k_vis = render_slider_paso_k("Ver hasta la iteración k:", max(1, len(df)), key="vis_nr_k")
    puntos_x = [row["x_k"] for row in res["historial"][:k_vis]] + [row["x_sig"] for row in res["historial"][:k_vis] if row["x_sig"] is not None]
    xmin, xmax = min(puntos_x) - 0.5, max(puntos_x) + 0.5
    xs, ys = _muestrear_curva(comp["fn"], xmin, xmax)

    fig = crear_figura_base_plotly(f"Newton-Raphson — Tangentes hasta k = {k_vis}", "x", "f(x)")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color="#2563eb", width=2.5)))
    for idx, row in enumerate(res["historial"][:k_vis], start=1):
        if row["x_sig"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=[row["x_k"], row["x_k"], row["x_sig"]],
                    y=[0.0, row["f_xk"], 0.0],
                    mode="lines+markers",
                    name=f"Tangente k={idx}",
                    opacity=1.0 if idx == k_vis else 0.45,
                )
            )
    st.plotly_chart(fig, use_container_width=True)


def _tab_punto_fijo_aitken() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("punto_fijo_aitken")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"f_str": "exp(x) - x**2 - 5", "g_str": "log(x**2 + 5)", "x0": 2.5, "tol": 1e-4, "max_iter": 50},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="raiz_pf",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"

    c1, c2 = st.columns(2)
    f_str = c1.text_input("Ecuación f(x) = 0:", value=str(st.session_state["raiz_pf_f_str"]), disabled=solo_lectura, key="ui_pf_f")
    g_str = c2.text_input("Despeje x = g(x):", value=str(st.session_state["raiz_pf_g_str"]), disabled=solo_lectura, key="ui_pf_g")
    c3, c4 = st.columns(2)
    x0_val = c3.number_input("Semilla x0:", value=float(st.session_state["raiz_pf_x0"]), format="%.4f", disabled=solo_lectura, key="ui_pf_x0")
    tol_val = c4.number_input("Tolerancia E:", value=float(st.session_state["raiz_pf_tol"]), min_value=1e-12, format="%.6f", disabled=solo_lectura, key="ui_pf_tol")

    if not solo_lectura:
        st.session_state["raiz_pf_f_str"] = f_str
        st.session_state["raiz_pf_g_str"] = g_str
        st.session_state["raiz_pf_x0"] = x0_val
        st.session_state["raiz_pf_tol"] = tol_val

    try:
        comp_g = compilar_g_punto_fijo(g_str)
        deriv_x0 = abs(comp_g["dgn"](x0_val))
    except Exception as exc:
        st.error(str(exc))
        return

    st.latex(rf"g'(x) = {comp_g['dg_latex']}, \qquad |g'(x_0)| = {deriv_x0:.4f}")
    if deriv_x0 < 1.0:
        st.success("✅ |g'(x₀)| < 1: El esquema es contractivo en el entorno de x₀.")
    else:
        st.warning("⚠️ |g'(x₀)| ≥ 1: El Punto Fijo simple puede diverger.")

    st.markdown("### ⚙️ Bloque 3 — Punto Fijo vs Aceleración Δ² de Aitken")
    res = metodo_punto_fijo_aitken(comp_g["gn"], x0_val, tol=tol_val, max_iter=int(st.session_state["raiz_pf_max_iter"]))
    pf, ait = res["punto_fijo"], res["aitken"]
    m1, m2 = st.columns(2)
    m1.metric("Punto Fijo simple", f"{pf['raiz']:.6f}", f"{pf['iteraciones']} iters | Err {pf['error_final']:.2e}")
    m2.metric("Aceleración Δ² de Aitken", f"{ait['raiz']:.6f}", f"{ait['ciclos']} ciclos | Err {ait['error_final']:.2e}")

    t_pf, t_ait = st.tabs(["Tabla Punto Fijo", "Tabla Aitken Δ²"])
    with t_pf:
        st.dataframe(pd.DataFrame(pf["historial"]), use_container_width=True, hide_index=True)
    with t_ait:
        st.dataframe(pd.DataFrame(ait["historial"]), use_container_width=True, hide_index=True)

    if pf["historial"]:
        st.markdown("### 📐 Bloque 4 — Diagrama Telaraña (Cobweb) Paso a Paso")
        k_vis = render_slider_paso_k("Ver hasta iteración de Punto Fijo k:", len(pf["historial"]), key="vis_pf_k", default_val=min(len(pf["historial"]), 8))
        sub = pf["historial"][:k_vis]
        pts = [r["x_ant"] for r in sub] + [r["x_sig"] for r in sub]
        xmin, xmax = min(pts) - 0.3, max(pts) + 0.3
        xs, ys = _muestrear_curva(comp_g["gn"], xmin, xmax)

        fig = crear_figura_base_plotly(f"Punto Fijo (y = g(x) vs y = x) hasta k = {k_vis}", "x", "y")
        fig.add_trace(go.Scatter(x=[xmin, xmax], y=[xmin, xmax], mode="lines", name="y = x", line=dict(color="#64748b", dash="dash")))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="y = g(x)", line=dict(color="#2563eb", width=2.5)))

        cob_x, cob_y = [sub[0]["x_ant"]], [sub[0]["x_ant"]]
        for r in sub:
            cob_x.extend([r["x_ant"], r["x_sig"]])
            cob_y.extend([r["x_sig"], r["x_sig"]])
        fig.add_trace(go.Scatter(x=cob_x, y=cob_y, mode="lines+markers", name="Trayectoria Telaraña", line=dict(color="#f59e0b", width=2)))
        st.plotly_chart(fig, use_container_width=True)


def vista_raices() -> None:
    inject_arandu_css()
    st.title("📉 Raíces de Ecuaciones No Lineales")
    st.caption("Localización, encierro garantizado (Bolzano) y métodos abiertos de orden lineal a cuadrático.")

    tabs = st.tabs(
        [
            "1️⃣ Bisección (y Descartes/Tanteo)",
            "2️⃣ Interpolación Lineal (Regula Falsi)",
            "3️⃣ Newton-Raphson (Fourier)",
            "4️⃣ Punto Fijo & Aitken Δ²",
        ]
    )
    with tabs[0]:
        _tab_biseccion()
    with tabs[1]:
        _tab_interpolacion_lineal()
    with tabs[2]:
        _tab_newton_raphson()
    with tabs[3]:
        _tab_punto_fijo_aitken()


vista_raices()
