"""
Vista de Sistemas de Ecuaciones Lineales y Autovalores - Arandú v3.0.
Gauss, Gauss-Jordan, LU (Doolittle), Gauss-Seidel, Potencias (Max/Min) y Faddeev-Leverrier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.sistemas_autovalores import (
    eliminacion_gauss,
    factorizacion_lu,
    gauss_jordan,
    gauss_seidel,
    metodo_faddeev_leverrier,
    metodo_potencias_max,
    metodo_potencias_min,
    verificar_dominancia_diagonal,
)
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    render_banner_edicion_profesora,
    render_bloque_1_teoria_y_ejemplo,
    render_slider_paso_k,
)


def _editor_sistema(clave_state: str, solo_lectura: bool, incluir_x0: bool = False) -> tuple[list[list[float]], list[float], list[float] | None]:
    A = st.session_state[f"{clave_state}_A"]
    b = st.session_state[f"{clave_state}_b"]
    n = len(A)
    cols_A = [f"a_{i+1}" for i in range(n)]
    df_mat = pd.DataFrame(A, columns=cols_A)
    df_mat["b"] = b
    if incluir_x0:
        x0 = st.session_state[f"{clave_state}_x0"]
        df_mat["x0"] = x0

    df_edit = st.data_editor(
        df_mat,
        use_container_width=True,
        hide_index=True,
        disabled=solo_lectura,
        key=f"{clave_state}__editor_df",
    )
    if not solo_lectura:
        A = df_edit[cols_A].astype(float).values.tolist()
        b = df_edit["b"].astype(float).tolist()
        st.session_state[f"{clave_state}_A"] = A
        st.session_state[f"{clave_state}_b"] = b
        if incluir_x0:
            x0 = df_edit["x0"].astype(float).tolist()
            st.session_state[f"{clave_state}_x0"] = x0
            return A, b, x0
    return A, b, (st.session_state[f"{clave_state}_x0"] if incluir_x0 else None)


def _tab_gauss() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("eliminacion_gauss")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
        },
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="sis_gauss",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"
    A, b, _ = _editor_sistema("sis_gauss", solo_lectura)

    st.markdown("### ⚙️ Bloque 3 — Escalonamiento y Sustitución Regresiva")
    try:
        res = eliminacion_gauss(A, b)
    except Exception as exc:
        st.error(str(exc))
        return

    cols = st.columns(len(res["solucion"]) + 1)
    for i, val in enumerate(res["solucion"]):
        cols[i].metric(f"x_{i+1}", f"{val:.6f}")
    cols[-1].metric("‖Ax - b‖₂", f"{res['norma_error']:.3e}")

    pasos = res["pasos_escalonamiento"]
    if pasos:
        st.markdown("### 📐 Bloque 4 — Matriz Aumentada Paso a Paso")
        k_vis = render_slider_paso_k("Ver paso de escalonamiento k:", len(pasos), key="vis_gauss_k")
        paso = pasos[k_vis - 1]
        st.info(paso["descripcion"])
        st.dataframe(pd.DataFrame(paso["matriz"]), use_container_width=True)

    fig = crear_figura_base_plotly("Solución xᵢ y Residuos por Ecuación", "Componente", "Valor", altura=320)
    fig.add_trace(go.Bar(x=[f"x_{i+1}" for i in range(len(res["solucion"]))], y=res["solucion"], name="Solución x", marker_color="#2563eb"))
    st.plotly_chart(fig, use_container_width=True)


def _tab_gauss_jordan() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("gauss_jordan")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
        },
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="sis_gj",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"
    A, b, _ = _editor_sistema("sis_gj", solo_lectura)
    modo_gj = st.radio("Objetivo de Gauss-Jordan:", ["Resolver Ax = b", "Calcular Inversa A⁻¹"], horizontal=True, key="ui_gj_modo")

    st.markdown("### ⚙️ Bloque 3 — Diagonalización Completa en Vivo")
    try:
        res = gauss_jordan(A, b if modo_gj.startswith("Resolver") else None)
    except Exception as exc:
        st.error(str(exc))
        return

    if res["modo"] == "sistema":
        cols = st.columns(len(res["solucion"]) + 1)
        for i, val in enumerate(res["solucion"]):
            cols[i].metric(f"x_{i+1}", f"{val:.6f}")
        cols[-1].metric("‖Ax - b‖₂", f"{res['norma_error']:.3e}")
    else:
        st.metric("‖A·A⁻¹ - I‖₂", f"{res['norma_error']:.3e}")
        st.dataframe(pd.DataFrame(res["inversa"]), use_container_width=True)

    pasos = res["pasos"]
    if pasos:
        st.markdown("### 📐 Bloque 4 — Transformación a Identidad Paso a Paso")
        k_vis = render_slider_paso_k("Ver paso de pivoteo k:", len(pasos), key="vis_gj_k")
        paso = pasos[k_vis - 1]
        st.info(paso["descripcion"])
        st.dataframe(pd.DataFrame(paso["matriz"]), use_container_width=True)


def _tab_lu() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("factorizacion_lu")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"A": [[4.0, -2.0, 1.0], [20.0, -7.0, 12.0], [-8.0, 13.0, 17.0]], "b": [11.0, 70.0, 17.0]},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="sis_lu",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"
    A, b, _ = _editor_sistema("sis_lu", solo_lectura)

    st.markdown("### ⚙️ Bloque 3 — Factores L, U y Solución (Ly=b, Ux=y)")
    try:
        res = factorizacion_lu(A, b)
    except Exception as exc:
        st.error(str(exc))
        return

    c1, c2 = st.columns(2)
    c1.markdown("**Matriz Triangular Inferior L:**")
    c1.dataframe(pd.DataFrame(res["L"]), use_container_width=True)
    c2.markdown("**Matriz Triangular Superior U:**")
    c2.dataframe(pd.DataFrame(res["U"]), use_container_width=True)

    cols = st.columns(len(res["solucion"]) + 1)
    for i, val in enumerate(res["solucion"]):
        cols[i].metric(f"x_{i+1}", f"{val:.6f}")
    cols[-1].metric("Error ‖LU - A‖", f"{res['error_LU']:.2e}")

    pasos = res["pasos_eliminacion"]
    if pasos:
        st.markdown("### 📐 Bloque 4 — Construcción Paso a Paso de L y U")
        k_vis = render_slider_paso_k("Ver columna eliminada k:", len(pasos), key="vis_lu_k")
        paso = pasos[k_vis - 1]
        c_l, c_u = st.columns(2)
        c_l.caption(f"L tras paso {k_vis}")
        c_l.dataframe(pd.DataFrame(paso["L_actual"]), use_container_width=True)
        c_u.caption(f"U tras paso {k_vis}")
        c_u.dataframe(pd.DataFrame(paso["U_actual"]), use_container_width=True)


def _tab_gauss_seidel() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("gauss_seidel")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
            "x0": [1.0, 1.0, 1.0],
            "tol": 1e-4,
            "max_iter": 50,
        },
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="sis_gs",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"
    A, b, x0 = _editor_sistema("sis_gs", solo_lectura, incluir_x0=True)

    c1, c2 = st.columns(2)
    tol_val = c1.number_input("Tolerancia E:", value=float(st.session_state["sis_gs_tol"]), min_value=1e-12, format="%.6f", disabled=solo_lectura, key="ui_gs_tol")
    max_it = c2.slider("Iteraciones máx:", 5, 100, int(st.session_state["sis_gs_max_iter"]), disabled=solo_lectura, key="ui_gs_max")
    if not solo_lectura:
        st.session_state["sis_gs_tol"] = tol_val
        st.session_state["sis_gs_max_iter"] = max_it

    dom = verificar_dominancia_diagonal(A)
    if dom["es_dominante"]:
        st.success(f"✅ {dom['diagnostico']}")
    else:
        st.warning(f"⚠️ {dom['diagnostico']}")

    st.markdown("### ⚙️ Bloque 3 — Iteraciones de Gauss-Seidel en Vivo")
    try:
        res = gauss_seidel(A, b, x0=x0, tol=tol_val, max_iter=max_it)
    except Exception as exc:
        st.error(str(exc))
        return

    cols = st.columns(len(res["solucion"]) + 2)
    for i, val in enumerate(res["solucion"]):
        cols[i].metric(f"x_{i+1}", f"{val:.6f}")
    cols[-2].metric("Iteraciones", res["iteraciones"])
    cols[-1].metric("Error final", f"{res['error_final']:.3e}")

    hist = res["historial"]
    df_hist = pd.DataFrame(
        [{"iteracion": h["iteracion"], **h["x_dict"], "error_max": h["error_max"], "residuo": h["norma_residuo"]} for h in hist]
    )
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    if len(df_hist) > 0:
        st.markdown("### 📐 Bloque 4 — Decaimiento Logarítmico del Error Paso a Paso")
        k_vis = render_slider_paso_k("Ver hasta la iteración k:", len(df_hist), key="vis_gs_k")
        df_vis = df_hist.iloc[:k_vis]
        fig = crear_figura_base_plotly(f"Convergencia de Gauss-Seidel hasta k = {k_vis}", "Iteración", "Error máximo (escala log)")
        fig.add_trace(go.Scatter(x=df_vis["iteracion"], y=df_vis["error_max"], mode="lines+markers", name="max |Δx|", line=dict(color="#2563eb", width=2.5)))
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)


def _tab_potencias() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("potencias")
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    defaults = {
        "A": raw_def.get("A", [[3.556, -1.778, 0.0], [-1.778, 3.556, -1.778], [0.0, -1.778, 3.556]]),
        "x0": raw_def.get("x0", [1.0, 1.0, 1.0]),
        "tol_pct": float(raw_def.get("tol", raw_def.get("tol_pct", 0.01))),
        "max_iter": int(raw_def.get("max_iter", 50)),
    }
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="sis_pot",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"

    A = st.session_state["sis_pot_A"]
    x0 = st.session_state["sis_pot_x0"]
    cols_A = [f"a_{i+1}" for i in range(len(A))]
    df_in = pd.DataFrame(A, columns=cols_A)
    df_in["x0"] = x0
    df_edit = st.data_editor(df_in, use_container_width=True, hide_index=True, disabled=solo_lectura, key="sis_pot__editor_df")
    if not solo_lectura:
        A = df_edit[cols_A].astype(float).values.tolist()
        x0 = df_edit["x0"].astype(float).tolist()
        st.session_state["sis_pot_A"] = A
        st.session_state["sis_pot_x0"] = x0

    c1, c2 = st.columns(2)
    tol_pct = c1.number_input("Tolerancia ε (%):", value=float(st.session_state["sis_pot_tol_pct"]), min_value=1e-6, format="%.5f", disabled=solo_lectura, key="ui_pot_tol")
    tipo = c2.radio("Autovalor buscado:", ["Mayor |λ_max| (Potencias Directo)", "Menor |λ_min| (Potencia Inversa)"], horizontal=True, key="ui_pot_tipo")
    if not solo_lectura:
        st.session_state["sis_pot_tol_pct"] = tol_pct

    st.markdown("### ⚙️ Bloque 3 — Iteración de Autovalores en Vivo")
    try:
        if tipo.startswith("Mayor"):
            res = metodo_potencias_max(A, x0=x0, tol=tol_pct, max_iter=int(st.session_state["sis_pot_max_iter"]))
            lam_final = res["autovalor_max"]
        else:
            res = metodo_potencias_min(A, x0=x0, tol=tol_pct, max_iter=int(st.session_state["sis_pot_max_iter"]))
            lam_final = res["autovalor_min"]
    except Exception as exc:
        st.error(str(exc))
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Autovalor estimado λ", f"{lam_final:.6f}")
    m2.metric("Iteraciones", res["iteraciones"])
    m3.metric("Error % final", f"{res['error_final_pct']:.4f}%")

    df_h = pd.DataFrame(res["historial"])
    st.dataframe(df_h, use_container_width=True, hide_index=True)

    if len(df_h) > 0:
        st.markdown("### 📐 Bloque 4 — Estabilización de λ_k Paso a Paso")
        k_vis = render_slider_paso_k("Ver hasta la iteración k:", len(df_h), key="vis_pot_k")
        df_vis = df_h.iloc[:k_vis]
        col_lam = "lambda" if "lambda" in df_vis.columns else "lambda_min"
        fig = crear_figura_base_plotly(f"Convergencia de λ_k hasta k = {k_vis}", "Iteración", "λ estimado")
        fig.add_trace(go.Scatter(x=df_vis["iteracion"], y=df_vis[col_lam], mode="lines+markers", name="λ_k", line=dict(color="#10b981", width=2.5)))
        st.plotly_chart(fig, use_container_width=True)


def _tab_faddeev() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("faddeev_leverrier")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"A": [[3.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 3.0]]},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="sis_fad",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    solo_lectura = modo == "catedra"

    A = st.session_state["sis_fad_A"]
    cols_A = [f"c_{i+1}" for i in range(len(A))]
    df_edit = st.data_editor(pd.DataFrame(A, columns=cols_A), use_container_width=True, hide_index=True, disabled=solo_lectura, key="sis_fad__editor_df")
    if not solo_lectura:
        A = df_edit[cols_A].astype(float).values.tolist()
        st.session_state["sis_fad_A"] = A

    st.markdown("### ⚙️ Bloque 3 — Polinomio Característico, Autovalores e Inversa")
    try:
        res = metodo_faddeev_leverrier(A)
    except Exception as exc:
        st.error(str(exc))
        return

    st.info(f"**Polinomio característico:** `{res['polinomio_texto']}`")
    m1, m2, m3 = st.columns(3)
    m1.metric("det(A)", f"{res['determinante']:.6f}")
    m2.metric("Autovalores", ", ".join(f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in res["autovalores"]))
    m3.metric("Error A⁻¹", f"{res['error_inversa']:.2e}" if res["error_inversa"] is not None else "Singular")

    if res["inversa"] is not None:
        with st.expander("🔁 Ver Matriz Inversa A⁻¹ obtenida por Faddeev"):
            st.dataframe(pd.DataFrame(res["inversa"]), use_container_width=True)

    st.markdown("### 📐 Bloque 4 — Raíces del Polinomio Característico P(λ) = 0")
    curva = res["curva_polinomio"]
    fig = crear_figura_base_plotly("Polinomio Característico P(λ)", "λ", "P(λ)")
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig.add_trace(go.Scatter(x=curva["lambda"], y=curva["p_lambda"], mode="lines", name="P(λ)", line=dict(color="#2563eb", width=2.5)))
    reales = res["autovalores_reales"]
    if reales:
        k_vis = render_slider_paso_k("Destacar hasta el autovalor real k:", len(reales), key="vis_fad_k")
        fig.add_trace(go.Scatter(x=reales[:k_vis], y=[0.0] * k_vis, mode="markers", name="Autovalores reales", marker=dict(size=12, color="#ef4444", symbol="star")))
    st.plotly_chart(fig, use_container_width=True)


def vista_sistemas() -> None:
    inject_arandu_css()
    st.title("📐 Sistemas Lineales y Autovalores")
    st.caption("Métodos directos, iterativos con dominancia diagonal y espectro característico.")

    tabs = st.tabs(
        [
            "1️⃣ Eliminación de Gauss",
            "2️⃣ Gauss-Jordan",
            "3️⃣ Factorización LU",
            "4️⃣ Gauss-Seidel",
            "5️⃣ Método de Potencias",
            "6️⃣ Faddeev-Leverrier",
        ]
    )
    with tabs[0]:
        _tab_gauss()
    with tabs[1]:
        _tab_gauss_jordan()
    with tabs[2]:
        _tab_lu()
    with tabs[3]:
        _tab_gauss_seidel()
    with tabs[4]:
        _tab_potencias()
    with tabs[5]:
        _tab_faddeev()


vista_sistemas()
