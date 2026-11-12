"""
views/view_lab4.py
==================
Vista interactiva para el Laboratorio 4: Sistemas de Ecuaciones Lineales y Autovalores.
- Eliminación de Gauss con pivoteo parcial y sustitución regresiva.
- Método de Gauss-Jordan (Resolución de sistemas e inversión matricial [A|I] -> [I|A^-1]).
- Factorización LU (Doolittle) con sustitución progresiva y regresiva.
- Método Iterativo de Gauss-Seidel con análisis previo de dominancia diagonal.
- Método de las Potencias (Autovalor máximo y mínimo con potencias inversas).
- Método de Faddeev-Leverrier (Polinomio característico, determinante, inversa y espectro).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.lab4_sistemas import (
    eliminacion_gauss,
    factorizacion_lu,
    gauss_jordan,
    gauss_seidel,
    metodo_faddeev_leverrier,
    metodo_potencias_max,
    metodo_potencias_min,
    verificar_dominancia_diagonal,
)


def vista_lab4():
    st.title("🏛️ Laboratorio 4: Sistemas de Ecuaciones Lineales y Espectro Matricial")
    st.markdown(
        r"""
        Este módulo aborda la resolución de sistemas lineales $A x = b$ mediante métodos directos
        (Gauss, Gauss-Jordan, Doolittle LU) e iterativos (Gauss-Seidel), así como el cálculo de
        autovalores y autovectores por Métodos de Potencias y Faddeev-Leverrier.
        """
    )

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "1. Eliminación de Gauss",
        "2. Gauss-Jordan e Inversa",
        "3. Factorización LU",
        "4. Gauss-Seidel",
        "5. Método de Potencias",
        "6. Faddeev-Leverrier",
    ])

    # Sistema por defecto de la guía de cátedra UNNE
    A_default = np.array([
        [18.72, 3.25, 1.25],
        [2.15, 15.68, 4.23],
        [1.05, 2.18, 14.52],
    ], dtype=float)
    b_default = np.array([25.43, 18.92, 12.35], dtype=float)

    # ==========================================================================
    # PESTAÑA 1: ELIMINACIÓN DE GAUSS
    # ==========================================================================
    with t1:
        st.subheader("1. Método de Eliminación de Gauss")
        st.latex(r"m_{ik} = \frac{a_{ik}^{(k)}}{a_{kk}^{(k)}}, \quad F_i \leftarrow F_i - m_{ik} F_k")

        st.markdown("#### Configuración del Sistema Ax = b (3x3)")
        cg1, cg2 = st.columns([3, 1])
        with cg1:
            df_A = pd.DataFrame(A_default, columns=["x₁", "x₂", "x₃"])
            edited_A = st.data_editor(df_A, key="v4_gauss_A")
            A_mat = edited_A.values
        with cg2:
            df_b = pd.DataFrame(b_default, columns=["b"])
            edited_b = st.data_editor(df_b, key="v4_gauss_b")
            b_vec = edited_b.values.flatten()

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            use_pivoteo = st.checkbox("Habilitar pivoteo parcial (por columnas)", value=True, key="v4_gauss_piv")
        with c_opt2:
            decs_g = st.slider("Decimales de redondeo:", 2, 8, 4, key="v4_gauss_dec")

        if st.button("Resolver por Eliminación de Gauss", key="btn_gauss"):
            try:
                res_g = eliminacion_gauss(A_mat, b_vec, pivoteo=use_pivoteo, decimales=decs_g)
                st.success(f"Sistema resuelto exitosamente en {res_g['duracion']*1000:.2f} ms")

                st.markdown("#### Vector Solución Encontrado")
                col_x = st.columns(len(res_g["solucion"]))
                for i, val in enumerate(res_g["solucion"]):
                    col_x[i].metric(f"x_{i+1}", f"{val:.{decs_g}f}")

                st.markdown("#### Pasos de Triangularización Superior")
                for paso in res_g["pasos_escalonamiento"]:
                    st.write(f"**Paso {paso['paso']}:** {paso['descripcion']}")
                    if paso["operaciones"]:
                        for op in paso["operaciones"]:
                            st.code(op, language="text")
                    st.dataframe(pd.DataFrame(paso["matriz"]), use_container_width=True)

                st.markdown("#### Sustitución Regresiva Detallada")
                st.dataframe(pd.DataFrame(res_g["pasos_sustitucion"]), use_container_width=True)

                st.markdown("#### Verificación de Residuales")
                r_norm = res_g["norma_error"]
                st.write(f"Norma del error $\\|Ax - b\\| = {r_norm:.2e}$")
            except Exception as e:
                st.error(f"Error en Eliminación de Gauss: {e}")

    # ==========================================================================
    # PESTAÑA 2: GAUSS-JORDAN
    # ==========================================================================
    with t2:
        st.subheader("2. Método de Gauss-Jordan (Resolución e Inversión Matricial)")
        st.latex(r"[A \mid I] \longrightarrow [I \mid A^{-1}]")

        modo_gj = st.radio("Objetivo:", ["Resolver Sistema Ax = b", "Calcular Matriz Inversa A⁻¹"], horizontal=True, key="v4_gj_modo")

        try:
            if modo_gj == "Resolver Sistema Ax = b":
                res_gj = gauss_jordan(A_mat, b_vec, pivoteo=True, decimales=4)
                st.success(f"Gauss-Jordan completado en {res_gj['duracion']*1000:.2f} ms")
                st.markdown("#### Solución Final")
                col_xj = st.columns(len(res_gj["solucion"]))
                for i, val in enumerate(res_gj["solucion"]):
                    col_xj[i].metric(f"x_{i+1}", f"{val:.4f}")

                st.markdown("#### Evolución de la Matriz Reducida por Filas")
                for paso in res_gj["pasos"]:
                    st.write(f"**Paso {paso['paso']}:** {paso['descripcion']}")
                    if paso["operaciones"]:
                        for op in paso["operaciones"]:
                            st.code(op, language="text")
                    st.dataframe(pd.DataFrame(paso["matriz"]), use_container_width=True)
            else:
                res_inv = gauss_jordan(A_mat, b_vector=None, pivoteo=True, decimales=4)
                st.success("Matriz Inversa calculada exitosamente.")
                st.markdown("#### Matriz Inversa Calculada A⁻¹")
                st.dataframe(pd.DataFrame(res_inv["inversa"]), use_container_width=True)
                st.markdown("#### Comprobación de Identidad A · A⁻¹")
                prod = np.dot(A_mat, res_inv["inversa"])
                st.dataframe(pd.DataFrame(np.round(prod, 4)), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Gauss-Jordan: {e}")

    # ==========================================================================
    # PESTAÑA 3: FACTORIZACIÓN LU (DOOLITTLE)
    # ==========================================================================
    with t3:
        st.subheader("3. Factorización LU (Algoritmo de Doolittle)")
        st.latex(r"A = L \cdot U, \quad L y = b \implies U x = y")
        st.markdown("Donde $L$ es triangular inferior con unos en su diagonal ($L_{ii} = 1$) y $U$ es triangular superior.")

        try:
            res_lu = factorizacion_lu(A_mat, b_vec, decimales=4)
            c_l, c_u = st.columns(2)
            with c_l:
                st.markdown("#### Matriz L (Triangular Inferior)")
                st.dataframe(pd.DataFrame(res_lu["L"]), use_container_width=True)
            with c_u:
                st.markdown("#### Matriz U (Triangular Superior)")
                st.dataframe(pd.DataFrame(res_lu["U"]), use_container_width=True)

            st.markdown("#### Sustitución Progresiva: Ly = b")
            st.write(f"Vector intermedio **y:** `{np.round(res_lu['y'], 4).tolist()}`")

            st.markdown("#### Sustitución Regresiva: Ux = y")
            st.write(f"Vector solución **x:** `{np.round(res_lu['x'], 4).tolist()}`")
            st.write(f"Error de descomposición $\\|LU - A\\| = {res_lu['error_LU']:.2e}$")
            st.write(f"Error del sistema $\\|Ax - b\\| = {res_lu['error_Ax']:.2e}$")
        except Exception as e:
            st.error(f"Error en Factorización LU: {e}")

    # ==========================================================================
    # PESTAÑA 4: GAUSS-SEIDEL
    # ==========================================================================
    with t4:
        st.subheader("4. Método Iterativo de Gauss-Seidel")
        st.latex(
            r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)"
        )

        st.markdown("#### Verificación Previa: Dominancia Diagonal Estricta")
        dom_info = verificar_dominancia_diagonal(A_mat)
        if dom_info["es_dominante"]:
            st.success("✅ La matriz es ESTRICTAMENTE DOMINANTE diagonalmente. El método converge para cualquier semilla inicial.")
        else:
            st.warning("⚠️ La matriz NO cumple dominancia diagonal estricta en todas sus filas. La convergencia no está garantizada teóricamente.")

        st.dataframe(pd.DataFrame(dom_info["filas"]), use_container_width=True)

        col_gs1, col_gs2 = st.columns(2)
        with col_gs1:
            tol_gs = st.number_input("Tolerancia:", value=0.01, format="%.4f", key="v4_gs_tol")
        with col_gs2:
            max_gs = st.number_input("Iteraciones máximas:", value=50, step=5, key="v4_gs_max")

        try:
            res_gs = gauss_seidel(A_mat, b_vec, tol=tol_gs, max_iter=int(max_gs))
            if res_gs["convergente"]:
                st.success(f"Convergencia lograda en **{res_gs['iteraciones']} iteraciones**.")
            else:
                st.warning(f"No convergió tras {res_gs['iteraciones']} iteraciones.")

            st.markdown("#### Historial de Iteraciones")
            df_gs = pd.DataFrame(res_gs["historial"])
            st.dataframe(df_gs, use_container_width=True)

            if len(df_gs) > 1 and "error_max" in df_gs.columns:
                fig_gs = go.Figure()
                fig_gs.add_trace(go.Scatter(
                    x=df_gs["iteracion"],
                    y=df_gs["error_max"],
                    mode="lines+markers",
                    line=dict(color="#636efa", width=2),
                    name="Error Máximo",
                ))
                fig_gs.update_layout(title="Decaimiento del Error en Gauss-Seidel", xaxis_title="Iteración", yaxis_title="Error Máximo", yaxis_type="log")
                st.plotly_chart(fig_gs, use_container_width=True)
        except Exception as e:
            st.error(f"Error en Gauss-Seidel: {e}")

    # ==========================================================================
    # PESTAÑA 5: MÉTODO DE LAS POTENCIAS
    # ==========================================================================
    with t5:
        st.subheader("5. Método de las Potencias (Autovalor Máximo y Mínimo)")
        st.latex(r"y_{k} = A x_{k-1}, \quad \lambda_k = \max(|y_k|), \quad x_k = \frac{y_k}{\lambda_k}")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### Autovalor Dominante (λ_max)")
            try:
                res_pmax = metodo_potencias_max(A_mat, tol=1e-3, max_iter=50)
                if res_pmax["convergente"]:
                    st.metric("λ_max", f"{res_pmax['autovalor']:.4f}")
                    st.write(f"**Iteraciones:** {res_pmax['iteraciones']}")
                    st.write(f"**Autovector normalizado:** `{np.round(res_pmax['autovector'], 4)}`")
                    st.dataframe(pd.DataFrame(res_pmax["historial"]), use_container_width=True)
                else:
                    st.error("No convergió el método de las potencias directas.")
            except Exception as e:
                st.error(f"Error en Potencias Máx: {e}")

        with col_p2:
            st.markdown("#### Autovalor Menor (λ_min) vía Potencias Inversas")
            st.latex(r"A^{-1} x = \frac{1}{\lambda} x")
            try:
                res_pmin = metodo_potencias_min(A_mat, tol=1e-3, max_iter=50)
                if res_pmin["convergente"]:
                    st.metric("λ_min", f"{res_pmin['autovalor']:.4f}")
                    st.write(f"**Iteraciones:** {res_pmin['iteraciones']}")
                    st.write(f"**Autovector normalizado:** `{np.round(res_pmin['autovector'], 4)}`")
                    st.dataframe(pd.DataFrame(res_pmin["historial"]), use_container_width=True)
                else:
                    st.error("No convergió el método de potencias inversas.")
            except Exception as e:
                st.error(f"Error en Potencias Mín: {e}")

    # ==========================================================================
    # PESTAÑA 6: FADDEEV-LEVERRIER
    # ==========================================================================
    with t6:
        st.subheader("6. Método de Faddeev-Leverrier")
        st.latex(
            r"B_k = A \cdot (B_{k-1} - b_{k-1} I), \quad b_k = \frac{1}{k} \operatorname{tr}(B_k)"
        )
        st.markdown(
            """
            Proporciona de forma simultánea el polinomio característico completo, el determinante,
            la matriz inversa analítica y todos los autovalores y autovectores.
            """
        )

        try:
            res_fl = metodo_faddeev_leverrier(A_mat)
            st.markdown("#### Polinomio Característico")
            coefs = res_fl["coeficientes_polinomio"]
            poly_str = " \\lambda^3 "
            if len(coefs) >= 4:
                poly_str += f"{coefs[1]:+.2f}\\lambda^2 {coefs[2]:+.2f}\\lambda {coefs[3]:+.2f}"
            st.latex(f"P(\\lambda) = {poly_str} = 0")

            c_fl1, c_fl2 = st.columns(2)
            c_fl1.metric("Determinante det(A)", f"{res_fl['determinante']:.4f}")
            c_fl2.metric("Error de Inversa ||A·A⁻¹ - I||", f"{res_fl['error_inversa']:.2e}")

            st.markdown("#### Espectro de Autovalores")
            autov_df = pd.DataFrame({
                "Autovalor": [f"λ_{i+1}" for i in range(len(res_fl["autovalores"]))],
                "Valor": [f"{val:.4f}" for val in res_fl["autovalores"]],
            })
            st.dataframe(autov_df, use_container_width=True)

            st.markdown("#### Matriz Inversa Obtenida por Faddeev")
            st.dataframe(pd.DataFrame(np.round(res_fl["inversa"], 4)), use_container_width=True)

            with st.expander("Ver Matrices Intermedias B_k y Trazas"):
                for p in res_fl["pasos_recurrentes"]:
                    st.write(f"**Paso k = {p['k']}:** Fórmula `{p['formula']}` &mdash; Traza = `{p['traza']:.4f}`, $b_k = {p['b_k']:.4f}$")
                    st.dataframe(pd.DataFrame(np.round(p["B_k"], 4)), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Faddeev-Leverrier: {e}")


vista_lab4()
