"""
views/interpolacion.py
======================
Categoría: Interpolación y Aproximación Polinómica.
Implementa el template estricto del 'Santo Grial':
1. Encabezado y Definición Teórica formal (LaTeX).
2. Bloque de Material de Estudio (PDFs de teoría y práctica resuelta).
3. Calculadora Interactiva (Workspace con core/, tablas y gráficos).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from core.lab5_interpolacion import (
    calcular_diferencias_ascendentes,
    calcular_diferencias_descendentes,
    construir_polinomio_simbolico_lagrange,
    interpolar_cuadratica_inversa,
    interpolar_lagrange,
    interpolar_lineal_directa,
    interpolar_lineal_inversa,
    interpolar_newton_gregory_ascendente,
    interpolar_newton_gregory_descendente,
)
from views.ui_helpers import bloque_material_estudio


def vista_interpolacion():
    st.title("📈 Interpolación y Aproximación Polinómica")
    st.markdown(
        """
        Esta sección agrupa los métodos para ajustar un polinomio único a un conjunto discreto
        de pares de datos $(x_i, y_i)$, abarcando formulaciones globales (Lagrange) y esquemas
        basados en operadores de diferencias finitas equiespaciadas (Newton-Gregory).
        """
    )

    t1, t2, t3, t4, t5 = st.tabs([
        "1. Interpolación Lineal",
        "2. Cuadrática Inversa",
        "3. Polinomio de Lagrange",
        "4. Newton-Gregory Ascendente",
        "5. Newton-Gregory Descendente",
    ])

    # ==========================================================================
    # MÉTODO 1: INTERPOLACIÓN LINEAL (DIRECTA E INVERSA)
    # ==========================================================================
    with t1:
        st.subheader("1. Interpolación Lineal Directa e Inversa")
        st.latex(r"y(x) = y_0 + \frac{y_1 - y_0}{x_1 - x_0}(x - x_0), \qquad x(y) = x_0 + \frac{x_1 - x_0}{y_1 - y_0}(y - y_0)")

        bloque_material_estudio(
            pdf_teoria="18_teoria_interpolacion_lineal_directa_inversa.pdf",
            pdf_practica="18_practica_interpolacion_lineal_tablas.pdf",
            titulo_metodo="Interpolación Lineal Directa e Inversa",
        )

        st.markdown("#### 🛠️ Workspace: Interpolación Lineal")
        cl1, cl2 = st.columns(2)
        with cl1:
            st.markdown("##### Nodos Extremos")
            x0_l = st.number_input("x₀:", value=2.0, key="int_x0_l")
            y0_l = st.number_input("y₀:", value=7.3891, key="int_y0_l")
            x1_l = st.number_input("x₁:", value=3.0, key="int_x1_l")
            y1_l = st.number_input("y₁:", value=20.0855, key="int_y1_l")

        with cl2:
            st.markdown("##### Modo de Evaluación")
            tipo_eval = st.radio("Modo:", ["Directa (evaluar x)", "Inversa (hallar x dado y)"], key="int_tipo_l")
            try:
                if "Directa" in tipo_eval:
                    x_val = st.number_input("Valor x a evaluar:", value=2.5, key="int_xeval_l")
                    res_dir = interpolar_lineal_directa([x0_l, x1_l], [y0_l, y1_l], x_val)
                    st.metric("y(x) interpolado", f"{res_dir['y_interpolado']:.6f}")
                    st.write(f"Pendiente m: `{res_dir['pendiente']:.6f}`")
                else:
                    y_val = st.number_input("Valor y buscado:", value=12.0, key="int_ybusc_l")
                    res_inv = interpolar_lineal_inversa([x0_l, x1_l], [y0_l, y1_l], y_val)
                    st.metric("x(y) interpolado", f"{res_inv['x_resultado']:.6f}")
            except Exception as e:
                st.error(f"Error en Interpolación Lineal: {e}")

    # ==========================================================================
    # MÉTODO 2: INTERPOLACIÓN CUADRÁTICA INVERSA
    # ==========================================================================
    with t2:
        st.subheader("2. Interpolación Cuadrática Inversa")
        st.latex(r"a u^2 + b u + c = 0, \qquad u = x - x_0, \qquad u = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")

        bloque_material_estudio(
            pdf_teoria="19_teoria_interpolacion_cuadratica_inversa.pdf",
            pdf_practica="19_practica_cuadratica_inversa_seleccion_raiz.pdf",
            titulo_metodo="Interpolación Cuadrática Inversa",
        )

        st.markdown("#### 🛠️ Workspace: Cuadrática Inversa")
        cq1, cq2 = st.columns(2)
        with cq1:
            puntos_q = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
            st.markdown("##### Tabla de 3 Nodos (x, y)")
            ed_q = st.data_editor(puntos_q, key="int_ed_cuad")
            xq = ed_q["x"].tolist()
            yq = ed_q["y"].tolist()

        with cq2:
            y_deseado = st.number_input("Valor y* fijado:", value=5.0, key="int_y_cuad")

        try:
            res_cuad = interpolar_cuadratica_inversa(xq, yq, y_deseado)
            st.success(f"Raíz físicamente admisible seleccionada: **x = {res_cuad['x_resultado']:.6f}**")
            st.latex(f"{res_cuad['a_coef']:.4f} u^2 + {res_cuad['b_coef']:.4f} u + {res_cuad['c_coef']:.4f} = 0")
            st.write(f"Discriminante: $\\Delta = {res_cuad['discriminante']:.6f}$")
            st.write(f"Raíces para u: $u_1 = {res_cuad['u1']:.6f}$, $u_2 = {res_cuad['u2']:.6f}$")
            st.write(f"Candidatos para x: $x_1 = {res_cuad['x_sol1']:.6f}$, $x_2 = {res_cuad['x_sol2']:.6f}$")
        except Exception as e:
            st.error(f"Error en Cuadrática Inversa: {e}")

    # ==========================================================================
    # MÉTODO 3: POLINOMIO DE LAGRANGE
    # ==========================================================================
    with t3:
        st.subheader("3. Polinomio de Interpolación de Lagrange")
        st.latex(r"P_n(x) = \sum_{i=0}^n y_i L_i(x), \qquad L_i(x) = \prod_{j=0, j \neq i}^n \frac{x - x_j}{x_i - x_j}")

        bloque_material_estudio(
            pdf_teoria="20_teoria_polinomio_lagrange_bases.pdf",
            pdf_practica="20_practica_lagrange_construccion_simbolica.pdf",
            titulo_metodo="Polinomio de Lagrange",
        )

        st.markdown("#### 🛠️ Workspace: Lagrange")
        df_lag = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.7183, 7.3891, 20.0855]})
        ed_lag = st.data_editor(df_lag, num_rows="dynamic", key="int_ed_lag")
        x_lag = ed_lag["x"].dropna().tolist()
        y_lag = ed_lag["y"].dropna().tolist()

        if len(x_lag) >= 2 and len(x_lag) == len(y_lag):
            x_eval_lag = st.number_input("Punto x a evaluar en el polinomio:", value=1.5, key="int_xeval_lag")
            try:
                res_lag = interpolar_lagrange(x_lag, y_lag, x_eval_lag)
                res_simb = construir_polinomio_simbolico_lagrange(x_lag, y_lag)

                st.metric("P(x) evaluado", f"{res_lag['y_interpolado']:.6f}")
                st.latex(f"P(x) = {res_simb['latex_expandido']}")

                x_min, x_max = min(x_lag), max(x_lag)
                span = x_max - x_min if x_max != x_min else 1.0
                x_grid = np.linspace(x_min - 0.2 * span, x_max + 0.2 * span, 200)

                p_sym_fn = sp.lambdify(sp.Symbol("x"), res_simb["polinomio_simbolico"], modules=["numpy", "math"])
                y_grid = [p_sym_fn(xi) for xi in x_grid]

                fig_lag = go.Figure()
                fig_lag.add_trace(go.Scatter(x=x_grid, y=y_grid, mode="lines", name="Polinomio P(x)", line=dict(color="#1f77b4", width=2)))
                fig_lag.add_trace(go.Scatter(x=x_lag, y=y_lag, mode="markers", name="Nodos de datos", marker=dict(size=10, color="#d62728")))
                fig_lag.add_trace(go.Scatter(
                    x=[x_eval_lag], y=[res_lag["y_interpolado"]],
                    mode="markers", name="Punto evaluado",
                    marker=dict(size=14, color="#2ca02c", symbol="star"),
                ))
                fig_lag.update_layout(title="Ajuste Polinómico de Lagrange P_n(x)", xaxis_title="x", yaxis_title="y")
                st.plotly_chart(fig_lag, use_container_width=True)

                with st.expander("Ver Polinomios Base Li(x) evaluados"):
                    df_bases = pd.DataFrame(res_lag["terminos_detalle"])
                    st.dataframe(df_bases, use_container_width=True)
            except Exception as e:
                st.error(f"Error en Lagrange: {e}")

    # ==========================================================================
    # MÉTODO 4: NEWTON-GREGORY ASCENDENTE
    # ==========================================================================
    with t4:
        st.subheader("4. Newton-Gregory Ascendente (Diferencias Progresivas)")
        st.latex(r"P_n(x) = y_0 + u \Delta y_0 + \frac{u(u-1)}{2!} \Delta^2 y_0 + \dots, \qquad u = \frac{x - x_0}{h}")

        bloque_material_estudio(
            pdf_teoria="21_teoria_newton_gregory_ascendente.pdf",
            pdf_practica="21_practica_diferencias_progresivas_tabla.pdf",
            titulo_metodo="Newton-Gregory Ascendente",
        )

        st.markdown("#### 🛠️ Workspace: Newton-Gregory Ascendente")
        df_ng_asc = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0], "y": [0.1736, 0.3420, 0.5000, 0.6428, 0.7660]})
        ed_asc = st.data_editor(df_ng_asc, key="int_ed_asc")
        x_asc = ed_asc["x"].tolist()
        y_asc = ed_asc["y"].tolist()

        x_eval_asc = st.number_input("Punto x a interpolar:", value=15.0, key="int_x_asc")

        try:
            res_asc = interpolar_newton_gregory_ascendente(x_asc, y_asc, x_eval_asc)
            c1, c2, c3 = st.columns(3)
            c1.metric("P(x) interpolado", f"{res_asc['y_interpolado']:.6f}")
            c2.metric("Paso equiespaciado (h)", f"{res_asc['h']:.4f}")
            c3.metric("Parámetro u", f"{res_asc['u']:.4f}")

            st.markdown("##### Tabla de Diferencias Finitas Progresivas (Δ)")
            st.dataframe(pd.DataFrame(res_asc["tabla_diferencias"]), use_container_width=True)

            st.markdown("##### Desglose de Términos del Polinomio")
            st.dataframe(pd.DataFrame(res_asc["terminos"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Newton-Gregory Ascendente: {e}")

    # ==========================================================================
    # MÉTODO 5: NEWTON-GREGORY DESCENDENTE
    # ==========================================================================
    with t5:
        st.subheader("5. Newton-Gregory Descendente (Diferencias Regresivas)")
        st.latex(r"P_n(x) = y_n + v \nabla y_n + \frac{v(v+1)}{2!} \nabla^2 y_n + \dots, \qquad v = \frac{x - x_n}{h}")

        bloque_material_estudio(
            pdf_teoria="22_teoria_newton_gregory_descendente.pdf",
            pdf_practica="22_practica_diferencias_regresivas_tabla.pdf",
            titulo_metodo="Newton-Gregory Descendente",
        )

        st.markdown("#### 🛠️ Workspace: Newton-Gregory Descendente")
        x_eval_desc = st.number_input("Punto x a interpolar:", value=45.0, key="int_x_desc")

        try:
            res_desc = interpolar_newton_gregory_descendente(x_asc, y_asc, x_eval_desc)
            cd1, cd2, cd3 = st.columns(3)
            cd1.metric("P(x) interpolado", f"{res_desc['y_interpolado']:.6f}")
            cd2.metric("Paso equiespaciado (h)", f"{res_desc['h']:.4f}")
            cd3.metric("Parámetro v", f"{res_desc['v']:.4f}")

            st.markdown("##### Tabla de Diferencias Regresivas (∇)")
            st.dataframe(pd.DataFrame(res_desc["tabla_diferencias"]), use_container_width=True)

            st.markdown("##### Desglose de Términos Regresivos")
            st.dataframe(pd.DataFrame(res_desc["terminos"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Newton-Gregory Descendente: {e}")


vista_interpolacion()

