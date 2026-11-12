"""
views/raices.py
===============
Categoría: Raíces de Ecuaciones No Lineales f(x) = 0.
Implementa el template estricto del 'Santo Grial':
1. Encabezado y Definición Teórica formal (LaTeX).
2. Bloque de Material de Estudio (PDFs de teoría y práctica resuelta).
3. Calculadora Interactiva (Workspace con core/, tablas y gráficos).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from core.lab3_raices import (
    analizar_descartes,
    barrido_tanteo,
    metodo_biseccion,
    metodo_interpolacion_lineal_bolzano,
    metodo_interpolacion_lineal_catedra,
    metodo_newton_raphson,
    metodo_punto_fijo_aitken,
    verificar_condiciones_fourier,
)
from views.ui_helpers import bloque_material_estudio


def parsear_expresion(expr_str: str, var_str: str = "x"):
    """Compila de forma segura una expresión matemática de SymPy a callable numérico."""
    x = sp.Symbol(var_str)
    expr_limpia = expr_str.replace("^", "**")
    expr_simb = sp.sympify(expr_limpia)
    fn_num = sp.lambdify(x, expr_simb, modules=["numpy", "math"])
    return x, expr_simb, fn_num


def vista_raices():
    st.title("📉 Raíces de Ecuaciones No Lineales")
    st.markdown(
        """
        Esta sección proporciona las herramientas para localizar y aproximar raíces reales de ecuaciones
        trascendentes y algebraicas $f(x) = 0$, abarcando métodos de intervalo (cerrados), métodos abiertos
        y esquemas de aceleración de convergencia.
        """
    )

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "1. Descartes y Bolzano",
        "2. Bisección",
        "3. Interpolación Lineal (Regula Falsi)",
        "4. Newton-Raphson",
        "5. Punto Fijo y Aitken",
        "6. Comparativa Multimétodo",
    ])

    # ==========================================================================
    # MÉTODO 1: DESCARTES Y BARRIDO DE BOLZANO
    # ==========================================================================
    with t1:
        st.subheader("1. Aislamiento Previo: Regla de Descartes y Teorema de Bolzano")
        st.latex(r"f(a) \cdot f(b) < 0 \implies \exists \, c \in (a, b) : f(c) = 0 \quad \text{(Teorema de Bolzano)}")
        st.latex(r"V\big(f(x)\big) - p = 2k \quad \text{(Regla de los Signos de Descartes para raíces positivas)}")

        bloque_material_estudio(
            pdf_teoria="06_teoria_descartes_bolzano_aislamiento.pdf",
            pdf_practica="06_practica_aislamiento_raices_tanteo.pdf",
            titulo_metodo="Regla de Descartes y Teorema de Bolzano",
        )

        st.markdown("#### 🛠️ Workspace: Aislamiento de Raíces")
        col1, col2 = st.columns([2, 1])
        with col1:
            expr_input = st.text_input("Función f(x):", value="x**3 - x - 1", key="r_f1")
        with col2:
            x_ini = st.number_input("x inicial de tanteo:", value=0.0, step=0.5, key="r_xini")
            dx_step = st.number_input("Paso de barrido (dx):", value=0.5, step=0.1, key="r_dx")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            st.latex(f"f(x) = {sp.latex(expr_simb)}")

            c_d, c_b = st.columns(2)
            with c_d:
                st.markdown("##### Análisis de Descartes")
                info_desc = analizar_descartes(expr_simb, x_sym)
                if info_desc["es_polinomio"]:
                    st.write(f"- Grado del polinomio: **{info_desc['grado']}**")
                    st.write(f"- Variaciones en f(x): **{info_desc['variaciones_pos']}** $\\to$ Posibles raíces (+): `{info_desc['posibles_raices_pos']}`")
                    st.write(f"- Variaciones en f(-x): **{info_desc['variaciones_neg']}** $\\to$ Posibles raíces (-): `{info_desc['posibles_raices_neg']}`")
                else:
                    st.info("La expresión no es un polinomio algebraico formal.")

            with c_b:
                st.markdown("##### Barrido de Bolzano")
                res_bolz = barrido_tanteo(fn_eval, x_ini=x_ini, dx=dx_step, max_pasos=20)
                if res_bolz["encontrado"]:
                    a_det, b_det = res_bolz["intervalo"]
                    st.success(f"✅ Cambio de signo detectado en: **[{a_det}, {b_det}]**")
                else:
                    st.warning("No se detectó cambio de signo en el rango de búsqueda.")

            if res_bolz["historial"]:
                st.dataframe(pd.DataFrame(res_bolz["historial"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error al analizar la función: {e}")

    # ==========================================================================
    # MÉTODO 2: BISECCIÓN
    # ==========================================================================
    with t2:
        st.subheader("2. Método de Bisección (Intervalo Medio)")
        st.latex(r"c_k = \frac{a_k + b_k}{2}, \qquad n \ge \frac{\ln(b-a) - \ln(\varepsilon)}{\ln 2}")

        bloque_material_estudio(
            pdf_teoria="07_teoria_metodo_biseccion.pdf",
            pdf_practica="07_practica_biseccion_tolerancia.pdf",
            titulo_metodo="Método de Bisección",
        )

        st.markdown("#### 🛠️ Workspace: Método de Bisección")
        cb1, cb2, cb3 = st.columns(3)
        with cb1:
            a_bis = st.number_input("Extremo inferior (a):", value=1.0, step=0.1, key="r_bis_a")
            b_bis = st.number_input("Extremo superior (b):", value=2.0, step=0.1, key="r_bis_b")
        with cb2:
            tol_bis = st.number_input("Tolerancia (epsilon):", value=1e-4, format="%.1e", key="r_bis_tol")
        with cb3:
            max_iter_bis = st.number_input("Iteraciones máximas:", value=50, step=5, key="r_bis_max")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            res_bis = metodo_biseccion(fn_eval, a=a_bis, b=b_bis, tol=tol_bis, max_iter=int(max_iter_bis))

            mb1, mb2, mb3, mb4 = st.columns(4)
            mb1.metric("Raíz Estimada", f"{res_bis['raiz']:.6f}")
            mb2.metric("f(raíz)", f"{res_bis['f_raiz']:.2e}")
            mb3.metric("Iteraciones", res_bis["iteraciones"])
            mb4.metric("Cota Teórica (n)", f"{res_bis['n_teorico_entero']} iter")

            st.dataframe(pd.DataFrame(res_bis["historial"]), use_container_width=True)

            x_grid = np.linspace(a_bis - 0.5, b_bis + 0.5, 300)
            y_grid = [fn_eval(val) for val in x_grid]
            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(x=x_grid, y=y_grid, mode="lines", name="f(x)", line=dict(color="#1f77b4", width=2)))
            fig_b.add_hline(y=0, line_dash="dash", line_color="black")
            fig_b.add_trace(go.Scatter(
                x=[res_bis["raiz"]], y=[res_bis["f_raiz"]],
                mode="markers", name="Raíz",
                marker=dict(color="#2ca02c", size=12, symbol="star"),
            ))
            fig_b.update_layout(title="Convergencia de Bisección", xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig_b, use_container_width=True)
        except Exception as e:
            st.error(f"Error en Bisección: {e}")

    # ==========================================================================
    # MÉTODO 3: INTERPOLACIÓN LINEAL (REGULA FALSI)
    # ==========================================================================
    with t3:
        st.subheader("3. Método de Interpolación Lineal (Regula Falsi)")
        st.latex(r"c_k = \frac{a_k f(b_k) - b_k f(a_k)}{f(b_k) - f(a_k)} \quad \text{(Clásica)}, \qquad x_{k+1} = x_k - \frac{f(x_k)}{f(x_k) - f(x_0)}(x_k - x_0) \quad \text{(Pivote Fijo)}")

        bloque_material_estudio(
            pdf_teoria="08_teoria_regula_falsi_interpolacion_lineal.pdf",
            pdf_practica="08_practica_regula_falsi_pivote_fijo.pdf",
            titulo_metodo="Método de Interpolación Lineal (Regula Falsi)",
        )

        st.markdown("#### 🛠️ Workspace: Regula Falsi")
        variante_rf = st.radio("Variante:", ["Variante Cátedra UNNE (Pivote Fijo)", "Variante Clásica (Bolzano)"], horizontal=True, key="r_rf_var")

        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            x0_rf = st.number_input("Punto x₀:", value=1.0, step=0.1, key="r_rf_x0")
            x1_rf = st.number_input("Punto x₁:", value=2.0, step=0.1, key="r_rf_x1")
        with cr2:
            tol_rf = st.number_input("Tolerancia:", value=1e-4, format="%.1e", key="r_rf_tol")
        with cr3:
            max_rf = st.number_input("Iteraciones máx:", value=50, step=5, key="r_rf_max")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            if "Cátedra" in variante_rf:
                res_rf = metodo_interpolacion_lineal_catedra(fn_eval, x0_fijo=x0_rf, x1_ini=x1_rf, tol=tol_rf, max_iter=int(max_rf))
            else:
                res_rf = metodo_interpolacion_lineal_bolzano(fn_eval, a=x0_rf, b=x1_rf, tol=tol_rf, max_iter=int(max_rf))

            mr1, mr2, mr3 = st.columns(3)
            mr1.metric("Raíz", f"{res_rf['raiz']:.6f}")
            mr2.metric("f(raíz)", f"{res_rf['f_raiz']:.2e}")
            mr3.metric("Iteraciones", res_rf["iteraciones"])

            st.dataframe(pd.DataFrame(res_rf["historial"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Regula Falsi: {e}")

    # ==========================================================================
    # MÉTODO 4: NEWTON-RAPHSON
    # ==========================================================================
    with t4:
        st.subheader("4. Método de Newton-Raphson (Tangentes)")
        st.latex(r"x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}, \qquad f(x_0) \cdot f''(x_0) > 0 \quad \text{(Condición de Fourier)}")

        bloque_material_estudio(
            pdf_teoria="09_teoria_newton_raphson_fourier.pdf",
            pdf_practica="09_practica_newton_raphson_convergencia.pdf",
            titulo_metodo="Método de Newton-Raphson",
        )

        st.markdown("#### 🛠️ Workspace: Newton-Raphson")
        cn1, cn2 = st.columns(2)
        with cn1:
            x0_nr = st.number_input("Semilla inicial (x₀):", value=2.0, step=0.1, key="r_nr_x0")
        with cn2:
            tol_nr = st.number_input("Tolerancia:", value=1e-5, format="%.1e", key="r_nr_tol")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            d1_expr = sp.diff(expr_simb, x_sym)
            d2_expr = sp.diff(d1_expr, x_sym)
            d1_num = sp.lambdify(x_sym, d1_expr, modules=["numpy", "math"])
            d2_num = sp.lambdify(x_sym, d2_expr, modules=["numpy", "math"])

            st.write(f"**Derivada primera:** $f'(x) = {sp.latex(d1_expr)}$")
            st.write(f"**Derivada segunda:** $f''(x) = {sp.latex(d2_expr)}$")

            cf_info = verificar_condiciones_fourier(fn_eval, d1_num, d2_num, a=min(1.0, x0_nr), b=max(2.0, x0_nr))
            if cf_info.get("fourier_cumple"):
                st.info(f"💡 Semilla óptima según Fourier: **x₀ = {cf_info['x0_sugerido']}** ({cf_info['detalle_fourier']})")

            res_nr = metodo_newton_raphson(fn_eval, d1=d1_num, x0=x0_nr, tol=tol_nr, max_iter=50)

            mn1, mn2, mn3 = st.columns(3)
            mn1.metric("Raíz", f"{res_nr['raiz']:.7f}")
            mn2.metric("f(raíz)", f"{res_nr['f_raiz']:.2e}")
            mn3.metric("Iteraciones", res_nr["iteraciones"])

            st.dataframe(pd.DataFrame(res_nr["historial"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Newton-Raphson: {e}")

    # ==========================================================================
    # MÉTODO 5: PUNTO FIJO Y AITKEN
    # ==========================================================================
    with t5:
        st.subheader("5. Método del Punto Fijo y Acelerador de Aitken Delta^2")
        st.latex(r"x_{k+1} = \varphi(x_k), \qquad \hat{x}_k = x_k - \frac{(x_{k+1} - x_k)^2}{x_{k+2} - 2x_{k+1} + x_k}")

        bloque_material_estudio(
            pdf_teoria="10_teoria_punto_fijo_aitken_delta2.pdf",
            pdf_practica="10_practica_punto_fijo_aceleracion.pdf",
            titulo_metodo="Punto Fijo y Aceleración de Aitken",
        )

        st.markdown("#### 🛠️ Workspace: Punto Fijo")
        cp1, cp2 = st.columns(2)
        with cp1:
            phi_str = st.text_input("Función contractiva phi(x):", value="(x + 1)**(1/3)", key="r_phi")
            x0_pf = st.number_input("Semilla inicial x₀:", value=1.5, step=0.1, key="r_pf_x0")
        with cp2:
            tol_pf = st.number_input("Tolerancia:", value=1e-4, format="%.1e", key="r_pf_tol")
            max_pf = st.number_input("Iteraciones máximas:", value=50, step=5, key="r_pf_max")

        try:
            x_sym, phi_simb, phi_num = parsear_expresion(phi_str)
            st.latex(f"\\varphi(x) = {sp.latex(phi_simb)}")

            res_ait = metodo_punto_fijo_aitken(phi_num, x0=x0_pf, tol=tol_pf, max_iter=int(max_pf))
            pf = res_ait["punto_fijo"]
            ait = res_ait["aitken"]

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("##### Punto Fijo Simple")
                st.metric("Raíz", f"{pf['raiz']:.6f}")
                st.metric("Iteraciones", pf["iteraciones"])
                st.dataframe(pd.DataFrame(pf["historial"]), use_container_width=True)

            with col_p2:
                st.markdown("##### Acelerador de Aitken")
                st.metric("Raíz Aitken", f"{ait['raiz']:.6f}")
                st.metric("Iteraciones", ait["iteraciones"])
                st.dataframe(pd.DataFrame(ait["historial"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Punto Fijo: {e}")

    # ==========================================================================
    # MÉTODO 6: COMPARATIVA MULTIMÉTODO
    # ==========================================================================
    with t6:
        st.subheader("6. Banco de Pruebas: Comparativa Multimétodo")
        st.markdown("Analiza la velocidad de convergencia y costo computacional entre los métodos estudiados.")

        bloque_material_estudio(
            pdf_teoria="11_teoria_comparativa_convergencia_raices.pdf",
            pdf_practica="11_practica_analisis_costo_computacional.pdf",
            titulo_metodo="Comparativa de Métodos para Raíces",
        )

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            d1_num = sp.lambdify(x_sym, sp.diff(expr_simb, x_sym), modules=["numpy", "math"])

            r_bis = metodo_biseccion(fn_eval, a=1.0, b=2.0, tol=1e-4, max_iter=50)
            r_fal = metodo_interpolacion_lineal_catedra(fn_eval, x0_fijo=1.0, x1_ini=2.0, tol=1e-4, max_iter=50)
            r_new = metodo_newton_raphson(fn_eval, d1=d1_num, x0=2.0, tol=1e-4, max_iter=50)

            resumen_data = [
                {"Método": "Bisección", "Raíz": r_bis["raiz"], "Iteraciones": r_bis["iteraciones"], "Error Final": r_bis["error_final"], "Tiempo (ms)": r_bis["duracion"] * 1000},
                {"Método": "Regula Falsi (Cátedra)", "Raíz": r_fal["raiz"], "Iteraciones": r_fal["iteraciones"], "Error Final": r_fal["error_final"], "Tiempo (ms)": r_fal["duracion"] * 1000},
                {"Método": "Newton-Raphson", "Raíz": r_new["raiz"], "Iteraciones": r_new["iteraciones"], "Error Final": r_new["error_final"], "Tiempo (ms)": r_new["duracion"] * 1000},
            ]
            st.dataframe(pd.DataFrame(resumen_data), use_container_width=True)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=[d["Método"] for d in resumen_data],
                y=[d["Iteraciones"] for d in resumen_data],
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                text=[f"{d['Iteraciones']} iter" for d in resumen_data],
                textposition="auto",
            ))
            fig_comp.update_layout(title="Iteraciones Requeridas para Converger (tol = 1e-4)", yaxis_title="Iteraciones")
            st.plotly_chart(fig_comp, use_container_width=True)
        except Exception as e:
            st.error(f"Error en Comparativa: {e}")


vista_raices()
