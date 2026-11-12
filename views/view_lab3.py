"""
views/view_lab3.py
==================
Vista interactiva para el Laboratorio 3: Búsqueda de Raíces de Ecuaciones No Lineales.
- Teoremas de Bolzano y Regla de Descartes.
- Método de Bisección (Intervalo Medio) con cálculo de cota teórica.
- Método de Interpolación Lineal (Regula Falsi): Variante Cátedra (pivote fijo) y Variante Bolzano.
- Método de Newton-Raphson (Tangentes) con verificación de condiciones de Fourier.
- Método del Punto Fijo con Aceleración de Aitken Delta^2.
- Comparativa Multimétodo de Convergencia.
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


def parsear_expresion(expr_str: str, var_str: str = "x"):
    """Compila de forma segura una expresión matemática de SymPy a callable numérico."""
    x = sp.Symbol(var_str)
    # Reemplazar '^' por '**' si el usuario lo ingresa
    expr_limpia = expr_str.replace("^", "**")
    expr_simb = sp.sympify(expr_limpia)
    fn_num = sp.lambdify(x, expr_simb, modules=["numpy", "math"])
    return x, expr_simb, fn_num


def vista_lab3():
    st.title("🎯 Laboratorio 3: Solución de Ecuaciones No Lineales")
    st.markdown(
        """
        En este laboratorio se analizan las técnicas numéricas clásicas para localizar y aproximar
        raíces reales de funciones no lineales $f(x) = 0$, analizando cotas de error, velocidad de
        convergencia y aceleración algorítmica.
        """
    )

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "1. Descartes y Bolzano",
        "2. Bisección",
        "3. Interpolación Lineal",
        "4. Newton-Raphson",
        "5. Punto Fijo y Aitken",
        "6. Comparativa Multimétodo",
    ])

    # ==========================================================================
    # PESTAÑA 1: DESCARTES Y BARRIDO DE BOLZANO
    # ==========================================================================
    with t1:
        st.subheader("1. Análisis Previo: Regla de Descartes y Tanteos Sistemáticos (Bolzano)")
        st.markdown(
            r"""
            Antes de aplicar algoritmos iterativos, se determina la posible cantidad de raíces reales
            y se acota un intervalo $[a, b]$ donde exista cambio de signo ($f(a) \cdot f(b) < 0$).
            """
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            expr_input = st.text_input(
                "Función f(x) (sintaxis SymPy o Python):",
                value="x**3 - x - 1",
                key="v3_t1_f",
                help="Ejemplo: x**3 - x - 1, x**3 - 4*x**2 - 4*x - 6, cos(x) - x",
            )
        with col2:
            x_ini = st.number_input("x inicial tanteo:", value=0.0, step=0.5, key="v3_t1_xini")
            dx_step = st.number_input("Paso de barrido (dx):", value=0.5, step=0.1, key="v3_t1_dx")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            st.latex(f"f(x) = {sp.latex(expr_simb)}")

            c_desc, c_bolz = st.columns(2)

            with c_desc:
                st.markdown("#### Regla de los Signos de Descartes")
                info_desc = analizar_descartes(expr_simb, x_sym)
                if info_desc["es_polinomio"]:
                    st.write(f"**Grado del polinomio:** {info_desc['grado']}")
                    st.markdown(f"- Variaciones de signo en $f(x)$: **{info_desc['variaciones_pos']}**")
                    st.write(f"  Posibles raíces reales positivas: `{info_desc['posibles_raices_pos']}`")
                    st.markdown(f"- Variaciones de signo en $f(-x)$: **{info_desc['variaciones_neg']}**")
                    st.write(f"  Posibles raíces reales negativas: `{info_desc['posibles_raices_neg']}`")
                else:
                    st.info("La expresión no es un polinomio algebraico formal; Descartes aplica estrictamente a polinomios.")

            with c_bolz:
                st.markdown("#### Barrido Sistemático de Bolzano")
                res_bolz = barrido_tanteo(fn_eval, x_ini=x_ini, dx=dx_step, max_pasos=20)
                if res_bolz["encontrado"]:
                    a_det, b_det = res_bolz["intervalo"]
                    st.success(f"✅ Cambio de signo detectado en el intervalo: **[{a_det}, {b_det}]**")
                else:
                    st.warning("No se encontró cambio de signo en los pasos analizados. Pruebe otro paso o punto inicial.")

            if res_bolz["historial"]:
                st.markdown("#### Historial de Pasos del Tanteo")
                df_tanteo = pd.DataFrame(res_bolz["historial"])
                st.dataframe(df_tanteo, use_container_width=True)

        except Exception as e:
            st.error(f"Error al analizar la expresión: {e}")

    # ==========================================================================
    # PESTAÑA 2: BISECCIÓN
    # ==========================================================================
    with t2:
        st.subheader("2. Método de Bisección (Intervalo Medio)")
        st.latex(r"c_k = \frac{a_k + b_k}{2}, \quad n \ge \frac{\ln(b-a) - \ln(\varepsilon)}{\ln 2}")

        cb1, cb2, cb3 = st.columns(3)
        with cb1:
            a_bis = st.number_input("Extremo a:", value=1.0, step=0.1, key="v3_bis_a")
            b_bis = st.number_input("Extremo b:", value=2.0, step=0.1, key="v3_bis_b")
        with cb2:
            tol_bis = st.number_input("Tolerancia (epsilon):", value=1e-4, format="%.1e", key="v3_bis_tol")
        with cb3:
            max_iter_bis = st.number_input("Iteraciones máximas:", value=50, step=5, key="v3_bis_max")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            res_bis = metodo_biseccion(fn_eval, a=a_bis, b=b_bis, tol=tol_bis, max_iter=int(max_iter_bis))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Raíz estimada", f"{res_bis['raiz']:.6f}")
            m2.metric("f(raíz)", f"{res_bis['f_raiz']:.2e}")
            m3.metric("Iteraciones", res_bis["iteraciones"])
            m4.metric("Cota teórica (n)", f"{res_bis['n_teorico_entero']} iter")

            st.markdown("#### Tabla de Iteraciones Paso a Paso")
            df_bis = pd.DataFrame(res_bis["historial"])
            st.dataframe(df_bis, use_container_width=True)

            # Gráfica de convergencia y función
            x_vals = np.linspace(min(a_bis, b_bis) - 0.5, max(a_bis, b_bis) + 0.5, 300)
            y_vals = [fn_eval(val) for val in x_vals]

            fig_bis = go.Figure()
            fig_bis.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x)", line=dict(color="#1f77b4", width=2)))
            fig_bis.add_hline(y=0, line_dash="dash", line_color="black")
            fig_bis.add_trace(go.Scatter(
                x=[res_bis["raiz"]],
                y=[res_bis["f_raiz"]],
                mode="markers",
                name="Raíz hallada",
                marker=dict(color="#2ca02c", size=12, symbol="star"),
            ))
            fig_bis.update_layout(title="Localización de la Raíz por Bisección", xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig_bis, use_container_width=True)

        except Exception as e:
            st.error(f"Error en Bisección: {e}")

    # ==========================================================================
    # PESTAÑA 3: INTERPOLACIÓN LINEAL (REGULA FALSI)
    # ==========================================================================
    with t3:
        st.subheader("3. Método de Interpolación Lineal (Regula Falsi)")
        st.markdown(
            """
            A diferencia de Bisección, utiliza los valores de la función para trazar una recta secante
            y acelerar la convergencia hacia el eje horizontal.
            """
        )

        variante = st.radio(
            "Seleccione Variante de Implementación:",
            ["Variante Cátedra UNNE (Pivote Fijo x₀)", "Variante Clásica (Bolzano / Actualización Dinámica)"],
            horizontal=True,
            key="v3_radio_falsi",
        )

        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            x0_f = st.number_input("Pivote / Extremo x₀:", value=1.0, step=0.1, key="v3_f_x0")
            x1_f = st.number_input("Punto inicial x₁:", value=2.0, step=0.1, key="v3_f_x1")
        with cf2:
            tol_f = st.number_input("Tolerancia:", value=1e-4, format="%.1e", key="v3_f_tol")
        with cf3:
            max_iter_f = st.number_input("Iteraciones máx:", value=50, step=5, key="v3_f_max")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            if "Cátedra" in variante:
                st.latex(r"x_{k+1} = x_k - \frac{f(x_k)}{f(x_k) - f(x_0)} (x_k - x_0)")
                res_falsi = metodo_interpolacion_lineal_catedra(fn_eval, x0_fijo=x0_f, x1_ini=x1_f, tol=tol_f, max_iter=int(max_iter_f))
            else:
                st.latex(r"c_k = \frac{a_k f(b_k) - b_k f(a_k)}{f(b_k) - f(a_k)}")
                res_falsi = metodo_interpolacion_lineal_bolzano(fn_eval, a=x0_f, b=x1_f, tol=tol_f, max_iter=int(max_iter_f))

            if not res_falsi["convergente"]:
                st.warning(res_falsi["mensaje"])

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Raíz hallada", f"{res_falsi['raiz']:.6f}")
            mc2.metric("f(raíz)", f"{res_falsi['f_raiz']:.2e}")
            mc3.metric("Iteraciones", res_falsi["iteraciones"])

            st.markdown("#### Historial de Iteraciones")
            st.dataframe(pd.DataFrame(res_falsi["historial"]), use_container_width=True)

        except Exception as e:
            st.error(f"Error en Regula Falsi: {e}")

    # ==========================================================================
    # PESTAÑA 4: NEWTON-RAPHSON
    # ==========================================================================
    with t4:
        st.subheader("4. Método de Newton-Raphson (Tangentes)")
        st.latex(r"x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}")

        cn1, cn2 = st.columns([2, 1])
        with cn1:
            st.markdown("#### Verificación de Condiciones de Fourier")
            st.latex(r"f(a) \cdot f(b) < 0, \quad f'(x) \neq 0, \quad f''(x) \neq 0, \quad f(x_0) \cdot f''(x_0) > 0")
        with cn2:
            x0_newt = st.number_input("Semilla inicial x₀:", value=2.0, step=0.1, key="v3_newt_x0")
            tol_newt = st.number_input("Tolerancia:", value=1e-5, format="%.1e", key="v3_newt_tol")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            d1_expr = sp.diff(expr_simb, x_sym)
            d2_expr = sp.diff(d1_expr, x_sym)
            d1_num = sp.lambdify(x_sym, d1_expr, modules=["numpy", "math"])
            d2_num = sp.lambdify(x_sym, d2_expr, modules=["numpy", "math"])

            st.write(f"**Derivada primera:** $f'(x) = {sp.latex(d1_expr)}$")
            st.write(f"**Derivada segunda:** $f''(x) = {sp.latex(d2_expr)}$")

            # Comprobar Fourier en [x0-1, x0+1] o intervalo de prueba
            cf_info = verificar_condiciones_fourier(fn_eval, d1_num, d2_num, a=min(1.0, x0_newt), b=max(2.0, x0_newt))
            if cf_info.get("fourier_cumple"):
                st.info(f"💡 Semilla óptima según Fourier: **x₀ = {cf_info['x0_sugerido']}** ({cf_info['detalle_fourier']})")

            res_newt = metodo_newton_raphson(fn_eval, d1=d1_num, x0=x0_newt, tol=tol_newt, max_iter=50)

            mn1, mn2, mn3 = st.columns(3)
            mn1.metric("Raíz", f"{res_newt['raiz']:.7f}")
            mn2.metric("f(raíz)", f"{res_newt['f_raiz']:.2e}")
            mn3.metric("Iteraciones", res_newt["iteraciones"])

            st.markdown("#### Historial de Newton-Raphson")
            df_newt = pd.DataFrame(res_newt["historial"])
            st.dataframe(df_newt, use_container_width=True)

            # Gráfica de tangentes
            fig_nr = go.Figure()
            x_vals = np.linspace(res_newt["raiz"] - 1.5, res_newt["raiz"] + 1.5, 200)
            y_vals = [fn_eval(val) for val in x_vals]
            fig_nr.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x)", line=dict(color="#1f77b4", width=2)))
            fig_nr.add_hline(y=0, line_dash="dash", line_color="black")

            # Dibujar un par de tangentes del historial
            for p in res_newt["historial"][:4]:
                xk = p["x_k"]
                yk = p["f_xk"]
                dk = p["f_prima_xk"]
                if dk is not None and not math.isnan(dk) and dk != 0:
                    xt_line = np.array([xk - 0.5, xk + 0.5])
                    yt_line = yk + dk * (xt_line - xk)
                    fig_nr.add_trace(go.Scatter(
                        x=xt_line, y=yt_line, mode="lines",
                        line=dict(dash="dot", width=1),
                        name=f"Tangente en x_{p['iteracion']}",
                    ))

            fig_nr.update_layout(title="Tangentes Sucesivas de Newton-Raphson", xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig_nr, use_container_width=True)

        except Exception as e:
            st.error(f"Error en Newton-Raphson: {e}")

    # ==========================================================================
    # PESTAÑA 5: PUNTO FIJO Y AITKEN
    # ==========================================================================
    with t5:
        st.subheader("5. Iteración de Punto Fijo y Aceleración de Aitken Delta^2")
        st.latex(r"x_{k+1} = \varphi(x_k), \qquad \hat{x}_k = x_k - \frac{(x_{k+1} - x_k)^2}{x_{k+2} - 2x_{k+1} + x_k}")

        st.markdown(
            r"""
            Para la ecuación $x^3 - x - 1 = 0$, despejando $x = \sqrt[3]{x + 1}$ se tiene una función contractiva
            $\varphi(x)$ con $|\varphi'(x)| < 1$.
            """
        )

        ca1, ca2 = st.columns(2)
        with ca1:
            phi_str = st.text_input("Función phi(x):", value="(x + 1)**(1/3)", key="v3_phi")
            x0_ait = st.number_input("Semilla x₀:", value=1.5, step=0.1, key="v3_x0_ait")
        with ca2:
            tol_ait = st.number_input("Tolerancia:", value=1e-4, format="%.1e", key="v3_tol_ait")
            max_iter_ait = st.number_input("Iteraciones máx:", value=50, step=5, key="v3_max_ait")

        try:
            x_sym, phi_simb, phi_num = parsear_expresion(phi_str)
            st.latex(f"\\varphi(x) = {sp.latex(phi_simb)}")

            res_ait = metodo_punto_fijo_aitken(phi_num, x0=x0_ait, tol=tol_ait, max_iter=int(max_iter_ait))
            pf = res_ait["punto_fijo"]
            ait = res_ait["aitken"]

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("#### Punto Fijo Simple")
                st.metric("Raíz Simple", f"{pf['raiz']:.6f}")
                st.metric("Iteraciones", pf["iteraciones"])
                st.dataframe(pd.DataFrame(pf["historial"]), use_container_width=True)

            with col_p2:
                st.markdown("#### Acelerador de Aitken Delta^2")
                st.metric("Raíz Aitken", f"{ait['raiz']:.6f}")
                st.metric("Iteraciones", ait["iteraciones"])
                st.dataframe(pd.DataFrame(ait["historial"]), use_container_width=True)

        except Exception as e:
            st.error(f"Error en Punto Fijo: {e}")

    # ==========================================================================
    # PESTAÑA 6: COMPARATIVA MULTIMÉTODO
    # ==========================================================================
    with t6:
        st.subheader("6. Comparativa Multimétodo de Eficiencia y Convergencia")
        st.markdown("Compara en paralelo los métodos estudiados para la función y tolerancia seleccionadas.")

        try:
            x_sym, expr_simb, fn_eval = parsear_expresion(expr_input)
            d1_num = sp.lambdify(x_sym, sp.diff(expr_simb, x_sym), modules=["numpy", "math"])

            r_bis = metodo_biseccion(fn_eval, a=1.0, b=2.0, tol=1e-4, max_iter=50)
            r_fal = metodo_interpolacion_lineal_catedra(fn_eval, x0_fijo=1.0, x1_ini=2.0, tol=1e-4, max_iter=50)
            r_new = metodo_newton_raphson(fn_eval, d1=d1_num, x0=2.0, tol=1e-4, max_iter=50)

            resumen_data = [
                {"Método": "Bisección", "Raíz": r_bis["raiz"], "Iteraciones": r_bis["iteraciones"], "Error Final": r_bis["error_final"], "Tiempo (ms)": r_bis["duracion"] * 1000},
                {"Método": "Interpolación Lineal (Cátedra)", "Raíz": r_fal["raiz"], "Iteraciones": r_fal["iteraciones"], "Error Final": r_fal["error_final"], "Tiempo (ms)": r_fal["duracion"] * 1000},
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
            fig_comp.update_layout(title="Cantidad de Iteraciones para Converger (tol = 1e-4)", yaxis_title="Iteraciones")
            st.plotly_chart(fig_comp, use_container_width=True)

        except Exception as e:
            st.error(f"Error en Comparativa: {e}")


vista_lab3()

