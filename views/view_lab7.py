"""
views/view_lab7.py
==================
Vista interactiva para el Laboratorio 7: Ecuaciones Diferenciales Ordinarias (EDO).
- Problemas de Valor Inicial (PVI): dy/dx = f(x, y), y(x0) = y0.
- Método de Euler y Euler Modificado (Heun).
- Métodos de Runge-Kutta de orden 2 (RK2) y orden 4 clásico (RK4).
- Método Predictor-Corrector Multipaso de Milne con arranque por RK4.
- Comparativa Multimétodo y contraste contra solución exacta analítica.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from core.lab7_edo import (
    comparar_metodos_edo,
    compilar_funcion_edo,
    metodo_euler,
    metodo_euler_modificado,
    metodo_milne,
    metodo_runge_kutta_2,
    metodo_runge_kutta_4,
)


def vista_lab7():
    st.title("🚀 Laboratorio 7: Ecuaciones Diferenciales Ordinarias (EDO)")
    st.markdown(
        r"""
        Este módulo modela numéricamente la evolución temporal de Problemas de Valor Inicial (PVI)
        mediante métodos de un solo paso (Euler, Heun, RK2, RK4) y métodos multipaso
        autocorrectores (Milne de orden 4), evaluando su estabilidad y convergencia.
        """
    )

    t1, t2, t3, t4 = st.tabs([
        "1. Euler y Euler Modificado",
        "2. Runge-Kutta (RK2 y RK4)",
        "3. Predictor-Corrector de Milne",
        "4. Comparativa Multimétodo",
    ])

    # Configuración compartida del PVI
    st.sidebar.header("⚙️ Parámetros del PVI")
    f_input = st.sidebar.text_input(
        "f(x, y) = dy/dx:",
        value="x + y",
        help="Sintaxis Python/SymPy: x + y, y - x**2 + 1, -2*x*y, etc.",
        key="v7_f_edo",
    )
    x0_val = st.sidebar.number_input("Condición inicial x₀:", value=0.0, step=0.1, key="v7_x0")
    y0_val = st.sidebar.number_input("Condición inicial y₀:", value=1.0, step=0.1, key="v7_y0")
    h_val = st.sidebar.number_input("Paso de integración (h):", value=0.1, min_value=0.001, max_value=1.0, step=0.05, format="%.3f", key="v7_h")
    n_pasos = st.sidebar.number_input("Número de pasos (n):", value=10, min_value=4, max_value=200, step=1, key="v7_n")

    try:
        fn_edo = compilar_funcion_edo(f_input)
    except Exception as e:
        st.error(f"Error al compilar la función f(x, y): {e}")
        return

    # ==========================================================================
    # PESTAÑA 1: EULER Y EULER MODIFICADO
    # ==========================================================================
    with t1:
        st.subheader("1. Método de Euler y Método de Heun (Euler Modificado)")
        st.latex(r"y_{k+1} = y_k + h \cdot f(x_k, y_k) \quad \text{(Euler Clásico, orden 1)}")
        st.latex(
            r"y_{k+1} = y_k + \frac{h}{2} \left[ f(x_k, y_k) + f(x_{k+1}, y_k + h f(x_k, y_k)) \right] \quad \text{(Heun, orden 2)}"
        )

        try:
            res_euler = metodo_euler(fn_edo, x0=x0_val, y0=y0_val, h=h_val, n_pasos=int(n_pasos))
            res_heun = metodo_euler_modificado(fn_edo, x0=x0_val, y0=y0_val, h=h_val, n_pasos=int(n_pasos))

            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.markdown("#### Resultados: Euler")
                st.metric("y(x_n) final", f"{res_euler['y'][-1]:.6f}")
                st.dataframe(pd.DataFrame(res_euler["historial"]), use_container_width=True)

            with c_e2:
                st.markdown("#### Resultados: Euler Modificado (Heun)")
                st.metric("y(x_n) final", f"{res_heun['y'][-1]:.6f}")
                st.dataframe(pd.DataFrame(res_heun["historial"]), use_container_width=True)

            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=res_euler["x"], y=res_euler["y"], mode="lines+markers", name="Euler", line=dict(color="#ef553b")))
            fig_e.add_trace(go.Scatter(x=res_heun["x"], y=res_heun["y"], mode="lines+markers", name="Heun (Euler Modificado)", line=dict(color="#00cc96")))
            fig_e.update_layout(title="Trayectorias: Euler vs Heun", xaxis_title="x", yaxis_title="y(x)")
            st.plotly_chart(fig_e, use_container_width=True)
        except Exception as e:
            st.error(f"Error en Euler/Heun: {e}")

    # ==========================================================================
    # PESTAÑA 2: RUNGE-KUTTA (RK2 Y RK4)
    # ==========================================================================
    with t2:
        st.subheader("2. Métodos de Runge-Kutta: RK2 y RK4 Clásico")
        st.latex(
            r"y_{k+1} = y_k + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4), \quad \begin{cases} k_1 = f(x_k, y_k) \\ k_2 = f\left(x_k + \frac{h}{2}, y_k + \frac{h}{2} k_1\right) \\ k_3 = f\left(x_k + \frac{h}{2}, y_k + \frac{h}{2} k_2\right) \\ k_4 = f(x_k + h, y_k + h k_3) \end{cases}"
        )

        try:
            res_rk2 = metodo_runge_kutta_2(fn_edo, x0=x0_val, y0=y0_val, h=h_val, n_pasos=int(n_pasos))
            res_rk4 = metodo_runge_kutta_4(fn_edo, x0=x0_val, y0=y0_val, h=h_val, n_pasos=int(n_pasos))

            mr1, mr2 = st.columns(2)
            mr1.metric("y(x_n) RK2", f"{res_rk2['y'][-1]:.6f}")
            mr2.metric("y(x_n) RK4 (Orden 4)", f"{res_rk4['y'][-1]:.6f}")

            st.markdown("#### Historial Paso a Paso de Runge-Kutta 4 (Pendientes k₁, k₂, k₃, k₄)")
            st.dataframe(pd.DataFrame(res_rk4["historial"]), use_container_width=True)

            fig_rk = go.Figure()
            fig_rk.add_trace(go.Scatter(x=res_rk2["x"], y=res_rk2["y"], mode="lines+markers", name="RK2", line=dict(color="#ab63fa")))
            fig_rk.add_trace(go.Scatter(x=res_rk4["x"], y=res_rk4["y"], mode="lines+markers", name="RK4", line=dict(color="#1f77b4", width=3)))
            fig_rk.update_layout(title="Convergencia de Runge-Kutta 2 vs Runge-Kutta 4", xaxis_title="x", yaxis_title="y(x)")
            st.plotly_chart(fig_rk, use_container_width=True)
        except Exception as e:
            st.error(f"Error en Runge-Kutta: {e}")

    # ==========================================================================
    # PESTAÑA 3: PREDICTOR-CORRECTOR DE MILNE
    # ==========================================================================
    with t3:
        st.subheader("3. Método Predictor-Corrector Multipaso de Milne (Orden 4)")
        st.latex(
            r"y_{k+1}^{(0)} = y_{k-3} + \frac{4h}{3} \left[ 2 f_{k-2} - f_{k-1} + 2 f_k \right] \quad \text{(Fórmula Predictora)}"
        )
        st.latex(
            r"y_{k+1}^{(m+1)} = y_{k-1} + \frac{h}{3} \left[ f_{k-1} + 4 f_k + f\left(x_{k+1}, y_{k+1}^{(m)}\right) \right] \quad \text{(Fórmula Correctora)}"
        )
        st.latex(
            r"R_M \approx \frac{1}{29} \left| y_{k+1}^{(\text{corrector})} - y_{k+1}^{(\text{predictor})} \right|"
        )

        st.info("ℹ️ Milne requiere 4 puntos iniciales independientes, calculados automáticamente con Runge-Kutta 4.")

        try:
            res_milne = metodo_milne(fn_edo, x0=x0_val, y0=y0_val, h=h_val, n_pasos=int(n_pasos), tolerancia_corrector=1e-6)
            st.metric("y(x_n) Milne Final", f"{res_milne['y'][-1]:.6f}")

            st.markdown("#### Historial Completo de Pasos (Arranque RK4 + Predicción + Corrección)")
            st.dataframe(pd.DataFrame(res_milne["historial"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Método de Milne: {e}")

    # ==========================================================================
    # PESTAÑA 4: COMPARATIVA MULTIMÉTODO
    # ==========================================================================
    with t4:
        st.subheader("4. Banco de Pruebas: Comparativa Simultánea de Métodos")
        st.markdown(
            "Permite contrastar todos los algoritmos en una misma gráfica. Opcionalmente ingrese la solución analítica exacta."
        )

        sol_exacta_str = st.text_input(
            "Solución Exacta y(x) analítica (opcional, para calcular error real):",
            value="2*exp(x) - x - 1",
            help="Para dy/dx = x + y con y(0) = 1, la solución exacta es 2*exp(x) - x - 1.",
            key="v7_sol_exacta",
        )

        fn_exacta = None
        if sol_exacta_str.strip():
            try:
                x_sym = sp.Symbol("x")
                expr_ex = sp.sympify(sol_exacta_str.replace("^", "**"))
                fn_exacta = sp.lambdify(x_sym, expr_ex, modules=["numpy", "math"])
                st.latex(f"y_{{\\text{{exacta}}}}(x) = {sp.latex(expr_ex)}")
            except Exception as e:
                st.warning(f"No se pudo compilar la solución analítica: {e}")

        try:
            res_comp = comparar_metodos_edo(fn_edo, x0=x0_val, y0=y0_val, h=h_val, n_pasos=int(n_pasos), solucion_exacta_fn=fn_exacta)

            st.markdown("#### Tabla Comparativa Paso a Paso")
            st.dataframe(pd.DataFrame(res_comp["tabla_comparativa"]), use_container_width=True)

            fig_all = go.Figure()
            colores = {
                "euler": "#ef553b",
                "heun": "#00cc96",
                "rk2": "#ab63fa",
                "rk4": "#1f77b4",
                "milne": "#ff7f0e",
            }
            nombres = {
                "euler": "Euler",
                "heun": "Euler Modificado (Heun)",
                "rk2": "Runge-Kutta 2",
                "rk4": "Runge-Kutta 4",
                "milne": "Milne (P-C)",
            }

            for metodo_clave, datos in res_comp["metodos"].items():
                fig_all.add_trace(go.Scatter(
                    x=datos["x"],
                    y=datos["y"],
                    mode="lines+markers",
                    name=nombres.get(metodo_clave, metodo_clave),
                    line=dict(color=colores.get(metodo_clave, "#333333")),
                ))

            if fn_exacta is not None:
                x_fine = np.linspace(x0_val, x0_val + n_pasos * h_val, 100)
                y_fine = [fn_exacta(xi) for xi in x_fine]
                fig_all.add_trace(go.Scatter(
                    x=x_fine,
                    y=y_fine,
                    mode="lines",
                    name="Solución Exacta Analítica",
                    line=dict(color="#000000", width=3, dash="dash"),
                ))

            fig_all.update_layout(title="Comparativa de Trayectorias Numéricas en EDO", xaxis_title="x", yaxis_title="y(x)")
            st.plotly_chart(fig_all, use_container_width=True)
        except Exception as e:
            st.error(f"Error en Comparativa: {e}")


vista_lab7()
