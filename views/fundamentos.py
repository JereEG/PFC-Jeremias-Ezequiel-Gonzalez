"""
views/fundamentos.py
====================
Categoría: Fundamentos y Teoría de Errores.
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

from core.lab1_2_teoria import (
    analizar_cancelacion_cuadratica,
    calcular_errores,
    calcular_pi_leibniz_historial,
    calcular_pi_leibniz_tolerancia,
    calcular_serie_euler,
    casos_estudio_asociatividad,
    evaluar_polinomio_horner,
    generar_fibonacci,
    obtener_casos_estudio_errores,
    operar_en_maquina,
)
from views.ui_helpers import bloque_material_estudio


def vista_fundamentos():
    st.title("🧮 Fundamentos Matemáticos, Aritmética de Máquina y Errores")
    st.markdown(
        """
        Esta sección aborda la representación numérica en computadoras, los límites de la aritmética
        de punto flotante finita, la propagación de errores y la aproximación por series de Taylor.
        """
    )

    t1, t2, t3, t4, t5 = st.tabs([
        "1. Series de Taylor y Criterio de Scarborough",
        "2. Aritmética de Máquina (k dígitos)",
        "3. Cancelación Catastrófica",
        "4. Esquema de Horner",
        "5. Sucesión de Fibonacci y Razón Áurea",
    ])

    # ==========================================================================
    # MÉTODO 1: SERIES DE TAYLOR Y CRITERIO DE SCARBOROUGH
    # ==========================================================================
    with t1:
        st.subheader("1. Series de Taylor y Convergencia por Criterio de Scarborough")
        st.latex(r"e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}, \qquad \frac{\pi}{4} = \sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1}")
        st.latex(r"\varepsilon_s = \left( 0.5 \times 10^{2-n} \right)\% \quad \text{(Criterio de Scarborough para } n \text{ cifras seguras)}")

        bloque_material_estudio(
            pdf_teoria="01_teoria_series_taylor_scarborough.pdf",
            pdf_practica="01_practica_series_taylor_ejercicio.pdf",
            titulo_metodo="Series de Taylor y Criterio de Scarborough",
        )

        st.markdown("#### 🛠️ Calculadora Interactiva: Serie de Euler e^x")
        c1, c2 = st.columns(2)
        with c1:
            x_val = st.number_input("Valor de x en e^x:", value=1.0, step=0.1, key="f_x_val")
        with c2:
            max_term = st.slider("Términos máximos de la serie (n):", min_value=3, max_value=50, value=15, key="f_max_term")

        res_euler = calcular_serie_euler(n_max=int(max_term), x_val=x_val)
        m1, m2, m3 = st.columns(3)
        m1.metric("Valor Aproximado", f"{res_euler['aproximacion_float']:.8f}")
        m2.metric("Términos Evaluados", res_euler["n_max"])
        m3.metric("Error Relativo %", f"{res_euler['error_relativo_final_pct']:.2e}%")

        st.markdown("#### Tabla de Convergencia Paso a Paso")
        df_euler = pd.DataFrame(res_euler["historial"])
        st.dataframe(df_euler, use_container_width=True)

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=df_euler["k"],
            y=df_euler["error_relativo_pct"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            name="Error Relativo %",
        ))
        fig_e.update_layout(title="Convergencia del Error Relativo en la Serie de Taylor", xaxis_title="Término (k)", yaxis_title="Error %", yaxis_type="log")
        st.plotly_chart(fig_e, use_container_width=True)

    # ==========================================================================
    # MÉTODO 2: ARITMÉTICA DE MÁQUINA DE K DÍGITOS
    # ==========================================================================
    with t2:
        st.subheader("2. Aritmética de Punto Flotante Normalizada de k Dígitos")
        st.latex(r"x = \pm 0.d_1 d_2 \dots d_k \times 10^n, \quad d_1 \neq 0")
        st.markdown(
            "Compara las operaciones elementales simulando una máquina de precisión finita de $k$ dígitos "
            "bajo modo de **redondeo simétrico** vs. **truncamiento/corte**."
        )

        bloque_material_estudio(
            pdf_teoria="02_teoria_aritmetica_punto_flotante.pdf",
            pdf_practica="02_practica_perdida_asociatividad.pdf",
            titulo_metodo="Aritmética de Máquina y Asociatividad",
        )

        st.markdown("#### 🛠️ Workspace: Operaciones Aritméticas Finita")
        ca1, ca2, ca3, ca4 = st.columns(4)
        with ca1:
            u_val = st.number_input("Operando u:", value=4.0/5.0, format="%.6f", key="f_u_val")
        with ca2:
            v_val = st.number_input("Operando v:", value=1.0/3.0, format="%.6f", key="f_v_val")
        with ca3:
            k_digits = st.slider("Cifras mantisa (k):", min_value=2, max_value=8, value=3, key="f_k_digits")
        with ca4:
            modo_op = st.selectbox("Modo de representación:", ["redondeo", "truncamiento"], key="f_modo_op")

        tabla_ops = []
        for operacion in ["+", "-", "*", "/"]:
            info_op = operar_en_maquina(u_val, v_val, operacion=operacion, k=k_digits)
            datos_modo = info_op[modo_op]
            tabla_ops.append({
                "Operación": f"u {operacion} v",
                "Resultado Máquina": datos_modo["norm_str"],
                "Valor Numérico": datos_modo["valor"],
                "Resultado Exacto": info_op["exacto"],
                "Error Absoluto": datos_modo["error_absoluto"],
                "Error Relativo %": datos_modo["error_relativo_pct"],
            })
        st.dataframe(pd.DataFrame(tabla_ops), use_container_width=True)

        st.markdown("#### Pérdida de Asociatividad: Casos Testigo de Cátedra")
        casos_asoc = casos_estudio_asociatividad()
        st.dataframe(pd.DataFrame(casos_asoc["tabla"]), use_container_width=True)
        an = casos_asoc["analisis_asociatividad"]
        st.info(f"💡 **Análisis de Asociatividad:** {an['conclusion']}")

    # ==========================================================================
    # MÉTODO 3: CANCELACIÓN CATASTRÓFICA
    # ==========================================================================
    with t3:
        st.subheader("3. Cancelación Catastrófica y Racionalización")
        st.latex(r"x_1 = \frac{-b + \sqrt{b^2 - 4ac}}{2a} \quad \xrightarrow{\text{racionalización}} \quad x_1 = \frac{-2c}{b + \sqrt{b^2 - 4ac}}")
        st.markdown(
            "Cuando $b^2 \gg 4ac$, la resta en el numerador cancela los dígitos significativos más relevantes, "
            "produciendo errores catastróficos que se evitan racionalizando el numerador."
        )

        bloque_material_estudio(
            pdf_teoria="03_teoria_cancelacion_catastrofica.pdf",
            pdf_practica="03_practica_cuadratica_racionalizada.pdf",
            titulo_metodo="Cancelación Catastrófica en la Ecuación Cuadrática",
        )

        st.markdown("#### 🛠️ Workspace: Ecuación Testigo ax² + bx + c = 0")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            a_c = st.number_input("Coeficiente a:", value=1.0, key="f_a_c")
        with cc2:
            b_c = st.number_input("Coeficiente b:", value=62.10, key="f_b_c")
        with cc3:
            c_c = st.number_input("Coeficiente c:", value=1.0, key="f_c_c")
        with cc4:
            k_c = st.slider("Cifras significativas (k):", 2, 6, 3, key="f_k_c")

        res_canc = analizar_cancelacion_cuadratica(a_c, b_c, c_c, k=k_c)
        df_canc = pd.DataFrame(res_canc["tabla_comparativa"])
        st.dataframe(df_canc, use_container_width=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_canc["metodo"],
            y=df_canc["error_relativo_pct"],
            marker_color=["#ef553b", "#00cc96", "#00cc96", "#ef553b"],
            text=[f"{val:.3f}%" if not math.isnan(val) else "N/A (Div 0)" for val in df_canc["error_relativo_pct"]],
            textposition="auto",
        ))
        fig_bar.update_layout(title="Error Relativo (%) Tradicional vs. Racionalizada", yaxis_title="Error Relativo (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================================================
    # MÉTODO 4: ESQUEMA DE HORNER
    # ==========================================================================
    with t4:
        st.subheader("4. Esquema Anidado de Horner")
        st.latex(r"P(x) = a_n x^n + \dots + a_1 x + a_0 = \Big( \dots \big( a_n x + a_{n-1} \big) x + \dots \Big) x + a_0")
        st.markdown(
            "La forma anidada reduce la cantidad de multiplicaciones de $O(n^2)$ a $O(n)$, "
            "minimizando drásticamente el error de redondeo acumulado."
        )

        bloque_material_estudio(
            pdf_teoria="04_teoria_algoritmo_horner.pdf",
            pdf_practica="04_practica_horner_polinomio_cubico.pdf",
            titulo_metodo="Evaluación Polinómica con Esquema de Horner",
        )

        st.markdown("#### 🛠️ Workspace: Polinomio P(x) = x³ - 6x² + 3x - 0.149")
        ch1, ch2 = st.columns(2)
        with ch1:
            x_h = st.number_input("Punto x a evaluar:", value=4.71, step=0.01, format="%.4f", key="f_x_h")
        with ch2:
            k_h = st.slider("Cifras de máquina (k):", 2, 6, 3, key="f_k_h")

        res_h = evaluar_polinomio_horner(x_h, k_h)

        st.markdown("##### Pasos en Forma Anidada de Horner")
        st.dataframe(pd.DataFrame(res_h["tabla_horner_pasos"]), use_container_width=True)

        st.markdown("##### Comparativa de Precisión")
        st.dataframe(pd.DataFrame(res_h["tabla_errores"]), use_container_width=True)

    # ==========================================================================
    # MÉTODO 5: SUCESIÓN DE FIBONACCI Y RAZÓN ÁUREA
    # ==========================================================================
    with t5:
        st.subheader("5. Sucesiones Recurrentes: Fibonacci y Razón Áurea")
        st.latex(r"F_n = F_{n-1} + F_{n-2}, \qquad \lim_{n \to \infty} \frac{F_n}{F_{n-1}} = \varphi = \frac{1 + \sqrt{5}}{2} \approx 1.6180339887\dots")

        bloque_material_estudio(
            pdf_teoria="05_teoria_sucesiones_recurrentes_fibonacci.pdf",
            pdf_practica="05_practica_fibonacci_razon_aurea.pdf",
            titulo_metodo="Sucesiones de Fibonacci y Razón Áurea",
        )

        st.markdown("#### 🛠️ Workspace: Generación de Términos (N hasta 200 sin desbordamiento)")
        n_fib = st.slider("Cantidad de términos a calcular (N):", min_value=5, max_value=200, value=25, key="f_n_fib")
        res_fib = generar_fibonacci(n_fib)

        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            st.metric("Razón Áurea Estimada (F_n / F_{n-1})", f"{res_fib['tabla'][-1]['cociente']:.10f}")
        with col_fb2:
            st.metric("Valor Teórico Exacto (φ)", f"{res_fib['phi_exacto']:.10f}")

        st.dataframe(pd.DataFrame(res_fib["tabla"]), use_container_width=True)


vista_fundamentos()
