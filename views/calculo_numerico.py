"""
views/calculo_numerico.py
=========================
Categoría: Integración y Diferenciación Numérica.
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

from core.lab6_integracion import (
    calcular_simpson_1_3,
    calcular_simpson_3_8,
    calcular_simpson_combinado,
    calcular_trapecio_compuesto,
    cota_error_simpson_1_3,
    cota_error_trapecio,
    derivada_newton_gregory_ascendente,
    derivada_newton_gregory_descendente,
    integrar_funcion_simbolica,
)
from views.ui_helpers import bloque_material_estudio


def vista_calculo_numerico():
    st.title("∬ Integración y Diferenciación Numérica")
    st.markdown(
        """
        Esta sección presenta los esquemas de cálculo infinitesimal computacional:
        diferenciación numérica por operadores de diferencias finitas y cuadraturas numéricas
        de Newton-Cotes simples y compuestas con sus correspondientes cotas de error.
        """
    )

    t1, t2, t3, t4 = st.tabs([
        "1. Diferenciación Numérica",
        "2. Regla del Trapecio",
        "3. Reglas de Simpson",
        "4. Cuadratura Simbólica vs Numérica",
    ])

    x_def = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_def = [0.0, 0.2474, 0.4794, 0.6816, 0.8415]

    # ==========================================================================
    # MÉTODO 1: DIFERENCIACIÓN NUMÉRICA
    # ==========================================================================
    with t1:
        st.subheader("1. Derivación Numérica por Diferencias Finitas (Newton-Gregory)")
        st.latex(r"f'(x_0) \approx \frac{1}{h} \left( \Delta y_0 - \frac{1}{2}\Delta^2 y_0 + \frac{1}{3}\Delta^3 y_0 - \dots \right), \quad f''(x_0) \approx \frac{1}{h^2} \left( \Delta^2 y_0 - \Delta^3 y_0 + \frac{11}{12}\Delta^4 y_0 - \dots \right)")

        bloque_material_estudio(
            pdf_teoria="23_teoria_diferenciacion_newton_gregory.pdf",
            pdf_practica="23_practica_derivadas_progresivas_regresivas.pdf",
            titulo_metodo="Diferenciación Numérica",
        )

        st.markdown("#### 🛠️ Workspace: Diferenciación Numérica")
        df_pts = pd.DataFrame({"x": [1.0, 1.1, 1.2, 1.3, 1.4], "y": [0.8415, 0.8912, 0.9320, 0.9636, 0.9854]})
        ed_pts = st.data_editor(df_pts, key="calc_ed_diff")
        xd = ed_pts["x"].tolist()
        yd = ed_pts["y"].tolist()

        cd1, cd2 = st.columns(2)
        with cd1:
            esquema = st.radio("Esquema:", ["Progresivo (Ascendente / Inicio)", "Regresivo (Descendente / Final)"], key="calc_esq_diff")
        with cd2:
            orden_d = st.selectbox("Orden de la derivada:", [1, 2], key="calc_orden_diff")

        try:
            if "Progresivo" in esquema:
                res_d = derivada_newton_gregory_ascendente(xd, yd, idx_punto=0, orden_derivada=orden_d)
            else:
                res_d = derivada_newton_gregory_descendente(xd, yd, idx_punto=-1, orden_derivada=orden_d)

            m_d1, m_d2, m_d3 = st.columns(3)
            m_d1.metric(f"f^{{({orden_d})}}(x) estimada", f"{res_d['derivada']:.6f}")
            m_d2.metric("Punto evaluado x", f"{res_d['x_eval']:.4f}")
            m_d3.metric("Paso equiespaciado h", f"{res_d['h']:.4f}")

            st.markdown("##### Desglose de Términos de la Serie")
            st.dataframe(pd.DataFrame(res_d["terminos_detalle"]), use_container_width=True)

            st.markdown("##### Tabla de Diferencias Finitas")
            st.dataframe(pd.DataFrame(res_d["tabla_diferencias"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error en Derivación: {e}")

    # ==========================================================================
    # MÉTODO 2: REGLA DEL TRAPECIO COMPUESTO
    # ==========================================================================
    with t2:
        st.subheader("2. Regla del Trapecio Compuesto")
        st.latex(r"I_T = \frac{h}{2} \left[ y_0 + 2 \sum_{i=1}^{n-1} y_i + y_n \right] = \frac{h}{2} (E + 2P + 2I), \qquad R_T = - \frac{(b-a) h^2}{12} f''(\xi)")

        bloque_material_estudio(
            pdf_teoria="24_teoria_regla_trapecio_compuesto.pdf",
            pdf_practica="24_practica_trapecio_cota_error.pdf",
            titulo_metodo="Regla del Trapecio Compuesto",
        )

        st.markdown("#### 🛠️ Workspace: Regla del Trapecio")
        df_trap = pd.DataFrame({"x": x_def, "y": y_def})
        ed_trap = st.data_editor(df_trap, key="calc_ed_trap")
        xt = ed_trap["x"].tolist()
        yt = ed_trap["y"].tolist()

        try:
            res_trap = calcular_trapecio_compuesto(xt, yt)
            mt1, mt2, mt3 = st.columns(3)
            mt1.metric("Integral (Área Trapecio)", f"{res_trap['area']:.6f}")
            mt2.metric("Subintervalos (n)", res_trap["n_intervalos"])
            mt3.metric("Paso h", f"{res_trap['h']:.4f}")

            st.markdown("##### Términos Cátedra: Extremos (E), Pares (P) e Impares (I)")
            c_e, c_p, c_i = st.columns(3)
            c_e.metric("Suma Extremos (E)", f"{res_trap['E']:.4f}")
            c_p.metric("Suma Pares (P)", f"{res_trap['P']:.4f}")
            c_i.metric("Suma Impares (I)", f"{res_trap['I']:.4f}")

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=xt, y=yt, mode="lines+markers", name="f(x)", line=dict(color="#1f77b4", width=2)))
            for i in range(len(xt) - 1):
                fig_t.add_trace(go.Scatter(
                    x=[xt[i], xt[i], xt[i+1], xt[i+1]],
                    y=[0, yt[i], yt[i+1], 0],
                    fill="toself",
                    fillcolor="rgba(31, 119, 180, 0.2)",
                    line=dict(color="rgba(31, 119, 180, 0.5)", width=1),
                    showlegend=(i == 0),
                    name="Trapecios",
                ))
            fig_t.update_layout(title="Aproximación por Trapecios Compuestos", xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig_t, use_container_width=True)
        except Exception as e:
            st.error(f"Error en Trapecio: {e}")

    # ==========================================================================
    # MÉTODO 3: REGLAS DE SIMPSON
    # ==========================================================================
    with t3:
        st.subheader("3. Reglas de Simpson: 1/3, 3/8 y Simpson Combinado")
        st.latex(r"I_{S1/3} = \frac{h}{3} (E + 2P + 4I), \qquad I_{S3/8} = \frac{3h}{8} \left[ y_0 + 3y_1 + 3y_2 + 2y_3 + \dots + y_n \right]")

        bloque_material_estudio(
            pdf_teoria="25_teoria_reglas_simpson_paridad.pdf",
            pdf_practica="25_practica_simpson_combinado_hibrido.pdf",
            titulo_metodo="Reglas de Simpson (1/3, 3/8 y Combinado)",
        )

        st.markdown("#### 🛠️ Workspace: Reglas de Simpson")
        tipo_simp = st.selectbox(
            "Variante de Simpson:",
            ["Simpson Combinado (Automático)", "Simpson 1/3 Compuesto", "Simpson 3/8"],
            key="calc_sel_simp",
        )

        try:
            if "Combinado" in tipo_simp:
                res_s = calcular_simpson_combinado(xt, yt)
                st.info(f"Esquema aplicado: **{res_s['metodo']}**")
            elif "1/3" in tipo_simp:
                res_s = calcular_simpson_1_3(xt, yt)
            else:
                res_s = calcular_simpson_3_8(xt, yt)

            ms1, ms2, ms3 = st.columns(3)
            ms1.metric("Integral Simpson", f"{res_s['area']:.6f}")
            ms2.metric("Subintervalos (n)", res_s["n_intervalos"])
            ms3.metric("Paso h", f"{res_s['h']:.4f}")

            if "I" in res_s:
                st.markdown("##### Desglose Cátedra")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("E (Extremos)", f"{res_s['E']:.4f}")
                sc2.metric("P (Pares)", f"{res_s['P']:.4f}")
                sc3.metric("I (Impares)", f"{res_s['I']:.4f}")
        except Exception as e:
            st.error(f"Error en Simpson: {e}")

    # ==========================================================================
    # MÉTODO 4: CUADRATURA SIMBÓLICA VS NUMÉRICA
    # ==========================================================================
    with t4:
        st.subheader("4. Banco de Pruebas: Cuadratura Numérica vs. Solución Exacta SymPy")
        st.latex(r"I_{\text{exacta}} = \int_a^b f(x) \, dx, \qquad \text{Error} = |I_{\text{exacta}} - I_{\text{num}}|")

        bloque_material_estudio(
            pdf_teoria="26_teoria_cuadratura_analitica_vs_numerica.pdf",
            pdf_practica="26_practica_comparativa_errores_cuadratura.pdf",
            titulo_metodo="Integración Analítica vs Numérica",
        )

        st.markdown("#### 🛠️ Workspace: Comparación de Métodos de Integración")
        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            f_analitica = st.text_input("Función f(x):", value="exp(x)", key="calc_f_ana")
        with ci2:
            a_int = st.number_input("Límite inferior a:", value=0.0, step=0.1, key="calc_a_int")
            b_int = st.number_input("Límite superior b:", value=1.0, step=0.1, key="calc_b_int")
        with ci3:
            n_subint = st.number_input("Cantidad de subintervalos (n):", value=6, min_value=2, max_value=200, step=2, key="calc_n_subint")

        try:
            res_comp_trap = integrar_funcion_simbolica(f_analitica, a_int, b_int, int(n_subint), metodo="trapecio")
            res_comp_s13 = integrar_funcion_simbolica(f_analitica, a_int, b_int, int(n_subint), metodo="simpson_1_3")
            res_comp_comb = integrar_funcion_simbolica(f_analitica, a_int, b_int, int(n_subint), metodo="simpson_combinado")

            st.latex(rf"\int_{{{a_int}}}^{{{b_int}}} \left( {res_comp_trap['funcion_str']} \right) dx = {res_comp_trap['integral_exacta']:.8f}")

            tabla_comparativa = [
                {
                    "Método": "Trapecio Compuesto",
                    "Integral Numérica": res_comp_trap["resultado_aproximado"],
                    "Integral Exacta": res_comp_trap["integral_exacta"],
                    "Error Absoluto": res_comp_trap["error_absoluto"],
                    "Error Relativo (%)": res_comp_trap["error_relativo_pct"],
                },
                {
                    "Método": "Simpson 1/3 Compuesto",
                    "Integral Numérica": res_comp_s13["resultado_aproximado"],
                    "Integral Exacta": res_comp_s13["integral_exacta"],
                    "Error Absoluto": res_comp_s13["error_absoluto"],
                    "Error Relativo (%)": res_comp_s13["error_relativo_pct"],
                },
                {
                    "Método": "Simpson Combinado",
                    "Integral Numérica": res_comp_comb["resultado_aproximado"],
                    "Integral Exacta": res_comp_comb["integral_exacta"],
                    "Error Absoluto": res_comp_comb["error_absoluto"],
                    "Error Relativo (%)": res_comp_comb["error_relativo_pct"],
                },
            ]
            st.dataframe(pd.DataFrame(tabla_comparativa), use_container_width=True)

            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                x=[row["Método"] for row in tabla_comparativa],
                y=[row["Error Absoluto"] for row in tabla_comparativa],
                marker_color=["#ef553b", "#00cc96", "#ab63fa"],
                text=[f"{row['Error Absoluto']:.2e}" for row in tabla_comparativa],
                textposition="auto",
            ))
            fig_err.update_layout(title=f"Error Absoluto por Cuadratura (n = {n_subint})", yaxis_title="Error Absoluto", yaxis_type="log")
            st.plotly_chart(fig_err, use_container_width=True)
        except Exception as e:
            st.error(f"Error en Integrador Simbólico: {e}")


vista_calculo_numerico()

