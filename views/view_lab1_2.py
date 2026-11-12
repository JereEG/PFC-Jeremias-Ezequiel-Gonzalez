"""
views/view_lab1_2.py
===================
Vista interactiva para los Laboratorios 1 y 2:
- Lab 1: Modelos algorítmicos, tiempos CPU/reloj, Fibonacci, cuadrática y series de Taylor (e y pi).
- Lab 2: Errores absolutos/relativos, aritmética finita (k dígitos), asociatividad, cancelación catastrófica y Horner.
"""

import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from core.lab1_2_teoria import (
    sumar_n_positivos,
    sumar_pares,
    generar_fibonacci,
    resolver_cuadratica,
    calcular_serie_euler,
    calcular_pi_leibniz_tolerancia,
    calcular_pi_leibniz_historial,
    conteo_terminos_serie_impar,
    calcular_errores,
    obtener_casos_estudio_errores,
    operar_en_maquina,
    casos_estudio_asociatividad,
    analizar_cancelacion_cuadratica,
    evaluar_polinomio_horner,
)


def vista_lab1_2():
    st.title("🔬 Módulo 1 y 2: Algoritmos, Series y Teoría de Errores")
    st.caption("FaCENA - UNNE &bull; Estudio de complejidad algorítmica, convergencia analítica y estabilidad en precisión finita.")
    st.markdown("---")

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "➕ 1. Sumas y Tiempos",
        "🌱 2. Fibonacci y Áureo",
        "♾️ 3. Series de e y π",
        "📏 4. Teoría de Errores",
        "🖥️ 5. Aritmética de Máquina",
        "⚠️ 6. Cancelación Catastrófica",
        "📉 7. Esquema de Horner",
    ])

    # ==========================================================================
    # PESTAÑA 1: SUMAS Y TIEMPOS
    # ==========================================================================
    with t1:
        st.subheader("1. Suma de $N$ Números y Rendimiento (Reloj vs. CPU)")
        st.markdown(
            """
            Contraste entre **Tiempo de Reloj** (*Wall-Clock Time*) y **Tiempo de Procesador (CPU)** 
            al acumular grandes volúmenes de datos.
            """
        )
        st.latex(r"S = \sum_{i=1}^{N} X_i \quad \text{con } X_i > 0")

        c1, c2 = st.columns(2)
        with c1:
            n_sum = st.slider("Cantidad de números N a sumar:", 100, 10000, 2000, step=500, key="v1_nsum")
        with c2:
            tipo_suma = st.radio("Operación:", ["Suma de todos los positivos", "Suma condicional de pares"], horizontal=True, key="v1_tsuma")

        if tipo_suma == "Suma de todos los positivos":
            import random
            rng = random.Random(42)
            datos = [round(rng.uniform(0.1, 100.0), 3) for _ in range(n_sum)]
            res = sumar_n_positivos(datos)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Datos Sumados (N)", f"{res['n']:,}")
            m2.metric("Suma Total S", f"{res['total']:,.3f}")
            m3.metric("Tiempo de Reloj", f"{res['t_clock'] * 1e6:.2f} µs")
            m4.metric("Tiempo de CPU", f"{res['t_cpu'] * 1e6:.2f} µs")

            if res["muestra_acumulada"]:
                st.caption("Muestra de los primeros términos acumulados:")
                st.dataframe(pd.DataFrame(res["muestra_acumulada"]), use_container_width=True)
        else:
            import random
            rng = random.Random(42)
            datos_ent = [rng.randint(1, 200) for _ in range(n_sum)]
            res_p = sumar_pares(datos_ent)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Evaluados (N)", f"{res_p['total_evaluados']:,}")
            m2.metric("Cantidad de Pares", f"{res_p['cant_pares']:,} ({res_p['porcentaje_pares']:.1f}%)")
            m3.metric("Suma de Pares", f"{res_p['suma_pares']:,}")
            m4.metric("Tiempo de Reloj", f"{res_p['t_clock'] * 1e6:.2f} µs")

    # ==========================================================================
    # PESTAÑA 2: FIBONACCI Y RAZÓN ÁUREA
    # ==========================================================================
    with t2:
        st.subheader("2. Sucesión de Fibonacci y Convergencia a la Razón Áurea")
        st.latex(
            r"F_n = F_{n-1} + F_{n-2}, \quad \lim_{n \to \infty} \frac{F_n}{F_{n-1}} = \varphi = \frac{1 + \sqrt{5}}{2} \approx 1.6180339887\dots"
        )
        n_fib = st.slider("Términos N a generar (hasta N = 200 sin desbordamiento):", 5, 200, 50, step=5, key="v1_nfib")
        res_fib = generar_fibonacci(n_fib)

        mf1, mf2, mf3 = st.columns(3)
        mf1.metric("Términos Calculados", f"{len(res_fib['terminos'])}")
        mf2.metric("Cifras de F_N", f"{res_fib['cifras_ultimo']} dígitos")
        mf3.metric("Tiempo de Cómputo", f"{res_fib['t_clock'] * 1e6:.2f} µs")

        st.code(f"F_{n_fib} = {res_fib['ultimo_termino']}")

        df_fib = pd.DataFrame(res_fib["tabla"])
        st.dataframe(df_fib, use_container_width=True)

        # Gráfico de convergencia del cociente a phi
        df_plot = df_fib[df_fib["cociente"].notnull()].head(35)
        fig_fib = go.Figure()
        fig_fib.add_trace(go.Scatter(
            x=df_plot["orden"],
            y=df_plot["cociente"],
            mode="lines+markers",
            name="Cociente F_n / F_{n-1}",
            line=dict(color="#1f77b4", width=2.5),
        ))
        fig_fib.add_hline(
            y=res_fib["phi_exacto"],
            line_dash="dash",
            line_color="#2ca02c",
            annotation_text=f"Razón Áurea φ ≈ {res_fib['phi_exacto']:.8f}",
        )
        fig_fib.update_layout(
            title="Oscilación Amortiguada del Cociente hacia el Número Áureo φ",
            xaxis_title="Orden n",
            yaxis_title="Cociente F_n / F_{n-1}",
            hovermode="x unified",
        )
        st.plotly_chart(fig_fib, use_container_width=True)

    # ==========================================================================
    # PESTAÑA 3: SERIES DE TAYLOR (e y pi)
    # ==========================================================================
    with t3:
        st.subheader("3. Aproximación de Constantes y Funciones por Series de Taylor")
        sub_e, sub_pi, sub_imp = st.tabs(["Exponencial e^x", "Número π (Leibniz)", "Conteo de Términos"])

        with sub_e:
            st.latex(r"e^x = \sum_{k=0}^{N} \frac{x^k}{k!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots")
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                n_e = st.slider("Términos N:", 1, 30, 15, key="v1_ne")
            with c_e2:
                x_e = st.number_input("Exponente x (use x = 1 para e):", value=1.0, step=0.5, key="v1_xe")

            res_e = calcular_serie_euler(n_e, x_e)
            me1, me2, me3 = st.columns(3)
            me1.metric("Aproximación S_N (float)", f"{res_e['aproximacion_float']:.15f}")
            me2.metric("Valor Exacto", f"{res_e['valor_exacto']:.15f}")
            me3.metric("Error Absoluto Final", f"{res_e['error_absoluto_final']:.4e}")

            st.caption(f"🔬 **Precisión Arbitraria Decimal (50 dígitos):** `{res_e['aproximacion_decimal_str']}`")
            st.dataframe(pd.DataFrame(res_e["historial"]), use_container_width=True)

        with sub_pi:
            st.latex(r"\pi = 4 \sum_{k=0}^{\infty} \frac{(-1)^k}{2k + 1}, \qquad \text{Criterio Scarborough: } E_a < 0.5 \times 10^{-d}")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                dec_solic = st.slider("Cifras decimales exactas deseadas:", 1, 5, 3, key="v1_pidec")
            with c_p2:
                res_tol = calcular_pi_leibniz_tolerancia(dec_solic)
                st.metric("Términos Requeridos", f"{res_tol['terminos_utilizados']:,}")
                st.metric("Aproximación de π", f"{res_tol['pi_aproximado']:.8f}")

            res_hist = calcular_pi_leibniz_historial(500)
            df_pi = pd.DataFrame(res_hist["historial"])
            fig_pi = go.Figure()
            fig_pi.add_trace(go.Scatter(x=df_pi["k"], y=df_pi["pi_aprox"], mode="lines", name="S_k (Leibniz)", line=dict(color="#9467bd")))
            fig_pi.add_hline(y=math.pi, line_dash="dash", line_color="#2ca02c", annotation_text="π exacto")
            fig_pi.update_layout(title="Convergencia Lenta y Oscilante de Gregory-Leibniz", xaxis_title="Término k", yaxis_title="π aprox")
            st.plotly_chart(fig_pi, use_container_width=True)

        with sub_imp:
            st.markdown(r"Deducción de cantidad de términos para series con denominadores impares: $1 + 1/3 + \dots + 1/(2i+1)$.")
            den_imp = st.number_input("Denominador del último término:", min_value=1, value=17, step=2, key="v1_denimp")
            c_res = conteo_terminos_serie_impar(int(den_imp))
            st.success(f"Para denominador {den_imp}, el índice es $i = {c_res['indice_final_i']}$ y la serie posee **$N = {c_res['n_terminos']}$ términos**.")

    # ==========================================================================
    # PESTAÑA 4: TEORÍA DE ERRORES
    # ==========================================================================
    with t4:
        st.subheader("4. Errores Absolutos, Relativos y Cifras Significativas")
        st.latex(r"E_a = |p - p^*|, \quad E_r = \frac{|p - p^*|}{|p|}, \quad E_r \le 0.5 \times 10^{1-t}")

        st.markdown("#### Calculadora Interactiva de Error")
        ce1, ce2 = st.columns(2)
        with ce1:
            val_p = st.number_input("Valor Exacto p:", value=math.pi, format="%.8f", key="v1_pe")
        with ce2:
            val_ps = st.number_input("Aproximación p*:", value=22.0 / 7.0, format="%.8f", key="v1_pa")

        info_calc = calcular_errores(val_p, val_ps)
        me1, me2, me3, me4 = st.columns(4)
        me1.metric("Error Absoluto (E_a)", f"{info_calc['error_absoluto']:.6e}")
        me2.metric("Error Relativo (E_r)", f"{info_calc['error_relativo']:.6e}")
        me3.metric("Error Porcentual", f"{info_calc['error_porcentual']:.4f}%")
        me4.metric("Cifras Significativas", f"{info_calc['cifras_significativas']} dígitos")

        st.markdown("#### Casos de Estudio Oficiales de la Cátedra (TP 2)")
        casos_tp = obtener_casos_estudio_errores()
        st.dataframe(pd.DataFrame(casos_tp), use_container_width=True)

    # ==========================================================================
    # PESTAÑA 5: ARITMÉTICA DE MÁQUINA Y ASOCIATIVIDAD
    # ==========================================================================
    with t5:
        st.subheader("5. Aritmética de Punto Flotante Decimal de $k$ Dígitos y Asociatividad")
        st.latex(r"fl(x) = \pm 0.d_1 d_2 \dots d_k \times 10^n")

        k_maq = st.slider("Cifras significativas de la máquina (k):", 2, 6, 3, key="v1_kmaq")
        st.markdown(f"#### Demostración de Pérdida de Asociatividad con $k = {k_maq}$ dígitos")
        asoc_res = casos_estudio_asociatividad(k_maq)
        st.dataframe(pd.DataFrame(asoc_res["tabla"]), use_container_width=True)
        st.warning(asoc_res["analisis_asociatividad"]["conclusion"])

        st.markdown("#### Calculadora Aritmética en Máquina")
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            op_x = st.number_input("Operando x:", value=121.0, format="%.4f", key="v1_opx")
        with ca2:
            op_simb = st.selectbox("Operación:", ["+", "-", "*", "/"], key="v1_opsimb")
        with ca3:
            op_y = st.number_input("Operando y:", value=0.327, format="%.4f", key="v1_opy")

        res_op = operar_en_maquina(op_x, op_y, op_simb, k_maq)
        mo1, mo2, mo3 = st.columns(3)
        mo1.metric("Valor Exacto", f"{res_op['exacto']:.6g}")
        mo2.metric(f"Redondeo ({k_maq} dígitos)", res_op["redondeo"]["norm_str"], f"Err: {res_op['redondeo']['error_relativo_pct']:.2f}%")
        mo3.metric(f"Truncamiento ({k_maq} dígitos)", res_op["truncamiento"]["norm_str"], f"Err: {res_op['truncamiento']['error_relativo_pct']:.2f}%")

    # ==========================================================================
    # PESTAÑA 6: CANCELACIÓN CATASTRÓFICA
    # ==========================================================================
    with t6:
        st.subheader("6. Estabilidad Numérica y Cancelación Catastrófica en la Ecuación Cuadrática")
        st.latex(
            r"x^2 + 62.10x + 1 = 0 \quad (b^2 \gg 4ac) \implies \sqrt{b^2 - 4ac} \approx |b|"
        )
        st.markdown(
            """
            En la fórmula tradicional para $x_1$, restar cantidades casi idénticas en el numerador provoca 
            **cancelación catastrófica**. La **fórmula racionalizada** evita este defecto:
            """
        )
        st.latex(r"x_1^{\text{trad}} = \frac{-b + \sqrt{b^2 - 4ac}}{2a}, \qquad x_1^{\text{rat}} = \frac{-2c}{b + \sqrt{b^2 - 4ac}}")

        kc_quad = st.slider("Dígitos de máquina (k) para simulación:", 3, 6, 4, key="v1_kcquad")
        res_canc = analizar_cancelacion_cuadratica(1.0, 62.10, 1.0, kc_quad)

        st.dataframe(pd.DataFrame(res_canc["tabla_comparativa"]), use_container_width=True)

        # Gráfico comparativo de error
        df_canc = pd.DataFrame(res_canc["tabla_comparativa"])
        fig_canc = go.Figure()
        fig_canc.add_trace(go.Bar(
            x=df_canc["metodo"],
            y=df_canc["error_relativo_pct"],
            marker_color=["#ef553b", "#00cc96", "#00cc96", "#ef553b"],
            text=[f"{v:.4f}%" for v in df_canc["error_relativo_pct"]],
            textposition="auto",
        ))
        fig_canc.update_layout(title="Error Relativo (%) Tradicional vs. Racionalizada", yaxis_title="Error Relativo (%)")
        st.plotly_chart(fig_canc, use_container_width=True)

    # ==========================================================================
    # PESTAÑA 7: ESQUEMA DE HORNER
    # ==========================================================================
    with t7:
        st.subheader("7. Evaluación Polinómica: Forma Estándar vs. Forma Anidada de Horner")
        st.latex(r"P(x) = x^3 - 6x^2 + 3x - 0.149 = \big((x - 6)x + 3\big)x - 0.149")
        st.markdown("La forma de Horner reduce multiplicaciones y atenúa la acumulación de errores de redondeo.")

        ch1, ch2 = st.columns(2)
        with ch1:
            xh_val = st.number_input("Punto x a evaluar:", value=4.71, step=0.01, format="%.4f", key="v1_xh")
        with ch2:
            kh_val = st.slider("Cifras significativas (k):", 2, 6, 3, key="v1_kh")

        res_h = evaluar_polinomio_horner(xh_val, kh_val)

        st.markdown("#### Términos de Potencias en Forma Estándar")
        st.dataframe(pd.DataFrame(res_h["tabla_terminos"]), use_container_width=True)

        st.markdown("#### Pasos de Evaluación en Forma Anidada de Horner")
        st.dataframe(pd.DataFrame(res_h["tabla_horner_pasos"]), use_container_width=True)

        st.markdown("#### Comparación Final de Errores")
        st.dataframe(pd.DataFrame(res_h["tabla_errores"]), use_container_width=True)


vista_lab1_2()

