"""
views/historico_labs.py
=======================
Anexo Curricular: Histórico de Laboratorios de Cursada (Labs 1 al 7).
Cátedra de Métodos Computacionales — FaCENA - Universidad Nacional del Nordeste (UNNE).
Mapea los trabajos prácticos originales con los nuevos módulos del Santo Grial.
"""

from __future__ import annotations

import streamlit as st
from views.ui_helpers import bloque_material_estudio


def vista_historico_labs():
    st.title("📚 Histórico de Cursada: Laboratorios 1 al 7")
    st.markdown(
        """
        Este anexo preserva la estructura curricular original de los **7 Trabajos Prácticos de Laboratorio** 
        de la cátedra de **Métodos Computacionales (FaCENA - UNNE)**, permitiendo consultar las guías de estudio,
        enunciados de exámenes y acceder directamente al módulo matemático correspondiente en el **Santo Grial**.
        """
    )

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "Lab 1: Modelos y Series",
        "Lab 2: Teoría de Errores",
        "Lab 3: Raíces No Lineales",
        "Lab 4: Sistemas y Autovalores",
        "Lab 5: Interpolación Polinomial",
        "Lab 6: Integración y Derivación",
        "Lab 7: Ecuaciones Diferenciales",
    ])

    # ==========================================================================
    # LAB 1
    # ==========================================================================
    with t1:
        st.subheader("Laboratorio 1: Modelos Algorítmicos, Tiempos y Series de Taylor")
        st.markdown(
            """
            - **Objetivo:** Introducción a la algoritmia científica, medición de tiempos de CPU vs. Reloj y deducción de series elementales.
            - **Contenidos Oficiales:** Suma de $N$ positivos, suma de pares, Fibonacci hasta $N=200$, Razón Áurea $\\varphi$, series de $e$ y $\\pi$ (Gregory-Leibniz con criterio de Scarborough).
            """
        )
        bloque_material_estudio(
            pdf_teoria="lab1_guia_teoria_modelos_series.pdf",
            pdf_practica="lab1_practica_resueltos.pdf",
            titulo_metodo="Guía Oficial Lab 1 (FaCENA - UNNE)",
        )
        if st.button("Ir al Módulo de Fundamentos y Errores", key="btn_h_l1", type="primary"):
            st.switch_page("views/fundamentos.py")

    # ==========================================================================
    # LAB 2
    # ==========================================================================
    with t2:
        st.subheader("Laboratorio 2: Teoría de Errores y Aritmética de Máquina")
        st.markdown(
            """
            - **Objetivo:** Estudio formal de los errores numéricos, representación normalizada de punto flotante y pérdida de propiedades algebraicas.
            - **Contenidos Oficiales:** Error absoluto/relativo, cifras significativas, aritmética de $k$ dígitos, redondeo vs. corte, pérdida de asociatividad, cancelación catastrófica en raíces cuadráticas y esquema de Horner.
            """
        )
        bloque_material_estudio(
            pdf_teoria="lab2_guia_teoria_errores_maquina.pdf",
            pdf_practica="lab2_practica_resueltos.pdf",
            titulo_metodo="Guía Oficial Lab 2 (FaCENA - UNNE)",
        )
        if st.button("Ir al Módulo de Fundamentos y Errores", key="btn_h_l2", type="primary"):
            st.switch_page("views/fundamentos.py")

    # ==========================================================================
    # LAB 3
    # ==========================================================================
    with t3:
        st.subheader("Laboratorio 3: Búsqueda de Raíces de Ecuaciones No Lineales")
        st.markdown(
            """
            - **Objetivo:** Aislamiento geométrico y algoritmos de aproximación de raíces para funciones escalares $f(x) = 0$.
            - **Contenidos Oficiales:** Regla de los signos de Descartes, barrido de Bolzano, Método de Bisección, Interpolación Lineal (*Regula Falsi* variante cátedra de pivote fijo), Newton-Raphson con Fourier y Punto Fijo con Aitken $\\Delta^2$.
            """
        )
        bloque_material_estudio(
            pdf_teoria="lab3_guia_teoria_raices.pdf",
            pdf_practica="lab3_practica_resueltos.pdf",
            titulo_metodo="Guía Oficial Lab 3 (FaCENA - UNNE)",
        )
        if st.button("Ir al Módulo de Raíces de Ecuaciones", key="btn_h_l3", type="primary"):
            st.switch_page("views/raices.py")

    # ==========================================================================
    # LAB 4
    # ==========================================================================
    with t4:
        st.subheader("Laboratorio 4: Sistemas Lineales y Autovalores/Autovectores")
        st.markdown(
            """
            - **Objetivo:** Resolución de sistemas matriciales grandes y análisis espectral de transformaciones lineales.
            - **Contenidos Oficiales:** Eliminación de Gauss con pivoteo parcial, Gauss-Jordan, inversión $[A|I]\\to[I|A^{-1}]$, Factorización $LU$ de Doolittle, relajación de Gauss-Seidel con dominancia diagonal, Método de las Potencias directas e inversas y Faddeev-Leverrier.
            """
        )
        bloque_material_estudio(
            pdf_teoria="lab4_guia_teoria_sistemas_autovalores.pdf",
            pdf_practica="lab4_practica_resueltos.pdf",
            titulo_metodo="Guía Oficial Lab 4 (FaCENA - UNNE)",
        )
        if st.button("Ir al Módulo de Álgebra Lineal Numérica", key="btn_h_l4", type="primary"):
            st.switch_page("views/algebra_lineal.py")

    # ==========================================================================
    # LAB 5
    # ==========================================================================
    with t5:
        st.subheader("Laboratorio 5: Interpolación Polinomial y Diferencias Finitas")
        st.markdown(
            """
            - **Objetivo:** Ajuste de curvas polinómicas a datos tabulados experimentales.
            - **Contenidos Oficiales:** Interpolación lineal directa e inversa, interpolación cuadrática inversa ($au^2+bu+c=0$), Polinomio de Lagrange con deducción analítica de bases y esquemas progresivo (ascendente) y regresivo (descendente) de Newton-Gregory.
            """
        )
        bloque_material_estudio(
            pdf_teoria="lab5_guia_teoria_interpolacion.pdf",
            pdf_practica="lab5_practica_resueltos.pdf",
            titulo_metodo="Guía Oficial Lab 5 (FaCENA - UNNE)",
        )
        if st.button("Ir al Módulo de Interpolación Polinomial", key="btn_h_l5", type="primary"):
            st.switch_page("views/interpolacion.py")

    # ==========================================================================
    # LAB 6
    # ==========================================================================
    with t6:
        st.subheader("Laboratorio 6: Integración y Derivación Numérica")
        st.markdown(
            """
            - **Objetivo:** Aproximación numérica de operadores diferenciales e integrales definidas sobre datos discretos.
            - **Contenidos Oficiales:** Diferenciación de Newton-Gregory de 1° y 2° orden, Regla del Trapecio compuesto con fórmula de cátedra ($E, 2P, 2I$), Reglas de Simpson 1/3, 3/8 y algoritmo híbrido combinado para particiones arbitrarias.
            """
        )
        bloque_material_estudio(
            pdf_teoria="lab6_guia_teoria_integracion_derivacion.pdf",
            pdf_practica="lab6_practica_resueltos.pdf",
            titulo_metodo="Guía Oficial Lab 6 (FaCENA - UNNE)",
        )
        if st.button("Ir al Módulo de Integración y Derivación", key="btn_h_l6", type="primary"):
            st.switch_page("views/calculo_numerico.py")

    # ==========================================================================
    # LAB 7
    # ==========================================================================
    with t7:
        st.subheader("Laboratorio 7: Ecuaciones Diferenciales Ordinarias (EDO)")
        st.markdown(
            """
            - **Objetivo:** Integración temporal de Problemas de Valor Inicial (PVI) de primer orden.
            - **Contenidos Oficiales:** Método de Euler explícito, Euler Modificado (Heun), familia Runge-Kutta de orden 2 y orden 4 clásico, y método multipaso predictor-corrector de Milne con arranque por RK4.
            """
        )
        bloque_material_estudio(
            pdf_teoria="lab7_guia_teoria_edo.pdf",
            pdf_practica="lab7_practica_resueltos.pdf",
            titulo_metodo="Guía Oficial Lab 7 (FaCENA - UNNE)",
        )
        if st.button("Ir al Módulo de Ecuaciones Diferenciales", key="btn_h_l7", type="primary"):
            st.switch_page("views/edo.py")


vista_historico_labs()
