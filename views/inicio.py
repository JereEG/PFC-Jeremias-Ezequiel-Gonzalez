"""
views/inicio.py
===============
Catálogo Principal e Inicio de la Suite Educativa de Métodos Computacionales.
Diseñado con tarjetas visuales interactivas para navegación docente y estudiantil en 1 clic.
"""

from __future__ import annotations

import streamlit as st


def vista_inicio():
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1f77b4 0%, #00a86b 100%); padding: 24px 30px; border-radius: 12px; color: white; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: white;">🎓 Métodos Computacionales — Suite Educativa</h1>
            <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                Facultad de Ciencias Exactas y Naturales y Agrimensura (FaCENA - UNNE) &bull; Proyecto Final de Carrera (PFC)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Bienvenido a la **Suite Educativa Interactiva de Métodos Computacionales**. Esta plataforma unifica los fundamentos teóricos, 
        el desarrollo algorítmico paso a paso y la visualización gráfica de los **Laboratorios 1 al 7**, permitiendo experimentar con los 
        ejemplos oficiales de la cátedra o ingresar ecuaciones y datos propios.

        Seleccione cualquiera de los módulos a continuación para comenzar:
        """
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown("### 🔬 Módulo 1 y 2: Algoritmos, Series y Teoría de Errores")
            st.caption("Modelos iterativos, tiempos CPU/reloj, series de Taylor, aritmética finita y estabilidad.")
            st.markdown(
                """
                - **Lab 1:** Suma de $N$ números, pares, Fibonacci ($N=200$, Razón Áurea $\\varphi$), series de $e^x$ y $\\pi$ (Gregory-Leibniz).
                - **Lab 2:** Error absoluto/relativo, cifras significativas, aritmética de máquina de $k$ dígitos, pérdida de asociatividad, cancelación catastrófica en cuadráticas y forma anidada de Horner.
                """
            )
            if st.button("🚀 Explorar Módulo 1 y 2", key="btn_nav_l12", use_container_width=True, type="primary"):
                st.switch_page("views/view_lab1_2.py")

        with st.container(border=True):
            st.markdown("### 🏛️ Módulo 4: Sistemas Lineales y Autovalores")
            st.caption("Métodos directos matriciales, descomposición, relajación iterativa y espectro matricial.")
            st.markdown(
                """
                - **Eliminación de Gauss y Gauss-Jordan:** Escalonamiento por filas, pivoteo parcial y cálculo de matriz inversa.
                - **Factorización $LU$ (Doolittle):** Sustitución hacia adelante ($Ly=b$) y hacia atrás ($Ux=y$).
                - **Gauss-Seidel:** Condición de dominancia diagonal e iteración con actualización inmediata.
                - **Autovalores:** Método de las Potencias (máximo y mínimo) y Polinomio Característico de **Faddeev-Leverrier**.
                """
            )
            if st.button("🚀 Explorar Módulo 4", key="btn_nav_l4", use_container_width=True, type="primary"):
                st.switch_page("views/view_lab4.py")

        with st.container(border=True):
            st.markdown("### 📐 Módulo 6: Diferenciación e Integración Numérica")
            st.caption("Derivación por diferencias finitas y fórmulas de cuadratura compuesta de Newton-Cotes.")
            st.markdown(
                """
                - **Diferenciación Numérica:** 1° y 2° derivada por fórmulas de Newton-Gregory (ascendente y descendente).
                - **Regla del Trapecio:** Trapecio simple y compuesto $A = \\frac{h}{2}(E + 2P + 2I)$ con cota $O(h^2)$.
                - **Reglas de Simpson:** Simpson 1/3 ($O(h^4)$), Simpson 3/8 y Simpson Combinado para particiones impares.
                - **Contraste Simbólico:** Verificación analítica exacta mediante SymPy.
                """
            )
            if st.button("🚀 Explorar Módulo 6", key="btn_nav_l6", use_container_width=True, type="primary"):
                st.switch_page("views/view_lab6.py")

    with col2:
        with st.container(border=True):
            st.markdown("### 🎯 Módulo 3: Búsqueda de Raíces No Lineales")
            st.caption("Aislamiento analítico y métodos iterativos para ecuaciones trascendentes y polinómicas.")
            st.markdown(
                """
                - **Análisis Previo:** Regla de los Signos de Descartes y Método de Tanteos (Teorema de Bolzano).
                - **Método de Bisección:** Intervalo medio con estimación de iteraciones teóricas mínimas.
                - **Interpolación Lineal (*Regula Falsi*):** Variante de cátedra (pivote fijo) y variante clásica.
                - **Newton-Raphson:** Tangentes con verificación de condiciones de convergencia de Fourier ($f \\cdot f'' > 0$).
                - **Punto Fijo y Aceleración de Aitken:** $x_{n+1} = \\varphi(x_n)$ con acelerador $\\Delta^2$.
                """
            )
            if st.button("🚀 Explorar Módulo 3", key="btn_nav_l3", use_container_width=True, type="primary"):
                st.switch_page("views/view_lab3.py")

        with st.container(border=True):
            st.markdown("### 📈 Módulo 5: Interpolación Polinomial")
            st.caption("Aproximación polinómica exacta sobre datos tabulados o funciones continuas.")
            st.markdown(
                """
                - **Interpolación Lineal:** Tramos lineales directos e inversión analítica $x(y)$.
                - **Interpolación Cuadrática Inversa:** Resolución de $a u^2 + b u + c = 0$ para $u = x - x_0$.
                - **Polinomio de Lagrange:** Polinomios base $L_i(x)$ y reconstrucción simbólica simplificada con SymPy.
                - **Newton-Gregory:** Tablas de diferencias finitas progresivas (ascendente, $\\Delta$) y retrógradas (descendente, $\\nabla$).
                """
            )
            if st.button("🚀 Explorar Módulo 5", key="btn_nav_l5", use_container_width=True, type="primary"):
                st.switch_page("views/view_lab5.py")

        with st.container(border=True):
            st.markdown("### 🚀 Módulo 7: Ecuaciones Diferenciales (PVI)")
            st.caption("Resolución numérica de Problemas de Valor Inicial $y' = f(x, y)$, $y(x_0) = y_0$.")
            st.markdown(
                """
                - **Métodos de un paso:** Euler explícito $O(h)$ y Euler Modificado (Heun $O(h^2)$).
                - **Familia Runge-Kutta:** RK2 estándar, RK2 Punto Medio y el clásico RK4 de 4° orden ($O(h^4)$).
                - **Método de Milne:** Predictor-corrector multipaso con arranque automático mediante RK4.
                - **Comparativa Lado a Lado:** Análisis de convergencia y errores relativos frente a soluciones exactas.
                """
            )
            if st.button("🚀 Explorar Módulo 7", key="btn_nav_l7", use_container_width=True, type="primary"):
                st.switch_page("views/view_lab7.py")

    st.markdown("---")
    st.info(
        "💡 **Sugerencia de Uso Docente:** Cada módulo incluye cargadores de ejemplos predeterminados idénticos a las guías de estudio "
        "y exámenes de FaCENA - UNNE. Puede utilizarlos como demostración en clase o modificar los parámetros libremente para explorar el comportamiento de los algoritmos."
    )


# Ejecución de la vista al ser cargada por Streamlit
vista_inicio()
