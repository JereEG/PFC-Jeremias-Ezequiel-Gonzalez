"""
views/inicio.py
===============
Métodos Computacionales / Numéricos — Portal de Referencia Matemática.
Catálogo central organizado por ramas y métodos numéricos, con navegación directa en 1 clic.
"""

from __future__ import annotations

import streamlit as st


def vista_inicio():
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00c9ff 100%); 
                    padding: 32px 35px; border-radius: 16px; color: white; margin-bottom: 25px; 
                    box-shadow: 0 8px 24px rgba(0,0,0,0.18);">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 8px;">
                <span style="font-size: 2.8rem;">🏛️</span>
                <div>
                    <h1 style="margin: 0; font-size: 2.3rem; font-weight: 800; color: white; letter-spacing: -0.5px;">
                        Métodos Computacionales / Numéricos
                    </h1>
                    <p style="margin: 4px 0 0 0; font-size: 1.15rem; opacity: 0.95; font-weight: 300;">
                        Plataforma de Referencia Teórica, Algorítmica y Visual &bull; FaCENA - UNNE
                    </p>
                </div>
            </div>
            <div style="margin-top: 14px; font-size: 0.95rem; opacity: 0.9; line-height: 1.5; border-top: 1px solid rgba(255,255,255,0.25); padding-top: 12px;">
                Estructura centrada en <b>Métodos Numéricos</b>. Cada método integra su fundamento teórico formal, notas de cátedra en PDF (teoría y ejercicios resueltos) y una calculadora interactiva con deducción paso a paso y visualizaciones geométricas.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧭 Catálogo de Especialidades Matemáticas")
    st.markdown(
        "Seleccione cualquier categoría matemática para acceder a sus métodos numéricos y herramientas de cálculo:"
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### 🧮 Fundamentos y Errores")
            st.caption("Aritmética de máquina, propagación, estabilidad y convergencia en series.")
            st.markdown(
                """
                - **Series de Taylor:** Aproximación de funciones continuas, serie de $e^x$ y $\\pi$ con Scarborough.
                - **Aritmética Finita ($k$ dígitos):** Truncamiento vs. redondeo, pérdida de asociatividad.
                - **Cancelación Catastrófica:** Racionalización de diferencias de números casi idénticos.
                - **Esquema de Horner:** Evaluación polinómica anidada minimizando operaciones de punto flotante.
                - **Sucesiones y Fibonacci:** Modelos iterativos y cálculo exacto de la Razón Áurea $\\varphi$.
                """
            )
            if st.button("Explorar Fundamentos y Errores", key="btn_nav_fund", use_container_width=True, type="primary"):
                st.switch_page("views/fundamentos.py")

        with col1:
            with st.container(border=True):
                st.markdown("### 📐 Álgebra Lineal Numérica")
                st.caption("Resolución de sistemas $Ax=b$, factorización matricial y espectro de autovalores.")
                st.markdown(
                    """
                    - **Eliminación de Gauss:** Triangularización con pivoteo parcial y sustitución regresiva.
                    - **Gauss-Jordan:** Forma escalonada reducida por filas (RREF) y cálculo de matriz inversa $A^{-1}$.
                    - **Factorización $LU$:** Algoritmo de Doolittle con sustitución progresiva y regresiva.
                    - **Gauss-Seidel:** Relajación iterativa y análisis de dominancia diagonal estricta.
                    - **Método de las Potencias:** Detección de autovalores extremos $\\lambda_{\\max}$ y $\\lambda_{\\min}$.
                    - **Faddeev-Leverrier:** Polinomio característico completo, determinante e inversa analítica.
                    """
                )
                if st.button("Explorar Álgebra Lineal", key="btn_nav_alg", use_container_width=True, type="primary"):
                    st.switch_page("views/algebra_lineal.py")

        with col1:
            with st.container(border=True):
                st.markdown("### 📊 Integración y Derivación")
                st.caption("Cuadraturas de Newton-Cotes y esquemas de diferenciación finita.")
                st.markdown(
                    """
                    - **Diferenciación Numérica:** Fórmulas de Gregory progresivas y regresivas (1° y 2° derivada).
                    - **Regla del Trapecio Compuesto:** Desglose cátedra ($E, 2P, 2I$) con cota de error $O(h^2)$.
                    - **Reglas de Simpson:** Simpson 1/3 ($O(h^4)$), Simpson 3/8 y algoritmo híbrido combinado.
                    - **Cuadratura Simbólica:** Banco de pruebas contra solución analítica de SymPy.
                    """
                )
                if st.button("Explorar Integración y Derivación", key="btn_nav_calc", use_container_width=True, type="primary"):
                    st.switch_page("views/calculo_numerico.py")

    with col2:
        with st.container(border=True):
            st.markdown("### 📉 Raíces de Ecuaciones")
            st.caption("Aislamiento analítico y convergencia iterativa para ecuaciones no lineales $f(x) = 0$.")
            st.markdown(
                """
                - **Análisis Previo:** Regla de los signos de Descartes y barrido sistemático de Bolzano.
                - **Método de Bisección:** Búsqueda en intervalo medio con cota teórica de iteraciones mínimas.
                - **Interpolación Lineal (*Regula Falsi*):** Variante clásica y variante de cátedra (pivote fijo).
                - **Newton-Raphson:** Método de tangentes con verificación previa de Fourier ($f \\cdot f'' > 0$).
                - **Punto Fijo y Aitken:** Iteración $x_{k+1} = \\varphi(x_k)$ acelerada mediante $\\Delta^2$.
                - **Comparativa Multimétodo:** Benchmarking de velocidad y precisión en paralelo.
                """
            )
            if st.button("Explorar Raíces de Ecuaciones", key="btn_nav_raices", use_container_width=True, type="primary"):
                st.switch_page("views/raices.py")

        with col2:
            with st.container(border=True):
                st.markdown("### 📈 Interpolación Polinomial")
                st.caption("Ajuste exacto sobre conjuntos discretos de puntos y diferencias finitas.")
                st.markdown(
                    """
                    - **Interpolación Lineal:** Tramos lineales directos y búsqueda inversa $x(y)$.
                    - **Cuadrática Inversa:** Resolución de $au^2 + bu + c = 0$ para estimar raíces físicas.
                    - **Polinomio de Lagrange:** Polinomios base $L_i(x)$ y reconstrucción simbólica simplificada.
                    - **Newton-Gregory Ascendente:** Tabla de diferencias progresivas ($\\Delta$) para inicio de tabla.
                    - **Newton-Gregory Descendente:** Tabla de diferencias regresivas ($\\nabla$) para final de tabla.
                    """
                )
                if st.button("Explorar Interpolación", key="btn_nav_interp", use_container_width=True, type="primary"):
                    st.switch_page("views/interpolacion.py")

        with col2:
            with st.container(border=True):
                st.markdown("### ⏱️ Ecuaciones Diferenciales (EDO)")
                st.caption("Modelado numérico de Problemas de Valor Inicial (PVI) $y' = f(x, y)$, $y(x_0)=y_0$.")
                st.markdown(
                    """
                    - **Métodos de un Paso:** Euler explícito $O(h)$ y Euler Modificado (Heun $O(h^2)$).
                    - **Runge-Kutta:** Familia RK2 (punto medio y estándar) y el clásico RK4 de 4° orden.
                    - **Predictor-Corrector de Milne:** Esquema multipaso de orden 4 con arranque por RK4.
                    - **Comparativa Multimétodo:** Traqueo simultáneo de trayectorias y contraste con solución exacta.
                    """
                )
                if st.button("Explorar Ecuaciones Diferenciales", key="btn_nav_edo", use_container_width=True, type="primary"):
                    st.switch_page("views/edo.py")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### 📚 Histórico de Cursada (Laboratorios 1 al 7)")
        st.caption("Anexo curricular con los trabajos prácticos de la cátedra de Métodos Computacionales (FaCENA - UNNE).")
        st.markdown(
            "Consulte la correlación de los laboratorios originales de la carrera con los nuevos módulos matemáticos, "
            "incluyendo enunciados de examen, guías oficiales y casos testigo."
        )
        if st.button("Ver Histórico de Laboratorios", key="btn_nav_hist", use_container_width=True):
            st.switch_page("views/historico_labs.py")


vista_inicio()
