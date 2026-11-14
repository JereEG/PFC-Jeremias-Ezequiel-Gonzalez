"""
Vista de Inicio - Arandú: Métodos Computacionales (v3.0).
Portal estilo Refactoring.Guru + acceso destacado a Arena Versus.
"""

from __future__ import annotations

import streamlit as st

from views.theme import obtener_logo_arandu_src
from views.ui_shared import FEEDBACK_FORMS_URL, inject_arandu_css


def vista_inicio() -> None:
    inject_arandu_css()

    logo_uri = obtener_logo_arandu_src(prefer_svg=False)
    logo_html = (
        f'<img src="{logo_uri}" class="arandu-hero-logo" alt="Isotipo Oficial de Arandú" />'
        if logo_uri
        else ""
    )

    st.markdown(
        f"""
        <div class="arandu-hero">
            <div class="arandu-hero-grid">
                {logo_html}
                <div style="flex:1; min-width:260px;">
                    <span class="arandu-badge">Arandú Org • v3.0</span>
                    <span class="arandu-badge">FaCENA - UNNE</span>
                    <span class="arandu-badge">Dark Botanical &amp; Golden Wisdom</span>
                    <h1>Arandú — Métodos Computacionales</h1>
                    <p style="margin:0; font-size:1.02rem; color:#EDE4D3;">
                        Plataforma interactiva de la organización <b>Arandú</b> para comprender, editar y comparar métodos numéricos con rigor matemático y estética orgánica:
                    </p>
                    <div class="arandu-pillars">
                        <div class="arandu-pillar-item">
                            <strong>🌱 Raíces:</strong> Fundamentos teóricos y análisis del error.
                        </div>
                        <div class="arandu-pillar-item">
                            <strong>🪵 Tronco:</strong> Algoritmos interactivos paso a paso y prácticas de cátedra.
                        </div>
                        <div class="arandu-pillar-item">
                            <strong>✨ Copa y Astros:</strong> Visualización geométrica y Arena Versus de convergencia.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        c_txt, c_btn = st.columns([3.2, 1.3])
        with c_txt:
            st.markdown("### ⚔️ Arena Versus: Duelo Directo de Métodos (Versus.com)")
            st.write(
                "Enfrenta dos algoritmos de la **misma familia matemática** sobre un mismo problema, "
                "descubre qué esperamos ver teóricamente (incluyendo casos patológicos) y compara "
                "convergencia, costo y error en gráficos superpuestos."
            )
        with c_btn:
            st.write("")
            if st.button("⚔️ Entrar a Arena Versus", type="primary", use_container_width=True):
                st.switch_page("views/arena_versus.py")

    st.markdown("### 🧭 Catálogo de Métodos Matemáticos")
    st.caption("Cada módulo sigue el esquema de 4 bloques: Teoría Guru + Ejemplo Oficial → Práctica Editable → Controles en Vivo → Geometría Paso a Paso.")

    tarjetas = [
        (
            "🧮 Fundamentos y Errores",
            "Series de Taylor (Scarborough & Leibniz), aritmética flotante de k dígitos, pérdida de asociatividad, cancelación catastrófica y algoritmo anidado de Horner.",
            "views/view_fundamentos.py",
            "Abrir Fundamentos",
        ),
        (
            "📉 Raíces de Ecuaciones",
            "Regla de Descartes, Barrido por Tanteos, Bisección, Regula Falsi (Cátedra y Bolzano), Newton-Raphson (Fourier) y Punto Fijo acelerado con Δ² de Aitken.",
            "views/view_raices.py",
            "Abrir Raíces",
        ),
        (
            "📐 Sistemas Lineales y Autovalores",
            "Eliminación de Gauss, Gauss-Jordan, Factorización LU (Doolittle), Gauss-Seidel (dominancia diagonal), Método de Potencias y Faddeev-Leverrier.",
            "views/view_sistemas.py",
            "Abrir Sistemas y Autovalores",
        ),
        (
            "📈 Interpolación Polinomial",
            "Interpolación Lineal, Cuadrática Inversa, Polinomio de Lagrange y Newton-Gregory Ascendente/Descendente con detector de nodos no equidistantes.",
            "views/view_interpolacion.py",
            "Abrir Interpolación",
        ),
        (
            "📊 Integración y Derivación",
            "Diferenciación con Gregory (y punto libre) + Trapecio, Simpson 1/3, Simpson 3/8 y reglas combinadas automáticas al cambiar Δh o n en vivo.",
            "views/view_integracion.py",
            "Abrir Integración y Derivación",
        ),
        (
            "⏱️ Ecuaciones Diferenciales (EDO)",
            "Problemas de Valor Inicial: Euler, Heun (corrector iterativo), RK2, RK4 y Predictor-Corrector de Milne con malla Δh interactiva y solución exacta.",
            "views/view_edo.py",
            "Abrir EDO / PVI",
        ),
    ]

    for idx in range(0, len(tarjetas), 2):
        col_izq, col_der = st.columns(2)
        for col, item in zip((col_izq, col_der), tarjetas[idx : idx + 2]):
            titulo, desc, ruta, texto_btn = item
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {titulo}")
                    st.write(desc)
                    if st.button(texto_btn, key=f"card_{ruta}", use_container_width=True):
                        st.switch_page(ruta)

    with st.container(border=True):
        c1, c2 = st.columns([3.2, 1.3])
        with c1:
            st.markdown("#### 📚 Histórico de Cursada (Laboratorios 1 al 7)")
            st.write(
                "Anexo curricular con los experimentos clásicos de los Trabajos Prácticos "
                "(Fibonacci N=200, razón áurea φ, tiempos y comparativas integradas por laboratorio)."
            )
        with c2:
            if st.button("📚 Ver Laboratorios", use_container_width=True):
                st.switch_page("views/view_laboratorios.py")

    with st.container(border=True):
        cf1, cf2 = st.columns([3.2, 1.3])
        with cf1:
            st.markdown("#### 💬 Tu opinión nos ayuda a mejorar Arandú")
            st.write(
                "Si sos docente o estudiante y querés sugerir mejoras en las explicaciones, "
                "reportar algún caso particular o evaluar la experiencia de uso para el PFC, "
                "completá nuestro breve formulario de feedback."
            )
        with cf2:
            st.write("")
            st.link_button(
                "📝 Enviar Feedback",
                FEEDBACK_FORMS_URL,
                type="primary",
                width="stretch",
            )


vista_inicio()
