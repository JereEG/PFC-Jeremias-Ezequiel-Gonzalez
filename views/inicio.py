"""
Vista de Inicio - Arandú: Métodos Computacionales.
Portal estilo Refactoring.Guru + acceso destacado a Arena Versus y catálogo pedagógico.
"""

from __future__ import annotations

import streamlit as st

from core.version import APP_VERSION, es_entorno_offline_wasm
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
    badge_entorno = (
        "PWA Offline (WebAssembly)" if es_entorno_offline_wasm() else "Cloud Server (Streamlit)"
    )

    st.markdown(
        f"""
        <div class="arandu-hero">
            <div class="arandu-hero-grid">
                {logo_html}
                <div style="flex:1; min-width:260px;">
                    <span class="arandu-badge">Arandú Org • v{APP_VERSION}</span>
                    <span class="arandu-badge">FaCENA - UNNE</span>
                    <span class="arandu-badge">{badge_entorno}</span>
                    <h1>Arandú — Métodos Computacionales</h1>
                    <p style="margin:0; font-size:1.02rem; color:#EDE4D3;">
                        Plataforma interactiva de <b>Arandú</b> para comprender, editar y comparar algoritmos numéricos con rigor académico y lectura enfocada en sistemas:
                    </p>
                    <div class="arandu-pillars">
                        <div class="arandu-pillar-item">
                            <strong>🌱 Raíces:</strong> Intuición computacional, teoría en 3 niveles y análisis del error de máquina.
                        </div>
                        <div class="arandu-pillar-item">
                            <strong>🪵 Tronco:</strong> Simulación en vivo con fragmentos aislados y ejercicios editables de cátedra.
                        </div>
                        <div class="arandu-pillar-item">
                            <strong>✨ Copa y Astros:</strong> Geometría interactiva y duelos de convergencia en Arena Versus.
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
            st.markdown("### ⚔️ Arena Versus: Duelo Directo de Métodos")
            st.write(
                "Enfrenta dos algoritmos de la **misma familia matemática** sobre un mismo problema, "
                "anticipa qué comportamiento veremos en ejecución (incluyendo casos límite) y compara "
                "convergencia, costo de iteraciones y error en gráficos superpuestos."
            )
        with c_btn:
            st.write("")
            if st.button("⚔️ Entrar a Arena Versus", type="primary", use_container_width=True):
                st.switch_page("views/arena_versus.py")

    st.markdown("### 🧭 Catálogo de Métodos Matemáticos")
    st.caption(
        "Lectura progresiva en cada módulo: Resumen Ejecutivo → Intuición y Fórmulas (con desarrollo opcional) → "
        "Criterio de Elección → Laboratorio Interactivo paso a paso."
    )

    tarjetas = [
        (
            "🧮 Fundamentos y Errores",
            r"Series iterativas ($e^x$, $\ln(1+x)$, $\pi$) en la CPU, aritmética finita de $k$ dígitos, pérdida de asociatividad, cancelación catastrófica y algoritmo de Horner.",
            "views/view_fundamentos.py",
            "Abrir Fundamentos",
        ),
        (
            "📉 Raíces de Ecuaciones",
            r"Regla de Descartes, Barrido por Tanteos, Bisección, Regula Falsi (Cátedra y cambio de signo), Newton-Raphson y Punto Fijo acelerado con $\Delta^2$ de Aitken.",
            "views/view_raices.py",
            "Abrir Raíces",
        ),
        (
            "📐 Sistemas Lineales y Autovalores",
            "Eliminación de Gauss, Gauss-Jordan, Factorización $LU$ (Doolittle), Gauss-Seidel (dominancia diagonal), Método de Potencias y Faddeev-Leverrier.",
            "views/view_sistemas.py",
            "Abrir Sistemas y Autovalores",
        ),
        (
            "📈 Interpolación Polinomial",
            "Interpolación Lineal, Cuadrática Inversa, Polinomio de Lagrange y diferencias finitas de Newton-Gregory Ascendente y Descendente.",
            "views/view_interpolacion.py",
            "Abrir Interpolación",
        ),
        (
            "📊 Integración y Derivación",
            r"Derivación numérica con Gregory y reglas de Trapecio, Simpson $1/3$, Simpson $3/8$ y combinaciones adaptativas al ajustar $\Delta h$ o $n$ en vivo.",
            "views/view_integracion.py",
            "Abrir Integración y Derivación",
        ),
        (
            "⏱️ Ecuaciones Diferenciales (EDO)",
            r"Problemas de Valor Inicial (PVI): Euler, Heun (Euler Modificado), RK2, RK4 y Predictor-Corrector de Milne con paso $\Delta h$ ajustable y solución exacta.",
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
                    st.markdown(desc)
                    if st.button(texto_btn, key=f"card_{ruta}", use_container_width=True):
                        st.switch_page(ruta)

    with st.container(border=True):
        c1, c2 = st.columns([3.2, 1.3])
        with c1:
            st.markdown("#### 📚 Histórico de Cursada (Laboratorios 1 al 7)")
            st.markdown(
                r"Anexo curricular con los experimentos clásicos de los Trabajos Prácticos "
                r"(Fibonacci con $N = 200$, razón áurea $\varphi$, sumatorias y mapa completo de la cursada)."
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
                use_container_width=True,
            )


vista_inicio()
