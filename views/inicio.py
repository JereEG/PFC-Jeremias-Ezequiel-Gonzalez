"""
Vista de Inicio - Arandú: Métodos Computacionales.
Portal estilo Refactoring.Guru + acceso destacado a Arena Versus y catálogo pedagógico.
"""

from __future__ import annotations

import streamlit as st

from core.version import APP_VERSION, es_entorno_offline_wasm
from views.theme import construir_html_tarjeta_catalogo, obtener_logo_arandu_src
from views.ui_shared import FEEDBACK_FORMS_URL, inject_arandu_css, render_tarjeta_catalogo


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
                            <strong>Fundamentos:</strong> Intuición computacional, teoría en 3 niveles y análisis del error de máquina.
                        </div>
                        <div class="arandu-pillar-item">
                            <strong>Simulación:</strong> Ejercicios editables en vivo con fragmentos aislados y datos de cátedra.
                        </div>
                        <div class="arandu-pillar-item">
                            <strong>Comparativa:</strong> Geometría interactiva y duelos de convergencia en Arena Versus.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Catálogo de Métodos y Herramientas")
    st.caption(
        "Seleccioná un módulo para recorrer su explicación progresiva (Resumen → Intuición → Criterios) y simular paso a paso."
    )

    tarjetas = [
        (
            "fundamentos",
            "Fundamentos y Errores",
            "Traduce funciones a bucles de CPU y mide el error al redondear dígitos.",
            "views/view_fundamentos.py",
            "Explorar Fundamentos",
            "UNIDAD 01 • ERRORES",
            False,
        ),
        (
            "raices",
            "Raíces de Ecuaciones",
            "Halla dónde una curva cruza cero acotando intervalos o trazando tangentes.",
            "views/view_raices.py",
            "Explorar Raíces",
            "UNIDAD 02 • NO LINEALES",
            False,
        ),
        (
            "sistemas",
            "Álgebra Lineal y Autovalores",
            "Resuelve sistemas Ax = b por triangulación o iteración y extrae eigenvalores.",
            "views/view_sistemas.py",
            "Explorar Sistemas",
            "UNIDAD 03–04 • MATRICES",
            False,
        ),
        (
            "interpolacion",
            "Interpolación Polinomial",
            "Construye curvas continuas que pasan exactamente por puntos medidos.",
            "views/view_interpolacion.py",
            "Explorar Interpolación",
            "UNIDAD 05 • POLINOMIOS",
            False,
        ),
        (
            "integracion",
            "Integración y Derivación",
            "Estima pendientes locales y áreas acumuladas adaptando subintervalos en vivo.",
            "views/view_integracion.py",
            "Explorar Integración",
            "UNIDAD 06 • CUADRATURA",
            False,
        ),
        (
            "edo",
            "Ecuaciones Diferenciales (EDO)",
            "Simula trayectorias paso a paso desde un punto inicial con error controlado.",
            "views/view_edo.py",
            "Explorar EDO / PVI",
            "UNIDAD 07 • PVI",
            False,
        ),
        (
            "arena_versus",
            "Arena Versus (Duelo)",
            "Enfrenta dos algoritmos de la misma familia y compara velocidad y error.",
            "views/arena_versus.py",
            "Entrar a Arena Versus",
            "COMPARADOR DIRECTO",
            True,
        ),
        (
            "laboratorios",
            "Histórico de Cursada",
            "Repasa los TPs 1 al 7, la serie de Fibonacci N=200 y la razón áurea.",
            "views/view_laboratorios.py",
            "Ver Laboratorios",
            "ANEXO CURRICULAR",
            False,
        ),
    ]

    for idx in range(0, 6, 3):
        cols = st.columns(3)
        for col, item in zip(cols, tarjetas[idx : idx + 3]):
            clave_svg, titulo, desc, ruta, texto_btn, tag, primario = item
            with col:
                render_tarjeta_catalogo(
                    clave_svg=clave_svg,
                    titulo=titulo,
                    descripcion=desc,
                    ruta=ruta,
                    texto_boton=texto_btn,
                    key=f"card_{ruta}",
                    subtitulo_tag=tag,
                    primary=primario,
                )

    cols_extra = st.columns(3)
    for col, item in zip(cols_extra[:2], tarjetas[6:8]):
        clave_svg, titulo, desc, ruta, texto_btn, tag, primario = item
        with col:
            render_tarjeta_catalogo(
                clave_svg=clave_svg,
                titulo=titulo,
                descripcion=desc,
                ruta=ruta,
                texto_boton=texto_btn,
                key=f"card_{ruta}",
                subtitulo_tag=tag,
                primary=primario,
            )

    with cols_extra[2]:
        with st.container():
            st.markdown(
                construir_html_tarjeta_catalogo(
                    clave_svg="feedback",
                    titulo="Feedback y Aportes",
                    descripcion="Envía sugerencias de cátedra o reporta casos límite del simulador.",
                    subtitulo_tag="COMUNIDAD • PFC",
                ),
                unsafe_allow_html=True,
            )
            st.link_button(
                "Enviar Feedback",
                FEEDBACK_FORMS_URL,
                use_container_width=True,
            )


vista_inicio()
