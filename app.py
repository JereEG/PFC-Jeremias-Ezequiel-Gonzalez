"""
app.py
======
Punto de Entrada Definitivo (Entrypoint) de la Suite Educativa de Métodos Computacionales.
Proyecto Final de Carrera (PFC) — FaCENA - Universidad Nacional del Nordeste (UNNE).

Desarrollado con arquitectura Cloud-Ready para Streamlit Community Cloud utilizando
el sistema moderno de navegación nativo st.navigation y st.Page por archivos.
"""

from __future__ import annotations

import streamlit as st

# Configuración global de la página
st.set_page_config(
    page_title="Métodos Computacionales - FaCENA UNNE",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definición de páginas mediante rutas de archivo reconocidas por Streamlit
pg_inicio = st.Page(
    "views/inicio.py",
    title="Catálogo Principal",
    icon="🏠",
    default=True,
    url_path="inicio",
)

pg_lab1_2 = st.Page(
    "views/view_lab1_2.py",
    title="Labs 1 y 2: Teoría y Errores",
    icon="🔬",
    url_path="lab1_2",
)

pg_lab3 = st.Page(
    "views/view_lab3.py",
    title="Lab 3: Búsqueda de Raíces",
    icon="🎯",
    url_path="lab3",
)

pg_lab4 = st.Page(
    "views/view_lab4.py",
    title="Lab 4: Sistemas y Autovalores",
    icon="🏛️",
    url_path="lab4",
)

pg_lab5 = st.Page(
    "views/view_lab5.py",
    title="Lab 5: Interpolación Polinómica",
    icon="📈",
    url_path="lab5",
)

pg_lab6 = st.Page(
    "views/view_lab6.py",
    title="Lab 6: Integración y Derivación",
    icon="📐",
    url_path="lab6",
)

pg_lab7 = st.Page(
    "views/view_lab7.py",
    title="Lab 7: Ecuaciones Diferenciales",
    icon="🚀",
    url_path="lab7",
)

# Estructura del menú de navegación
nav = st.navigation({
    "Inicio": [pg_inicio],
    "Laboratorios": [
        pg_lab1_2,
        pg_lab3,
        pg_lab4,
        pg_lab5,
        pg_lab6,
        pg_lab7,
    ],
})

# Panel lateral con información institucional
with st.sidebar:
    st.markdown("### 🎓 PFC Métodos Computacionales")
    st.caption("FaCENA — Universidad Nacional del Nordeste (UNNE)")
    st.markdown("---")
    st.markdown(
        """
        **Alumno:** Jeremías Ezequiel González  
        **Plataforma:** Python 3 + Streamlit  
        **Entorno:** Cloud-Ready (Free Tier)
        """
    )
    st.markdown("---")
    st.caption("v1.0.0 • Producción")

# Ejecutar el ruteo de la página seleccionada
nav.run()
