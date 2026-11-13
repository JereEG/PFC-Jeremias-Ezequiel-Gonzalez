"""
Re-exportaciones de componentes UX/UI compartidos para compatibilidad hacia atrás.
La implementación canónica reside en ``views.ui_shared``.
"""

from __future__ import annotations

from views.ui_shared import (
    FEEDBACK_FORMS_URL,
    LOGO_SVG_PATH,
    crear_figura_base_plotly,
    inicializar_estado_practica,
    inject_arandu_css,
    obtener_logo_data_uri,
    render_banner_edicion_profesora,
    render_bloque_1_teoria_y_ejemplo,
    render_slider_paso_k,
    restaurar_estado_practica,
)

__all__ = [
    "FEEDBACK_FORMS_URL",
    "LOGO_SVG_PATH",
    "crear_figura_base_plotly",
    "inicializar_estado_practica",
    "inject_arandu_css",
    "obtener_logo_data_uri",
    "render_banner_edicion_profesora",
    "render_bloque_1_teoria_y_ejemplo",
    "render_slider_paso_k",
    "restaurar_estado_practica",
]
