"""
Re-exportaciones diferidas (PEP 562) de componentes UX/UI compartidos para compatibilidad.
La implementación canónica reside en ``views.ui_shared`` y ``views.theme``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_THEME_EXPORTS = {
    "ARANDU_BARK",
    "ARANDU_CARD_BORDER",
    "ARANDU_GOLD",
    "ARANDU_GOLD_FILL",
    "ARANDU_GRID",
    "ARANDU_IVORY",
    "ARANDU_MUTED",
    "ARANDU_NODE_IVORY",
    "ARANDU_OBSIDIAN",
    "ARANDU_PLOT_BG",
    "ARANDU_SAGE",
    "ARANDU_SAGE_BORDER",
    "ARANDU_SAGE_FILL",
    "ARANDU_TERRACOTTA",
    "ARANDU_TERRACOTTA_DARK",
    "ARANDU_TERRACOTTA_FILL",
    "COLORWAY_ARANDU",
    "FEEDBACK_FORMS_URL",
    "LOGO_JPG_PATH",
    "LOGO_SVG_PATH",
    "PALETA_ARANDU",
    "PLOTLY_ARANDU_TEMPLATE",
    "PLOTLY_CONFIG_ARANDU",
    "aplicar_estilo_arandu",
    "construir_html_carta_tarot",
    "construir_html_mini_tarjetas",
    "construir_html_tarjeta_catalogo",
    "estilizar_figura_arandu",
    "leer_asset_bytes",
    "leer_asset_texto",
    "obtener_asset_base64",
    "obtener_asset_data_uri",
    "obtener_css_arandu",
    "obtener_logo_arandu_src",
    "obtener_logos_app",
    "obtener_plotly_arandu_template",
    "obtener_svg_tarjeta_arandu",
    "obtener_svg_tarot_arandu",
}

_UI_SHARED_EXPORTS = {
    "crear_figura_base_plotly",
    "inicializar_estado_practica",
    "inject_arandu_css",
    "limpiar_valores_plotly",
    "muestrear_para_grafico",
    "obtener_logo_data_uri",
    "render_banner_edicion_profesora",
    "render_bloque_1_teoria_y_ejemplo",
    "render_controles_edicion_practica",
    "render_dataframe_acotado",
    "render_mazo_tarot_metodos",
    "render_mini_catalogo_vista",
    "render_plotly_arandu",
    "render_practica_catedra_info",
    "render_slider_paso_k",
    "render_tarjeta_catalogo",
    "rerun_fragmento_seguro",
    "restaurar_estado_practica",
}

__all__ = sorted(_THEME_EXPORTS | _UI_SHARED_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _THEME_EXPORTS:
        return getattr(import_module("views.theme"), name)
    if name in _UI_SHARED_EXPORTS:
        return getattr(import_module("views.ui_shared"), name)
    raise AttributeError(f"module 'views' has no attribute {name!r}")
