"""
Motor de Estilo e Identidad Visual de Arandú ("Dark Botanical & Golden Wisdom").
Centraliza la paleta cromática oficial, tipografías, CSS de Streamlit y Template Plotly.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    import plotly.graph_objects as go

ARANDU_GOLD = "#D4A359"
ARANDU_OBSIDIAN = "#0A0B09"
ARANDU_BARK = "#161814"
ARANDU_PLOT_BG = "#121410"
ARANDU_IVORY = "#EDE4D3"
ARANDU_NODE_IVORY = "#FFFDF9"
ARANDU_SAGE = "#7A9A65"
ARANDU_SAGE_FILL = "rgba(122, 154, 101, 0.25)"
ARANDU_SAGE_BORDER = "#687D5B"
ARANDU_TERRACOTTA = "#C86D44"
ARANDU_TERRACOTTA_DARK = "#B5673F"
ARANDU_TERRACOTTA_FILL = "rgba(200, 109, 68, 0.25)"
ARANDU_GOLD_FILL = "rgba(212, 163, 89, 0.25)"
ARANDU_GRID = "#26241E"
ARANDU_CARD_BORDER = "#2E2920"
ARANDU_MUTED = "#9E9585"

PALETA_ARANDU: dict[str, str] = {
    "primary_gold": ARANDU_GOLD,
    "background_obsidian": ARANDU_OBSIDIAN,
    "surface_bark": ARANDU_BARK,
    "plot_bg": ARANDU_PLOT_BG,
    "text_ivory": ARANDU_IVORY,
    "node_ivory": ARANDU_NODE_IVORY,
    "sage_green": ARANDU_SAGE,
    "sage_fill": ARANDU_SAGE_FILL,
    "sage_border": ARANDU_SAGE_BORDER,
    "terracotta": ARANDU_TERRACOTTA,
    "terracotta_dark": ARANDU_TERRACOTTA_DARK,
    "terracotta_fill": ARANDU_TERRACOTTA_FILL,
    "gold_fill": ARANDU_GOLD_FILL,
    "grid_line": ARANDU_GRID,
    "card_border": ARANDU_CARD_BORDER,
    "muted_text": ARANDU_MUTED,
}

COLORWAY_ARANDU: list[str] = [
    ARANDU_GOLD,
    ARANDU_SAGE,
    ARANDU_TERRACOTTA,
    ARANDU_NODE_IVORY,
    ARANDU_TERRACOTTA_DARK,
    ARANDU_SAGE_BORDER,
]

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_JPG_PATH = _ASSETS_DIR / "logo_arandu.jpg"
LOGO_SVG_PATH = _ASSETS_DIR / "svg" / "arandu.svg"

FEEDBACK_FORMS_URL = "https://forms.gle/Sy6M1NWzHRxMbuiBA"
PLOTLY_CONFIG_ARANDU: dict[str, object] = {
    "displayModeBar": "hover",
    "displaylogo": False,
}

_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
}


def _resolver_ruta_asset(ruta_relativa: str) -> Path | None:
    limpia = ruta_relativa.strip().replace("\\", "/").lstrip("/")
    if limpia.startswith("assets/"):
        limpia = limpia[7:]
    for raiz in (_ASSETS_DIR, Path("assets")):
        base = raiz.resolve()
        destino = (base / limpia).resolve()
        if destino != base and destino.is_relative_to(base) and destino.is_file():
            return destino
    return None


@st.cache_resource(show_spinner=False)
def obtener_plotly_arandu_template():
    """Instancia una única referencia global de la plantilla Plotly de Arandú."""
    import plotly.graph_objects as go
    import plotly.io as pio

    template = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(10, 11, 9, 0)",
            plot_bgcolor=ARANDU_PLOT_BG,
            font=dict(
                color=ARANDU_IVORY,
                family="'Plus Jakarta Sans', sans-serif",
                size=13,
            ),
            title=dict(
                font=dict(
                    color=ARANDU_IVORY,
                    family="'Cinzel', 'Fraunces', serif",
                    size=17,
                )
            ),
            colorway=COLORWAY_ARANDU,
            legend=dict(
                bgcolor="rgba(22, 24, 20, 0.72)",
                bordercolor=ARANDU_CARD_BORDER,
                borderwidth=1,
                font=dict(color=ARANDU_IVORY, size=12),
            ),
            hoverlabel=dict(
                bgcolor=ARANDU_BARK,
                bordercolor=ARANDU_TERRACOTTA_DARK,
                font=dict(color=ARANDU_IVORY, family="'JetBrains Mono', monospace", size=12),
            ),
            xaxis=dict(
                gridcolor=ARANDU_GRID,
                linecolor=ARANDU_CARD_BORDER,
                zeroline=True,
                zerolinecolor=ARANDU_TERRACOTTA_DARK,
                zerolinewidth=1.6,
                tickfont=dict(color=ARANDU_IVORY, family="'JetBrains Mono', monospace", size=11),
            ),
            yaxis=dict(
                gridcolor=ARANDU_GRID,
                linecolor=ARANDU_CARD_BORDER,
                zeroline=True,
                zerolinecolor=ARANDU_TERRACOTTA_DARK,
                zerolinewidth=1.6,
                tickfont=dict(color=ARANDU_IVORY, family="'JetBrains Mono', monospace", size=11),
            ),
        )
    )
    pio.templates["arandu"] = template
    pio.templates.default = "arandu"
    return template


def __getattr__(name: str):
    if name == "PLOTLY_ARANDU_TEMPLATE":
        return obtener_plotly_arandu_template()
    raise AttributeError(f"module 'views.theme' has no attribute {name!r}")


def estilizar_figura_arandu(fig: go.Figure) -> go.Figure:
    """Aplica la plantilla y colores orgánicos de Arandú a cualquier figura Plotly."""
    template = obtener_plotly_arandu_template()
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(10, 11, 9, 0)",
        plot_bgcolor=ARANDU_PLOT_BG,
        font=dict(color=ARANDU_IVORY, family="'Plus Jakarta Sans', sans-serif"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=ARANDU_GRID,
        zeroline=True,
        zerolinecolor=ARANDU_TERRACOTTA_DARK,
        zerolinewidth=1.6,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=ARANDU_GRID,
        zeroline=True,
        zerolinecolor=ARANDU_TERRACOTTA_DARK,
        zerolinewidth=1.6,
    )
    return fig


@st.cache_data(show_spinner=False)
def leer_asset_bytes(ruta_relativa: str) -> bytes | None:
    """Lee un archivo binario de assets/ una sola vez y lo retiene en caché."""
    ruta = _resolver_ruta_asset(ruta_relativa)
    return ruta.read_bytes() if ruta is not None else None


@st.cache_data(show_spinner=False)
def leer_asset_texto(ruta_relativa: str) -> str | None:
    """Lee un archivo de texto (p. ej. SVG) de assets/ una sola vez en caché."""
    raw = leer_asset_bytes(ruta_relativa)
    return raw.decode("utf-8") if raw is not None else None


@st.cache_data(show_spinner=False)
def obtener_asset_base64(ruta_relativa: str) -> str | None:
    """Retorna el contenido de un archivo de assets/ (imagen o PDF) codificado en base64."""
    raw = leer_asset_bytes(ruta_relativa)
    return base64.b64encode(raw).decode("ascii") if raw is not None else None


@st.cache_data(show_spinner=False)
def obtener_asset_data_uri(ruta_relativa: str, mime_type: str | None = None) -> str | None:
    """Construye y cachea un data URI RFC 2397 para imágenes o PDFs dentro de assets/."""
    encoded = obtener_asset_base64(ruta_relativa)
    if encoded is None:
        return None
    ext = Path(ruta_relativa).suffix.lower()
    mime = mime_type or _MIME_TYPES.get(ext, "application/octet-stream")
    return f"data:{mime};base64,{encoded}"


@st.cache_data(show_spinner=False)
def obtener_logo_arandu_src(prefer_svg: bool = False) -> str | None:
    """Retorna un data URI cacheado (JPEG o SVG transparente) del logo oficial Arandú."""
    candidatos = ("svg/arandu.svg", "logo_arandu.jpg") if prefer_svg else ("logo_arandu.jpg", "svg/arandu.svg")
    for rel in candidatos:
        uri = obtener_asset_data_uri(rel)
        if uri is not None:
            return uri
    return None


@st.cache_data(show_spinner=False)
def obtener_logos_app() -> tuple[bytes | str | None, str | bytes]:
    """Retorna (sidebar_logo, page_icon) cacheados para app.py sin lecturas repetidas a disco."""
    jpg_bytes = leer_asset_bytes("logo_arandu.jpg")
    svg_txt = leer_asset_texto("svg/arandu.svg")
    sidebar_logo = jpg_bytes if jpg_bytes is not None else svg_txt
    page_icon: str | bytes = svg_txt if svg_txt is not None else (jpg_bytes if jpg_bytes is not None else "🌳")
    return sidebar_logo, page_icon


@st.cache_data(show_spinner=False)
def obtener_css_arandu() -> str:
    """Genera y cachea la hoja CSS y tipografías de Arandú una sola vez en memoria."""
    return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Fraunces:opsz,wght@9..144,600&family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #EDE4D3;
        }
        h1, h2, .arandu-brand-title {
            font-family: 'Cinzel', 'Fraunces', serif !important;
            letter-spacing: 0.02em;
            color: #EDE4D3;
        }
        h3, h4 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #EDE4D3;
        }
        code, pre, .stCode, [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
        }

        .arandu-sidebar-brand {
            text-align: center;
            padding: 0.75rem 0.6rem 0.9rem 0.6rem;
            border-radius: 12px;
            background: linear-gradient(160deg, #161814 0%, #11120F 100%);
            border: 1px solid #2E2920;
            border-bottom: 2px solid #D4A359;
            margin-bottom: 0.75rem;
        }
        .arandu-sidebar-brand img {
            width: 78px;
            height: 78px;
            border-radius: 50%;
            border: 2px solid #D4A359;
            object-fit: cover;
            margin-bottom: 0.45rem;
            box-shadow: 0 4px 14px rgba(212, 163, 89, 0.22);
        }
        .arandu-sidebar-brand-name {
            font-family: 'Cinzel', 'Fraunces', serif;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            color: #D4A359;
            margin: 0;
        }
        .arandu-sidebar-brand-sub {
            font-size: 0.75rem;
            color: #EDE4D3;
            opacity: 0.85;
            margin: 0.2rem 0 0 0;
        }

        .arandu-hero {
            padding: 1.55rem 1.85rem;
            border-radius: 14px;
            border: 1px solid #2E2920;
            border-left: 4px solid #D4A359;
            background: radial-gradient(circle at top right, rgba(122, 154, 101, 0.14), transparent 55%),
                        linear-gradient(135deg, #161814 0%, #0E100C 100%);
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
            margin-bottom: 1.25rem;
        }
        .arandu-hero-grid {
            display: flex;
            align-items: center;
            gap: 1.6rem;
            flex-wrap: wrap;
        }
        .arandu-hero-logo {
            width: 132px;
            height: 132px;
            flex-shrink: 0;
            border-radius: 50%;
            background: #0A0B09;
            border: 2px solid rgba(212, 163, 89, 0.65);
            box-shadow: 0 6px 22px rgba(212, 163, 89, 0.20);
            object-fit: cover;
        }
        .arandu-hero h1 {
            margin: 0.35rem 0 0.45rem 0;
            font-size: 1.95rem;
            color: #EDE4D3;
        }
        .arandu-pillars {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
            gap: 0.65rem;
            margin-top: 0.85rem;
        }
        .arandu-pillar-item {
            padding: 0.55rem 0.75rem;
            border-radius: 9px;
            background-color: rgba(10, 11, 9, 0.55);
            border: 1px solid #2E2920;
            border-left: 3px solid #7A9A65;
            font-size: 0.86rem;
            line-height: 1.35;
        }
        .arandu-pillar-item strong {
            color: #D4A359;
        }

        .arandu-abstract {
            padding: 0.85rem 1.15rem;
            border-radius: 10px;
            border: 1px solid #2E2920;
            border-left: 4px solid #D4A359;
            background: linear-gradient(90deg, rgba(212, 163, 89, 0.10), rgba(22, 24, 20, 0.92));
            margin: 0.35rem 0 0.85rem 0;
            font-size: 0.98rem;
            line-height: 1.5;
        }
        .arandu-verdict-card {
            padding: 0.95rem 1.15rem;
            border-radius: 12px;
            border: 1px solid #2E2920;
            background-color: #161814;
            margin: 0.55rem 0 1rem 0;
        }
        .arandu-verdict-pro {
            border-left: 4px solid #7A9A65 !important;
        }
        .arandu-verdict-warn {
            border-left: 4px solid #C86D44 !important;
        }

        .arandu-card,
        .arandu-guru-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid #2E2920;
            border-left: 4px solid #D4A359;
            background-color: #161814;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.32);
            margin-bottom: 0.8rem;
            height: 100%;
        }
        .arandu-card-sage {
            border-left: 4px solid #687D5B !important;
        }
        .arandu-card h4,
        .arandu-guru-card h4 {
            margin: 0 0 0.45rem 0;
            font-size: 1.02rem;
            color: #D4A359;
        }
        .arandu-card-sage h4 {
            color: #7A9A65;
        }

        .arandu-ejemplo-box {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid #2E2920;
            border-left: 4px solid #687D5B;
            background-color: #161814;
            margin: 0.6rem 0 1rem 0;
        }
        .arandu-edit-banner {
            padding: 0.8rem 1.05rem;
            border-radius: 10px;
            border: 2px dashed #D4A359;
            background-color: rgba(212, 163, 89, 0.10);
            color: #EDE4D3;
            font-weight: 600;
            margin: 0.65rem 0 0.85rem 0;
        }
        div[data-testid="stDataEditor"] {
            border: 2px dashed #D4A359 !important;
            border-radius: 10px !important;
            padding: 0.35rem !important;
            background-color: #161814 !important;
        }

        .arandu-versus-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid #2E2920;
            background-color: #161814;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.30);
            margin-bottom: 0.85rem;
        }
        .arandu-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            color: #D4A359;
            background-color: rgba(212, 163, 89, 0.12);
            border: 1px solid rgba(212, 163, 89, 0.45);
            margin-right: 0.35rem;
        }

        div.stButton > button,
        div.stLinkButton > a {
            border: 1px solid #D4A359 !important;
            background-color: #1E201B !important;
            color: #EDE4D3 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button:hover,
        div.stLinkButton > a:hover {
            background-color: rgba(181, 103, 63, 0.26) !important;
            border-color: #B5673F !important;
            color: #FFFDF9 !important;
            box-shadow: 0 0 12px rgba(212, 163, 89, 0.24) !important;
        }

        button[data-baseweb="tab"] {
            color: #EDE4D3 !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #C86D44 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #D4A359 !important;
            border-bottom-color: #D4A359 !important;
        }

        .arandu-nav-card {
            background-color: #161814;
            border: 1px solid #2E2920;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 0.45rem;
        }
        .arandu-nav-card:hover {
            transform: translateY(-4px);
            border-color: #D4A359;
            box-shadow: 0 10px 24px rgba(212, 163, 89, 0.16);
        }
        .arandu-nav-card-header {
            height: 120px;
            width: 100%;
            background-color: #121410;
            border-bottom: 1px solid #2E2920;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            padding: 0.45rem 0.65rem;
            box-sizing: border-box;
        }
        .arandu-nav-card-header svg {
            width: 100%;
            height: 100%;
            max-height: 106px;
            display: block;
        }
        .arandu-nav-card-body {
            padding: 0.9rem 1.05rem 0.95rem 1.05rem;
        }
        .arandu-nav-card-tag {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: #7A9A65;
            margin-bottom: 0.35rem;
            letter-spacing: 0.02em;
        }
        .arandu-nav-card-title {
            margin: 0 0 0.4rem 0;
            font-family: 'Cinzel', 'Fraunces', serif;
            font-size: 1.02rem;
            font-weight: 700;
            color: #EDE4D3;
            line-height: 1.25;
            min-height: 2.5em;
            display: flex;
            align-items: flex-start;
        }
        .arandu-nav-card:hover .arandu-nav-card-title {
            color: #D4A359;
        }
        .arandu-nav-card-desc {
            margin: 0;
            font-size: 0.87rem;
            line-height: 1.45;
            color: #C7BFA9;
            height: 2.9em;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        .arandu-mini-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
            gap: 0.6rem;
            margin: 0.35rem 0 0.95rem 0;
        }
        .arandu-mini-card {
            padding: 0.65rem 0.85rem;
            border-radius: 10px;
            background-color: #161814;
            border: 1px solid #2E2920;
            border-left: 3px solid #D4A359;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .arandu-mini-card:hover {
            transform: translateY(-2px);
            border-color: #D4A359;
        }
        .arandu-mini-card-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: #EDE4D3;
            margin: 0 0 0.2rem 0;
        }
        .arandu-mini-card-desc {
            font-size: 0.8rem;
            line-height: 1.35;
            color: #C7BFA9;
            margin: 0;
        }
        @media (max-width: 768px) {
            .arandu-nav-card-header {
                height: auto;
                min-height: 108px;
            }
            .arandu-nav-card-title {
                min-height: auto;
            }
            .arandu-nav-card-desc {
                height: auto;
                min-height: 2.9em;
                -webkit-line-clamp: 3;
            }
            .arandu-mini-grid {
                grid-template-columns: 1fr;
            }
        }

        [data-testid="stSkeleton"],
        .arandu-skeleton-box {
            background: linear-gradient(
                90deg,
                #161814 25%,
                #26241E 50%,
                #161814 75%
            ) !important;
            background-size: 200% 100% !important;
            border: 1px solid #2E2920 !important;
            border-radius: 10px !important;
            animation: arandu-skeleton-shimmer 1.6s ease-in-out infinite !important;
        }
        @keyframes arandu-skeleton-shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        </style>
        """


@st.cache_data(show_spinner=False)
def obtener_svg_tarjeta_arandu(clave_svg: str) -> str:
    """Retorna una ilustración SVG minimalista e inline en paleta Arandú para tarjetas del catálogo."""
    svgs: dict[str, str] = {
        "fundamentos": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Precisión binaria y series escalonadas">'
            '<line x1="22" y1="98" x2="298" y2="98" stroke="#2E2920" stroke-width="1.5"/>'
            '<rect x="34" y="64" width="24" height="34" rx="3" fill="#687D5B" opacity="0.45" stroke="#687D5B"/>'
            '<rect x="64" y="50" width="24" height="48" rx="3" fill="#687D5B" opacity="0.65" stroke="#687D5B"/>'
            '<rect x="94" y="38" width="24" height="60" rx="3" fill="#D4A359" opacity="0.45" stroke="#D4A359"/>'
            '<rect x="124" y="29" width="24" height="69" rx="3" fill="#D4A359" opacity="0.72" stroke="#D4A359"/>'
            '<path d="M164,86 Q216,74 286,24" fill="none" stroke="#EDE4D3" stroke-width="2" stroke-dasharray="4 3"/>'
            '<polyline points="164,86 196,86 196,70 228,70 228,51 260,51 260,31 286,31" fill="none" stroke="#D4A359" stroke-width="2.4"/>'
            '<rect x="260" y="24" width="12" height="7" rx="1.5" fill="#B5673F"/>'
            '<circle cx="286" cy="24" r="4" fill="#D4A359"/>'
            '</svg>'
        ),
        "raices": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Curva cruzando el eje X con tangente">'
            '<line x1="24" y1="74" x2="296" y2="74" stroke="#687D5B" stroke-width="1.5" stroke-dasharray="4 4"/>'
            '<line x1="54" y1="16" x2="54" y2="104" stroke="#2E2920" stroke-width="1.4"/>'
            '<path d="M40,98 C105,94 148,54 190,42 C232,30 258,20 282,16" fill="none" stroke="#EDE4D3" stroke-width="2.6"/>'
            '<line x1="92" y1="102" x2="262" y2="18" stroke="#D4A359" stroke-width="2" stroke-dasharray="5 3"/>'
            '<line x1="224" y1="36" x2="224" y2="74" stroke="#B5673F" stroke-width="1.5" stroke-dasharray="3 3"/>'
            '<circle cx="224" cy="36" r="4.5" fill="#B5673F" stroke="#EDE4D3" stroke-width="1.2"/>'
            '<circle cx="149" cy="74" r="5.5" fill="#D4A359" stroke="#121410" stroke-width="1.5"/>'
            '<circle cx="131" cy="74" r="3.5" fill="#687D5B"/>'
            '</svg>'
        ),
        "sistemas": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Matriz diagonal iluminada y autovectores">'
            '<g transform="translate(36,20)">'
            '<rect x="0" y="0" width="84" height="80" rx="6" fill="none" stroke="#2E2920" stroke-width="1.5"/>'
            '<rect x="8" y="8" width="15" height="14" rx="2.5" fill="#D4A359"/>'
            '<rect x="27" y="8" width="15" height="14" rx="2.5" fill="#687D5B" opacity="0.35"/>'
            '<rect x="46" y="8" width="15" height="14" rx="2.5" fill="#687D5B" opacity="0.25"/>'
            '<rect x="65" y="8" width="13" height="14" rx="2.5" fill="#687D5B" opacity="0.2"/>'
            '<rect x="8" y="25" width="15" height="14" rx="2.5" fill="#2E2920" opacity="0.4"/>'
            '<rect x="27" y="25" width="15" height="14" rx="2.5" fill="#D4A359"/>'
            '<rect x="46" y="25" width="15" height="14" rx="2.5" fill="#687D5B" opacity="0.35"/>'
            '<rect x="65" y="25" width="13" height="14" rx="2.5" fill="#687D5B" opacity="0.25"/>'
            '<rect x="8" y="42" width="15" height="14" rx="2.5" fill="#2E2920" opacity="0.35"/>'
            '<rect x="27" y="42" width="15" height="14" rx="2.5" fill="#2E2920" opacity="0.4"/>'
            '<rect x="46" y="42" width="15" height="14" rx="2.5" fill="#D4A359"/>'
            '<rect x="65" y="42" width="13" height="14" rx="2.5" fill="#687D5B" opacity="0.35"/>'
            '<rect x="8" y="59" width="15" height="13" rx="2.5" fill="#2E2920" opacity="0.3"/>'
            '<rect x="27" y="59" width="15" height="13" rx="2.5" fill="#2E2920" opacity="0.35"/>'
            '<rect x="46" y="59" width="15" height="13" rx="2.5" fill="#2E2920" opacity="0.4"/>'
            '<rect x="65" y="59" width="13" height="13" rx="2.5" fill="#D4A359"/>'
            '</g>'
            '<line x1="168" y1="94" x2="286" y2="94" stroke="#2E2920" stroke-width="1.5"/>'
            '<line x1="168" y1="94" x2="168" y2="20" stroke="#2E2920" stroke-width="1.5"/>'
            '<line x1="168" y1="94" x2="266" y2="32" stroke="#D4A359" stroke-width="2.6"/>'
            '<polygon points="272,27 260,30 266,38" fill="#D4A359"/>'
            '<line x1="168" y1="94" x2="244" y2="76" stroke="#B5673F" stroke-width="2.2"/>'
            '<polygon points="251,74 240,72 243,81" fill="#B5673F"/>'
            '<circle cx="168" cy="94" r="4" fill="#EDE4D3"/>'
            '</svg>'
        ),
        "interpolacion": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Nodos discretos unidos por curva polinomial">'
            '<line x1="22" y1="96" x2="298" y2="96" stroke="#2E2920" stroke-width="1.4"/>'
            '<path d="M38,84 C68,36 102,26 136,54 C168,80 198,86 232,46 C252,24 270,22 286,36" fill="none" stroke="#D4A359" stroke-width="2.6"/>'
            '<line x1="184" y1="74" x2="184" y2="96" stroke="#687D5B" stroke-width="1.5" stroke-dasharray="3 3"/>'
            '<circle cx="38" cy="84" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.6"/>'
            '<circle cx="92" cy="36" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.6"/>'
            '<circle cx="146" cy="61" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.6"/>'
            '<circle cx="222" cy="56" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.6"/>'
            '<circle cx="276" cy="28" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.6"/>'
            '<circle cx="184" cy="74" r="5" fill="#687D5B" stroke="#EDE4D3" stroke-width="1.5"/>'
            '</svg>'
        ),
        "integracion": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Área bajo la curva con trapecios y arcos parabólicos">'
            '<line x1="24" y1="96" x2="296" y2="96" stroke="#2E2920" stroke-width="1.5"/>'
            '<polygon points="46,96 46,72 98,54 98,96" fill="#687D5B" opacity="0.32" stroke="#687D5B" stroke-width="1.3"/>'
            '<polygon points="98,96 98,54 150,36 150,96" fill="#687D5B" opacity="0.24" stroke="#687D5B" stroke-width="1.3"/>'
            '<polygon points="150,96 150,36 202,44 202,96" fill="#D4A359" opacity="0.24" stroke="#D4A359" stroke-width="1.3"/>'
            '<polygon points="202,96 202,44 254,66 254,96" fill="#D4A359" opacity="0.18" stroke="#D4A359" stroke-width="1.3"/>'
            '<path d="M46,72 Q98,42 150,36 Q202,32 254,66" fill="none" stroke="#EDE4D3" stroke-width="2.5"/>'
            '<path d="M150,36 Q202,28 254,66" fill="none" stroke="#D4A359" stroke-width="2" stroke-dasharray="4 3"/>'
            '<circle cx="46" cy="72" r="3.5" fill="#EDE4D3"/>'
            '<circle cx="98" cy="54" r="3.5" fill="#EDE4D3"/>'
            '<circle cx="150" cy="36" r="4" fill="#D4A359"/>'
            '<circle cx="202" cy="44" r="3.5" fill="#B5673F"/>'
            '<circle cx="254" cy="66" r="3.5" fill="#EDE4D3"/>'
            '</svg>'
        ),
        "edo": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Campo de pendientes y trayectoria paso a paso">'
            '<g stroke="#687D5B" stroke-width="1.4" opacity="0.72">'
            '<line x1="36" y1="30" x2="52" y2="24"/><line x1="92" y1="28" x2="108" y2="23"/><line x1="148" y1="32" x2="164" y2="28"/><line x1="204" y1="40" x2="220" y2="34"/><line x1="260" y1="30" x2="276" y2="22"/>'
            '<line x1="36" y1="60" x2="52" y2="52"/><line x1="92" y1="56" x2="108" y2="47"/><line x1="148" y1="58" x2="164" y2="50"/><line x1="204" y1="64" x2="220" y2="54"/><line x1="260" y1="56" x2="276" y2="44"/>'
            '<line x1="36" y1="90" x2="52" y2="80"/><line x1="92" y1="86" x2="108" y2="75"/><line x1="148" y1="88" x2="164" y2="78"/><line x1="204" y1="92" x2="220" y2="82"/><line x1="260" y1="84" x2="276" y2="70"/>'
            '</g>'
            '<polyline points="42,92 98,74 156,56 214,42 272,24" fill="none" stroke="#D4A359" stroke-width="2.5"/>'
            '<circle cx="42" cy="92" r="4" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.4"/>'
            '<circle cx="98" cy="74" r="3.8" fill="#D4A359"/>'
            '<circle cx="156" cy="56" r="3.8" fill="#D4A359"/>'
            '<circle cx="214" cy="42" r="3.8" fill="#D4A359"/>'
            '<circle cx="272" cy="24" r="4.5" fill="#B5673F" stroke="#EDE4D3" stroke-width="1.3"/>'
            '</svg>'
        ),
        "arena_versus": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dos curvas de convergencia compitiendo hacia un punto">'
            '<line x1="24" y1="94" x2="296" y2="94" stroke="#2E2920" stroke-width="1.5"/>'
            '<line x1="24" y1="58" x2="296" y2="58" stroke="#687D5B" stroke-width="1.4" stroke-dasharray="4 4"/>'
            '<path d="M36,24 C86,28 112,56 172,58 L268,58" fill="none" stroke="#D4A359" stroke-width="2.7"/>'
            '<path d="M36,88 C108,86 164,74 228,62 L268,58" fill="none" stroke="#B5673F" stroke-width="2.5"/>'
            '<circle cx="172" cy="58" r="4" fill="#D4A359"/>'
            '<circle cx="268" cy="58" r="6" fill="#EDE4D3" stroke="#D4A359" stroke-width="2.2"/>'
            '<circle cx="36" cy="24" r="3.5" fill="#D4A359"/>'
            '<circle cx="36" cy="88" r="3.5" fill="#B5673F"/>'
            '</svg>'
        ),
        "laboratorios": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Progresión de Fibonacci hacia la razón áurea">'
            '<line x1="24" y1="98" x2="296" y2="98" stroke="#2E2920" stroke-width="1.5"/>'
            '<line x1="24" y1="38" x2="296" y2="38" stroke="#D4A359" stroke-width="1.4" stroke-dasharray="5 4"/>'
            '<rect x="40" y="82" width="24" height="16" rx="2.5" fill="#687D5B" opacity="0.6"/>'
            '<rect x="76" y="72" width="24" height="26" rx="2.5" fill="#687D5B" opacity="0.75"/>'
            '<rect x="112" y="60" width="24" height="38" rx="2.5" fill="#D4A359" opacity="0.6"/>'
            '<rect x="148" y="48" width="24" height="50" rx="2.5" fill="#D4A359" opacity="0.8"/>'
            '<rect x="184" y="38" width="24" height="60" rx="2.5" fill="#B5673F" opacity="0.8"/>'
            '<polyline points="52,74 88,52 124,42 160,39 196,38 268,38" fill="none" stroke="#EDE4D3" stroke-width="2.2"/>'
            '<circle cx="268" cy="38" r="4.5" fill="#D4A359"/>'
            '</svg>'
        ),
        "feedback": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Aportes y evaluación docente y estudiantil">'
            '<rect x="78" y="24" width="164" height="64" rx="10" fill="#161814" stroke="#D4A359" stroke-width="1.8"/>'
            '<line x1="98" y1="44" x2="202" y2="44" stroke="#EDE4D3" stroke-width="2.4" stroke-linecap="round"/>'
            '<line x1="98" y1="58" x2="220" y2="58" stroke="#687D5B" stroke-width="2.2" stroke-linecap="round"/>'
            '<line x1="98" y1="72" x2="174" y2="72" stroke="#B5673F" stroke-width="2.2" stroke-linecap="round"/>'
            '<circle cx="242" cy="24" r="9" fill="#D4A359"/>'
            '<polygon points="104,88 122,88 104,102" fill="#161814" stroke="#D4A359" stroke-width="1.6"/>'
            '</svg>'
        ),
    }
    return svgs.get(clave_svg, svgs["arena_versus"])


@st.cache_data(show_spinner=False)
def construir_html_tarjeta_catalogo(
    clave_svg: str,
    titulo: str,
    descripcion: str,
    subtitulo_tag: str = "",
) -> str:
    """Construye y cachea el HTML completo de una tarjeta visual Refactoring.Guru en Arandú."""
    svg_inline = obtener_svg_tarjeta_arandu(clave_svg)
    tag_html = (
        f'<span class="arandu-nav-card-tag">{html.escape(subtitulo_tag)}</span>'
        if subtitulo_tag
        else ""
    )
    return (
        '<div class="arandu-nav-card">'
        f'<div class="arandu-nav-card-header">{svg_inline}</div>'
        '<div class="arandu-nav-card-body">'
        f"{tag_html}"
        f'<div class="arandu-nav-card-title">{html.escape(titulo)}</div>'
        f'<p class="arandu-nav-card-desc">{html.escape(descripcion)}</p>'
        "</div>"
        "</div>"
    )


@st.cache_data(show_spinner=False)
def construir_html_mini_tarjetas(items: tuple[tuple[str, str], ...]) -> str:
    """Construye y cachea una fila compacta de mini-tarjetas explicativas de métodos."""
    cards = "".join(
        '<div class="arandu-mini-card">'
        f'<div class="arandu-mini-card-title">{html.escape(titulo)}</div>'
        f'<p class="arandu-mini-card-desc">{html.escape(desc)}</p>'
        "</div>"
        for titulo, desc in items
    )
    return f'<div class="arandu-mini-grid">{cards}</div>'


def aplicar_estilo_arandu() -> None:
    """Inyecta tipografías, tarjetas Guru, tablas doradas y controles orgánicos de Arandú."""
    st.markdown(obtener_css_arandu(), unsafe_allow_html=True)

