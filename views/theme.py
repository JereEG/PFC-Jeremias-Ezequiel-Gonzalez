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

        .arandu-tarot-card {
            background: linear-gradient(175deg, #171A14 0%, #10120E 58%, #16130E 100%);
            border: 1px solid #2E2920;
            border-radius: 14px;
            padding: 0.72rem 0.78rem 0.82rem 0.78rem;
            box-shadow: inset 0 0 0 4px #11130F, inset 0 0 0 5px rgba(212, 163, 89, 0.22), 0 8px 20px rgba(0, 0, 0, 0.44);
            margin-bottom: 0.45rem;
            transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
            position: relative;
            overflow: hidden;
        }
        .arandu-tarot-card:hover {
            transform: translateY(-6px) rotate(-0.45deg);
            border-color: #D4A359;
            box-shadow: inset 0 0 0 4px #11130F, inset 0 0 0 5px rgba(212, 163, 89, 0.56), 0 14px 28px rgba(212, 163, 89, 0.18);
        }
        .arandu-tarot-card.is-active {
            background: radial-gradient(circle at 50% 34%, rgba(212, 163, 89, 0.15), transparent 64%),
                        linear-gradient(175deg, #1E2018 0%, #121410 62%, #1C170F 100%);
            border: 1.5px solid #D4A359;
            transform: translateY(-4px);
            box-shadow: inset 0 0 0 4px #121410, inset 0 0 0 5px rgba(212, 163, 89, 0.84), 0 14px 28px rgba(212, 163, 89, 0.26);
        }
        .arandu-tarot-top {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
            margin-bottom: 0.14rem;
        }
        .arandu-tarot-top::before,
        .arandu-tarot-top::after {
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(212, 163, 89, 0.45), transparent);
        }
        .arandu-tarot-num {
            font-family: 'Cinzel', 'Fraunces', serif;
            font-size: 0.92rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            color: #D4A359;
            line-height: 1;
        }
        .arandu-tarot-badge {
            display: inline-block;
            margin-left: 0.25rem;
            padding: 0.08rem 0.38rem;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #121410;
            background-color: #D4A359;
        }
        .arandu-tarot-suit {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.64rem;
            letter-spacing: 0.09em;
            color: #7A9A65;
            text-align: center;
            margin: 0.1rem 0 0.42rem 0;
            text-transform: uppercase;
        }
        .arandu-tarot-card.is-active .arandu-tarot-suit {
            color: #D4A359;
        }
        .arandu-tarot-art {
            width: 100%;
            height: 158px;
            background: #0C0E0A;
            border: 1px solid #2E2920;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.3rem;
            box-sizing: border-box;
            margin-bottom: 0.56rem;
            overflow: hidden;
        }
        .arandu-tarot-card.is-active .arandu-tarot-art {
            border-color: rgba(212, 163, 89, 0.65);
            background: radial-gradient(circle at 50% 45%, rgba(212, 163, 89, 0.09), #0C0E0A 72%);
        }
        .arandu-tarot-art svg {
            width: 100%;
            height: 100%;
            max-height: 148px;
            display: block;
        }
        .arandu-tarot-title {
            font-family: 'Cinzel', 'Fraunces', serif;
            font-size: 0.92rem;
            font-weight: 700;
            color: #EDE4D3;
            text-align: center;
            line-height: 1.24;
            min-height: 2.55em;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 0 0.32rem 0;
        }
        .arandu-tarot-card:hover .arandu-tarot-title,
        .arandu-tarot-card.is-active .arandu-tarot-title {
            color: #D4A359;
        }
        .arandu-tarot-desc {
            font-size: 0.79rem;
            line-height: 1.38;
            color: #C7BFA9;
            text-align: center;
            height: 2.8em;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            margin: 0 0 0.48rem 0;
        }
        .arandu-tarot-foot {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: #D4A359;
            border-top: 1px solid rgba(212, 163, 89, 0.24);
            padding-top: 0.38rem;
            text-align: center;
            letter-spacing: 0.03em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .arandu-tarot-mesa-banner {
            padding: 0.65rem 1.05rem;
            border-radius: 10px;
            border: 1px solid #2E2920;
            border-left: 4px solid #D4A359;
            background: linear-gradient(90deg, rgba(212, 163, 89, 0.12), #161814 55%);
            margin: 0.55rem 0 0.9rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .arandu-tarot-mesa-titulo {
            font-family: 'Cinzel', 'Fraunces', serif;
            font-size: 1.02rem;
            font-weight: 700;
            color: #D4A359;
        }
        .arandu-tarot-mesa-pista {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: #7A9A65;
        }
        @media (max-width: 768px) {
            .arandu-tarot-art {
                height: 132px;
            }
            .arandu-tarot-title,
            .arandu-tarot-desc {
                min-height: auto;
                height: auto;
                -webkit-line-clamp: 3;
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


_TAROT_FRAME_PREFIX = (
    '<svg viewBox="0 0 180 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}">'
    '<rect x="5" y="5" width="170" height="180" rx="8" fill="#0C0E0A" stroke="#2E2920" stroke-width="1.2"/>'
    '<rect x="10" y="10" width="160" height="170" rx="5" fill="none" stroke="#D4A359" stroke-opacity="0.28" stroke-width="0.9"/>'
    '<path d="M22,164 L22,66 A68,48 0 0,1 158,66 L158,164 Z" fill="#121410" stroke="#2E2920" stroke-width="1.2"/>'
    '<polygon points="18,15 21,18 18,21 15,18" fill="#D4A359" opacity="0.7"/>'
    '<polygon points="162,15 165,18 162,21 159,18" fill="#D4A359" opacity="0.7"/>'
    '<polygon points="18,169 21,172 18,175 15,172" fill="#D4A359" opacity="0.7"/>'
    '<polygon points="162,169 165,172 162,175 159,172" fill="#D4A359" opacity="0.7"/>'
    '<circle cx="90" cy="24" r="2.8" fill="#D4A359"/>'
    '<line x1="64" y1="24" x2="82" y2="24" stroke="#D4A359" stroke-opacity="0.45" stroke-width="1"/>'
    '<line x1="98" y1="24" x2="116" y2="24" stroke="#D4A359" stroke-opacity="0.45" stroke-width="1"/>'
)

_TAROT_BODY_SVGS: dict[str, tuple[str, str]] = {
    "tarot_taylor": (
        "Arcano Series de Taylor",
        '<line x1="30" y1="146" x2="150" y2="146" stroke="#2E2920" stroke-width="1.2"/>'
        '<path d="M36,142 Q96,128 144,48" fill="none" stroke="#EDE4D3" stroke-width="2" stroke-dasharray="4 3"/>'
        '<polyline points="36,142 64,142 64,124 92,124 92,98 120,98 120,66 144,66" fill="none" stroke="#D4A359" stroke-width="2.4"/>'
        '<rect x="120" y="54" width="24" height="12" rx="2" fill="#687D5B" opacity="0.55" stroke="#687D5B"/>'
        '<circle cx="144" cy="48" r="4.5" fill="#D4A359" stroke="#121410" stroke-width="1.2"/>',
    ),
    "tarot_aritmetica": (
        "Arcano Aritmética de Máquina",
        '<line x1="90" y1="42" x2="90" y2="148" stroke="#D4A359" stroke-width="1.8"/>'
        '<polygon points="72,148 108,148 96,138 84,138" fill="#2E2920" stroke="#D4A359" stroke-width="1"/>'
        '<line x1="40" y1="74" x2="140" y2="56" stroke="#EDE4D3" stroke-width="2.2"/>'
        '<line x1="44" y1="74" x2="36" y2="108" stroke="#687D5B" stroke-width="1.3"/><line x1="44" y1="74" x2="54" y2="108" stroke="#687D5B" stroke-width="1.3"/>'
        '<path d="M32,108 Q45,118 58,108 Z" fill="#687D5B" opacity="0.7" stroke="#687D5B"/>'
        '<line x1="136" y1="56" x2="126" y2="92" stroke="#B5673F" stroke-width="1.3"/><line x1="136" y1="56" x2="146" y2="92" stroke="#B5673F" stroke-width="1.3"/>'
        '<path d="M122,92 Q136,102 150,92 Z" fill="#B5673F" opacity="0.7" stroke="#B5673F"/>'
        '<circle cx="90" cy="65" r="4" fill="#D4A359"/>',
    ),
    "tarot_cancelacion": (
        "Arcano Cancelación Cuadrática",
        '<line x1="30" y1="134" x2="150" y2="134" stroke="#2E2920" stroke-width="1.2"/>'
        '<line x1="36" y1="68" x2="138" y2="94" stroke="#EDE4D3" stroke-width="2.2"/>'
        '<line x1="36" y1="74" x2="138" y2="97" stroke="#B5673F" stroke-width="2" stroke-dasharray="3 2"/>'
        '<circle cx="138" cy="95" r="14" fill="none" stroke="#D4A359" stroke-width="1.8"/>'
        '<line x1="126" y1="95" x2="150" y2="95" stroke="#D4A359" stroke-width="1.4"/>'
        '<circle cx="138" cy="95" r="3.5" fill="#D4A359"/>'
        '<path d="M46,120 Q90,104 134,120" fill="none" stroke="#687D5B" stroke-width="1.6"/>',
    ),
    "tarot_horner": (
        "Arcano Forma Anidada de Horner",
        '<rect x="34" y="52" width="112" height="96" rx="12" fill="none" stroke="#687D5B" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<rect x="48" y="66" width="84" height="68" rx="9" fill="none" stroke="#D4A359" stroke-opacity="0.7" stroke-width="1.6"/>'
        '<rect x="64" y="82" width="52" height="36" rx="6" fill="#181A14" stroke="#D4A359" stroke-width="2"/>'
        '<polygon points="90,89 99,100 90,111 81,100" fill="#D4A359"/>'
        '<circle cx="34" cy="100" r="3.5" fill="#EDE4D3"/><circle cx="146" cy="100" r="3.5" fill="#B5673F"/>',
    ),
    "tarot_biseccion": (
        "Arcano Bisección y Bolzano",
        '<line x1="28" y1="100" x2="152" y2="100" stroke="#687D5B" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<line x1="46" y1="54" x2="46" y2="148" stroke="#687D5B" stroke-width="1.6"/>'
        '<line x1="134" y1="54" x2="134" y2="148" stroke="#B5673F" stroke-width="1.6"/>'
        '<line x1="90" y1="46" x2="90" y2="152" stroke="#D4A359" stroke-width="2" stroke-dasharray="3 2"/>'
        '<path d="M36,138 C64,128 112,72 144,58" fill="none" stroke="#EDE4D3" stroke-width="2.4"/>'
        '<circle cx="46" cy="134" r="3.8" fill="#687D5B"/><circle cx="134" cy="63" r="3.8" fill="#B5673F"/>'
        '<polygon points="90,91 96,100 90,109 84,100" fill="#D4A359" stroke="#121410" stroke-width="1"/>',
    ),
    "tarot_regula_falsi": (
        "Arcano Regula Falsi Secante",
        '<line x1="28" y1="100" x2="152" y2="100" stroke="#687D5B" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<path d="M38,142 Q96,126 142,54" fill="none" stroke="#EDE4D3" stroke-width="2.3"/>'
        '<line x1="42" y1="140" x2="138" y2="60" stroke="#D4A359" stroke-width="2"/>'
        '<line x1="138" y1="60" x2="138" y2="100" stroke="#B5673F" stroke-width="1.2" stroke-dasharray="2 2"/>'
        '<circle cx="42" cy="140" r="4" fill="#687D5B"/><circle cx="138" cy="60" r="4" fill="#B5673F"/>'
        '<circle cx="90" cy="100" r="5" fill="#D4A359" stroke="#121410" stroke-width="1.3"/>'
        '<circle cx="107" cy="100" r="3.2" fill="#EDE4D3"/>',
    ),
    "tarot_newton_raphson": (
        "Arcano Newton-Raphson Tangente",
        '<line x1="28" y1="126" x2="152" y2="126" stroke="#687D5B" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<path d="M40,148 Q106,124 142,46" fill="none" stroke="#EDE4D3" stroke-width="2.4"/>'
        '<line x1="74" y1="144" x2="148" y2="48" stroke="#D4A359" stroke-width="2.2"/>'
        '<line x1="130" y1="71" x2="130" y2="126" stroke="#B5673F" stroke-width="1.4" stroke-dasharray="3 2"/>'
        '<circle cx="130" cy="71" r="4.5" fill="#B5673F" stroke="#EDE4D3" stroke-width="1.1"/>'
        '<circle cx="88" cy="126" r="5.5" fill="#D4A359" stroke="#121410" stroke-width="1.4"/>'
        '<circle cx="103" cy="126" r="3.2" fill="#687D5B"/>',
    ),
    "tarot_punto_fijo": (
        "Arcano Punto Fijo y Aitken",
        '<line x1="36" y1="146" x2="144" y2="48" stroke="#687D5B" stroke-width="1.6" stroke-dasharray="4 3"/>'
        '<path d="M34,116 Q88,108 144,64" fill="none" stroke="#EDE4D3" stroke-width="2.3"/>'
        '<polyline points="48,146 48,113 84,113 84,99 100,99 100,91" fill="none" stroke="#D4A359" stroke-width="1.9"/>'
        '<path d="M48,146 Q92,138 106,86" fill="none" stroke="#B5673F" stroke-width="1.6" stroke-dasharray="3 2"/>'
        '<circle cx="106" cy="83" r="5" fill="#D4A359" stroke="#121410" stroke-width="1.3"/>',
    ),
    "tarot_gauss": (
        "Arcano Eliminación de Gauss",
        '<g transform="translate(46,56)">'
        '<rect x="0" y="0" width="24" height="24" rx="3" fill="#D4A359"/><rect x="30" y="0" width="24" height="24" rx="3" fill="#687D5B" opacity="0.65"/><rect x="60" y="0" width="24" height="24" rx="3" fill="#687D5B" opacity="0.45"/>'
        '<rect x="0" y="30" width="24" height="24" rx="3" fill="#2E2920" opacity="0.35"/><rect x="30" y="30" width="24" height="24" rx="3" fill="#D4A359"/><rect x="60" y="30" width="24" height="24" rx="3" fill="#687D5B" opacity="0.55"/>'
        '<rect x="0" y="60" width="24" height="24" rx="3" fill="#2E2920" opacity="0.25"/><rect x="30" y="60" width="24" height="24" rx="3" fill="#2E2920" opacity="0.35"/><rect x="60" y="60" width="24" height="24" rx="3" fill="#D4A359"/>'
        '</g>',
    ),
    "tarot_gauss_jordan": (
        "Arcano Gauss-Jordan Diagonal",
        '<g transform="translate(46,56)">'
        '<rect x="0" y="0" width="24" height="24" rx="3" fill="#D4A359"/><rect x="30" y="0" width="24" height="24" rx="3" fill="#2E2920" opacity="0.3"/><rect x="60" y="0" width="24" height="24" rx="3" fill="#2E2920" opacity="0.3"/>'
        '<rect x="0" y="30" width="24" height="24" rx="3" fill="#2E2920" opacity="0.3"/><rect x="30" y="30" width="24" height="24" rx="3" fill="#D4A359"/><rect x="60" y="30" width="24" height="24" rx="3" fill="#2E2920" opacity="0.3"/>'
        '<rect x="0" y="60" width="24" height="24" rx="3" fill="#2E2920" opacity="0.3"/><rect x="30" y="60" width="24" height="24" rx="3" fill="#2E2920" opacity="0.3"/><rect x="60" y="60" width="24" height="24" rx="3" fill="#D4A359"/>'
        '<line x1="-4" y1="-4" x2="88" y2="88" stroke="#EDE4D3" stroke-width="1.6" stroke-dasharray="3 3"/>'
        '</g>',
    ),
    "tarot_lu": (
        "Arcano Factorización LU",
        '<polygon points="44,62 44,142 124,142" fill="#687D5B" opacity="0.55" stroke="#687D5B" stroke-width="2"/>'
        '<polygon points="56,56 136,56 136,136" fill="#D4A359" opacity="0.52" stroke="#D4A359" stroke-width="2"/>'
        '<line x1="40" y1="54" x2="140" y2="144" stroke="#EDE4D3" stroke-width="1.6" stroke-dasharray="4 3"/>',
    ),
    "tarot_gauss_seidel": (
        "Arcano Iteración Gauss-Seidel",
        '<circle cx="90" cy="100" r="38" fill="none" stroke="#2E2920" stroke-width="1.5"/>'
        '<path d="M90,62 A38,38 0 0,1 128,100" fill="none" stroke="#D4A359" stroke-width="2.4"/>'
        '<path d="M128,100 A38,38 0 0,1 90,138" fill="none" stroke="#687D5B" stroke-width="2.4"/>'
        '<path d="M90,138 A38,38 0 0,1 52,100" fill="none" stroke="#B5673F" stroke-width="2.4"/>'
        '<path d="M52,100 A38,38 0 0,1 90,62" fill="none" stroke="#EDE4D3" stroke-width="2"/>'
        '<circle cx="90" cy="62" r="4.5" fill="#D4A359"/><circle cx="128" cy="100" r="4.5" fill="#687D5B"/>'
        '<circle cx="90" cy="138" r="4.5" fill="#B5673F"/><circle cx="52" cy="100" r="4.5" fill="#EDE4D3"/>'
        '<circle cx="90" cy="100" r="6" fill="#D4A359"/>',
    ),
    "tarot_potencias": (
        "Arcano Método de Potencias",
        '<line x1="42" y1="142" x2="144" y2="142" stroke="#2E2920" stroke-width="1.3"/>'
        '<line x1="42" y1="142" x2="42" y2="54" stroke="#2E2920" stroke-width="1.3"/>'
        '<line x1="42" y1="142" x2="128" y2="122" stroke="#687D5B" stroke-width="1.8" stroke-dasharray="3 2"/>'
        '<line x1="42" y1="142" x2="136" y2="96" stroke="#B5673F" stroke-width="2"/>'
        '<line x1="42" y1="142" x2="138" y2="62" stroke="#D4A359" stroke-width="2.8"/>'
        '<polygon points="144,57 132,61 139,69" fill="#D4A359"/>'
        '<circle cx="42" cy="142" r="4" fill="#EDE4D3"/>',
    ),
    "tarot_faddeev": (
        "Arcano Faddeev-Leverrier",
        '<line x1="30" y1="100" x2="150" y2="100" stroke="#2E2920" stroke-width="1.3"/>'
        '<path d="M36,64 C56,144 74,144 90,100 C106,56 124,56 144,136" fill="none" stroke="#D4A359" stroke-width="2.4"/>'
        '<circle cx="51" cy="100" r="4.5" fill="#687D5B" stroke="#EDE4D3" stroke-width="1"/>'
        '<circle cx="90" cy="100" r="5" fill="#D4A359" stroke="#121410" stroke-width="1.3"/>'
        '<circle cx="129" cy="100" r="4.5" fill="#B5673F" stroke="#EDE4D3" stroke-width="1"/>',
    ),
    "tarot_interp_lineal": (
        "Arcano Interpolación Lineal",
        '<line x1="30" y1="142" x2="150" y2="142" stroke="#2E2920" stroke-width="1.2"/>'
        '<line x1="46" y1="126" x2="134" y2="66" stroke="#D4A359" stroke-width="2.6"/>'
        '<line x1="90" y1="96" x2="90" y2="142" stroke="#687D5B" stroke-width="1.5" stroke-dasharray="3 2"/>'
        '<circle cx="46" cy="126" r="5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.5"/>'
        '<circle cx="134" cy="66" r="5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.5"/>'
        '<circle cx="90" cy="96" r="4.5" fill="#687D5B" stroke="#EDE4D3" stroke-width="1.3"/>',
    ),
    "tarot_cuad_inversa": (
        "Arcano Cuadrática Inversa",
        '<line x1="34" y1="98" x2="146" y2="98" stroke="#687D5B" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<path d="M40,66 Q90,158 140,66" fill="none" stroke="#EDE4D3" stroke-width="2.3"/>'
        '<circle cx="46" cy="76" r="4.2" fill="#B5673F"/><circle cx="90" cy="112" r="4.2" fill="#B5673F"/><circle cx="134" cy="76" r="4.2" fill="#B5673F"/>'
        '<circle cx="114" cy="98" r="5" fill="#D4A359" stroke="#121410" stroke-width="1.3"/>',
    ),
    "tarot_lagrange": (
        "Arcano Polinomio de Lagrange",
        '<path d="M36,136 C64,56 106,142 144,62" fill="none" stroke="#D4A359" stroke-width="2.5"/>'
        '<path d="M36,78 Q90,136 144,122" fill="none" stroke="#687D5B" stroke-width="1.4" stroke-dasharray="3 3"/>'
        '<circle cx="44" cy="114" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.4"/>'
        '<circle cx="72" cy="82" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.4"/>'
        '<circle cx="112" cy="110" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.4"/>'
        '<circle cx="138" cy="70" r="4.5" fill="#EDE4D3" stroke="#B5673F" stroke-width="1.4"/>',
    ),
    "tarot_newton_gregory": (
        "Arcano Newton-Gregory Diferencias",
        '<line x1="34" y1="142" x2="146" y2="142" stroke="#2E2920" stroke-width="1.2"/>'
        '<path d="M44,128 Q92,102 138,56" fill="none" stroke="#D4A359" stroke-width="2.5"/>'
        '<polyline points="44,128 74,128 74,110 106,110 106,86 138,86 138,56" fill="none" stroke="#687D5B" stroke-width="1.6" stroke-dasharray="3 2"/>'
        '<circle cx="44" cy="128" r="4" fill="#EDE4D3"/><circle cx="74" cy="110" r="4" fill="#EDE4D3"/>'
        '<circle cx="106" cy="86" r="4" fill="#EDE4D3"/><circle cx="138" cy="56" r="4.5" fill="#B5673F"/>',
    ),
    "tarot_diferenciacion": (
        "Arcano Diferenciación Gregory",
        '<path d="M36,138 C64,72 108,128 144,62" fill="none" stroke="#EDE4D3" stroke-width="2.4"/>'
        '<line x1="48" y1="128" x2="132" y2="68" stroke="#D4A359" stroke-width="2.2"/>'
        '<circle cx="90" cy="98" r="11" fill="none" stroke="#687D5B" stroke-width="1.5"/>'
        '<circle cx="90" cy="98" r="4.5" fill="#D4A359" stroke="#121410" stroke-width="1.2"/>',
    ),
    "tarot_trapecios": (
        "Arcano Regla de Trapecios",
        '<line x1="32" y1="142" x2="148" y2="142" stroke="#2E2920" stroke-width="1.3"/>'
        '<polygon points="42,142 42,114 74,84 74,142" fill="#687D5B" opacity="0.34" stroke="#687D5B" stroke-width="1.3"/>'
        '<polygon points="74,142 74,84 106,70 106,142" fill="#D4A359" opacity="0.28" stroke="#D4A359" stroke-width="1.3"/>'
        '<polygon points="106,142 106,70 138,98 138,142" fill="#687D5B" opacity="0.34" stroke="#687D5B" stroke-width="1.3"/>'
        '<path d="M42,114 Q90,56 138,98" fill="none" stroke="#EDE4D3" stroke-width="2.2"/>',
    ),
    "tarot_simpson_13": (
        "Arcano Simpson Un Tercio",
        '<line x1="32" y1="142" x2="148" y2="142" stroke="#2E2920" stroke-width="1.3"/>'
        '<path d="M42,142 L42,112 Q90,52 138,98 L138,142 Z" fill="#D4A359" opacity="0.24" stroke="#D4A359" stroke-width="1.8"/>'
        '<line x1="90" y1="74" x2="90" y2="142" stroke="#687D5B" stroke-width="1.4" stroke-dasharray="3 2"/>'
        '<circle cx="42" cy="112" r="3.8" fill="#EDE4D3"/><circle cx="90" cy="74" r="4.5" fill="#D4A359"/><circle cx="138" cy="98" r="3.8" fill="#EDE4D3"/>',
    ),
    "tarot_simpson_38": (
        "Arcano Simpson Tres Octavos",
        '<line x1="30" y1="142" x2="150" y2="142" stroke="#2E2920" stroke-width="1.3"/>'
        '<path d="M38,142 L38,116 C64,62 114,58 142,108 L142,142 Z" fill="#687D5B" opacity="0.24" stroke="#D4A359" stroke-width="1.8"/>'
        '<line x1="72" y1="78" x2="72" y2="142" stroke="#B5673F" stroke-width="1.3" stroke-dasharray="3 2"/>'
        '<line x1="108" y1="74" x2="108" y2="142" stroke="#B5673F" stroke-width="1.3" stroke-dasharray="3 2"/>'
        '<circle cx="38" cy="116" r="3.5" fill="#EDE4D3"/><circle cx="72" cy="78" r="3.8" fill="#B5673F"/>'
        '<circle cx="108" cy="74" r="3.8" fill="#B5673F"/><circle cx="142" cy="108" r="3.5" fill="#EDE4D3"/>',
    ),
    "tarot_euler_heun": (
        "Arcano Euler y Heun",
        '<path d="M40,136 Q92,102 142,56" fill="none" stroke="#EDE4D3" stroke-width="2" stroke-dasharray="4 3"/>'
        '<polyline points="40,136 90,118 140,86" fill="none" stroke="#B5673F" stroke-width="2"/>'
        '<polyline points="40,136 90,104 140,60" fill="none" stroke="#D4A359" stroke-width="2.4"/>'
        '<circle cx="40" cy="136" r="4" fill="#EDE4D3"/><circle cx="90" cy="104" r="4" fill="#D4A359"/><circle cx="140" cy="60" r="4.5" fill="#D4A359"/>',
    ),
    "tarot_runge_kutta": (
        "Arcano Runge-Kutta RK4",
        '<g stroke="#687D5B" stroke-width="1.5" opacity="0.85">'
        '<line x1="46" y1="128" x2="64" y2="116"/><line x1="74" y1="108" x2="94" y2="92"/><line x1="104" y1="84" x2="124" y2="66"/>'
        '</g>'
        '<path d="M42,136 C74,116 106,88 142,56" fill="none" stroke="#D4A359" stroke-width="2.6"/>'
        '<circle cx="42" cy="136" r="4" fill="#EDE4D3"/><circle cx="74" cy="112" r="3.8" fill="#687D5B"/>'
        '<circle cx="108" cy="84" r="3.8" fill="#B5673F"/><circle cx="142" cy="56" r="4.8" fill="#D4A359"/>',
    ),
    "tarot_milne": (
        "Arcano Predictor-Corrector Milne",
        '<polyline points="38,136 62,120 86,102 110,84" fill="none" stroke="#687D5B" stroke-width="2.2"/>'
        '<line x1="110" y1="84" x2="142" y2="74" stroke="#B5673F" stroke-width="1.8" stroke-dasharray="3 2"/>'
        '<line x1="110" y1="84" x2="142" y2="56" stroke="#D4A359" stroke-width="2.4"/>'
        '<line x1="142" y1="74" x2="142" y2="56" stroke="#EDE4D3" stroke-width="1.4" stroke-dasharray="2 2"/>'
        '<circle cx="38" cy="136" r="3.5" fill="#687D5B"/><circle cx="62" cy="120" r="3.5" fill="#687D5B"/>'
        '<circle cx="86" cy="102" r="3.5" fill="#687D5B"/><circle cx="110" cy="84" r="4" fill="#EDE4D3"/>'
        '<circle cx="142" cy="74" r="3.5" fill="#B5673F"/><circle cx="142" cy="56" r="4.8" fill="#D4A359"/>',
    ),
}


@st.cache_data(show_spinner=False)
def obtener_svg_tarot_arandu(clave_svg: str) -> str:
    """Retorna la ilustración vertical SVG inline estilo Tarot para cada método numérico."""
    label, body = _TAROT_BODY_SVGS.get(clave_svg, _TAROT_BODY_SVGS["tarot_biseccion"])
    prefix = _TAROT_FRAME_PREFIX.format(label=html.escape(label))
    return f"{prefix}{body}</svg>"


@st.cache_data(show_spinner=False)
def construir_html_carta_tarot(
    romano: str,
    clave_svg: str,
    titulo: str,
    subtitulo_arcano: str,
    resumen: str,
    pista: str,
    activa: bool = False,
) -> str:
    """Construye y cachea el HTML de una carta coleccionable estilo Tarot Arandú."""
    clase_activa = " is-active" if activa else ""
    badge = '<span class="arandu-tarot-badge">ACTIVA</span>' if activa else ""
    svg_tarot = obtener_svg_tarot_arandu(clave_svg)
    return (
        f'<div class="arandu-tarot-card{clase_activa}">'
        '<div class="arandu-tarot-top">'
        f'<span class="arandu-tarot-num">{html.escape(romano)}{badge}</span>'
        "</div>"
        f'<div class="arandu-tarot-suit">{html.escape(subtitulo_arcano)}</div>'
        f'<div class="arandu-tarot-art">{svg_tarot}</div>'
        f'<div class="arandu-tarot-title">{html.escape(titulo)}</div>'
        f'<p class="arandu-tarot-desc">{html.escape(resumen)}</p>'
        f'<div class="arandu-tarot-foot">{html.escape(pista)}</div>'
        "</div>"
    )


def aplicar_estilo_arandu() -> None:
    """Inyecta tipografías, tarjetas Guru, tablas doradas y controles orgánicos de Arandú."""
    st.markdown(obtener_css_arandu(), unsafe_allow_html=True)

