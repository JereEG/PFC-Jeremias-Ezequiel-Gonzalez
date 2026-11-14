"""
Motor de Estilo e Identidad Visual de Arandú ("Dark Botanical & Golden Wisdom").
Centraliza la paleta cromática oficial, tipografías, CSS de Streamlit y Template Plotly.
"""

from __future__ import annotations

import base64
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

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

PLOTLY_ARANDU_TEMPLATE = go.layout.Template(
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

pio.templates["arandu"] = PLOTLY_ARANDU_TEMPLATE
pio.templates.default = "arandu"


def estilizar_figura_arandu(fig: go.Figure) -> go.Figure:
    """Aplica la plantilla y colores orgánicos de Arandú a cualquier figura Plotly."""
    fig.update_layout(
        template=PLOTLY_ARANDU_TEMPLATE,
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
def obtener_logo_arandu_src(prefer_svg: bool = False) -> str | None:
    """Retorna un data URI (PNG/JPEG o SVG transparente) del logo oficial Arandú."""
    candidatos = (LOGO_SVG_PATH, LOGO_JPG_PATH) if prefer_svg else (LOGO_JPG_PATH, LOGO_SVG_PATH)
    for ruta in candidatos:
        if ruta.is_file():
            mime = "image/svg+xml" if ruta.suffix.lower() == ".svg" else "image/jpeg"
            encoded = base64.b64encode(ruta.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return None


def aplicar_estilo_arandu() -> None:
    """Inyecta tipografías, tarjetas Guru, tablas doradas y controles orgánicos de Arandú."""
    st.markdown(
        """
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
        </style>
        """,
        unsafe_allow_html=True,
    )

