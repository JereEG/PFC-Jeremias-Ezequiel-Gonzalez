"""
Motor de Estilo e Identidad Visual de Arandú ("Dark Botanical & Golden Wisdom").
Centraliza la paleta cromática oficial, tipografías, CSS de Streamlit y Template Plotly.
"""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    import plotly.graph_objects as go

PALETA_ARANDU: dict[str, str] = {
    "bg_principal": "#0A0B09",
    "bg_secundario": "#161814",
    "bg_lienzo": "#121410",
    "oro": "#D4A359",
    "salvia": "#7A9A65",
    "terracota": "#C86D44",
    "texto_principal": "#EDE4D3",
    "texto_secundario": "#C7BFA9",
    "borde": "#736958",
    "salvia_borde": "#687D5B",
    "terracota_oscuro": "#B5673F",
    "marfil_nodo": "#FFFDF9",
    "texto_atenuado": "#9E9585",
    "grilla": "#26241E",
    "bg_elevado": "#1E201B",
    "bg_marco": "#0C0E0A",
    "bg_sombra": "#11120F",
    "bg_profundo": "#0E100C",
    "bg_carta_top": "#171A14",
    "bg_carta_mid": "#10120E",
    "bg_carta_bot": "#16130E",
    "bg_carta_activa_top": "#1E2018",
    "bg_carta_activa_bot": "#1C170F",
    "bg_carta_inset": "#11130F",
    "bg_horner": "#181A14",
    "negro_sombra": "#000000",
}


PALETA_ARANDU_CLARO: dict[str, str] = {
    "bg_principal": "#F4EFE6",
    "bg_secundario": "#EAE3D5",
    "bg_lienzo": "#F8F4EC",
    "oro": "#7A500F",
    "salvia": "#3A5730",
    "terracota": "#8E3B18",
    "texto_principal": "#1F241D",
    "texto_secundario": "#444036",
    "borde": "#8B8170",
    "salvia_borde": "#46643C",
    "terracota_oscuro": "#7A3112",
    "marfil_nodo": "#FFFDF9",
    "texto_atenuado": "#564F43",
    "grilla": "#DDD4C2",
    "bg_elevado": "#EFE8DA",
    "bg_marco": "#EBE3D4",
    "bg_sombra": "#E4DBC9",
    "bg_profundo": "#EFE9DC",
    "bg_carta_top": "#F1ECE1",
    "bg_carta_mid": "#F7F3EA",
    "bg_carta_bot": "#EAE2D2",
    "bg_carta_activa_top": "#EDE5D4",
    "bg_carta_activa_bot": "#E5DAC4",
    "bg_carta_inset": "#F6F1E7",
    "bg_horner": "#EBE3D5",
    "negro_sombra": "#4A4030",
}


def obtener_modo_tema() -> str:
    """Retorna el tema activo ('oscuro' o 'claro') desde session_state."""
    return "claro" if st.session_state.get("arandu_tema_modo") == "claro" else "oscuro"


def obtener_paleta_activa(modo: str | None = None) -> dict[str, str]:
    """Retorna la paleta cromática correspondiente al tema activo."""
    modo_eff = modo or obtener_modo_tema()
    return PALETA_ARANDU_CLARO if modo_eff == "claro" else PALETA_ARANDU


def hex_a_rgba(hex_color: str, alpha: float) -> str:
    """Convierte un color hexadecimal de PALETA_ARANDU a cadena rgba CSS/Plotly."""
    limpio = hex_color.lstrip("#")
    r, g, b = int(limpio[0:2], 16), int(limpio[2:4], 16), int(limpio[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


PALETA_ARANDU.update(
    {
        "oro_fill": hex_a_rgba(PALETA_ARANDU["oro"], 0.25),
        "salvia_fill": hex_a_rgba(PALETA_ARANDU["salvia"], 0.25),
        "terracota_fill": hex_a_rgba(PALETA_ARANDU["terracota"], 0.25),
        "primary_gold": PALETA_ARANDU["oro"],
        "background_obsidian": PALETA_ARANDU["bg_principal"],
        "surface_bark": PALETA_ARANDU["bg_secundario"],
        "plot_bg": PALETA_ARANDU["bg_lienzo"],
        "text_ivory": PALETA_ARANDU["texto_principal"],
        "node_ivory": PALETA_ARANDU["marfil_nodo"],
        "sage_green": PALETA_ARANDU["salvia"],
        "sage_fill": hex_a_rgba(PALETA_ARANDU["salvia"], 0.25),
        "sage_border": PALETA_ARANDU["salvia_borde"],
        "terracotta": PALETA_ARANDU["terracota"],
        "terracotta_dark": PALETA_ARANDU["terracota_oscuro"],
        "terracotta_fill": hex_a_rgba(PALETA_ARANDU["terracota"], 0.25),
        "gold_fill": hex_a_rgba(PALETA_ARANDU["oro"], 0.25),
        "grid_line": PALETA_ARANDU["grilla"],
        "card_border": PALETA_ARANDU["borde"],
        "muted_text": PALETA_ARANDU["texto_atenuado"],
    }
)

ARANDU_GOLD = PALETA_ARANDU["oro"]
ARANDU_OBSIDIAN = PALETA_ARANDU["bg_principal"]
ARANDU_BARK = PALETA_ARANDU["bg_secundario"]
ARANDU_PLOT_BG = PALETA_ARANDU["bg_lienzo"]
ARANDU_IVORY = PALETA_ARANDU["texto_principal"]
ARANDU_NODE_IVORY = PALETA_ARANDU["marfil_nodo"]
ARANDU_SAGE = PALETA_ARANDU["salvia"]
ARANDU_SAGE_FILL = PALETA_ARANDU["salvia_fill"]
ARANDU_SAGE_BORDER = PALETA_ARANDU["salvia_borde"]
ARANDU_TERRACOTTA = PALETA_ARANDU["terracota"]
ARANDU_TERRACOTTA_DARK = PALETA_ARANDU["terracota_oscuro"]
ARANDU_TERRACOTTA_FILL = PALETA_ARANDU["terracota_fill"]
ARANDU_GOLD_FILL = PALETA_ARANDU["oro_fill"]
ARANDU_GRID = PALETA_ARANDU["grilla"]
ARANDU_CARD_BORDER = PALETA_ARANDU["borde"]
ARANDU_MUTED = PALETA_ARANDU["texto_atenuado"]

COLORWAY_ARANDU: list[str] = [
    PALETA_ARANDU["oro"],
    PALETA_ARANDU["salvia"],
    PALETA_ARANDU["terracota"],
    PALETA_ARANDU["marfil_nodo"],
    PALETA_ARANDU["terracota_oscuro"],
    PALETA_ARANDU["salvia_borde"],
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
def obtener_plotly_arandu_template(modo: str = "oscuro"):
    """Instancia la plantilla Plotly de Arandú para el tema ('oscuro' o 'claro')."""
    import plotly.graph_objects as go
    import plotly.io as pio

    paleta = PALETA_ARANDU_CLARO if modo == "claro" else PALETA_ARANDU
    grid_suave = hex_a_rgba(paleta["grilla"], 0.42 if modo == "oscuro" else 0.52)
    colorway = [
        paleta["oro"],
        paleta["salvia"],
        paleta["terracota"],
        paleta["texto_principal"],
        paleta["terracota_oscuro"],
        paleta["salvia_borde"],
    ]
    template = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=hex_a_rgba(paleta["bg_principal"], 0),
            plot_bgcolor=paleta["bg_lienzo"],
            margin=dict(l=50, r=30, t=60, b=50),
            font=dict(
                color=paleta["texto_principal"],
                family="'Inter', sans-serif",
                size=13,
            ),
            title=dict(
                y=0.98,
                yanchor="top",
                automargin=True,
                pad=dict(b=8),
                font=dict(
                    color=paleta["texto_principal"],
                    family="'Outfit', sans-serif",
                    size=17,
                ),
            ),
            colorway=colorway,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor=paleta["bg_secundario"],
                bordercolor=paleta["borde"],
                borderwidth=1,
                font=dict(color=paleta["texto_principal"], size=13),
            ),
            hoverlabel=dict(
                bgcolor=paleta["bg_secundario"],
                bordercolor=paleta["terracota_oscuro"],
                font=dict(
                    color=paleta["texto_principal"],
                    family="'JetBrains Mono', monospace",
                    size=12,
                ),
            ),
            xaxis=dict(
                gridcolor=grid_suave,
                minor=dict(showgrid=False),
                linecolor=paleta["borde"],
                zeroline=True,
                zerolinecolor=paleta["terracota_oscuro"],
                zerolinewidth=1.6,
                tickfont=dict(
                    color=paleta["texto_principal"],
                    family="'JetBrains Mono', monospace",
                    size=13,
                ),
                title_font=dict(
                    color=paleta["texto_principal"],
                    family="'Inter', sans-serif",
                    size=14,
                ),
            ),
            yaxis=dict(
                gridcolor=grid_suave,
                minor=dict(showgrid=False),
                linecolor=paleta["borde"],
                zeroline=True,
                zerolinecolor=paleta["terracota_oscuro"],
                zerolinewidth=1.6,
                tickfont=dict(
                    color=paleta["texto_principal"],
                    family="'JetBrains Mono', monospace",
                    size=13,
                ),
                title_font=dict(
                    color=paleta["texto_principal"],
                    family="'Inter', sans-serif",
                    size=14,
                ),
            ),
        )
    )
    nombre_tpl = "arandu_claro" if modo == "claro" else "arandu"
    pio.templates[nombre_tpl] = template
    if modo == "oscuro":
        pio.templates.default = "arandu"
    return template


def __getattr__(name: str):
    if name == "PLOTLY_ARANDU_TEMPLATE":
        return obtener_plotly_arandu_template()
    raise AttributeError(f"module 'views.theme' has no attribute {name!r}")


def _mapa_colores_oscuro_a_claro() -> dict[str, str]:
    """Construye el mapeo de colores de paleta oscura hacia modo claro de alto contraste."""
    claves = (
        "oro",
        "salvia",
        "terracota",
        "terracota_oscuro",
        "salvia_borde",
        "texto_principal",
        "texto_secundario",
        "texto_atenuado",
        "borde",
        "grilla",
        "bg_lienzo",
        "bg_principal",
        "bg_secundario",
    )
    mapa = {PALETA_ARANDU[k].lower(): PALETA_ARANDU_CLARO[k] for k in claves}
    for k_base in ("oro", "salvia", "terracota"):
        for alpha in (0.22, 0.25):
            mapa[hex_a_rgba(PALETA_ARANDU[k_base], alpha).lower()] = hex_a_rgba(
                PALETA_ARANDU_CLARO[k_base], alpha
            )
    return mapa


def _remapear_color(valor, mapa: dict[str, str]):
    if isinstance(valor, str):
        return mapa.get(valor.lower(), valor)
    if isinstance(valor, (list, tuple)):
        convertidos = [mapa.get(v.lower(), v) if isinstance(v, str) else v for v in valor]
        return tuple(convertidos) if isinstance(valor, tuple) else convertidos
    return valor


def _adaptar_colores_figura_al_tema(fig: go.Figure, modo: str) -> None:
    """Reemplaza colores oscuros en trazos, formas y anotaciones cuando el modo claro está activo."""
    if modo != "claro":
        return
    mapa = _mapa_colores_oscuro_a_claro()
    mapa_linea = {**mapa, PALETA_ARANDU["marfil_nodo"].lower(): PALETA_ARANDU_CLARO["texto_principal"]}

    for traza in getattr(fig, "data", ()) or ():
        if getattr(traza, "fillcolor", None) is not None:
            traza.fillcolor = _remapear_color(traza.fillcolor, mapa)
        linea = getattr(traza, "line", None)
        if linea is not None and getattr(linea, "color", None) is not None:
            linea.color = _remapear_color(linea.color, mapa_linea)
        marcador = getattr(traza, "marker", None)
        if marcador is not None:
            m_line = getattr(marcador, "line", None)
            if m_line is not None and getattr(m_line, "color", None) is not None:
                m_line.color = _remapear_color(m_line.color, mapa_linea)
            if getattr(marcador, "color", None) is not None:
                m_color = marcador.color
                if (
                    isinstance(m_color, str)
                    and m_color.lower() == PALETA_ARANDU["marfil_nodo"].lower()
                    and not (m_line is not None and getattr(m_line, "color", None))
                ):
                    marcador.color = PALETA_ARANDU_CLARO["texto_principal"]
                else:
                    marcador.color = _remapear_color(m_color, mapa)

    for forma in getattr(fig.layout, "shapes", ()) or ():
        if getattr(forma, "fillcolor", None) is not None:
            forma.fillcolor = _remapear_color(forma.fillcolor, mapa)
        f_line = getattr(forma, "line", None)
        if f_line is not None and getattr(f_line, "color", None) is not None:
            f_line.color = _remapear_color(f_line.color, mapa_linea)

    for anotacion in getattr(fig.layout, "annotations", ()) or ():
        if getattr(anotacion, "font", None) is not None and getattr(anotacion.font, "color", None) is not None:
            anotacion.font.color = _remapear_color(anotacion.font.color, mapa_linea)


def _calcular_margen_superior_arandu(fig: go.Figure) -> int:
    """Reserva margen superior suficiente para evitar solape entre título y leyendas multilínea."""
    nombres = [
        str(getattr(tr, "name", "") or "")
        for tr in (getattr(fig, "data", ()) or ())
        if getattr(tr, "showlegend", True) is not False and getattr(tr, "name", None)
    ]
    if not nombres:
        return 60
    ancho_estimado = sum(54 + len(nom) * 8 for nom in nombres)
    filas = min(max(1, math.ceil(ancho_estimado / 440)), 4)
    return 60 + filas * 34


def estilizar_figura_arandu(fig: go.Figure) -> go.Figure:
    """Aplica la plantilla y colores orgánicos de Arandú según el tema activo."""
    modo = obtener_modo_tema()
    paleta = obtener_paleta_activa(modo)
    grid_suave = hex_a_rgba(paleta["grilla"], 0.42 if modo == "oscuro" else 0.52)
    template = obtener_plotly_arandu_template(modo)
    margen_prev = getattr(getattr(fig.layout, "margin", None), "t", None)
    t_prev = int(margen_prev) if isinstance(margen_prev, (int, float)) else 60
    margen_calc = _calcular_margen_superior_arandu(fig)
    margen_sup = max(margen_calc, t_prev)
    delta_t = max(margen_sup - t_prev, 0)
    h_actual = getattr(fig.layout, "height", None)
    layout_kwargs: dict[str, object] = {}
    if isinstance(h_actual, (int, float)) and delta_t > 0:
        layout_kwargs["height"] = int(h_actual) + delta_t

    fig.update_layout(
        template=template,
        paper_bgcolor=hex_a_rgba(paleta["bg_principal"], 0),
        plot_bgcolor=paleta["bg_lienzo"],
        margin=dict(l=50, r=30, t=margen_sup, b=50),
        font=dict(color=paleta["texto_principal"], family="'Inter', sans-serif", size=13),
        title=dict(
            y=0.98,
            yanchor="top",
            automargin=True,
            pad=dict(b=8),
            font=dict(color=paleta["texto_principal"], family="'Outfit', sans-serif", size=17),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor=paleta["bg_secundario"],
            bordercolor=paleta["borde"],
            borderwidth=1,
            font=dict(color=paleta["texto_principal"], size=13),
        ),
        hoverlabel=dict(
            bgcolor=paleta["bg_secundario"],
            bordercolor=paleta["terracota_oscuro"],
            font=dict(color=paleta["texto_principal"], family="'JetBrains Mono', monospace", size=12),
        ),
        **layout_kwargs,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_suave,
        minor=dict(showgrid=False),
        zeroline=True,
        zerolinecolor=paleta["terracota_oscuro"],
        zerolinewidth=1.6,
        tickfont=dict(color=paleta["texto_principal"], family="'JetBrains Mono', monospace", size=13),
        title_font=dict(color=paleta["texto_principal"], family="'Inter', sans-serif", size=14),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_suave,
        minor=dict(showgrid=False),
        zeroline=True,
        zerolinecolor=paleta["terracota_oscuro"],
        zerolinewidth=1.6,
        tickfont=dict(color=paleta["texto_principal"], family="'JetBrains Mono', monospace", size=13),
        title_font=dict(color=paleta["texto_principal"], family="'Inter', sans-serif", size=14),
    )
    _adaptar_colores_figura_al_tema(fig, modo)
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
    page_icon: str | bytes = (
        svg_txt
        if svg_txt is not None
        else (jpg_bytes if jpg_bytes is not None else ":material/calculate:")
    )
    return sidebar_logo, page_icon


@st.cache_data(show_spinner=False)
def obtener_css_arandu(modo: str = "oscuro") -> str:
    """Genera y cachea la hoja CSS y variables de diseño de Arandú para el tema ('oscuro' o 'claro')."""
    paleta = PALETA_ARANDU_CLARO if modo == "claro" else PALETA_ARANDU
    tokens_root = {
        "--arandu-bg": paleta["bg_principal"],
        "--arandu-surface": paleta["bg_secundario"],
        "--arandu-canvas": paleta["bg_lienzo"],
        "--arandu-gold": paleta["oro"],
        "--arandu-sage": paleta["salvia"],
        "--arandu-terracotta": paleta["terracota"],
        "--arandu-text": paleta["texto_principal"],
        "--arandu-muted": paleta["texto_secundario"],
        "--arandu-border": paleta["borde"],
        "--arandu-sage-border": paleta["salvia_borde"],
        "--arandu-terracotta-dark": paleta["terracota_oscuro"],
        "--arandu-node-ivory": paleta["marfil_nodo"],
        "--arandu-text-faint": paleta["texto_atenuado"],
        "--arandu-grid": paleta["grilla"],
        "--arandu-elevated": paleta["bg_elevado"],
        "--arandu-frame-bg": paleta["bg_marco"],
        "--arandu-shadow-bg": paleta["bg_sombra"],
        "--arandu-deep-bg": paleta["bg_profundo"],
        "--arandu-card-top": paleta["bg_carta_top"],
        "--arandu-card-mid": paleta["bg_carta_mid"],
        "--arandu-card-bot": paleta["bg_carta_bot"],
        "--arandu-card-active-top": paleta["bg_carta_activa_top"],
        "--arandu-card-active-bot": paleta["bg_carta_activa_bot"],
        "--arandu-card-inset": paleta["bg_carta_inset"],
        "--arandu-gold-09": hex_a_rgba(paleta["oro"], 0.09),
        "--arandu-gold-10": hex_a_rgba(paleta["oro"], 0.10),
        "--arandu-gold-12": hex_a_rgba(paleta["oro"], 0.12),
        "--arandu-gold-15": hex_a_rgba(paleta["oro"], 0.15),
        "--arandu-gold-16": hex_a_rgba(paleta["oro"], 0.16),
        "--arandu-gold-18": hex_a_rgba(paleta["oro"], 0.18),
        "--arandu-gold-20": hex_a_rgba(paleta["oro"], 0.20),
        "--arandu-gold-22": hex_a_rgba(paleta["oro"], 0.22),
        "--arandu-gold-24": hex_a_rgba(paleta["oro"], 0.24),
        "--arandu-gold-26": hex_a_rgba(paleta["oro"], 0.26),
        "--arandu-gold-45": hex_a_rgba(paleta["oro"], 0.45),
        "--arandu-gold-55": hex_a_rgba(paleta["oro"], 0.55),
        "--arandu-gold-56": hex_a_rgba(paleta["oro"], 0.56),
        "--arandu-gold-65": hex_a_rgba(paleta["oro"], 0.65),
        "--arandu-gold-84": hex_a_rgba(paleta["oro"], 0.84),
        "--arandu-sage-14": hex_a_rgba(paleta["salvia"], 0.14),
        "--arandu-sage-18": hex_a_rgba(paleta["salvia_borde"], 0.18),
        "--arandu-sage-22": hex_a_rgba(paleta["salvia_borde"], 0.22),
        "--arandu-sage-55": hex_a_rgba(paleta["salvia"], 0.55),
        "--arandu-terracotta-12": hex_a_rgba(paleta["terracota"], 0.12),
        "--arandu-terracotta-18": hex_a_rgba(paleta["terracota_oscuro"], 0.18),
        "--arandu-terracotta-26": hex_a_rgba(paleta["terracota_oscuro"], 0.26),
        "--arandu-terracotta-45": hex_a_rgba(paleta["terracota"], 0.45),
        "--arandu-text-10": hex_a_rgba(paleta["texto_principal"], 0.10),
        "--arandu-text-45": hex_a_rgba(paleta["texto_principal"], 0.45),
        "--arandu-surface-92": hex_a_rgba(paleta["bg_secundario"], 0.92),
        "--arandu-bg-55": hex_a_rgba(paleta["bg_principal"], 0.55),
        "--arandu-shadow-30": hex_a_rgba(paleta["negro_sombra"], 0.14 if modo == "claro" else 0.30),
        "--arandu-shadow-32": hex_a_rgba(paleta["negro_sombra"], 0.16 if modo == "claro" else 0.32),
        "--arandu-shadow-44": hex_a_rgba(paleta["negro_sombra"], 0.20 if modo == "claro" else 0.44),
        "--arandu-shadow-45": hex_a_rgba(paleta["negro_sombra"], 0.18 if modo == "claro" else 0.45),
    }
    root_lines = "\n".join(f"            {clave}: {valor};" for clave, valor in tokens_root.items())
    css_claro_streamlit = ""
    if modo == "claro":
        css_claro_streamlit = """
        :root, html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: var(--arandu-bg) !important;
            color: var(--arandu-text) !important;
            color-scheme: light;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: var(--arandu-bg) !important;
            color: var(--arandu-text) !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background-color: var(--arandu-surface) !important;
            color: var(--arandu-text) !important;
            border-right: 1px solid var(--arandu-border);
        }
        [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] span,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span, [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"], label, .katex {
            color: var(--arandu-text) !important;
        }
        [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
        input, textarea {
            background-color: var(--arandu-canvas) !important;
            color: var(--arandu-text) !important;
            border-color: var(--arandu-border) !important;
        }
        button[kind="secondary"], [data-testid="stBaseButton-secondary"],
        [data-testid="stLinkButton"] a {
            background-color: var(--arandu-canvas) !important;
            color: var(--arandu-text) !important;
            border: 1px solid var(--arandu-gold) !important;
            border-radius: 8px !important;
        }
        button[kind="primary"], [data-testid="stBaseButton-primary"] {
            background-color: var(--arandu-gold) !important;
            color: var(--arandu-canvas) !important;
            border-color: var(--arandu-gold) !important;
            border-radius: 8px !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"],
        [data-testid="stMetric"] {
            background-color: var(--arandu-surface) !important;
            border-color: var(--arandu-border) !important;
            color: var(--arandu-text) !important;
        }
        [data-testid="stAlert"] {
            color: var(--arandu-text) !important;
        }
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
            color: var(--arandu-text) !important;
        }
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpanderDetails"] {
            background-color: var(--arandu-surface) !important;
            color: var(--arandu-text) !important;
            border-color: var(--arandu-border) !important;
        }
        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpander"] summary:focus-visible,
        [data-testid="stExpander"] summary:active {
            background-color: var(--arandu-elevated) !important;
            color: var(--arandu-text) !important;
        }
        [data-testid="stExpander"] svg {
            color: var(--arandu-text) !important;
            fill: currentColor !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        ul[role="listbox"],
        li[role="option"],
        [data-baseweb="menu"] li {
            background-color: var(--arandu-canvas) !important;
            color: var(--arandu-text) !important;
            border-color: var(--arandu-border) !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        [data-baseweb="menu"] li:hover {
            background-color: var(--arandu-surface) !important;
            color: var(--arandu-text) !important;
        }
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInputStepUp"],
        [data-testid="stNumberInputStepDown"] {
            background-color: var(--arandu-surface) !important;
            color: var(--arandu-text) !important;
            border-color: var(--arandu-border) !important;
        }
        [data-testid="stToast"],
        [data-testid="stCode"],
        pre,
        [data-testid="stMarkdownContainer"] code {
            background-color: var(--arandu-surface) !important;
            color: var(--arandu-text) !important;
            border-color: var(--arandu-border) !important;
        }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] p {
            color: var(--arandu-text) !important;
        }
        """
    return f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@600;700&display=swap');

        :root {{
{root_lines}
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: var(--arandu-text);
            line-height: 1.7;
        }}
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            font-family: 'Inter', sans-serif;
            color: var(--arandu-text);
            font-size: 1.08rem;
            line-height: 1.7;
        }}
        .stMain [data-testid="stMarkdownContainer"] > p,
        .stMain [data-testid="stMarkdownContainer"] > ul,
        .stMain [data-testid="stMarkdownContainer"] > ol {{
            max-width: 74ch;
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {{
            font-size: 0.94rem;
            line-height: 1.55;
            max-width: none;
        }}
        .katex {{
            font-size: 1.15em !important;
        }}
        .katex-display,
        [data-testid="stLatex"] .katex-display {{
            overflow-x: auto;
            overflow-y: hidden;
            padding: 0.25rem 0;
        }}
        .katex-display > .katex,
        [data-testid="stLatex"] .katex {{
            font-size: 1.22em !important;
        }}
        h1, h2, h3, .arandu-brand-title {{
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--arandu-text);
        }}
        h4 {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--arandu-text);
        }}
        code, pre, .stCode, [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace !important;
        }}
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] *,
        [data-testid="stDataEditor"],
        [data-testid="stDataEditor"] *,
        [data-testid="stTable"],
        [data-testid="stTable"] * {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.98rem !important;
            --gdg-font-family: 'JetBrains Mono', monospace;
            --gdg-base-font-style: 0.98rem 'JetBrains Mono', monospace;
            --gdg-header-font-style: 600 0.98rem 'JetBrains Mono', monospace;
            --gdg-bg-cell: var(--arandu-canvas);
            --gdg-bg-cell-medium: var(--arandu-surface);
            --gdg-bg-header: var(--arandu-surface);
            --gdg-bg-header-hovered: var(--arandu-elevated);
            --gdg-bg-header-has-focus: var(--arandu-elevated);
            --gdg-text-dark: var(--arandu-text);
            --gdg-text-medium: var(--arandu-muted);
            --gdg-text-light: var(--arandu-text-faint);
            --gdg-text-header: var(--arandu-gold);
            --gdg-text-header-selected: var(--arandu-text);
            --gdg-border-color: var(--arandu-border);
            --gdg-horizontal-border-color: var(--arandu-grid);
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--arandu-border) !important;
            border-radius: 6px !important;
            padding: 0.35rem !important;
            background-color: var(--arandu-surface) !important;
        }}
        .arandu-eyebrow {{
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-gold);
            margin-bottom: 0.28rem;
            text-align: left;
        }}
        .arandu-tbl-tag-edit {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--arandu-gold);
            margin: 0.45rem 0 0.28rem 0;
            text-align: left;
        }}
        .arandu-tbl-tag-readonly {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--arandu-muted);
            margin: 0.45rem 0 0.25rem 0;
            text-align: left;
        }}
        .arandu-ctrl-hint {{
            font-family: 'Inter', sans-serif;
            font-size: 0.94rem;
            font-weight: 500;
            color: var(--arandu-muted);
            margin: 0.15rem 0 0.5rem 0;
            text-align: left;
        }}
        [data-testid="stAlert"] [data-testid="stAlertContainer"] {{
            border: none !important;
            border-top: none !important;
            border-right: none !important;
            border-bottom: none !important;
            border-left: 4px solid var(--arandu-sage) !important;
            border-radius: 0 4px 4px 0 !important;
            background-color: var(--arandu-sage-14) !important;
            box-shadow: none !important;
            text-align: left !important;
            padding: 0.9rem 1.1rem !important;
        }}
        [data-testid="stAlert"] [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
            border-left-color: var(--arandu-sage) !important;
            background-color: var(--arandu-sage-14) !important;
        }}
        [data-testid="stAlert"] [data-testid="stAlertContentInfo"]::before {{
            content: "AYUDA DE USO";
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-sage);
            margin-bottom: 0.28rem;
        }}
        [data-testid="stAlert"] [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
            border-left-color: var(--arandu-gold) !important;
            background-color: var(--arandu-gold-09) !important;
        }}
        [data-testid="stAlert"] [data-testid="stAlertContentSuccess"]::before {{
            content: "NOTA DE CÁTEDRA";
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-gold);
            margin-bottom: 0.28rem;
        }}
        [data-testid="stAlert"] [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]),
        [data-testid="stAlert"] [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
            border-left-color: var(--arandu-terracotta) !important;
            background-color: var(--arandu-terracotta-18) !important;
        }}
        [data-testid="stAlert"] [data-testid="stAlertContentWarning"]::before,
        [data-testid="stAlert"] [data-testid="stAlertContentError"]::before {{
            content: "CRITERIO TÉCNICO";
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-terracotta);
            margin-bottom: 0.28rem;
        }}

        div[data-testid="stHorizontalBlock"] {{
            align-items: stretch;
        }}

        .arandu-wasm-banner {{
            padding: 0.72rem 0.95rem;
            border-radius: 0 4px 4px 0;
            background-color: var(--arandu-sage-14);
            border: none;
            border-left: 4px solid var(--arandu-sage);
            color: var(--arandu-text);
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            line-height: 1.55;
            text-align: left;
            margin-top: 0.5rem;
        }}
        .arandu-wasm-banner::before {{
            content: "AYUDA DE USO";
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-sage);
            margin-bottom: 0.22rem;
        }}

        .arandu-sidebar-brand {{
            text-align: center;
            padding: 0.75rem 0.6rem 0.9rem 0.6rem;
            border-radius: 12px;
            background: linear-gradient(160deg, var(--arandu-surface) 0%, var(--arandu-shadow-bg) 100%);
            border: 1px solid var(--arandu-border);
            border-bottom: 2px solid var(--arandu-gold);
            margin-bottom: 0.75rem;
        }}
        .arandu-sidebar-brand img {{
            width: 78px;
            height: 78px;
            border-radius: 50%;
            border: 2px solid var(--arandu-gold);
            object-fit: cover;
            margin-bottom: 0.45rem;
            box-shadow: 0 4px 14px var(--arandu-gold-22);
        }}
        .arandu-sidebar-brand-name {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.28rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--arandu-gold);
            margin: 0;
        }}
        .arandu-sidebar-brand-sub {{
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            color: var(--arandu-text);
            opacity: 0.88;
            line-height: 1.45;
            margin: 0.2rem 0 0 0;
        }}

        .arandu-hero {{
            padding: 1.55rem 1.85rem;
            border-radius: 14px;
            border: 1px solid var(--arandu-border);
            border-left: 4px solid var(--arandu-gold);
            background: radial-gradient(circle at top right, var(--arandu-sage-14), transparent 55%),
                        linear-gradient(135deg, var(--arandu-surface) 0%, var(--arandu-deep-bg) 100%);
            box-shadow: 0 10px 28px var(--arandu-shadow-45);
            margin-bottom: 1.25rem;
        }}
        .arandu-hero-grid {{
            display: flex;
            align-items: center;
            gap: 1.6rem;
            flex-wrap: wrap;
        }}
        .arandu-hero-logo {{
            width: 132px;
            height: 132px;
            flex-shrink: 0;
            border-radius: 50%;
            background: var(--arandu-bg);
            border: 2px solid var(--arandu-gold-65);
            box-shadow: 0 6px 22px var(--arandu-gold-20);
            object-fit: cover;
        }}
        .arandu-hero h1 {{
            margin: 0.35rem 0 0.45rem 0;
            font-size: 1.95rem;
            color: var(--arandu-text);
        }}
        .arandu-pillars {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
            gap: 0.65rem;
            margin-top: 0.85rem;
        }}
        .arandu-pillar-item {{
            padding: 0.55rem 0.75rem;
            border-radius: 9px;
            background-color: var(--arandu-bg-55);
            border: 1px solid var(--arandu-border);
            border-left: 3px solid var(--arandu-sage);
            font-size: 0.86rem;
            line-height: 1.45;
        }}
        .arandu-pillar-item strong {{
            color: var(--arandu-gold);
        }}

        .arandu-abstract {{
            padding: 0.9rem 1.15rem;
            border-radius: 0 4px 4px 0;
            border: none;
            border-left: 4px solid var(--arandu-gold);
            background-color: var(--arandu-gold-09);
            text-align: left;
            margin: 0.35rem 0 0.85rem 0;
            font-size: 1.02rem;
            line-height: 1.7;
        }}
        .arandu-abstract::before {{
            content: "NOTA DE CÁTEDRA";
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-gold);
            margin-bottom: 0.25rem;
        }}
        .arandu-verdict-card {{
            padding: 0.95rem 1.15rem;
            border-radius: 12px;
            border: 1px solid var(--arandu-border);
            background-color: var(--arandu-surface);
            margin: 0.55rem 0 1rem 0;
        }}
        .arandu-verdict-pro {{
            border-left: 4px solid var(--arandu-sage) !important;
        }}
        .arandu-verdict-warn {{
            border-left: 4px solid var(--arandu-terracotta) !important;
        }}

        .arandu-card,
        .arandu-guru-card {{
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid var(--arandu-border);
            border-left: 4px solid var(--arandu-gold);
            background-color: var(--arandu-surface);
            box-shadow: 0 6px 18px var(--arandu-shadow-32);
            margin-bottom: 0.8rem;
            height: 100%;
        }}
        .arandu-card-sage {{
            border-left: 4px solid var(--arandu-sage-border) !important;
        }}
        .arandu-card h4,
        .arandu-guru-card h4 {{
            margin: 0 0 0.45rem 0;
            font-family: 'Outfit', sans-serif;
            font-size: 1.04rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--arandu-gold);
        }}
        .arandu-card-sage h4 {{
            color: var(--arandu-sage);
        }}

        .arandu-ejemplo-box {{
            padding: 1rem 1.2rem;
            border-radius: 0 4px 4px 0;
            border: none;
            border-left: 4px solid var(--arandu-sage);
            background-color: var(--arandu-sage-14);
            text-align: left;
            margin: 0.6rem 0 1rem 0;
        }}
        .arandu-ejemplo-box::before {{
            content: "NOTA DE CÁTEDRA";
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-sage);
            margin-bottom: 0.25rem;
        }}
        .arandu-edit-banner {{
            padding: 0.85rem 1.05rem;
            border-radius: 0 4px 4px 0;
            border: none;
            border-left: 4px solid var(--arandu-gold);
            background-color: var(--arandu-gold-09);
            color: var(--arandu-text);
            font-weight: 500;
            text-align: left;
            margin: 0.65rem 0 0.85rem 0;
        }}
        .arandu-edit-banner::before {{
            content: "AYUDA DE USO";
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-gold);
            margin-bottom: 0.25rem;
        }}
        div[data-testid="stDataEditor"] {{
            border: 1px dashed var(--arandu-gold) !important;
            border-radius: 6px !important;
            padding: 0.4rem !important;
            background-color: var(--arandu-gold-09) !important;
        }}

        .arandu-versus-card {{
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid var(--arandu-border);
            background-color: var(--arandu-surface);
            box-shadow: 0 6px 18px var(--arandu-shadow-30);
            margin-bottom: 0.85rem;
        }}
        .arandu-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--arandu-gold);
            background-color: var(--arandu-gold-12);
            border: 1px solid var(--arandu-gold-45);
            margin-right: 0.35rem;
        }}

        div.stButton > button,
        div.stLinkButton > a {{
            border: 1px solid var(--arandu-gold) !important;
            background-color: var(--arandu-elevated) !important;
            color: var(--arandu-text) !important;
            border-radius: 8px !important;
            min-height: 44px !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            box-shadow: 0 3px 10px var(--arandu-shadow-30) !important;
            transition: transform 0.18s ease, background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
        }}
        div.stButton > button[kind="primary"],
        div.stButton > [data-testid="stBaseButton-primary"] {{
            background-color: var(--arandu-gold) !important;
            color: var(--arandu-canvas) !important;
            border-color: var(--arandu-gold) !important;
        }}
        div.stButton > button:hover,
        div.stLinkButton > a:hover,
        div.stButton > button[kind="primary"]:hover,
        div.stButton > [data-testid="stBaseButton-primary"]:hover {{
            transform: translateY(-2px) !important;
            background-color: var(--arandu-terracotta-26) !important;
            border-color: var(--arandu-terracotta-dark) !important;
            color: var(--arandu-text) !important;
            box-shadow: 0 6px 16px var(--arandu-gold-24) !important;
        }}

        button[data-baseweb="tab"] {{
            color: var(--arandu-text) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            min-height: 44px !important;
        }}
        [data-testid="stSidebarNav"] a,
        [data-testid="stExpander"] summary {{
            min-height: 44px !important;
        }}
        [data-baseweb="input"],
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] {{
            border-color: var(--arandu-text-faint) !important;
            min-height: 44px;
        }}
        [data-testid="stMarkdownContainer"] a {{
            color: var(--arandu-gold) !important;
            text-decoration: underline !important;
            text-underline-offset: 3px;
        }}
        .js-plotly-plot .plotly .modebar {{
            background-color: var(--arandu-surface) !important;
            border: 1px solid var(--arandu-border) !important;
            border-radius: 6px !important;
        }}
        .js-plotly-plot .plotly .modebar-btn svg path {{
            fill: var(--arandu-muted) !important;
        }}
        .js-plotly-plot .plotly .modebar-btn:hover svg path,
        .js-plotly-plot .plotly .modebar-btn.active svg path {{
            fill: var(--arandu-gold) !important;
        }}
        a:focus-visible,
        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        select:focus-visible,
        summary:focus-visible,
        [tabindex]:focus-visible,
        [role="button"]:focus-visible,
        [role="tab"]:focus-visible,
        [role="slider"]:focus-visible {{
            outline: 3px solid var(--arandu-gold) !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 5px var(--arandu-gold-24) !important;
        }}
        .arandu-sr-only {{
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            clip-path: inset(50%) !important;
            border: 0 !important;
        }}
        .arandu-sidebar-heading {{
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            color: var(--arandu-text) !important;
            margin: 0 0 0.35rem 0 !important;
            line-height: 1.3 !important;
        }}
        .arandu-vs-divider {{
            text-align: center !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            color: var(--arandu-gold) !important;
            padding-top: 1.1rem !important;
            margin: 0 !important;
            line-height: 1.2 !important;
        }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }}
        }}
        button[data-baseweb="tab"]:hover {{
            color: var(--arandu-text) !important;
            border-bottom-color: var(--arandu-gold) !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--arandu-gold) !important;
            border-bottom-color: var(--arandu-gold) !important;
        }}

        .arandu-nav-card {{
            display: flex;
            flex-direction: column;
            height: 100%;
            background-color: var(--arandu-surface);
            border: 1px solid var(--arandu-border);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 0.45rem;
        }}
        .arandu-nav-card:hover {{
            transform: translateY(-4px);
            border-color: var(--arandu-gold);
            box-shadow: 0 10px 24px var(--arandu-gold-16);
        }}
        .arandu-nav-card-header {{
            height: 120px;
            width: 100%;
            flex-shrink: 0;
            background-color: var(--arandu-canvas);
            border-bottom: 1px solid var(--arandu-border);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            padding: 0.45rem 0.65rem;
            box-sizing: border-box;
        }}
        .arandu-nav-card-header svg {{
            width: 100%;
            height: 100%;
            max-height: 106px;
            display: block;
        }}
        .arandu-nav-card-body {{
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.45rem;
            padding: 0.95rem 1.05rem 1.05rem 1.05rem;
        }}
        .arandu-nav-card-top {{
            display: flex;
            flex-direction: column;
            gap: 0.28rem;
        }}
        .arandu-nav-card-tag {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            font-weight: 500;
            color: var(--arandu-sage);
            letter-spacing: 0.01em;
        }}
        .arandu-nav-card-title {{
            margin: 0;
            font-family: 'Outfit', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--arandu-text);
            line-height: 1.3;
        }}
        .arandu-nav-card:hover .arandu-nav-card-title {{
            color: var(--arandu-gold);
        }}
        .arandu-nav-card-desc {{
            margin: 0;
            font-family: 'Inter', sans-serif;
            font-size: 0.96rem;
            font-weight: 400;
            line-height: 1.58;
            color: var(--arandu-text);
        }}
        .arandu-mini-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
            gap: 0.6rem;
            margin: 0.35rem 0 0.95rem 0;
        }}
        .arandu-mini-card {{
            padding: 0.65rem 0.85rem;
            border-radius: 10px;
            background-color: var(--arandu-surface);
            border: 1px solid var(--arandu-border);
            border-left: 3px solid var(--arandu-gold);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .arandu-mini-card:hover {{
            transform: translateY(-2px);
            border-color: var(--arandu-gold);
        }}
        .arandu-mini-card-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.92rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--arandu-text);
            margin: 0 0 0.25rem 0;
        }}
        .arandu-mini-card-desc {{
            font-family: 'Inter', sans-serif;
            font-size: 0.96rem;
            font-weight: 400;
            line-height: 1.55;
            color: var(--arandu-text);
            margin: 0;
        }}
        @media (max-width: 768px) {{
            .arandu-nav-card-header {{
                height: auto;
                min-height: 108px;
            }}
            .arandu-mini-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .arandu-nav-card-header img.arandu-card-media-img,
        .arandu-ficha-art img.arandu-card-media-img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            border-radius: 6px;
        }}
        .arandu-ficha-card {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
            background: linear-gradient(175deg, var(--arandu-card-top) 0%, var(--arandu-card-mid) 58%, var(--arandu-card-bot) 100%);
            border: 1px solid var(--arandu-border);
            border-radius: 14px;
            padding: 0.75rem 0.82rem 0.85rem 0.82rem;
            box-shadow: inset 0 0 0 4px var(--arandu-card-inset), inset 0 0 0 5px var(--arandu-gold-22), 0 8px 20px var(--arandu-shadow-44);
            margin-bottom: 0.45rem;
            transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
            position: relative;
        }}
        .arandu-ficha-card:hover {{
            transform: translateY(-5px);
            border-color: var(--arandu-gold);
            box-shadow: inset 0 0 0 4px var(--arandu-card-inset), inset 0 0 0 5px var(--arandu-gold-56), 0 14px 28px var(--arandu-gold-18);
        }}
        .arandu-ficha-card.is-active {{
            background: radial-gradient(circle at 50% 34%, var(--arandu-gold-15), transparent 64%),
                        linear-gradient(175deg, var(--arandu-card-active-top) 0%, var(--arandu-canvas) 62%, var(--arandu-card-active-bot) 100%);
            border: 1.5px solid var(--arandu-gold);
            transform: translateY(-4px);
            box-shadow: inset 0 0 0 4px var(--arandu-canvas), inset 0 0 0 5px var(--arandu-gold-84), 0 14px 28px var(--arandu-gold-26);
        }}
        .arandu-ficha-main {{
            display: flex;
            flex-direction: column;
            flex: 1 1 auto;
        }}
        .arandu-ficha-top {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
            margin-bottom: 0.22rem;
        }}
        .arandu-ficha-top::before,
        .arandu-ficha-top::after {{
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--arandu-gold-45), transparent);
        }}
        .arandu-ficha-num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.55rem;
            height: 1.55rem;
            padding: 0 0.42rem;
            border-radius: 999px;
            border: 1px solid var(--arandu-gold-55);
            background-color: var(--arandu-gold-12);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--arandu-gold);
            line-height: 1;
        }}
        .arandu-ficha-card.is-active .arandu-ficha-num {{
            background-color: var(--arandu-gold);
            color: var(--arandu-canvas);
            border-color: var(--arandu-gold);
        }}
        .arandu-ficha-badge {{
            display: inline-block;
            margin-left: 0.35rem;
            padding: 0.14rem 0.5rem;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: var(--arandu-canvas);
            background-color: var(--arandu-gold);
        }}
        .arandu-ficha-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.01em;
            color: var(--arandu-sage);
            text-align: center;
            margin: 0.12rem 0 0.45rem 0;
        }}
        .arandu-ficha-card.is-active .arandu-ficha-tag {{
            color: var(--arandu-gold);
        }}
        .arandu-ficha-art {{
            width: 100%;
            height: 158px;
            background: var(--arandu-frame-bg);
            border: 1px solid var(--arandu-border);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.3rem;
            box-sizing: border-box;
            margin-bottom: 0.6rem;
            overflow: hidden;
        }}
        .arandu-ficha-card.is-active .arandu-ficha-art {{
            border-color: var(--arandu-gold-65);
            background: radial-gradient(circle at 50% 45%, var(--arandu-gold-09), var(--arandu-frame-bg) 72%);
        }}
        .arandu-ficha-art svg {{
            width: 100%;
            height: 100%;
            max-height: 148px;
            display: block;
        }}
        .arandu-ficha-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.96rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--arandu-text);
            text-align: center;
            line-height: 1.28;
            margin: 0 0 0.38rem 0;
        }}
        .arandu-ficha-card:hover .arandu-ficha-title,
        .arandu-ficha-card.is-active .arandu-ficha-title {{
            color: var(--arandu-gold);
        }}
        .arandu-ficha-desc {{
            font-family: 'Inter', sans-serif;
            font-size: 0.96rem;
            font-weight: 400;
            line-height: 1.58;
            color: var(--arandu-text);
            text-align: center;
            margin: 0 0 0.6rem 0;
        }}
        .arandu-ficha-foot {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: var(--arandu-gold);
            border-top: 1px solid var(--arandu-gold-24);
            padding-top: 0.45rem;
            text-align: center;
            line-height: 1.4;
            letter-spacing: 0.01em;
            margin-top: auto;
        }}
        .arandu-ficha-activa-banner {{
            padding: 0.85rem 1.1rem;
            border-radius: 0 4px 4px 0;
            border: none;
            border-left: 4px solid var(--arandu-gold);
            background-color: var(--arandu-gold-09);
            text-align: left;
            margin: 0.55rem 0 0.9rem 0;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 0.2rem;
        }}
        .arandu-ficha-activa-titulo {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.04rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--arandu-gold);
        }}
        .arandu-ficha-activa-pista {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: var(--arandu-sage);
        }}
        .arandu-verdict-head {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.98rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            padding: 0.35rem 0.65rem;
            border-radius: 7px;
            margin-bottom: 0.55rem;
        }}
        .arandu-verdict-head-pro {{
            color: var(--arandu-text);
            background-color: var(--arandu-sage-18);
            border-left: 3px solid var(--arandu-sage);
        }}
        .arandu-verdict-head-warn {{
            color: var(--arandu-text);
            background-color: var(--arandu-terracotta-18);
            border-left: 3px solid var(--arandu-terracotta);
        }}
        @media (max-width: 768px) {{
            .arandu-ficha-art {{
                height: 132px;
            }}
        }}

        [data-testid="stSkeleton"],
        .arandu-skeleton-box {{
            background: linear-gradient(
                90deg,
                var(--arandu-surface) 25%,
                var(--arandu-grid) 50%,
                var(--arandu-surface) 75%
            ) !important;
            background-size: 200% 100% !important;
            border: 1px solid var(--arandu-border) !important;
            border-radius: 10px !important;
            animation: arandu-skeleton-shimmer 1.6s ease-in-out infinite !important;
        }}
        @keyframes arandu-skeleton-shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        .arandu-ng-banner {{
            padding: 0.85rem 1.05rem;
            border-radius: 0 4px 4px 0;
            border: none;
            font-size: 0.96rem;
            line-height: 1.6;
            text-align: left;
            color: var(--arandu-text);
            margin: 0.45rem 0 0.85rem 0;
        }}
        .arandu-ng-banner::before {{
            content: "CRITERIO TÉCNICO";
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--arandu-gold);
            margin-bottom: 0.25rem;
        }}
        .arandu-ng-banner-asc {{
            background-color: var(--arandu-sage-14);
            border-left: 4px solid var(--arandu-sage);
        }}
        .arandu-ng-banner-asc::before {{
            color: var(--arandu-sage);
        }}
        .arandu-ng-banner-desc {{
            background-color: var(--arandu-gold-09);
            border-left: 4px solid var(--arandu-gold);
        }}
        .arandu-ng-summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem 0;
        }}
        .arandu-ng-summary-card {{
            border-radius: 12px;
            padding: 0.85rem 1rem;
            border: 1px solid var(--arandu-border);
            background-color: var(--arandu-surface);
        }}
        .arandu-ng-summary-asc {{
            background-color: var(--arandu-sage-14);
            border-color: var(--arandu-sage-55);
        }}
        .arandu-ng-summary-desc {{
            background-color: var(--arandu-gold-12);
            border-color: var(--arandu-gold-45);
        }}
        .arandu-ng-summary-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--arandu-text);
        }}
        .arandu-ng-summary-val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.32rem;
            font-weight: 700;
            color: var(--arandu-text);
            margin: 0.3rem 0 0.18rem 0;
        }}
        .arandu-ng-summary-sub {{
            font-size: 0.76rem;
            color: var(--arandu-muted);
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.45;
        }}
        .arandu-ng-pill-asc,
        .arandu-ng-pill-desc,
        .arandu-ng-pill-muted {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.12rem 0.45rem;
            border-radius: 6px;
        }}
        .arandu-ng-pill-asc {{
            background-color: var(--arandu-sage-22);
            color: var(--arandu-text);
            border: 1px solid var(--arandu-sage-55);
        }}
        .arandu-ng-pill-desc {{
            background-color: var(--arandu-gold-18);
            color: var(--arandu-text);
            border: 1px solid var(--arandu-gold-45);
        }}
        .arandu-ng-pill-muted {{
            background-color: var(--arandu-canvas);
            color: var(--arandu-muted);
            border: 1px solid var(--arandu-border);
        }}
        .arandu-ng-diff-wrap {{
            overflow-x: auto;
            border: 1px solid var(--arandu-border);
            border-radius: 12px;
            background-color: var(--arandu-canvas);
            padding: 0.4rem;
            margin-bottom: 0.75rem;
        }}
        .arandu-ng-diff-table {{
            width: 100%;
            border-collapse: collapse;
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.92rem;
        }}
        .arandu-ng-diff-th {{
            background-color: var(--arandu-surface);
            color: var(--arandu-gold);
            font-weight: 600;
            padding: 0.48rem 0.55rem;
            border: 1px solid var(--arandu-border);
        }}
        .arandu-ng-diff-idx {{
            background-color: var(--arandu-surface);
            color: var(--arandu-muted);
            padding: 0.32rem 0.5rem;
            border: 1px solid var(--arandu-border);
        }}
        .arandu-ng-diff-empty {{
            border: 1px solid var(--arandu-grid);
            background-color: var(--arandu-canvas);
        }}
        .arandu-ng-cell {{
            padding: 0.36rem 0.55rem;
            border: 1px solid var(--arandu-border);
            background-color: var(--arandu-surface);
            color: var(--arandu-text);
            vertical-align: middle;
        }}
        .arandu-ng-cell-asc {{
            background-color: var(--arandu-sage-18);
            border: 1.5px solid var(--arandu-sage);
            color: var(--arandu-text);
            font-weight: 700;
        }}
        .arandu-ng-cell-desc {{
            background-color: var(--arandu-gold-16);
            border: 1.5px solid var(--arandu-gold);
            color: var(--arandu-text);
            font-weight: 700;
        }}
        .arandu-ng-cell-both {{
            background-color: var(--arandu-terracotta-18);
            border: 1.5px solid var(--arandu-terracotta);
            color: var(--arandu-text);
            font-weight: 700;
        }}
        .arandu-ng-cell-trunc {{
            opacity: 0.72;
            border-style: dashed !important;
        }}
        .arandu-ng-badge-asc,
        .arandu-ng-badge-desc {{
            display: block;
            font-size: 0.65rem;
            font-weight: 600;
            margin-bottom: 0.12rem;
        }}
        .arandu-ng-badge-asc {{
            color: var(--arandu-sage);
        }}
        .arandu-ng-badge-desc {{
            color: var(--arandu-gold);
        }}
        .arandu-ng-formula-card {{
            border-radius: 14px;
            padding: 1rem;
            background-color: var(--arandu-surface);
            border: 1px solid var(--arandu-border);
        }}
        .arandu-ng-formula-card-asc {{
            border-color: var(--arandu-sage-55);
        }}
        .arandu-ng-formula-card-desc {{
            border-color: var(--arandu-gold-45);
        }}
        .arandu-ng-formula-muted {{
            opacity: 0.72;
            border-style: dashed !important;
        }}
        .arandu-ng-formula-head {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
            padding-bottom: 0.65rem;
            margin-bottom: 0.65rem;
            border-bottom: 1px solid var(--arandu-border);
        }}
        .arandu-ng-formula-icon-asc,
        .arandu-ng-formula-icon-desc {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--arandu-canvas);
            flex-shrink: 0;
        }}
        .arandu-ng-formula-icon-asc {{
            background-color: var(--arandu-sage);
        }}
        .arandu-ng-formula-icon-desc {{
            background-color: var(--arandu-gold);
        }}
        .arandu-ng-step-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.4rem;
            padding: 0.42rem 0.65rem;
            border-radius: 8px;
            background-color: var(--arandu-canvas);
            border: 1px solid var(--arandu-border);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            margin-bottom: 0.35rem;
        }}
        .arandu-ng-step-val-asc {{
            color: var(--arandu-sage);
            font-weight: 700;
        }}
        .arandu-ng-step-val-desc {{
            color: var(--arandu-gold);
            font-weight: 700;
        }}
        .arandu-ng-step-total-asc,
        .arandu-ng-step-total-desc {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.55rem 0.75rem;
            border-radius: 9px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            margin-top: 0.5rem;
            color: var(--arandu-canvas);
        }}
        .arandu-ng-step-total-asc {{
            background-color: var(--arandu-sage);
        }}
        .arandu-ng-step-total-desc {{
            background-color: var(--arandu-gold);
        }}
        .arandu-ng-banner-gprog {{
            background-color: var(--arandu-terracotta-12);
            border-left: 4px solid var(--arandu-terracotta);
        }}
        .arandu-ng-banner-gprog::before {{
            color: var(--arandu-terracotta);
        }}
        .arandu-ng-banner-greg {{
            background-color: var(--arandu-text-10);
            border-left: 4px solid var(--arandu-text);
        }}
        .arandu-ng-banner-greg::before {{
            color: var(--arandu-text);
        }}
        .arandu-ng-cell-gprog {{
            background-color: var(--arandu-terracotta-18);
            border: 2px solid var(--arandu-terracotta);
            color: var(--arandu-text);
            font-weight: 700;
        }}
        .arandu-ng-cell-greg {{
            background-color: var(--arandu-text-10);
            border: 2px double var(--arandu-text);
            color: var(--arandu-text);
            font-weight: 700;
        }}
        .arandu-ng-cell-anchor {{
            box-shadow: inset 0 0 0 2px var(--arandu-gold-65);
        }}
        .arandu-ng-badge-gprog,
        .arandu-ng-badge-greg {{
            display: block;
            font-size: 0.68rem;
            font-weight: 700;
            margin-bottom: 0.12rem;
        }}
        .arandu-ng-badge-gprog {{
            color: var(--arandu-terracotta);
        }}
        .arandu-ng-badge-greg {{
            color: var(--arandu-text);
        }}
        .arandu-ng-formula-card-gprog {{
            border-color: var(--arandu-terracotta-45);
        }}
        .arandu-ng-formula-card-greg {{
            border-color: var(--arandu-text-45);
        }}
        .arandu-ng-formula-icon-gprog,
        .arandu-ng-formula-icon-greg {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--arandu-canvas);
            flex-shrink: 0;
        }}
        .arandu-ng-formula-icon-gprog {{
            background-color: var(--arandu-terracotta);
        }}
        .arandu-ng-formula-icon-greg {{
            background-color: var(--arandu-text);
        }}
        .arandu-ng-step-val-gprog {{
            color: var(--arandu-terracotta);
            font-weight: 700;
        }}
        .arandu-ng-step-val-greg {{
            color: var(--arandu-text);
            font-weight: 700;
        }}
        .arandu-ng-step-total-gprog,
        .arandu-ng-step-total-greg {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.55rem 0.75rem;
            border-radius: 9px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            margin-top: 0.5rem;
            color: var(--arandu-canvas);
        }}
        .arandu-ng-step-total-gprog {{
            background-color: var(--arandu-terracotta);
        }}
        .arandu-ng-step-total-greg {{
            background-color: var(--arandu-text);
        }}
        .arandu-gauss-path-strip {{
            padding: 0.6rem 0.9rem;
            border-radius: 0 4px 4px 0;
            border: none;
            border-left: 4px solid var(--arandu-gold);
            background-color: var(--arandu-surface);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.84rem;
            line-height: 1.55;
            color: var(--arandu-text);
            margin-bottom: 0.85rem;
        }}
        .arandu-gauss-comp-wrap {{
            overflow-x: auto;
            border: 1px solid var(--arandu-border);
            border-radius: 10px;
            background-color: var(--arandu-surface);
            margin: 0.4rem 0 0.85rem 0;
        }}
        .arandu-gauss-comp-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.86rem;
            text-align: left;
        }}
        .arandu-gauss-comp-table th {{
            padding: 0.5rem 0.65rem;
            background-color: var(--arandu-canvas);
            color: var(--arandu-gold);
            border-bottom: 1px solid var(--arandu-border);
            font-weight: 600;
        }}
        .arandu-gauss-comp-table td {{
            padding: 0.45rem 0.65rem;
            border-bottom: 1px solid var(--arandu-grid);
            color: var(--arandu-text);
        }}
        .arandu-gauss-comp-row-active {{
            background-color: var(--arandu-gold-10);
            font-weight: 700;
        }}
        .arandu-gauss-comp-best {{
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            background-color: var(--arandu-sage-22);
            color: var(--arandu-text);
            border: 1px solid var(--arandu-sage-55);
            margin-left: 0.35rem;
        }}
        a.arandu-skip-link {{
            position: fixed;
            top: -120px;
            left: 1rem;
            z-index: 9999;
            padding: 0.65rem 1.1rem;
            border-radius: 8px;
            background-color: var(--arandu-surface-elevated);
            color: var(--arandu-gold) !important;
            border: 2px solid var(--arandu-gold);
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 0.96rem;
            text-decoration: none;
            box-shadow: 0 6px 18px var(--arandu-black-35);
        }}
        a.arandu-skip-link:focus,
        a.arandu-skip-link:focus-visible {{
            top: 0.75rem;
        }}
        [data-baseweb="checkbox"] span,
        [data-baseweb="radio"] div[role="radio"],
        [data-baseweb="select"] > div {{
            border-color: var(--arandu-text-faint) !important;
        }}
        .stMain,
        .arandu-hero,
        .arandu-nav-card,
        .arandu-ficha-card,
        .arandu-callout,
        .arandu-error-callout,
        .arandu-chart-desc {{
            overflow-wrap: break-word;
            word-break: normal;
        }}
        .arandu-chart-desc {{
            border-left: 3px solid var(--arandu-sage);
            background-color: var(--arandu-sage-14);
            padding: 0.5rem 0.85rem;
            margin: 0.3rem 0 0.55rem 0;
            border-radius: 0 4px 4px 0;
            font-size: 0.92rem;
            color: var(--arandu-text);
            line-height: 1.5;
            text-align: left;
        }}
        .arandu-chart-table-wrap {{
            overflow-x: auto;
            max-width: 100%;
            border: 1px solid var(--arandu-border);
            border-radius: 6px;
            padding: 0.25rem;
            background-color: var(--arandu-surface);
        }}
        table.arandu-chart-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            color: var(--arandu-text);
        }}
        table.arandu-chart-table caption {{
            text-align: left;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--arandu-gold);
            padding: 0.35rem 0.5rem;
        }}
        table.arandu-chart-table th {{
            background-color: var(--arandu-surface-elevated);
            color: var(--arandu-gold);
            font-weight: 700;
            text-align: left;
            padding: 0.4rem 0.55rem;
            border-bottom: 2px solid var(--arandu-gold);
        }}
        table.arandu-chart-table td {{
            padding: 0.35rem 0.55rem;
            border-bottom: 1px solid var(--arandu-border);
        }}
        .arandu-error-callout {{
            border: none;
            border-left: 4px solid var(--arandu-terracotta);
            border-radius: 0 4px 4px 0;
            background-color: var(--arandu-terr-18);
            padding: 0.8rem 1.05rem;
            margin: 0.5rem 0 0.9rem 0;
            color: var(--arandu-text);
            line-height: 1.58;
            text-align: left;
        }}
        .arandu-error-tag {{
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--arandu-terracotta);
            margin-bottom: 0.25rem;
        }}
        .arandu-footer {{
            margin-top: 1.1rem;
            padding-top: 0.65rem;
            border-top: 1px solid var(--arandu-border);
            font-size: 0.86rem;
            color: var(--arandu-muted);
            line-height: 1.45;
        }}
        @media (max-width: 900px) {{
            .arandu-ng-summary-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        @media (max-width: 640px) {{
            .arandu-hero {{
                padding: 1rem 1.05rem;
            }}
            .arandu-pillars {{
                grid-template-columns: 1fr;
            }}
        }}
        {css_claro_streamlit}
        </style>
        """


@st.cache_data(show_spinner=False)
def obtener_svg_tarjeta_arandu(clave_svg: str, modo: str = "oscuro") -> str:
    """Retorna una ilustración SVG minimalista e inline en paleta Arandú para tarjetas del catálogo."""
    paleta = PALETA_ARANDU_CLARO if modo == "claro" else PALETA_ARANDU
    colores_svg = {
        "borde": paleta["borde"],
        "salvia": paleta["salvia_borde"],
        "oro": paleta["oro"],
        "texto": paleta["texto_principal"],
        "terracota": paleta["terracota_oscuro"],
        "lienzo": paleta["bg_lienzo"],
        "secundario": paleta["bg_secundario"],
    }
    svgs: dict[str, str] = {
        "fundamentos": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Precisión binaria y series escalonadas">'
            '<line x1="22" y1="98" x2="298" y2="98" stroke="{borde}" stroke-width="1.5"/>'
            '<rect x="34" y="64" width="24" height="34" rx="3" fill="{salvia}" opacity="0.45" stroke="{salvia}"/>'
            '<rect x="64" y="50" width="24" height="48" rx="3" fill="{salvia}" opacity="0.65" stroke="{salvia}"/>'
            '<rect x="94" y="38" width="24" height="60" rx="3" fill="{oro}" opacity="0.45" stroke="{oro}"/>'
            '<rect x="124" y="29" width="24" height="69" rx="3" fill="{oro}" opacity="0.72" stroke="{oro}"/>'
            '<path d="M164,86 Q216,74 286,24" fill="none" stroke="{texto}" stroke-width="2" stroke-dasharray="4 3"/>'
            '<polyline points="164,86 196,86 196,70 228,70 228,51 260,51 260,31 286,31" fill="none" stroke="{oro}" stroke-width="2.4"/>'
            '<rect x="260" y="24" width="12" height="7" rx="1.5" fill="{terracota}"/>'
            '<circle cx="286" cy="24" r="4" fill="{oro}"/>'
            '</svg>'
        ),
        "raices": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Curva cruzando el eje X con tangente">'
            '<line x1="24" y1="74" x2="296" y2="74" stroke="{salvia}" stroke-width="1.5" stroke-dasharray="4 4"/>'
            '<line x1="54" y1="16" x2="54" y2="104" stroke="{borde}" stroke-width="1.4"/>'
            '<path d="M40,98 C105,94 148,54 190,42 C232,30 258,20 282,16" fill="none" stroke="{texto}" stroke-width="2.6"/>'
            '<line x1="92" y1="102" x2="262" y2="18" stroke="{oro}" stroke-width="2" stroke-dasharray="5 3"/>'
            '<line x1="224" y1="36" x2="224" y2="74" stroke="{terracota}" stroke-width="1.5" stroke-dasharray="3 3"/>'
            '<circle cx="224" cy="36" r="4.5" fill="{terracota}" stroke="{texto}" stroke-width="1.2"/>'
            '<circle cx="149" cy="74" r="5.5" fill="{oro}" stroke="{lienzo}" stroke-width="1.5"/>'
            '<circle cx="131" cy="74" r="3.5" fill="{salvia}"/>'
            '</svg>'
        ),
        "sistemas": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Matriz diagonal iluminada y autovectores">'
            '<g transform="translate(36,20)">'
            '<rect x="0" y="0" width="84" height="80" rx="6" fill="none" stroke="{borde}" stroke-width="1.5"/>'
            '<rect x="8" y="8" width="15" height="14" rx="2.5" fill="{oro}"/>'
            '<rect x="27" y="8" width="15" height="14" rx="2.5" fill="{salvia}" opacity="0.35"/>'
            '<rect x="46" y="8" width="15" height="14" rx="2.5" fill="{salvia}" opacity="0.25"/>'
            '<rect x="65" y="8" width="13" height="14" rx="2.5" fill="{salvia}" opacity="0.2"/>'
            '<rect x="8" y="25" width="15" height="14" rx="2.5" fill="{borde}" opacity="0.4"/>'
            '<rect x="27" y="25" width="15" height="14" rx="2.5" fill="{oro}"/>'
            '<rect x="46" y="25" width="15" height="14" rx="2.5" fill="{salvia}" opacity="0.35"/>'
            '<rect x="65" y="25" width="13" height="14" rx="2.5" fill="{salvia}" opacity="0.25"/>'
            '<rect x="8" y="42" width="15" height="14" rx="2.5" fill="{borde}" opacity="0.35"/>'
            '<rect x="27" y="42" width="15" height="14" rx="2.5" fill="{borde}" opacity="0.4"/>'
            '<rect x="46" y="42" width="15" height="14" rx="2.5" fill="{oro}"/>'
            '<rect x="65" y="42" width="13" height="14" rx="2.5" fill="{salvia}" opacity="0.35"/>'
            '<rect x="8" y="59" width="15" height="13" rx="2.5" fill="{borde}" opacity="0.3"/>'
            '<rect x="27" y="59" width="15" height="13" rx="2.5" fill="{borde}" opacity="0.35"/>'
            '<rect x="46" y="59" width="15" height="13" rx="2.5" fill="{borde}" opacity="0.4"/>'
            '<rect x="65" y="59" width="13" height="13" rx="2.5" fill="{oro}"/>'
            '</g>'
            '<line x1="168" y1="94" x2="286" y2="94" stroke="{borde}" stroke-width="1.5"/>'
            '<line x1="168" y1="94" x2="168" y2="20" stroke="{borde}" stroke-width="1.5"/>'
            '<line x1="168" y1="94" x2="266" y2="32" stroke="{oro}" stroke-width="2.6"/>'
            '<polygon points="272,27 260,30 266,38" fill="{oro}"/>'
            '<line x1="168" y1="94" x2="244" y2="76" stroke="{terracota}" stroke-width="2.2"/>'
            '<polygon points="251,74 240,72 243,81" fill="{terracota}"/>'
            '<circle cx="168" cy="94" r="4" fill="{texto}"/>'
            '</svg>'
        ),
        "interpolacion": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Nodos discretos unidos por curva polinomial">'
            '<line x1="22" y1="96" x2="298" y2="96" stroke="{borde}" stroke-width="1.4"/>'
            '<path d="M38,84 C68,36 102,26 136,54 C168,80 198,86 232,46 C252,24 270,22 286,36" fill="none" stroke="{oro}" stroke-width="2.6"/>'
            '<line x1="184" y1="74" x2="184" y2="96" stroke="{salvia}" stroke-width="1.5" stroke-dasharray="3 3"/>'
            '<circle cx="38" cy="84" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.6"/>'
            '<circle cx="92" cy="36" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.6"/>'
            '<circle cx="146" cy="61" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.6"/>'
            '<circle cx="222" cy="56" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.6"/>'
            '<circle cx="276" cy="28" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.6"/>'
            '<circle cx="184" cy="74" r="5" fill="{salvia}" stroke="{texto}" stroke-width="1.5"/>'
            '</svg>'
        ),
        "integracion": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Área bajo la curva con trapecios y arcos parabólicos">'
            '<line x1="24" y1="96" x2="296" y2="96" stroke="{borde}" stroke-width="1.5"/>'
            '<polygon points="46,96 46,72 98,54 98,96" fill="{salvia}" opacity="0.32" stroke="{salvia}" stroke-width="1.3"/>'
            '<polygon points="98,96 98,54 150,36 150,96" fill="{salvia}" opacity="0.24" stroke="{salvia}" stroke-width="1.3"/>'
            '<polygon points="150,96 150,36 202,44 202,96" fill="{oro}" opacity="0.24" stroke="{oro}" stroke-width="1.3"/>'
            '<polygon points="202,96 202,44 254,66 254,96" fill="{oro}" opacity="0.18" stroke="{oro}" stroke-width="1.3"/>'
            '<path d="M46,72 Q98,42 150,36 Q202,32 254,66" fill="none" stroke="{texto}" stroke-width="2.5"/>'
            '<path d="M150,36 Q202,28 254,66" fill="none" stroke="{oro}" stroke-width="2" stroke-dasharray="4 3"/>'
            '<circle cx="46" cy="72" r="3.5" fill="{texto}"/>'
            '<circle cx="98" cy="54" r="3.5" fill="{texto}"/>'
            '<circle cx="150" cy="36" r="4" fill="{oro}"/>'
            '<circle cx="202" cy="44" r="3.5" fill="{terracota}"/>'
            '<circle cx="254" cy="66" r="3.5" fill="{texto}"/>'
            '</svg>'
        ),
        "edo": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Campo de pendientes y trayectoria paso a paso">'
            '<g stroke="{salvia}" stroke-width="1.4" opacity="0.72">'
            '<line x1="36" y1="30" x2="52" y2="24"/><line x1="92" y1="28" x2="108" y2="23"/><line x1="148" y1="32" x2="164" y2="28"/><line x1="204" y1="40" x2="220" y2="34"/><line x1="260" y1="30" x2="276" y2="22"/>'
            '<line x1="36" y1="60" x2="52" y2="52"/><line x1="92" y1="56" x2="108" y2="47"/><line x1="148" y1="58" x2="164" y2="50"/><line x1="204" y1="64" x2="220" y2="54"/><line x1="260" y1="56" x2="276" y2="44"/>'
            '<line x1="36" y1="90" x2="52" y2="80"/><line x1="92" y1="86" x2="108" y2="75"/><line x1="148" y1="88" x2="164" y2="78"/><line x1="204" y1="92" x2="220" y2="82"/><line x1="260" y1="84" x2="276" y2="70"/>'
            '</g>'
            '<polyline points="42,92 98,74 156,56 214,42 272,24" fill="none" stroke="{oro}" stroke-width="2.5"/>'
            '<circle cx="42" cy="92" r="4" fill="{texto}" stroke="{terracota}" stroke-width="1.4"/>'
            '<circle cx="98" cy="74" r="3.8" fill="{oro}"/>'
            '<circle cx="156" cy="56" r="3.8" fill="{oro}"/>'
            '<circle cx="214" cy="42" r="3.8" fill="{oro}"/>'
            '<circle cx="272" cy="24" r="4.5" fill="{terracota}" stroke="{texto}" stroke-width="1.3"/>'
            '</svg>'
        ),
        "arena_versus": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dos curvas de convergencia compitiendo hacia un punto">'
            '<line x1="24" y1="94" x2="296" y2="94" stroke="{borde}" stroke-width="1.5"/>'
            '<line x1="24" y1="58" x2="296" y2="58" stroke="{salvia}" stroke-width="1.4" stroke-dasharray="4 4"/>'
            '<path d="M36,24 C86,28 112,56 172,58 L268,58" fill="none" stroke="{oro}" stroke-width="2.7"/>'
            '<path d="M36,88 C108,86 164,74 228,62 L268,58" fill="none" stroke="{terracota}" stroke-width="2.5"/>'
            '<circle cx="172" cy="58" r="4" fill="{oro}"/>'
            '<circle cx="268" cy="58" r="6" fill="{texto}" stroke="{oro}" stroke-width="2.2"/>'
            '<circle cx="36" cy="24" r="3.5" fill="{oro}"/>'
            '<circle cx="36" cy="88" r="3.5" fill="{terracota}"/>'
            '</svg>'
        ),
        "laboratorios": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Progresión de Fibonacci hacia la razón áurea">'
            '<line x1="24" y1="98" x2="296" y2="98" stroke="{borde}" stroke-width="1.5"/>'
            '<line x1="24" y1="38" x2="296" y2="38" stroke="{oro}" stroke-width="1.4" stroke-dasharray="5 4"/>'
            '<rect x="40" y="82" width="24" height="16" rx="2.5" fill="{salvia}" opacity="0.6"/>'
            '<rect x="76" y="72" width="24" height="26" rx="2.5" fill="{salvia}" opacity="0.75"/>'
            '<rect x="112" y="60" width="24" height="38" rx="2.5" fill="{oro}" opacity="0.6"/>'
            '<rect x="148" y="48" width="24" height="50" rx="2.5" fill="{oro}" opacity="0.8"/>'
            '<rect x="184" y="38" width="24" height="60" rx="2.5" fill="{terracota}" opacity="0.8"/>'
            '<polyline points="52,74 88,52 124,42 160,39 196,38 268,38" fill="none" stroke="{texto}" stroke-width="2.2"/>'
            '<circle cx="268" cy="38" r="4.5" fill="{oro}"/>'
            '</svg>'
        ),
        "feedback": (
            '<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Aportes y evaluación docente y estudiantil">'
            '<rect x="78" y="24" width="164" height="64" rx="10" fill="{secundario}" stroke="{oro}" stroke-width="1.8"/>'
            '<line x1="98" y1="44" x2="202" y2="44" stroke="{texto}" stroke-width="2.4" stroke-linecap="round"/>'
            '<line x1="98" y1="58" x2="220" y2="58" stroke="{salvia}" stroke-width="2.2" stroke-linecap="round"/>'
            '<line x1="98" y1="72" x2="174" y2="72" stroke="{terracota}" stroke-width="2.2" stroke-linecap="round"/>'
            '<circle cx="242" cy="24" r="9" fill="{oro}"/>'
            '<polygon points="104,88 122,88 104,102" fill="{secundario}" stroke="{oro}" stroke-width="1.6"/>'
            '</svg>'
        ),
    }
    return svgs.get(clave_svg, svgs["arena_versus"]).format(**colores_svg)


_EXTENSIONES_IMAGEN_METODO = (".png", ".jpg", ".jpeg", ".svg")


def _buscar_asset_metodo(clave_metodo: str) -> tuple[str, int, str] | None:
    """Busca si existe un archivo personalizado en assets/metodos/<clave>.(png|jpg|jpeg|svg)."""
    limpia = clave_metodo.strip().replace("\\", "/").split("/")[-1]
    if not limpia:
        return None
    for ext in _EXTENSIONES_IMAGEN_METODO:
        ruta = _resolver_ruta_asset(f"metodos/{limpia}{ext}")
        if ruta is not None:
            mime = _MIME_TYPES.get(ext, "application/octet-stream")
            return str(ruta), ruta.stat().st_mtime_ns, mime
    return None


@st.cache_data(show_spinner=False)
def _leer_asset_metodo_html(ruta_str: str, mtime_ns: int, mime: str, alt: str) -> str:
    """Codifica y cachea una imagen personalizada local para tarjetas sin dependencias externas."""
    _ = mtime_ns
    raw = Path(ruta_str).read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    alt_esc = html.escape(alt)
    return f'<img src="data:{mime};base64,{encoded}" class="arandu-card-media-img" alt="{alt_esc}" />'


@st.cache_data(show_spinner=False)
def _construir_html_tarjeta_catalogo_cached(
    clave_svg: str,
    titulo: str,
    descripcion: str,
    subtitulo_tag: str,
    firma_asset: tuple[str, int, str] | None,
    modo: str,
) -> str:
    if firma_asset is not None:
        media_html = _leer_asset_metodo_html(*firma_asset, alt=titulo)
    else:
        media_html = obtener_svg_tarjeta_arandu(clave_svg, modo=modo)
    tag_html = (
        f'<span class="arandu-nav-card-tag">{html.escape(subtitulo_tag)}</span>'
        if subtitulo_tag
        else ""
    )
    return (
        f'<article class="arandu-nav-card" aria-label="{html.escape(titulo)}">'
        f'<div class="arandu-nav-card-header">{media_html}</div>'
        '<div class="arandu-nav-card-body">'
        '<div class="arandu-nav-card-top">'
        f"{tag_html}"
        f'<h3 class="arandu-nav-card-title">{html.escape(titulo)}</h3>'
        "</div>"
        f'<p class="arandu-nav-card-desc">{html.escape(descripcion)}</p>'
        "</div>"
        "</article>"
    )


def construir_html_tarjeta_catalogo(
    clave_svg: str,
    titulo: str,
    descripcion: str,
    subtitulo_tag: str = "",
) -> str:
    """Construye el HTML de una tarjeta visual del catálogo principal con soporte de imagen en assets/metodos/."""
    firma_asset = _buscar_asset_metodo(clave_svg)
    return _construir_html_tarjeta_catalogo_cached(
        clave_svg=clave_svg,
        titulo=titulo,
        descripcion=descripcion,
        subtitulo_tag=subtitulo_tag,
        firma_asset=firma_asset,
        modo=obtener_modo_tema(),
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


_FICHA_SVG_FRAME_PREFIX = (
    '<svg viewBox="0 0 180 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}">'
    '<rect x="5" y="5" width="170" height="180" rx="8" fill="{marco}" stroke="{borde}" stroke-width="1.2"/>'
    '<rect x="10" y="10" width="160" height="170" rx="5" fill="none" stroke="{oro}" stroke-opacity="0.28" stroke-width="0.9"/>'
    '<rect x="20" y="34" width="140" height="132" rx="6" fill="{lienzo}" stroke="{borde}" stroke-width="1.1"/>'
    '<line x1="18" y1="22" x2="26" y2="22" stroke="{oro}" stroke-opacity="0.55" stroke-width="1.2"/>'
    '<line x1="154" y1="22" x2="162" y2="22" stroke="{oro}" stroke-opacity="0.55" stroke-width="1.2"/>'
    '<line x1="64" y1="22" x2="116" y2="22" stroke="{oro}" stroke-opacity="0.35" stroke-width="1"/>'
    '<circle cx="90" cy="22" r="2.2" fill="{oro}"/>'
)

_BORRADOR_METODO_SVGS: dict[str, tuple[str, str]] = {
    "taylor": (
        "Esquema Series de Taylor",
        '<line x1="30" y1="146" x2="150" y2="146" stroke="{borde}" stroke-width="1.2"/>'
        '<path d="M36,142 Q96,128 144,48" fill="none" stroke="{texto}" stroke-width="2" stroke-dasharray="4 3"/>'
        '<polyline points="36,142 64,142 64,124 92,124 92,98 120,98 120,66 144,66" fill="none" stroke="{oro}" stroke-width="2.4"/>'
        '<rect x="120" y="54" width="24" height="12" rx="2" fill="{salvia}" opacity="0.55" stroke="{salvia}"/>'
        '<circle cx="144" cy="48" r="4.5" fill="{oro}" stroke="{lienzo}" stroke-width="1.2"/>',
    ),
    "aritmetica_maquina": (
        "Esquema Aritmética de Máquina",
        '<line x1="90" y1="44" x2="90" y2="148" stroke="{oro}" stroke-width="1.8"/>'
        '<polygon points="72,148 108,148 96,138 84,138" fill="{borde}" stroke="{oro}" stroke-width="1"/>'
        '<line x1="40" y1="74" x2="140" y2="56" stroke="{texto}" stroke-width="2.2"/>'
        '<line x1="44" y1="74" x2="36" y2="108" stroke="{salvia}" stroke-width="1.3"/><line x1="44" y1="74" x2="54" y2="108" stroke="{salvia}" stroke-width="1.3"/>'
        '<path d="M32,108 Q45,118 58,108 Z" fill="{salvia}" opacity="0.7" stroke="{salvia}"/>'
        '<line x1="136" y1="56" x2="126" y2="92" stroke="{terracota}" stroke-width="1.3"/><line x1="136" y1="56" x2="146" y2="92" stroke="{terracota}" stroke-width="1.3"/>'
        '<path d="M122,92 Q136,102 150,92 Z" fill="{terracota}" opacity="0.7" stroke="{terracota}"/>'
        '<circle cx="90" cy="65" r="4" fill="{oro}"/>',
    ),
    "cancelacion": (
        "Esquema Cancelación Cuadrática",
        '<line x1="30" y1="134" x2="150" y2="134" stroke="{borde}" stroke-width="1.2"/>'
        '<line x1="36" y1="68" x2="138" y2="94" stroke="{texto}" stroke-width="2.2"/>'
        '<line x1="36" y1="74" x2="138" y2="97" stroke="{terracota}" stroke-width="2" stroke-dasharray="3 2"/>'
        '<circle cx="138" cy="95" r="14" fill="none" stroke="{oro}" stroke-width="1.8"/>'
        '<line x1="126" y1="95" x2="150" y2="95" stroke="{oro}" stroke-width="1.4"/>'
        '<circle cx="138" cy="95" r="3.5" fill="{oro}"/>'
        '<path d="M46,120 Q90,104 134,120" fill="none" stroke="{salvia}" stroke-width="1.6"/>',
    ),
    "horner": (
        "Esquema Forma Anidada de Horner",
        '<rect x="34" y="52" width="112" height="96" rx="12" fill="none" stroke="{salvia}" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<rect x="48" y="66" width="84" height="68" rx="9" fill="none" stroke="{oro}" stroke-opacity="0.7" stroke-width="1.6"/>'
        '<rect x="64" y="82" width="52" height="36" rx="6" fill="{horner}" stroke="{oro}" stroke-width="2"/>'
        '<polygon points="90,89 99,100 90,111 81,100" fill="{oro}"/>'
        '<circle cx="34" cy="100" r="3.5" fill="{texto}"/><circle cx="146" cy="100" r="3.5" fill="{terracota}"/>',
    ),
    "biseccion": (
        "Esquema Bisección y Bolzano",
        '<line x1="28" y1="100" x2="152" y2="100" stroke="{salvia}" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<line x1="46" y1="54" x2="46" y2="148" stroke="{salvia}" stroke-width="1.6"/>'
        '<line x1="134" y1="54" x2="134" y2="148" stroke="{terracota}" stroke-width="1.6"/>'
        '<line x1="90" y1="46" x2="90" y2="152" stroke="{oro}" stroke-width="2" stroke-dasharray="3 2"/>'
        '<path d="M36,138 C64,128 112,72 144,58" fill="none" stroke="{texto}" stroke-width="2.4"/>'
        '<circle cx="46" cy="134" r="3.8" fill="{salvia}"/><circle cx="134" cy="63" r="3.8" fill="{terracota}"/>'
        '<polygon points="90,91 96,100 90,109 84,100" fill="{oro}" stroke="{lienzo}" stroke-width="1"/>',
    ),
    "regula_falsi": (
        "Esquema Regula Falsi Secante",
        '<line x1="28" y1="100" x2="152" y2="100" stroke="{salvia}" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<path d="M38,142 Q96,126 142,54" fill="none" stroke="{texto}" stroke-width="2.3"/>'
        '<line x1="42" y1="140" x2="138" y2="60" stroke="{oro}" stroke-width="2"/>'
        '<line x1="138" y1="60" x2="138" y2="100" stroke="{terracota}" stroke-width="1.2" stroke-dasharray="2 2"/>'
        '<circle cx="42" cy="140" r="4" fill="{salvia}"/><circle cx="138" cy="60" r="4" fill="{terracota}"/>'
        '<circle cx="90" cy="100" r="5" fill="{oro}" stroke="{lienzo}" stroke-width="1.3"/>'
        '<circle cx="107" cy="100" r="3.2" fill="{texto}"/>',
    ),
    "newton_raphson": (
        "Esquema Newton-Raphson Tangente",
        '<line x1="28" y1="126" x2="152" y2="126" stroke="{salvia}" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<path d="M40,148 Q106,124 142,46" fill="none" stroke="{texto}" stroke-width="2.4"/>'
        '<line x1="74" y1="144" x2="148" y2="48" stroke="{oro}" stroke-width="2.2"/>'
        '<line x1="130" y1="71" x2="130" y2="126" stroke="{terracota}" stroke-width="1.4" stroke-dasharray="3 2"/>'
        '<circle cx="130" cy="71" r="4.5" fill="{terracota}" stroke="{texto}" stroke-width="1.1"/>'
        '<circle cx="88" cy="126" r="5.5" fill="{oro}" stroke="{lienzo}" stroke-width="1.4"/>'
        '<circle cx="103" cy="126" r="3.2" fill="{salvia}"/>',
    ),
    "punto_fijo": (
        "Esquema Punto Fijo y Aitken",
        '<line x1="36" y1="146" x2="144" y2="48" stroke="{salvia}" stroke-width="1.6" stroke-dasharray="4 3"/>'
        '<path d="M34,116 Q88,108 144,64" fill="none" stroke="{texto}" stroke-width="2.3"/>'
        '<polyline points="48,146 48,113 84,113 84,99 100,99 100,91" fill="none" stroke="{oro}" stroke-width="1.9"/>'
        '<path d="M48,146 Q92,138 106,86" fill="none" stroke="{terracota}" stroke-width="1.6" stroke-dasharray="3 2"/>'
        '<circle cx="106" cy="83" r="5" fill="{oro}" stroke="{lienzo}" stroke-width="1.3"/>',
    ),
    "gauss": (
        "Esquema Eliminación de Gauss",
        '<g transform="translate(46,56)">'
        '<rect x="0" y="0" width="24" height="24" rx="3" fill="{oro}"/><rect x="30" y="0" width="24" height="24" rx="3" fill="{salvia}" opacity="0.65"/><rect x="60" y="0" width="24" height="24" rx="3" fill="{salvia}" opacity="0.45"/>'
        '<rect x="0" y="30" width="24" height="24" rx="3" fill="{borde}" opacity="0.35"/><rect x="30" y="30" width="24" height="24" rx="3" fill="{oro}"/><rect x="60" y="30" width="24" height="24" rx="3" fill="{salvia}" opacity="0.55"/>'
        '<rect x="0" y="60" width="24" height="24" rx="3" fill="{borde}" opacity="0.25"/><rect x="30" y="60" width="24" height="24" rx="3" fill="{borde}" opacity="0.35"/><rect x="60" y="60" width="24" height="24" rx="3" fill="{oro}"/>'
        '</g>',
    ),
    "gauss_jordan": (
        "Esquema Gauss-Jordan Diagonal",
        '<g transform="translate(46,56)">'
        '<rect x="0" y="0" width="24" height="24" rx="3" fill="{oro}"/><rect x="30" y="0" width="24" height="24" rx="3" fill="{borde}" opacity="0.3"/><rect x="60" y="0" width="24" height="24" rx="3" fill="{borde}" opacity="0.3"/>'
        '<rect x="0" y="30" width="24" height="24" rx="3" fill="{borde}" opacity="0.3"/><rect x="30" y="30" width="24" height="24" rx="3" fill="{oro}"/><rect x="60" y="30" width="24" height="24" rx="3" fill="{borde}" opacity="0.3"/>'
        '<rect x="0" y="60" width="24" height="24" rx="3" fill="{borde}" opacity="0.3"/><rect x="30" y="60" width="24" height="24" rx="3" fill="{borde}" opacity="0.3"/><rect x="60" y="60" width="24" height="24" rx="3" fill="{oro}"/>'
        '<line x1="-4" y1="-4" x2="88" y2="88" stroke="{texto}" stroke-width="1.6" stroke-dasharray="3 3"/>'
        '</g>',
    ),
    "lu": (
        "Esquema Factorización LU",
        '<polygon points="44,62 44,142 124,142" fill="{salvia}" opacity="0.55" stroke="{salvia}" stroke-width="2"/>'
        '<polygon points="56,56 136,56 136,136" fill="{oro}" opacity="0.52" stroke="{oro}" stroke-width="2"/>'
        '<line x1="40" y1="54" x2="140" y2="144" stroke="{texto}" stroke-width="1.6" stroke-dasharray="4 3"/>',
    ),
    "gauss_seidel": (
        "Esquema Iteración Gauss-Seidel",
        '<circle cx="90" cy="100" r="38" fill="none" stroke="{borde}" stroke-width="1.5"/>'
        '<path d="M90,62 A38,38 0 0,1 128,100" fill="none" stroke="{oro}" stroke-width="2.4"/>'
        '<path d="M128,100 A38,38 0 0,1 90,138" fill="none" stroke="{salvia}" stroke-width="2.4"/>'
        '<path d="M90,138 A38,38 0 0,1 52,100" fill="none" stroke="{terracota}" stroke-width="2.4"/>'
        '<path d="M52,100 A38,38 0 0,1 90,62" fill="none" stroke="{texto}" stroke-width="2"/>'
        '<circle cx="90" cy="62" r="4.5" fill="{oro}"/><circle cx="128" cy="100" r="4.5" fill="{salvia}"/>'
        '<circle cx="90" cy="138" r="4.5" fill="{terracota}"/><circle cx="52" cy="100" r="4.5" fill="{texto}"/>'
        '<circle cx="90" cy="100" r="6" fill="{oro}"/>',
    ),
    "potencias": (
        "Esquema Método de Potencias",
        '<line x1="42" y1="142" x2="144" y2="142" stroke="{borde}" stroke-width="1.3"/>'
        '<line x1="42" y1="142" x2="42" y2="54" stroke="{borde}" stroke-width="1.3"/>'
        '<line x1="42" y1="142" x2="128" y2="122" stroke="{salvia}" stroke-width="1.8" stroke-dasharray="3 2"/>'
        '<line x1="42" y1="142" x2="136" y2="96" stroke="{terracota}" stroke-width="2"/>'
        '<line x1="42" y1="142" x2="138" y2="62" stroke="{oro}" stroke-width="2.8"/>'
        '<polygon points="144,57 132,61 139,69" fill="{oro}"/>'
        '<circle cx="42" cy="142" r="4" fill="{texto}"/>',
    ),
    "faddeev": (
        "Esquema Faddeev-Leverrier",
        '<line x1="30" y1="100" x2="150" y2="100" stroke="{borde}" stroke-width="1.3"/>'
        '<path d="M36,64 C56,144 74,144 90,100 C106,56 124,56 144,136" fill="none" stroke="{oro}" stroke-width="2.4"/>'
        '<circle cx="51" cy="100" r="4.5" fill="{salvia}" stroke="{texto}" stroke-width="1"/>'
        '<circle cx="90" cy="100" r="5" fill="{oro}" stroke="{lienzo}" stroke-width="1.3"/>'
        '<circle cx="129" cy="100" r="4.5" fill="{terracota}" stroke="{texto}" stroke-width="1"/>',
    ),
    "interp_lineal": (
        "Esquema Interpolación Lineal",
        '<line x1="30" y1="142" x2="150" y2="142" stroke="{borde}" stroke-width="1.2"/>'
        '<line x1="46" y1="126" x2="134" y2="66" stroke="{oro}" stroke-width="2.6"/>'
        '<line x1="90" y1="96" x2="90" y2="142" stroke="{salvia}" stroke-width="1.5" stroke-dasharray="3 2"/>'
        '<circle cx="46" cy="126" r="5" fill="{texto}" stroke="{terracota}" stroke-width="1.5"/>'
        '<circle cx="134" cy="66" r="5" fill="{texto}" stroke="{terracota}" stroke-width="1.5"/>'
        '<circle cx="90" cy="96" r="4.5" fill="{salvia}" stroke="{texto}" stroke-width="1.3"/>',
    ),
    "cuadratica_inversa": (
        "Esquema Cuadrática Inversa",
        '<line x1="34" y1="98" x2="146" y2="98" stroke="{salvia}" stroke-width="1.4" stroke-dasharray="4 3"/>'
        '<path d="M40,66 Q90,158 140,66" fill="none" stroke="{texto}" stroke-width="2.3"/>'
        '<circle cx="46" cy="76" r="4.2" fill="{terracota}"/><circle cx="90" cy="112" r="4.2" fill="{terracota}"/><circle cx="134" cy="76" r="4.2" fill="{terracota}"/>'
        '<circle cx="114" cy="98" r="5" fill="{oro}" stroke="{lienzo}" stroke-width="1.3"/>',
    ),
    "lagrange": (
        "Esquema Polinomio de Lagrange",
        '<path d="M36,136 C64,56 106,142 144,62" fill="none" stroke="{oro}" stroke-width="2.5"/>'
        '<path d="M36,78 Q90,136 144,122" fill="none" stroke="{salvia}" stroke-width="1.4" stroke-dasharray="3 3"/>'
        '<circle cx="44" cy="114" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.4"/>'
        '<circle cx="72" cy="82" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.4"/>'
        '<circle cx="112" cy="110" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.4"/>'
        '<circle cx="138" cy="70" r="4.5" fill="{texto}" stroke="{terracota}" stroke-width="1.4"/>',
    ),
    "newton_gregory": (
        "Esquema Newton-Gregory Diferencias",
        '<line x1="34" y1="142" x2="146" y2="142" stroke="{borde}" stroke-width="1.2"/>'
        '<path d="M44,128 Q92,102 138,56" fill="none" stroke="{oro}" stroke-width="2.5"/>'
        '<polyline points="44,128 74,128 74,110 106,110 106,86 138,86 138,56" fill="none" stroke="{salvia}" stroke-width="1.6" stroke-dasharray="3 2"/>'
        '<circle cx="44" cy="128" r="4" fill="{texto}"/><circle cx="74" cy="110" r="4" fill="{texto}"/>'
        '<circle cx="106" cy="86" r="4" fill="{texto}"/><circle cx="138" cy="56" r="4.5" fill="{terracota}"/>',
    ),
    "formula_gauss": (
        "Esquema Fórmula de Gauss Central en Zig-Zag",
        '<line x1="34" y1="142" x2="146" y2="142" stroke="{borde}" stroke-width="1.2"/>'
        '<line x1="90" y1="54" x2="90" y2="142" stroke="{oro}" stroke-width="1.3" stroke-dasharray="3 2"/>'
        '<path d="M42,126 C64,112 76,102 90,94 C106,84 120,76 138,60" fill="none" stroke="{oro}" stroke-width="2.4"/>'
        '<polyline points="90,94 106,104 122,86 138,96" fill="none" stroke="{terracota}" stroke-width="1.8"/>'
        '<polyline points="90,94 74,82 58,100 42,88" fill="none" stroke="{salvia}" stroke-width="1.5" stroke-dasharray="3 2"/>'
        '<circle cx="42" cy="126" r="3.4" fill="{texto}"/>'
        '<circle cx="66" cy="110" r="3.6" fill="{salvia}"/>'
        '<circle cx="90" cy="94" r="5.2" fill="{oro}" stroke="{lienzo}" stroke-width="1.5"/>'
        '<circle cx="114" cy="78" r="3.6" fill="{terracota}"/>'
        '<circle cx="138" cy="60" r="3.4" fill="{texto}"/>',
    ),
    "diferenciacion": (
        "Esquema Diferenciación Gregory",
        '<path d="M36,138 C64,72 108,128 144,62" fill="none" stroke="{texto}" stroke-width="2.4"/>'
        '<line x1="48" y1="128" x2="132" y2="68" stroke="{oro}" stroke-width="2.2"/>'
        '<circle cx="90" cy="98" r="11" fill="none" stroke="{salvia}" stroke-width="1.5"/>'
        '<circle cx="90" cy="98" r="4.5" fill="{oro}" stroke="{lienzo}" stroke-width="1.2"/>',
    ),
    "trapecios": (
        "Esquema Regla de Trapecios",
        '<line x1="32" y1="142" x2="148" y2="142" stroke="{borde}" stroke-width="1.3"/>'
        '<polygon points="42,142 42,114 74,84 74,142" fill="{salvia}" opacity="0.34" stroke="{salvia}" stroke-width="1.3"/>'
        '<polygon points="74,142 74,84 106,70 106,142" fill="{oro}" opacity="0.28" stroke="{oro}" stroke-width="1.3"/>'
        '<polygon points="106,142 106,70 138,98 138,142" fill="{salvia}" opacity="0.34" stroke="{salvia}" stroke-width="1.3"/>'
        '<path d="M42,114 Q90,56 138,98" fill="none" stroke="{texto}" stroke-width="2.2"/>',
    ),
    "simpson_13": (
        "Esquema Simpson Un Tercio",
        '<line x1="32" y1="142" x2="148" y2="142" stroke="{borde}" stroke-width="1.3"/>'
        '<path d="M42,142 L42,112 Q90,52 138,98 L138,142 Z" fill="{oro}" opacity="0.24" stroke="{oro}" stroke-width="1.8"/>'
        '<line x1="90" y1="74" x2="90" y2="142" stroke="{salvia}" stroke-width="1.4" stroke-dasharray="3 2"/>'
        '<circle cx="42" cy="112" r="3.8" fill="{texto}"/><circle cx="90" cy="74" r="4.5" fill="{oro}"/><circle cx="138" cy="98" r="3.8" fill="{texto}"/>',
    ),
    "simpson_38": (
        "Esquema Simpson Tres Octavos",
        '<line x1="30" y1="142" x2="150" y2="142" stroke="{borde}" stroke-width="1.3"/>'
        '<path d="M38,142 L38,116 C64,62 114,58 142,108 L142,142 Z" fill="{salvia}" opacity="0.24" stroke="{oro}" stroke-width="1.8"/>'
        '<line x1="72" y1="78" x2="72" y2="142" stroke="{terracota}" stroke-width="1.3" stroke-dasharray="3 2"/>'
        '<line x1="108" y1="74" x2="108" y2="142" stroke="{terracota}" stroke-width="1.3" stroke-dasharray="3 2"/>'
        '<circle cx="38" cy="116" r="3.5" fill="{texto}"/><circle cx="72" cy="78" r="3.8" fill="{terracota}"/>'
        '<circle cx="108" cy="74" r="3.8" fill="{terracota}"/><circle cx="142" cy="108" r="3.5" fill="{texto}"/>',
    ),
    "euler_heun": (
        "Esquema Euler y Heun",
        '<path d="M40,136 Q92,102 142,56" fill="none" stroke="{texto}" stroke-width="2" stroke-dasharray="4 3"/>'
        '<polyline points="40,136 90,118 140,86" fill="none" stroke="{terracota}" stroke-width="2"/>'
        '<polyline points="40,136 90,104 140,60" fill="none" stroke="{oro}" stroke-width="2.4"/>'
        '<circle cx="40" cy="136" r="4" fill="{texto}"/><circle cx="90" cy="104" r="4" fill="{oro}"/><circle cx="140" cy="60" r="4.5" fill="{oro}"/>',
    ),
    "runge_kutta": (
        "Esquema Runge-Kutta RK4",
        '<g stroke="{salvia}" stroke-width="1.5" opacity="0.85">'
        '<line x1="46" y1="128" x2="64" y2="116"/><line x1="74" y1="108" x2="94" y2="92"/><line x1="104" y1="84" x2="124" y2="66"/>'
        '</g>'
        '<path d="M42,136 C74,116 106,88 142,56" fill="none" stroke="{oro}" stroke-width="2.6"/>'
        '<circle cx="42" cy="136" r="4" fill="{texto}"/><circle cx="74" cy="112" r="3.8" fill="{salvia}"/>'
        '<circle cx="108" cy="84" r="3.8" fill="{terracota}"/><circle cx="142" cy="56" r="4.8" fill="{oro}"/>',
    ),
    "milne": (
        "Esquema Predictor-Corrector Milne",
        '<polyline points="38,136 62,120 86,102 110,84" fill="none" stroke="{salvia}" stroke-width="2.2"/>'
        '<line x1="110" y1="84" x2="142" y2="74" stroke="{terracota}" stroke-width="1.8" stroke-dasharray="3 2"/>'
        '<line x1="110" y1="84" x2="142" y2="56" stroke="{oro}" stroke-width="2.4"/>'
        '<line x1="142" y1="74" x2="142" y2="56" stroke="{texto}" stroke-width="1.4" stroke-dasharray="2 2"/>'
        '<circle cx="38" cy="136" r="3.5" fill="{salvia}"/><circle cx="62" cy="120" r="3.5" fill="{salvia}"/>'
        '<circle cx="86" cy="102" r="3.5" fill="{salvia}"/><circle cx="110" cy="84" r="4" fill="{texto}"/>'
        '<circle cx="142" cy="74" r="3.5" fill="{terracota}"/><circle cx="142" cy="56" r="4.8" fill="{oro}"/>',
    ),
}


@st.cache_data(show_spinner=False)
def obtener_svg_borrador_metodo(clave_svg: str, modo: str = "oscuro") -> str:
    """Retorna la ilustración vertical SVG borrador por defecto para una ficha de método."""
    paleta = PALETA_ARANDU_CLARO if modo == "claro" else PALETA_ARANDU
    colores_ficha = {
        "marco": paleta["bg_marco"],
        "borde": paleta["borde"],
        "oro": paleta["oro"],
        "lienzo": paleta["bg_lienzo"],
        "texto": paleta["texto_principal"],
        "salvia": paleta["salvia_borde"],
        "terracota": paleta["terracota_oscuro"],
        "horner": paleta["bg_horner"],
    }
    label, body = _BORRADOR_METODO_SVGS.get(clave_svg, _BORRADOR_METODO_SVGS["biseccion"])
    prefix = _FICHA_SVG_FRAME_PREFIX.format(label=html.escape(label), **colores_ficha)
    return f"{prefix}{body.format(**colores_ficha)}</svg>"


@st.cache_data(show_spinner=False)
def _construir_html_ficha_metodo_cached(
    numero: str,
    clave_svg: str,
    titulo: str,
    subtitulo: str,
    resumen: str,
    pista: str,
    activa: bool,
    firma_asset: tuple[str, int, str] | None,
    modo: str,
) -> str:
    clase_activa = " is-active" if activa else ""
    badge = '<span class="arandu-ficha-badge">Activo</span>' if activa else ""
    if firma_asset is not None:
        media_html = _leer_asset_metodo_html(*firma_asset, alt=titulo)
    else:
        media_html = obtener_svg_borrador_metodo(clave_svg, modo=modo)
    return (
        f'<article class="arandu-ficha-card{clase_activa}" aria-label="Método {html.escape(numero)}: {html.escape(titulo)}">'
        '<div class="arandu-ficha-main">'
        '<div class="arandu-ficha-top">'
        f'<span class="arandu-ficha-num">{html.escape(numero)}</span>{badge}'
        "</div>"
        f'<div class="arandu-ficha-art">{media_html}</div>'
        f'<div class="arandu-ficha-title">{html.escape(titulo)}</div>'
        f'<p class="arandu-ficha-desc">{html.escape(resumen)}</p>'
        "</div>"
        f'<div class="arandu-ficha-foot">{html.escape(pista)}</div>'
        "</article>"
    )


def construir_html_ficha_metodo(
    numero: str,
    clave_svg: str,
    titulo: str,
    subtitulo: str,
    resumen: str,
    pista: str,
    activa: bool = False,
) -> str:
    """Construye el HTML de una ficha coleccionable de estudio verificando imagen personalizada en assets/metodos/."""
    firma_asset = _buscar_asset_metodo(clave_svg)
    return _construir_html_ficha_metodo_cached(
        numero=numero,
        clave_svg=clave_svg,
        titulo=titulo,
        subtitulo=subtitulo,
        resumen=resumen,
        pista=pista,
        activa=activa,
        firma_asset=firma_asset,
        modo=obtener_modo_tema(),
    )


def aplicar_estilo_arandu() -> None:
    """Inyecta tipografías, tarjetas Guru, tablas doradas y controles orgánicos de Arandú."""
    st.markdown(obtener_css_arandu(obtener_modo_tema()), unsafe_allow_html=True)

