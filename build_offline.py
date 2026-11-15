"""
Generador de distribución estática PWA Offline (stlite + WebAssembly) para Arandú.

Empaqueta exactamente los mismos archivos fuente que Streamlit Community Cloud
(`app.py`, `core/`, `views/`, `.streamlit/config.toml` y `assets/`) dentro de
`dist_offline/` listo para desplegar en Vercel o GitHub Pages sin duplicar lógica.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.version import APP_VERSION
from views.theme import PALETA_ARANDU, hex_a_rgba

ROOT_DIR = Path(__file__).resolve().parent
DIST_DIR = ROOT_DIR / "dist_offline"
STLITE_CDN_BASE = "https://cdn.jsdelivr.net/npm/@stlite/browser@1.12.2/build"
REQUIRED_PACKAGES = ["sympy", "numpy", "pandas", "plotly"]


def _preparar_directorio_salida() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)


def _copiar_activos_y_config() -> None:
    assets_src = ROOT_DIR / "assets"
    assets_dst = DIST_DIR / "assets"
    if assets_src.exists():
        shutil.copytree(
            assets_src,
            assets_dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".gitkeep"),
        )

    config_src = ROOT_DIR / ".streamlit" / "config.toml"
    if config_src.exists():
        config_dst = DIST_DIR / ".streamlit" / "config.toml"
        config_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_src, config_dst)


def _recolectar_vfs() -> dict[str, object]:
    """Construye el mapa de archivos virtuales (POSIX) para montar en stlite."""
    archivos: dict[str, object] = {}

    for rel in ("app.py", ".streamlit/config.toml", "assets/svg/arandu.svg"):
        ruta = ROOT_DIR / rel
        if ruta.exists():
            archivos[rel] = ruta.read_text(encoding="utf-8")

    for carpeta in ("core", "views"):
        dir_path = ROOT_DIR / carpeta
        if not dir_path.exists():
            continue
        for py_file in sorted(dir_path.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel_posix = py_file.relative_to(ROOT_DIR).as_posix()
            archivos[rel_posix] = py_file.read_text(encoding="utf-8")

    logo_jpg = ROOT_DIR / "assets" / "logo_arandu.jpg"
    if logo_jpg.exists():
        archivos["assets/logo_arandu.jpg"] = {"url": "./assets/logo_arandu.jpg"}

    metodos_dir = ROOT_DIR / "assets" / "metodos"
    if metodos_dir.exists():
        for img_file in sorted(metodos_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg"):
                rel_posix = img_file.relative_to(ROOT_DIR).as_posix()
                if img_file.suffix.lower() == ".svg":
                    archivos[rel_posix] = img_file.read_text(encoding="utf-8")
                else:
                    archivos[rel_posix] = {"url": f"./{rel_posix}"}

    return archivos


def _generar_index_html(archivos_vfs: dict[str, object]) -> None:
    vfs_json = json.dumps(archivos_vfs, ensure_ascii=False).replace("</", "<\\/")
    pkgs_json = json.dumps(REQUIRED_PACKAGES, ensure_ascii=False)
    streamlit_cfg_json = json.dumps(
        {
            "theme.base": "dark",
            "theme.primaryColor": PALETA_ARANDU["oro"],
            "theme.backgroundColor": PALETA_ARANDU["bg_principal"],
            "theme.secondaryBackgroundColor": PALETA_ARANDU["bg_secundario"],
            "theme.textColor": PALETA_ARANDU["texto_principal"],
            "theme.font": "sans serif",
            "client.toolbarMode": "minimal",
            "client.showErrorDetails": False,
            "browser.gatherUsageStats": False,
        },
        ensure_ascii=False,
    )
    css_root_lines = "\n".join(
        f"      {clave}: {valor};"
        for clave, valor in {
            "--arandu-bg": PALETA_ARANDU["bg_principal"],
            "--arandu-surface": PALETA_ARANDU["bg_secundario"],
            "--arandu-gold": PALETA_ARANDU["oro"],
            "--arandu-sage": PALETA_ARANDU["salvia"],
            "--arandu-sage-border": PALETA_ARANDU["salvia_borde"],
            "--arandu-text": PALETA_ARANDU["texto_principal"],
            "--arandu-muted": PALETA_ARANDU["texto_secundario"],
            "--arandu-border": PALETA_ARANDU["borde"],
            "--arandu-elevated": PALETA_ARANDU["bg_elevado"],
            "--arandu-gold-14": hex_a_rgba(PALETA_ARANDU["oro"], 0.14),
            "--arandu-gold-20": hex_a_rgba(PALETA_ARANDU["oro"], 0.20),
            "--arandu-gold-35": hex_a_rgba(PALETA_ARANDU["oro"], 0.35),
            "--arandu-surface-94": hex_a_rgba(PALETA_ARANDU["bg_secundario"], 0.94),
            "--arandu-shadow-45": hex_a_rgba(PALETA_ARANDU["negro_sombra"], 0.45),
        }.items()
    )

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <meta name="theme-color" content="{PALETA_ARANDU["bg_principal"]}" />
  <meta name="description" content="Arandú — Métodos Computacionales v{APP_VERSION} (Modo Offline WebAssembly / PWA)" />
  <title>Arandú — Métodos Computacionales (Offline v{APP_VERSION})</title>
  <link rel="manifest" href="./manifest.json" />
  <link rel="icon" type="image/svg+xml" href="./assets/svg/arandu.svg" />
  <link rel="apple-touch-icon" href="./assets/logo_arandu.jpg" />
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@600;700&display=swap" />
  <link rel="stylesheet" href="{STLITE_CDN_BASE}/stlite.css" />
  <style>
    :root {{
{css_root_lines}
    }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background-color: var(--arandu-bg);
      color: var(--arandu-text);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.65;
      overflow-x: hidden;
    }}
    #root {{
      width: 100%;
      min-height: 100vh;
    }}
    #arandu-splash {{
      position: fixed;
      inset: 0;
      z-index: 9999;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem;
      background: radial-gradient(circle at top center, var(--arandu-gold-14), var(--arandu-bg) 62%);
      color: var(--arandu-text);
      text-align: center;
      transition: opacity 0.4s ease;
    }}
    #arandu-splash.oculto {{
      opacity: 0;
      pointer-events: none;
    }}
    .splash-logo {{
      width: 96px;
      height: 96px;
      margin-bottom: 1.1rem;
      filter: drop-shadow(0 0 18px var(--arandu-gold-35));
    }}
    .splash-title {{
      margin: 0;
      font-family: 'Outfit', sans-serif;
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--arandu-text);
    }}
    .splash-badge {{
      display: inline-block;
      margin-top: 0.55rem;
      padding: 0.25rem 0.75rem;
      border-radius: 999px;
      border: 1px solid var(--arandu-gold);
      background: var(--arandu-gold-14);
      color: var(--arandu-gold);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .splash-spinner {{
      width: 38px;
      height: 38px;
      margin: 1.4rem 0 0.9rem 0;
      border-radius: 50%;
      border: 3px solid var(--arandu-gold-20);
      border-top-color: var(--arandu-gold);
      animation: arandu-spin 0.9s linear infinite;
    }}
    @keyframes arandu-spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .splash-status {{
      max-width: 520px;
      font-size: 0.92rem;
      line-height: 1.6;
      color: var(--arandu-muted);
      margin-bottom: 1rem;
    }}
    .splash-skeleton-grid {{
      width: min(760px, 92vw);
      display: grid;
      grid-template-columns: 200px 1fr;
      gap: 0.9rem;
      margin-top: 0.35rem;
      text-align: left;
    }}
    @media (max-width: 640px) {{
      .splash-skeleton-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    .sk-panel {{
      background-color: var(--arandu-surface);
      border: 1px solid var(--arandu-border);
      border-radius: 12px;
      padding: 0.85rem;
    }}
    .sk-bar {{
      height: 12px;
      border-radius: 6px;
      margin-bottom: 0.65rem;
      background: linear-gradient(90deg, var(--arandu-elevated) 25%, var(--arandu-border) 50%, var(--arandu-elevated) 75%);
      background-size: 200% 100%;
      animation: arandu-shimmer 1.5s infinite linear;
    }}
    .sk-bar.w-60 {{ width: 60%; }}
    .sk-bar.w-80 {{ width: 80%; }}
    .sk-bar.w-45 {{ width: 45%; }}
    .sk-bar.tall {{ height: 48px; border-radius: 10px; }}
    .sk-cards {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-top: 0.5rem;
    }}
    @keyframes arandu-shimmer {{
      to {{ background-position: -200% 0; }}
    }}
    .splash-retry {{
      display: none;
      margin-top: 1rem;
      padding: 0.55rem 1.15rem;
      border-radius: 10px;
      border: 1px solid var(--arandu-gold);
      background: var(--arandu-elevated);
      color: var(--arandu-text);
      font-weight: 600;
      cursor: pointer;
    }}
    .splash-retry.visible {{
      display: inline-block;
    }}
    #arandu-net-pill {{
      display: none;
      position: fixed;
      right: 1rem;
      bottom: 1rem;
      z-index: 9998;
      padding: 0.45rem 0.85rem;
      border-radius: 999px;
      background: var(--arandu-surface-94);
      border: 1px solid var(--arandu-sage-border);
      border-left: 4px solid var(--arandu-sage);
      color: var(--arandu-text);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      font-weight: 600;
      box-shadow: 0 6px 18px var(--arandu-shadow-45);
    }}
    #arandu-net-pill.visible {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }}
  </style>
</head>
<body>
  <div id="arandu-splash" role="status" aria-live="polite">
    <img src="./assets/svg/arandu.svg" class="splash-logo" alt="Isotipo de Arandú" />
    <h1 class="splash-title">Arandú — Métodos Computacionales</h1>
    <span class="splash-badge">v{APP_VERSION} • PWA Offline WebAssembly</span>
    <div class="splash-spinner" id="arandu-spinner"></div>
    <p id="arandu-status" class="splash-status">
      Inicializando Python y librerías científicas (<code>numpy</code>, <code>sympy</code>, <code>pandas</code>, <code>plotly</code>) en tu navegador.<br />
      La primera carga guarda todo localmente para funcionar 100% sin conexión.
    </p>
    <div class="splash-skeleton-grid" aria-hidden="true">
      <div class="sk-panel">
        <div class="sk-bar w-60"></div>
        <div class="sk-bar w-80"></div>
        <div class="sk-bar w-45"></div>
        <div class="sk-bar w-80"></div>
      </div>
      <div class="sk-panel">
        <div class="sk-bar tall"></div>
        <div class="sk-cards">
          <div class="sk-bar tall"></div>
          <div class="sk-bar tall"></div>
        </div>
      </div>
    </div>
    <button type="button" id="arandu-retry-btn" class="splash-retry" onclick="window.location.reload()">
      Reintentar carga
    </button>
  </div>
  <div id="arandu-net-pill" role="status" aria-live="polite">
    Sin conexión • Motor WebAssembly local activo
  </div>
  <div id="root"></div>

  <script type="module">
    import {{ mount }} from "{STLITE_CDN_BASE}/stlite.js";

    const safeStorageGet = (key) => {{
      try {{
        return window.localStorage ? window.localStorage.getItem(key) : null;
      }} catch (_err) {{
        return null;
      }}
    }};

    const safeStorageSet = (key, value) => {{
      try {{
        if (window.localStorage) window.localStorage.setItem(key, value);
      }} catch (_err) {{}}
    }};

    const actualizarEstadoRed = () => {{
      const pill = document.getElementById("arandu-net-pill");
      if (!pill) return;
      const offline = typeof navigator !== "undefined" && navigator.onLine === false;
      pill.classList.toggle("visible", offline);
    }};
    window.addEventListener("offline", actualizarEstadoRed);
    window.addEventListener("online", actualizarEstadoRed);
    actualizarEstadoRed();

    const mostrarModoOfflineSinCache = (detalle) => {{
      const status = document.getElementById("arandu-status");
      const spinner = document.getElementById("arandu-spinner");
      const retryBtn = document.getElementById("arandu-retry-btn");
      if (spinner) spinner.style.display = "none";
      if (retryBtn) retryBtn.classList.add("visible");
      if (status) {{
        status.innerHTML =
          "<b>Sin conexión a internet antes de completar la caché inicial.</b><br />" +
          "Conectate una primera vez para descargar Pyodide o tocá <i>Reintentar carga</i> cuando vuelva la red." +
          (detalle ? "<br /><small>" + detalle + "</small>" : "");
      }}
    }};

    const ocultarSplash = () => {{
      safeStorageSet("arandu_wasm_ready_v{APP_VERSION}", "1");
      const splash = document.getElementById("arandu-splash");
      if (splash && !splash.classList.contains("oculto")) {{
        splash.classList.add("oculto");
        setTimeout(() => splash.remove(), 450);
      }}
    }};

    // Registro seguro de Service Worker para caché offline
    try {{
      if ("serviceWorker" in navigator) {{
        window.addEventListener("load", () => {{
          navigator.serviceWorker.register("./sw.js").catch(() => {{}});
        }});
      }}
    }} catch (_err) {{}}

    const yaCacheado = safeStorageGet("arandu_wasm_ready_v{APP_VERSION}") === "1";
    if (typeof navigator !== "undefined" && navigator.onLine === false && !yaCacheado) {{
      mostrarModoOfflineSinCache("Aún no se guardaron paquetes en este navegador.");
    }}

    // Ocultar splash apenas Streamlit renderice el contenedor principal
    const observer = new MutationObserver(() => {{
      const root = document.getElementById("root");
      if (root && root.querySelector('[data-testid="stAppViewContainer"]')) {{
        ocultarSplash();
        observer.disconnect();
      }}
    }});
    observer.observe(document.getElementById("root"), {{ childList: true, subtree: true }});

    const files = {vfs_json};
    const requirements = {pkgs_json};
    const streamlitConfig = {streamlit_cfg_json};

    try {{
      const app = mount(
        {{
          requirements,
          entrypoint: "app.py",
          files,
          streamlitConfig,
        }},
        document.getElementById("root")
      );
      if (app && typeof app.then === "function") {{
        app.then(() => setTimeout(ocultarSplash, 800)).catch((err) => {{
          if (typeof navigator !== "undefined" && navigator.onLine === false) {{
            mostrarModoOfflineSinCache(err?.message || "");
            return;
          }}
          const status = document.getElementById("arandu-status");
          const retryBtn = document.getElementById("arandu-retry-btn");
          if (retryBtn) retryBtn.classList.add("visible");
          if (status) {{
            status.innerHTML = "<b>No se pudo iniciar el entorno WebAssembly:</b> " + (err?.message || err);
          }}
        }});
      }}
    }} catch (err) {{
      if (typeof navigator !== "undefined" && navigator.onLine === false) {{
        mostrarModoOfflineSinCache(err?.message || "");
      }} else {{
        const status = document.getElementById("arandu-status");
        const retryBtn = document.getElementById("arandu-retry-btn");
        if (retryBtn) retryBtn.classList.add("visible");
        if (status) {{
          status.innerHTML = "<b>Error cargando Arandú Offline:</b> " + (err?.message || err);
        }}
      }}
    }}
  </script>
</body>
</html>
"""
    (DIST_DIR / "index.html").write_text(html, encoding="utf-8")


def _generar_manifest() -> None:
    manifest = {
        "name": "Arandú — Métodos Computacionales",
        "short_name": "Arandú",
        "description": "Plataforma educativa de Métodos Numéricos con ejecución Python offline en el navegador.",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "background_color": PALETA_ARANDU["bg_principal"],
        "theme_color": PALETA_ARANDU["oro"],
        "lang": "es-AR",
        "icons": [
            {
                "src": "./assets/svg/arandu.svg",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any",
            },
            {
                "src": "./assets/logo_arandu.jpg",
                "sizes": "512x512",
                "type": "image/jpeg",
                "purpose": "any",
            },
        ],
    }
    (DIST_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _generar_service_worker() -> None:
    precache = [
        "./",
        "./index.html",
        "./manifest.json",
        "./assets/svg/arandu.svg",
        "./assets/logo_arandu.jpg",
        "./.streamlit/config.toml",
        f"{STLITE_CDN_BASE}/stlite.css",
        f"{STLITE_CDN_BASE}/stlite.js",
    ]
    metodos_dir = ROOT_DIR / "assets" / "metodos"
    if metodos_dir.exists():
        for img_file in sorted(metodos_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg"):
                precache.append(f"./{img_file.relative_to(ROOT_DIR).as_posix()}")
    precache_json = json.dumps(precache, indent=2, ensure_ascii=False)

    sw_js = f"""const CACHE_SHELL = "arandu-offline-shell-v{APP_VERSION}";
const CACHE_RUNTIME = "arandu-offline-runtime-v{APP_VERSION}";

const PRECACHE_URLS = {precache_json};

self.addEventListener("install", (event) => {{
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_SHELL).then(async (cache) => {{
      await Promise.allSettled(PRECACHE_URLS.map((u) => cache.add(u)));
    }})
  );
}});

self.addEventListener("activate", (event) => {{
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_SHELL && k !== CACHE_RUNTIME)
          .map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
}});

const esRecursoCdnInmutable = (url) => {{
  const host = url.hostname;
  if (
    host.includes("cdn.jsdelivr.net") ||
    host.includes("pypi.org") ||
    host.includes("files.pythonhosted.org") ||
    host.includes("fonts.googleapis.com") ||
    host.includes("fonts.gstatic.com")
  ) {{
    return true;
  }}
  return /\\.(wasm|whl|zip|data|so|js|css|woff2?)$/i.test(url.pathname);
}};

self.addEventListener("fetch", (event) => {{
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.protocol !== "http:" && url.protocol !== "https:") return;

  if (esRecursoCdnInmutable(url)) {{
    // Cache-First para Pyodide, wheels científicas y runtime @stlite/browser
    event.respondWith(
      caches.match(req).then((cached) => {{
        if (cached) return cached;
        return fetch(req)
          .then((resp) => {{
            if (resp && (resp.ok || resp.type === "opaque")) {{
              const clone = resp.clone();
              caches.open(CACHE_RUNTIME).then((cache) => cache.put(req, clone));
            }}
            return resp;
          }})
          .catch(() => caches.match(req).then((r) => r || new Response("", {{ status: 503, statusText: "Offline" }})));
      }})
    );
    return;
  }}

  // Stale-While-Revalidate para app shell y assets locales
  event.respondWith(
    caches.match(req).then((cached) => {{
      const networkFetch = fetch(req)
        .then((resp) => {{
          if (resp && resp.ok) {{
            const clone = resp.clone();
            caches.open(CACHE_SHELL).then((cache) => cache.put(req, clone));
          }}
          return resp;
        }})
        .catch(() => {{
          if (req.mode === "navigate") {{
            return caches.match("./index.html");
          }}
          return cached;
        }});
      return cached || networkFetch;
    }})
  );
}});
"""
    (DIST_DIR / "sw.js").write_text(sw_js, encoding="utf-8")


def _generar_vercel_config() -> None:
    vercel_cfg = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "headers": [
            {
                "source": "/(.*)",
                "headers": [
                    {"key": "X-Content-Type-Options", "value": "nosniff"},
                    {"key": "X-Frame-Options", "value": "DENY"},
                    {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
                    {"key": "Permissions-Policy", "value": "camera=(), microphone=(), geolocation=()"},
                ],
            },
            {
                "source": "/sw.js",
                "headers": [
                    {"key": "Cache-Control", "value": "no-cache, no-store, must-revalidate"},
                    {"key": "Service-Worker-Allowed", "value": "/"},
                ],
            },
            {
                "source": "/manifest.json",
                "headers": [
                    {"key": "Content-Type", "value": "application/manifest+json; charset=utf-8"},
                    {"key": "Cache-Control", "value": "public, max-age=3600"},
                ],
            },
            {
                "source": "/assets/(.*)",
                "headers": [
                    {"key": "Cache-Control", "value": "public, max-age=86400, stale-while-revalidate=604800"},
                ],
            },
        ],
    }
    (DIST_DIR / "vercel.json").write_text(
        json.dumps(vercel_cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build() -> Path:
    """Ejecuta el empaquetado completo en `dist_offline/` y devuelve la ruta."""
    _preparar_directorio_salida()
    _copiar_activos_y_config()
    archivos_vfs = _recolectar_vfs()
    _generar_index_html(archivos_vfs)
    _generar_manifest()
    _generar_service_worker()
    _generar_vercel_config()
    return DIST_DIR


if __name__ == "__main__":
    salida = build()
    print(f"[Arandú v{APP_VERSION}] Build PWA Offline generado en: {salida}")

