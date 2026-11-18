# 🦉 Arandú: Métodos Computacionales (v3.2.0)

Plataforma educativa interactiva desarrollada en **Python + Streamlit** para la cátedra de **Métodos Computacionales** (Facultad de Ciencias Exactas y Naturales y Agrimensura — **FaCENA, UNNE**), como parte del Proyecto Final de Carrera (PFC).

**"Arandú"** (sabiduría / entendimiento en guaraní) unifica el rigor académico de la cátedra con una estética **Dark Botanical & Golden Wisdom**, una arquitectura **Static-First** optimizada para producción y una capa pedagógica pensada para estudiantes de Sistemas:

1. **Estructura Didáctica en 3 Niveles (estilo Refactoring.Guru):**
   - **Nivel A — Resumen Ejecutivo:** Qué resuelve el método y su intuición computacional en 2–3 líneas.
   - **Nivel B — Intuición y Fórmulas (con Profundización Opcional):** Ecuaciones oficiales en LaTeX + bloque desplegable opcional con demostración y criterio formal.
   - **Nivel C — Cuándo Elegir / Evitar:** Reglas prácticas de ingeniería numérica y casos límite.
   - **Laboratorio Aislado (`@st.fragment`):** Teoría estática cargada una única vez; tablas editables (`st.data_editor`), controles de paso $\Delta h$ y gráficos Plotly interactivos aislados por fragmento.

2. **Estilo Versus.com (`⚔️ Arena Versus`):**
   - Comparador directo restringido estrictamente a métodos de la **misma familia matemática** (*Raíces*, *Sistemas Lineales*, *Autovalores*, *Interpolación*, *Integración* y *EDO/PVI*).
   - Contrasta fichas técnicas en Markdown/KaTeX nativo, comportamiento esperado, casos límite y métricas/curvas superpuestas en vivo.

---

## 🏗️ Arquitectura Dual de Despliegue (Cloud + Offline PWA v3.2.0)

Ambos entornos comparten exactamente el **mismo código fuente** (`app.py`, `core/`, `views/`, `.streamlit/config.toml` y `assets/`), detectando el entorno en tiempo de ejecución mediante `core.version.es_entorno_offline_wasm()`:

| Modo | Motor | Caso de Uso |
| :--- | :--- | :--- |
| **1. Streamlit Community Cloud** | Servidor Python + Tornado | Acceso instantáneo desde celulares, demos en clase y presentaciones ante jurado. |
| **2. PWA Offline (`stlite` + WebAssembly)** | Pyodide + `@stlite/browser` + Service Worker | Instalación como app en escritorio/móvil (Vercel / GitHub Pages) capaz de calcular sin internet tras la primera carga. |

### Flujo de Publicación Dual
- **Streamlit Cloud:**
  1. Conecta este repositorio en [share.streamlit.io](https://share.streamlit.io/) apuntando a `app.py` en la rama `main`.
  2. La configuración `.streamlit/config.toml` desactiva file-watchers y telemetría para minimizar consumo en la nube.
- **PWA Offline Estática (Vercel / GitHub Pages):**
  1. Ejecuta el empaquetador estático desde la raíz del proyecto:
     ```bash
     python build_offline.py
     ```
  2. Se generará la carpeta `dist_offline/` (excluida de Git) lista para desplegar en Vercel o GitHub Pages con `index.html`, `manifest.json`, `sw.js`, `vercel.json`, `assets/` y `.streamlit/config.toml`.
  3. Tras la primera carga en el navegador, el Service Worker cachea el runtime Pyodide, los paquetes científicos (`numpy`, `sympy`, `pandas`, `plotly`) y los módulos de Arandú para operar **100% sin conexión**.

---

## 🏗️ Estructura del Proyecto

```text
PFC-Jeremias-Ezequiel-Gonzalez/
├── app.py                          # Entrypoint definitivo (st.navigation + puente Cloud/WASM)
├── build_offline.py                # Generador de distribución estática PWA Offline (dist_offline/)
├── requirements.txt                # Dependencias para ejecución local y Streamlit Cloud
├── .streamlit/config.toml          # Tema Dark Botanical & directivas Static-First de producción
├── assets/                         # Logo JPG/SVG e identidades cargadas con @st.cache_data
├── core/                           # Lógica matemática pura + precomputación + cacheo compilado
│   ├── __init__.py                 # Re-exportaciones diferidas (PEP 562)
│   ├── _cache_utils.py             # Cacheo @st.cache_data/@st.cache_resource y vectorización NumPy
│   ├── version.py                  # APP_VERSION ("3.2.0"), URL PWA y detector WASM/Pyodide
│   ├── catalogo_ejercicios.py      # Catálogo 3 niveles, ejemplos precomputados y Fichas Versus
│   ├── fundamentos_errores.py      # Series (Taylor, Leibniz), Punto Flotante, Cancelación, Horner
│   ├── raices.py                   # Descartes, Tanteos, Bisección, Regula Falsi, Newton, Aitken
│   ├── sistemas_autovalores.py     # Gauss, Gauss-Jordan, LU, Gauss-Seidel, Potencias, Faddeev
│   ├── interpolacion.py            # Lineal, Cuadrática Inversa, Lagrange, Newton-Gregory + Δh
│   ├── integracion_derivacion.py   # Derivación Gregory (libre) + Cuadratura Adaptativa en Vivo
│   └── edo_pvi.py                  # Euler, Heun (iterado), RK2, RK4 y Milne (Orden 4)
└── views/                          # Vistas modulares Streamlit con @st.fragment
    ├── __init__.py                 # Re-exportaciones diferidas UI
    ├── theme.py                    # CSS Arandú + template Plotly cacheado + loaders de assets
    ├── ui_shared.py                # Renderizadores 3 niveles, paginación >300 filas y Plotly config
    ├── inicio.py                   # Portal principal y tarjetas de navegación
    ├── arena_versus.py             # ⚔️ Comparador directo de métodos
    ├── view_fundamentos.py         # 🧮 Fundamentos y Errores
    ├── view_raices.py              # 📉 Raíces de Ecuaciones
    ├── view_sistemas.py            # 📐 Sistemas Lineales y Autovalores
    ├── view_interpolacion.py       # 📈 Interpolación Polinomial
    ├── view_integracion.py         # 📊 Integración y Derivación Numérica
    ├── view_edo.py                 # ⏱️ Ecuaciones Diferenciales Ordinarias (PVI)
    └── view_laboratorios.py        # 📚 Anexo Curricular (Laboratorios 1 al 7)
```

---

## 🚀 Ejecución Local

```bash
pip install -r requirements.txt
streamlit run app.py
```
En Windows también puedes hacer doble clic en `ejecutar.bat`.

---

## 📚 Bibliografía, Créditos Académicos y Licencias

- **Licencia del software:** Código fuente bajo **MIT License** ([`LICENSE`](./LICENSE)) — © 2026 Jeremías Ezequiel González (Proyecto Final de Carrera, **FaCENA — UNNE**).
- **Bibliografía académica base:**
  - **Pace, Guido José** *(citado también como Pacce, Guido)*. *Métodos Numéricos*. Corrientes: **EUDENE** (UNNE), 1997. ISBN `950-656-017-X`.
  - **Chapra, Steven C.; Canale, Raymond P.** *Métodos numéricos para ingenieros*. McGraw-Hill Interamericana.
  - **Burden, Richard L.; Faires, J. Douglas.** *Análisis numérico*. Cengage Learning / Editorial Sudamericana.
  - **Cordero Barbero, A. y otros.** *Problemas resueltos de métodos numéricos*. Thomson / Paraninfo, 2006. ISBN `84-9732-409-9`.
- **Material docente de cátedra:** Guías de Laboratorio 1–7, filminas teóricas y propuestas de solución de la cátedra **Métodos Computacionales (FaCENA — UNNE)**, acreditando a su equipo docente (incluyendo guías/presentaciones de la Prof. **Flavia S. Pucheta**).
- **Dependencias y tipografías:** Detalle completo de licencias open-source (`Apache-2.0`, `BSD`, `MIT`, `MPL-2.0`) y fuentes (`SIL OFL 1.1`) en [`CREDITOS_Y_LICENCIAS.md`](./CREDITOS_Y_LICENCIAS.md).
