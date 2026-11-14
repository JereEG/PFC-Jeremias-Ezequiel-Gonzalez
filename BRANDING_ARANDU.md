# 🌳 Manual de Identidad Visual — Organización "Arandú"

> **Estética:** *Dark Botanical & Golden Wisdom*  
> **Origen:** Proyecto Final de Carrera — Métodos Computacionales (FaCENA — UNNE)  
> **Propósito:** Definir el lenguaje visual, cromático, tipográfico y comunicacional de **Arandú**, la futura organización dedicada a crear herramientas educativas interactivas de matemática aplicada y ciencia de la computación.

---

## 1. Manifiesto de Arandú

En lengua guaraní, **Arandú** significa *sabiduría*, *conocimiento profundo* y *capacidad de comprender el orden del mundo a partir de la experiencia y la razón*.  

Creemos que la enseñanza de los Métodos Computacionales no debe esconderse detrás de cajas negras ni interfaces frías o genéricas:
1. **El conocimiento debe tener raíces visibles:** cada algoritmo nace de un teorema y de una cota de error que el estudiante debe poder inspeccionar.
2. **Aprender es intervenir los datos:** ninguna fórmula se comprende del todo hasta que se modifica un parámetro, un paso $\Delta h$ o una fila de una matriz y se observa el impacto geométrico en tiempo real.
3. **La belleza visual sirve a la claridad cognitiva:** una paleta cálida, de alto contraste reposado sobre fondo negro obsidiana, reduce la fatiga visual y concentra la atención en la curva, el error y la convergencia.

---

## 2. Simbología del Isotipo Oficial

El isotipo oficial (`assets/logo_arandu.jpg` y su versión vectorial transparente `assets/svg/arandu.svg`) representa al sabio unido al árbol cósmico sobre fondo negro obsidiana. Cada parte del emblema estructura también nuestra arquitectura pedagógica:

| Elemento del Isotipo | Significado en la Plataforma |
| :--- | :--- |
| 🌱 **Raíces** | **Fundamentos teóricos y análisis del error:** Series de Taylor, aritmética flotante de máquina, cancelación catastrófica y cotas de estabilidad. |
| 🪵 **Tronco** | **Algoritmos interactivos paso a paso y prácticas de cátedra:** Tablas editables en vivo, resolución con $\Delta h$ dinámico y trazabilidad completa de cada iteración. |
| ✨ **Copa, Viento y Astros** | **Visualización geométrica y Arena Versus:** Rectas tangentes, secantes, arcos parabólicos, trayectorias de EDOs y duelos comparativos de convergencia entre métodos de una misma familia. |

---

## 3. Paleta Cromática Oficial (*Dark Botanical & Golden Wisdom*)

### A. Superficies y Tipografía Base (`.streamlit/config.toml` y CSS)

| Token (`views/theme.py`) | Hex / RGBA | Rol Visual |
| :--- | :--- | :--- |
| `ARANDU_OBSIDIAN` | `#0A0B09` | Fondo principal profundo de la aplicación (`backgroundColor`). |
| `ARANDU_BARK` | `#161814` | Superficie de tarjetas (`arandu-card`), sidebar y contenedores (`secondaryBackgroundColor`). |
| `ARANDU_PLOT_BG` | `#121410` | Fondo interior del lienzo de gráficos Plotly (`plot_bgcolor`). |
| `ARANDU_CARD_BORDER` | `#2E2920` | Bordes sutiles de tarjetas y contenedores. |
| `ARANDU_GRID` | `#26241E` | Líneas de grilla discretas en ejes `x` e `y` de Plotly. |
| `ARANDU_IVORY` | `#EDE4D3` | Color principal de texto y ejes (`textColor`), optimizado para lectura prolongada sin encandilamiento. |
| `ARANDU_MUTED` | `#9E9585` | Trazos auxiliares de referencia ($y = x$, curvas de grado completo punteadas). |

### B. Acentos de Marca e Interacción

| Token (`views/theme.py`) | Hex | Uso Destinado |
| :--- | :--- | :--- |
| `ARANDU_GOLD` | `#D4A359` | **Color Primario** (`primaryColor`): marca, títulos destacados, bordes de tablas editables (`st.data_editor`), borde izquierdo de tarjetas principales. |
| `ARANDU_SAGE` / `ARANDU_SAGE_BORDER` | `#7A9A65` / `#687D5B` | Acento botánico secundario: tarjetas complementarias (`arandu-card-sage`), estabilidad y convergencia favorable. |
| `ARANDU_TERRACOTTA` / `ARANDU_TERRACOTTA_DARK` | `#C86D44` / `#B5673F` | Estado `hover` en botones/pestañas, ejes cero (`zeroline`) en Plotly y alertas cálidas. |

### C. Ciclo Oficial de Colores Matemáticos (Plotly)

Todos los gráficos heredan `PLOTLY_ARANDU_TEMPLATE` y deben respetar la siguiente jerarquía semántica:

1. **Función principal $f(x)$ o Solución Exacta:** `#D4A359` (*Oro Arandú*, línea sólida o punteada de `2.5px`–`2.8px`).
2. **Aproximación, Polinomio Interpolador, Secantes o Simpson 1/3:** `#7A9A65` (*Verde Salvia*, relleno en áreas con `rgba(122, 154, 101, 0.25)`).
3. **Rectas Tangentes, Contendiente B / Método Alternativo o Trapecios:** `#C86D44` (*Terracota Cálido*, relleno en áreas con `rgba(200, 109, 68, 0.25)`).
4. **Raíces Halladas, Nodos y Puntos Críticos:** `#FFFDF9` (*Marfil Brillante*) con borde `#B5673F` de `1.8px`.

> Quedan prohibidos los colores estridentes por defecto (`#2563eb`, cyan neón, magenta, amarillo puro `#ffff00` o fondos blancos en gráficos).

---

## 4. Tipografías Oficiales

Se cargan desde Google Fonts en `aplicar_estilo_arandu()` (`views/theme.py`):

- **Títulos de Marca y Encabezados Principales (`h1`, `h2`):** `'Cinzel'` (alternativa `'Fraunces'`). Aporta carácter editorial y atemporal asociado a la sabiduría clásica.
- **Cuerpo de Lectura, Explicaciones Teóricas y Controles:** `'Plus Jakarta Sans'`. Proporciona legibilidad moderna, moderna apertura de caracteres y lectura ágil en bloques estilo *Refactoring.Guru*.
- **Bloques Numéricos, Código y Métricas (`st.metric`, `st.code`):** `'JetBrains Mono'`. Asegura alineación tabular limpia de mantisas, matrices, residuos y errores relativos.

---

## 5. Reglas de Tono de Comunicación

> **"Claridad pedagógica, rigor matemático, cero complejidad innecesaria."**

1. **Directo al concepto:** explicar en 2 o 3 líneas *qué problema resuelve*, *cuál es su intuición geométrica*, *cuándo usarlo* y *cuándo falla*. Evitar párrafos académicos extensos antes de ver el método en acción.
2. **Rigor sin opacidad:** conservar las definiciones formales en LaTeX, criterios de paro oficiales (Scarborough, Fourier, Bolzano, dominancia diagonal) y terminología exacta de cátedra.
3. **Invitación explícita a experimentar:** cada tabla o matriz editable debe lucir visualmente destacada (borde punteado dorado `#D4A359`) y permitir restaurar los datos originales con un clic.
4. **Comparación honesta (Arena Versus):** mostrar no solo quién converge más rápido, sino también los escenarios patológicos donde un método superior diverge o falla.

---

## 6. Cómo Aplicar esta Identidad en Futuros Repositorios de Arandú Org

Para unificar cualquier nueva herramienta o librería web creada bajo la organización **Arandú**:

1. **Tema nativo de Streamlit:** copiar el archivo `.streamlit/config.toml` en la raíz del nuevo repositorio para inicializar el servidor directamente en modo oscuro obsidiana/oro.
2. **Activos gráficos:** incluir `assets/logo_arandu.jpg` (fondo `#0A0B09`) para cabeceras circulares / `st.logo` y `assets/svg/arandu.svg` para íconos vectoriales y favicons adaptables.
3. **Motor de estilo centralizado:** reutilizar `views/theme.py` (o publicarlo como paquete compartido `arandu-theme`) e invocar:
   ```python
   import streamlit as st
   from views.theme import LOGO_JPG_PATH, PLOTLY_ARANDU_TEMPLATE, aplicar_estilo_arandu

   st.set_page_config(page_title="Arandú — Módulo", page_icon=str(LOGO_JPG_PATH), layout="wide")
   st.logo(str(LOGO_JPG_PATH), size="large")
   aplicar_estilo_arandu()
   ```
4. **Gráficos consistentes:** crear las figuras pasando por un constructor común que asigne `template=PLOTLY_ARANDU_TEMPLATE` y respete el ciclo cromático `#D4A359` → `#7A9A65` → `#C86D44` → `#FFFDF9`.

