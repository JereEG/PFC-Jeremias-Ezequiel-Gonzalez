# 🦉 Arandú: Métodos Computacionales (v3.0)

Plataforma educativa interactiva desarrollada en **Python + Streamlit** para la cátedra de **Métodos Computacionales** (Facultad de Ciencias Exactas y Naturales y Agrimensura — **FaCENA, UNNE**), como parte del Proyecto Final de Carrera (PFC).

**"Arandú"** (sabiduría / entendimiento en guaraní) unifica el rigor matemático de la cátedra con dos filosofías modernas de aprendizaje visual:

1. **Estilo Refactoring.Guru (en cada Método):**
   - **Bloque 1 — Teoría Intuitiva y Ejemplo Oficial:** Qué resuelve, intuición geométrica, cuándo usar, cuándo falla, fórmulas oficiales en LaTeX y desarrollo paso a paso del Ejemplo Teórico Oficial.
   - **Bloque 2 — Práctica de Cátedra Interactiva:** Datasets y ecuaciones oficiales precargados con tablas/matrices editables en vivo (`st.data_editor`), indicador visual de edición, selector `Ejercicio de Cátedra` vs `Edición Manual` y botón `🔄 Restaurar valores originales de la profesora`.
   - **Bloque 3 — Controles en Vivo (incluyendo paso $\Delta h$ dinámico):** Ajuste interactivo de tolerancias, iteraciones máximas, dígitos de máquina $k$ y tamaño de paso $\Delta h$ / subintervalos $n$. En Integración Numérica, si un cambio de $\Delta h$ o $n$ viola las condiciones de paridad/divisibilidad (ej. $n$ impar en Simpson 1/3), el motor adaptativo (`integrar_adaptativo_en_vivo`) combina automáticamente reglas válidas (**Simpson 3/8 + 1/3** o **Simpson 1/3 + Trapecio**) y lo explica pedagógicamente. En Interpolación, detecta nodos no equidistantes invalidando Newton-Gregory y ofreciendo respaldo por Lagrange.
   - **Bloque 4 — Visualización Geométrica Paso a Paso:** Gráficos interactivos en **Plotly** con deslizador `"Ver hasta la iteración / paso k"` (tangentes de Newton, cuerdas secantes, intervalos de Bolzano, telarañas de Punto Fijo, polinomio $P(\lambda)$ de Faddeev, sub-bloques coloreados bajo integrales y trayectorias de EDO frente a la solución exacta).

2. **Estilo Versus.com (`⚔️ Arena Versus`):**
   - Comparador directo restringido estrictamente a métodos de la **misma familia matemática** (*Raíces*, *Sistemas Lineales*, *Autovalores*, *Interpolación*, *Integración* y *EDO/PVI*).
   - Presenta fichas técnicas paralelas, bloque `"🧠 ¿Qué esperamos ver teóricamente?"`, advertencias sobre excepciones/casos patológicos y duelo en vivo con métricas y curvas superpuestas.

---

## 🏗️ Arquitectura del Proyecto (Cloud-Ready)

```text
PFC-Jeremias-Ezequiel-Gonzalez/
├── app.py                          # Entrypoint definitivo (st.navigation + st.Page)
├── requirements.txt                # Dependencias para ejecución local y Streamlit Cloud
├── ejecutar.bat                    # Lanzador rápido en Windows
├── README.md
├── core/                           # Lógica matemática pura (sin imports de Streamlit ni I/O)
│   ├── __init__.py
│   ├── catalogo_ejercicios.py      # Catálogo Guru, Ejemplo Teórico, Prácticas y Fichas Versus
│   ├── fundamentos_errores.py      # Series (Taylor, Leibniz), Punto Flotante, Cancelación, Horner
│   ├── raices.py                   # Descartes, Tanteos, Bisección, Regula Falsi, Newton, Aitken
│   ├── sistemas_autovalores.py     # Gauss, Gauss-Jordan, LU, Gauss-Seidel, Potencias, Faddeev
│   ├── interpolacion.py            # Lineal, Cuadrática Inversa, Lagrange, Newton-Gregory + Δh
│   ├── integracion_derivacion.py   # Derivación Gregory (libre) + Cuadratura Adaptativa en Vivo
│   └── edo_pvi.py                  # Euler, Heun (iterado), RK2, RK4 y Milne (Orden 4)
└── views/                          # Vistas modulares Streamlit (solo importan desde core/ y views/)
    ├── __init__.py                 # CSS compartido, renderizadores Bloque 1/2 y helpers Plotly
    ├── inicio.py                   # Portal principal y tarjetas de navegación
    ├── arena_versus.py             # ⚔️ Comparador directo estilo Versus.com
    ├── view_fundamentos.py         # 🧮 Fundamentos y Errores
    ├── view_raices.py              # 📉 Raíces de Ecuaciones
    ├── view_sistemas.py            # 📐 Sistemas Lineales y Autovalores
    ├── view_interpolacion.py       # 📈 Interpolación Polinomial
    ├── view_integracion.py         # 📊 Integración y Derivación Numérica
    ├── view_edo.py                 # ⏱️ Ecuaciones Diferenciales Ordinarias (PVI)
    └── view_laboratorios.py        # 📚 Anexo Curricular (Laboratorios 1 al 7)
```

---

## 🚀 Ejecución Local y Despliegue en Streamlit Community Cloud

### Local (Windows / Linux / macOS)
```bash
pip install -r requirements.txt
streamlit run app.py
```
En Windows también puedes hacer doble clic en `ejecutar.bat`.

### Streamlit Community Cloud
1. Conecta este repositorio en [share.streamlit.io](https://share.streamlit.io/).
2. Selecciona la rama principal y define `app.py` como **Main file path**.
3. La aplicación es **100% autónoma**: no lee carpetas temporales ni requiere archivos binarios externos en tiempo de ejecución.
