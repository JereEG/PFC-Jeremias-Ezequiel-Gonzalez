# 🏛️ El Santo Grial de los Métodos Numéricos (FaCENA - UNNE)
## Proyecto Final de Carrera (PFC) — Jeremías Ezequiel González

Plataforma integral, educativa y de referencia interactiva para el estudio y aplicación de **Métodos Numéricos**, desarrollada para la cátedra de **Métodos Computacionales** de la **Facultad de Ciencias Exactas y Naturales y Agrimensura (FaCENA - UNNE)**.

Arquitectura desacoplada en capas y 100% **Cloud-Ready** para su despliegue en **Streamlit Community Cloud**.

---

## 🧭 Arquitectura del Sistema: Enfoque Centrado en Métodos

A diferencia de un catálogo lineal por laboratorios, la plataforma organiza el conocimiento en torno a los **Métodos Matemáticos** como núcleo principal, manteniendo un anexo histórico curricular:

```text
/
├── assets/
│   └── teoria/                        # PDFs de cátedra: fundamentos teóricos y ejercicios resueltos
│
├── core/                              # MÓDULO MATEMÁTICO PURO (Funciones puras, 0 UI/consola)
│   ├── __init__.py
│   ├── lab1_2_teoria.py               # Series de Taylor, Fibonacci, Aritmética k-dígitos y Errores
│   ├── lab3_raices.py                 # Descartes, Bolzano, Bisección, Regula Falsi, Newton-Raphson, Aitken
│   ├── lab4_sistemas.py               # Gauss, Gauss-Jordan, LU Doolittle, Gauss-Seidel, Potencias y Faddeev
│   ├── lab5_interpolacion.py          # Lagrange, Newton-Gregory Asc/Desc, Cuadrática Inversa
│   ├── lab6_integracion.py            # Trapecio, Simpson 1/3, 3/8, Híbrido y Diferenciación Numérica
│   └── lab7_edo.py                    # Euler, Heun, Runge-Kutta (RK2/RK4) y Milne P-C
│
├── views/                             # INTERFAZ GRÁFICA SEMÁNTICA (Streamlit + Plotly + LaTeX)
│   ├── __init__.py
│   ├── ui_helpers.py                  # Visor de PDFs en base64 y bloque estandarizado de material de estudio
│   ├── inicio.py                      # Portal principal y catálogo interactivo del Santo Grial
│   ├── fundamentos.py                 # 🧮 Fundamentos, Aritmética de Máquina y Errores
│   ├── raices.py                      # 📉 Raíces de Ecuaciones No Lineales
│   ├── algebra_lineal.py              # 📐 Álgebra Lineal Numérica y Espectro Matricial
│   ├── interpolacion.py               # 📈 Interpolación y Aproximación Polinomial
│   ├── calculo_numerico.py            # ∬ Integración y Diferenciación Numérica
│   ├── edo.py                         # ⏱️ Ecuaciones Diferenciales Ordinarias (EDO - PVI)
│   └── historico_labs.py              # 📚 Histórico de Cursada (Laboratorios 1 al 7)
│
├── requirements.txt                   # Dependencias estrictas para Streamlit Cloud
├── ejecutar.bat                       # Script de 1 clic para Windows (python -m streamlit run app.py)
├── app.py                             # Entrypoint de Streamlit (Catálogo con st.navigation semántico)
└── README.md                          # Documentación formal del PFC
```

---

## 📐 Template Estricto de cada Método Numérico

Cada método numérico en la plataforma sigue rigurosamente el layout trifásico:

1. **Encabezado y Definición Formal**: Título y formalismo matemático con fórmulas en LaTeX (`st.latex`).
2. **Material de Estudio y Notas de Cátedra (PDFs)**:
   - *Pestaña A:* Fundamento Teórico y Deducción (PDF embebido nativamente desde `assets/teoria/`).
   - *Pestaña B:* Ejercicio Resuelto en Clase (PDF de la resolución paso a paso).
3. **Calculadora Interactiva (El Workspace)**:
   - Controles interactivos para carga de datos, matrices y expresiones simbólicas (`sympy`).
   - Ejecución mediante las funciones puras de `core/`.
   - Resultados en tablas detalladas (`st.dataframe`) y visualizaciones interactivas (`plotly`).

---

## 🚀 Despliegue y Ejecución

### 1. Ejecución Local con 1 Clic (Windows)
Hacer doble clic en **`ejecutar.bat`**.

### 2. Ejecución Manual por Terminal
```bash
# 1. Crear entorno virtual (opcional)
python -m venv .venv
.venv\Scripts\activate   # En Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar la aplicación
streamlit run app.py
```

### 3. Despliegue en Streamlit Community Cloud
- **Repository**: Seleccionar este repositorio en Streamlit Cloud.
- **Main file path**: `app.py`
- **Python version**: 3.10+
- **Requirements file**: `requirements.txt`

---

## ⚖️ Licencia y Créditos
Desarrollado para la cátedra de Métodos Computacionales — Facultad de Ciencias Exactas y Naturales y Agrimensura (FaCENA - UNNE).
