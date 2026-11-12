# 🎓 Suite Interactiva de Métodos Computacionales (FaCENA - UNNE)
## Proyecto Final de Carrera (PFC) — Jeremías Ezequiel González

Suite educativa e interactiva para la cátedra de **Métodos Computacionales** de la **Facultad de Ciencias Exactas y Naturales y Agrimensura (FaCENA - UNNE)**, construida con una arquitectura modular desacoplada y cloud-ready para **Streamlit Community Cloud**.

---

## 🏛️ Arquitectura del Sistema

El proyecto sigue una separación estricta entre la **capa de cálculo matemático (funciones puras)** y la **capa de presentación visual (Streamlit)**:

```text
/
├── requirements.txt                   # Dependencias estrictas para Streamlit Cloud
├── ejecutar.bat                       # Script de 1 clic para Windows (python -m streamlit run app.py)
├── app.py                             # Entrypoint de Streamlit (Catálogo con st.navigation)
├── README.md                          # Documentación formal del PFC
│
├── core/                              # MÓDULO MATEMÁTICO PURO (Sin dependencias de consola)
│   ├── __init__.py
│   ├── lab1_2_teoria.py               # Series, Fibonacci, Errores y Aritmética de Máquina
│   ├── lab3_raices.py                 # Newton-Raphson, Bisección, Regula Falsi, Punto Fijo
│   ├── lab4_sistemas.py               # Gauss, LU, Gauss-Seidel, Potencias, Faddeev-Leverrier
│   ├── lab5_interpolacion.py          # Lagrange, Newton-Gregory Asc/Desc, Cuadrática Inversa
│   ├── lab6_integracion.py            # Trapecio, Simpson 1/3, 3/8 y Diferenciación Numérica
│   └── lab7_edo.py                    # Euler, Heun, Runge-Kutta (RK2/RK4) y Milne P-C
│
└── views/                             # INTERFAZ GRÁFICA INTERACTIVA (Streamlit + Plotly)
    ├── __init__.py
    ├── inicio.py                      # Menú principal visual con tarjetas de navegación
    ├── view_lab1_2.py                 # Vistas interactivas de Laboratorios 1 y 2
    ├── view_lab3.py                   # Vistas interactivas de Búsqueda de Raíces
    ├── view_lab4.py                   # Vistas interactivas de Sistemas y Autovalores
    ├── view_lab5.py                   # Vistas interactivas de Interpolación
    ├── view_lab6.py                   # Vistas interactivas de Integración y Derivación
    └── view_lab7.py                   # Vistas interactivas de EDO - PVI
```

---

## 🚀 Despliegue y Ejecución

### 1. Ejecución Local con 1 Clic (Windows)
Hacer doble clic en `ejecutar.bat`. El script detectará automáticamente el entorno virtual o la instalación de Python, instalará las dependencias si faltan y lanzará el navegador.

### 2. Ejecución Manual por Terminal
```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar la aplicación
streamlit run app.py
```

### 3. Despliegue en Streamlit Community Cloud
El repositorio está preparado para despliegue directo sin configuración adicional:
- **Repository**: Seleccionar este repositorio en Streamlit Cloud.
- **Main file path**: `app.py`
- **Python version**: 3.10+
- **Requirements file**: `requirements.txt`

---

## 📚 Módulos y Laboratorios

| Módulo | Nombre | Temas Clave |
| :--- | :--- | :--- |
| **Lab 1 & 2** | Teoría, Algoritmos y Errores | Series de Taylor ($e^x$, $\pi$), Fibonacci, Razón Áurea, Errores Absoluto/Relativo, Punto Flotante, Cancelación Catastrófica, Esquema de Horner. |
| **Lab 3** | Búsqueda de Raíces | Regla de Descartes, Tanteos (Bolzano), Bisección, Regula Falsi, Newton-Raphson (Fourier), Punto Fijo y Aitken. |
| **Lab 4** | Sistemas Lineales y Autovalores | Eliminación Gaussiana, Gauss-Jordan, Factorización LU, Gauss-Seidel, Método de las Potencias, Faddeev-Leverrier. |
| **Lab 5** | Interpolación Polinomial | Interpolación Lineal, Cuadrática Inversa, Polinomio de Lagrange, Newton-Gregory Ascendente y Descendente. |
| **Lab 6** | Integración y Derivación | Derivadas por diferencias finitas de Gregory, Regla del Trapecio (simple/compuesta), Reglas de Simpson 1/3, 3/8 y combinada. |
| **Lab 7** | Ecuaciones Diferenciales (PVI) | Euler explícito, Euler modificado (Heun), Runge-Kutta 2° orden y 4° orden (RK4), Método predictor-corrector de Milne. |

---

## ⚖️ Licencia y Créditos
Desarrollado para la cátedra de Métodos Computacionales — FaCENA - UNNE.
