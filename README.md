# 🏛️ Métodos Computacionales — El Santo Grial de los Métodos Numéricos
## Facultad de Ciencias Exactas y Naturales y Agrimensura (FaCENA - UNNE)
### Proyecto Final de Carrera (PFC) • Jeremías Ezequiel González

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Release](https://img.shields.io/badge/Versi%C3%B3n-v2.0.0-success)](#)
[![Cloud-Ready](https://img.shields.io/badge/Streamlit%20Cloud-Ready-brightgreen)](https://share.streamlit.io/)
[![Licencia](https://img.shields.io/badge/Instituci%C3%B3n-FaCENA%20--%20UNNE-blue)](#)

Plataforma interactiva, educativa y de referencia matemática diseñada y construida para la cátedra de **Métodos Computacionales** de la **Facultad de Ciencias Exactas y Naturales y Agrimensura (FaCENA - Universidad Nacional del Nordeste, UNNE)**, Corrientes, Argentina.

El sistema unifica el rigor teórico formal, la deducción paso a paso de los algoritmos clásicos y modernos, la visualización geométrica interactiva con Plotly y la integración nativa de material de cátedra en formato PDF. Desarrollado con una arquitectura desacoplada en capas y 100% **Cloud-Ready** para su despliegue continuo en el tier gratuito de **Streamlit Community Cloud**.

---

## 🧭 Filosofía y Arquitectura del Sistema (v2.0.0)

A diferencia de una simple recopilación de trabajos prácticos de cursada, la versión **v2.0.0** adopta un **enfoque centrado en Métodos Matemáticos**, estructurando la plataforma como un compendio enciclopédico e interactivo de métodos numéricos, manteniendo la correlación histórica con los laboratorios curriculares en un anexo especializado.

```text
/
├── assets/
│   └── teoria/                        # Documentos PDF de cátedra (teoría y ejercicios resueltos)
│
├── core/                              # CAPA DE CÁLCULO NUMÉRICO PURO (Funciones puras, 0 UI/consola)
│   ├── __init__.py
│   ├── lab1_2_teoria.py               # Series de Taylor, Fibonacci, Aritmética k-dígitos, Errores y Horner
│   ├── lab3_raices.py                 # Descartes, Bolzano, Bisección, Regula Falsi, Newton-Raphson y Aitken
│   ├── lab4_sistemas.py               # Gauss, Gauss-Jordan, Factorización LU, Gauss-Seidel, Potencias y Faddeev
│   ├── lab5_interpolacion.py          # Lagrange, Newton-Gregory Progresivo/Regresivo, Cuadrática Inversa
│   ├── lab6_integracion.py            # Diferenciación Finita, Trapecio Compuesto, Reglas de Simpson y Cuadratura
│   └── lab7_edo.py                    # Euler, Heun, Runge-Kutta 2 y 4, Predictor-Corrector de Milne
│
├── views/                             # CAPA DE PRESENTACIÓN VISUAL (Streamlit + Plotly + LaTeX)
│   ├── __init__.py
│   ├── ui_helpers.py                  # Visor de PDFs en base64 y bloque estandarizado de material de estudio
│   ├── inicio.py                      # 🏠 Catálogo principal del Santo Grial con accesos directos
│   ├── fundamentos.py                 # 🧮 Fundamentos, Aritmética de Máquina y Teoría de Errores
│   ├── raices.py                      # 📉 Raíces de Ecuaciones No Lineales
│   ├── algebra_lineal.py              # 📐 Álgebra Lineal Numérica y Espectro Matricial
│   ├── interpolacion.py               # 📈 Interpolación y Aproximación Polinomial
│   ├── calculo_numerico.py            # 📊 Integración y Diferenciación Numérica
│   ├── edo.py                         # ⏱️ Ecuaciones Diferenciales Ordinarias (EDO - PVI)
│   └── historico_labs.py              # 📚 Histórico de Cursada (Mapeo de Laboratorios 1 al 7)
│
├── requirements.txt                   # Dependencias científicas congeladas para Streamlit Cloud
├── ejecutar.bat                       # Launcher automatizado de 1 clic para Windows (detecta Python / venv)
├── app.py                             # Entrypoint principal con ruteo nativo st.navigation y st.Page
└── README.md                          # Memoria técnica completa del proyecto
```

### Principios de Diseño
1. **Desacoplamiento Estricto:** Toda la matemática reside en funciones puras dentro de `core/`. No contienen llamadas a `print()`, `input()`, ni dependencias de Streamlit. Devuelven estructuras de datos limpias (diccionarios, tuplas, listas de pasos, dataframes) listas para ser consumidas por cualquier interfaz (CLI, Web, API).
2. **Robustez Numérica:** Manejo de singularidades, matrices mal condicionadas, divisiones por cero, indeterminaciones `NaN`/`Inf` en cancelaciones catastróficas y criterios de parada rigurosos (tolerancia absoluta, relativa y cotas de iteraciones).
3. **Soporte Simbólico y Arbitrario:** Procesamiento de funciones matemáticas arbitrarias mediante `sympy`, garantizando cálculo analítico exacto de derivadas e integrales para contraste con las soluciones numéricas.
4. **Cloud-Ready:** Compatible al 100% con los límites de memoria, CPU y stateless runtime de Streamlit Community Cloud.

---

## 📐 Template Estricto de 3 Fases por Método Numérico

Cada método dentro de las vistas (`views/`) implementa un diseño vertical estandarizado de tres etapas:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Encabezado y Fundamentación Matemática Formal                       │
│    - Definición rigurosa, deducción de fórmulas y cotas de error en    │
│      formato LaTeX mediante st.latex().                                │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Material de Estudio Embebido (PDFs de Cátedra)                      │
│    - Panel desplegable st.expander con dos pestañas embebidas en       │
│      base64 nativo:                                                    │
│        • Pestaña A: 📖 Fundamento Teórico y Deducción (PDF).           │
│        • Pestaña B: 📝 Ejercicio Resuelto en Clase (PDF).              │
│    - Fallback inteligente: si el archivo no está en assets/teoria/,    │
│      muestra una tarjeta informativa con el nombre exacto esperado.    │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Calculadora Interactiva y Espacio de Trabajo (Workspace)            │
│    - Formularios de entrada dinámica (matrices, inputs, funciones).    │
│    - Botón de cómputo que invoca la función pura de core/.             │
│    - Tabla de iteraciones paso a paso con st.dataframe().              │
│    - Gráfico interactivo con plotly (zoom, hover, tangentes, áreas).   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Índice Exhaustivo de Módulos y Métodos

### 1. 🧮 Fundamentos, Aritmética de Máquina y Errores (`views/fundamentos.py`)
- **Series de Taylor:** Aproximación de funciones continuas, cálculo de la constante $e$ y de $\pi$ (Serie de Gregory-Leibniz) aplicando el **Criterio de Parada de Scarborough** ($\varepsilon_s = 0.5 \times 10^{2-n}\%$).
- **Aritmética de Punto Flotante de $k$ Dígitos:** Simulación de una computadora decimal de precisión finita en formato normalizado $\pm 0.d_1 d_2 \dots d_k \times 10^e$, contrastando **Redondeo Simétrico vs. Truncamiento**.
- **Pérdida de Asociatividad:** Demostración computacional de que $(A - B) - C \neq (A - C) - B$ en aritmética finita mediante casos testigo de cátedra.
- **Cancelación Catastrófica en Cuadráticas:** Diagnóstico de pérdida de dígitos significativos al restar raíces de magnitudes casi idénticas en la resolvente clásica ($x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$) y solución mediante la **fórmula racionalizada** (Viète/Muller).
- **Forma Anidada de Horner:** Reducción del costo computacional de evaluación polinómica de $O(n^2)$ a $O(n)$ sumas y multiplicaciones, mitigando el error de redondeo acumulado.
- **Sucesión de Fibonacci y Razón Áurea:** Cómputo de términos arbitrarios ($N$ hasta 200 con precisión infinita mediante enteros de Python) y análisis de convergencia del cociente $\frac{F_{n}}{F_{n-1}}$ hacia $\varphi = \frac{1 + \sqrt{5}}{2}$.

### 2. 📉 Raíces de Ecuaciones No Lineales (`views/raices.py`)
- **Aislamiento de Raíces:**
  - **Regla de los Signos de Descartes:** Determinación de las cotas máximas de raíces reales positivas y negativas según la alternancia de signos en los coeficientes.
  - **Teorema de Bolzano:** Barrido sistemático por intervalos para detectar cambios de signo $f(a) \cdot f(b) < 0$.
- **Método de Bisección (Intervalo Medio):** Algoritmo cerrado de convergencia garantizada, con cálculo automático de la **cota teórica de iteraciones mínimas**:
  $$n \ge \frac{\ln(b - a) - \ln(\varepsilon)}{\ln 2}$$
- **Interpolación Lineal (*Regula Falsi*):**
  - **Variante Cátedra UNNE:** Mantención del pivote fijo inicial $x_0$ para evaluar convergencia monotónica unilateral.
  - **Variante Clásica de Bolzano:** Actualización dinámica del subintervalo que encierra el cambio de signo.
- **Método de Newton-Raphson:** Esquema abierto de convergencia cuadrática. Verificación previa de las **Condiciones de Fourier** ($f(x_0) \cdot f''(x_0) > 0$) para asegurar convergencia estricta sin oscilaciones ni pendientes nulas.
- **Método de Punto Fijo y Aceleración de Aitken:** Transformación $x = g(x)$ y aceleración de sucesiones lineales mediante el operador de diferencias $\Delta^2$ de Aitken:
  $$x_{\text{acelerado}} = x_k - \frac{(x_{k+1} - x_k)^2}{x_{k+2} - 2x_{k+1} + x_k}$$
- **Comparativa Multimétodo:** Ejecución en paralelo sobre la misma función para contrastar velocidad de convergencia, número de iteraciones y tiempos de CPU.

### 3. 📐 Álgebra Lineal Numérica y Espectro Matricial (`views/algebra_lineal.py`)
- **Eliminación Gaussiana:** Triangularización superior con **pivoteo parcial por columnas** para mitigar la amplificación de errores por división de elementos diagonales pequeños, registro de multiplicadores $m_{ik}$ y sustitución regresiva.
- **Método de Gauss-Jordan:** Reducción completa a la matriz identidad $[A|b] \to [I|x]$ y cálculo simultáneo de la **Matriz Inversa** $[A|I] \to [I|A^{-1}]$ con verificación de norma de residuo $\|A \cdot A^{-1} - I\|$.
- **Factorización $LU$ (Doolittle):** Descomposición $A = L \cdot U$ con diagonal unitaria en $L$, resolución encadenada de dos sistemas triangulares: $L y = b$ (sustitución progresiva) y $U x = y$ (sustitución regresiva).
- **Método Iterativo de Gauss-Seidel:** Verificación previa de **Dominancia Diagonal Estricta** como condición suficiente de convergencia, cálculo de errores relativos porcentuales en norma infinita y gráfico de convergencia logarítmica.
- **Método de las Potencias:**
  - **Potencias Directas:** Estimación del autovalor de mayor módulo $|\lambda_{\max}|$ y su autovector asociado mediante normalización iterativa.
  - **Potencias Inversas:** Cálculo del autovalor de menor módulo $|\lambda_{\min}|$ aplicando el método sobre la inversa $A^{-1}$.
- **Método de Faddeev-Leverrier:** Algoritmo de trazas recursivas que obtiene en un solo procedimiento:
  - Los coeficientes del **Polinomio Característico** $P(\lambda) = \det(\lambda I - A)$.
  - El determinante exacto $\det(A)$.
  - La **matriz inversa** $A^{-1}$ analítica.
  - Todas las raíces polinómicas (autovalores reales y complejos) y sus correspondientes autovectores.

### 4. 📈 Interpolación y Aproximación Polinomial (`views/interpolacion.py`)
- **Interpolación Lineal Directa e Inversa:** Estimación de valores intermedios $y(x)$ y cálculo inverso de la abscisa $x(y)$ asociada a un valor funcional dado.
- **Interpolación Cuadrática Inversa:** Ajuste sobre 3 nodos continuos y resolución de la ecuación cuadrática $a u^2 + b u + c = 0$ ($u = x - x_0$) con selección física de la raíz admisible dentro del soporte tabular.
- **Polinomio Interpolador de Lagrange:** Construcción explícita de los polinomios base:
  $$L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$
  y generación simbólica simplificada del polinomio global $P_n(x)$ mediante SymPy.
- **Newton-Gregory Ascendente (Diferencias Progresivas):** Construcción de la tabla completa de diferencias $\Delta^k y_0$, cálculo del parámetro $u = \frac{x - x_0}{h}$ e interpolación óptima para puntos situados en el tercio inicial de la tabla.
- **Newton-Gregory Descendente (Diferencias Retrógradas):** Construcción de la tabla de diferencias $\nabla^k y_n$, cálculo del parámetro $v = \frac{x - x_n}{h}$ e interpolación óptima para puntos cercanos al final de la tabla.

### 5. 📊 Integración y Diferenciación Numérica (`views/calculo_numerico.py`)
- **Diferenciación Numérica:** Fórmulas de diferencias finitas de Gregory hacia adelante (inicio del intervalo) y hacia atrás (final del intervalo) para la primera y segunda derivada ($f'(x)$ y $f''(x)$).
- **Regla del Trapecio Compuesto:** Cuadratura numérica con desglose formal de cátedra:
  $$I_T = \frac{h}{2} [E + 2P + 2I]$$
  donde $E$ es la suma de extremos, $P$ la suma de pares e $I$ la suma de impares, con análisis del error de truncamiento $O(h^2)$.
- **Reglas de Simpson:**
  - **Simpson 1/3 Compuesto:** Para particiones con número par de subintervalos:
    $$I_{S1/3} = \frac{h}{3} [E + 2P + 4I] \quad \text{con cota de error } O(h^4)$$
  - **Simpson 3/8:** Para subintervalos múltiplos de 3.
  - **Simpson Híbrido (Combinado):** Algoritmo inteligente para particiones con número impar de intervalos: aplica Simpson 3/8 en los primeros 3 paneles y Simpson 1/3 en los restantes $n-3$, garantizando precisión de orden $O(h^4)$.
- **Cuadratura Simbólica vs. Numérica:** Integración simbólica analítica mediante SymPy contra Trapecio y Simpson, con cálculo de discrepancia absoluta y porcentual.

### 6. ⏱️ Ecuaciones Diferenciales Ordinarias (EDO) (`views/edo.py`)
- **Métodos de un Solo Paso:**
  - **Euler Explícito:** Esquema tangencial de primer orden: $y_{k+1} = y_k + h f(x_k, y_k)$.
  - **Euler Modificado (Heun):** Predictor-corrector de segundo orden $O(h^2)$.
- **Familia de Runge-Kutta:**
  - **RK2:** Esquemas de orden 2 (Heun y punto medio).
  - **RK4 Clásico:** Algoritmo de cuarto orden $O(h^4)$ con desglose interactivo de las cuatro pendientes evaluadas en cada paso ($k_1, k_2, k_3, k_4$):
    $$y_{k+1} = y_k + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
- **Método Multipaso de Milne (Predictor-Corrector):** Algoritmo de 4° orden con procedimiento de arranque automático mediante RK4 para generar los primeros 3 nodos y aplicación iterativa del par:
  - Predictor abierto de Milne: $y_{k+1}^{(0)} = y_{k-3} + \frac{4h}{3}[2f_k - f_{k-1} + 2f_{k-2}]$
  - Corrector cerrado de Simpson: $y_{k+1}^{(j)} = y_{k-1} + \frac{h}{3}[f(x_{k+1}, y_{k+1}^{(j-1)}) + 4f_k + f_{k-1}]$
- **Comparativa Multimétodo y Solución Analítica:** Trazado superpuesto de trayectorias numéricas frente a la curva analítica exacta resuelta simbólicamente por SymPy.

### 7. 📚 Histórico de Cursada (`views/historico_labs.py`)
- Mapeo didáctico que vincula los 7 Trabajos Prácticos originales de la cátedra de FaCENA con los nuevos módulos de la plataforma, permitiendo a docentes y alumnos consultar los enunciados oficiales, notas de laboratorio y acceder con un solo clic a las herramientas correspondientes.

---

## 📄 Guía de Incorporación de Documentos PDF (`assets/teoria/`)

La plataforma cuenta con visores embebidos responsivos en Base64. Para activar las lecturas oficiales y guías prácticas resueltas, simplemente copie los documentos correspondientes en la carpeta [`assets/teoria/`](file:///c:/UNNE/PFC/PFC-Jeremias-Ezequiel-Gonzalez/assets/teoria/):

| Archivo PDF Sugerido | Módulo / Método | Contenido de Cátedra Recomendado |
| :--- | :--- | :--- |
| `01_teoria_series_taylor_scarborough.pdf` | Fundamentos | Deducción de series de Taylor y criterio de Scarborough |
| `01_practica_series_taylor_ejercicio.pdf` | Fundamentos | Ejercicios prácticos de aproximación de $e$ y $\pi$ |
| `02_teoria_aritmetica_maquina_redondeo.pdf` | Fundamentos | Aritmética finita, redondeo simétrico y truncamiento |
| `03_practica_asociatividad_errores.pdf` | Fundamentos | Ejemplos de pérdida de asociatividad en punto flotante |
| `04_teoria_cancelacion_catastrofica.pdf` | Fundamentos | Cancelación sustractiva y fórmula de Muller/Viète |
| `05_teoria_esquema_horner.pdf` | Fundamentos | Demostración formal del algoritmo anidado de Horner |
| `06_teoria_descartes_bolzano_aislamiento.pdf` | Raíces | Regla de Descartes y Teorema de Bolzano |
| `07_teoria_metodo_biseccion.pdf` | Raíces | Demostración de convergencia y cota teórica de Bisección |
| `08_teoria_regula_falsi.pdf` | Raíces | Interpolación lineal y variantes de pivote fijo vs. Bolzano |
| `09_teoria_newton_raphson_fourier.pdf` | Raíces | Deducción de tangentes y condiciones de Fourier |
| `10_teoria_punto_fijo_aitken.pdf` | Raíces | Convergencia de $x=g(x)$ y acelerador delta cuadrado |
| `12_teoria_eliminacion_gauss_pivoteo.pdf` | Álgebra Lineal | Eliminación Gaussiana con pivoteo parcial |
| `13_teoria_gauss_jordan_inversa.pdf` | Álgebra Lineal | Reducción de Jordan y cálculo de matriz inversa |
| `14_teoria_factorizacion_lu_doolittle.pdf` | Álgebra Lineal | Factorización $A = LU$ según Doolittle |
| `15_teoria_gauss_seidel_dominancia.pdf` | Álgebra Lineal | Métodos iterativos y dominancia diagonal estricta |
| `16_teoria_metodo_potencias.pdf` | Álgebra Lineal | Algoritmo de potencias directas e inversas |
| `17_teoria_faddeev_leverrier.pdf` | Álgebra Lineal | Algoritmo de Leverrier-Faddeev para espectro y trazas |
| `18_teoria_interpolacion_lineal.pdf` | Interpolación | Interpolación lineal directa e inversa |
| `19_teoria_cuadratica_inversa.pdf` | Interpolación | Ajuste parabólico y resolución inversa |
| `20_teoria_polinomio_lagrange_bases.pdf` | Interpolación | Construcción de polinomios base de Lagrange |
| `21_teoria_newton_gregory_ascendente.pdf` | Interpolación | Diferencias finitas progresivas $\Delta$ |
| `22_teoria_newton_gregory_descendente.pdf` | Interpolación | Diferencias finitas retrógradas $\nabla$ |
| `23_teoria_diferenciacion_numerica.pdf` | Cálculo Numérico | Fórmulas de derivadas primera y segunda por diferencias |
| `24_teoria_regla_trapecio_compuesto.pdf` | Cálculo Numérico | Cuadratura de Trapecio ($E, 2P, 2I$) y cota de error |
| `25_teoria_reglas_simpson.pdf` | Cálculo Numérico | Deducción de Simpson 1/3, 3/8 y esquema combinado |
| `26_teoria_edo_metodo_euler.pdf` | EDO | Métodos de Euler clásico y modificado (Heun) |
| `28_teoria_runge_kutta_orden2_orden4.pdf` | EDO | Deducción y análisis del método clásico de Runge-Kutta 4 |
| `29_teoria_milne_predictor_corrector.pdf` | EDO | Métodos multipaso y predictor-corrector de Milne |
| `lab1_guia_teoria_modelos_series.pdf` | Histórico | Guía original del Laboratorio 1 (FaCENA - UNNE) |

> [!NOTE]
> La presencia de los PDFs es opcional: si un archivo aún no se encuentra en la carpeta, la plataforma desplegará un placeholder informativo con indicaciones claras sin interrumpir ninguna de las funciones interactivas ni el cómputo numérico.

---

## 🚀 Despliegue y Ejecución

### 1. Ejecución Rápida con 1 Clic (Windows)
Haga doble clic en el archivo **[`ejecutar.bat`](file:///c:/UNNE/PFC/PFC-Jeremias-Ezequiel-Gonzalez/ejecutar.bat)**.
El script detecta automáticamente su entorno virtual (`.venv` o `venv`), verifica las librerías necesarias y lanza el servidor de Streamlit en su navegador predeterminado.

### 2. Ejecución Manual por Consola
```bash
# 1. Clonar el repositorio
git clone https://github.com/JereEG/PFC-Jeremias-Ezequiel-Gonzalez.git
cd PFC-Jeremias-Ezequiel-Gonzalez

# 2. Crear y activar entorno virtual (recomendado)
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux / macOS:
source .venv/bin/activate

# 3. Instalar dependencias científicas
pip install -r requirements.txt

# 4. Iniciar la aplicación
streamlit run app.py
```

### 3. Despliegue Continuo en Streamlit Community Cloud
El proyecto está optimizado para su despliegue gratuito en la nube:
1. Conecte su cuenta de GitHub en [share.streamlit.io](https://share.streamlit.io/).
2. Seleccione el repositorio: `JereEG/PFC-Jeremias-Ezequiel-Gonzalez`.
3. Configure los datos de inicio:
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Python version:** `3.10` o superior
4. Presione **Deploy**. La plataforma se desplegará de forma automática con HTTPS y soporte continuo.

---

## 🛠️ Stack Tecnológico

| Herramienta | Rol en el Proyecto |
| :--- | :--- |
| **Python 3.10+** | Lenguaje núcleo del proyecto |
| **Streamlit 1.36+** | Framework web reactivo (`st.navigation`, `st.Page`, `st.data_editor`, `st.expander`) |
| **SymPy** | Motor de álgebra computacional (cálculo simbólico, derivadas, integrales y parsing de fórmulas) |
| **NumPy** | Álgebra matricial numérica, vectores, normas euclídeas y cálculo espectral |
| **Pandas** | Generación estructurada de tablas de iteraciones y operadores de diferencias finitas |
| **Plotly Graph Objects** | Gráficos científicos interactivos con zoom dinámico, tooltips y renderizado WebGL |

---

## 📜 Historial de Versiones

- **`v1.0.0` (Extracción y Suite Base):**
  - Desacoplamiento completo del código legado de consola de los Laboratorios 1 al 7 de FaCENA - UNNE.
  - Creación de módulos de cálculo puro en `core/` con funciones libres de `print()` o `input()`.
  - Primera interfaz interactiva en Streamlit por laboratorios.
- **`v2.0.0` (El Santo Grial de los Métodos Numéricos):**
  - Pivot arquitectónico total de UX/UI centrado en **Métodos Matemáticos**.
  - Ruteo moderno con `st.navigation` y `st.Page`.
  - Estandarización del layout de 3 fases (Teoría $\LaTeX$, PDFs embebidos en Base64 y Workspace interactivo).
  - Integración de resolvente racionalizada para cancelación catastrófica, dominancia diagonal en Gauss-Seidel, Simpson combinado híbrido y soporte integral de fórmulas simbólicas con SymPy.
  - Creación de anexo curricular "Histórico de Cursada".

---

## ⚖️ Autor y Créditos Académicos

- **Autor:** Jeremías Ezequiel González
- **Carrera:** Licenciatura en Sistemas de Información / Proyecto Final de Carrera (PFC)
- **Cátedra:** Métodos Computacionales
- **Institución:** Facultad de Ciencias Exactas y Naturales y Agrimensura (FaCENA) — Universidad Nacional del Nordeste (UNNE), Corrientes, Argentina.
