"""
core/catalogo_ejercicios.py
===========================
Catálogo estructurado de conocimiento matemático, ejemplos teóricos oficiales
y ejercicios prácticos de cátedra para "Arandú: Métodos Computacionales (v3.0)".

Contiene:
- CATALOGO_METODOS: Información estilo Refactoring.Guru (qué resuelve, intuición geométrica,
  cuándo usar, cuándo falla), fórmulas oficiales LaTeX, Ejemplo Teórico Oficial y
  Práctica de la Profesora con datos iniciales exactos y solución de referencia.
- FICHAS_VERSUS: Fichas técnicas comparativas por familia matemática para Arena Versus,
  incluyendo expectativas teóricas y notas pedagógicas sobre excepciones y casos patológicos.

100% autónomo: no depende de archivos externos ni de `_temp_legacy/`.
"""

from __future__ import annotations

from typing import Any


CATALOGO_METODOS: dict[str, dict[str, Any]] = {
    # ==========================================================================
    # 1. FUNDAMENTOS Y ERRORES
    # ==========================================================================
    "series_taylor": {
        "nombre": "Series de Taylor y Maclaurin (Criterio de Scarborough)",
        "familia": "fundamentos",
        "icono": "🧮",
        "resumen_corto": "Aproxima funciones trascendentes mediante sumas infinitas de polinomios controladas por tolerancia.",
        "guru": {
            "que_resuelve": "Permite calcular funciones como e^x, sin(x), ln(1+x) o π usando únicamente sumas, potencias y factoriales.",
            "intuicion_geometrica": "Cada nuevo término agrega una curvatura polinómica (recta tangente, parábola osculadora, cúbica...) que se abraza más fielmente a la curva original alrededor de un punto.",
            "cuando_usar": [
                "Cuando necesitas evaluar funciones analíticas con precisión configurable mediante N términos o cifras significativas.",
                "Para estimar el error de truncamiento de algoritmos numéricos.",
            ],
            "cuando_falla": [
                "Lejos del radio de convergencia (por ejemplo, ln(1+x) fuera de (-1, 1]).",
                "En series de convergencia extremadamente lenta como Gregory-Leibniz para π (requiere ~200.000 términos para 5 decimales).",
            ],
        },
        "formulas_latex": [
            r"e^x = \sum_{k=0}^{n-1} \frac{x^k}{k!}, \qquad \pi = 4\sum_{k=0}^{\infty} \frac{(-1)^k}{2k+1}",
            r"E_{rp} = \left|\frac{\text{aprox. actual} - \text{aprox. anterior}}{\text{aprox. actual}}\right| \times 100 < E_s = (0.5 \times 10^{2-n})\%",
        ],
        "ejemplo_teorico": {
            "titulo": "Aproximación de e^(0.5) asegurando 3 cifras significativas (Scarborough)",
            "enunciado": "Estimar e^(0.5) por serie de Maclaurin garantizando al menos n = 3 cifras significativas exactas.",
            "pasos": [
                "Cota de Scarborough para n = 3: E_s = 0.5 × 10^(2-3) % = 0.05%.",
                "T_1 = 1",
                "T_2 = 1 + 0.5 = 1.5 (E_rp = 33.33%)",
                "T_3 = 1.5 + 0.5²/2! = 1.625 (E_rp = 7.69%)",
                "T_4 = 1.625 + 0.5³/3! = 1.645833... (E_rp = 1.26%)",
                "T_6 ≈ 1.6486979 cumple |E_rp| < 0.05%.",
            ],
            "resultado_final": "e^(0.5) ≈ 1.6487 con 3 cifras significativas garantizadas.",
        },
        "practica_profesora": {
            "titulo": "Lab 1 (Ej. 5 y 6): Cálculo de e y π con criterio de error",
            "enunciado": "Calcular la serie de e^x (por defecto x = 1, N = 8 términos) y la serie de Leibniz para π con tolerancia Ea < 0.5 × 10^(-n).",
            "datos_iniciales": {
                "x_euler": 1.0,
                "n_terminos_euler": 8,
                "cifras_scarborough": 3,
                "cifras_leibniz": 3,
            },
            "solucion_referencia": "Para N=8 en x=1: e ≈ 2.71827877 (Ea = 3.0586e-6). Para Leibniz n=3 (tol=0.0005): 2000 términos, π ≈ 3.1410927.",
        },
    },
    "aritmetica_maquina": {
        "nombre": "Aritmética de Punto Flotante (k dígitos) y Asociatividad",
        "familia": "fundamentos",
        "icono": "💾",
        "resumen_corto": "Simula la representación finita normalizada ±0.d1...dk × 10^n comparando redondeo simétrico y truncamiento.",
        "guru": {
            "que_resuelve": "Explica y cuantifica por qué la computadora no almacena números reales infinitos y cómo el orden de las operaciones altera el resultado.",
            "intuicion_geometrica": "La recta real continua se convierte en una regla con muescas discretas: todo cálculo cae entre dos muescas y se ajusta a la más cercana (redondeo) o a la inferior (truncamiento).",
            "cuando_usar": [
                "Para auditar errores de redondeo acumulados y elegir el mejor orden de suma (sumar primero los números pequeños).",
            ],
            "cuando_falla": [
                "Al operar números de órdenes de magnitud muy distintos, donde el pequeño se 'absorbe' y desaparece.",
            ],
        },
        "formulas_latex": [
            r"fl(y) = \pm 0.d_1 d_2 \dots d_k \times 10^n, \qquad \left|\frac{y - fl(y)}{y}\right| \le \begin{cases} 10^{-k+1} & \text{(truncamiento)} \\ 0.5 \times 10^{-k+1} & \text{(redondeo)} \end{cases}",
            r"(A \ominus B) \ominus C \neq (A \ominus C) \ominus B \quad \text{en aritmética finita}",
        ],
        "ejemplo_teorico": {
            "titulo": "Operaciones con x = 1/3 e y = 5/7 a k = 5 dígitos (Truncamiento)",
            "enunciado": "Representar fl(x) = 0.33333 × 10^0 y fl(y) = 0.71428 × 10^0 y calcular x+y y y-x.",
            "pasos": [
                "x + y = fl(0.33333 + 0.71428) = fl(1.04761) = 0.10476 × 10^1 (Er = 1.82e-5).",
                "y - x = fl(0.71428 - 0.33333) = fl(0.38095) = 0.38095 × 10^0 (Er = 6.25e-6).",
            ],
            "resultado_final": "El truncamiento introduce un error relativo acotado por 10^(-4), salvo cuando hay resta de términos cercanos.",
        },
        "practica_profesora": {
            "titulo": "TP2 Ej. 2 y 3: Pérdida de Asociatividad con k = 3 dígitos",
            "enunciado": "Comparar (121 - 119) - 0.327 frente a (121 - 0.327) - 119 usando aritmética normalizada de k = 3 dígitos (redondeo y truncamiento).",
            "datos_iniciales": {
                "k_digitos": 3,
                "u_val": 121.0,
                "v_val": 119.0,
                "w_val": 0.327,
            },
            "solucion_referencia": "Exacto = 1.673. Redondeo k=3: (121-119)-0.327 = 0.167×10¹ (Er 0.18%); (121-0.327)-119 = 0.200×10¹ (Er 19.55%!). Truncamiento: (121-0.327)-119 = 0.100×10¹ (Er 40.23%!).",
        },
    },
    "cancelacion_cuadratica": {
        "nombre": "Cancelación Catastrófica en Resolvente Cuadrática",
        "familia": "fundamentos",
        "icono": "💥",
        "resumen_corto": "Evita la pérdida severa de cifras significativas al restar dos números casi iguales mediante racionalización.",
        "guru": {
            "que_resuelve": "Corrige el error numérico grave de la fórmula cuadrática tradicional cuando b² >> 4ac.",
            "intuicion_geometrica": "Si el vértice está muy alejado de la raíz pequeña, √(b²-4ac) es casi igual a |b|. Restarlos destruye los primeros dígitos exactos y deja solo ruido de redondeo.",
            "cuando_usar": [
                "Siempre que b² sea mucho mayor que |4ac| al resolver ax² + bx + c = 0.",
            ],
            "cuando_falla": [
                "Aplicar la fórmula racionalizada a la raíz donde los signos se suman (en lugar de restarse) provoca el problema inverso.",
            ],
        },
        "formulas_latex": [
            r"x_{1,2} = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \quad \xrightarrow{\text{Racionalización}} \quad x_1 = \frac{-2c}{b + \sqrt{b^2 - 4ac}} \;\; (b > 0)",
        ],
        "ejemplo_teorico": {
            "titulo": "Caso testigo de Chapra (Ej. 3.7)",
            "enunciado": "Resolver x² + 3000.001x + 3 = 0 con precisión finita.",
            "pasos": [
                "Raíces exactas: x_1 = -0.001, x_2 = -3000.",
                "Con pocos dígitos en la fórmula tradicional, -b + √(b² - 4ac) cancela los dígitos líderes y distorsiona x_1.",
            ],
            "resultado_final": "Usar la forma racionalizada para x_1 y la estándar sumando en el numerador para x_2.",
        },
        "practica_profesora": {
            "titulo": "TP2 Ej. 4: x² + 62.10x + 1 = 0 con redondeo a k = 4 dígitos",
            "enunciado": "Comparar la resolvente estándar vs. racionalizada para x² + 62.10x + 1 = 0 usando k = 4 dígitos.",
            "datos_iniciales": {
                "a": 1.0,
                "b": 62.10,
                "c": 1.0,
                "k_digitos": 4,
            },
            "solucion_referencia": "Exactas: x1 = -0.01610723, x2 = -62.08390. Estándar k=4 da x1 = -0.02000 (Er = 24.17%); Racionalizada da x1 = -0.01610 (Er = 0.045%).",
        },
    },
    "horner": {
        "nombre": "Esquema Anidado de Horner",
        "familia": "fundamentos",
        "icono": "🪆",
        "resumen_corto": "Evalúa polinomios en O(n) multiplicaciones reduciendo drásticamente el error de máquina.",
        "guru": {
            "que_resuelve": "Reemplaza el cálculo costoso de potencias explícitas x^k por multiplicaciones sucesivas anidadas.",
            "intuicion_geometrica": "En lugar de elevar un número grande varias veces (amplificando el error en cada potencia), va factorizando x paso a paso desde el término de mayor grado.",
            "cuando_usar": [
                "Siempre que se evalúe cualquier polinomio de grado >= 2 en código o en calculadoras.",
            ],
            "cuando_falla": [
                "No cambia el mal condicionamiento propio de raíces múltiples cercanas (como el polinomio de Wilkinson).",
            ],
        },
        "formulas_latex": [
            r"P(x) = a_n x^n + \dots + a_1 x + a_0 = ((\dots(a_n x + a_{n-1})x + \dots)x + a_1)x + a_0",
        ],
        "ejemplo_teorico": {
            "titulo": "Conteo de operaciones en grado 3",
            "enunciado": "Comparar evaluación directa vs. anidada para un polinomio cúbico.",
            "pasos": [
                "Directa: requiere calcular x², x³ y multiplicar cada término (4 multiplicaciones + 3 sumas).",
                "Horner: ((a_3 x + a_2)x + a_1)x + a_0 (3 multiplicaciones, y solo 2 si a_3 = 1).",
            ],
            "resultado_final": "Menor cantidad de operaciones en punto flotante implica menor propagación de error de redondeo/truncamiento.",
        },
        "practica_profesora": {
            "titulo": "TP2 Ej. 5/6: f(x) = x³ - 6x² + 3x - 0.149 en x = 4.71 (k = 3 dígitos)",
            "enunciado": "Evaluar f(4.71) con aritmética de 3 dígitos (truncamiento y redondeo) en forma estándar y en forma anidada.",
            "datos_iniciales": {
                "coefs_str": "1, -6, 3, -0.149",
                "x_eval": 4.71,
                "k_digitos": 3,
            },
            "solucion_referencia": "Exacto: -14.636489. Estándar (k=3): -14.0 (Er 4.35%). Horner Redondeo (k=3): -14.6 (Er 0.25%). Horner Truncamiento: -14.5 (Er 0.93%).",
        },
    },

    # ==========================================================================
    # 2. RAÍCES DE ECUACIONES
    # ==========================================================================
    "biseccion": {
        "nombre": "Descartes, Tanteos y Bisección (Intervalo Medio)",
        "familia": "raices",
        "icono": "⚖️",
        "resumen_corto": "Divide iterativamente un intervalo con cambio de signo por la mitad hasta acorralar la raíz.",
        "guru": {
            "que_resuelve": "Encuentra raíces reales de ecuaciones no lineales f(x) = 0 dentro de un intervalo cerrado [a, b].",
            "intuicion_geometrica": "Si la función es continua y cruza el eje X (cambia de signo entre a y b), cortamos el intervalo al medio y nos quedamos con la mitad donde sigue habiendo cruce.",
            "cuando_usar": [
                "Cuando necesitas garantía absoluta de convergencia (siempre que f(a)·f(b) < 0).",
                "Como etapa inicial para acercarse a una raíz antes de usar Newton-Raphson.",
            ],
            "cuando_falla": [
                "Si la función es tangente al eje X (raíz múltiple de orden par, sin cambio de signo).",
                "Convergencia lenta (lineal): reduce el error apenas a la mitad por cada paso.",
            ],
        },
        "formulas_latex": [
            r"x_r = \frac{a + b}{2}, \qquad |b_n - a_n| = \frac{b - a}{2^n} \le E \implies n \ge \frac{\ln(b - a) - \ln E}{\ln 2}",
        ],
        "ejemplo_teorico": {
            "titulo": "Aproximación de √2 mediante f(x) = x² - 2 en [1, 3]",
            "enunciado": "Aproximar la raíz positiva de f(x) = x² - 2 en el intervalo [1, 3].",
            "pasos": [
                "f(1) = -1 < 0 y f(3) = 7 > 0 → existe raíz en [1, 3].",
                "Iter 1: x1 = (1+3)/2 = 2, f(2) = 2 > 0 → nuevo intervalo [1, 2].",
                "Iter 2: x2 = 1.5, f(1.5) = 0.25 > 0 → [1, 1.5].",
                "Iter 3: x3 = 1.25 → [1.25, 1.5] ... Iter 8: x8 = 1.4140625.",
            ],
            "resultado_final": "4 cifras exactas de √2 alcanzadas en 8 bisecciones.",
        },
        "practica_profesora": {
            "titulo": "TP3 Ej. 1: -2.75x³ + 18x² - 21x - 12 = 0 con E < 0.01",
            "enunciado": "Aplicar Regla de Descartes, separar raíces con tanteo Δx = 0.5 y aproximar la raíz en (2, 2.5) por Intervalo Medio con E < 0.01.",
            "datos_iniciales": {
                "f_str": "-2.75*x**3 + 18*x**2 - 21*x - 12",
                "a": 2.0,
                "b": 2.5,
                "tol": 0.01,
                "dx_tanteo": 0.5,
                "x_min_tanteo": -1.0,
                "x_max_tanteo": 5.0,
                "max_iter": 50,
            },
            "solucion_referencia": "Descartes: 2 o 0 positivas, 1 negativa. Cota teórica: n ≥ 5.64 → 6 iteraciones. Raíz: x6 = 2.2265625 (error = 0.0078125 < 0.01).",
        },
    },
    "interpolacion_lineal_raices": {
        "nombre": "Interpolación Lineal (Regula Falsi / Partes Proporcionales)",
        "familia": "raices",
        "icono": "📏",
        "resumen_corto": "Une los extremos del intervalo con una recta secante y usa su corte con el eje X como nueva aproximación.",
        "guru": {
            "que_resuelve": "Encuentra raíces reales aprovechando no solo el signo, sino las magnitudes de f(a) y f(b).",
            "intuicion_geometrica": "Trazamos una cuerda recta entre (a, f(a)) y (b, f(b)). Donde esa cuerda corta al eje X estará mucho más cerca de la raíz real que el punto medio geométrico.",
            "cuando_usar": [
                "Cuando la función es casi lineal y continua en el intervalo [a, b].",
            ],
            "cuando_falla": [
                "Si la curva es muy cóncava o convexa, uno de los extremos queda anclado (pivote fijo) y la convergencia puede volverse muy lenta de un solo lado.",
            ],
        },
        "formulas_latex": [
            r"x_{n+1} = \frac{x_{n-1} f(x_n) - x_n f(x_{n-1})}{f(x_n) - f(x_{n-1})} = x_n - f(x_n)\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}",
        ],
        "ejemplo_teorico": {
            "titulo": "Esquema con extremo fijo P1(x1, y1)",
            "enunciado": "Aproximar la raíz de una función monótona en [x1, x2] con f(x1)·f(x2) < 0 y concavidad constante.",
            "pasos": [
                "Se traza la secante entre (x1, y1) y (x2, y2), obteniendo la intersección x3.",
                "Si la concavidad mantiene signo, el extremo x1 permanece fijo como pivote y cada nueva secante parte desde (x1, y1) hacia (x_k, y_k).",
            ],
            "resultado_final": "Convergencia monótona por un lado hacia la raíz real.",
        },
        "practica_profesora": {
            "titulo": "TP3 Ej. 2: 2x⁴ - 6x² + 7x - 2 = 0 (menor raíz) con E < 0.001",
            "enunciado": "Hallar la raíz negativa de f(x) = 2x⁴ - 6x² + 7x - 2 en [-2.5, -2.0] usando Interpolación Lineal con E < 0.001.",
            "datos_iniciales": {
                "f_str": "2*x**4 - 6*x**2 + 7*x - 2",
                "a": -2.5,
                "b": -2.0,
                "tol": 0.001,
                "max_iter": 50,
            },
            "solucion_referencia": "Pivote fijo x0 = -2.5 (f=21.125). En 6 pasos (x2=-2.137339 ... x6=-2.191747) alcanza |x6-x5| = 0.0007 < 0.001.",
        },
    },
    "newton_raphson": {
        "nombre": "Método de Newton-Raphson (Tangentes)",
        "familia": "raices",
        "icono": "📐",
        "resumen_corto": "Usa la recta tangente en cada punto para saltar hacia la raíz con velocidad cuadrática.",
        "guru": {
            "que_resuelve": "Encuentra raíces reales de funciones derivables en muy pocas iteraciones.",
            "intuicion_geometrica": "Nos paramos en (x_n, f(x_n)), deslizamos por la recta tangente hasta chocar con el eje X y ese punto es el nuevo x_(n+1). Cada paso prácticamente duplica los decimales exactos.",
            "cuando_usar": [
                "Cuando f'(x) es fácil de calcular y se verifican las Condiciones de Fourier en un extremo del intervalo.",
            ],
            "cuando_falla": [
                "Si f'(x_n) ≈ 0 (tangente horizontal → división por cero o salto al infinito).",
                "Si el punto inicial x_0 no cumple f(x_0)·f''(x_0) > 0, pudiendo oscilar o salir del intervalo.",
            ],
        },
        "formulas_latex": [
            r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}, \qquad \text{Condición de Fourier: } f(x_0)\cdot f''(x_0) > 0",
        ],
        "ejemplo_teorico": {
            "titulo": "Cálculo de √2 con f(x) = x² - 2 y x0 = 3",
            "enunciado": "Aproximar √2 aplicando x_(n+1) = x_n - (x_n² - 2)/(2x_n) partiendo de x0 = 3.",
            "pasos": [
                "x1 = 3 - 7/6 = 1.833333...",
                "x2 = 1.462121...",
                "x3 = 1.414998...",
                "x4 = 1.41421378... → x5 = 1.414213562 (>10 cifras exactas en solo 5 iteraciones).",
            ],
            "resultado_final": "Convergencia cuadrática en 5 iteraciones.",
        },
        "practica_profesora": {
            "titulo": "TP3 Ej. 3: x³ - 2x - 5 = 0 en [2, 2.5] con E < 0.0001",
            "enunciado": "Evaluar Fourier en [2, 2.5] para elegir x0 y aproximar la raíz de x³ - 2x - 5 = 0 con E < 0.0001.",
            "datos_iniciales": {
                "f_str": "x**3 - 2*x - 5",
                "a": 2.0,
                "b": 2.5,
                "x0": 2.5,
                "tol": 0.0001,
                "max_iter": 50,
            },
            "solucion_referencia": "Fourier falla en x=2 ((-1)(12)<0) y cumple en x=2.5 ((5.625)(15)>0 → x0=2.5). Converge en 5 pasos a x5 = 2.094551.",
        },
    },
    "punto_fijo_aitken": {
        "nombre": "Punto Fijo y Aceleración Δ² de Aitken",
        "familia": "raices",
        "icono": "🌀",
        "resumen_corto": "Reescribe f(x)=0 como x = g(x) y acelera la convergencia lineal mediante extrapolación Δ².",
        "guru": {
            "que_resuelve": "Encuentra el punto donde la curva y = g(x) corta a la recta identidad y = x, acelerando sucesiones lentas.",
            "intuicion_geometrica": "Forma una 'escalera' o 'telaraña' rebotando entre y = g(x) e y = x. Cada 2 rebotes, Aitken proyecta geométricamente dónde se intersectarían para saltar casi directo a la raíz.",
            "cuando_usar": [
                "Cuando se puede despejar x = g(x) con |g'(x)| < 1 en el entorno de la raíz.",
            ],
            "cuando_falla": [
                "Diverge exponencialmente si |g'(x)| >= 1 en el punto de inicio.",
            ],
        },
        "formulas_latex": [
            r"x_{n+1} = g(x_n) \;\; (|g'(x)| < 1), \qquad S = x_{n+1} - \frac{(x_{n+1} - x_n)^2}{x_{n+1} - 2x_n + x_{n-1}}",
        ],
        "ejemplo_teorico": {
            "titulo": "e^x - x² - 3 = 0 en (1, 2) con Aitken",
            "enunciado": "Usar x_(n+1) = ln(x_n² + 3) desde x0 = 1 con aceleración Δ² de Aitken.",
            "pasos": [
                "x0 = 1 → x1 = 1.38629 → x2 = 1.59367.",
                "Salto de Aitken: x3 = 1.83405.",
                "Nuevo ciclo PF: x4 = 1.85062, x5 = 1.86016 → Aitken: x6 = 1.87311.",
            ],
            "resultado_final": "r ≈ 1.87311 en 2 ciclos de Aitken.",
        },
        "practica_profesora": {
            "titulo": "TP3 Ej. 4: e^x - x² - 5 = 0 con x0 = 2.5 y E < 0.0001",
            "enunciado": "Despejar g(x) = ln(x² + 5), comparar Iteración Simple (10 pasos) frente a Aceleración de Aitken (4 pasos).",
            "datos_iniciales": {
                "f_str": "exp(x) - x**2 - 5",
                "g_str": "log(x**2 + 5)",
                "x0": 2.5,
                "tol": 0.0001,
                "max_iter": 50,
            },
            "solucion_referencia": "Iteración simple tarda 10 iteraciones (2.3563986). Aitken salta de x0=2.5, x1=2.420368, x2=2.3849189 a x3=2.356476 y converge en x4=2.356408.",
        },
    },

    # ==========================================================================
    # 3. ÁLGEBRA LINEAL NUMÉRICA Y AUTOVALORES
    # ==========================================================================
    "eliminacion_gauss": {
        "nombre": "Eliminación Gaussiana (con Pivoteo Parcial)",
        "familia": "sistemas",
        "icono": "📐",
        "resumen_corto": "Triangulariza la matriz ampliada [A|b] y resuelve las incógnitas mediante sustitución regresiva.",
        "guru": {
            "que_resuelve": "Sistemas lineales cuadrados Ax = b de tamaño pequeño a mediano en una cantidad fija de operaciones.",
            "intuicion_geometrica": "Combina ecuaciones (planos) multiplicando por factores m_ik = a_ik/a_kk para anular variables debajo de la diagonal, dejando la última ecuación con 1 sola incógnita.",
            "cuando_usar": [
                "Sistemas densos de hasta ~25 ecuaciones cuando se busca solución exacta directa en un solo vector b.",
            ],
            "cuando_falla": [
                "Pivotes nulos o cercanos a cero sin pivoteo parcial.",
                "Sistemas muy grandes (>1000 variables) o mal condicionados donde se acumula error de redondeo.",
            ],
        },
        "formulas_latex": [
            r"a'_{ij} = a_{ij} - \frac{a_{ik}}{a_{kk}} a_{kj}, \qquad x_i = \frac{a_{im} - \sum_{j=i+1}^n a_{ij} x_j}{a_{ii}}",
        ],
        "ejemplo_teorico": {
            "titulo": "Sistema 3x3 triangularizado de Unidad 4",
            "enunciado": "Resolver 2x1 - 2x2 + 5x3 = 13; 2x1 + 3x2 + 4x3 = 20; 3x1 - x2 + 3x3 = 10.",
            "pasos": [
                "Eliminación lleva la matriz a [[2, -2, 5 | 13], [0, 5, -1 | 7], [0, 0, -4.1 | -12.3]].",
                "Sustitución hacia atrás: x3 = 3 → x2 = 2 → x1 = 1.",
            ],
            "resultado_final": "x = [1, 2, 3]^T.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 5: Sistema 3x3 de Cátedra",
            "enunciado": "Resolver mediante Eliminación Gaussiana: 18.72x + 8.2y + 8.76z = 121.280; 6.4x + 15.9y + 7.18z = 126.321; 5.22x + 6.4y + 14.31z = 118.522.",
            "datos_iniciales": {
                "A": [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]],
                "b": [121.280, 126.321, 118.522],
            },
            "solucion_referencia": "Reducción final: 10.553z = 58.055 → x = 1.836, y = 4.721, z = 5.501.",
        },
    },
    "gauss_jordan": {
        "nombre": "Método de Gauss-Jordan y Matriz Inversa",
        "familia": "sistemas",
        "icono": "🔁",
        "resumen_corto": "Reduce [A|b|I] directamente a la matriz identidad [I|x|A⁻¹] eliminando por arriba y por abajo.",
        "guru": {
            "que_resuelve": "Encuentra simultáneamente la solución del sistema Ax = b y la matriz inversa A⁻¹ sin sustitución regresiva.",
            "intuicion_geometrica": "Transforma paso a paso la base de los vectores columnas hasta convertirla en los ejes canónicos ortogonales.",
            "cuando_usar": [
                "Cuando necesitas calcular la matriz inversa A⁻¹ explícita o resolver varios vectores b simultáneamente.",
            ],
            "cuando_falla": [
                "Requiere ~50% más multiplicaciones que Eliminación de Gauss si solo te interesa resolver un único vector b.",
            ],
        },
        "formulas_latex": [
            r"[A \mid b \mid I] \xrightarrow{\text{Operaciones Elementales}} [I \mid x \mid A^{-1}]",
        ],
        "ejemplo_teorico": {
            "titulo": "Reducción completa del sistema 2x1 - 2x2 + 5x3 = 13...",
            "enunciado": "Normalizar cada fila pivote y anular tanto arriba como abajo de la diagonal.",
            "pasos": [
                "Normaliza fila 1 dividiendo por 2 y anula col 1 en filas 2 y 3.",
                "Repite en cols 2 y 3 hasta obtener I en el bloque izquierdo.",
            ],
            "resultado_final": "x = [1, 2, 3]^T directamente en la última columna.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 6: Gauss-Jordan sobre el sistema 3x3 de Cátedra",
            "enunciado": "Aplicar el método de rotación / Gauss-Jordan sobre el sistema de 18.72x + 8.2y + 8.76z = 121.280...",
            "datos_iniciales": {
                "A": [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]],
                "b": [121.280, 126.321, 118.522],
            },
            "solucion_referencia": "Solución PDF (2 dec): [1.82, 4.72, 5.50]; cálculo de alta precisión: [1.8358, 4.7213, 5.5013].",
        },
    },
    "factorizacion_lu": {
        "nombre": "Factorización LU (Doolittle)",
        "familia": "sistemas",
        "icono": "🧩",
        "resumen_corto": "Descompone A = L·U separando los multiplicadores de Gauss (L) de la matriz escalonada (U).",
        "guru": {
            "que_resuelve": "Desacopla la matriz A del vector independiente b, permitiendo resolver múltiples escenarios Ax = b_k en tiempo O(n²).",
            "intuicion_geometrica": "Guarda la 'receta de transformaciones' (L) y la 'matriz escalonada' (U). Si mañana cambia el vector b, solo aplicas L y U sin volver a triangularizar A.",
            "cuando_usar": [
                "Cuando la matriz del sistema es fija pero los términos independientes b cambian repetidamente.",
            ],
            "cuando_falla": [
                "Sin permutaciones (pivoteo), falla si aparece un cero en la diagonal U_kk.",
            ],
        },
        "formulas_latex": [
            r"A = L \cdot U \implies \begin{cases} L y = b & \text{(sustitución progresiva)} \\ U x = y & \text{(sustitución regresiva)} \end{cases}",
        ],
        "ejemplo_teorico": {
            "titulo": "Descomposición 3x3 exacta de Unidad 4",
            "enunciado": "Factorizar A = [[4, -2, 1], [20, -7, 12], [-8, 13, 17]] y resolver para b = [11, 70, 17].",
            "pasos": [
                "Multiplicadores: f21 = 5, f31 = -2, f32 = 3 → forman L.",
                "U = [[4, -2, 1], [0, 3, 7], [0, 0, -2]].",
                "Ly = b → y = [11, 15, -6]^T; Ux = y → x = [1, -2, 3]^T.",
            ],
            "resultado_final": "x = [1, -2, 3]^T.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 7: Sistema 4x1 - 2x2 + x3 = 11",
            "enunciado": "Obtener L y U, resolver el vector auxiliar y mediante Ly = b y hallar x mediante Ux = y.",
            "datos_iniciales": {
                "A": [[4.0, -2.0, 1.0], [20.0, -7.0, 12.0], [-8.0, 13.0, 17.0]],
                "b": [11.0, 70.0, 17.0],
            },
            "solucion_referencia": "y = [11, 15, -6], x = [1, -2, 3].",
        },
    },
    "gauss_seidel": {
        "nombre": "Método Iterativo de Gauss-Seidel",
        "familia": "sistemas",
        "icono": "🔄",
        "resumen_corto": "Aproxima la solución de Ax = b iterativamente reutilizando de inmediato cada variable recién calculada.",
        "guru": {
            "que_resuelve": "Sistemas lineales grandes o dispersos donde la eliminación directa sería muy costosa.",
            "intuicion_geometrica": "Despeja cada variable de su ecuación principal y rebota secuencialmente entre hiperplanos hasta converger al punto de intersección.",
            "cuando_usar": [
                "Cuando la matriz es diagonalmente dominante: |a_ii| >= sum_{j != i} |a_ij|.",
            ],
            "cuando_falla": [
                "Si la matriz NO tiene dominancia diagonal (o radio espectral >= 1), puede diverger al infinito.",
            ],
        },
        "formulas_latex": [
            r"x_i^{(k+1)} = \frac{b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)}}{a_{ii}}",
        ],
        "ejemplo_teorico": {
            "titulo": "Reordenamiento para lograr dominancia diagonal",
            "enunciado": "Resolver 2x1+10x2+x3=51; x1+2x2+10x3=61; 10x1+x2+2x3=44 con inicio (0,0,0) y E <= 0.02.",
            "pasos": [
                "Despejar la variable dominante en cada fila: x2 de ec.1, x3 de ec.2, x1 de ec.3.",
                "Iter 1: (2.874, 5.1, 5.08) → Iter 3: (2.9999, 3.9998, 5.0004).",
            ],
            "resultado_final": "Converge a x = [3, 4, 5] en 3 iteraciones.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 8: Sistema con Dominancia Diagonal y x0 = [1, 1, 1]",
            "enunciado": "Verificar dominancia diagonal (18.72 > 16.96; 15.9 > 13.58; 14.31 > 11.62) e iterar desde [1, 1, 1] con tolerancia 0.01.",
            "datos_iniciales": {
                "A": [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]],
                "b": [121.280, 126.321, 118.522],
                "x0": [1.0, 1.0, 1.0],
                "tol": 0.01,
                "max_iter": 50,
            },
            "solucion_referencia": "En 6 iteraciones converge a [1.835, 4.721, 5.502] con errores <= 0.005.",
        },
    },
    "potencias": {
        "nombre": "Método de las Potencias (Directo e Inverso)",
        "familia": "autovalores",
        "icono": "⚡",
        "resumen_corto": "Multiplica repetidamente un vector inicial por A (o A⁻¹) para extraer los autovalores extremos λ_max y λ_min.",
        "guru": {
            "que_resuelve": "Halla el autovalor dominante (mayor módulo) o mínimo (aplicado sobre A⁻¹) y su autovector asociado sin construir polinomios.",
            "intuicion_geometrica": "Al multiplicar A·v repetidamente, la componente alineada con la dirección de máximo estiramiento (λ_max) crece mucho más rápido que las demás hasta dominar por completo al vector.",
            "cuando_usar": [
                "Matrices grandes donde solo interesan los modos extremos (ej. frecuencia fundamental o límite de estabilidad).",
            ],
            "cuando_falla": [
                "Si |λ_1| ≈ |λ_2| la convergencia se vuelve lentísima.",
                "Si el vector inicial X0 es exactamente ortogonal al autovector dominante.",
                "Para λ_min exige que A sea invertible (det(A) != 0).",
            ],
        },
        "formulas_latex": [
            r"A X_{k-1} = \bar{\lambda}_k X_k \quad (\lambda_{\max}), \qquad A^{-1} X_{k-1} = \frac{1}{\bar{\lambda}_k} X_k \quad (\lambda_{\min})",
        ],
        "ejemplo_teorico": {
            "titulo": "Normalización por componente máxima en norma infinito",
            "enunciado": "A partir de X0 arbitrario, calcular Y_k = A·X_{k-1}, tomar m_k = componente de mayor valor absoluto como aproximación de λ y normalizar X_k = Y_k / m_k.",
            "pasos": [
                "Iterar hasta que |λ_k - λ_{k-1}| / |λ_k| < tolerancia.",
            ],
            "resultado_final": "m_k → λ_max y X_k → autovector normalizado.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 10: Matriz Tridiagonal Simétrica 3x3 con X0 = [1, 1, 1]",
            "enunciado": "Hallar el máximo y mínimo autovalor de A = [[3.556, -1.778, 0], [-1.778, 3.556, -1.778], [0, -1.778, 3.556]] partiendo de X0 = [1, 1, 1].",
            "datos_iniciales": {
                "A": [[3.556, -1.778, 0.0], [-1.778, 3.556, -1.778], [0.0, -1.778, 3.556]],
                "x0": [1.0, 1.0, 1.0],
                "tol": 0.05,
                "max_iter": 30,
            },
            "solucion_referencia": "λ_max ≈ 6.095 con v = [-0.708, 1, -0.708]^T (5 pasos). λ_min ≈ 1.037 con v = [0.709, 1, 0.709]^T (3 pasos).",
        },
    },
    "faddeev_leverrier": {
        "nombre": "Método de Faddeev-Leverrier (Espectro Completo)",
        "familia": "autovalores",
        "icono": "🔬",
        "resumen_corto": "Calcula recursivamente matrices B_k y trazas b_k para obtener el polinomio característico completo y A⁻¹.",
        "guru": {
            "que_resuelve": "Determina TODOS los autovalores simultáneamente (vía polinomio característico), el determinante y la matriz inversa analítica.",
            "intuicion_geometrica": "Extrae invariantes geométricos de la matriz A a través de sus trazas sucesivas, reconstruyendo la ecuación polinómica cuyas raíces son exactamente las direcciones propias.",
            "cuando_usar": [
                "Matrices pequeñas o medianas (n <= 6) cuando se requiere conocer todo el espectro de autovalores y la expresión del polinomio.",
            ],
            "cuando_falla": [
                "Inestabilidad numérica para matrices de gran dimensión (n > 8) al evaluar raíces de polinomios de alto grado.",
                "Para obtener A⁻¹ exige b_n != 0 (det(A) != 0).",
            ],
        },
        "formulas_latex": [
            r"B_1 = A, \;\; b_1 = \operatorname{tr}(B_1); \qquad B_k = A(B_{k-1} - b_{k-1}I), \;\; b_k = \frac{1}{k}\operatorname{tr}(B_k)",
            r"P(\lambda) = (-1)^n \left(\lambda^n - b_1 \lambda^{n-1} - \dots - b_n\right) = 0, \qquad A^{-1} = \frac{1}{b_n}(B_{n-1} - b_{n-1}I)",
        ],
        "ejemplo_teorico": {
            "titulo": "Matriz simétrica clásica 3x3 de Unidad 4",
            "enunciado": "Calcular polinomio característico, autovalores e inversa de A = [[3, 2, 4], [2, 0, 2], [4, 2, 3]].",
            "pasos": [
                "B1 = A → b1 = 6.",
                "B2 = [[11, 2, 4], [2, 8, 2], [4, 2, 11]] → b2 = 15.",
                "B3 = 8I → b3 = 8.",
                "Polinomio: λ³ - 6λ² - 15λ - 8 = 0 → raíces λ = 8, -1, -1.",
            ],
            "resultado_final": "Autovalores {8, -1, -1}; A⁻¹ = (1/8)(B2 - 15I).",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 9: Faddeev-Leverrier + Autovector de λ = 8",
            "enunciado": "Obtener coeficientes b1, b2, b3, polinomio característico y espectro completo de la matriz [[3, 2, 4], [2, 0, 2], [4, 2, 3]].",
            "datos_iniciales": {
                "A": [[3.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 3.0]],
            },
            "solucion_referencia": "b1=6, b2=15, b3=8. P(λ) = -λ³ + 6λ² + 15λ + 8. Raíces: 8, -1, -1. Para λ=8, v = [-1, -0.5, -1]^T.",
        },
    },

    # ==========================================================================
    # 4. INTERPOLACIÓN POLINOMIAL
    # ==========================================================================
    "interpolacion_lineal": {
        "nombre": "Interpolación Lineal Directa e Inversa",
        "familia": "interpolacion",
        "icono": "📍",
        "resumen_corto": "Conecta dos puntos adyacentes con una recta secante para estimar y(x) o despejar x(y).",
        "guru": {
            "que_resuelve": "Estima valores intermedios entre dos datos consecutivos de una tabla o encuentra qué x produce una altura y deseada.",
            "intuicion_geometrica": "Tira una cuerda recta entre (x0, y0) y (x1, y1) y lee la coordenada sobre esa recta.",
            "cuando_usar": [
                "Cuando el paso h es pequeño y la función se comporta casi linealmente en el subintervalo.",
            ],
            "cuando_falla": [
                "Cuando la función tiene curvatura pronunciada (Δ²y grande): subestima o sobreestima sistemáticamente.",
            ],
        },
        "formulas_latex": [
            r"P_1(x) = y_0 + \frac{\Delta y_0}{h}(x - x_0) \quad \Longleftrightarrow \quad x = x_0 + \frac{(y - y_0)\,h}{\Delta y_0}",
        ],
        "ejemplo_teorico": {
            "titulo": "Estimación de ln(2) en tabla lineal",
            "enunciado": "Comparación gráfica de Chapra mostrando cómo el segmento secante entre x=1 y x=4 aproxima la curva logarítmica con error de curvatura.",
            "pasos": [
                "Se toma la pendiente secante Δy0 / h.",
                "La diferencia respecto al valor real decrece al acortar h.",
            ],
            "resultado_final": "Exacta únicamente si los puntos son colineales (Δ²y = 0).",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. Inversa: Viscosidad del Agua (y = 1.702)",
            "enunciado": "Dada la tabla de viscosidad x=[0, 5, 10, 15], y=[1.792, 1.519, 1.308, 1.140], hallar la temperatura x para y = 1.702.",
            "datos_iniciales": {
                "x": [0.0, 5.0, 10.0, 15.0],
                "y": [1.792, 1.519, 1.308, 1.140],
                "y_objetivo": 1.702,
                "x_objetivo": 2.5,
            },
            "solucion_referencia": "1.702 = 1.792 + (-0.273/5)(x - 0) → x ≈ 1.64835 °C (en PDF truncado a 1.64 °C).",
        },
    },
    "cuadratica_inversa": {
        "nombre": "Interpolación Cuadrática Inversa (Newton-Gregory Grado 2)",
        "familia": "interpolacion",
        "icono": "➰",
        "resumen_corto": "Ajusta una parábola con 3 nodos y resuelve la ecuación cuadrática resultante para hallar x dado y.",
        "guru": {
            "que_resuelve": "Mejora drásticamente la interpolación inversa cuando los datos siguen una relación curva.",
            "intuicion_geometrica": "Toma 3 puntos para dibujar una parábola y corta horizontalmente a la altura y_obj para leer la abscisa x dentro del rango de la tabla.",
            "cuando_usar": [
                "En tablas equidistantes con al menos 3 nodos donde la recta lineal introduce error por curvatura.",
            ],
            "cuando_falla": [
                "Si el discriminante b² - 4ac < 0 (la altura y_obj no corta a la parábola real).",
                "Requiere nodos equidistantes en x si se formula vía Newton-Gregory de 2° orden.",
            ],
        },
        "formulas_latex": [
            r"y = y_0 + \frac{\Delta y_0}{h}(x - x_0) + \frac{\Delta^2 y_0}{2!\,h^2}(x - x_0)(x - x_1) \implies a u^2 + b u + c = 0 \;\; (u = x - x_0)",
        ],
        "ejemplo_teorico": {
            "titulo": "Inversión de tabla en presencia de curvatura",
            "enunciado": "Cuando se invierte x = f^(-1)(y), los valores de y rara vez son equidistantes, por lo que Newton-Gregory de 2° grado plantea una ecuación cuadrática en x.",
            "pasos": [
                "Se expande Newton-Gregory hasta Δ²y0 igualado al y_objetivo.",
                "Se resuelve por Baskara y se descarta la raíz fuera del intervalo tabular.",
            ],
            "resultado_final": "La raíz dentro del intervalo representa la interpolación cuadrática inversa.",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. 2b: Viscosidad del Agua (y = 1.702) con 3 nodos",
            "enunciado": "Con la misma tabla x=[0, 5, 10, 15], y=[1.792, 1.519, 1.308, 1.140], calcular la temperatura para y = 1.702 incorporando Δ²y0 = 0.062.",
            "datos_iniciales": {
                "x": [0.0, 5.0, 10.0, 15.0],
                "y": [1.792, 1.519, 1.308, 1.140],
                "y_objetivo": 1.702,
            },
            "solucion_referencia": "0.00124 x² - 0.0608 x + 0.09 = 0 → x1 = 47.50 (descartada), x2 ≈ 1.5296 °C (PDF: 1.52 °C).",
        },
    },
    "lagrange": {
        "nombre": "Polinomio Interpolador de Lagrange",
        "familia": "interpolacion",
        "icono": "🌊",
        "resumen_corto": "Construye el único polinomio de grado <= n-1 que pasa por puntos no necesariamente equidistantes.",
        "guru": {
            "que_resuelve": "Interpola valores sobre tablas experimentales con espaciamiento irregular (h variable).",
            "intuicion_geometrica": "Crea un polinomio base L_i(x) por cada dato que vale exactamente 1 en x_i y 0 en todos los demás nodos. La combinación sum(y_i * L_i(x)) pasa exactamente por todos los puntos.",
            "cuando_usar": [
                "Siempre que los nodos x_i NO estén equiespaciados.",
                "Para construir explícitamente el polinomio analítico P(x).",
            ],
            "cuando_falla": [
                "Si hay nodos x_i duplicados (división por cero).",
                "Con demasiados puntos (n > 10) puede sufrir oscilaciones severas en los bordes (Fenómeno de Runge).",
            ],
        },
        "formulas_latex": [
            r"P_n(x) = \sum_{k=0}^{n} y_k L_k(x), \qquad L_k(x) = \prod_{\substack{j=0 \\ j \neq k}}^{n} \frac{x - x_j}{x_k - x_j}",
        ],
        "ejemplo_teorico": {
            "titulo": "Forma factorizada vs. polinomio de Newton (no equidistantes)",
            "enunciado": "Dado un conjunto de 3 nodos no equidistantes (x=[1, 4, 6], y=[0, 1.386294, 1.791759] de ln(x)), evaluar en x = 2.",
            "pasos": [
                "Se calculan los polinomios base L0(2), L1(2), L2(2).",
                "Se suma y0·L0(2) + y1·L1(2) + y2·L2(2) = 0.5658442.",
            ],
            "resultado_final": "El polinomio de Lagrange coincide algebraicamente con el de Newton de diferencias divididas (unicidad del polinomio interpolador).",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. 2: Presión de Vapor de Agua a T = 180 °C",
            "enunciado": "Calcular la presión de vapor de agua a 180 °C a partir de datos no equidistantes T=[100, 120, 200, 220], P=[1.01, 3.61, 15.5, 23.2].",
            "datos_iniciales": {
                "x": [100.0, 120.0, 200.0, 220.0],
                "y": [1.01, 3.61, 15.5, 23.2],
                "x_eval": 180.0,
            },
            "solucion_referencia": "Producto (180-100)(180-120)(180-200)(180-220) = 3.840.000 → P(180) ≈ 10.562 Pascal.",
        },
    },
    "newton_gregory_asc": {
        "nombre": "Newton-Gregory Ascendente (Diferencias Progresivas)",
        "familia": "interpolacion",
        "icono": "📈",
        "resumen_corto": "Usa la diagonal superior de la tabla de diferencias finitas Δ^k y0 para interpolar al inicio de tablas equidistantes.",
        "guru": {
            "que_resuelve": "Interpola rápidamente cerca del comienzo de una tabla con paso h constante.",
            "intuicion_geometrica": "Toma y0 y le va sumando correcciones basadas en la velocidad (Δy0), aceleración (Δ²y0) y curvaturas superiores leídas desde el borde inicial.",
            "cuando_usar": [
                "Exclusivamente cuando los nodos x están equidistantes (h constante) y el punto buscado está cerca de x_0.",
            ],
            "cuando_falla": [
                "NO aplica si el usuario altera un nodo x_i rompiendo la equidistancia (debe usarse Lagrange).",
            ],
        },
        "formulas_latex": [
            r"u = \frac{x - x_0}{h}, \qquad P_n(x) = y_0 + u\,\Delta y_0 + \frac{u(u-1)}{2!}\Delta^2 y_0 + \frac{u(u-1)(u-2)}{3!}\Delta^3 y_0 + \dots",
        ],
        "ejemplo_teorico": {
            "titulo": "Tabla de diferencias progresivas de y = x² en x = [0, 1, 2, 3, 4]",
            "enunciado": "Para un polinomio cuadrático, las primeras diferencias son [1, 3, 5, 7], las segundas son constantes e iguales a [2, 2, 2], y Δ³y = 0.",
            "pasos": [
                "Construir la tabla Δ^k y_i.",
                "Verificar que para un polinomio de grado n, las diferencias de orden n son constantes y las de orden n+1 se anulan.",
            ],
            "resultado_final": "Permite detectar el grado exacto de una tabla polinómica o errores de tipeo en un dato.",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. 1a: Newton-Gregory Ascendente en x = 0.35",
            "enunciado": "Dada la tabla con h = 0.3: x=[0.2, 0.5, 0.8, 1.1, 1.4], y=[0.258, 2.535, 4.902, 6.722, 10.704], interpolar f(0.35).",
            "datos_iniciales": {
                "x": [0.2, 0.5, 0.8, 1.1, 1.4],
                "y": [0.258, 2.535, 4.902, 6.722, 10.704],
                "x_eval": 0.35,
            },
            "solucion_referencia": "u = (0.35 - 0.2)/0.3 = 0.5; diagonal [0.258, 2.277, 0.09, -0.637, 3.346] → P(0.35) = 1.214734375.",
        },
    },
    "newton_gregory_desc": {
        "nombre": "Newton-Gregory Descendente (Diferencias Retrógradas)",
        "familia": "interpolacion",
        "icono": "📉",
        "resumen_corto": "Usa la diagonal inferior de la tabla de diferencias ∇^k yn para interpolar al final de tablas equidistantes.",
        "guru": {
            "que_resuelve": "Interpola con máxima comodidad cerca del extremo final x_n en tablas equidistantes.",
            "intuicion_geometrica": "Ancla el polinomio en el último dato (x_n, y_n) y retrocede hacia la izquierda sumando contribuciones con signos positivos u(u+1).",
            "cuando_usar": [
                "Cuando x está próximo a x_n y los nodos están equidistantes.",
            ],
            "cuando_falla": [
                "Requiere equidistancia estricta h constante entre todos los x_i.",
            ],
        },
        "formulas_latex": [
            r"v = \frac{x - x_n}{h}, \qquad P_n(x) = y_n + v\,\nabla y_n + \frac{v(v+1)}{2!}\nabla^2 y_n + \frac{v(v+1)(v+2)}{3!}\nabla^3 y_n + \dots",
        ],
        "ejemplo_teorico": {
            "titulo": "Cambio de origen a x_n en tabla equiespaciada",
            "enunciado": "Al centrar u = (x - x_n)/h, el polinomio utiliza las diferencias acumuladas hacia atrás hasta el último punto disponible.",
            "pasos": [
                "Evaluar v < 0 (cuando x < x_n) con factores v(v+1)...(v+k-1)/k!.",
            ],
            "resultado_final": "Reproduce exactamente el mismo polinomio único de grado n que la forma ascendente.",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. 1b: Newton-Gregory Descendente en x = 1.37",
            "enunciado": "Con la misma tabla x=[0.2, 0.5, 0.8, 1.1, 1.4], y=[0.258, 2.535, 4.902, 6.722, 10.704], interpolar f(1.37).",
            "datos_iniciales": {
                "x": [0.2, 0.5, 0.8, 1.1, 1.4],
                "y": [0.258, 2.535, 4.902, 6.722, 10.704],
                "x_eval": 1.37,
            },
            "solucion_referencia": "v = (1.37 - 1.4)/0.3 = -0.1; diagonal inferior [10.704, 3.982, 2.162, 2.709, 3.346] → P(1.37) = 10.06216678.",
        },
    },

    # ==========================================================================
    # 5. INTEGRACIÓN Y DERIVACIÓN NUMÉRICA
    # ==========================================================================
    "diferenciacion_numerica": {
        "nombre": "Diferenciación Numérica (Gregory Ascendente y Descendente)",
        "familia": "integracion",
        "icono": "📐",
        "resumen_corto": "Calcula derivadas de 1° y 2° orden en tablas discretas derivando los polinomios de Newton-Gregory.",
        "guru": {
            "que_resuelve": "Obtiene f'(x) y f''(x) cuando no conocemos la fórmula analítica o solo disponemos de mediciones tabulares.",
            "intuicion_geometrica": "Construye la curva polinómica local que une los puntos de la tabla y calcula exactamente la pendiente de la tangente y la concavidad en el nodo elegido.",
            "cuando_usar": [
                "Ascendente al inicio de la tabla (para aprovechar diferencias progresivas Δ^k); Descendente al final de la tabla (∇^k).",
            ],
            "cuando_falla": [
                "Pasos h extremadamente pequeños (h < 10^-8) amplifican la cancelación numérica y disparan el error.",
                "Muy sensible al ruido experimental en los datos y_i.",
            ],
        },
        "formulas_latex": [
            r"f'(x_0) \approx \frac{1}{h}\left[\Delta y_0 - \frac{1}{2}\Delta^2 y_0 + \frac{1}{3}\Delta^3 y_0 - \frac{1}{4}\Delta^4 y_0\right], \qquad f''(x_0) \approx \frac{1}{h^2}\left[\Delta^2 y_0 - \Delta^3 y_0 + \frac{11}{12}\Delta^4 y_0\right]",
            r"f'(x_n) \approx \frac{1}{h}\left[\nabla y_n + \frac{1}{2}\nabla^2 y_n + \frac{1}{3}\nabla^3 y_n + \frac{1}{4}\nabla^4 y_n\right], \qquad f''(x_n) \approx \frac{1}{h^2}\left[\nabla^2 y_n + \nabla^3 y_n + \frac{11}{12}\nabla^4 y_n\right]",
        ],
        "ejemplo_teorico": {
            "titulo": "Paradoja del paso h -> 0 en f(x) = e^x en x = 1",
            "enunciado": "Evaluar (f(1+h)-f(1))/h reduciendo h desde 10^-1 hasta 10^-14 (exacto e = 2.71828182845905).",
            "pasos": [
                "El error baja hasta h = 10^-8 (error = 5.10e-9).",
                "Para h <= 10^-9 la resta f(1+h)-f(1) pierde dígitos por cancelación y el error vuelve a crecer hasta 0.0537 en h = 10^-14.",
            ],
            "resultado_final": "Existe un paso h óptimo que balancea error de truncamiento y error de redondeo.",
        },
        "practica_profesora": {
            "titulo": "TP6 Ej. Diferenciación: Tabla h = 1 (x = 12..17)",
            "enunciado": "Dada la tabla x=[12, 13, 14, 15, 16, 17], y=[0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151], calcular f'(12) (ascendente) y f''(17) (descendente).",
            "datos_iniciales": {
                "x": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                "y": [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151],
            },
            "solucion_referencia": "f'(12) = -0.0083175; f'(12.4) = -0.0153909; f''(17) = 0.011254.",
        },
    },
    "trapecio": {
        "nombre": "Regla de los Trapecios Compuesta",
        "familia": "integracion",
        "icono": "🪜",
        "resumen_corto": "Aproxima el área bajo la curva uniendo cada par de nodos consecutivos con segmentos rectos.",
        "guru": {
            "que_resuelve": "Calcula integrales definidas de cualquier función o tabla discreta sin importar si n es par o impar.",
            "intuicion_geometrica": "Divide el área en franjas verticales y sustituye la parte superior curva por un techo plano recto (un trapecio por subintervalo).",
            "cuando_usar": [
                "Funciona para CUALQUIER cantidad de subintervalos n >= 1.",
                "Útil cuando los datos tienen bordes con paso distinto o como tramo residual en reglas combinadas.",
            ],
            "cuando_falla": [
                "Orden de error O(h²): necesita muchos subintervalos para igualar la precisión que Simpson logra con pocos puntos en curvas suaves.",
            ],
        },
        "formulas_latex": [
            r"A = \frac{h}{2}\left[E + 2P + 2I\right], \qquad E_a \le \frac{(b - a)^3}{12\,n^2}\max|f''(x)|",
        ],
        "ejemplo_teorico": {
            "titulo": "Polinomio de Chapra en [0, 0.8] con n = 2 (h = 0.4)",
            "enunciado": "Integrar f(x) = 0.2 + 25x - 200x² + 675x³ - 900x⁴ + 400x⁵ en [0, 0.8] (exacto = 1.640533).",
            "pasos": [
                "n=2: f(0)=0.2, f(0.4)=2.456, f(0.8)=0.232.",
                "A = 0.8 * (0.2 + 2(2.456) + 0.232)/4 = 1.0688 (error relativo 34.9%).",
            ],
            "resultado_final": "Al aumentar n a 10, el error cae rápidamente de acuerdo a O(1/n²).",
        },
        "practica_profesora": {
            "titulo": "TP6 Ej. 1: Regla del Trapecio (y = x²/5, h = 5)",
            "enunciado": "Integrar la tabla x=[0.5, 5.5, 10.5, 15.5, 20.5, 25.5], y=[0.05, 6.05, 22.05, 48.05, 84.05, 130.05] por Trapecios.",
            "datos_iniciales": {
                "x": [0.5, 5.5, 10.5, 15.5, 20.5, 25.5],
                "y": [0.05, 6.05, 22.05, 48.05, 84.05, 130.05],
                "f_str": "x**2 / 5",
            },
            "solucion_referencia": "E=130.10, P=54.10, I=106.10 → A = (5/2)[130.10 + 2(54.10) + 2(106.10)] = 1126.25 (Exacto = 1105.417).",
        },
    },
    "simpson_1_3": {
        "nombre": "Regla de Simpson 1/3 Compuesta (y Motor Adaptativo)",
        "familia": "integracion",
        "icono": "🥣",
        "resumen_corto": "Aproxima la integral uniendo cada 3 puntos consecutivos con arcos parabólicos de 2° grado.",
        "guru": {
            "que_resuelve": "Integrales definidas con altísima precisión O(h⁴) y costo computacional bajo.",
            "intuicion_geometrica": "En lugar de techos rectos (Trapecio), coloca parábolas suaves sobre cada par de subintervalos (3 nodos), capturando la curvatura exacta hasta grado 3.",
            "cuando_usar": [
                "Cuando el número de subintervalos n es PAR y los nodos están equiespaciados.",
                "Si al variar Δh en vivo el número de intervalos pasa a ser IMPAR, debe combinarse automáticamente con Simpson 3/8 o Trapecio.",
            ],
            "cuando_falla": [
                "No puede aplicarse puro cuando n es impar ni cuando el paso h cambia a mitad de la tabla.",
            ],
        },
        "formulas_latex": [
            r"A = \frac{h}{3}\left[E + 4I + 2P\right], \qquad E_a \le \frac{(b - a)^5}{180\,n^4}\max|f^{(4)}(x)|",
        ],
        "ejemplo_teorico": {
            "titulo": "Regla simple y compuesta Newton-Cotes de 2° orden",
            "enunciado": "La integración de un polinomio interpolador de Lagrange de grado 2 sobre tres nodos equiespaciados genera exactamente los pesos (1, 4, 1)·h/3.",
            "pasos": [
                "Aunque se deduce con parábolas (grado 2), su término de error depende de f^(4), por lo que es EXACTA incluso para polinomios cúbicos.",
            ],
            "resultado_final": "Orden de convergencia O(h⁴): reducir h a la mitad reduce el error en un factor de 16.",
        },
        "practica_profesora": {
            "titulo": "TP6 Ej. 2: Simpson 1/3 en [0.5, 20.5] (n = 4, h = 5)",
            "enunciado": "Integrar los primeros 5 puntos x=[0.5, 5.5, 10.5, 15.5, 20.5], y=[0.05, 6.05, 22.05, 48.05, 84.05] con Simpson 1/3.",
            "datos_iniciales": {
                "x": [0.5, 5.5, 10.5, 15.5, 20.5],
                "y": [0.05, 6.05, 22.05, 48.05, 84.05],
                "f_str": "x**2 / 5",
            },
            "solucion_referencia": "E=84.10, I=54.10, P=22.05 → A = (5/3)[84.10 + 4(54.10) + 2(22.05)] = 574.333... (Exacta para x²/5).",
        },
    },
    "simpson_3_8_combinado": {
        "nombre": "Simpson 3/8 y Esquema Combinado (3/8 + 1/3)",
        "familia": "integracion",
        "icono": "🧩",
        "resumen_corto": "Usa cúbicas sobre bloques de 3 subintervalos y se combina con 1/3 para resolver particiones impares.",
        "guru": {
            "que_resuelve": "Permite mantener precisión de 4° orden O(h⁴) cuando n es impar (n >= 5) sin tener que degradar a Trapecio.",
            "intuicion_geometrica": "Toma 3 paneles con un polinomio cúbico (Simpson 3/8) para 'consumir' la imparidad, y resuelve todos los pares restantes con parábolas de Simpson 1/3.",
            "cuando_usar": [
                "Simpson 3/8 puro cuando n es múltiplo de 3; Combinado (3/8 + 1/3) para cualquier n impar >= 5.",
            ],
            "cuando_falla": [
                "Requiere paso h uniforme en cada sub-bloque donde se aplique.",
            ],
        },
        "formulas_latex": [
            r"A_{3/8} = \frac{3h}{8}\left[y_0 + 3y_1 + 3y_2 + y_3\right], \qquad A_{\text{comb}} = A_{3/8}(\text{primeros }3) + A_{1/3}(\text{restantes})",
        ],
        "ejemplo_teorico": {
            "titulo": "Estrategia mixta en tablas impares",
            "enunciado": "Para n = 5 subintervalos no es posible aplicar Simpson 1/3 (exige par) ni Simpson 3/8 (exige múltiplo de 3) de forma pura.",
            "pasos": [
                "Se divide el dominio en [x0..x3] (3 subintervalos → Simpson 3/8) y [x3..x5] (2 subintervalos → Simpson 1/3).",
            ],
            "resultado_final": "Mantiene error global O(h⁴) en todo el dominio.",
        },
        "practica_profesora": {
            "titulo": "TP6 Ej. 3: Simpson 3/8 + 1/3 en [0.5, 25.5] (n = 5, h = 5)",
            "enunciado": "Integrar la tabla completa de 6 puntos (n = 5 subintervalos) combinando Simpson 3/8 en [0.5, 15.5] y Simpson 1/3 en [15.5, 25.5].",
            "datos_iniciales": {
                "x": [0.5, 5.5, 10.5, 15.5, 20.5, 25.5],
                "y": [0.05, 6.05, 22.05, 48.05, 84.05, 130.05],
                "f_str": "x**2 / 5",
            },
            "solucion_referencia": "A1 (3/8) = 248.25; A2 (1/3) = 857.1667 → Total = 1105.417 (Exacta!).",
        },
    },

    # ==========================================================================
    # 6. ECUACIONES DIFERENCIALES ORDINARIAS (EDO - PVI)
    # ==========================================================================
    "euler_heun": {
        "nombre": "Euler Clásico y Euler Modificado (Heun)",
        "familia": "edo",
        "icono": "🧭",
        "resumen_corto": "Integra paso a paso un PVI avanzando con la pendiente inicial (Euler) o promediando predicción-corrección (Heun).",
        "guru": {
            "que_resuelve": "Estima la trayectoria y(x) de una ecuación diferencial y' = f(x, y) a partir de una condición inicial (x0, y0).",
            "intuicion_geometrica": "Euler camina en línea recta siguiendo la brújula de la pendiente inicial durante todo el paso h (se desvía en curvas). Euler Modificado da un paso tentativo, mide la pendiente en la llegada, promedia ambas pendientes y corrige.",
            "cuando_usar": [
                "Euler como introducción conceptual o con pasos h muy finos; Heun cuando se quiere orden O(h²) con evaluación iterada del corrector.",
            ],
            "cuando_falla": [
                "Con pasos h grandes, Euler acumula un error global O(h) muy notable y puede volverse inestable en ecuaciones rígidas.",
            ],
        },
        "formulas_latex": [
            r"y_{i+1} = y_i + h\,f(x_i, y_i) \;\; (\text{Euler}), \qquad C^{(m)}(y_{i+1}) = y_i + \frac{h}{2}\left[f(x_i,y_i) + f(x_{i+1}, C^{(m-1)})\right] \;\; (\text{Heun})",
        ],
        "ejemplo_teorico": {
            "titulo": "Modelo de caída libre / paracaidista (Unidad 1 ec. 1.12)",
            "enunciado": "dv/dt = g - (c/m)v se discretiza como v(t_{i+1}) = v(t_i) + [g - (c/m)v(t_i)] Δt.",
            "pasos": [
                "La pendiente en t_i se multiplica por el paso h = Δt.",
                "En Heun de cátedra, se corrige iterativamente hasta |C^1 - C| < E.",
            ],
            "resultado_final": "Euler Modificado reduce drásticamente el error frente a Euler clásico.",
        },
        "practica_profesora": {
            "titulo": "TP7 Ej. 1: y' = 6x²/y, y(4) = 16 en [4.0, 4.4] con h = 0.1",
            "enunciado": "Resolver el PVI y' = 6x²/y con y(4) = 16 hasta x = 4.4 (h = 0.1, E < 0.0001 en Heun), cuya solución exacta es y = 2x√x.",
            "datos_iniciales": {
                "f_str": "6*x**2 / y",
                "x0": 4.0,
                "y0": 16.0,
                "xf": 4.4,
                "h": 0.1,
                "tol_corrector": 0.0001,
                "y_exacta_str": "2*x*sqrt(x)",
            },
            "solucion_referencia": "Exacta y(4.4) = 18.459036. Euler: y(4.4) = 18.44514 (err 0.0139). Euler Modificado: y(4.4) = 18.45901 (err 0.00002).",
        },
    },
    "runge_kutta": {
        "nombre": "Métodos de Runge-Kutta (RK2 y RK4 Clásico)",
        "familia": "edo",
        "icono": "🚀",
        "resumen_corto": "Logra precisión de 2° o 4° orden muestreando múltiples pendientes ponderadas dentro de cada paso h.",
        "guru": {
            "que_resuelve": "Es el caballo de batalla estándar para resolver EDO con altísima precisión sin necesitar derivadas analíticas de orden superior.",
            "intuicion_geometrica": "RK4 evalúa 4 pendientes exploratorias (al inicio k1, dos en el punto medio k2 y k3, y una al final k4) y calcula un promedio ponderado (1 + 2 + 2 + 1)/6 equivalente a la regla de Simpson.",
            "cuando_usar": [
                "Como método principal monopaso para casi cualquier PVI no rígido.",
                "Para generar los 4 valores de arranque necesarios en métodos multipaso como Milne.",
            ],
            "cuando_falla": [
                "Requiere 4 evaluaciones de f(x,y) por paso (mayor costo por paso que métodos multipaso).",
            ],
        },
        "formulas_latex": [
            r"k_1 = h f(x_i, y_i), \;\; k_2 = h f\!\left(x_i+\frac{h}{2}, y_i+\frac{k_1}{2}\right), \;\; k_3 = h f\!\left(x_i+\frac{h}{2}, y_i+\frac{k_2}{2}\right), \;\; k_4 = h f(x_i+h, y_i+k_3)",
            r"y_{i+1} = y_i + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)",
        ],
        "ejemplo_teorico": {
            "titulo": "Arranque multipaso y ponderación simpsoniana",
            "enunciado": "Cuando f(x, y) solo depende de x, la fórmula de RK4 se reduce exactamente a la Regla de Simpson 1/3 sobre [x_i, x_{i+1}].",
            "pasos": [
                "k2 y k3 promedian la pendiente en el punto medio x_i + h/2.",
                "Peso central total 4/6 y extremos 1/6 cada uno.",
            ],
            "resultado_final": "Error local O(h^5) y error global O(h^4).",
        },
        "practica_profesora": {
            "titulo": "TP7 Ej. 2: RK2 y RK4 en y' = 6x²/y, y(4) = 16 en [4.0, 4.3] (h = 0.1)",
            "enunciado": "Calcular los valores de y(4.1), y(4.2) e y(4.3) con RK2 y RK4 para el PVI y' = 6x²/y, y(4) = 16.",
            "datos_iniciales": {
                "f_str": "6*x**2 / y",
                "x0": 4.0,
                "y0": 16.0,
                "xf": 4.3,
                "h": 0.1,
                "y_exacta_str": "2*x*sqrt(x)",
            },
            "solucion_referencia": "Exacta y(4.3) = 17.83333956. RK2: 17.83352. RK4: 17.83333 (prácticamente exacta).",
        },
    },
    "milne": {
        "nombre": "Método Multipaso Predictor-Corrector de Milne",
        "familia": "edo",
        "icono": "🔮",
        "resumen_corto": "Reutiliza los 4 pasos previos mediante fórmula abierta predictora y fórmula cerrada correctora (Simpson).",
        "guru": {
            "que_resuelve": "Avanza a 4° orden con apenas 2 evaluaciones de f por paso (mucho más rápido por paso que RK4).",
            "intuicion_geometrica": "Mira el historial reciente de 4 pendientes pasadas para proyectar una extrapolación futura (Predictor) y luego ajusta esa predicción integrando con Simpson sobre los últimos 3 puntos (Corrector).",
            "cuando_usar": [
                "En intervalos largos cuando evaluar f(x, y) es costoso y ya se dispone de 4 puntos iniciales generados por RK4.",
            ],
            "cuando_falla": [
                "No es autoiniciable: requiere obligatoriamente n >= 4 subintervalos (los primeros 3 pasos los resuelve RK4).",
                "Puede sufrir inestabilidad numérica débil si el paso h no es lo bastante pequeño (oscilaciones parásitas en soluciones decrecientes).",
            ],
        },
        "formulas_latex": [
            r"P(y_{i+1}) = y_{i-3} + \frac{4h}{3}\left(2y'_i - y'_{i-1} + 2y'_{i-2}\right)",
            r"C(y_{i+1}) = y_{i-1} + \frac{h}{3}\left(y'_{i-1} + 4y'_i + f(x_{i+1}, y^{(m-1)}_{i+1})\right)",
        ],
        "ejemplo_teorico": {
            "titulo": "Esquema oficial de Doble Corrección con Arranque RK4",
            "enunciado": "La cátedra arranca con (y0, y1, y2, y3) calculados por RK4, aplica el predictor abierto de Milne y ejecuta dos iteraciones de corrección C y C¹.",
            "pasos": [
                "Verificar que hay al menos 4 pasos en el intervalo [x0, xf].",
                "Nota: en la variante 2026 se usa la fórmula estándar (2y'_i - y'_{i-1} + 2y'_{i-2}).",
            ],
            "resultado_final": "Logra precisión O(h⁴) comparable a RK4 con la mitad de evaluaciones por paso.",
        },
        "practica_profesora": {
            "titulo": "TP7 Ej. 3: Milne en y' = 6x²/y, y(4) = 16 en [4.0, 4.4] (h = 0.1)",
            "enunciado": "Tomando y(4.0..4.3) con RK4, calcular y(4.4) mediante Milne con doble corrección (y comparar con la exacta 18.459036).",
            "datos_iniciales": {
                "f_str": "6*x**2 / y",
                "x0": 4.0,
                "y0": 16.0,
                "xf": 4.4,
                "h": 0.1,
                "y_exacta_str": "2*x*sqrt(x)",
            },
            "solucion_referencia": "P(y4) = 18.45903 → C(y4) = 18.45902 → C¹(y4) = 18.45902 (exacta = 18.45903573).",
        },
    },
}


# ==============================================================================
# FICHAS TÉCNICAS Y REGLAS PARA ARENA VERSUS (Estilo Versus.com)
# ==============================================================================

FICHAS_VERSUS: dict[str, dict[str, Any]] = {
    "raices": {
        "titulo": "📉 Raíces de Ecuaciones No Lineales",
        "descripcion": "Enfrenta métodos cerrados (garantizados) contra métodos abiertos (rápidos) sobre una misma ecuación f(x) = 0.",
        "metodos": {
            "biseccion": {
                "nombre": "Bisección (Intervalo Medio)",
                "orden": "Lineal (Orden 1, razón 0.5)",
                "costo": "1 evaluación de f(x) por paso",
                "estabilidad": "Máxima (siempre converge si hay Bolzano)",
                "requisitos": "Continuidad y cambio de signo f(a)·f(b) < 0",
                "pros": ["Imposible que diverja dentro del intervalo", "Número exacto de pasos predecible a priori", "No requiere derivadas"],
                "contras": ["Lento (reducción fija a la mitad)", "No detecta raíces tangentes sin cambio de signo"],
            },
            "regula_falsi": {
                "nombre": "Interpolación Lineal (Regula Falsi)",
                "orden": "Lineal / Superlineal (~1.0 - 1.6)",
                "costo": "1 evaluación nueva de f(x) por paso",
                "estabilidad": "Alta (encierra siempre la raíz)",
                "requisitos": "Monotonía en (a, b) y f(a)·f(b) < 0",
                "pros": ["Aprovecha la altura de f(a) y f(b)", "Generalmente más rápido que Bisección en funciones poco curvas"],
                "contras": ["Si un extremo queda fijo (pivote), puede volverse mucho más lento que Bisección"],
            },
            "newton_raphson": {
                "nombre": "Newton-Raphson (Tangentes)",
                "orden": "Cuadrático (Orden 2)",
                "costo": "2 evaluaciones por paso: f(x) y f'(x)",
                "estabilidad": "Condicional (depende de Fourier y x0)",
                "requisitos": "f derivable, f'(x) != 0, Condición de Fourier f(x0)·f''(x0) > 0",
                "pros": ["Convergencia explosiva: duplica dígitos exactos por paso", "Ideal para altísima precisión"],
                "contras": ["Puede diverger si f'(x) ≈ 0 o hay puntos de inflexión", "Exige derivada analítica"],
            },
            "aitken": {
                "nombre": "Punto Fijo + Aceleración Aitken",
                "orden": "Lineal acelerado (~Superlineal)",
                "costo": "3 pasos por ciclo completo",
                "estabilidad": "Condicional a |g'(x)| < 1",
                "requisitos": "Despeje contractivo x = g(x) con |g'(x)| < 1",
                "pros": ["Acelera drásticamente una iteración lenta sin derivar", "Útil cuando ya se tiene x = g(x)"],
                "contras": ["Diverge si |g'(x)| >= 1", "No es universal: depende de cómo se despeje g(x)"],
            },
        },
        "expectativas": {
            ("biseccion", "newton_raphson"): {
                "que_esperamos": "En condiciones normales (cumpliendo las Condiciones de Fourier), **Newton-Raphson** converge cuadráticamente en muchísimas menos iteraciones (ej. 3 a 5) que **Bisección** (que avanza rígidamente partiendo a la mitad).",
                "excepciones": "Si x0 cae cerca de un mínimo/máximo local donde f'(x) ≈ 0, o si hay un punto de inflexión entre x0 y la raíz, Newton-Raphson puede oscilar o dispararse al infinito mientras Bisección sigue avanzando sin fallar.",
            },
            ("biseccion", "regula_falsi"): {
                "que_esperamos": "En funciones casi rectas o suavemente monótonas, **Interpolación Lineal (Regula Falsi)** suele llegar en menos iteraciones que **Bisección** porque la secante apunta cerca de la raíz.",
                "excepciones": "En funciones con fuerte curvatura (como x¹⁰ - 1 o exponenciales pronunciadas), Regula Falsi deja un extremo estancado y su paso se achica muy lento, siendo superado claramente por Bisección.",
            },
            ("newton_raphson", "regula_falsi"): {
                "que_esperamos": "**Newton-Raphson** (orden 2) suele vencer ampliamente a **Regula Falsi** en cantidad de iteraciones y error final.",
                "excepciones": "Cuando f'(x) no se conoce fácilmente, es discontinua o muy cercana a cero cerca de la raíz, Regula Falsi es más seguro al mantener el intervalo acotado.",
            },
            ("aitken", "biseccion"): {
                "que_esperamos": "Con un despeje contractivo g(x) bien elegido (|g'(x)| pequeño), **Punto Fijo + Aitken** logra saltar muy cerca de la raíz en pocos ciclos frente a la progresión geométrica fija de **Bisección**.",
                "excepciones": "Si |g'(x)| >= 1 en el entorno de x0, Punto Fijo diverge de inmediato y Aitken no logra estabilizarlo; Bisección no sufre esa limitación.",
            },
            ("aitken", "newton_raphson"): {
                "que_esperamos": "**Newton-Raphson** mantiene ventaja al tener convergencia cuadrática natural sin necesitar transformar f(x) en g(x).",
                "excepciones": "Si g'(x) es extremadamente chico cerca de la raíz y f'(x) es costosa de evaluar, Aitken puede competir muy de cerca con menos esfuerzo algebraico.",
            },
            ("aitken", "regula_falsi"): {
                "que_esperamos": "Ambos dependen de la curvatura/pendiente local; si |g'(x)| < 0.3, **Aitken** suele cerrar rápido en 2–3 ciclos; si Regula Falsi no queda anclado, avanza directo con la secante.",
                "excepciones": "Aitken diverge si falla la contractividad de g(x), mientras Regula Falsi mantiene la raíz encerrada.",
            },
        },
    },
    "sistemas": {
        "titulo": "📐 Sistemas de Ecuaciones Lineales (Ax = b)",
        "descripcion": "Compara métodos directos (Gauss, Gauss-Jordan, LU) contra métodos iterativos (Gauss-Seidel).",
        "metodos": {
            "gauss": {
                "nombre": "Eliminación de Gauss (Pivoteo Parcial)",
                "orden": "Directo (finito)",
                "costo": "~ (1/3) n³ operaciones",
                "estabilidad": "Alta con pivoteo parcial",
                "requisitos": "det(A) != 0 (matriz no singular)",
                "pros": ["El método directo más eficiente para un solo vector b", "Pivoteo parcial evita ceros y errores grandes"],
                "contras": ["Si cambia b hay que rehacer toda la eliminación", "Costoso en sistemas dispersos de miles de ecuaciones"],
            },
            "gauss_jordan": {
                "nombre": "Método de Gauss-Jordan",
                "orden": "Directo (finito)",
                "costo": "~ (1/2) n³ operaciones",
                "estabilidad": "Alta con pivoteo parcial",
                "requisitos": "det(A) != 0",
                "pros": ["Entrega la solución directa sin sustitución atrás", "Permite obtener A⁻¹ en el mismo proceso"],
                "contras": ["Realiza ~50% más operaciones aritméticas que Gauss para un solo sistema"],
            },
            "lu": {
                "nombre": "Factorización LU (Doolittle)",
                "orden": "Directo (finito)",
                "costo": "~ (1/3) n³ factorizar + O(n²) por cada nuevo b",
                "estabilidad": "Media (sin pivoteo exige pivotes != 0)",
                "requisitos": "Menores principales no nulos (U_kk != 0)",
                "pros": ["Insuperable cuando se resuelven múltiples vectores b con la misma matriz A"],
                "contras": ["Sin pivoteo, falla si un pivote diagonal intermedio resulta cero"],
            },
            "gauss_seidel": {
                "nombre": "Gauss-Seidel (Iterativo)",
                "orden": "Iterativo (Lineal según radio espectral)",
                "costo": "O(n²) por iteración (O(nnz) en dispersas)",
                "estabilidad": "Sujeta a Dominancia Diagonal",
                "requisitos": "a_ii != 0 y idealmente Dominancia Diagonal Estricta",
                "pros": ["No modifica la matriz A (ideal para matrices gigantes dispersas)", "Permite controlar precisión con tolerancia E"],
                "contras": ["Puede diverger si falta dominancia diagonal", "No entrega solución exacta en un paso fijo"],
            },
        },
        "expectativas": {
            ("gauss", "gauss_jordan"): {
                "que_esperamos": "Ambos llegan exactamente a la misma solución, pero **Eliminación de Gauss** realiza menos operaciones de punto flotante que **Gauss-Jordan**.",
                "excepciones": "Si además de x necesitamos calcular A⁻¹ o resolver simultáneamente varios vectores independientes, Gauss-Jordan resulta más cómodo.",
            },
            ("gauss", "gauss_seidel"): {
                "que_esperamos": "En sistemas pequeños (3x3, 4x4), **Gauss** entrega la solución exacta directamente; en matrices diagonalmente dominantes, **Gauss-Seidel** converge rápidamente reduciendo el error exponencialmente en escala logarítmica.",
                "excepciones": "Si se desordena la matriz perdiendo la dominancia diagonal, Gauss-Seidel diverge al infinito mientras Eliminación de Gauss con pivoteo resuelve sin problema.",
            },
            ("gauss", "lu"): {
                "que_esperamos": "La matriz triangular U de Doolittle coincide exactamente con la escalonada de **Gauss** (sin pivoteo), y ambos obtienen igual resultado en la primera ejecución.",
                "excepciones": "Si un pivote diagonal es 0, Gauss estándar con pivoteo intercambia filas y resuelve; Doolittle puro sin permutaciones se detiene por división por cero.",
            },
            ("gauss_jordan", "gauss_seidel"): {
                "que_esperamos": "**Gauss-Jordan** garantiza el vector exacto en n pasos de columna, mientras **Gauss-Seidel** aproxima iterativamente según la tolerancia elegida.",
                "excepciones": "En matrices mal condicionadas o sin dominancia diagonal, Gauss-Seidel no converge; pero en sistemas enormes y ralos, Gauss-Jordan agota la memoria.",
            },
            ("gauss_jordan", "lu"): {
                "que_esperamos": "**Factorización LU** cuesta aproximadamente un 33% menos de operaciones que **Gauss-Jordan** y permite reutilizar A.",
                "excepciones": "Si el objetivo es hallar explícitamente la inversa A⁻¹, Gauss-Jordan sobre [A|I] es directo.",
            },
            ("gauss_seidel", "lu"): {
                "que_esperamos": "**LU** encuentra la solución exacta en tiempo fijo sin depender del vector inicial X0; **Gauss-Seidel** depende de X0 y de la dominancia diagonal.",
                "excepciones": "Para sistemas masivos ralos (como mallas de diferencias finitas), LU llena de ceros no nulos la matriz (fill-in) mientras Gauss-Seidel solo opera sobre los valores no nulos.",
            },
        },
    },
    "autovalores": {
        "titulo": "⚡ Autovalores y Espectro Matricial",
        "descripcion": "Compara la búsqueda iterativa de extremos (Potencias) contra la extracción exacta del polinomio característico (Faddeev-Leverrier).",
        "metodos": {
            "potencias_max": {
                "nombre": "Potencias Directas (λ_max)",
                "orden": "Iterativo (depende de |λ2 / λ1|)",
                "costo": "O(n²) por iteración",
                "estabilidad": "Alta para el autovalor dominante",
                "requisitos": "|λ1| > |λ2| y X0 no ortogonal al autovector dominante",
                "pros": ["Muy rápido y escalable para matrices grandes", "Da directo λ_max y su autovector sin factorizar"],
                "contras": ["Solo encuentra UN autovalor (el mayor en módulo)", "Lento si |λ1| y |λ2| son muy cercanos"],
            },
            "potencias_min": {
                "nombre": "Potencias Inversas (λ_min)",
                "orden": "Iterativo sobre A⁻¹",
                "costo": "Requiere A⁻¹ + O(n²) por iteración",
                "estabilidad": "Sensible a cercanía de λ_min a 0",
                "requisitos": "det(A) != 0 (A invertible)",
                "pros": ["Extrae el modo propio menor (λ_min) crítico en vibraciones y pandeo"],
                "contras": ["Exige invertir o factorizar A previamente"],
            },
            "faddeev": {
                "nombre": "Faddeev-Leverrier (Espectro Total)",
                "orden": "Directo exacto (n pasos matriciales)",
                "costo": "O(n⁴) — muy costoso si n crece",
                "estabilidad": "Buena para n <= 6; sensible si n es grande",
                "requisitos": "Matriz cuadrada n×n",
                "pros": ["Obtiene TODOS los autovalores, el polinomio característico P(λ), det(A) y A⁻¹ en un solo algoritmo"],
                "contras": ["Costoso computacionalmente (n multiplicaciones de matrices n×n)", "Las raíces de polinomios de grado alto son sensibles al redondeo"],
            },
        },
        "expectativas": {
            ("faddeev", "potencias_max"): {
                "que_esperamos": "**Faddeev-Leverrier** revela el espectro completo (todos los λ y el polinomio característico), mientras **Potencias Directas** converge rápidamente al autovalor dominante λ_max coincidiendo con la raíz de mayor módulo de Faddeev.",
                "excepciones": "Si la matriz tiene dos autovalores dominantes opuestos o muy cercanos (|λ1| ≈ |λ2|) o complejos conjugados, el método de las Potencias oscila o no converge, mientras Faddeev sigue entregando el polinomio exacto.",
            },
            ("faddeev", "potencias_min"): {
                "que_esperamos": "**Potencias Inversas** converge hacia el autovalor de menor módulo que aparece más próximo a λ = 0 en la curva de **Faddeev-Leverrier**.",
                "excepciones": "Si la matriz es singular (λ = 0 es autovalor, b_n = 0), Potencias Inversas no puede invertir A, mientras Faddeev-Leverrier lo detecta directamente viendo b_n = 0.",
            },
            ("potencias_max", "potencias_min"): {
                "que_esperamos": "**Potencias Directas** captura el extremo superior del espectro (|λ_max|) y **Potencias Inversas** captura el extremo inferior (|λ_min|), acotando el rango espectral y el número de condición κ = |λ_max / λ_min|.",
                "excepciones": "Si el vector inicial X0 es ortogonal al autovector dominante o mínimo, cualquiera de los dos puede converger al segundo autovalor en lugar del extremo real.",
            },
        },
    },
    "interpolacion": {
        "titulo": "📈 Interpolación Polinomial",
        "descripcion": "Compara diferencias finitas equidistantes (Newton-Gregory) frente a polinomios para nodos arbitrarios (Lagrange).",
        "metodos": {
            "lagrange": {
                "nombre": "Polinomio de Lagrange",
                "orden": "Exacto de grado <= n-1",
                "costo": "O(n²) por evaluación",
                "estabilidad": "Apto para cualquier espaciamiento",
                "requisitos": "Abscisas x_i todas distintas entre sí",
                "pros": ["Funciona perfectamente con nodos NO equidistantes (h variable)", "Fórmula explícita simétrica y directa"],
                "contras": ["Si agregas un punto nuevo hay que recalcular todos los L_i(x)"],
            },
            "gregory_asc": {
                "nombre": "Newton-Gregory Ascendente",
                "orden": "Exacto de grado <= n-1",
                "costo": "Tabla O(n²) + evaluación O(n)",
                "estabilidad": "Excelente al inicio de la tabla (cerca de x0)",
                "requisitos": "Nodos estrictamente EQUIDISTANTES (h constante)",
                "pros": ["Tabla de diferencias fácil de auditar y reutilizar si se agrega un nodo al final", "Permite detectar el grado polinómico de los datos"],
                "contras": ["NO es válido si un nodo pierde equidistancia"],
            },
            "gregory_desc": {
                "nombre": "Newton-Gregory Descendente",
                "orden": "Exacto de grado <= n-1",
                "costo": "Tabla O(n²) + evaluación O(n)",
                "estabilidad": "Excelente al final de la tabla (cerca de x_n)",
                "requisitos": "Nodos estrictamente EQUIDISTANTES (h constante)",
                "pros": ["Óptimo cerca del extremo derecho de la tabla (datos más recientes)"],
                "contras": ["NO es válido si los x_i tienen espaciamiento irregular"],
            },
        },
        "expectativas": {
            ("gregory_asc", "lagrange"): {
                "que_esperamos": "Por el Teorema de Unicidad del Polinomio Interpolador, sobre nodos equidistantes **Newton-Gregory Ascendente** y **Lagrange** producen EXACTAMENTE el mismo valor y la misma curva.",
                "excepciones": "Si modificas un solo nodo x_i rompiendo la equidistancia (h variable), Newton-Gregory Ascendente deja de ser matemáticamente válido mientras Lagrange sigue siendo exacto. Además, para >10 puntos equiespaciados ambos sufren el Fenómeno de Runge en los extremos.",
            },
            ("gregory_desc", "lagrange"): {
                "que_esperamos": "Sobre una tabla equiespaciada producen el mismo polinomio único P(x) y valor interpolado idéntico.",
                "excepciones": "Con nodos no equidistantes, solo Lagrange es aplicable directamente; Newton-Gregory Descendente exige paso h constante.",
            },
            ("gregory_asc", "gregory_desc"): {
                "que_esperamos": "Ambos generan algebraicamente el mismo polinomio cuando se utilizan todos los nodos de la tabla.",
                "excepciones": "Si se trunca el grado del polinomio (usando menos diferencias que n-1), **Ascendente** es mucho más preciso cerca de x0 y **Descendente** es mucho más preciso cerca de x_n.",
            },
        },
    },
    "integracion": {
        "titulo": "📊 Integración Numérica (Cuadratura)",
        "descripcion": "Compara reglas lineales (Trapecio) contra reglas parabólicas/cúbicas (Simpson 1/3, 3/8 y Adaptativo) al variar Δh.",
        "metodos": {
            "trapecio": {
                "nombre": "Regla del Trapecio Compuesto",
                "orden": "O(h²) — Exacto hasta grado 1",
                "costo": "n+1 evaluaciones, sin restricciones de paridad",
                "estabilidad": "Muy alta incluso en funciones rugosas o con picos",
                "requisitos": "n >= 1 subintervalos",
                "pros": ["Funciona para CUALQUIER cantidad de intervalos n (par o impar)", "Muy simple y robusto frente a datos ruidosos"],
                "contras": ["Error O(h²) significativamente mayor que Simpson en curvas suaves"],
            },
            "simpson_1_3": {
                "nombre": "Simpson 1/3 (y Adaptativo ante Δh)",
                "orden": "O(h⁴) — Exacto hasta grado 3",
                "costo": "Igual número de nodos que Trapecio, mucha mayor precisión",
                "estabilidad": "Excelente en funciones suaves C⁴",
                "requisitos": "n PAR de intervalos (si cambia a impar requiere adaptación 3/8 o Trapecio)",
                "pros": ["Orden O(h⁴): reduce el error 16 veces al dividir h a la mitad", "Exacta incluso para polinomios cúbicos"],
                "contras": ["No se puede aplicar puro si n es impar o si h cambia en medio de la tabla"],
            },
            "simpson_3_8": {
                "nombre": "Simpson 3/8 / Combinado",
                "orden": "O(h⁴) — Exacto hasta grado 3",
                "costo": "Bloques de 3 subintervalos + resto",
                "estabilidad": "Excelente en mallas impares",
                "requisitos": "n múltiplo de 3 (puro) o n >= 5 impar (combinado con 1/3)",
                "pros": ["Permite resolver particiones impares manteniendo orden O(h⁴)"],
                "contras": ["Ligeramente menos compacto que 1/3 puro cuando n ya es par"],
            },
        },
        "expectativas": {
            ("simpson_1_3", "trapecio"): {
                "que_esperamos": "Para cualquier curva suave (polinomios de grado >= 2, exponenciales, trigonométricas), **Simpson 1/3** (orden O(h⁴)) logra un error órdenes de magnitud menor que **Trapecio** (orden O(h²)) usando exactamente los mismos puntos.",
                "excepciones": "1) Si el paso Δh cambia dejando n impar, Simpson 1/3 puro no aplica y debe combinarse con Simpson 3/8 o Trapecio. 2) En funciones lineales a trozos, rectas puras o funciones con picos/discontinuidades en derivadas, Trapecio puede igualar o incluso superar a Simpson (que oscila al intentar ajustar parábolas sobre esquinas).",
            },
            ("simpson_3_8", "trapecio"): {
                "que_esperamos": "**Simpson 3/8 (o Combinado)** supera ampliamente a **Trapecio** en precisión (O(h⁴) vs. O(h²)) manteniendo los mismos datos.",
                "excepciones": "Con n=1 o n=2 subintervalos sueltos no alcanza para un bloque de 3/8 (4 puntos), por lo que Trapecio o 1/3 toman la posta.",
            },
            ("simpson_1_3", "simpson_3_8"): {
                "que_esperamos": "Ambos poseen orden teórico O(h⁴) y suelen dar precisiones muy similares (1/3 tiene una constante de error levemente menor: 1/180 vs 1/80 cuando se comparan a igual n).",
                "excepciones": "Si n es impar (ej. n=5), Simpson 1/3 puro falla y necesita del aporte de un bloque de 3/8 para cubrir los 3 primeros paneles.",
            },
        },
    },
    "edo": {
        "titulo": "⏱️ Ecuaciones Diferenciales Ordinarias (PVI)",
        "descripcion": "Compara métodos monopaso (Euler, Heun, RK2, RK4) frente al esquema multipaso de Milne al variar el paso h.",
        "metodos": {
            "euler": {
                "nombre": "Método de Euler Clásico",
                "orden": "O(h) — Orden 1",
                "costo": "1 evaluación de f(x, y) por paso",
                "estabilidad": "Limitada (deriva rápida si h no es minúsculo)",
                "requisitos": "Condición inicial (x0, y0) y paso h",
                "pros": ["Extremadamente simple y fácil de implementar"],
                "contras": ["Muy impreciso (error O(h)); requiere h diminuto"],
            },
            "heun": {
                "nombre": "Euler Modificado (Heun Iterado)",
                "orden": "O(h²) — Orden 2",
                "costo": "2+ evaluaciones por paso (según corrector)",
                "estabilidad": "Buena",
                "requisitos": "(x0, y0), h y tolerancia del corrector",
                "pros": ["Salto enorme de precisión frente a Euler con bajo costo"],
                "contras": ["Inferior en orden frente a RK4 o Milne"],
            },
            "rk2": {
                "nombre": "Runge-Kutta Orden 2 (RK2)",
                "orden": "O(h²) — Orden 2",
                "costo": "2 evaluaciones de f(x, y) por paso",
                "estabilidad": "Buena",
                "requisitos": "(x0, y0) y paso h",
                "pros": ["Autoiniciable, buena relación costo/precisión"],
                "contras": ["Menor orden que RK4 cuando se exige alta precisión"],
            },
            "rk4": {
                "nombre": "Runge-Kutta Clásico (RK4)",
                "orden": "O(h⁴) — Orden 4",
                "costo": "4 evaluaciones de f(x, y) por paso",
                "estabilidad": "Muy alta en EDOs no rígidas",
                "requisitos": "(x0, y0) y paso h",
                "pros": ["Altísima precisión O(h⁴) desde el primer paso", "Autoiniciable"],
                "contras": ["4 evaluaciones de la función por cada paso avanzado"],
            },
            "milne": {
                "nombre": "Predictor-Corrector de Milne",
                "orden": "O(h⁴) — Orden 4 (Multipaso)",
                "costo": "~2-3 evaluaciones por paso (tras el arranque)",
                "estabilidad": "Puede presentar inestabilidad débil para h grande",
                "requisitos": "Mínimo 4 pasos (n >= 4) con arranque previo por RK4",
                "pros": ["Misma precisión O(h⁴) que RK4 con casi la mitad de evaluaciones por paso en tramos largos", "Permite estimar el error en línea comparando P vs C"],
                "contras": ["No es autoiniciable", "Falla si h produce n < 4 subintervalos"],
            },
        },
        "expectativas": {
            ("euler", "rk4"): {
                "que_esperamos": "**RK4** (orden O(h⁴)) sigue fielmente la solución exacta casi sin error apreciable incluso con pasos h moderados, mientras **Euler** (orden O(h)) se despega visiblemente formando una poligonal tangencial rezagada.",
                "excepciones": "Si f(x, y) es constante (y' = cte, recta), Euler ya es exacto en 1 paso sin que RK4 aporte ventaja adicional; en ecuaciones rígidas ambos exigen pasos suficientemente chicos.",
            },
            ("euler", "heun"): {
                "que_esperamos": "**Euler Modificado (Heun)** reduce el error en órdenes de magnitud frente a **Euler Clásico** promediando las pendientes de inicio y fin de paso.",
                "excepciones": "En pasos h muy grandes en EDO explosivas (ej. y' = x² + y² cerca de asíntotas), tanto Euler como Heun pueden diverger si el paso excede el radio de estabilidad.",
            },
            ("milne", "rk4"): {
                "que_esperamos": "Ambos alcanzan precisión de 4° orden O(h⁴) prácticamente indistinguible frente a la curva exacta, pero **Milne** requiere menos evaluaciones de f(x, y) por paso una vez arrancado.",
                "excepciones": "1) Si el paso Δh es grande y el intervalo tiene menos de 4 subintervalos, Milne ni siquiera puede arrancar (RK4 sí funciona desde 1 paso). 2) En EDO con solución decreciente rápida (ej. y' = -λy), Milne puede sufrir inestabilidad oscilatoria parásita si h no es suficientemente pequeño.",
            },
            ("heun", "rk4"): {
                "que_esperamos": "**RK4** (orden 4) supera claramente en precisión a **Heun** (orden 2) para el mismo paso h.",
                "excepciones": "Si Heun re-itera el corrector varias veces y el paso h es muy pequeño, el error puede ser comparable a costa de mayor número de evaluaciones.",
            },
            ("rk2", "rk4"): {
                "que_esperamos": "**RK4** ofrece error O(h⁴) frente a O(h²) de **RK2**, reduciendo drásticamente la desviación acumulada al final del intervalo.",
                "excepciones": "Cuando la evaluación de f(x, y) es extremadamente costosa y solo se requiere 1-2% de precisión, RK2 usa la mitad de evaluaciones por paso.",
            },
        },
    },
}


def obtener_metodo(clave: str) -> dict[str, Any]:
    """Retorna la ficha completa de un método del catálogo o lanza KeyError claro."""
    if clave not in CATALOGO_METODOS:
        raise KeyError(f"Método '{clave}' no encontrado en CATALOGO_METODOS.")
    return CATALOGO_METODOS[clave]


def obtener_expectativa_versus(familia: str, m_a: str, m_b: str) -> dict[str, str]:
    """Busca la ficha teórica de expectativa y excepciones entre Método A y Método B."""
    ficha = FICHAS_VERSUS.get(familia, {})
    exp_map = ficha.get("expectativas", {})
    par = tuple(sorted((m_a, m_b)))
    if par in exp_map:
        return exp_map[par]
    if (m_a, m_b) in exp_map:
        return exp_map[(m_a, m_b)]
    if (m_b, m_a) in exp_map:
        return exp_map[(m_b, m_a)]
    return {
        "que_esperamos": "Se comparan dos algoritmos pertenecientes a la misma familia matemática según su orden de convergencia teórico y estabilidad numérica.",
        "excepciones": "Verifique los criterios de validez individuales (derivadas no nulas, dominancia diagonal, equidistancia de nodos o tamaño de paso h) para detectar posibles casos patológicos.",
    }

