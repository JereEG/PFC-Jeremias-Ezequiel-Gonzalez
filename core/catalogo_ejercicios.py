r"""
core/catalogo_ejercicios.py
===========================
Catálogo pedagógico de métodos numéricos, precomputaciones estáticas y fichas
comparativas para "Arandú: Métodos Computacionales (v3.2)".

Diseñado para estudiantes de Sistemas bajo arquitectura de lectura en 3 niveles
(Resumen Ejecutivo, Intuición/Algoritmo + Profundización opcional, Veredicto Práctico),
traducción de jerga matemática a lógica computacional y precomputación total de
expresiones de cátedra para evitar sobrecarga de SymPy en tiempo real.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ==============================================================================
# PRECOMPUTACIONES ESTÁTICAS (ZERO SYMPY EN EJERCICIOS DE CÁTEDRA)
# ==============================================================================

EXPRESIONES_PRECOMPUTADAS_RAICES: dict[str, dict[str, Any]] = {
    "-2.75*x**3 + 18*x**2 - 21*x - 12": {
        "f_latex": r"- 2.75 x^{3} + 18 x^{2} - 21 x - 12",
        "d1_latex": r"- 8.25 x^{2} + 36 x - 21",
        "d2_latex": r"36 - 16.5 x",
        "fn_np": lambda x: -2.75 * (x**3) + 18.0 * (x**2) - 21.0 * x - 12.0,
        "d1_np": lambda x: -8.25 * (x**2) + 36.0 * x - 21.0,
        "d2_np": lambda x: 36.0 - 16.5 * x,
        "descartes": {
            "es_polinomio": True,
            "grado": 3,
            "f_pos_latex": r"- 2.75 x^{3} + 18 x^{2} - 21 x - 12",
            "f_neg_latex": r"2.75 x^{3} + 18 x^{2} + 21 x - 12",
            "coefs_pos": [-2.75, 18.0, -21.0, -12.0],
            "signos_pos": ["-", "+", "-", "-"],
            "variaciones_pos": 2,
            "posibles_raices_pos": [2, 0],
            "coefs_neg": [2.75, 18.0, 21.0, -12.0],
            "signos_neg": ["+", "+", "+", "-"],
            "variaciones_neg": 1,
            "posibles_raices_neg": [1],
            "max_raices_positivas": 2,
            "max_raices_negativas": 1,
        },
    },
    "2*x**4 - 6*x**2 + 7*x - 2": {
        "f_latex": r"2 x^{4} - 6 x^{2} + 7 x - 2",
        "d1_latex": r"8 x^{3} - 12 x + 7",
        "d2_latex": r"24 x^{2} - 12",
        "fn_np": lambda x: 2.0 * (x**4) - 6.0 * (x**2) + 7.0 * x - 2.0,
        "d1_np": lambda x: 8.0 * (x**3) - 12.0 * x + 7.0,
        "d2_np": lambda x: 24.0 * (x**2) - 12.0,
        "descartes": {
            "es_polinomio": True,
            "grado": 4,
            "f_pos_latex": r"2 x^{4} - 6 x^{2} + 7 x - 2",
            "f_neg_latex": r"2 x^{4} - 6 x^{2} - 7 x - 2",
            "coefs_pos": [2.0, -6.0, 7.0, -2.0],
            "signos_pos": ["+", "-", "+", "-"],
            "variaciones_pos": 3,
            "posibles_raices_pos": [3, 1],
            "coefs_neg": [2.0, -6.0, -7.0, -2.0],
            "signos_neg": ["+", "-", "-", "-"],
            "variaciones_neg": 1,
            "posibles_raices_neg": [1],
            "max_raices_positivas": 3,
            "max_raices_negativas": 1,
        },
    },
    "x**3 - 2*x - 5": {
        "f_latex": r"x^{3} - 2 x - 5",
        "d1_latex": r"3 x^{2} - 2",
        "d2_latex": r"6 x",
        "fn_np": lambda x: (x**3) - 2.0 * x - 5.0,
        "d1_np": lambda x: 3.0 * (x**2) - 2.0,
        "d2_np": lambda x: 6.0 * x,
        "descartes": {
            "es_polinomio": True,
            "grado": 3,
            "f_pos_latex": r"x^{3} - 2 x - 5",
            "f_neg_latex": r"- x^{3} + 2 x - 5",
            "coefs_pos": [1.0, -2.0, -5.0],
            "signos_pos": ["+", "-", "-"],
            "variaciones_pos": 1,
            "posibles_raices_pos": [1],
            "coefs_neg": [-1.0, 2.0, -5.0],
            "signos_neg": ["-", "+", "-"],
            "variaciones_neg": 2,
            "posibles_raices_neg": [2, 0],
            "max_raices_positivas": 1,
            "max_raices_negativas": 2,
        },
    },
    "exp(x) - x**2 - 5": {
        "f_latex": r"- x^{2} + e^{x} - 5",
        "d1_latex": r"- 2 x + e^{x}",
        "d2_latex": r"e^{x} - 2",
        "fn_np": lambda x: np.exp(x) - (x**2) - 5.0,
        "d1_np": lambda x: np.exp(x) - 2.0 * x,
        "d2_np": lambda x: np.exp(x) - 2.0,
        "descartes": {
            "es_polinomio": False,
            "mensaje": "La función no es polinómica: el conteo de cambios de signo (Regla de Descartes) aplica únicamente a polinomios.",
        },
    },
}

EXPRESIONES_PRECOMPUTADAS_PUNTO_FIJO: dict[str, dict[str, Any]] = {
    "log(x**2 + 5)": {
        "g_latex": r"\log{\left(x^{2} + 5 \right)}",
        "dg_latex": r"\frac{2 x}{x^{2} + 5}",
        "gn_np": lambda x: np.log((x**2) + 5.0),
        "dgn_np": lambda x: (2.0 * x) / ((x**2) + 5.0),
    },
    "(2*x + 5)**(1/3)": {
        "g_latex": r"\sqrt[3]{2 x + 5}",
        "dg_latex": r"\frac{2}{3 \left(2 x + 5\right)^{\frac{2}{3}}}",
        "gn_np": lambda x: np.power(2.0 * x + 5.0, 1.0 / 3.0),
        "dgn_np": lambda x: 2.0 / (3.0 * np.power(2.0 * x + 5.0, 2.0 / 3.0)),
    },
}

INTEGRALES_PRECOMPUTADAS: dict[str, dict[str, Any]] = {
    "x**2 / 5": {
        "f_latex": r"\frac{x^{2}}{5}",
        "fn_np": lambda x: (np.asarray(x, dtype=float) ** 2) / 5.0,
        "exact_eval": lambda a, b: ((float(b) ** 3) - (float(a) ** 3)) / 15.0,
    },
}

EDO_PRECOMPUTADAS: dict[str, dict[str, Any]] = {
    "f": {
        "6*x**2 / y": {
            "f_latex": r"\frac{6 x^{2}}{y}",
            "fn_np": lambda x, y: (6.0 * (x**2)) / y,
        },
    },
    "exacta": {
        "2*x*sqrt(x)": {
            "y_latex": r"2 x^{\frac{3}{2}}",
            "fn_np": lambda x: 2.0 * x * np.sqrt(x),
        },
    },
}

LAGRANGE_PRECOMPUTADOS: dict[tuple[tuple[float, ...], tuple[float, ...]], dict[str, Any]] = {
    ((100.0, 120.0, 200.0, 220.0), (1.01, 3.61, 15.5, 23.2)): {
        "latex_expandido": r"1.81458333333333 \cdot 10^{-5} x^{3} - 0.007435 x^{2} + 1.10519166666667 x - 53.305",
        "bases_latex": [
            r"- 4.16666666666667 \cdot 10^{-6} x^{3} + 0.00225 x^{2} - 0.393333333333333 x + 22.0",
            r"\left(0.01 x - 2.2\right) \left(0.0125 x - 2.5\right) \left(0.05 x - 5.0\right)",
            r"- 6.25 \cdot 10^{-6} x^{3} + 0.00275 x^{2} - 0.3775 x + 16.5",
            r"\left(0.00833333333333333 x - 0.833333333333333\right) \left(0.01 x - 1.2\right) \left(0.05 x - 10.0\right)",
        ],
    },
}

FADDEEV_PRECOMPUTADOS: dict[tuple[tuple[float, ...], ...], dict[str, Any]] = {
    ((3.0, 2.0, 4.0), (2.0, 0.0, 2.0), (4.0, 2.0, 3.0)): {
        "polinomio_texto": "-λ^3 + 6λ^2 + 15λ + 8 = 0",
        "b_coeficientes": [6.0, 15.0, 8.0],
        "determinante": 8.0,
        "autovalores": [8.0, -1.0, -1.0],
        "inversa": [
            [-0.5, 0.25, 0.5],
            [0.25, -0.875, 0.25],
            [0.5, 0.25, -0.5],
        ],
    },
}


# ==============================================================================
# CATÁLOGO PEDAGÓGICO DE MÉTODOS (ESTRUCTURA EN 3 NIVELES PARA SISTEMAS)
# ==============================================================================

CATALOGO_METODOS: dict[str, dict[str, Any]] = {
    # ==========================================================================
    # 1. FUNDAMENTOS Y ERRORES
    # ==========================================================================
    "series_taylor": {
        "nombre": "Series de Potencias y Corte por Tolerancia",
        "familia": "fundamentos",
        "icono": "🧮",
        "resumen_corto": (
            "Una CPU solo sabe sumar, restar, multiplicar y dividir. Las **Series de Taylor y Maclaurin** "
            "traducen funciones como $e^x$, $\\ln(1+x)$ o $\\pi$ a un bucle `for` que acumula potencias "
            "y factoriales hasta alcanzar la precisión deseada."
        ),
        "guru": {
            "que_resuelve": (
                "Calcula funciones matemáticas complejas usando únicamente las cuatro operaciones aritméticas básicas de la ALU."
            ),
            "intuicion_geometrica": (
                "Cada iteración del bucle suma un polinomio de mayor grado que pega mejor la aproximación a la curva real alrededor de un punto."
            ),
            "algoritmo_pasos": [
                "Se parte del término base ($k = 0$) y un acumulador `suma`.",
                "En cada paso del `for`, se calcula el nuevo término usando potencias y factoriales y se actualiza `suma`.",
                "**Criterio de parada para asegurar $n$ cifras decimales exactas (Criterio de Scarborough):** detenemos el ciclo cuando el cambio relativo $|E_{rp}|$ cae por debajo de $E_s = 0.5 \\times 10^{2-n}\\%$.",
            ],
            "profundizacion": [
                r"El error relativo porcentual aproximado entre iteraciones consecutivas se calcula como $E_{rp} = \left|\frac{S_k - S_{k-1}}{S_k}\right| \times 100\%$.",
                r"La serie de Maclaurin para $\ln(1+x) = \sum_{k=1}^{\infty} (-1)^{k-1}\frac{x^k}{k}$ converge solo para $x \in (-1, 1]$, mientras que $e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}$ converge para todo $x \in \mathbb{R}$.",
                r"La serie alternante de Gregory-Leibniz $\pi = 4\sum_{k=0}^{\infty} \frac{(-1)^k}{2k+1}$ acota su error por el primer término omitido $\frac{4}{2n+1}$, por lo que su costo es $\mathcal{O}(1/\varepsilon)$ iteraciones.",
            ],
            "cuando_usar": [
                "Cuando necesitás implementar funciones elementales sin librerías externas o controlar dinámicamente cuántas cifras exactas querés calcular.",
                "Para acotar el error de corte al truncar un bucle infinito a un número finito de pasos.",
            ],
            "cuando_falla": [
                "Si evaluás fuera de su dominio válido (por ejemplo $\\ln(1+x)$ para $|x| > 1$), la suma explota en lugar de converger.",
                "En series muy lentas como $\\pi$ por Gregory-Leibniz: conseguir $5$ decimales exactos requiere cerca de $200\\,000$ iteraciones.",
            ],
        },
        "formulas_latex": [
            r"e^x = \sum_{k=0}^{n-1} \frac{x^k}{k!}, \qquad \ln(1+x) = \sum_{k=1}^{n} (-1)^{k-1}\frac{x^k}{k}, \qquad \pi = 4\sum_{k=0}^{\infty} \frac{(-1)^k}{2k+1}",
            r"E_{rp} = \left|\frac{S_{\text{actual}} - S_{\text{anterior}}}{S_{\text{actual}}}\right| \times 100 < E_s = 0.5 \times 10^{2-n}\%",
        ],
        "ejemplo_teorico": {
            "titulo": "Aproximación de $e^{0.5}$ asegurando $3$ cifras significativas",
            "enunciado": "Estimar $e^{0.5}$ con un bucle acumulador garantizando al menos $n = 3$ cifras significativas exactas.",
            "pasos": [
                "Cota para $n = 3$ cifras (Criterio de Scarborough): $E_s = 0.5 \\times 10^{2-3}\\% = 0.05\\%$.",
                "Iteración 1: $T_1 = 1$",
                "Iteración 2: $T_2 = 1 + 0.5 = 1.5$ ($E_{rp} = 33.33\\%$)",
                "Iteración 3: $T_3 = 1.5 + \\frac{0.5^2}{2!} = 1.625$ ($E_{rp} = 7.69\\%$)",
                "Iteración 4: $T_4 = 1.625 + \\frac{0.5^3}{3!} = 1.645833\\dots$ ($E_{rp} = 1.26\\%$)",
                "Iteración 6: $T_6 \\approx 1.6486979$ cumple $|E_{rp}| < 0.05\\%$ y detiene el ciclo.",
            ],
            "resultado_final": "$e^{0.5} \\approx 1.6487$ con $3$ cifras significativas garantizadas.",
        },
        "practica_profesora": {
            "titulo": "Lab 1 (Ej. 5 y 6): Cálculo de $e^x$ y $\\pi$ con criterio de corte",
            "enunciado": "Simular el bucle de la serie para $e^x$ (por defecto $x = 1$, $N = 8$ términos) y la serie alternante de $\\pi$ con tolerancia $E_a < 0.5 \\times 10^{-n}$.",
            "datos_iniciales": {
                "x_euler": 1.0,
                "n_terminos_euler": 8,
                "cifras_scarborough": 3,
                "cifras_leibniz": 3,
            },
            "solucion_referencia": "Para $N = 8$ en $x = 1$: $e \\approx 2.71827877$ ($E_a = 3.0586 \\times 10^{-6}$). Para $\\pi$ con $n = 3$ ($E_a < 0.0005$): $2000$ términos, $\\pi \\approx 3.1410927$.",
        },
    },
    "aritmetica_maquina": {
        "nombre": "Aritmética de Punto Flotante ($k$ dígitos) y Asociatividad",
        "familia": "fundamentos",
        "icono": "💾",
        "resumen_corto": (
            "La memoria RAM tiene bits finitos y no puede guardar infinitos decimales. "
            "Modelar números con mantisa de $k$ dígitos explica por qué el **orden de tus sumas y restas** "
            "cambia el resultado final dentro de un programa."
        ),
        "guru": {
            "que_resuelve": (
                "Mide cuánto error se acumula al redondear o truncar cada operación intermedia en una computadora."
            ),
            "intuicion_geometrica": (
                "En lugar de una recta continua, la CPU trabaja sobre una regla con marcas fijas: cualquier número entre dos marcas se corta (truncamiento) o se mueve a la marca más cercana (redondeo)."
            ),
            "algoritmo_pasos": [
                "Todo número no nulo se normaliza en notación científica $\\pm 0.d_1 d_2 \\dots d_k \\times 10^n$.",
                "Al ejecutar cada `+`, `-`, `*` o `/`, el resultado vuelve a recortarse a exactamente $k$ dígitos de mantisa.",
                "Si sumás un número muy chico a uno gigante, los dígitos del chico caen fuera de los $k$ dígitos y desaparecen."
            ],
            "profundizacion": [
                r"La cota del error relativo de representación es $10^{-k+1}$ bajo truncamiento y $0.5 \times 10^{-k+1}$ bajo redondeo simétrico.",
                r"La propiedad asociativa $(A - B) - C = (A - C) - B$ deja de cumplirse cuando los resultados intermedios sufren ajustes distintos a $k$ dígitos.",
            ],
            "cuando_usar": [
                "Para diagnosticar diferencias de precisión al acumular miles de datos (regla práctica: acumulá primero los valores pequeños).",
            ],
            "cuando_falla": [
                "Al mezclar magnitudes muy dispares ($10^{12} + 10^{-5}$): el término pequeño es absorbido por completo y no modifica la variable acumulada.",
            ],
        },
        "formulas_latex": [
            r"fl(y) = \pm 0.d_1 d_2 \dots d_k \times 10^n, \qquad \left|\frac{y - fl(y)}{y}\right| \le \begin{cases} 10^{-k+1} & \text{(truncamiento)} \\ 0.5 \times 10^{-k+1} & \text{(redondeo)} \end{cases}",
            r"(A \ominus B) \ominus C \neq (A \ominus C) \ominus B \quad \text{en aritmética finita}",
        ],
        "ejemplo_teorico": {
            "titulo": "Operaciones con $x = 1/3$ e $y = 5/7$ usando $k = 5$ dígitos (Truncamiento)",
            "enunciado": "Representar $fl(x) = 0.33333 \\times 10^0$ y $fl(y) = 0.71428 \\times 10^0$ y evaluar $x + y$ e $y - x$.",
            "pasos": [
                "$x + y = fl(0.33333 + 0.71428) = fl(1.04761) = 0.10476 \\times 10^1$ ($E_r = 1.82 \\times 10^{-5}$).",
                "$y - x = fl(0.71428 - 0.33333) = fl(0.38095) = 0.38095 \\times 10^0$ ($E_r = 6.25 \\times 10^{-6}$).",
            ],
            "resultado_final": "El truncamiento mantiene el error relativo por debajo de $10^{-4}$, excepto si se restan términos casi idénticos.",
        },
        "practica_profesora": {
            "titulo": "TP2 Ej. 2 y 3: Pérdida de asociatividad con $k = 3$ dígitos",
            "enunciado": "Comparar $(121 - 119) - 0.327$ frente a $(121 - 0.327) - 119$ simulando registros de $k = 3$ dígitos.",
            "datos_iniciales": {
                "k_digitos": 3,
                "u_val": 121.0,
                "v_val": 119.0,
                "w_val": 0.327,
            },
            "solucion_referencia": "Exacto: $1.673$. Redondeo ($k = 3$): $(121 - 119) - 0.327 = 0.167 \\times 10^1$ ($E_r = 0.18\\%$); $(121 - 0.327) - 119 = 0.200 \\times 10^1$ ($E_r = 19.55\\%$). Truncamiento: $(121 - 0.327) - 119 = 0.100 \\times 10^1$ ($E_r = 40.23\\%$).",
        },
    },
    "cancelacion_cuadratica": {
        "nombre": "Estabilidad en Ecuaciones Cuadráticas (Evitar Restas Riesgosas)",
        "familia": "fundamentos",
        "icono": "💥",
        "resumen_corto": (
            "Al aplicar la resolvente clásica en una PC cuando $b^2 \\gg 4ac$, restar dos números casi idénticos "
            "borra los dígitos útiles. **Racionalizar el numerador** convierte la resta peligrosa en una suma segura."
        ),
        "guru": {
            "que_resuelve": (
                "Evita errores gigantes en la raíz pequeña de $ax^2 + bx + c = 0$ sin aumentar la precisión de los floats."
            ),
            "intuicion_geometrica": (
                "**Error de resta entre números casi iguales (Cancelación catastrófica):** al restar $-b + \\sqrt{b^2 - 4ac}$ en punto flotante, los dígitos iniciales idénticos se anulan entre sí y el resultado queda lleno de ruido de redondeo."
            ),
            "algoritmo_pasos": [
                "Calculamos $\\Delta = \\sqrt{b^2 - 4ac}$.",
                "Para la raíz donde el signo de $-b$ y $\\pm\\Delta$ tienen **igual signo**, sumamos directo con la fórmula clásica.",
                "Para la raíz donde se **restarían**, multiplicamos arriba y abajo por el conjugado para transformar la resta del numerador en suma en el denominador.",
            ],
            "profundizacion": [
                r"Multiplicando por $\frac{-b - \sqrt{b^2 - 4ac}}{-b - \sqrt{b^2 - 4ac}}$ cuando $b > 0$, obtenemos $x_1 = \frac{-2c}{b + \sqrt{b^2 - 4ac}}$, donde todos los términos del denominador tienen el mismo signo y no hay pérdida de dígitos.",
            ],
            "cuando_usar": [
                "En cualquier cálculo de raíces cuadráticas en producción cuando $b^2$ domina ampliamente a $|4ac|$.",
            ],
            "cuando_falla": [
                "No uses la versión racionalizada para la raíz que ya sumaba sus signos originalmente, porque volverías a introducir una resta en el denominador.",
            ],
        },
        "formulas_latex": [
            r"x_{1,2} = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \quad \xrightarrow{\text{Racionalización}} \quad x_1 = \frac{-2c}{b + \sqrt{b^2 - 4ac}} \;\; (b > 0)",
        ],
        "ejemplo_teorico": {
            "titulo": "Caso clásico de inestabilidad numérica ($x^2 + 3000.001x + 3 = 0$)",
            "enunciado": "Calcular ambas raíces con aritmética finita para $x^2 + 3000.001x + 3 = 0$.",
            "pasos": [
                "Raíces exactas: $x_1 = -0.001$ y $x_2 = -3000$.",
                "Con pocos dígitos en la fórmula tradicional, $-b + \\sqrt{b^2 - 4ac}$ cancela casi todos los dígitos líderes y distorsiona $x_1$.",
            ],
            "resultado_final": "Usar la fórmula racionalizada para $x_1$ y la tradicional (que suma dos negativos) para $x_2$.",
        },
        "practica_profesora": {
            "titulo": "TP2 Ej. 4: $x^2 + 62.10x + 1 = 0$ con redondeo a $k = 4$ dígitos",
            "enunciado": "Comparar la resolvente estándar frente a la racionalizada en $x^2 + 62.10x + 1 = 0$ simulando floats decimales de $k = 4$ dígitos.",
            "datos_iniciales": {
                "a": 1.0,
                "b": 62.10,
                "c": 1.0,
                "k_digitos": 4,
            },
            "solucion_referencia": "Exactas: $x_1 = -0.01610723$, $x_2 = -62.08390$. Estándar ($k=4$) da $x_1 = -0.02000$ ($E_r = 24.17\\%$); Racionalizada logra $x_1 = -0.01610$ ($E_r = 0.045\\%$).",
        },
    },
    "horner": {
        "nombre": "Esquema Anidado de Horner",
        "familia": "fundamentos",
        "icono": "🪆",
        "resumen_corto": (
            "Refactoriza un polinomio factorizando $x$ de forma anidada para evaluarlo en tiempo lineal $\\mathcal{O}(n)$. "
            "Reduce a la mitad las operaciones y minimiza la propagación de error de redondeo."
        ),
        "guru": {
            "que_resuelve": (
                "Evalúa polinomios en código sin calcular potencias caras ni acumular errores en cada exponente $x^k$."
            ),
            "intuicion_geometrica": (
                "En lugar de computar $x^2, x^3, \\dots, x^n$ por separado, va acumulando el resultado multiplicando por $x$ y sumando el siguiente coeficiente dentro de un único bucle lineal."
            ),
            "algoritmo_pasos": [
                "Inicializás una variable `res = a_n` (el coeficiente del término de mayor grado).",
                "Recorrés el resto de los coeficientes de mayor a menor en un `for`: `res = res * x + a_i`.",
                "Con apenas $n$ multiplicaciones y $n$ sumas obtenés el valor exacto del polinomio.",
            ],
            "profundizacion": [
                r"Evaluar un polinomio de grado $n$ de forma ingenua calculando potencias desde cero cuesta $\frac{n(n+1)}{2}$ multiplicaciones ($\mathcal{O}(n^2)$), mientras que Horner solo usa $n$ multiplicaciones y $n$ sumas ($\mathcal{O}(n)$).",
            ],
            "cuando_usar": [
                "Siempre que programes la evaluación de cualquier polinomio de grado $\\ge 2$.",
            ],
            "cuando_falla": [
                "No elimina la inestabilidad intrínseca de polinomios mal condicionados donde un cambio infinitesimal en los coeficientes mueve muchísimo las raíces (Polinomio de Wilkinson).",
            ],
        },
        "formulas_latex": [
            r"P(x) = a_n x^n + \dots + a_1 x + a_0 = \left(\dots\left((a_n x + a_{n-1})x + \dots\right)x + a_1\right)x + a_0",
        ],
        "ejemplo_teorico": {
            "titulo": "Conteo de instrucciones en grado $3$",
            "enunciado": "Comparar el costo en CPU entre evaluación directa y esquema anidado para un polinomio cúbico.",
            "pasos": [
                "Forma directa: calcula $x^2$, $x^3$ y multiplica cada término ($5$ multiplicaciones + $3$ sumas).",
                "Esquema de Horner: $((a_3 x + a_2)x + a_1)x + a_0$ ($3$ multiplicaciones, y solo $2$ si $a_3 = 1$).",
            ],
            "resultado_final": "Menos operaciones en punto flotante significan menos código, mayor velocidad y menor error acumulado.",
        },
        "practica_profesora": {
            "titulo": "TP2 Ej. 5/6: $f(x) = x^3 - 6x^2 + 3x - 0.149$ en $x = 4.71$ ($k = 3$ dígitos)",
            "enunciado": "Evaluar $f(4.71)$ con registros de $3$ dígitos (truncamiento y redondeo) en forma directa vs. forma anidada de Horner.",
            "datos_iniciales": {
                "coefs_str": "1, -6, 3, -0.149",
                "x_eval": 4.71,
                "k_digitos": 3,
            },
            "solucion_referencia": "Exacto: $-14.636489$. Estándar ($k=3$): $-14.0$ ($E_r = 4.35\\%$). Horner Redondeo ($k=3$): $-14.6$ ($E_r = 0.25\\%$). Horner Truncamiento: $-14.5$ ($E_r = 0.93\\%$).",
        },
    },

    # ==========================================================================
    # 2. RAÍCES DE ECUACIONES
    # ==========================================================================
    "biseccion": {
        "nombre": "Descartes, Tanteos y Bisección (Intervalo Medio)",
        "familia": "raices",
        "icono": "⚖️",
        "resumen_corto": (
            "Es el equivalente numérico de la **búsqueda binaria** sobre una función continua. "
            "Si los signos en $[a, b]$ son opuestos, divide el intervalo a la mitad en cada paso hasta encerrar la raíz."
        ),
        "guru": {
            "que_resuelve": (
                "Encuentra ceros de funciones reales $f(x) = 0$ garantizando que el algoritmo nunca diverja si arranca bien."
            ),
            "intuicion_geometrica": (
                "**Chequeo de arranque seguro (Teorema de Bolzano):** si $f(a)$ y $f(b)$ tienen signos opuestos ($f(a)\\cdot f(b) < 0$), la curva continua obligatoriamente cruza el eje $X$. Partimos al medio y descartamos la mitad que no tiene cruce."
            ),
            "algoritmo_pasos": [
                "Calculás el punto medio $x_r = \\frac{a + b}{2}$ y evaluás $f(x_r)$.",
                "Si $f(a) \\cdot f(x_r) < 0$, la raíz está en la mitad izquierda (`b = x_r`); caso contrario, en la derecha (`a = x_r`).",
                "Repite hasta que el ancho del intervalo o el error $|x_r - x_{\\text{ant}}|$ sea menor que la tolerancia $E$.",
            ],
            "profundizacion": [
                r"Opcionalmente, en polinomios podés contar los cambios de signo de $f(x)$ y $f(-x)$ para estimar cuántas raíces positivas y negativas existen antes de buscar (**Regla de los Signos de Descartes**).",
                r"Como el ancho $\frac{b-a}{2^n}$ se reduce exactamente a la mitad en cada ciclo, podés calcular de antemano cuántas iteraciones exactas requiere el `for`: $n \ge \frac{\ln(b-a) - \ln E}{\ln 2}$.",
            ],
            "cuando_usar": [
                "Cuando necesitás convergencia 100% garantizada sin calcular derivadas ni riesgos de explosión numérica.",
                "Como fase previa de acercamiento antes de lanzar un método veloz como Newton-Raphson.",
            ],
            "cuando_falla": [
                "Si la curva apenas toca el eje $X$ y rebota sin cruzar de signo (raíces múltiples de orden par como $x^2 = 0$).",
                "Convergencia lenta (lineal): solo gana un bit de precisión por cada evaluación.",
            ],
        },
        "formulas_latex": [
            r"x_r = \frac{a + b}{2}, \qquad |b_n - a_n| = \frac{b - a}{2^n} \le E \implies n \ge \frac{\ln(b - a) - \ln E}{\ln 2}",
        ],
        "ejemplo_teorico": {
            "titulo": "Aproximación de $\\sqrt{2}$ mediante $f(x) = x^2 - 2$ en $[1, 3]$",
            "enunciado": "Acorralar la raíz positiva de $f(x) = x^2 - 2$ en el intervalo $[1, 3]$.",
            "pasos": [
                "$f(1) = -1 < 0$ y $f(3) = 7 > 0 \\implies$ existe raíz en $[1, 3]$.",
                "Paso 1: $x_1 = \\frac{1+3}{2} = 2$, $f(2) = 2 > 0 \\implies$ nuevo intervalo $[1, 2]$.",
                "Paso 2: $x_2 = 1.5$, $f(1.5) = 0.25 > 0 \\implies [1, 1.5]$.",
                "Paso 3: $x_3 = 1.25 \\implies [1.25, 1.5] \\dots$ Paso 8: $x_8 = 1.4140625$.",
            ],
            "resultado_final": "$4$ cifras exactas de $\\sqrt{2}$ alcanzadas en $8$ divisiones binarias.",
        },
        "practica_profesora": {
            "titulo": "TP3 Ej. 1: $-2.75x^3 + 18x^2 - 21x - 12 = 0$ con $E < 0.01$",
            "enunciado": "Contar cambios de signo (Descartes), barrer intervalos con $\\Delta x = 0.5$ y aproximar la raíz en $(2, 2.5)$ por Intervalo Medio con $E < 0.01$.",
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
            "solucion_referencia": "Descartes: $2$ o $0$ raíces positivas y $1$ negativa. Cota teórica: $n \\ge 5.64 \\implies 6$ iteraciones. Raíz: $x_6 = 2.2265625$ (error $0.0078125 < 0.01$).",
        },
    },
    "interpolacion_lineal_raices": {
        "nombre": "Interpolación Lineal (Regula Falsi / Secante Acotada)",
        "familia": "raices",
        "icono": "📏",
        "resumen_corto": (
            "En vez de cortar ciegamente a la mitad, une los extremos del intervalo con una línea recta "
            "y usa el punto donde corta al eje $X$ para acercarse más rápido cuando un lado está mucho más cerca de cero."
        ),
        "guru": {
            "que_resuelve": (
                "Acelera la búsqueda en un intervalo cerrado aprovechando las alturas $f(a)$ y $f(b)$ y no solo sus signos."
            ),
            "intuicion_geometrica": (
                "Trazamos una cuerda recta entre $(a, f(a))$ y $(b, f(b))$. Donde esa cuerda corta al eje $X$ suele caer mucho más cerca de la raíz que el centro geométrico."
            ),
            "algoritmo_pasos": [
                "Se calcula la intersección de la recta secante con el eje horizontal mediante semejanza de triángulos.",
                "En la variante estándar de cátedra con curvatura fija, un extremo permanece fijo como pivote mientras el otro avanza hacia la raíz.",
                "En la variante acotada moderna, se reemplaza el extremo del mismo signo que $f(x_{\\text{nuevo}})$ manteniendo la raíz atrapada.",
            ],
            "profundizacion": [
                r"Por semejanza de triángulos rectángulos, $x_{n+1} = x_n - f(x_n)\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$, que puede escribirse de forma equivalente como $\frac{x_{n-1} f(x_n) - x_n f(x_{n-1})}{f(x_n) - f(x_{n-1})}$.",
            ],
            "cuando_usar": [
                "Cuando la función varía suavemente casi como una recta dentro de $[a, b]$.",
            ],
            "cuando_falla": [
                "Si la curva es muy pronunciada (exponenciales o potencias altas), uno de los extremos queda clavado sin moverse (pivote anclado) y el algoritmo se vuelve más lento que Bisección.",
            ],
        },
        "formulas_latex": [
            r"x_{n+1} = \frac{x_{n-1} f(x_n) - x_n f(x_{n-1})}{f(x_n) - f(x_{n-1})} = x_n - f(x_n)\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}",
        ],
        "ejemplo_teorico": {
            "titulo": "Esquema con extremo pivote $P_1(x_1, y_1)$",
            "enunciado": "Aproximar la raíz de una función monótona en $[x_1, x_2]$ con $f(x_1)\\cdot f(x_2) < 0$ y curvatura constante.",
            "pasos": [
                "Se traza la secante entre $(x_1, y_1)$ y $(x_2, y_2)$, hallando su corte $x_3$ con el eje $X$.",
                "Mientras la concavidad mantenga el signo, $x_1$ queda como ancla fija y cada nueva cuerda conecta $(x_1, y_1)$ con $(x_k, y_k)$.",
            ],
            "resultado_final": "Aproximación monótona unilateral hacia la raíz.",
        },
        "practica_profesora": {
            "titulo": "TP3 Ej. 2: $2x^4 - 6x^2 + 7x - 2 = 0$ (raíz menor) con $E < 0.001$",
            "enunciado": "Hallar la raíz negativa de $f(x) = 2x^4 - 6x^2 + 7x - 2$ en $[-2.5, -2.0]$ usando Interpolación Lineal con tolerancia $E < 0.001$.",
            "datos_iniciales": {
                "f_str": "2*x**4 - 6*x**2 + 7*x - 2",
                "a": -2.5,
                "b": -2.0,
                "tol": 0.001,
                "max_iter": 50,
            },
            "solucion_referencia": "Pivote fijo $x_0 = -2.5$ ($f = 21.125$). En $6$ iteraciones ($x_2 = -2.137339 \\dots x_6 = -2.191747$) logra $|x_6 - x_5| = 0.0007 < 0.001$.",
        },
    },
    "newton_raphson": {
        "nombre": "Método de Newton-Raphson (Rectas Tangentes)",
        "familia": "raices",
        "icono": "📐",
        "resumen_corto": (
            "El método más veloz cuando conocés la derivada: reemplaza la curva por su recta tangente "
            "y salta hacia donde toca el eje $X$, duplicando las cifras decimales exactas en cada paso."
        ),
        "guru": {
            "que_resuelve": (
                "Encuentra raíces de ecuaciones suaves en muy pocas iteraciones usando una aproximación lineal local."
            ),
            "intuicion_geometrica": (
                "Nos paramos en el punto $(x_n, f(x_n))$, nos deslizamos por la pendiente tangente $f'(x_n)$ hasta cruzar el eje $X$ y repetimos desde ese nuevo punto."
            ),
            "algoritmo_pasos": [
                "**Chequeo de arranque seguro (Teoremas de Bolzano y Fourier):** verificamos que la curva cruce el eje $X$ en el intervalo y que en $x_0$ se cumpla $f(x_0)\\cdot f''(x_0) > 0$ para que la pendiente inicial nos apunte hacia adentro del intervalo y no nos dispare lejos.",
                "En cada paso actualizamos `x = x - f(x) / df(x)`.",
                "Detenemos el bucle cuando $|x_{n+1} - x_n| < E$.",
            ],
            "profundizacion": [
                r"Proviene de truncar la serie de Taylor en grado $1$: $f(x_{n+1}) \approx f(x_n) + f'(x_n)(x_{n+1} - x_n) = 0$.",
                r"Su error verifica $\varepsilon_{n+1} \approx \left|\frac{f''(r)}{2f'(r)}\right|\varepsilon_n^2$ (**convergencia cuadrática**): si en un paso tenés $2$ decimales exactos, en el siguiente pasás a $4$, luego a $8$, y así sucesivamente.",
            ],
            "cuando_usar": [
                "Cuando necesitás alta precisión en mínimas iteraciones y podés calcular fácilmente $f'(x)$.",
            ],
            "cuando_falla": [
                "Si la curva se aplana ($f'(x_n) \\approx 0$), dividís por casi cero y la tangente casi horizontal dispara el punto muy lejos.",
                "Si arrancás desde un $x_0$ mal elegido (violando Fourier), el algoritmo puede entrar en ciclos infinitos o escapar del intervalo.",
            ],
        },
        "formulas_latex": [
            r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}, \qquad \text{Chequeo de arranque (Fourier): } f(x_0)\cdot f''(x_0) > 0",
        ],
        "ejemplo_teorico": {
            "titulo": "Cálculo de $\\sqrt{2}$ con $f(x) = x^2 - 2$ y $x_0 = 3$",
            "enunciado": "Aproximar $\\sqrt{2}$ aplicando $x_{n+1} = x_n - \\frac{x_n^2 - 2}{2x_n}$ partiendo desde $x_0 = 3$.",
            "pasos": [
                "$x_1 = 3 - \\frac{7}{6} = 1.833333\\dots$",
                "$x_2 = 1.462121\\dots$",
                "$x_3 = 1.414998\\dots$",
                "$x_4 = 1.41421378\\dots \\to x_5 = 1.414213562$ (más de $10$ cifras exactas en apenas $5$ pasos).",
            ],
            "resultado_final": "Convergencia cuadrática alcanzada en $5$ iteraciones.",
        },
        "practica_profesora": {
            "titulo": "TP3 Ej. 3: $x^3 - 2x - 5 = 0$ en $[2, 2.5]$ con $E < 0.0001$",
            "enunciado": "Evaluar el chequeo de arranque seguro (Fourier) en $[2, 2.5]$ para elegir $x_0$ y aproximar la raíz de $x^3 - 2x - 5 = 0$ con $E < 0.0001$.",
            "datos_iniciales": {
                "f_str": "x**3 - 2*x - 5",
                "a": 2.0,
                "b": 2.5,
                "x0": 2.5,
                "tol": 0.0001,
                "max_iter": 50,
            },
            "solucion_referencia": "En $x = 2$ falla el chequeo de arranque: $(-1)(12) < 0$. En $x = 2.5$ sí cumple: $(5.625)(15) > 0 \\implies x_0 = 2.5$. Converge en $5$ pasos a $x_5 = 2.094551$.",
        },
    },
    "punto_fijo_aitken": {
        "nombre": "Iteración de Punto Fijo y Aceleración $\\Delta^2$ de Aitken",
        "familia": "raices",
        "icono": "🌀",
        "resumen_corto": (
            "Reescribe la ecuación $f(x)=0$ como una asignación iterativa `x = g(x)` y usa la aceleración de "
            "Aitken para dar grandes saltos geométricos cuando la iteración normal avanza lento."
        ),
        "guru": {
            "que_resuelve": (
                "Halla la raíz buscando el valor donde la curva $y = g(x)$ corta a la recta diagonal $y = x$, sin necesidad de derivar durante el bucle principal."
            ),
            "intuicion_geometrica": (
                "Al ejecutar `x = g(x)` en un `while`, el algoritmo rebota como una escalera o telaraña entre $y = g(x)$ y la diagonal $y = x$. Aitken observa $3$ puntos consecutivos de esa secuencia y extrapola directo hacia el cruce."
            ),
            "algoritmo_pasos": [
                "Despejás $x$ algebraicamente como $x = g(x)$ verificando que la pendiente alrededor de la raíz achique las distancias (**condición contractiva:** $|g'(x)| < 1$).",
                "Con un bucle simple se evalúan $x_1 = g(x_0)$ y $x_2 = g(x_1)$.",
                "Cada $2$ pasos simples, Aitken calcula una predicción acelerada $S$ usando las diferencias de segundo orden para reiniciar mucho más cerca de la solución.",
            ],
            "profundizacion": [
                r"Si el error disminuye a razón constante $e_{n+1} \approx K e_n$ con $K = g'(r)$, tres valores consecutivos forman una progresión geométrica cuyo límite estimado es $S = x_{n+1} - \frac{(x_{n+1} - x_n)^2}{x_{n+1} - 2x_n + x_{n-1}}$.",
            ],
            "cuando_usar": [
                "Cuando podés despejar una fórmula estable $x = g(x)$ cuya pendiente cumpla $|g'(x)| < 1$ en la zona de búsqueda.",
            ],
            "cuando_falla": [
                "Si la pendiente $|g'(x)| \\ge 1$, cada rebote se aleja más de la solución y la secuencia explota hacia el infinito.",
            ],
        },
        "formulas_latex": [
            r"x_{n+1} = g(x_n) \;\; \left(|g'(x)| < 1\right), \qquad S = x_{n+1} - \frac{(x_{n+1} - x_n)^2}{x_{n+1} - 2x_n + x_{n-1}}",
        ],
        "ejemplo_teorico": {
            "titulo": "Resolución de $e^x - x^2 - 3 = 0$ en $(1, 2)$ con Aitken",
            "enunciado": "Iterar $x_{n+1} = \\ln(x_n^2 + 3)$ partiendo desde $x_0 = 1$ aplicando saltos acelerados de Aitken.",
            "pasos": [
                "$x_0 = 1 \\to x_1 = 1.38629 \\to x_2 = 1.59367$.",
                "Salto extrapolado de Aitken: $x_3 = 1.83405$.",
                "Nuevo ciclo simple: $x_4 = 1.85062$, $x_5 = 1.86016 \\to$ Aitken: $x_6 = 1.87311$.",
            ],
            "resultado_final": "$r \\approx 1.87311$ alcanzado en solo $2$ ciclos de aceleración.",
        },
        "practica_profesora": {
            "titulo": "TP3 Ej. 4: $e^x - x^2 - 5 = 0$ con $x_0 = 2.5$ y $E < 0.0001$",
            "enunciado": "Usar el despeje $g(x) = \\ln(x^2 + 5)$ y comparar la Iteración Simple ($10$ pasos) frente a la Aceleración $\\Delta^2$ de Aitken ($4$ pasos).",
            "datos_iniciales": {
                "f_str": "exp(x) - x**2 - 5",
                "g_str": "log(x**2 + 5)",
                "x0": 2.5,
                "tol": 0.0001,
                "max_iter": 50,
            },
            "solucion_referencia": "La iteración simple demanda $10$ pasos ($2.3563986$). Aitken salta de $x_0=2.5, x_1=2.420368, x_2=2.3849189$ a $x_3=2.356476$ y converge en $x_4=2.356408$.",
        },
    },

    # ==========================================================================
    # 3. ÁLGEBRA LINEAL NUMÉRICA Y AUTOVALORES
    # ==========================================================================
    "eliminacion_gauss": {
        "nombre": "Eliminación Gaussiana (con Pivoteo Parcial)",
        "familia": "sistemas",
        "icono": "📐",
        "resumen_corto": (
            "El método clásico para resolver sistemas lineales $A x = b$: convierte la matriz en un triángulo superior "
            "con dos bucles anidados y luego despeja las incógnitas de abajo hacia arriba."
        ),
        "guru": {
            "que_resuelve": (
                "Resuelve sistemas de $n$ ecuaciones lineales con $n$ incógnitas de tamaño pequeño o mediano en una cantidad fija de pasos."
            ),
            "intuicion_geometrica": (
                "Elimina una variable por columna combinando filas hasta dejar la última ecuación con una sola incógnita, y desde ahí resuelve en cascada inversa."
            ),
            "algoritmo_pasos": [
                "**Intercambio preventivo de filas (Pivoteo parcial):** antes de dividir por un elemento diagonal, sube la fila con el mayor valor absoluto en esa columna para evitar divisiones por cero y reducir error de redondeo.",
                "Elimina los elementos debajo de la diagonal mediante multiplicadores $m_{ik} = \\frac{a_{ik}}{a_{kk}}$.",
                "Ejecuta un bucle hacia atrás (sustitución regresiva) desde $x_n$ hasta $x_1$.",
            ],
            "profundizacion": [
                r"El costo total de operaciones aritméticas escala en orden cúbico: $\approx \frac{2}{3}n^3$ operaciones en la fase de eliminación más $n^2$ en la sustitución regresiva.",
            ],
            "cuando_usar": [
                "Para resolver sistemas densos de tamaño moderado cuando te interesa una solución directa para un único vector $b$.",
            ],
            "cuando_falla": [
                "Si la matriz tiene determinante nulo (ecuaciones repetidas o dependientes) no existe solución única.",
                "En matrices enormes con miles de ceros ($n > 1000$), rellena la memoria inutilizando el almacenamiento disperso.",
            ],
        },
        "formulas_latex": [
            r"a'_{ij} = a_{ij} - \frac{a_{ik}}{a_{kk}} a_{kj}, \qquad x_i = \frac{a_{im} - \sum_{j=i+1}^n a_{ij} x_j}{a_{ii}}",
        ],
        "ejemplo_teorico": {
            "titulo": "Sistema $3 \\times 3$ triangularizado",
            "enunciado": "Resolver $2x_1 - 2x_2 + 5x_3 = 13$, $2x_1 + 3x_2 + 4x_3 = 20$ y $3x_1 - x_2 + 3x_3 = 10$.",
            "pasos": [
                "La fase de eliminación reduce la matriz aumentada a $\\begin{bmatrix} 2 & -2 & 5 & | & 13 \\\\ 0 & 5 & -1 & | & 7 \\\\ 0 & 0 & -4.1 & | & -12.3 \\end{bmatrix}$.",
                "Despejando de abajo hacia arriba: $x_3 = 3 \\to x_2 = 2 \\to x_1 = 1$.",
            ],
            "resultado_final": "Vector solución $x = [1,\\, 2,\\, 3]^T$.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 5: Sistema $3 \\times 3$ de Cátedra",
            "enunciado": "Resolver mediante Eliminación Gaussiana: $18.72x + 8.2y + 8.76z = 121.280$, $6.4x + 15.9y + 7.18z = 126.321$ y $5.22x + 6.4y + 14.31z = 118.522$.",
            "datos_iniciales": {
                "A": [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]],
                "b": [121.280, 126.321, 118.522],
            },
            "solucion_referencia": "Última ecuación reducida: $10.553z = 58.055 \\implies x = 1.836,\\; y = 4.721,\\; z = 5.501$.",
        },
    },
    "gauss_jordan": {
        "nombre": "Método de Gauss-Jordan y Matriz Inversa",
        "familia": "sistemas",
        "icono": "🔁",
        "resumen_corto": (
            "Lleva la eliminación un paso más allá: en vez de dejar un triángulo, limpia ceros **por arriba y por abajo** "
            "de la diagonal hasta obtener directamente el vector solución y la inversa $A^{-1}$."
        ),
        "guru": {
            "que_resuelve": (
                "Entrega al mismo tiempo la solución de $Ax = b$ y la matriz inversa explícita $A^{-1}$ sin paso regresivo posterior."
            ),
            "intuicion_geometrica": (
                "Escala cada fila para poner un $1$ en su diagonal principal y anula todo el resto de la columna, desacoplando cada incógnita."
            ),
            "algoritmo_pasos": [
                "Se arma el bloque extendido $[A \\mid b \\mid I]$ en memoria.",
                "Para cada columna $k$, dividís su fila por el pivote diagonal para normalizarlo a $1$.",
                "Eliminás los coeficientes de todas las demás filas (tanto superiores como inferiores).",
            ],
            "profundizacion": [
                r"Su complejidad asciende a $\approx n^3$ operaciones ($\sim 50\%$ más costo de CPU que Eliminación Gaussiana estándar) cuando solo se busca un vector $x$.",
            ],
            "cuando_usar": [
                "Cuando tu programa necesita explícitamente calcular la matriz inversa $A^{-1}$ o resolver múltiples columnas $b$ a la vez.",
            ],
            "cuando_falla": [
                "Si solo querés resolver un sistema lineal $Ax = b$, es redundante frente a Gauss estándar o factorización $LU$.",
            ],
        },
        "formulas_latex": [
            r"[A \mid b \mid I] \xrightarrow{\text{Operaciones Elementales por Fila}} [I \mid x \mid A^{-1}]",
        ],
        "ejemplo_teorico": {
            "titulo": "Reducción completa del sistema $3 \\times 3$",
            "enunciado": "Normalizar cada fila pivote de $2x_1 - 2x_2 + 5x_3 = 13\\dots$ eliminando tanto arriba como abajo de la diagonal.",
            "pasos": [
                "Divide la fila $1$ por $2$ y anula la columna $1$ en filas $2$ y $3$.",
                "Repite en columnas $2$ y $3$ hasta obtener la matriz identidad $I$ a la izquierda.",
            ],
            "resultado_final": "$x = [1,\\, 2,\\, 3]^T$ aparece directamente en la columna de resultados.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 6: Gauss-Jordan sobre el sistema $3 \\times 3$ de Cátedra",
            "enunciado": "Aplicar el método de Gauss-Jordan y obtener $A^{-1}$ sobre el sistema $18.72x + 8.2y + 8.76z = 121.280\\dots$",
            "datos_iniciales": {
                "A": [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]],
                "b": [121.280, 126.321, 118.522],
            },
            "solucion_referencia": "Redondeo de apunte ($2$ dec.): $[1.82,\\, 4.72,\\, 5.50]$; cálculo numérico completo: $[1.8358,\\, 4.7213,\\, 5.5013]$.",
        },
    },
    "factorizacion_lu": {
        "nombre": "Factorización $LU$ (Esquema de Doolittle)",
        "familia": "sistemas",
        "icono": "🧩",
        "resumen_corto": (
            "Guarda en caché el trabajo pesado de Gauss factorizando $A = L \\cdot U$. "
            "Si mañana cambia el vector de entradas $b$, resolvés el nuevo sistema en tiempo $\\mathcal{O}(n^2)$ sin recalcular $A$."
        ),
        "guru": {
            "que_resuelve": (
                "Desacopla la matriz de coeficientes $A$ del vector independiente $b$ para acelerar múltiples simulaciones con la misma estructura."
            ),
            "intuicion_geometrica": (
                "Separa los multiplicadores de la eliminación en una matriz triangular inferior ($L$) y guarda la matriz escalonada final en una triangular superior ($U$)."
            ),
            "algoritmo_pasos": [
                "Se descompone una sola vez $A = L \\cdot U$ (donde $L$ tiene $1$ en su diagonal y almacena los multiplicadores).",
                "Para cada nuevo vector $b$, primero resolvés hacia adelante $L y = b$ (sustitución progresiva).",
                "Finalmente resolvés hacia atrás $U x = y$ (sustitución regresiva).",
            ],
            "profundizacion": [
                r"La descomposición cuesta $\approx \frac{1}{3}n^3$ multiplicaciones, idéntico a Gauss, pero cada vector adicional $b_k$ cuesta apenas $n^2$ operaciones.",
            ],
            "cuando_usar": [
                "En motores gráficos, simuladores físicos o grafos donde la matriz $A$ es fija y cambia el vector $b$ en cada frame.",
            ],
            "cuando_falla": [
                "La versión básica de Doolittle sin intercambio de filas se detiene si algún elemento diagonal $U_{kk}$ da cero.",
            ],
        },
        "formulas_latex": [
            r"A = L \cdot U \implies \begin{cases} L y = b & \text{(sustitución progresiva)} \\ U x = y & \text{(sustitución regresiva)} \end{cases}",
        ],
        "ejemplo_teorico": {
            "titulo": "Descomposición $3 \\times 3$ exacta de Cátedra",
            "enunciado": "Factorizar $A = [[4, -2, 1], [20, -7, 12], [-8, 13, 17]]$ y resolver para $b = [11, 70, 17]^T$.",
            "pasos": [
                "Multiplicadores almacenados en $L$: $f_{21} = 5$, $f_{31} = -2$, $f_{32} = 3$.",
                "Matriz escalonada $U = [[4, -2, 1], [0, 3, 7], [0, 0, -2]]$.",
                "Paso progresivo $L y = b \\implies y = [11, 15, -6]^T$; paso regresivo $U x = y \\implies x = [1, -2, 3]^T$.",
            ],
            "resultado_final": "Vector solución $x = [1,\\, -2,\\, 3]^T$.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 7: Sistema $4x_1 - 2x_2 + x_3 = 11$",
            "enunciado": "Factorizar $A$ en matrices triangulares $L$ y $U$, calcular el vector auxiliar $y$ mediante $L y = b$ y despejar $x$ desde $U x = y$.",
            "datos_iniciales": {
                "A": [[4.0, -2.0, 1.0], [20.0, -7.0, 12.0], [-8.0, 13.0, 17.0]],
                "b": [11.0, 70.0, 17.0],
            },
            "solucion_referencia": "$y = [11,\\, 15,\\, -6]^T$ y solución final $x = [1,\\, -2,\\, 3]^T$.",
        },
    },
    "gauss_seidel": {
        "nombre": "Método Iterativo de Gauss-Seidel",
        "familia": "sistemas",
        "icono": "🔄",
        "resumen_corto": (
            "Resuelve sistemas enormes mediante un bucle `while` sin modificar la matriz original: despeja cada variable "
            "y usa inmediatamente los valores recién actualizados en esa misma vuelta."
        ),
        "guru": {
            "que_resuelve": (
                "Aproxima $A x = b$ en sistemas de miles de ecuaciones dispersas (con muchos ceros) donde almacenar matrices triangulares llenaría la RAM."
            ),
            "intuicion_geometrica": (
                "Despeja $x_i$ de la ecuación $i$-ésima y ajusta una coordenada por vez rebote tras rebote hasta estabilizarse en la solución."
            ),
            "algoritmo_pasos": [
                "**Chequeo de estabilidad antes de iterar (Dominancia diagonal):** verificamos que en cada fila el elemento de la diagonal $|a_{ii}|$ pese más que la suma de los demás coeficientes de su fila ($|a_{ii}| \\ge \\sum_{j \\ne i} |a_{ij}|$).",
                "Arrancás con una suposición inicial (por ejemplo $x^{(0)} = [0, 0, 0]$ o $[1, 1, 1]$) y recorres las variables actualizándolas in-place.",
                "Se detiene cuando el mayor cambio entre iteraciones $|x^{(k+1)} - x^{(k)}|$ cae debajo de la tolerancia $E$.",
            ],
            "profundizacion": [
                r"La convergencia ocurre si y solo si el factor de amplificación asintótica de la matriz de iteración (radio espectral $\rho(T_{GS})$) es estrictamente menor que $1$. La dominancia diagonal estricta es una condición suficiente fácil de programar en $\mathcal{O}(n^2)$ antes de arrancar.",
            ],
            "cuando_usar": [
                "En matrices grandes con mayoría de ceros (mallas físicas, grafos) que cumplen dominancia diagonal o permiten reordenar filas para lograrla.",
            ],
            "cuando_falla": [
                "Si la matriz NO es diagonalmente dominante, los valores suelen oscilar creciendo cada iteración hasta desbordar (`Overflow`/`Inf`).",
            ],
        },
        "formulas_latex": [
            r"x_i^{(k+1)} = \frac{b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)}}{a_{ii}}",
        ],
        "ejemplo_teorico": {
            "titulo": "Reordenar filas para lograr dominancia diagonal",
            "enunciado": "Resolver $2x_1 + 10x_2 + x_3 = 51$, $x_1 + 2x_2 + 10x_3 = 61$ y $10x_1 + x_2 + 2x_3 = 44$ desde $(0,0,0)$ con $E \\le 0.02$.",
            "pasos": [
                "Reordenamos filas para ubicar el mayor coeficiente en la diagonal: $x_1$ desde la tercera ecuación, $x_2$ desde la primera y $x_3$ desde la segunda.",
                "Iteración 1: $(2.874,\\, 5.1,\\, 5.08) \\to$ Iteración 3: $(2.9999,\\, 3.9998,\\, 5.0004)$.",
            ],
            "resultado_final": "Converge a $x = [3,\\, 4,\\, 5]^T$ en apenas $3$ pasadas.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 8: Sistema con Dominancia Diagonal y $X_0 = [1, 1, 1]$",
            "enunciado": "Comprobar la dominancia diagonal ($18.72 > 16.96$; $15.9 > 13.58$; $14.31 > 11.62$) e iterar desde $[1, 1, 1]$ con tolerancia $E = 0.01$.",
            "datos_iniciales": {
                "A": [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]],
                "b": [121.280, 126.321, 118.522],
                "x0": [1.0, 1.0, 1.0],
                "tol": 0.01,
                "max_iter": 50,
            },
            "solucion_referencia": "Converge en $6$ iteraciones a $[1.835,\\, 4.721,\\, 5.502]$ con errores $\\le 0.005$.",
        },
    },
    "potencias": {
        "nombre": "Método de las Potencias (Directo e Inverso)",
        "familia": "autovalores",
        "icono": "⚡",
        "resumen_corto": (
            "Encuentra el autovalor más grande ($\\lambda_{\\max}$) o más chico ($\\lambda_{\\min}$) multiplicando "
            "una matriz por un vector repetidas veces dentro de un bucle y reescalando en cada paso."
        ),
        "guru": {
            "que_resuelve": (
                "Calcula los factores de escala extremos (máximo y mínimo) y sus direcciones propias sin construir polinomios gigantes."
            ),
            "intuicion_geometrica": (
                "Al multiplicar repetidamente un vector por la matriz $A$, la dirección de mayor estiramiento crece mucho más rápido que las demás y termina dominando por completo al vector."
            ),
            "algoritmo_pasos": [
                "Se multiplica `Y = A @ X` (o `Y = inv(A) @ X` para extraer $\\lambda_{\\min}$).",
                "Se toma el componente de mayor módulo de `Y` como estimación actual de $\\lambda$ y se divide todo el vector por ese número para normalizarlo (evitando desbordes).",
                "Se repite hasta que el cambio relativo entre $\\lambda_k$ y $\\lambda_{k-1}$ sea menor que la tolerancia.",
            ],
            "profundizacion": [
                r"La razón de convergencia depende de la distancia relativa entre los dos autovalores más grandes: $\left|\frac{\lambda_2}{\lambda_1}\right|^k \to 0$. Es la base conceptual del algoritmo original de PageRank para ordenar grafos web.",
            ],
            "cuando_usar": [
                "En matrices grandes (redes, estabilidad estructural, grafos) donde solo interesa el modo dominante $\\lambda_{\\max}$ o el menor $\\lambda_{\\min}$.",
            ],
            "cuando_falla": [
                "Si dos autovalores dominantes empatan en tamaño ($|\\lambda_1| \\approx |\\lambda_2|$), la convergencia se estanca o queda oscilando.",
                "Si el vector inicial $X_0$ arranca exactamente perpendicular al autovector dominante.",
                "La versión inversa para $\\lambda_{\\min}$ falla si la matriz no tiene inversa ($\\det(A) = 0$).",
            ],
        },
        "formulas_latex": [
            r"A X_{k-1} = \bar{\lambda}_k X_k \quad (\lambda_{\max}), \qquad A^{-1} X_{k-1} = \frac{1}{\bar{\lambda}_k} X_k \quad (\lambda_{\min})",
        ],
        "ejemplo_teorico": {
            "titulo": "Normalización por componente máxima (norma infinito)",
            "enunciado": "Dado un vector inicial $X_0$, calcular $Y_k = A X_{k-1}$, extraer $m_k$ igual al componente de mayor magnitud y normalizar $X_k = Y_k / m_k$.",
            "pasos": [
                "Repetir el ciclo mientras $\\frac{|\\lambda_k - \\lambda_{k-1}|}{|\\lambda_k|} \\ge E$.",
            ],
            "resultado_final": "$m_k \\to \\lambda_{\\max}$ y $X_k$ converge al autovector asociado.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 10: Matriz Tridiagonal Simétrica $3 \\times 3$ con $X_0 = [1, 1, 1]$",
            "enunciado": "Calcular el mayor y menor autovalor de $A = [[3.556, -1.778, 0], [-1.778, 3.556, -1.778], [0, -1.778, 3.556]]$ partiendo de $X_0 = [1, 1, 1]$.",
            "datos_iniciales": {
                "A": [[3.556, -1.778, 0.0], [-1.778, 3.556, -1.778], [0.0, -1.778, 3.556]],
                "x0": [1.0, 1.0, 1.0],
                "tol": 0.05,
                "max_iter": 30,
            },
            "solucion_referencia": "$\\lambda_{\\max} \\approx 6.095$ con $v = [-0.708,\\, 1,\\, -0.708]^T$ ($5$ iteraciones). $\\lambda_{\\min} \\approx 1.037$ con $v = [0.709,\\, 1,\\, 0.709]^T$ ($3$ iteraciones).",
        },
    },
    "faddeev_leverrier": {
        "nombre": "Método de Faddeev-Leverrier (Espectro Completo)",
        "familia": "autovalores",
        "icono": "🔬",
        "resumen_corto": (
            "Un algoritmo en cadena de exactamente $n$ pasos matriciales que reconstruye todos los coeficientes "
            "del polinomio característico, obtiene **todos los autovalores** y de regalo entrega $A^{-1}$."
        ),
        "guru": {
            "que_resuelve": (
                "Calcula simultáneamente el espectro completo de autovalores, $\\det(A)$ y la matriz inversa $A^{-1}$ en matrices pequeñas."
            ),
            "intuicion_geometrica": (
                "En cada paso multiplica matrices acumuladas y lee la suma de su diagonal (la traza), extrayendo los coeficientes del polinomio cuyas raíces son todos los autovalores."
            ),
            "algoritmo_pasos": [
                "Inicializás $B_1 = A$ y tomás $b_1 = \\operatorname{tr}(B_1)$ (suma de la diagonal).",
                "Para $k = 2 \\dots n$, calculás $B_k = A(B_{k-1} - b_{k-1} I)$ y promediás su diagonal $b_k = \\frac{1}{k}\\operatorname{tr}(B_k)$.",
                "Con $[1, -b_1, \\dots, -b_n]$ obtenés el polinomio característico $P(\\lambda) = 0$ y con el penúltimo bloque armás $A^{-1}$ gratis.",
            ],
            "profundizacion": [
                r"Su costo computacional es $\mathcal{O}(n^4)$ porque realiza $n-1$ multiplicaciones de matrices $n \times n$ de costo $\mathcal{O}(n^3)$ cada una, además de requerir calcular las raíces de un polinomio de grado $n$.",
            ],
            "cuando_usar": [
                "Para matrices pequeñas ($n \\le 6$) donde necesitás conocer **todos** los autovalores y la expresión simbólica del polinomio característico.",
            ],
            "cuando_falla": [
                "En matrices de dimensión media o grande ($n > 8$): es lento ($\\mathcal{O}(n^4)$) y encontrar raíces de polinomios de grado alto se vuelve muy sensible al error de máquina.",
                "No puede devolver $A^{-1}$ si la matriz es singular ($b_n = 0$).",
            ],
        },
        "formulas_latex": [
            r"B_1 = A, \;\; b_1 = \operatorname{tr}(B_1); \qquad B_k = A(B_{k-1} - b_{k-1}I), \;\; b_k = \frac{1}{k}\operatorname{tr}(B_k)",
            r"P(\lambda) = (-1)^n \left(\lambda^n - b_1 \lambda^{n-1} - \dots - b_n\right) = 0, \qquad A^{-1} = \frac{1}{b_n}(B_{n-1} - b_{n-1}I)",
        ],
        "ejemplo_teorico": {
            "titulo": "Matriz simétrica clásica $3 \\times 3$ de Unidad 4",
            "enunciado": "Calcular el polinomio característico, los autovalores y la inversa de $A = [[3, 2, 4], [2, 0, 2], [4, 2, 3]]$.",
            "pasos": [
                "$B_1 = A \\implies b_1 = 6$.",
                "$B_2 = [[11, 2, 4], [2, 8, 2], [4, 2, 11]] \\implies b_2 = 15$.",
                "$B_3 = 8I \\implies b_3 = 8$.",
                "Polinomio: $-\\lambda^3 + 6\\lambda^2 + 15\\lambda + 8 = 0 \\implies \\lambda \\in \\{8,\\, -1,\\, -1\\}$.",
            ],
            "resultado_final": "Autovalores $\\{8,\\, -1,\\, -1\\}$ e inversa $A^{-1} = \\frac{1}{8}(B_2 - 15I)$.",
        },
        "practica_profesora": {
            "titulo": "TP4 Ej. 9: Faddeev-Leverrier + Autovector de $\\lambda = 8$",
            "enunciado": "Obtener coeficientes $b_1, b_2, b_3$, polinomio característico y espectro completo de la matriz $[[3, 2, 4], [2, 0, 2], [4, 2, 3]]$.",
            "datos_iniciales": {
                "A": [[3.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 3.0]],
            },
            "solucion_referencia": "$b_1=6$, $b_2=15$, $b_3=8$. $P(\\lambda) = -\\lambda^3 + 6\\lambda^2 + 15\\lambda + 8$. Raíces: $8, -1, -1$. Para $\\lambda=8$: $v = [-1,\\, -0.5,\\, -1]^T$.",
        },
    },

    # ==========================================================================
    # 4. INTERPOLACIÓN POLINOMIAL
    # ==========================================================================
    "interpolacion_lineal": {
        "nombre": "Interpolación Lineal Directa e Inversa",
        "familia": "interpolacion",
        "icono": "📍",
        "resumen_corto": (
            "Conecta dos puntos consecutivos de una tabla con un segmento de recta para estimar valores intermedios "
            "$y(x)$ o despejar al revés qué valor de $x$ alcanza cierta altura $y$."
        ),
        "guru": {
            "que_resuelve": (
                "Estima un dato faltante entre dos filas vecinas de una tabla o realiza búsqueda inversa de entrada $x$."
            ),
            "intuicion_geometrica": (
                "Tira una línea recta entre $(x_0, y_0)$ y $(x_1, y_1)$ y evalúa la ecuación de la recta en el punto objetivo."
            ),
            "algoritmo_pasos": [
                "Identificás los dos puntos de la tabla que rodean al objetivo.",
                "Calculás la pendiente local $\\frac{y_1 - y_0}{x_1 - x_0} = \\frac{\\Delta y_0}{h}$.",
                "Para interpolar directo despejás $y(x)$; para inversa despejás $x(y)$.",
            ],
            "profundizacion": [
                r"El error de interpolación lineal depende de la curvatura (segunda derivada): $E(x) = \frac{1}{2} f''(\xi)(x - x_0)(x - x_1)$, siendo máximo justo en la mitad del tramo.",
            ],
            "cuando_usar": [
                "Cuando la distancia $h$ entre mediciones es chica o la variable crece de manera casi lineal.",
            ],
            "cuando_falla": [
                "En tablas muy espaciadas con curvatura fuerte ($\\Delta^2 y$ grande), donde la línea recta pasa sistemáticamente por encima o por debajo de la curva real.",
            ],
        },
        "formulas_latex": [
            r"P_1(x) = y_0 + \frac{\Delta y_0}{h}(x - x_0) \quad \Longleftrightarrow \quad x = x_0 + \frac{(y - y_0)\,h}{\Delta y_0}",
        ],
        "ejemplo_teorico": {
            "titulo": "Aproximación lineal sobre $\\ln(x)$",
            "enunciado": "Analizar cómo la cuerda entre $x = 1$ y $x = 4$ aproxima una curva logarítmica dejando una brecha por curvatura.",
            "pasos": [
                "Se calcula la pendiente secante $\\frac{\\Delta y_0}{h}$.",
                "Acortar el paso $h$ reduce cuadráticamente la distancia entre el segmento recto y la función.",
            ],
            "resultado_final": "Exacta únicamente cuando los datos reales están alineados ($\\Delta^2 y = 0$).",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. Inversa: Viscosidad del Agua ($y = 1.702$)",
            "enunciado": "Dada la tabla $x = [0, 5, 10, 15]$, $y = [1.792, 1.519, 1.308, 1.140]$, calcular la temperatura $x$ correspondiente a $y = 1.702$.",
            "datos_iniciales": {
                "x": [0.0, 5.0, 10.0, 15.0],
                "y": [1.792, 1.519, 1.308, 1.140],
                "y_objetivo": 1.702,
                "x_objetivo": 2.5,
            },
            "solucion_referencia": "$1.702 = 1.792 + \\frac{-0.273}{5}(x - 0) \\implies x \\approx 1.64835\\ ^\\circ\\text{C}$ (en apunte truncado a $1.64\\ ^\\circ\\text{C}$).",
        },
    },
    "cuadratica_inversa": {
        "nombre": "Interpolación Cuadrática Inversa (Grado 2)",
        "familia": "interpolacion",
        "icono": "➰",
        "resumen_corto": (
            "En vez de una recta usa una **parábola ajustada con $3$ puntos** de la tabla y aplica la fórmula "
            "cuadrática para despejar con mucha mayor exactitud qué $x$ genera un $y$ objetivo."
        ),
        "guru": {
            "que_resuelve": (
                "Corrige el error por curvatura de la interpolación lineal inversa usando un nodo adicional."
            ),
            "intuicion_geometrica": (
                "Traza una parábola suave por $3$ puntos vecinos y la corta con la línea horizontal $y = y_{\\text{obj}}$ para leer la abscisa dentro del intervalo."
            ),
            "algoritmo_pasos": [
                "Se arma el polinomio de grado $2$ con las diferencias $\\Delta y_0$ y $\\Delta^2 y_0$.",
                "Igualando al valor deseado $y_{\\text{obj}}$ se obtiene una ecuación cuadrática $au^2 + bu + c = 0$ en $u = x - x_0$.",
                "Se calculan las dos raíces de la cuadrática y se elige la que pertenece al rango de la tabla.",
            ],
            "profundizacion": [
                r"Aunque los $x_i$ sean equidistantes, los valores tabulados $y_i$ no lo son; por eso no puede invertirse la tabla como diferencias finitas en $y$ y se plantea la cuadrática en función de $x$.",
            ],
            "cuando_usar": [
                "En tablas con espaciamiento uniforme $h$ donde disponés de $3$ o más puntos y una recta queda corta.",
            ],
            "cuando_falla": [
                "Si el discriminante $b^2 - 4ac < 0$, la altura $y_{\\text{obj}}$ queda por fuera del alcance de la parábola real.",
                "Requiere que los nodos $x_i$ estén equiespaciados para usar el término con paso $h$ constante.",
            ],
        },
        "formulas_latex": [
            r"y = y_0 + \frac{\Delta y_0}{h}(x - x_0) + \frac{\Delta^2 y_0}{2!\,h^2}(x - x_0)(x - x_1) \implies a u^2 + b u + c = 0 \;\; (u = x - x_0)",
        ],
        "ejemplo_teorico": {
            "titulo": "Inversión de una curva no lineal con $3$ puntos",
            "enunciado": "Ajustar $x = f^{-1}(y)$ cuando los valores $y$ no son equidistantes resolviendo la parábola en $x$.",
            "pasos": [
                "Se desarrolla el polinomio de grado $2$ hasta $\\Delta^2 y_0$ igualado a $y_{\\text{obj}}$.",
                "Se aplica la resolvente cuadrática descartando la raíz externa al tramo de datos.",
            ],
            "resultado_final": "La raíz ubicada dentro del intervalo tabular entrega la estimación curva inversa.",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. 2b: Viscosidad del Agua ($y = 1.702$) con $3$ nodos",
            "enunciado": "Con $x = [0, 5, 10, 15]$ e $y = [1.792, 1.519, 1.308, 1.140]$, calcular la temperatura para $y = 1.702$ sumando el término cuadrático $\\Delta^2 y_0 = 0.062$.",
            "datos_iniciales": {
                "x": [0.0, 5.0, 10.0, 15.0],
                "y": [1.792, 1.519, 1.308, 1.140],
                "y_objetivo": 1.702,
            },
            "solucion_referencia": "$0.00124 x^2 - 0.0608 x + 0.09 = 0 \\implies x_1 = 47.50$ (fuera de rango), $x_2 \\approx 1.5296\\ ^\\circ\\text{C}$ (apunte: $1.52\\ ^\\circ\\text{C}$).",
        },
    },
    "lagrange": {
        "nombre": "Polinomio Interpolador de Lagrange",
        "familia": "interpolacion",
        "icono": "🌊",
        "resumen_corto": (
            "Construye una suma ponderada de polinomios de 'interruptor' $L_k(x)$ que pasan exactamente por todos los "
            "datos, **incluso si tus mediciones en $x$ tienen espaciamiento irregular**."
        ),
        "guru": {
            "que_resuelve": (
                "Interpola sobre datasets del mundo real (sensores, timestamps, experimentos) donde los intervalos entre $x_i$ no son iguales."
            ),
            "intuicion_geometrica": (
                "Crea un filtro base $L_k(x)$ por cada fila de datos: ese filtro vale exactamente $1$ en su punto $x_k$ y $0$ en todos los demás. Al multiplicarlo por $y_k$ y sumar, la curva final toca cada punto exacto."
            ),
            "algoritmo_pasos": [
                "Para cada índice $k$, un doble bucle `for` calcula el producto $\\prod_{j \\ne k} \\frac{x - x_j}{x_k - x_j}$.",
                "Se multiplica ese peso $L_k(x)$ por $y_k$ y se acumula en el total.",
                "**Garantía de resultado idéntico (Teorema de Unicidad):** cuando los datos están igualmente espaciados, este cálculo da exactamente el mismo polinomio que Newton-Gregory.",
            ],
            "profundizacion": [
                r"El algoritmo evalúa en $\mathcal{O}(n^2)$ operaciones por punto y requiere que todas las abscisas $x_i$ sean distintas entre sí para no dividir por cero en $x_k - x_j$.",
            ],
            "cuando_usar": [
                "Siempre que los valores $x_i$ de tu tabla **no** estén a distancia constante ($h$ variable).",
                "Cuando querés obtener la fórmula polinómica explícita completa.",
            ],
            "cuando_falla": [
                "Si existen dos coordenadas $x_i$ iguales, aparece una división por cero.",
                "**Oscilación artificial al forzar demasiados puntos (Fenómeno de Runge):** usar un único polinomio con muchos nodos ($n > 10$) genera picos gigantes en los bordes del gráfico.",
            ],
        },
        "formulas_latex": [
            r"P_n(x) = \sum_{k=0}^{n} y_k L_k(x), \qquad L_k(x) = \prod_{\substack{j=0 \\ j \neq k}}^{n} \frac{x - x_j}{x_k - x_j}",
        ],
        "ejemplo_teorico": {
            "titulo": "Interpolación sobre nodos no equidistantes de $\\ln(x)$",
            "enunciado": "Evaluar $P(2)$ dados $3$ nodos desiguales $x = [1, 4, 6]$ con $y = [0,\\, 1.386294,\\, 1.791759]$.",
            "pasos": [
                "Se computan los pesos base $L_0(2)$, $L_1(2)$ y $L_2(2)$.",
                "Se evalúa la combinación $y_0 L_0(2) + y_1 L_1(2) + y_2 L_2(2) = 0.5658442$.",
            ],
            "resultado_final": "Produce exactamente el único polinomio cuadrático que atraviesa esos tres puntos.",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. 2: Presión de Vapor de Agua a $T = 180\\ ^\\circ\\text{C}$",
            "enunciado": "Calcular la presión de vapor a $180\\ ^\\circ\\text{C}$ con datos no equiespaciados $T = [100, 120, 200, 220]$, $P = [1.01, 3.61, 15.5, 23.2]$.",
            "datos_iniciales": {
                "x": [100.0, 120.0, 200.0, 220.0],
                "y": [1.01, 3.61, 15.5, 23.2],
                "x_eval": 180.0,
            },
            "solucion_referencia": "Producto $(180-100)(180-120)(180-200)(180-220) = 3\\,840\\,000 \\implies P(180) \\approx 10.562\\ \\text{Pa}$.",
        },
    },
    "newton_gregory_asc": {
        "nombre": "Newton-Gregory Ascendente (Diferencias Progresivas)",
        "familia": "interpolacion",
        "icono": "📈",
        "resumen_corto": (
            "Cuando la tabla tiene paso fijo $h$, precalcula una matriz triangular de restas sucesivas $\\Delta^k y_0$ "
            "para ir refinando la estimación grado a grado cerca del comienzo de la tabla."
        ),
        "guru": {
            "que_resuelve": (
                "Interpola rápidamente en tablas con intervalo constante $h$ cerca de los primeros nodos ($x_0$)."
            ),
            "intuicion_geometrica": (
                "Arranca en $y_0$ y suma correcciones basadas en la pendiente inicial ($\\Delta y_0$), la curvatura ($\\Delta^2 y_0$) y órdenes siguientes sin rearmar todo desde cero."
            ),
            "algoritmo_pasos": [
                "Se verifica que el paso $h = x_{i+1} - x_i$ sea igual en toda la tabla y se normaliza $u = \\frac{x - x_0}{h}$.",
                "Se arma la tabla triangular de restas consecutivas hacia abajo (diferencias progresivas $\\Delta^k y_i$).",
                "Se suman los términos tomando la fila diagonal superior $[y_0,\\, \\Delta y_0,\\, \\Delta^2 y_0, \\dots]$.",
            ],
            "profundizacion": [
                r"Una ventaja computacional clave de esta tabla triangular es detectar automáticamente el grado real de los datos: si las mediciones pertenecen a un polinomio de grado $k$, la columna $\Delta^{k+1} y$ se anula completamente (útil también para detectar errores de carga en una fila).",
            ],
            "cuando_usar": [
                "Exclusivamente cuando todos los $x_i$ están igualmente espaciados ($h$ constante) y el valor buscado $x^*$ está cerca del principio.",
            ],
            "cuando_falla": [
                "**Invalidado si se rompe la equidistancia:** si modificás un solo $x_i$ cambiando $h$, la fórmula deja de aplicar y debés pasar a Lagrange.",
            ],
        },
        "formulas_latex": [
            r"u = \frac{x - x_0}{h}, \qquad P_n(x) = y_0 + u\,\Delta y_0 + \frac{u(u-1)}{2!}\Delta^2 y_0 + \frac{u(u-1)(u-2)}{3!}\Delta^3 y_0 + \dots",
        ],
        "ejemplo_teorico": {
            "titulo": "Detección de grado en $y = x^2$ sobre $x = [0, 1, 2, 3, 4]$",
            "enunciado": "Observar cómo las diferencias progresivas revelan el grado cuadrático.",
            "pasos": [
                "Primera resta ($\\Delta y$): $[1, 3, 5, 7]$; segunda ($\\Delta^2 y$): $[2, 2, 2]$ constantes; tercera ($\\Delta^3 y$): $[0, 0]$.",
            ],
            "resultado_final": "Toda columna nula indica que el polinomio ya capturó la función exacta.",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. 1a: Newton-Gregory Ascendente en $x = 0.35$",
            "enunciado": "Con paso fijo $h = 0.3$ ($x = [0.2, 0.5, 0.8, 1.1, 1.4]$, $y = [0.258, 2.535, 4.902, 6.722, 10.704]$), estimar $f(0.35)$.",
            "datos_iniciales": {
                "x": [0.2, 0.5, 0.8, 1.1, 1.4],
                "y": [0.258, 2.535, 4.902, 6.722, 10.704],
                "x_eval": 0.35,
            },
            "solucion_referencia": "$u = \\frac{0.35 - 0.2}{0.3} = 0.5$; diagonal $[0.258,\\, 2.277,\\, 0.09,\\, -0.637,\\, 3.346] \\implies P(0.35) = 1.214734375$.",
        },
    },
    "newton_gregory_desc": {
        "nombre": "Newton-Gregory Descendente (Diferencias Retrógradas)",
        "familia": "interpolacion",
        "icono": "📉",
        "resumen_corto": (
            "Es la versión simétrica anclada al final de la tabla ($x_n$). Recorre la diagonal inferior "
            "de la matriz de diferencias $\\nabla^k y_n$ para lograr mejor precisión sobre los últimos datos."
        ),
        "guru": {
            "que_resuelve": (
                "Interpola cerca del extremo final de una serie temporal o tabla con paso constante $h$."
            ),
            "intuicion_geometrica": (
                "Toma el último punto $(x_n, y_n)$ como ancla y retrocede hacia la izquierda acumulando correcciones con signos positivos $v(v+1)$."
            ),
            "algoritmo_pasos": [
                "Se define la coordenada relativa al extremo derecho $v = \\frac{x - x_n}{h}$ (habitualmente negativa cuando $x < x_n$).",
                "Se leen los valores desde el borde inferior de la misma matriz de restas sucesivas ($\\nabla^k y_n$).",
                "Si se truncan pocos grados, conserva mucha más precisión cerca de $x_n$ que la versión ascendente.",
            ],
            "profundizacion": [
                r"Cuando se incluyen todas las diferencias hasta el grado $n$, la curva final obtenida coincide algebraicamente punto por punto con Newton-Gregory Ascendente y con Lagrange (**Teorema de Unicidad del Polinomio Interpolador**).",
            ],
            "cuando_usar": [
                "Cuando el punto de consulta $x^*$ está próximo al último nodo $x_n$ y todos los $x_i$ están a igual distancia $h$.",
            ],
            "cuando_falla": [
                "Deja de ser válido si un dato de entrada rompe la distancia uniforme $h$.",
            ],
        },
        "formulas_latex": [
            r"v = \frac{x - x_n}{h}, \qquad P_n(x) = y_n + v\,\nabla y_n + \frac{v(v+1)}{2!}\nabla^2 y_n + \frac{v(v+1)(v+2)}{3!}\nabla^3 y_n + \dots",
        ],
        "ejemplo_teorico": {
            "titulo": "Cambio de ancla al extremo $x_n$ en tabla uniforme",
            "enunciado": "Centrar el parámetro $v = \\frac{x - x_n}{h}$ leyendo la diagonal inferior acumulada.",
            "pasos": [
                "Evaluar $v < 0$ con factores combinatorios positivos $v(v+1)\\dots(v+k-1)/k!$.",
            ],
            "resultado_final": "Reproduce el mismo polinomio interpolador único cuando se utiliza el grado completo.",
        },
        "practica_profesora": {
            "titulo": "TP5 Ej. 1b: Newton-Gregory Descendente en $x = 1.37$",
            "enunciado": "Con la misma tabla $x = [0.2, 0.5, 0.8, 1.1, 1.4]$, $y = [0.258, 2.535, 4.902, 6.722, 10.704]$, estimar $f(1.37)$.",
            "datos_iniciales": {
                "x": [0.2, 0.5, 0.8, 1.1, 1.4],
                "y": [0.258, 2.535, 4.902, 6.722, 10.704],
                "x_eval": 1.37,
            },
            "solucion_referencia": "$v = \\frac{1.37 - 1.4}{0.3} = -0.1$; diagonal inferior $[10.704,\\, 3.982,\\, 2.162,\\, 2.709,\\, 3.346] \\implies P(1.37) = 10.06216678$.",
        },
    },
    "formula_gauss": {
        "nombre": "Fórmula de Gauss (Diferencias Centrales en Zig-Zag)",
        "familia": "interpolacion",
        "icono": "",
        "resumen_corto": (
            "Ancla el desarrollo en el nodo central de la tabla equidistante y avanza en zig-zag "
            "para mantener mínimo el parámetro adimensional $|u| \\le 0.5$ cuando $x^*$ está en el centro."
        ),
        "guru": {
            "que_resuelve": (
                "Interpola con máxima precisión en el tercio central de una tabla equiespaciada cuando se trunca antes del grado máximo."
            ),
            "intuicion_geometrica": (
                "En lugar de anclarse en un borde ($x_0$ o $x_n$), parte del corazón de la tabla y alterna nodos arriba y abajo en zig-zag para rodear simétricamente a $x^*$."
            ),
            "algoritmo_pasos": [
                "Elegí como ancla central $x_0^{\\mathrm{c}}$ el nodo más próximo a $x^*$ en el centro de la tabla y definí $u = (x - x_0^{\\mathrm{c}})/h$.",
                "Gauss Progresiva ($0 \\le u \\le 0.5$): da el primer paso hacia abajo en la tabla escalonada ($\\Delta f(x_0)$) y luego zigzaguea arriba/abajo con factores $u, (u-1)u, (u-1)u(u+1), \\dots$.",
                "Gauss Regresiva ($-0.5 \\le u < 0$): da el primer paso hacia arriba ($\\Delta f(x_0 - h)$) y alterna simétricamente con factores $u, u(u+1), (u-1)u(u+1), \\dots$.",
            ],
            "profundizacion": [
                r"Con grado completo ($k = n - 1$), Gauss Progresiva, Gauss Regresiva y ambas variantes de Newton-Gregory generan exactamente el mismo polinomio por Unicidad; su ventaja decisiva aparece al truncar ($k < n - 1$), porque cada nueva diferencia incorpora el nodo vecino más cercano al centro.",
            ],
            "cuando_usar": [
                "Cuando la malla tiene paso constante $h$ y el punto $x^*$ está próximo al centro de la tabla ($-0.5 \\le u \\le 0.5$).",
            ],
            "cuando_falla": [
                "Se invalida si los nodos $x_i$ pierden equidistancia, y pierde su ventaja por sobre Newton-Gregory si $x^*$ está pegado a $x_0$ o a $x_n$.",
            ],
        },
        "formulas_latex": [
            r"u = \frac{x - x_0}{h}, \quad P_n(x) = f(x_0) + \Delta f(x_0)\, u + \frac{\Delta^2 f(x_0 - h)}{2!}(u - 1)u + \frac{\Delta^3 f(x_0 - h)}{3!}(u - 1)u(u + 1) + \frac{\Delta^4 f(x_0 - 2h)}{4!}(u - 2)(u - 1)u(u + 1) + \dots",
            r"P_n^{\mathrm{reg}}(x) = f(x_0) + \Delta f(x_0 - h)\, u + \frac{\Delta^2 f(x_0 - h)}{2!}u(u + 1) + \frac{\Delta^3 f(x_0 - 2h)}{3!}(u - 1)u(u + 1) + \frac{\Delta^4 f(x_0 - 2h)}{4!}(u + 2)(u - 1)u(u + 1) + \dots",
        ],
        "ejemplo_teorico": {
            "titulo": "Trayectoria central en zig-zag sobre tabla de 5 nodos",
            "enunciado": "Centrar $x_0^{\\mathrm{c}} = x_2$ y alternar diferencias de la zona media de la tabla.",
            "pasos": [
                "Gauss Progresiva toma filas $2, 2, 1, 1, 0$ (baja primero en orden impar, sube en orden par).",
                "Gauss Regresiva toma filas $2, 1, 1, 0, 0$ (sube primero en orden impar, baja en orden par).",
            ],
            "resultado_final": "Al incluir el grado completo coincide exactamente con Lagrange y Newton-Gregory; al truncar en el centro logra el menor residuo.",
        },
        "practica_profesora": {
            "titulo": "TP5 Central: Fórmula de Gauss Progresiva en $x = 0.90$",
            "enunciado": "Sobre la tabla de TP5 Ej. 1 ($x = [0.2, 0.5, 0.8, 1.1, 1.4]$, $y = [0.258, 2.535, 4.902, 6.722, 10.704]$), estimar $f(0.90)$ anclando en el nodo central $x_2 = 0.8$.",
            "datos_iniciales": {
                "x": [0.2, 0.5, 0.8, 1.1, 1.4],
                "y": [0.258, 2.535, 4.902, 6.722, 10.704],
                "x_eval": 0.90,
            },
            "solucion_referencia": "$u = \\frac{0.90 - 0.8}{0.3} = 0.333333$; zig-zag $[4.902,\\, 1.820,\\, -0.547,\\, 2.709,\\, 3.346] \\implies P_4(0.90) = 5.50451440$.",
        },
    },

    # ==========================================================================
    # 5. INTEGRACIÓN Y DERIVACIÓN NUMÉRICA
    # ==========================================================================
    "diferenciacion_numerica": {
        "nombre": "Diferenciación Numérica (Polinomios de Gregory)",
        "familia": "integracion",
        "icono": "📐",
        "resumen_corto": (
            "Calcula derivadas primera y segunda $f'(x)$ y $f''(x)$ cuando no conocés la fórmula analítica "
            "sino solo una tabla de datos, derivando directamente el polinomio de Newton-Gregory."
        ),
        "guru": {
            "que_resuelve": (
                "Estima tasas de cambio (velocidad) y concavidad (aceleración) sobre registros discretos medidos."
            ),
            "intuicion_geometrica": (
                "Pasa una curva suave polinómica por los datos vecinos y calcula analíticamente la inclinación de su recta tangente en cualquier punto."
            ),
            "algoritmo_pasos": [
                "Se construye la matriz de diferencias progresivas ($\\Delta^k$) o retrógradas ($\\nabla^k$).",
                "Se aplica la fórmula correspondiente en función de $u = \\frac{x - x_0}{h}$ (inicio) o $v = \\frac{x - x_n}{h}$ (final).",
                "Se divide por $h$ para primera derivada y por $h^2$ para segunda derivada.",
            ],
            "profundizacion": [
                r"Existe un compromiso clásico entre error de truncamiento y error de máquina: si $h$ baja demasiado (por debajo de $10^{-8}$), la resta $\frac{f(x+h) - f(x)}{h}$ sufre **error de resta entre números casi iguales (cancelación catastrófica)** y el ruido de punto flotante amplifica el error al dividir por un $h$ minúsculo.",
            ],
            "cuando_usar": [
                "Versión ascendente al principio de una tabla; versión descendente cerca del final de la tabla.",
            ],
            "cuando_falla": [
                "Si el paso $h$ es excesivamente chico ($h < 10^{-8}$), la división amplifica la pérdida de cifras significativas.",
                "Si las mediciones tienen ruido aleatorio (sensores), derivar amplifica ese ruido drásticamente.",
            ],
        },
        "formulas_latex": [
            r"f'(x_0) \approx \frac{1}{h}\left[\Delta y_0 - \frac{1}{2}\Delta^2 y_0 + \frac{1}{3}\Delta^3 y_0 - \frac{1}{4}\Delta^4 y_0\right], \qquad f''(x_0) \approx \frac{1}{h^2}\left[\Delta^2 y_0 - \Delta^3 y_0 + \frac{11}{12}\Delta^4 y_0\right]",
            r"f'(x_n) \approx \frac{1}{h}\left[\nabla y_n + \frac{1}{2}\nabla^2 y_n + \frac{1}{3}\nabla^3 y_n + \frac{1}{4}\nabla^4 y_n\right], \qquad f''(x_n) \approx \frac{1}{h^2}\left[\nabla^2 y_n + \nabla^3 y_n + \frac{11}{12}\nabla^4 y_n\right]",
        ],
        "ejemplo_teorico": {
            "titulo": "Por qué no conviene hacer $h \\to 0$ indefinidamente en una PC",
            "enunciado": "Evaluar $\\frac{f(1+h)-f(1)}{h}$ para $f(x) = e^x$ achicando $h$ desde $10^{-1}$ hasta $10^{-14}$ (exacto: $e \\approx 2.718281828$).",
            "pasos": [
                "El error disminuye limpiamente hasta llegar a $h = 10^{-8}$ (error $5.10 \\times 10^{-9}$).",
                "Para $h \\le 10^{-9}$, $f(1+h)$ y $f(1)$ coinciden en casi todos los bits: restarlos deja basura de redondeo y el error sube a $0.0537$ en $h = 10^{-14}$.",
            ],
            "resultado_final": "Existe un paso óptimo $h$ que balancea el truncamiento matemático con la precisión del float.",
        },
        "practica_profesora": {
            "titulo": "TP6 Ej. Diferenciación: Tabla con $h = 1$ ($x = 12 \\dots 17$)",
            "enunciado": "Dada $x = [12, 13, 14, 15, 16, 17]$, $y = [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151]$, calcular $f'(12)$ (ascendente) y $f''(17)$ (descendente).",
            "datos_iniciales": {
                "x": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                "y": [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151],
            },
            "solucion_referencia": "$f'(12) = -0.0083175$; $f'(12.4) = -0.0153909$; $f''(17) = 0.011254$.",
        },
    },
    "trapecio": {
        "nombre": "Regla de los Trapecios Compuesta",
        "familia": "integracion",
        "icono": "🪜",
        "resumen_corto": (
            "Calcula áreas e integrales definidas sumando franjas verticales rematadas en línea recta (trapecios). "
            "Su gran ventaja práctica es que funciona para **cualquier cantidad de intervalos** $n$, par o impar."
        ),
        "guru": {
            "que_resuelve": (
                "Aproxima $\\int_a^b f(x)\\,dx$ sobre funciones o tablas discretas sin imponer restricciones de paridad ni divisibilidad."
            ),
            "intuicion_geometrica": (
                "Conecta cada par de nodos consecutivos $(x_i, y_i)$ y $(x_{i+1}, y_{i+1})$ con una línea recta recta y suma el área de cada trapecio resultante."
            ),
            "algoritmo_pasos": [
                "Sumás los dos valores extremos ($E = y_0 + y_n$).",
                "Sumás todos los puntos intermedios multiplicados por $2$ ($2P + 2I$).",
                "Multiplicás el total por $\\frac{h}{2}$.",
            ],
            "profundizacion": [
                r"Al aproximar con segmentos lineales, su error decrece en orden cuadrático $\mathcal{O}(h^2)$: al duplicar la cantidad de subintervalos $n$, el error se reduce aproximadamente $4$ veces.",
            ],
            "cuando_usar": [
                "Cuando la tabla tiene una cantidad impar o suelta de intervalos o datos con paso $h$ irregular.",
                "Como tramo de cierre residual en algoritmos combinados.",
            ],
            "cuando_falla": [
                "En curvas suaves requiere muchísimos más subintervalos que las reglas de Simpson para alcanzar la misma precisión.",
            ],
        },
        "formulas_latex": [
            r"A = \frac{h}{2}\left[E + 2P + 2I\right], \qquad E_a \le \frac{(b - a)^3}{12\,n^2}\max|f''(x)|",
        ],
        "ejemplo_teorico": {
            "titulo": "Polinomio en $[0, 0.8]$ con $n = 2$ ($h = 0.4$)",
            "enunciado": "Integrar $f(x) = 0.2 + 25x - 200x^2 + 675x^3 - 900x^4 + 400x^5$ en $[0, 0.8]$ (exacto: $1.640533$).",
            "pasos": [
                "Con $n=2$: $f(0)=0.2$, $f(0.4)=2.456$, $f(0.8)=0.232$.",
                "$A = \\frac{0.4}{2}(0.2 + 2(2.456) + 0.232) = 1.0688$ (error relativo $34.9\\%$).",
            ],
            "resultado_final": "Al aumentar a $n=10$, el error baja rápidamente siguiendo $\\mathcal{O}(1/n^2)$.",
        },
        "practica_profesora": {
            "titulo": "TP6 Ej. 1: Regla del Trapecio ($y = \\frac{x^2}{5}$, $h = 5$)",
            "enunciado": "Integrar $x = [0.5, 5.5, 10.5, 15.5, 20.5, 25.5]$, $y = [0.05, 6.05, 22.05, 48.05, 84.05, 130.05]$ mediante Trapecios.",
            "datos_iniciales": {
                "x": [0.5, 5.5, 10.5, 15.5, 20.5, 25.5],
                "y": [0.05, 6.05, 22.05, 48.05, 84.05, 130.05],
                "f_str": "x**2 / 5",
            },
            "solucion_referencia": "$E=130.10$, $P=54.10$, $I=106.10 \\implies A = \\frac{5}{2}[130.10 + 2(54.10) + 2(106.10)] = 1126.25$ (Exacto $= 1105.417$).",
        },
    },
    "simpson_1_3": {
        "nombre": "Regla de Simpson 1/3 Compuesta (y Motor Adaptativo)",
        "familia": "integracion",
        "icono": "🥣",
        "resumen_corto": (
            "En vez de techos planos, toma grupos de $3$ puntos consecutivos y los une con **parábolas**. "
            "Logra precisión de cuarto orden $\\mathcal{O}(h^4)$ con la misma cantidad de datos que Trapecio."
        ),
        "guru": {
            "que_resuelve": (
                "Calcula integrales de alta exactitud usando pesos alternados $(1, 4, 2, 4, \\dots, 1)$ sobre tramos pares."
            ),
            "intuicion_geometrica": (
                "Reemplaza cada pareja de subintervalos consecutivos por el área exacta debajo de un arco parabólico que atraviesa esos tres puntos."
            ),
            "algoritmo_pasos": [
                "Se comprueba que el paso $h$ sea uniforme y que el número de intervalos $n$ sea **PAR**.",
                "Se suman extremos ($E$), posiciones impares multiplicadas por $4$ ($4I$) y posiciones pares interiores por $2$ ($2P$).",
                "Si modificás $\\Delta h$ dejando un tramo final residual o un $n$ impar, nuestro motor adaptativo empalma automáticamente con Simpson 3/8 o Trapecio sin fallar.",
            ],
            "profundizacion": [
                r"Por simetría de los pesos sobre $[-h, h]$, los términos de error de tercer grado se cancelan entre sí: aunque usa parábolas de grado $2$, integra **sin ningún error** cualquier polinomio cúbico y tiene error $\mathcal{O}(h^4)$.",
            ],
            "cuando_usar": [
                "Como primera opción cuando el número de intervalos $n$ es par y los nodos están a paso constante.",
            ],
            "cuando_falla": [
                "No se puede aplicar puro sobre $n$ impar ni sobre mallas con $h$ cambiante sin combinar bloques.",
            ],
        },
        "formulas_latex": [
            r"A = \frac{h}{3}\left[E + 4I + 2P\right], \qquad E_a \le \frac{(b - a)^5}{180\,n^4}\max|f^{(4)}(x)|",
        ],
        "ejemplo_teorico": {
            "titulo": "Exactitud sobre polinomios hasta grado $3$",
            "enunciado": "Integrar un polinomio interpolante cuadrático sobre $3$ puntos equidistantes produce los pesos $\\frac{h}{3}(1,\\, 4,\\, 1)$.",
            "pasos": [
                "Como la cota de error depende de la cuarta derivada $f^{(4)}(x)$, cualquier polinomio cúbico ($f^{(4)} = 0$) se integra de manera **exacta**.",
            ],
            "resultado_final": "Orden $\\mathcal{O}(h^4)$: reducir $\\Delta h$ a la mitad multiplica la precisión por $16$.",
        },
        "practica_profesora": {
            "titulo": "TP6 Ej. 2: Simpson 1/3 en $[0.5, 20.5]$ ($n = 4$, $h = 5$)",
            "enunciado": "Integrar los primeros $5$ puntos $x = [0.5, 5.5, 10.5, 15.5, 20.5]$, $y = [0.05, 6.05, 22.05, 48.05, 84.05]$ con Simpson 1/3.",
            "datos_iniciales": {
                "x": [0.5, 5.5, 10.5, 15.5, 20.5],
                "y": [0.05, 6.05, 22.05, 48.05, 84.05],
                "f_str": "x**2 / 5",
            },
            "solucion_referencia": "$E=84.10$, $I=54.10$, $P=22.05 \\implies A = \\frac{5}{3}[84.10 + 4(54.10) + 2(22.05)] = 574.333\\dots$ (Exacta para $x^2/5$).",
        },
    },
    "simpson_3_8_combinado": {
        "nombre": "Simpson 3/8 y Esquema Combinado ($3/8 + 1/3$)",
        "familia": "integracion",
        "icono": "🧩",
        "resumen_corto": (
            "Ajusta **polinomios cúbicos** sobre bloques de $3$ subintervalos ($4$ puntos). Combinado con Simpson 1/3 "
            "resuelve tablas con cantidad impar de subintervalos sin bajar de orden $\\mathcal{O}(h^4)$."
        ),
        "guru": {
            "que_resuelve": (
                "Mantiene precisión de cuarto orden en mallas con $n$ múltiplo de $3$ o $n$ impar ($n \\ge 5$)."
            ),
            "intuicion_geometrica": (
                "Cuando $n$ es impar, toma un primer bloque de $3$ subintervalos con una cúbica ($3/8$) dejando atrás un resto de intervalos **par** para completarlo con parábolas ($1/3$)."
            ),
            "algoritmo_pasos": [
                "Si $n$ es múltiplo de $3$, aplica módulos sucesivos con coeficientes $(1, 3, 3, 1)\\frac{3h}{8}$.",
                "Si $n \\ge 5$ es impar no múltiplo de $3$, procesa los primeros $3$ intervalos con Simpson 3/8 y pasa el resto par a Simpson 1/3.",
            ],
            "profundizacion": [
                r"Cada bloque cúbico de Simpson 3/8 posee error local $-\frac{3h^5}{80} f^{(4)}(\xi)$, manteniendo exactamente el mismo orden global $\mathcal{O}(h^4)$ que Simpson 1/3.",
            ],
            "cuando_usar": [
                "Para cualquier partición uniforme de $n$ impar con al menos $5$ subintervalos o cuando $n$ es múltiplo de $3$.",
            ],
            "cuando_falla": [
                "Exige espaciamiento $h$ uniforme dentro de cada bloque donde se aplica.",
            ],
        },
        "formulas_latex": [
            r"A_{3/8} = \frac{3h}{8}\left[y_0 + 3y_1 + 3y_2 + y_3\right], \qquad A_{\text{comb}} = A_{3/8}(\text{primeros }3) + A_{1/3}(\text{restantes})",
        ],
        "ejemplo_teorico": {
            "titulo": "Estrategia híbrida cuando $n = 5$",
            "enunciado": "Para $n = 5$ subintervalos no aplica ni Simpson 1/3 puro ($n$ par) ni Simpson 3/8 puro ($n$ múltiplo de $3$).",
            "pasos": [
                "Se divide el dominio en $[x_0, x_3]$ ($3$ intervalos $\\to$ Simpson 3/8) más $[x_3, x_5]$ ($2$ intervalos $\\to$ Simpson 1/3).",
            ],
            "resultado_final": "Conserva precisión $\\mathcal{O}(h^4)$ sobre todo el intervalo completo.",
        },
        "practica_profesora": {
            "titulo": "TP6 Ej. 3: Simpson 3/8 + 1/3 en $[0.5, 25.5]$ ($n = 5$, $h = 5$)",
            "enunciado": "Integrar la tabla completa de $6$ puntos ($n = 5$ intervalos) usando Simpson 3/8 en $[0.5, 15.5]$ y Simpson 1/3 en $[15.5, 25.5]$.",
            "datos_iniciales": {
                "x": [0.5, 5.5, 10.5, 15.5, 20.5, 25.5],
                "y": [0.05, 6.05, 22.05, 48.05, 84.05, 130.05],
                "f_str": "x**2 / 5",
            },
            "solucion_referencia": "$A_1 (3/8) = 248.25$, $A_2 (1/3) = 857.1667 \\implies \\text{Total} = 1105.417$ (Exacta).",
        },
    },

    # ==========================================================================
    # 6. ECUACIONES DIFERENCIALES ORDINARIAS (EDO - PVI)
    # ==========================================================================
    "euler_heun": {
        "nombre": "Euler Clásico y Euler Modificado (Heun)",
        "familia": "edo",
        "icono": "🧭",
        "resumen_corto": (
            "Simulan paso a paso la evolución de un sistema $y' = f(x, y)$ desde un estado inicial $(x_0, y_0)$. "
            "**Euler** avanza siguiendo solo la pendiente inicial; **Heun** predice, mide en el destino y promedia."
        ),
        "guru": {
            "que_resuelve": (
                "Aproxima la trayectoria de una ecuación diferencial ordinaria con valor inicial en física, biología o simulaciones temporales."
            ),
            "intuicion_geometrica": (
                "Euler camina en línea recta durante todo un paso $\\Delta h$ usando la pendiente inicial (como avanzar en un juego por ticks sin mirar si la curva dobló). Heun hace un paso de prueba, mira la pendiente en el destino, promedia ambas y corrige."
            ),
            "algoritmo_pasos": [
                "**Euler:** $y_{i+1} = y_i + \\Delta h \\cdot f(x_i, y_i)$.",
                "**Predictor de Heun:** estima un valor tentativo $P(y_{i+1})$ usando Euler.",
                "**Corrector iterado de Heun:** promedia la pendiente inicial y la del punto predicho repetidamente hasta que el cambio sea menor que $E$.",
            ],
            "profundizacion": [
                r"Euler equivale a truncar Taylor en primer grado y acumula un error global $\mathcal{O}(h)$. Heun corresponde al método predictor-corrector trapezoidal de segundo orden $\mathcal{O}(h^2)$.",
            ],
            "cuando_usar": [
                "Euler para entender la discretización básica o con pasos $\\Delta h$ diminutos; Heun cuando necesitás orden $\\mathcal{O}(h^2)$ con poco código.",
            ],
            "cuando_falla": [
                "Con pasos $\\Delta h$ grandes, Euler se desvía rápidamente de la curva real.",
                "**Inestabilidad en dinámicas con escalas muy rápidas (Ecuaciones rígidas):** si una componente decae muy rápido frente al paso $\\Delta h$, los métodos explícitos oscilan y explotan.",
            ],
        },
        "formulas_latex": [
            r"y_{i+1} = y_i + h\,f(x_i, y_i) \;\; (\text{Euler}), \qquad C^{(m)}(y_{i+1}) = y_i + \frac{h}{2}\left[f(x_i,y_i) + f\!\left(x_{i+1}, C^{(m-1)}\right)\right] \;\; (\text{Heun})",
        ],
        "ejemplo_teorico": {
            "titulo": "Simulación discreta por ticks (Caída con fricción)",
            "enunciado": "Discretizar $\\frac{dv}{dt} = g - \\frac{c}{m}v$ avanzando con paso temporal $h = \\Delta t$.",
            "pasos": [
                "Euler multiplica la aceleración actual por $\\Delta t$ para sumar a la velocidad.",
                "Heun refina la predicción promediando las pendientes al inicio y fin de cada tick hasta cumplir $|C^{(m)} - C^{(m-1)}| < E$.",
            ],
            "resultado_final": "Heun elimina la deriva lineal típica de Euler simple.",
        },
        "practica_profesora": {
            "titulo": "TP7 Ej. 1: $y' = \\frac{6x^2}{y}$, $y(4) = 16$ en $[4.0, 4.4]$ con $\\Delta h = 0.1$",
            "enunciado": "Resolver $y' = \\frac{6x^2}{y}$ con $y(4) = 16$ hasta $x = 4.4$ ($\\Delta h = 0.1$, tolerancia de corrector $E < 0.0001$), cuya solución analítica exacta es $y(x) = 2x\\sqrt{x}$.",
            "datos_iniciales": {
                "f_str": "6*x**2 / y",
                "x0": 4.0,
                "y0": 16.0,
                "xf": 4.4,
                "h": 0.1,
                "tol_corrector": 0.0001,
                "y_exacta_str": "2*x*sqrt(x)",
            },
            "solucion_referencia": "Exacta $y(4.4) = 18.459036$. Euler: $18.44514$ (error $0.0139$). Euler Modificado: $18.45901$ (error $0.00002$).",
        },
    },
    "runge_kutta": {
        "nombre": "Métodos de Runge-Kutta ($RK2$ y $RK4$ Clásico)",
        "familia": "edo",
        "icono": "🚀",
        "resumen_corto": (
            "El estándar industrial para simular sistemas dinámicos: sondea **cuatro pendientes exploratorias** "
            "dentro de cada paso $\\Delta h$ y calcula un promedio ponderado con precisión de cuarto orden $\\mathcal{O}(h^4)$."
        ),
        "guru": {
            "que_resuelve": (
                "Resuelve EDOs no rígidas con altísima precisión desde el primer paso sin requerir derivadas analíticas extra."
            ),
            "intuicion_geometrica": (
                "En lugar de mirar solo el inicio del intervalo, $RK4$ evalúa una pendiente en el arranque ($k_1$), dos en la mitad del tramo ($k_2, k_3$) y una al final ($k_4$), dándole peso doble a las pendientes centrales."
            ),
            "algoritmo_pasos": [
                "Calculás $k_1$ al comienzo, $k_2$ en $x_i + h/2$ usando $k_1$, $k_3$ en el punto medio usando $k_2$ y $k_4$ en $x_i + h$ usando $k_3$.",
                "Combinás las cuatro sondas con pesos $(1 + 2 + 2 + 1)/6$, equivalente directo a Simpson 1/3.",
                "No depende de pasos anteriores: arranca desde $i = 0$ en un solo bucle `for`.",
            ],
            "profundizacion": [
                r"Cuando $f(x, y)$ depende únicamente de $x$, $k_2 = k_3 = h f(x_i + h/2)$ y la fórmula de $RK4$ se colapsa algebraicamente en la Regla de Simpson 1/3. Su error local por paso es $\mathcal{O}(h^5)$ y el acumulado global $\mathcal{O}(h^4)$.",
            ],
            "cuando_usar": [
                "Como algoritmo principal por defecto para casi cualquier ecuación diferencial.",
                "Para generar los primeros $4$ puntos de arranque exigidos por métodos multipaso como Milne.",
            ],
            "cuando_falla": [
                "Ejecuta $4$ llamadas a la función `f(x, y)` por cada paso, siendo más costoso por iteración que un método multipaso si `f` es muy pesada.",
            ],
        },
        "formulas_latex": [
            r"k_1 = h f(x_i, y_i), \;\; k_2 = h f\!\left(x_i+\frac{h}{2}, y_i+\frac{k_1}{2}\right), \;\; k_3 = h f\!\left(x_i+\frac{h}{2}, y_i+\frac{k_2}{2}\right), \;\; k_4 = h f(x_i+h, y_i+k_3)",
            r"y_{i+1} = y_i + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)",
        ],
        "ejemplo_teorico": {
            "titulo": "Equivalencia entre $RK4$ y Simpson 1/3",
            "enunciado": "Cuando $f(x, y)$ no depende de $y$, la suma ponderada $(1, 2, 2, 1)/6$ reproduce exactamente la integración parabólica de Simpson.",
            "pasos": [
                "$k_2$ y $k_3$ muestrean el punto medio $x_i + \\frac{h}{2}$ sumando peso central $4/6$.",
                "Los extremos reciben $1/6$ cada uno.",
            ],
            "resultado_final": "Error local $\\mathcal{O}(h^5)$ y error acumulado global $\\mathcal{O}(h^4)$.",
        },
        "practica_profesora": {
            "titulo": "TP7 Ej. 2: $RK2$ y $RK4$ en $y' = \\frac{6x^2}{y}$, $y(4) = 16$ en $[4.0, 4.3]$ ($\\Delta h = 0.1$)",
            "enunciado": "Calcular $y(4.1)$, $y(4.2)$ e $y(4.3)$ usando $RK2$ y $RK4$ para el problema $y' = \\frac{6x^2}{y}$, $y(4) = 16$.",
            "datos_iniciales": {
                "f_str": "6*x**2 / y",
                "x0": 4.0,
                "y0": 16.0,
                "xf": 4.3,
                "h": 0.1,
                "y_exacta_str": "2*x*sqrt(x)",
            },
            "solucion_referencia": "Exacta $y(4.3) = 17.83333956$. $RK2$: $17.83352$. $RK4$: $17.83333$ (coincide a $5$ decimales).",
        },
    },
    "milne": {
        "nombre": "Método Multipaso Predictor-Corrector de Milne",
        "familia": "edo",
        "icono": "🔮",
        "resumen_corto": (
            "En vez de evaluar $4$ veces la función por paso como $RK4$, reutiliza el historial guardado en caché "
            "de los $4$ puntos anteriores para proyectar el siguiente valor y corregirlo en $2$ evaluaciones."
        ),
        "guru": {
            "que_resuelve": (
                "Acelera simulaciones largas con precisión $\\mathcal{O}(h^4)$ cuando evaluar `f(x, y)` resulta costoso."
            ),
            "intuicion_geometrica": (
                "Usa una ventana deslizante con los últimos $4$ pasos para extrapolar un valor futuro (Predictor) y luego lo refina integrando sobre los últimos $3$ puntos con Simpson (Corrector)."
            ),
            "algoritmo_pasos": [
                "Fase de arranque: calcula los primeros $4$ nodos ($y_0, y_1, y_2, y_3$) mediante $RK4$.",
                "Fase Predictor: proyecta $P(y_{i+1})$ usando las pendientes pasadas $y'_{i-2}, y'_{i-1}, y'_i$.",
                "Fase Corrector: evalúa la pendiente predicha y corrige $C(y_{i+1})$ dos veces según esquema de cátedra.",
            ],
            "profundizacion": [
                r"Además de ahorrar evaluaciones de $f$ frente a $RK4$, comparar la diferencia $|P(y_{i+1}) - C(y_{i+1})|$ provee una estimación gratuita del error local en tiempo real para ajustar el paso $\Delta h$.",
            ],
            "cuando_usar": [
                "En simulaciones largas donde tenés al menos $4$ subintervalos ($n \\ge 4$) y evaluar `f(x, y)` consume mucho CPU.",
            ],
            "cuando_falla": [
                "**No puede arrancar solo:** si el paso $\\Delta h$ es muy grande y produce menos de $4$ intervalos en el rango, el método falla antes de empezar.",
                "En ecuaciones con decaimiento exponencial fuerte ($y' = -\\lambda y$) puede desarrollar oscilaciones parásitas crecientes si $\\Delta h$ no es lo bastante chico.",
            ],
        },
        "formulas_latex": [
            r"P(y_{i+1}) = y_{i-3} + \frac{4h}{3}\left(2y'_i - y'_{i-1} + 2y'_{i-2}\right)",
            r"C(y_{i+1}) = y_{i-1} + \frac{h}{3}\left(y'_{i-1} + 4y'_i + f\!\left(x_{i+1}, y^{(m-1)}_{i+1}\right)\right)",
        ],
        "ejemplo_teorico": {
            "titulo": "Esquema de Doble Corrección con arranque $RK4$",
            "enunciado": "Generar $(y_0, y_1, y_2, y_3)$ mediante $RK4$, proyectar $P(y_4)$ y ejecutar dos correcciones $C$ y $C^1$.",
            "pasos": [
                "Verificar que la partición cubra al menos $4$ pasos en $[x_0, x_f]$.",
                "Aplicar el predictor con $(2y'_i - y'_{i-1} + 2y'_{i-2})$ y refinar iterando el corrector simpsoniano.",
            ],
            "resultado_final": "Alcanza exactitud de orden $\\mathcal{O}(h^4)$ con la mitad de llamadas a `f(x, y)` por paso.",
        },
        "practica_profesora": {
            "titulo": "TP7 Ej. 3: Milne en $y' = \\frac{6x^2}{y}$, $y(4) = 16$ en $[4.0, 4.4]$ ($\\Delta h = 0.1$)",
            "enunciado": "Arrancando $y(4.0 \\dots 4.3)$ con $RK4$, calcular $y(4.4)$ por Milne con doble corrección frente al valor exacto $18.459036$.",
            "datos_iniciales": {
                "f_str": "6*x**2 / y",
                "x0": 4.0,
                "y0": 16.0,
                "xf": 4.4,
                "h": 0.1,
                "y_exacta_str": "2*x*sqrt(x)",
            },
            "solucion_referencia": "$P(y_4) = 18.45903 \\to C(y_4) = 18.45902 \\to C^1(y_4) = 18.45902$ (exacta: $18.45903573$).",
        },
    },
}


# ==============================================================================
# FICHAS TÉCNICAS Y REGLAS PARA ARENA VERSUS (LATEX Y LENGUAJE DE SISTEMAS)
# ==============================================================================

FICHAS_VERSUS: dict[str, dict[str, Any]] = {
    "raices": {
        "titulo": "📉 Raíces de Ecuaciones No Lineales",
        "descripcion": "Enfrenta algoritmos acotados (convergencia asegurada) contra métodos abiertos (máxima velocidad) sobre una misma ecuación $f(x) = 0$.",
        "metodos": {
            "biseccion": {
                "nombre": "Bisección (Intervalo Medio)",
                "orden": "Lineal (Orden $1$, razón $0.5$)",
                "costo": "$1$ evaluación de $f(x)$ por paso",
                "estabilidad": "Máxima (siempre converge si cruza de signo)",
                "requisitos": "Curva continua con signos opuestos $f(a)\\cdot f(b) < 0$",
                "pros": ["Imposible que diverja dentro del intervalo", "Cantidad exacta de iteraciones calculable antes del bucle", "No necesita derivadas"],
                "contras": ["Lento: solo recorta el intervalo a la mitad", "No detecta raíces tangentes que rebotan sin cambiar de signo"],
            },
            "regula_falsi": {
                "nombre": "Interpolación Lineal (Regula Falsi)",
                "orden": "Lineal / Superlineal ($\\sim 1.0 - 1.6$)",
                "costo": "$1$ evaluación nueva de $f(x)$ por paso",
                "estabilidad": "Alta (mantiene la raíz encerrada)",
                "requisitos": "Función monótona en $(a, b)$ y $f(a)\\cdot f(b) < 0$",
                "pros": ["Aprovecha las magnitudes de $f(a)$ y $f(b)$", "Suele ganar a Bisección en curvas poco pronunciadas"],
                "contras": ["Si un extremo queda anclado (pivote fijo), puede volverse más lento que Bisección"],
            },
            "newton_raphson": {
                "nombre": "Newton-Raphson (Tangentes)",
                "orden": "Cuadrático (Orden $2$)",
                "costo": "$2$ evaluaciones por paso: $f(x)$ y $f'(x)$",
                "estabilidad": "Condicional (depende del punto de arranque $x_0$)",
                "requisitos": "$f$ derivable, $f'(x) \\ne 0$ y chequeo de arranque $f(x_0)\\cdot f''(x_0) > 0$ (Fourier)",
                "pros": ["Convergencia explosiva: duplica los dígitos exactos en cada vuelta", "Ideal para altísima precisión"],
                "contras": ["Se dispara o divide por cero si $f'(x) \\approx 0$", "Exige calcular la derivada $f'(x)$"],
            },
            "aitken": {
                "nombre": "Punto Fijo + Aceleración de Aitken",
                "orden": "Lineal acelerado ($\\sim$ Superlineal)",
                "costo": "$3$ pasos por ciclo completo",
                "estabilidad": "Sujeto a pendiente contractiva $|g'(x)| < 1$",
                "requisitos": "Despeje iterativo $x = g(x)$ con $|g'(x)| < 1$",
                "pros": ["Acelera drásticamente secuencias lentas sin usar derivadas en el ciclo", "Natural cuando el problema ya viene en formato `x = g(x)`"],
                "contras": ["Diverge si $|g'(x)| \\ge 1$", "El éxito depende de cómo despejes $g(x)$"],
            },
        },
        "expectativas": {
            ("biseccion", "newton_raphson"): {
                "que_esperamos": "Cumpliendo un arranque seguro (Fourier: $f(x_0)\\cdot f''(x_0) > 0$), **Newton-Raphson** termina en muy pocas vueltas ($3$ a $5$ iteraciones) mientras **Bisección** avanza paso a paso reduciendo a la mitad.",
                "excepciones": "Si $x_0$ cae cerca de una meseta plana donde $f'(x) \\approx 0$, Newton-Raphson puede oscilar o dispararse fuera del rango mientras Bisección sigue convergiendo sin problemas.",
            },
            ("biseccion", "regula_falsi"): {
                "que_esperamos": "En curvas suaves y casi rectas, **Regula Falsi** traza una secante que cae mucho más cerca de la raíz y gana en iteraciones a **Bisección**.",
                "excepciones": "En funciones muy empinadas (como $x^{10} - 1$ o exponenciales fuertes), Regula Falsi deja un extremo estancado y avanza tan lento que Bisección lo supera claramente.",
            },
            ("newton_raphson", "regula_falsi"): {
                "que_esperamos": "**Newton-Raphson** (orden cuadrático) vence ampliamente en velocidad de convergencia y error final a **Regula Falsi**.",
                "excepciones": "Si $f'(x)$ es difícil de calcular o se acerca a cero cerca de la raíz, Regula Falsi ofrece mayor seguridad al no poder escapar del intervalo.",
            },
            ("aitken", "biseccion"): {
                "que_esperamos": "Con un despeje estable donde las distancias se encogen ($|g'(x)| < 1$), **Punto Fijo + Aitken** salta directo hacia el cruce en pocos ciclos frente al corte rígido de **Bisección**.",
                "excepciones": "Si $|g'(x)| \\ge 1$ cerca de $x_0$, Punto Fijo explota inmediatamente; Bisección no sufre esa fragilidad.",
            },
            ("aitken", "newton_raphson"): {
                "que_esperamos": "**Newton-Raphson** conserva la ventaja al tener orden $2$ directo sobre $f(x) = 0$ sin requerir despejar $g(x)$.",
                "excepciones": "Si la pendiente $|g'(x)|$ es muy cercana a cero cerca de la solución y evaluar $f'(x)$ es costoso, Aitken compite muy de cerca.",
            },
            ("aitken", "regula_falsi"): {
                "que_esperamos": "Si $|g'(x)| < 0.3$, **Aitken** suele cerrar en apenas $2$ o $3$ ciclos; si Regula Falsi no queda anclado en un pivote, avanza firme con la secante.",
                "excepciones": "Aitken diverge si falla la condición de pendiente $< 1$, mientras Regula Falsi nunca suelta la raíz del intervalo.",
            },
        },
    },
    "sistemas": {
        "titulo": "📐 Sistemas de Ecuaciones Lineales ($A x = b$)",
        "descripcion": "Compara algoritmos directos (Gauss, Gauss-Jordan, $LU$) contra métodos iterativos en bucle (Gauss-Seidel).",
        "metodos": {
            "gauss": {
                "nombre": "Eliminación de Gauss (Pivoteo Parcial)",
                "orden": "Directo (finito)",
                "costo": "$\\approx \\frac{1}{3} n^3$ operaciones",
                "estabilidad": "Alta con intercambio preventivo de filas (pivoteo)",
                "requisitos": "Sistema compatible determinado ($\\det(A) \\ne 0$)",
                "pros": ["El algoritmo directo más eficiente para un único vector $b$", "El pivoteo parcial evita ceros diagonales y errores graves"],
                "contras": ["Si cambiás $b$, tenés que repetir toda la eliminación desde cero", "Costoso en memoria si la matriz es gigante y rala"],
            },
            "gauss_jordan": {
                "nombre": "Método de Gauss-Jordan",
                "orden": "Directo (finito)",
                "costo": "$\\approx \\frac{1}{2} n^3$ operaciones",
                "estabilidad": "Alta con pivoteo parcial",
                "requisitos": "$\\det(A) \\ne 0$",
                "pros": ["Entrega la solución directa sin bucle hacia atrás", "Permite obtener $A^{-1}$ en el mismo procedimiento"],
                "contras": ["Consume $\\sim 50\\%$ más operaciones que Gauss para resolver un único sistema"],
            },
            "lu": {
                "nombre": "Factorización $LU$ (Doolittle)",
                "orden": "Directo (finito)",
                "costo": "$\\approx \\frac{1}{3} n^3$ cachear $L, U$ + $\\mathcal{O}(n^2)$ por cada nuevo $b$",
                "estabilidad": "Media (sin pivoteo requiere diagonales $U_{kk} \\ne 0$)",
                "requisitos": "Pivotes diagonales intermedios no nulos ($U_{kk} \\ne 0$)",
                "pros": ["Imbatible cuando se resuelven muchos vectores $b$ con la misma matriz $A$"],
                "contras": ["Sin permutar filas, falla si aparece un cero en la diagonal intermedia de $U$"],
            },
            "gauss_seidel": {
                "nombre": "Gauss-Seidel (Iterativo)",
                "orden": "Iterativo (convergencia lineal)",
                "costo": "$\\mathcal{O}(n^2)$ por vuelta ($\\mathcal{O}(\\text{nnz})$ en matrices ralas)",
                "estabilidad": "Sujeta a Dominancia Diagonal",
                "requisitos": "$a_{ii} \\ne 0$ e idealmente peso diagonal dominante en cada fila",
                "pros": ["No altera ni llena de ceros no nulos la matriz $A$", "Permite cortar el `while` con cualquier tolerancia $E$"],
                "contras": ["Puede diverger si la matriz no tiene dominancia diagonal", "No garantiza solución exacta en una cantidad fija de pasos"],
            },
        },
        "expectativas": {
            ("gauss", "gauss_jordan"): {
                "que_esperamos": "Ambos calculan exactamente la misma solución $x$, pero **Eliminación de Gauss** realiza menos operaciones en CPU que **Gauss-Jordan**.",
                "excepciones": "Si además de $x$ necesitás la matriz inversa $A^{-1}$ o resolvés varios vectores $b$ simultáneos, Gauss-Jordan resulta más práctico.",
            },
            ("gauss", "gauss_seidel"): {
                "que_esperamos": "En sistemas chicos ($3 \\times 3$, $4 \\times 4$), **Gauss** obtiene el resultado exacto directo; en matrices con dominancia diagonal, **Gauss-Seidel** cae rápidamente hacia la solución en pocas iteraciones.",
                "excepciones": "Si desordenás las filas perdiendo dominancia diagonal, Gauss-Seidel explota al infinito mientras Gauss con pivoteo parcial resuelve sin inmutarse.",
            },
            ("gauss", "lu"): {
                "que_esperamos": "La matriz triangular $U$ de Doolittle coincide exactamente con la salida escalonada de **Gauss** (sin pivoteo), y ambos dan idéntico resultado inicial.",
                "excepciones": "Si aparece un $0$ en la diagonal, Gauss intercambia filas automáticamente; Doolittle puro sin permutaciones falla por división por cero.",
            },
            ("gauss_jordan", "gauss_seidel"): {
                "que_esperamos": "**Gauss-Jordan** entrega el resultado exacto en $n$ pasos de columna; **Gauss-Seidel** aproxima iterativamente según la tolerancia $E$.",
                "excepciones": "Sin dominancia diagonal, Gauss-Seidel no converge; en cambio, sobre redes de decenas de miles de nodos casi vacíos, Gauss-Jordan agota la RAM.",
            },
            ("gauss_jordan", "lu"): {
                "que_esperamos": "**Factorización $LU$** ejecuta alrededor de un $33\\%$ menos de operaciones que **Gauss-Jordan** y permite reutilizar $A$ en memoria.",
                "excepciones": "Si querés extraer la matriz inversa $A^{-1}$ directamente de la matriz aumentada $[A \\mid I]$, Gauss-Jordan es inmediato.",
            },
            ("gauss_seidel", "lu"): {
                "que_esperamos": "**$LU$** resuelve en tiempo fijo independiente del arranque $X_0$; **Gauss-Seidel** depende tanto de $X_0$ como de la dominancia diagonal.",
                "excepciones": "En mallas masivas llenas de ceros, $LU$ crea valores no nulos en posiciones originalmente vacías (fill-in) consumiendo RAM, mientras Gauss-Seidel solo opera sobre los valores existentes.",
            },
        },
    },
    "autovalores": {
        "titulo": "⚡ Autovalores y Espectro Matricial",
        "descripcion": "Compara la búsqueda iterativa de extremos (**Potencias**) frente al cálculo completo del polinomio característico (**Faddeev-Leverrier**).",
        "metodos": {
            "potencias_max": {
                "nombre": "Potencias Directas ($\\lambda_{\\max}$)",
                "orden": "Iterativo (razón $|\\lambda_2 / \\lambda_1|$)",
                "costo": "$\\mathcal{O}(n^2)$ por iteración",
                "estabilidad": "Alta para el autovalor dominante",
                "requisitos": "$|\\lambda_1| > |\\lambda_2|$ y $X_0$ no perpendicular al autovector dominante",
                "pros": ["Muy veloz y escalable para matrices enormes", "Calcula $\\lambda_{\\max}$ y su autovector sin hallar raíces de polinomios"],
                "contras": ["Solo encuentra UN autovalor (el de mayor magnitud)", "Lento si $|\\lambda_1|$ y $|\\lambda_2|$ tienen tamaños muy parecidos"],
            },
            "potencias_min": {
                "nombre": "Potencias Inversas ($\\lambda_{\\min}$)",
                "orden": "Iterativo sobre $A^{-1}$",
                "costo": "Requiere $A^{-1}$ + $\\mathcal{O}(n^2)$ por paso",
                "estabilidad": "Sensible cuando $\\lambda_{\\min}$ está muy cerca de $0$",
                "requisitos": "$\\det(A) \\ne 0$ ($A$ debe admitir inversa)",
                "pros": ["Aísla el modo propio menor ($\\lambda_{\\min}$), clave en estabilidad"],
                "contras": ["Exige invertir o factorizar $A$ previamente"],
            },
            "faddeev": {
                "nombre": "Faddeev-Leverrier (Espectro Total)",
                "orden": "Directo exacto ($n$ pasos matriciales)",
                "costo": "$\\mathcal{O}(n^4)$ — elevado cuando $n$ crece",
                "estabilidad": "Buena para $n \\le 6$; sensible en grados altos",
                "requisitos": "Matriz cuadrada $n \\times n$",
                "pros": ["Obtiene TODOS los autovalores, el polinomio $P(\\lambda)$, $\\det(A)$ y $A^{-1}$ en un mismo flujo"],
                "contras": ["Costo $\\mathcal{O}(n^4)$ muy alto para matrices grandes", "Hallar raíces de polinomios de grado grande es inestable en punto flotante"],
            },
        },
        "expectativas": {
            ("faddeev", "potencias_max"): {
                "que_esperamos": "**Faddeev-Leverrier** calcula el espectro completo, mientras **Potencias Directas** converge rápidamente al mayor autovalor $|\\lambda_{\\max}|$ coincidiendo con la raíz extrema de Faddeev.",
                "excepciones": "Si la matriz tiene dos autovalores dominantes opuestos, empatados en módulo o complejos conjugados, el bucle de Potencias oscila sin fijar valor mientras Faddeev sí construye el polinomio.",
            },
            ("faddeev", "potencias_min"): {
                "que_esperamos": "**Potencias Inversas** converge hacia el autovalor más cercano a cero visible en el gráfico de $P(\\lambda)$ generado por **Faddeev-Leverrier**.",
                "excepciones": "Si $A$ es singular ($\\lambda = 0$ es raíz, $b_n = 0$), Potencias Inversas no puede invertir $A$, mientras Faddeev lo detecta viendo $b_n = 0$.",
            },
            ("potencias_max", "potencias_min"): {
                "que_esperamos": "**Potencias Directas** encuentra la cota superior $|\\lambda_{\\max}|$ y **Potencias Inversas** la cota inferior $|\\lambda_{\\min}|$, dando el rango completo de amplificación $\\kappa = |\\lambda_{\\max} / \\lambda_{\\min}|$.",
                "excepciones": "Si $X_0$ arranca exactamente ortogonal al autovector extremo buscado, el algoritmo converge al segundo autovalor en magnitud en lugar del verdadero extremo.",
            },
        },
    },
    "interpolacion": {
        "titulo": "📈 Interpolación Polinomial",
        "descripcion": "Compara diferencias finitas en los bordes (**Newton-Gregory**), centrales (**Gauss**) y nodos arbitrarios (**Lagrange**).",
        "metodos": {
            "lagrange": {
                "nombre": "Polinomio de Lagrange",
                "orden": "Exacto de grado $\\le n-1$",
                "costo": "$\\mathcal{O}(n^2)$ por cada punto evaluado",
                "estabilidad": "Válido para cualquier distancia entre $x_i$",
                "requisitos": "Valores de $x_i$ todos distintos entre sí",
                "pros": ["Funciona perfectamente con nodos NO equidistantes ($h$ variable)", "Expresión directa y simétrica"],
                "contras": ["Al añadir un dato nuevo debés recomputar todas las bases $L_i(x)$"],
            },
            "gregory_asc": {
                "nombre": "Newton-Gregory Ascendente",
                "orden": "Exacto de grado $\\le n-1$",
                "costo": "Tabla $\\mathcal{O}(n^2)$ + evaluación $\\mathcal{O}(n)$",
                "estabilidad": "Óptima al comienzo de la tabla (cerca de $x_0$)",
                "requisitos": "Nodos estrictamente EQUIDISTANTES ($h$ constante)",
                "pros": ["Permite agregar datos al final sin tirar la tabla anterior", "Las columnas nulas revelan automáticamente el grado del polinomio"],
                "contras": ["SE INVALIDA por completo si alterás un $x_i$ rompiendo la equidistancia"],
            },
            "gregory_desc": {
                "nombre": "Newton-Gregory Descendente",
                "orden": "Exacto de grado $\\le n-1$",
                "costo": "Tabla $\\mathcal{O}(n^2)$ + evaluación $\\mathcal{O}(n)$",
                "estabilidad": "Óptima al final de la tabla (cerca de $x_n$)",
                "requisitos": "Nodos estrictamente EQUIDISTANTES ($h$ constante)",
                "pros": ["Mayor precisión cerca de los últimos datos registrados ($x_n$)"],
                "contras": ["NO es válido si los $x_i$ tienen paso irregular"],
            },
            "gauss_prog": {
                "nombre": "Fórmula de Gauss Progresiva",
                "orden": "Exacto de grado $\\le n-1$",
                "costo": "Tabla $\\mathcal{O}(n^2)$ + zig-zag central $\\mathcal{O}(n)$",
                "estabilidad": "Óptima en el centro con $0 \\le u \\le 0.5$",
                "requisitos": "Nodos estrictamente EQUIDISTANTES ($h$ constante)",
                "pros": ["Minimiza $|u|$ en el núcleo central alternando diferencias hacia abajo primero"],
                "contras": ["Exige paso constante $h$ y pierde ventaja en los bordes de la tabla"],
            },
            "gauss_reg": {
                "nombre": "Fórmula de Gauss Regresiva",
                "orden": "Exacto de grado $\\le n-1$",
                "costo": "Tabla $\\mathcal{O}(n^2)$ + zig-zag central $\\mathcal{O}(n)$",
                "estabilidad": "Óptima en el centro con $-0.5 \\le u < 0$",
                "requisitos": "Nodos estrictamente EQUIDISTANTES ($h$ constante)",
                "pros": ["Rodea simétricamente a $x^*$ a la izquierda del nodo central subiendo primero en zig-zag"],
                "contras": ["Exige paso constante $h$ y pierde ventaja cerca de los extremos $x_0, x_n$"],
            },
        },
        "expectativas": {
            ("gregory_asc", "lagrange"): {
                "que_esperamos": "Cuando las abscisas $x_i$ tienen paso $h$ constante, todos los métodos entregan exactamente la misma curva (**Teorema de Unicidad del Polinomio Interpolador**).",
                "excepciones": "Si editás un solo nodo $x_i$ perdiendo la equidistancia, Newton-Gregory deja de ser válido y Lagrange sigue exacto. Además, con más de $10$ puntos equidistantes aparece vibración severa en los bordes (**Fenómeno de Runge**).",
            },
            ("gregory_desc", "lagrange"): {
                "que_esperamos": "En una tabla con paso constante $h$ generan el mismo polinomio único $P(x)$ e idéntico valor evaluado.",
                "excepciones": "Ante puntos de medición con intervalos desiguales, solo Lagrange se mantiene válido.",
            },
            ("gregory_asc", "gregory_desc"): {
                "que_esperamos": "Al usar todos los nodos disponibles producen exactamente el mismo polinomio algebraico.",
                "excepciones": "Si truncás el polinomio a grados más bajos ($k < n-1$), **Ascendente** es mucho más fiel cerca de $x_0$ y **Descendente** brilla cerca de $x_n$.",
            },
            ("gauss_prog", "gauss_reg"): {
                "que_esperamos": "En grado completo $k = n-1$ construyen exactamente el mismo polinomio único sobre los mismos nodos centrales.",
                "excepciones": "Al truncar ($k < n-1$) en el intervalo central, **Gauss Progresiva** favorece $u \\ge 0$ (primer paso abajo) y **Gauss Regresiva** favorece $u < 0$ (primer paso arriba).",
            },
            ("gauss_prog", "gregory_asc"): {
                "que_esperamos": "Coinciden idénticamente en $P_{n-1}(x)$ cuando se usan todas las columnas de la tabla equidistante.",
                "excepciones": "Con truncamiento parcial ($k < n-1$), **Gauss Progresiva** supera ampliamente a Ascendente cuando $x^*$ está en el centro ($0 \\le u \\le 0.5$), mientras Ascendente gana cerca de $x_0$.",
            },
            ("gauss_prog", "gregory_desc"): {
                "que_esperamos": "Idéntico polinomio completo de grado $n-1$ con distinta parametrización.",
                "excepciones": "Al truncar, **Gauss Progresiva** domina el centro de la tabla y **Descendente** el extremo final $x_n$.",
            },
            ("gauss_prog", "lagrange"): {
                "que_esperamos": "Sobre una tabla equidistante reproducen exactamente la misma curva interpolante.",
                "excepciones": "Si alterás cualquier $x_i$ rompiendo el paso uniforme $h$, Gauss se invalida y solo Lagrange continúa siendo exacto.",
            },
            ("gauss_reg", "gregory_asc"): {
                "que_esperamos": "Producen idéntico polinomio completo $P_{n-1}(x)$ sobre mallas equiespaciadas.",
                "excepciones": "Al truncar a grados bajos, **Gauss Regresiva** es más precisa inmediatamente a la izquierda del nodo central ($-0.5 \\le u < 0$).",
            },
            ("gauss_reg", "gregory_desc"): {
                "que_esperamos": "Idéntico polinomio completo cuando se utiliza toda la tabla equidistante.",
                "excepciones": "Al truncar, **Gauss Regresiva** prioriza el entorno del nodo central mientras **Descendente** prioriza el borde final $x_n$.",
            },
            ("gauss_reg", "lagrange"): {
                "que_esperamos": "Idéntico valor interpolado bajo paso $h$ constante en grado completo.",
                "excepciones": "Ante nodos con espaciamiento variable, solo **Lagrange** sigue siendo aplicable.",
            },
        },
    },
    "integracion": {
        "titulo": "📊 Integración Numérica (Cuadratura)",
        "descripcion": "Compara reglas con segmentos rectos (**Trapecio**) frente a arcos parabólicos/cúbicos (**Simpson 1/3 y 3/8**) ajustando el paso $\\Delta h$.",
        "metodos": {
            "trapecio": {
                "nombre": "Regla del Trapecio Compuesto",
                "orden": "$\\mathcal{O}(h^2)$ — Exacto hasta grado $1$",
                "costo": "$n + 1$ evaluaciones sin restricciones de paridad",
                "estabilidad": "Muy alta incluso en curvas con picos o datos ruidosos",
                "requisitos": "$n \\ge 1$ subintervalos",
                "pros": ["Funciona para CUALQUIER cantidad $n$ (par o impar)", "Robusto frente a cambios bruscos o esquinas en los datos"],
                "contras": ["Error $\\mathcal{O}(h^2)$ mucho más alto que Simpson sobre funciones suaves"],
            },
            "simpson_1_3": {
                "nombre": "Simpson 1/3 (y Adaptativo ante $\\Delta h$)",
                "orden": "$\\mathcal{O}(h^4)$ — Exacto hasta grado $3$",
                "costo": "Los mismos nodos que Trapecio pero muchísima más exactitud",
                "estabilidad": "Excelente en curvas suaves",
                "requisitos": "Cantidad PAR de subintervalos $n$ (si cambia a impar requiere adaptar con 3/8 o Trapecio)",
                "pros": ["Orden $\\mathcal{O}(h^4)$: achicar $\\Delta h$ a la mitad reduce el error $16$ veces", "Exacta incluso sobre polinomios cúbicos"],
                "contras": ["No aplica puro si $n$ es impar o si $\\Delta h$ varía a mitad de la tabla"],
            },
            "simpson_3_8": {
                "nombre": "Simpson 3/8 / Combinado",
                "orden": "$\\mathcal{O}(h^4)$ — Exacto hasta grado $3$",
                "costo": "Bloques de $3$ subintervalos más remanente",
                "estabilidad": "Excelente en mallas impares",
                "requisitos": "$n$ múltiplo de $3$ (puro) o $n \\ge 5$ impar (combinado con 1/3)",
                "pros": ["Resuelve mallas impares manteniendo precisión global de cuarto orden $\\mathcal{O}(h^4)$"],
                "contras": ["Ligeramente más complejo que Simpson 1/3 puro cuando $n$ ya es par"],
            },
        },
        "expectativas": {
            ("simpson_1_3", "trapecio"): {
                "que_esperamos": "Sobre cualquier función curva suave, **Simpson 1/3** ($\\mathcal{O}(h^4)$) obtiene errores muchísimo menores que **Trapecio** ($\\mathcal{O}(h^2)$) evaluando exactamente los mismos puntos.",
                "excepciones": "1) Si al variar $\\Delta h$ el total de intervalos $n$ pasa a ser impar, Simpson 1/3 puro se detiene y requiere combinarse con Simpson 3/8 o Trapecio. 2) Sobre funciones lineales a tramos o con esquinas agudas, Trapecio evita el sobrepaso curvo de las parábolas.",
            },
            ("simpson_3_8", "trapecio"): {
                "que_esperamos": "**Simpson 3/8 (o Combinado)** supera holgadamente a **Trapecio** en precisión ($\\mathcal{O}(h^4)$ vs. $\\mathcal{O}(h^2)$).",
                "excepciones": "Con $n=1$ o $n=2$ subintervalos sueltos no alcanza para completar un bloque cúbico de $3/8$ ($4$ puntos).",
            },
            ("simpson_1_3", "simpson_3_8"): {
                "que_esperamos": "Ambos tienen orden cuarto $\\mathcal{O}(h^4)$ y resultados muy cercanos, con leve ventaja de cota para 1/3 ($1/180$ frente a $1/80$ al comparar igual $n$).",
                "excepciones": "Cuando $n$ es impar ($n = 5, 7, \\dots$), Simpson 1/3 necesita que un bloque de 3/8 absorba $3$ subintervalos para poder operar sobre los pares restantes.",
            },
        },
    },
    "edo": {
        "titulo": "⏱️ Ecuaciones Diferenciales Ordinarias (PVI)",
        "descripcion": "Compara simuladores paso a paso (**Euler, Heun, $RK2, RK4$**) contra predicción con historial (**Milne**) variando $\\Delta h$.",
        "metodos": {
            "euler": {
                "nombre": "Método de Euler Clásico",
                "orden": "$\\mathcal{O}(h)$ — Orden $1$",
                "costo": "$1$ evaluación de $f(x, y)$ por paso",
                "estabilidad": "Limitada (acumula error rápido si $\\Delta h$ no es chico)",
                "requisitos": "Estado inicial $(x_0, y_0)$ y paso $\\Delta h$",
                "pros": ["Muy simple: un único cálculo lineal por tick"],
                "contras": ["Error global $\\mathcal{O}(h)$ elevado; exige pasos muy chicos"],
            },
            "heun": {
                "nombre": "Euler Modificado (Heun Iterado)",
                "orden": "$\\mathcal{O}(h^2)$ — Orden $2$",
                "costo": "$2$ o más evaluaciones por paso según tolerancia",
                "estabilidad": "Buena",
                "requisitos": "$(x_0, y_0)$, paso $\\Delta h$ y tolerancia de corrector",
                "pros": ["Mejora enormemente frente a Euler con bajo costo extra"],
                "contras": ["Superado por $RK4$ o Milne cuando se exige alta precisión"],
            },
            "rk2": {
                "nombre": "Runge-Kutta Orden 2 ($RK2$)",
                "orden": "$\\mathcal{O}(h^2)$ — Orden $2$",
                "costo": "$2$ evaluaciones de $f(x, y)$ por paso",
                "estabilidad": "Buena",
                "requisitos": "$(x_0, y_0)$ y paso $\\Delta h$",
                "pros": ["Autoiniciable desde $i = 0$ con buen balance costo/precisión"],
                "contras": ["Menor orden que $RK4$ para pasos grandes"],
            },
            "rk4": {
                "nombre": "Runge-Kutta Clásico ($RK4$)",
                "orden": "$\\mathcal{O}(h^4)$ — Orden $4$",
                "costo": "$4$ evaluaciones de $f(x, y)$ por paso",
                "estabilidad": "Muy alta en ecuaciones no rígidas",
                "requisitos": "$(x_0, y_0)$ y paso $\\Delta h$",
                "pros": ["Máxima precisión $\\mathcal{O}(h^4)$ desde el primer paso", "Autoiniciable sin requerir historial previo"],
                "contras": ["Ejecuta $4$ llamados a `f(x, y)` por cada avance"],
            },
            "milne": {
                "nombre": "Predictor-Corrector de Milne",
                "orden": "$\\mathcal{O}(h^4)$ — Orden $4$ (Multipaso)",
                "costo": "$\\sim 2\\text{ a }3$ evaluaciones por paso (luego del arranque)",
                "estabilidad": "Puede volverse oscilatorio si $\\Delta h$ es grande",
                "requisitos": "Al menos $4$ subintervalos ($n \\ge 4$) precalculados por $RK4$",
                "pros": ["Precisión de cuarto orden $\\mathcal{O}(h^4)$ con casi la mitad de llamados a `f` que $RK4$ en tramos largos", "Mide su propio error comparando $P$ y $C$"],
                "contras": ["No arranca por sí mismo", "Falla si el paso $\\Delta h$ produce menos de $4$ intervalos"],
            },
        },
        "expectativas": {
            ("euler", "rk4"): {
                "que_esperamos": "**$RK4$** ($\\mathcal{O}(h^4)$) se pega a la solución exacta incluso con pasos medianos, mientras **Euler** ($\\mathcal{O}(h)$) forma una poligonal recta que se separa cada vez más.",
                "excepciones": "Si la derivada es constante (recta pura), Euler ya es exacto en $1$ paso; si el sistema tiene dinámicas con escalas de decaimiento extremadamente rápidas (ecuaciones rígidas), ambos exigen $\\Delta h$ muy pequeños.",
            },
            ("euler", "heun"): {
                "que_esperamos": "**Heun** disminuye drásticamente el desvío frente a **Euler** al promediar las pendientes de salida y llegada.",
                "excepciones": "En ecuaciones explosivas con asíntotas verticales cercanas, un paso $\\Delta h$ demasiado grande hace diverger a ambos.",
            },
            ("milne", "rk4"): {
                "que_esperamos": "Ambos logran precisión $\\mathcal{O}(h^4)$ casi indistinguible frente a la exacta, pero **Milne** evalúa menos veces `f(x, y)` por paso tras arrancar.",
                "excepciones": "1) Si $\\Delta h$ es grande y deja menos de $4$ pasos en $[x_0, x_f]$, Milne no puede iniciar ($RK4$ funciona igual desde $1$ paso). 2) En modelos de decaimiento fuerte como $y' = -\\lambda y$, Milne puede oscilar si $\\Delta h$ no es suficientemente chico.",
            },
            ("heun", "rk4"): {
                "que_esperamos": "**$RK4$** (orden $4$) reduce mucho más el error acumulado que **Heun** (orden $2$) para el mismo $\\Delta h$.",
                "excepciones": "Con $\\Delta h$ minúsculos e iteración repetida del corrector, Heun se acerca a $RK4$ pero gastando más llamadas.",
            },
            ("rk2", "rk4"): {
                "que_esperamos": "**$RK4$** alcanza error global $\\mathcal{O}(h^4)$ frente a $\\mathcal{O}(h^2)$ de **$RK2$** con mucho menor desvío al final del rango.",
                "excepciones": "Si `f(x, y)` es sumamente costosa y basta una tolerancia relajada del $1\\% - 2\\%$, $RK2$ consume exactamente la mitad de tiempo de evaluación.",
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
    """Busca la ficha técnica comparativa de expectativa y excepciones entre Método A y Método B."""
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
        "que_esperamos": "Se comparan dos algoritmos de la misma familia matemática contrastando orden de convergencia, costo computacional y estabilidad numérica.",
        "excepciones": "Verificá los requisitos de cada algoritmo (derivadas no nulas, dominancia diagonal, equidistancia de nodos o tamaño de paso $\\Delta h$) para detectar posibles fallos.",
    }


# ==============================================================================
# DEDUCCIONES COMPLETAS PASO A PASO (TEÓRICO, EJEMPLO Y PRÁCTICA DE CÁTEDRA)
# ==============================================================================

DEDUCCIONES_PASO_A_PASO_METODOS: dict[str, dict[str, Any]] = {
    "series_taylor": {
        "deduccion_teorica": [
            r"**Planteo polinómico alrededor de $x_0$:** Buscamos aproximar una función derivable $f(x)$ cerca de un centro $x_0$ mediante un polinomio $P_n(x) = \sum_{k=0}^{n} c_k (x - x_0)^k$ de modo que la función y sus primeras $n$ derivadas coincidan en $x_0$.",
            r"**Cálculo de los coeficientes $c_k$:** Evaluando la derivada $k$-ésima $P_n^{(k)}(x_0) = k!\,c_k$ e igualándola a $f^{(k)}(x_0)$, despejamos directamente $c_k = \frac{f^{(k)}(x_0)}{k!}$.",
            r"**Caso Maclaurin ($x_0 = 0$) para $e^x$, $\ln(1+x)$ y $\arctan(x)$:** Para $f(x)=e^x$, todas las derivadas cumplen $f^{(k)}(0)=1$, obteniendo $e^x = \sum_{k=0}^{\infty}\frac{x^k}{k!}$. Integrando $\frac{1}{1+t}$ sale $\ln(1+x)$ y evaluando $\arctan(1) = \frac{\pi}{4} = \sum_{k=0}^{\infty}\frac{(-1)^k}{2k+1}$ se deduce la serie alternante de Gregory-Leibniz.",
            r"**Relación de recurrencia para el código:** Para no recalcular potencias y factoriales desde cero, cada término cumple $T_k = T_{k-1}\cdot \frac{x}{k}$, lo que reduce el costo de cada iteración a una multiplicación y una división.",
            r"**Deducción del resto y criterio de corte (Scarborough):** Por el Teorema del Resto de Lagrange, el error al truncar en $n$ términos es $R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$. En la práctica no conocemos el valor exacto, por lo que garantizamos $m$ cifras significativas exigiendo que la corrección relativa entre pasos cumpla $E_{rp} = \left|\frac{S_k - S_{k-1}}{S_k}\right| \times 100\% < E_s = 0.5 \times 10^{2-m}\%$.",
        ],
        "practica_pasos_deduccion": [
            r"**Definir función e hipótesis de trabajo:** En $e^x$ con $x = 1$ y $N = 8$ términos de truncamiento ($k = 0 \dots 7$), cada término vale $T_k = \frac{1^k}{k!}$ y el acumulador inicia en $S_0 = 1$.",
            r"**Acumulación iterativa término a término:** Calculamos $T_1 = 1$, $T_2 = \frac{1}{2} = 0.5$, $T_3 = \frac{1}{6} \approx 0.166667$, $T_4 = \frac{1}{24} \approx 0.041667$, $T_5 = \frac{1}{120}$, $T_6 = \frac{1}{720}$, $T_7 = \frac{1}{5040}$ y $T_8 = \frac{1}{40320}$.",
            r"**Verificación de parada por Scarborough ($n = 3$ cifras):** Calculamos $E_s = 0.5 \times 10^{2-3}\% = 0.05\%$. Hasta $k = 5$ ($E_{rp} \approx 0.306\%$) el bucle continúa; al incluir $k = 6$ se verifica $E_{rp} \approx 0.051\%$, y en $k = 7$ cae a $0.0073\% < 0.05\%$, asegurando ya $3$ cifras exactas.",
            r"**Evaluación final para $e^1$ y serie alternante de $\pi$:** Con $N = 8$ términos alcanzamos $S_8 \approx 2.71827877$ ($E_a = 3.0586 \times 10^{-6}$ frente a $e = 2.71828183$). Para $\pi = 4\sum \frac{(-1)^k}{2k+1}$, acotando por el primer término omitido $\frac{4}{2N+1} < 0.0005$ se deduce $2N + 1 > 8000$, es decir $N \ge 2000$ iteraciones.",
        ],
    },
    "aritmetica_maquina": {
        "deduccion_teorica": [
            r"**Representación normalizada de $k$ dígitos:** Todo real $y \ne 0$ se escribe en base $10$ como $y = \pm 0.d_1 d_2 \dots d_k d_{k+1}\dots \times 10^n$ con $1 \le d_1 \le 9$, desplazando la coma decimal mediante el exponente $n$.",
            r"**Reglas de truncamiento y redondeo simétrico:** Al recortar a $k$ dígitos, el truncamiento descarta $d_{k+1}\dots$ directamente; el redondeo suma $\frac{1}{2}\cdot 10^{n-k}$ al valor absoluto antes de truncar (o incrementa $d_k$ en $1$ si $d_{k+1} \ge 5$).",
            r"**Deducción de la cota del error relativo (épsilon de máquina):** Como $|y| \ge 0.1 \times 10^n$, en truncamiento $|y - fl(y)| < 10^{n-k} \implies \frac{|y - fl(y)|}{|y|} < \frac{10^{n-k}}{10^{n-1}} = 10^{1-k}$. En redondeo el error absoluto no supera $0.5 \times 10^{n-k}$, dando cota $\frac{|y - fl(y)|}{|y|} \le 0.5 \times 10^{1-k}$.",
            r"**Ruptura de la asociatividad en operaciones finitas:** En máquina cada operación binaria ejecuta $a \oplus b = fl(fl(a) + fl(b))$. Cuando dos números grandes cercanos se restan primero ($u - v$), su diferencia libera posiciones en la mantisa para el número chico $w$; en cambio, operar $u - w$ primero desplaza los dígitos menos significativos de $w$ fuera de los $k$ dígitos disponibles (absorción).",
        ],
        "ejemplo_teorico_pasos": [
            r"**Normalización inicial a $k = 5$ dígitos (truncamiento):** $x = 1/3 = 0.333333\dots \implies fl(x) = 0.33333 \times 10^0$; $y = 5/7 = 0.7142857\dots \implies fl(y) = 0.71428 \times 10^0$.",
            r"**Suma en aritmética finita $fl(fl(x) + fl(y))$:** Sumamos las mantisas alineadas: $0.33333 + 0.71428 = 1.04761 = 0.104761 \times 10^1$. Al truncar a $5$ dígitos se descarta el último $1$, quedando $0.10476 \times 10^1 = 1.0476$.",
            r"**Resta en aritmética finita $fl(fl(y) - fl(x))$:** Restamos $0.71428 - 0.33333 = 0.38095 = 0.38095 \times 10^0$, que ya posee $5$ dígitos exactos en mantisa normalizada.",
            r"**Comparación frente al valor exacto:** El valor exacto es $\frac{1}{3} + \frac{5}{7} = \frac{22}{21} \approx 1.047619$, dando error relativo $E_r = 1.82 \times 10^{-5} < 10^{-4}$, confirmando la cota teórica de truncamiento $10^{1-5}$.",
        ],
        "practica_pasos_deduccion": [
            r"**Normalización de los operandos a $k = 3$ dígitos:** $fl(121) = 0.121 \times 10^3$, $fl(119) = 0.119 \times 10^3$ y $fl(0.327) = 0.327 \times 10^0$. El valor exacto de referencia es $(121 - 119) - 0.327 = 1.673$.",
            r"**Camino 1 $(u \ominus v) \ominus w$ (términos cercanos primero):** Primero operamos $121 - 119 = 2.00 = 0.200 \times 10^1$ (sin pérdida de información). Luego restamos $2.00 - 0.327 = 1.673 = 0.1673 \times 10^1$, que a $k = 3$ dígitos da $0.167 \times 10^1$ tanto en redondeo como en truncamiento ($E_r = 0.18\%$).",
            r"**Camino 2 $(u \ominus w) \ominus v$ (absorción del término pequeño):** Alineamos exponentes para restar $121 - 0.327 = 120.673 = 0.120673 \times 10^3$. Al ajustar a $3$ dígitos: en redondeo queda $0.121 \times 10^3 = 121$ (el $0.327$ fue absorbido por completo); en truncamiento queda $0.120 \times 10^3 = 120$.",
            r"**Segundo paso del Camino 2 y conclusión:** Al restar $119$, el redondeo entrega $121 - 119 = 2.00$ ($E_r = 19.55\%$) y el truncamiento entrega $120 - 119 = 1.00$ ($E_r = 40.23\%$). Queda demostrado por qué conviene operar primero los términos de magnitud similar.",
        ],
    },
    "cancelacion_cuadratica": {
        "deduccion_teorica": [
            r"**Origen de la fórmula cuadrática clásica:** Dividiendo $ax^2 + bx + c = 0$ por $a \ne 0$ y completando cuadrados $\left(x + \frac{b}{2a}\right)^2 = \frac{b^2 - 4ac}{4a^2}$, se despeja $x_{1,2} = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.",
            r"**Detección del peligro cuando $b^2 \gg |4ac|$:** Si $b > 0$, la raíz $\sqrt{b^2 - 4ac}$ resulta muy próxima a $b$. En $x_1 = \frac{-b + \sqrt{b^2 - 4ac}}{2a}$, restar dos números casi iguales anula sus primeros dígitos significativos y deja solo ruido de redondeo (**cancelación catastrófica**), mientras que $x_2 = \frac{-b - \sqrt{b^2 - 4ac}}{2a}$ suma dos negativos con igual signo y es estable.",
            r"**Racionalización por el conjugado:** Multiplicando numerador y denominador de $x_1$ por el conjugado $(-b - \sqrt{b^2 - 4ac})$: $x_1 = \frac{(-b + \sqrt{b^2 - 4ac})(-b - \sqrt{b^2 - 4ac})}{2a(-b - \sqrt{b^2 - 4ac})} = \frac{b^2 - (b^2 - 4ac)}{-2a(b + \sqrt{b^2 - 4ac})}$.",
            r"**Simplificación algebraica estable (Vieta):** Cancelando $b^2 - b^2$ y simplificando $2a$, obtenemos $x_1 = \frac{-2c}{b + \sqrt{b^2 - 4ac}}$, donde todos los términos del denominador comparten signo positivo (cuando $b > 0$), eliminando por completo la resta peligrosa.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Evaluación del discriminante para $x^2 + 3000.001x + 3 = 0$:** Aquí $a = 1$, $b = 3000.001$, $c = 3$, por lo que $b^2 \approx 9 \times 10^6 \gg 4ac = 12$.",
            r"**Cálculo del radical:** $\sqrt{b^2 - 4ac} = \sqrt{9000006.000001 - 12} = 2999.999$, prácticamente indistinguible de $b = 3000.001$ si trabajamos con mantisa de pocos dígitos.",
            r"**Falla de la resolvente tradicional en $x_1$:** Con $6$ dígitos significativos, $b$ se redondea a $3000.00$ y $\sqrt{\Delta}$ también a $3000.00$, haciendo $-b + \sqrt{\Delta} = 0$, o produciendo errores relativos enormes en $x_1 = -0.001$.",
            r"**Corrección mediante la forma racionalizada:** Evaluamos $x_1 = \frac{-2(3)}{3000.001 + 2999.999} = \frac{-6}{6000} = -0.001$ exacto sin cancelar ni un solo dígito significativo, y $x_2 = \frac{-3000.001 - 2999.999}{2} = -3000$.",
        ],
        "practica_pasos_deduccion": [
            r"**Representación a $k = 4$ dígitos de $x^2 + 62.10x + 1 = 0$:** Guardamos $fl(a)=1.000$, $fl(b)=62.10$ y $fl(c)=1.000$. Las raíces exactas son $x_1 \approx -0.01610723$ y $x_2 \approx -62.08389$.",
            r"**Cálculo del discriminante en máquina ($k = 4$):** $b^2 = 62.10^2 = 3856.41 \xrightarrow{k=4} 3856$; luego $\Delta = fl(3856 - 4) = 3852$, y $\sqrt{\Delta} = \sqrt{3852} \approx 62.06448 \xrightarrow{k=4} 62.06$.",
            r"**Resolvente tradicional (resta catastrófica):** Al calcular $x_1 = fl\left(\frac{-62.10 + 62.06}{2}\right) = \frac{-0.04}{2} = -0.02000$, perdimos casi todas las cifras significativas y el error relativo salta al $24.17\%$.",
            r"**Resolvente racionalizada estable:** Aplicamos $x_1 = fl\left(\frac{-2c}{b + \sqrt{\Delta}}\right) = fl\left(\frac{-2.000}{62.10 + 62.06}\right) = fl\left(\frac{-2.000}{124.2}\right) = -0.01610$, reduciendo el error relativo a apenas $0.045\%$ (más de $500$ veces más exacto).",
        ],
    },
    "horner": {
        "deduccion_teorica": [
            r"**Partida desde la forma estándar de un polinomio de grado $n$:** $P_n(x) = a_n x^n + a_{n-1} x^{n-1} + \dots + a_1 x + a_0$. Evaluar cada potencia $x^k$ por separado exige $1 + 2 + \dots + n = \frac{n(n+1)}{2}$ multiplicaciones.",
            r"**Factorización recursiva de $x$ desde los términos superiores:** Sacando factor común $x$ en los primeros $n$ términos: $P_n(x) = (a_n x^{n-1} + a_{n-1} x^{n-2} + \dots + a_1)x + a_0$. Repitiendo la factorización dentro del paréntesis se llega a la forma completamente anidada $P_n(x) = (\dots((a_n x + a_{n-1})x + a_{n-2})\dots)x + a_0$.",
            r"**Definición de la recurrencia lineal ($b_k$):** Definimos $b_n = a_n$ y, para $k = n-1, n-2, \dots, 0$, calculamos $b_k = b_{k+1} x + a_k$. Al finalizar el ciclo en $k = 0$, se cumple exactamente $b_0 = P_n(x)$.",
            r"**Análisis de complejidad y propagación de error:** Se requieren exactamente $n$ multiplicaciones y $n$ sumas ($\mathcal{O}(n)$ en vez de $\mathcal{O}(n^2)$). Como los intermedios son más uniformes y hay menos redondeos por paso, el error de máquina acumulado disminuye notablemente.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Polinomio cúbico en forma expandida:** Sea $P_3(x) = a_3 x^3 + a_2 x^2 + a_1 x + a_0$. En cálculo directo primero computamos $x^2 = x\cdot x$ y $x^3 = x^2 \cdot x$ ($2$ productos).",
            r"**Multiplicaciones y sumas en evaluación directa:** Luego calculamos $a_3 x^3$, $a_2 x^2$ y $a_1 x$ ($3$ productos más, sumando $5$ multiplicaciones y $3$ sumas en total, con redondeo en cada potencia elevada).",
            r"**Transformación al esquema anidado de Horner:** Factorizando $x$ dos veces escribimos $P_3(x) = ((a_3 x + a_2)x + a_1)x + a_0$.",
            r"**Conteo de pasos e instrucciones:** Inicializamos $b_3 = a_3$, luego $b_2 = b_3 x + a_2$, $b_1 = b_2 x + a_1$ y $b_0 = b_1 x + a_0$: apenas $3$ multiplicaciones ($2$ si $a_3 = 1$) y $3$ sumas, eliminando el cálculo de potencias grandes.",
        ],
        "practica_pasos_deduccion": [
            r"**Factorización del polinomio del TP2:** Reescribimos $f(x) = x^3 - 6x^2 + 3x - 0.149$ como $f(x) = ((x - 6)x + 3)x - 0.149$ y normalizamos $x = 4.71 = 0.471 \times 10^1$ a $k = 3$ dígitos.",
            r"**Forma directa (redondeo $k = 3$):** Calculamos $x^2 \approx 22.2$, $x^3 \approx 105$; luego $6x^2 \approx 133$, $3x \approx 14.1$; sumando de izquierda a derecha $105 - 133 = -28.0$, $-28.0 + 14.1 = -13.9$, $-13.9 - 0.149 = -14.0$ ($E_r = 4.35\%$).",
            r"**Esquema de Horner paso a paso (redondeo $k = 3$):** 1) $b_3 = 1$; 2) $b_2 = fl(1 \times 4.71 - 6) = -1.29$; 3) $b_1 = fl(fl(-1.29 \times 4.71) + 3) = fl(-6.08 + 3) = -3.08$; 4) $b_0 = fl(fl(-3.08 \times 4.71) - 0.149) = fl(-14.5 - 0.149) = -14.6$.",
            r"**Comparación frente al exacto ($-14.636489$):** Con exactamente la misma aritmética de $3$ dígitos, Horner reduce el error relativo de $4.35\%$ a $0.25\%$ (redondeo) y $0.93\%$ (truncamiento, $-14.5$).",
        ],
    },
    "biseccion": {
        "deduccion_teorica": [
            r"**Condición de existencia (Teorema de Bolzano):** Si $f$ es continua en $[a, b]$ y cumple $f(a)\cdot f(b) < 0$ (signos opuestos en los extremos), por continuidad existe al menos una raíz real $r \in (a, b)$ tal que $f(r) = 0$.",
            r"**Partición binaria e invariante de encierro:** Tomamos el punto medio $x_k = \frac{a_k + b_k}{2}$ dividiendo $[a_k, b_k]$ en dos mitades de longitud $\frac{b_k - a_k}{2}$. Si $f(a_k)\cdot f(x_k) < 0$, el cruce está en $[a_k, x_k]$; de lo contrario, está en $[x_k, b_k]$. En ambos casos se preserva $f(a_{k+1})\cdot f(b_{k+1}) < 0$.",
            r"**Deducción de la cota de error tras $n$ pasos:** Al inicio la longitud es $L_0 = b - a$. Después de $1$ bisección es $L_1 = \frac{b-a}{2}$, y por inducción tras $n$ pasos $L_n = |b_n - a_n| = \frac{b - a}{2^n}$. Como tanto $r$ como la aproximación viven dentro de ese intervalo, el error queda acotado por $\frac{b - a}{2^n}$.",
            r"**Fórmula a priori del número de iteraciones:** Para garantizar error menor o igual que $E$, pedimos $\frac{b-a}{2^n} \le E \implies 2^n \ge \frac{b-a}{E}$. Aplicando logaritmo despejamos $n \ge \frac{\ln(b-a) - \ln E}{\ln 2}$.",
        ],
        "practica_pasos_deduccion": [
            r"**Análisis previo (Descartes y tanteo) sobre $f(x) = -2.75x^3 + 18x^2 - 21x - 12$:** Los signos en $f(x)$ son `[-, +, -, -]` ($2$ cambios $\implies 2$ o $0$ raíces positivas) y en $f(-x)$ son `[+, +, +, -]` ($1$ cambio $\implies 1$ raíz negativa exacta). El barrido $\Delta x = 0.5$ detecta cambio de signo en $[2.0, 2.5]$ pues $f(2) = 2.0 > 0$ y $f(2.5) = -3.46875 < 0$.",
            r"**Cota teórica de iteraciones para $E < 0.01$:** Sustituyendo $a=2.0$, $b=2.5$ y $E=0.01$: $n \ge \frac{\ln(0.5) - \ln(0.01)}{\ln 2} = \frac{\ln(50)}{\ln 2} \approx 5.6439 \implies$ se requieren exactamente $n = 6$ iteraciones.",
            r"**Primeras bisecciones paso a paso:** 1) $x_1 = 2.25$, $f(2.25) = -0.3242 < 0 \implies [2.0, 2.25]$; 2) $x_2 = 2.125$, $f(2.125) = 1.1519 > 0 \implies [2.125, 2.25]$; 3) $x_3 = 2.1875$, $f(2.1875) = 0.5067 > 0 \implies [2.1875, 2.25]$.",
            r"**Cierre por tolerancia en $k = 6$:** Continuando con $x_4 = 2.21875$, $x_5 = 2.234375$ y finalmente $x_6 = \frac{2.21875 + 2.234375}{2} = 2.2265625$, obtenemos $|x_6 - x_5| = 0.0078125 < 0.01$, coincidiendo con la cota predicha.",
        ],
    },
    "interpolacion_lineal_raices": {
        "deduccion_teorica": [
            r"**Planteo geométrico de la cuerda (secante):** Dados $(x_0, f(x_0))$ y $(x_1, f(x_1))$ con signos opuestos ($f(x_0)\cdot f(x_1) < 0$), aproximamos la curva en $[x_0, x_1]$ por la recta que une ambos puntos.",
            r"**Igualdad de pendientes por semejanza de triángulos:** La recta secante corta al eje $X$ en $(x_{\text{nuevo}}, 0)$. Comparando los triángulos rectángulos formados entre la pendiente total y el segmento hasta el corte: $\frac{0 - f(x_n)}{x_{n+1} - x_n} = \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}$.",
            r"**Despeje algebraico en forma de Newton-secante:** Multiplicando cruzado obtenemos $x_{n+1} = x_n - f(x_n)\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$.",
            r"**Forma simétrica compacta y criterio de reemplazo:** Llevando a denominador común $f(x_n) - f(x_{n-1})$, resulta $x_{n+1} = \frac{x_{n-1} f(x_n) - x_n f(x_{n-1})}{f(x_n) - f(x_{n-1})}$. Si $f''$ mantiene signo constante en $[a, b]$, el esquema de cátedra deja fijo como pivote al extremo que comparte signo con $f''$; en Regula Falsi general se reemplaza el extremo con mismo signo que $f(x_{n+1})$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Verificación de Bolzano y curvatura en $[x_1, x_2]$:** Confirmamos $f(x_1)\cdot f(x_2) < 0$ y comprobamos que $f''(x)$ no cambia de signo en el tramo (concavidad uniforme).",
            r"**Trazado de la primera cuerda:** Calculamos $x_3 = \frac{x_1 f(x_2) - x_2 f(x_1)}{f(x_2) - f(x_1)}$, ubicado en la intersección de la recta entre $(x_1, y_1)$ y $(x_2, y_2)$ con el eje $y = 0$.",
            r"**Elección del pivote fijo $P_1(x_1, y_1)$:** Si $f(x_1)$ y $f''$ tienen igual signo, toda cuerda que sale de $P_1$ cruza el eje $X$ siempre del mismo lado respecto a la raíz verdadera.",
            r"**Recurrencia monótona hacia la raíz:** Por eso cada nueva secante conecta el ancla fija $(x_1, f(x_1))$ con $(x_k, f(x_k))$, formando una sucesión monótona $x_2, x_3, x_4, \dots$ que converge a la raíz hasta cumplir $|x_{k+1} - x_k| < E$.",
        ],
        "practica_pasos_deduccion": [
            r"**Verificación inicial en $[-2.5, -2.0]$ para $f(x) = 2x^4 - 6x^2 + 7x - 2$:** Evaluamos los extremos: $f(-2.5) = 2(39.0625) - 6(6.25) - 17.5 - 2 = +21.125 > 0$ y $f(-2.0) = 32 - 24 - 14 - 2 = -8.0 < 0$ (cumple Bolzano). Además $f''(x) = 24x^2 - 12 > 0$ en todo $[-2.5, -2.0]$, por lo que el pivote fijo es $x_0 = -2.5$ ($f(x_0)f''(x_0) > 0$).",
            r"**Primera iteración ($x_2$):** Aplicamos $x_2 = \frac{x_0 f(x_1) - x_1 f(x_0)}{f(x_1) - f(x_0)} = \frac{(-2.5)(-8.0) - (-2.0)(21.125)}{-8.0 - 21.125} = \frac{62.25}{-29.125} \approx -2.137339$, con $f(x_2) \approx -2.2389 < 0$.",
            r"**Avance de cuerdas hacia la raíz:** Manteniendo $x_0 = -2.5$, iteración 2 da $x_3 = \frac{(-2.5)(-2.2389) - (-2.137339)(21.125)}{-2.2389 - 21.125} \approx -2.172074$, luego $x_4 \approx -2.185231$, $x_5 \approx -2.190025$.",
            r"**Parada por tolerancia $E < 0.001$:** En el paso $6$ obtenemos $x_6 = -2.191747$ (de hecho $|x_6 - x_5| \approx 0.0017$ y al paso siguiente $|x_7 - x_6| < 0.001$, logrando convergencia a $-2.1926$). Nótese cómo el pivote fijo retrasa ligeramente el avance frente a Newton.",
        ],
    },
    "newton_raphson": {
        "deduccion_teorica": [
            r"**Linealización mediante Serie de Taylor:** Desarrollamos $f(x)$ alrededor de la iteración actual $x_n$ hasta primer orden: $f(x) \approx f(x_n) + f'(x_n)(x - x_n)$ (ecuación de la recta tangente en $(x_n, f(x_n))$).",
            r"**Despeje de la raíz de la recta tangente:** Igualamos la aproximación lineal a cero buscando $x = x_{n+1}$: $f(x_n) + f'(x_n)(x_{n+1} - x_n) = 0 \implies x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$ siempre que $f'(x_n) \ne 0$.",
            r"**Deducción de las Condiciones de Fourier para arranque seguro:** En $[a, b]$ con $f(a)f(b) < 0$, $f' \ne 0$ y $f''$ de signo constante, la curva es monótona y convexa o cóncava. Para que la tangente trazada desde el extremo $x_0$ corte al eje $X$ *dentro* de $[a, b]$ del mismo lado que $x_0$ sin saltar fuera del intervalo, el punto inicial debe compartir signo con la curvatura: $f(x_0)\cdot f''(x_0) > 0$.",
            r"**Deducción del orden de convergencia cuadrático:** Restando $r$ en ambos miembros y usando Taylor de orden $2$ de $f(r)=0$: $e_{n+1} = e_n - \frac{f'(r)e_n + \frac{1}{2}f''(\xi)e_n^2 + \dots}{f'(x_n)} \approx \frac{f''(r)}{2f'(r)} e_n^2$. Al depender de $e_n^2$, la cantidad de decimales exactos aproximadamente se duplica en cada iteración.",
        ],
        "practica_pasos_deduccion": [
            r"**Cálculo simbólico de derivadas para $f(x) = x^3 - 2x - 5$:** Derivando término a término obtenemos $f'(x) = 3x^2 - 2$ y $f''(x) = 6x$. En $[2, 2.5]$ se cumple $f(2) = -1 < 0$, $f(2.5) = 5.625 > 0$ (Bolzano), $f'(x) \ge 10 > 0$ y $f''(x) \ge 12 > 0$.",
            r"**Verificación de Fourier para elegir $x_0$:** Probamos $x = 2$: $f(2)\cdot f''(2) = (-1)(12) = -12 < 0$ (NO cumple). Probamos $x = 2.5$: $f(2.5)\cdot f''(2.5) = (5.625)(15) = +84.375 > 0$ (SÍ cumple), por lo tanto elegimos $x_0 = 2.5$.",
            r"**Sustitución paso a paso en la fórmula de tangentes:** $x_1 = 2.5 - \frac{5.625}{3(2.5)^2 - 2} = 2.5 - \frac{5.625}{16.75} \approx 2.164179$; luego $x_2 = 2.164179 - \frac{0.807789}{12.05101} \approx 2.097135$; luego $x_3 \approx 2.094555$.",
            r"**Cierre con tolerancia $E < 0.0001$:** En $k = 4$, $x_4 = 2.09455148$, con $|x_4 - x_3| \approx 3.8 \times 10^{-6} < 10^{-4}$ (y en $k=5$ alcanza precisión de máquina $2.0945514815$).",
        ],
    },
    "punto_fijo_aitken": {
        "deduccion_teorica": [
            r"**Transformación de raíces en punto fijo:** Reescribimos $f(x) = 0$ de forma equivalente como $x = g(x)$. La solución $r$ es el punto donde la curva $y = g(x)$ intersecta la recta identidad $y = x$, sugiriendo la iteración $x_{n+1} = g(x_n)$.",
            r"**Teorema del Punto Fijo y condición contractiva:** Aplicando el Teorema del Valor Medio entre $x_n$ y la raíz $r = g(r)$: $e_{n+1} = |x_{n+1} - r| = |g(x_n) - g(r)| = |g'(\xi_n)|\,|x_n - r|$. Si existe $L = \max|g'(x)| < 1$ en un entorno de $r$, entonces $e_n \le L^n e_0 \to 0$ (**convergencia lineal garantizada**).",
            r"**Deducción de la aceleración $\Delta^2$ de Aitken:** Para $x_n$ cercanos a $r$, el error decae como progresión geométrica $x_{n+1} - r \approx K(x_n - r)$ con $K \approx g'(r)$. Formando dos pasos consecutivos $\frac{x_{n+2} - r}{x_{n+1} - r} \approx \frac{x_{n+1} - r}{x_n - r}$, eliminamos $K$ igualando razones.",
            r"**Despeje de la fórmula extrapolada $S$:** Multiplicando cruzado $(x_{n+2}-r)(x_n-r) = (x_{n+1}-r)^2$, los términos $r^2$ se cancelan y despejando $r = S$ queda $S = x_{n+2} - \frac{(x_{n+2} - x_{n+1})^2}{x_{n+2} - 2x_{n+1} + x_n}$ (análogo en notación de diferencias $\Delta^2$: $S = x_n - \frac{(\Delta x_n)^2}{\Delta^2 x_n}$), logrando saltos mucho más veloces.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Despeje iterativo de $e^x - x^2 - 3 = 0$ en $(1, 2)$:** Pasando $x^2 + 3$ al segundo miembro y aplicando logaritmo natural obtenemos $x = g(x) = \ln(x^2 + 3)$, cuya derivada es $g'(x) = \frac{2x}{x^2+3} \in [0.46, 0.54] < 1$ (contractiva).",
            r"**Primer bloque de dos iteraciones simples desde $x_0 = 1$:** Calculamos $x_1 = \ln(1^2 + 3) = \ln(4) \approx 1.386294$ y luego $x_2 = \ln(1.386294^2 + 3) \approx 1.593678$.",
            r"**Primer salto $\Delta^2$ de Aitken:** Sustituimos $(x_0, x_1, x_2)$ en $S_1 = x_2 - \frac{(x_2 - x_1)^2}{x_2 - 2x_1 + x_0} = 1.593678 - \frac{(0.207384)^2}{-0.178910} \approx 1.83405$ (saltando cerca de $r \approx 1.8731$).",
            r"**Segundo ciclo acelerado hasta convergencia:** Reiniciando desde $S_1 = 1.83405$, evaluamos $x_4 = 1.85062$, $x_5 = 1.86016$ y la nueva extrapolación da $x_6 = 1.87311$, alcanzando la raíz en apenas $2$ ciclos.",
        ],
        "practica_pasos_deduccion": [
            r"**Verificación de contractividad para $g(x) = \ln(x^2 + 5)$ en $x_0 = 2.5$:** Derivando obtenemos $g'(x) = \frac{2x}{x^2 + 5}$. Evaluado en $x_0 = 2.5$: $|g'(2.5)| = \frac{5}{6.25 + 5} = \frac{5}{11.25} \approx 0.4444 < 1$, por lo que el esquema es contractivo y convergente.",
            r"**Comparación inicial por Punto Fijo simple:** $x_1 = \ln(2.5^2 + 5) = \ln(11.25) \approx 2.420368$, $x_2 = \ln(2.420368^2 + 5) \approx 2.384919$; al avanzar a paso lineal necesita $10$ iteraciones para bajar de $E = 0.0001$ (llegando a $2.3563986$).",
            r"**Cálculo del primer salto de Aitken con $(x_0, x_1, x_2)$:** $\Delta x_0 = 2.420368 - 2.5 = -0.079632$, $\Delta x_1 = 2.384919 - 2.420368 = -0.035449$, y $\Delta^2 x_0 = x_2 - 2x_1 + x_0 = 0.044183$. Sustituyendo: $S_1 = 2.384919 - \frac{(-0.035449)^2}{0.044183} \approx 2.356476$.",
            r"**Cierre acelerado en pocos ciclos:** Desde $2.356476$ se recalculan dos pasos simples y en el ciclo $4$ Aitken converge a $2.356408$ con error inferior a $10^{-4}$, ahorrando más de la mitad de iteraciones.",
        ],
    },
    "eliminacion_gauss": {
        "deduccion_teorica": [
            r"**Invariancia bajo operaciones elementales de fila:** El sistema $Ax = b$ no altera su conjunto solución al intercambiar filas ($F_i \leftrightarrow F_j$) ni al reemplazar una fila por sí misma menos un múltiplo escalar de otra fila ($F_i \leftarrow F_i - m_{ik}F_k$).",
            r"**Triangulación progresiva con pivoteo parcial:** En la columna $k$, elegimos la fila $p \ge k$ que maximiza $|a_{pk}|$ y la subimos a la diagonal ($F_k \leftrightarrow F_p$) para evitar dividir por cero o por pivotes diminutos. Luego anulamos todo $a_{ik}$ ($i > k$) definiendo el multiplicador $m_{ik} = \frac{a_{ik}}{a_{kk}}$ y actualizando toda la fila aumentada $[A \mid b]$.",
            r"**Despeje mediante sustitución regresiva:** Al terminar los $n-1$ pasos, la matriz queda triangular superior $Ux = c$. La última ecuación es directa ($u_{nn}x_n = c_n \implies x_n = c_n / u_{nn}$), y subiendo para $i = n-1 \dots 1$ despejamos $x_i = \frac{c_i - \sum_{j=i+1}^n u_{ij}x_j}{u_{ii}}$.",
            r"**Conteo exacto de operaciones:** Eliminar debajo de la diagonal consume $\frac{2}{3}n^3 + \mathcal{O}(n^2)$ operaciones aritméticas y la sustitución regresiva apenas $n^2$ operaciones.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Matriz aumentada inicial:** $[2, -2, 5 \mid 13]$, $[2, 3, 4 \mid 20]$, $[3, -1, 3 \mid 10]$.",
            r"**Eliminación columna 1 ($a_{11} = 2$):** Con $m_{21} = 1$ y $m_{31} = 1.5$, restamos múltiplos de $F_1$ a $F_2$ y $F_3$ obteniendo $[0, 5, -1 \mid 7]$ y $[0, 2, -4.5 \mid -9.5]$.",
            r"**Eliminación columna 2 ($a'_{22} = 5$):** Aplicando $m_{32} = \frac{2}{5} = 0.4$ sobre $F_3$ queda la matriz triangular superior con última ecuación $-4.1 x_3 = -12.3$.",
            r"**Sustitución regresiva completa:** De la última fila $x_3 = \frac{-12.3}{-4.1} = 3$; subiendo a $F_2$, $5x_2 - 3 = 7 \implies x_2 = 2$; en $F_1$, $2x_1 - 2(2) + 5(3) = 13 \implies x_1 = 1$.",
        ],
        "practica_pasos_deduccion": [
            r"**Matriz aumentada inicial $[A \mid b]$:** Partimos con las ecuaciones de cátedra: Fila 1 $[18.72,\, 8.2,\, 8.76 \mid 121.280]$, Fila 2 $[6.4,\, 15.9,\, 7.18 \mid 126.321]$ y Fila 3 $[5.22,\, 6.4,\, 14.31 \mid 118.522]$. Como $|18.72|$ es el mayor de la columna 1, no hace falta permutar.",
            r"**Eliminación columna 1:** Multiplicadores $m_{21} = \frac{6.4}{18.72} \approx 0.341880$ y $m_{31} = \frac{5.22}{18.72} \approx 0.278846$. Restando a $F_2$ y $F_3$ obtenemos en $F_2'$ $[0,\, 13.09658,\, 4.18513 \mid 84.85778]$ y en $F_3'$ $[0,\, 4.11346,\, 11.86731 \mid 84.70354]$.",
            r"**Eliminación columna 2:** El pivote es $13.09658$ y $m_{32} = \frac{4.11346}{13.09658} \approx 0.314087$. Actualizando $F_3$ resulta la ecuación escalonada final $10.55282\,z = 58.05490$.",
            r"**Sustitución regresiva de abajo hacia arriba:** 1) $z = \frac{58.05490}{10.55282} \approx 5.50136$; 2) $y = \frac{84.85778 - 4.18513(5.50136)}{13.09658} \approx 4.72137$; 3) $x = \frac{121.280 - 8.2(4.72137) - 8.76(5.50136)}{18.72} \approx 1.83584$.",
        ],
    },
    "gauss_jordan": {
        "deduccion_teorica": [
            r"**De matriz triangular a matriz identidad:** Mientras Gauss anula solo por debajo de la diagonal y luego despeja hacia atrás, Gauss-Jordan convierte cada pivote en $1$ y anula *tanto por encima como por debajo* de la diagonal en cada columna $k$.",
            r"**Bloque ampliado doble $[A \mid b \mid I]$:** Si $A$ es invertible, aplicar la secuencia de operaciones elementales equivale a premultiplicar todo el bloque por $A^{-1}$: $A^{-1}[A \mid b \mid I] = [I \mid A^{-1}b \mid A^{-1}] = [I \mid x \mid A^{-1}]$.",
            r"**Pasos por columna $i = 1 \dots n$:** 1) Pivoteo parcial ($F_i \leftrightarrow F_p$); 2) Normalización de la fila pivote $F_i \leftarrow F_i / M_{ii}$ para dejar $1$ en la diagonal; 3) Para todo $k \ne i$, anular la columna con $F_k \leftarrow F_k - M_{ki}F_i$.",
            r"**Costo computacional:** Al eliminar también los ceros superiores en cada columna, realiza aproximadamente $n^3$ operaciones (un $50\%$ más que Gauss para un sistema único $Ax = b$, pero compensa si necesitamos explícitamente $A^{-1}$ o varios vectores $b$).",
        ],
        "ejemplo_teorico_pasos": [
            r"**Armado del bloque ampliado inicial:** Colocamos la matriz $A$ seguida de la columna $b = [13, 20, 10]^T$ (y opcionalmente $I_3$ si también se desea $A^{-1}$).",
            r"**Columna 1:** Dividimos la fila $1$ por el pivote $2$ para obtener $[1, -1, 2.5 \mid 6.5]$; luego restamos $2F_1$ a $F_2$ y $3F_1$ a $F_3$ dejando ceros en el resto de la columna $1$.",
            r"**Columna 2:** Normalizamos $F_2$ dividiendo por $5$ para fijar el pivote $1$, y anulamos la posición $(1,2)$ hacia arriba y la $(3,2)$ hacia abajo.",
            r"**Columna 3 y lectura directa:** Al normalizar el último pivote diagonal y eliminar las entradas superiores en $(1,3)$ y $(2,3)$, el bloque izquierdo queda convertido en $I_3$ y la columna aumentada entrega directamente $[1, 2, 3]^T$ sin sustitución trasera.",
        ],
        "practica_pasos_deduccion": [
            r"**Bloque inicial $[A \mid b \mid I_3]$:** Cargamos las filas de $A$ ($18.72, 8.2, 8.76$, etc.), el vector $b = [121.280, 126.321, 118.522]^T$ y la identidad $I_3$ a la derecha.",
            r"**Pivote 1 ($18.72$):** Dividimos $F_1 \leftarrow F_1 / 18.72 \implies [1, 0.43803, 0.46795 \mid 6.47863]$. Anulamos la columna 1 en $F_2$ ($F_2 - 6.4 F_1$) y en $F_3$ ($F_3 - 5.22 F_1$).",
            r"**Pivote 2 ($\approx 13.09658$):** Normalizamos $F_2$ a pivote $1$ y eliminamos tanto hacia arriba (entrada $(1,2)$ de $F_1$) como hacia abajo (entrada $(3,2)$ de $F_3$).",
            r"**Pivote 3 ($\approx 10.55282$) y lectura simultánea de $x$ y $A^{-1}$:** Normalizamos $F_3$ y anulamos $(1,3)$ y $(2,3)$. En la columna derecha de $b$ aparece directamente $x = [1.8358, 4.7214, 5.5014]^T$ (con $2$ decimales en apunte: $[1.82, 4.72, 5.50]$), y en el bloque derecho queda formada $A^{-1}$.",
        ],
    },
    "factorizacion_lu": {
        "deduccion_teorica": [
            r"**Relación entre Eliminación de Gauss y matrices elementales:** Cada operación de resta $F_i \leftarrow F_i - m_{ik}F_k$ debajo de la diagonal equivale a multiplicar a izquierda por una matriz elemental triangular inferior $E_{ik}$. La matriz escalonada final es $U = E \cdot A$, por lo que $A = E^{-1}U = L \cdot U$.",
            r"**Estructura del factor $L$ en Doolittle:** El inverso del producto de matrices elementales tiene una propiedad notable: la matriz $L$ conserva unos en su diagonal ($l_{ii} = 1$) y en cada entrada $(i, k)$ debajo de la diagonal guarda directamente el multiplicador $l_{ik} = m_{ik}$ calculado durante la eliminación de Gauss.",
            r"**Desacoplamiento del sistema $Ax = b$:** Reemplazando $A = LU$ en $LUx = b$ y definiendo el vector auxiliar $y = Ux$, el problema original de orden $\mathcal{O}(n^3)$ se divide en dos sistemas triangulares inmediatos.",
            r"**Doble barrido triangular $\mathcal{O}(n^2)$:** 1) **Sustitución progresiva** en $Ly = b$: $y_i = b_i - \sum_{j=1}^{i-1} l_{ij} y_j$; 2) **Sustitución regresiva** en $Ux = y$: $x_i = \frac{y_i - \sum_{j=i+1}^n u_{ij} x_j}{u_{ii}}$. Si cambia $b$, solo repetimos estos dos pasos baratos.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Cálculo de multiplicadores de $L$ en columnas 1 y 2:** $l_{21} = 20/4 = 5$, $l_{31} = -8/4 = -2$, y tras actualizar fila 3 a $[0, 9, 19]$, $l_{32} = 9/3 = 3$.",
            r"**Ensamblado de factores $L$ y $U$:** $L = [[1,0,0],[5,1,0],[-2,3,1]]$ y $U = [[4,-2,1],[0,3,7],[0,0,-2]]$, cumpliendo $L\cdot U = A$.",
            r"**Barrido progresivo $Ly = b$:** $y_1 = 11$; $y_2 = 70 - 5(11) = 15$; $y_3 = 17 - (-2)(11) - 3(15) = -6$.",
            r"**Barrido regresivo $Ux = y$:** $x_3 = -6/(-2) = 3$; $x_2 = (15 - 7(3))/3 = -2$; $x_1 = (11 + 2(-2) - 3)/4 = 1$.",
        ],
        "practica_pasos_deduccion": [
            r"**Eliminación columna 1 sobre $A$:** Con pivote $u_{11} = 4$, calculamos los multiplicadores $l_{21} = \frac{20}{4} = 5$ y $l_{31} = \frac{-8}{4} = -2$. Al actualizar filas $2$ y $3$ de $U$ quedan $[0, 3, 7]$ y $[0, 9, 19]$.",
            r"**Eliminación columna 2 y obtención de $L, U$:** Con pivote $u_{22} = 3$, resulta $l_{32} = \frac{9}{3} = 3$, anulando la entrada $(3,2)$ y dejando $u_{33} = 19 - 3(7) = -2$. Así $L = [[1,0,0],[5,1,0],[-2,3,1]]$ y $U = [[4,-2,1],[0,3,7],[0,0,-2]]$, verificando $L \cdot U = A$.",
            r"**Sustitución progresiva $Ly = b$ (de arriba hacia abajo):** $y_1 = 11$; $y_2 = 70 - 5(11) = 15$; $y_3 = 17 - (-2)(11) - 3(15) = 17 + 22 - 45 = -6$.",
            r"**Sustitución regresiva $Ux = y$ (de abajo hacia arriba):** $x_3 = \frac{-6}{-2} = 3$; $x_2 = \frac{15 - 7(3)}{3} = \frac{-6}{3} = -2$; $x_1 = \frac{11 - (-2)(-2) - 1(3)}{4} = \frac{4}{4} = 1$.",
        ],
    },
    "gauss_seidel": {
        "deduccion_teorica": [
            r"**Descomposición aditiva de $A$ y despeje coordenada a coordenada:** Escribimos la $i$-ésima ecuación $\sum_{j=1}^n a_{ij} x_j = b_i$ despejando $x_i$ (con $a_{ii} \ne 0$): $x_i = \frac{1}{a_{ii}}\left(b_i - \sum_{j < i} a_{ij} x_j - \sum_{j > i} a_{ij} x_j\right)$.",
            r"**Actualización inmediata (in-place) frente a Jacobi:** A diferencia de Jacobi (que usa solo valores del paso anterior $k$), Gauss-Seidel sustituye en la primera sumatoria ($j < i$) las variables $x_j^{(k+1)}$ que ya fueron recién actualizadas en la misma vuelta, reduciendo memoria y duplicando habitualmente la velocidad de convergencia.",
            r"**Forma matricial del operador de iteración:** Descomponiendo $A = D - E - F$ (diagonal, triangular inferior y superior cambiadas de signo), la iteración equivale a $(D - E)x^{(k+1)} = Fx^{(k)} + b$, es decir $x^{(k+1)} = T_{GS}x^{(k)} + c$ con $T_{GS} = (D-E)^{-1}F$.",
            r"**Deducción de la condición suficiente de Dominancia Diagonal:** La sucesión converge para cualquier semilla $x^{(0)}$ si la norma de $T_{GS}$ es $< 1$ (equivalente por Teorema del Punto Fijo a $\rho(T_{GS}) < 1$). Se demuestra que si $|a_{ii}| > \sum_{j \ne i} |a_{ij}|$ en todas las filas, $\|T_{GS}\|_{\infty} < 1$, garantizando convergencia sin calcular autovalores.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Reordenamiento de filas para asegurar dominancia diagonal:** Colocamos la ecuación de mayor $|a_{ii}|$ en cada fila: $10x_1 + x_2 + 2x_3 = 44$, $2x_1 + 10x_2 + x_3 = 51$, $x_1 + 2x_2 + 10x_3 = 61$.",
            r"**Despeje iterativo in-place:** $x_1 = (44 - x_2 - 2x_3)/10$, $x_2 = (51 - 2x_1 - x_3)/10$, $x_3 = (61 - x_1 - 2x_2)/10$.",
            r"**Iteración 1 desde $(0, 0, 0)$:** $x_1^{(1)} = 4.4$, luego $x_2^{(1)} = (51 - 2(4.4))/10 = 4.22$, luego $x_3^{(1)} = (61 - 4.4 - 2(4.22))/10 = 4.816$.",
            r"**Iteraciones 2 y 3 hasta $E \le 0.02$:** Los valores actualizados al instante contraen el error rápidamente hasta $[2.9999, 3.9998, 5.0004] \to [3, 4, 5]^T$.",
        ],
        "practica_pasos_deduccion": [
            r"**Chequeo de dominancia diagonal en las $3$ filas:** Fila 1: $|18.72| > 8.2 + 8.76 = 16.96$ (sí); Fila 2: $|15.9| > 6.4 + 7.18 = 13.58$ (sí); Fila 3: $|14.31| > 5.22 + 6.4 = 11.62$ (sí). La convergencia está garantizada desde cualquier $X_0$.",
            r"**Fórmulas de recurrencia despejadas:** $x^{(k+1)} = \frac{121.280 - 8.2 y^{(k)} - 8.76 z^{(k)}}{18.72}$, $y^{(k+1)} = \frac{126.321 - 6.4 x^{(k+1)} - 7.18 z^{(k)}}{15.9}$, $z^{(k+1)} = \frac{118.522 - 5.22 x^{(k+1)} - 6.4 y^{(k+1)}}{14.31}$.",
            r"**Primera vuelta desde $X_0 = [1, 1, 1]$:** 1) $x^{(1)} = \frac{121.280 - 8.2 - 8.76}{18.72} \approx 5.57265$; 2) usando $x^{(1)}$ al instante, $y^{(1)} = \frac{126.321 - 6.4(5.57265) - 7.18(1)}{15.9} \approx 5.25007$; 3) usando ambos, $z^{(1)} = \frac{118.522 - 5.22(5.57265) - 6.4(5.25007)}{14.31} \approx 3.90163$.",
            r"**Evolución hacia la convergencia ($E = 0.01$):** Cada vuelta contrae el error aproximadamente por un factor cercano a $0.4$; en la iteración $6$ se alcanza $[1.835, 4.721, 5.502]$ con máximo cambio entre pasos menor a $0.005 < 0.01$.",
        ],
    },
    "potencias": {
        "deduccion_teorica": [
            r"**Descomposición espectral del vector semilla $X_0$:** Si $A$ tiene una base de autovectores $\{v_1, v_2, \dots, v_n\}$ asociados a $|\lambda_1| > |\lambda_2| \ge \dots \ge |\lambda_n|$, cualquier vector $X_0$ no ortogonal a $v_1$ se escribe como $X_0 = c_1 v_1 + c_2 v_2 + \dots + c_n v_n$ con $c_1 \ne 0$.",
            r"**Multiplicación iterativa $A^k X_0$:** Como $A^k v_i = \lambda_i^k v_i$, obtenemos $A^k X_0 = \lambda_1^k \left[c_1 v_1 + c_2\left(\frac{\lambda_2}{\lambda_1}\right)^k v_2 + \dots + c_n\left(\frac{\lambda_n}{\lambda_1}\right)^k v_n\right]$. Como $|\lambda_i / \lambda_1| < 1$ para $i \ge 2$, todas las componentes secundarias tienden a cero exponencialmente.",
            r"**Reescalado para evitar desborde y extraer $\lambda_1$:** Para impedir `Overflow` o `Underflow` de $\lambda_1^k$, en cada paso evaluamos $Y_k = A X_{k-1}$, tomamos $m_k$ igual al componente de mayor magnitud (norma infinito) y normalizamos $X_k = Y_k / m_k$. Cuando $X_{k-1} \to v_1$, $A v_1 = \lambda_1 v_1$ implica $m_k \to \lambda_{\max}$.",
            r"**Extensión al menor autovalor ($\lambda_{\min}$) por Potencia Inversa:** Los autovalores de $A^{-1}$ son los recíprocos $1/\lambda_i$. Por tanto, el mayor autovalor en módulo de $A^{-1}$ corresponde a $1/\lambda_{\min}$, y aplicando el mismo algoritmo con $Y_k = A^{-1}X_{k-1}$ se obtiene $\lambda_{\min} = 1 / m_k$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Inicialización:** Se fija una semilla no nula $X_0$ (habitualmente $[1, 1, \dots, 1]^T$) y tolerancia relativa porcentual $E$.",
            r"**Multiplicación matricial $Y_k = A X_{k-1}$:** Proyecta al vector actual sobre el dominio lineal de $A$ amplificando su componente a lo largo de $v_1$ en razón $\lambda_1$.",
            r"**Normalización por norma infinito:** Hallamos la entrada de $Y_k$ con máximo valor absoluto $m_k = Y_k[idx_{\max}]$ y construimos el nuevo vector unitario $X_k = Y_k / m_k$, cuya componente máxima vale exactamente $1$.",
            r"**Criterio de corte:** Se mide $\varepsilon_k = \frac{|m_k - m_{k-1}|}{|m_k|}\times 100\%$ y al bajar de $E$ concluimos que $\lambda_{\max} \approx m_k$ y su autovector asociado es $X_k$.",
        ],
        "practica_pasos_deduccion": [
            r"**Potencias Directas ($\lambda_{\max}$) en $A$ simétrica con $X_0 = [1, 1, 1]^T$:** Paso 1: $Y_1 = A X_0 = [1.778, 0.000, 1.778]^T$; máximo módulo $m_1 = 1.778 \implies X_1 = [1, 0, 1]^T$.",
            r"**Siguientes iteraciones directas:** Paso 2: $Y_2 = A X_1 = [3.556, -3.556, 3.556]^T \implies m_2 = 3.556$, $X_2 = [1, -1, 1]^T$. Paso 3: $Y_3 = [5.334, -7.112, 5.334]^T \implies m_3 = -7.112$ (con signo en norma infinito) o reescalando a componente máxima positiva $7.112$, oscila hasta $m_5 \approx 6.095$ con autovector $[-0.708, 1, -0.708]^T$.",
            r"**Potencias Inversas ($\lambda_{\min}$) sobre $A^{-1}$:** Invertimos $A$ obteniendo $A^{-1}$ simétrica de filas positivas; aplicamos $Y_1 = A^{-1}X_0$, obteniendo en el paso 1 $m_1^{-1} = 1/\mu_1$ y vector normalizado positivo.",
            r"**Convergencia rápida de $\lambda_{\min}$:** Al tercer ciclo ($k=3$), $\mu_k$ estabiliza en $0.964$, cuyo recíproco es $\lambda_{\min} = \frac{1}{\mu_3} \approx 1.037$ con autovector $[0.709, 1, 0.709]^T$ (razón teórica $|\lambda_2^{-1}/\lambda_{\min}^{-1}| = |\lambda_{\min}/\lambda_2|$ muy favorable).",
        ],
    },
    "faddeev_leverrier": {
        "deduccion_teorica": [
            r"**Teorema de Cayley-Hamilton y Newton-Girard:** Toda matriz $A \in \mathbb{R}^{n\times n}$ anula su propio polinomio característico $P(\lambda) = (-1)^n(\lambda^n - b_1 \lambda^{n-1} - b_2\lambda^{n-2} - \dots - b_n) = 0$, cuyos coeficientes se conectan con las sumas de potencias de los autovalores $s_k = \sum_{i=1}^n \lambda_i^k = \operatorname{tr}(A^k)$ mediante las identidades de Newton.",
            r"**Recurrencia matricial de Faddeev:** Al definir $B_1 = A$, $b_1 = \operatorname{tr}(B_1)$ y para $k = 2 \dots n$ generar $B_k = A(B_{k-1} - b_{k-1}I)$ junto con $b_k = \frac{1}{k}\operatorname{tr}(B_k)$, se demuestra por inducción que el término final cumple $B_n = b_n I$ porque $B_n - b_n I = A^n - b_1 A^{n-1} - \dots - b_n I = P(A) = 0$ por Cayley-Hamilton.",
            r"**Deducción gratuita de $\det(A)$ y $A^{-1}$:** Al evaluar $P(0) = \det(A)$, resulta $\det(A) = (-1)^{n-1} b_n$. Además, de $A(B_{n-1} - b_{n-1}I) = B_n = b_n I$, dividiendo ambos lados por $b_n \ne 0$ y $A$ se deduce directamente $A^{-1} = \frac{1}{b_n}(B_{n-1} - b_{n-1}I)$.",
            r"**Obtención del espectro completo:** Hallando las raíces del polinomio característico construido $[-1]^n(\lambda^n - \sum_{k=1}^n b_k \lambda^{n-k}) = 0$ se recuperan *todos* los autovalores (reales o complejos), de costo $\mathcal{O}(n^4)$.",
        ],
        "practica_pasos_deduccion": [
            r"**Paso $k = 1$ sobre $A = [[3,2,4],[2,0,2],[4,2,3]]$:** $B_1 = A$. Su diagonal es $[3, 0, 3]$, por lo que $b_1 = \operatorname{tr}(B_1) = 3 + 0 + 3 = 6$. Formamos $B_1 - 6I = [[-3,2,4],[2,-6,2],[4,2,-3]]$.",
            r"**Paso $k = 2$:** Multiplicamos $B_2 = A(B_1 - 6I) = [[11, 2, 4], [2, 8, 2], [4, 2, 11]]$. Su traza vale $11 + 8 + 11 = 30$, luego $b_2 = \frac{30}{2} = 15$. Restando $15I$: $B_2 - 15I = [[-4,2,4],[2,-7,2],[4,2,-4]]$.",
            r"**Paso $k = 3$ y verificación de identidad:** Calculamos $B_3 = A(B_2 - 15I) = [[8, 0, 0], [0, 8, 0], [0, 0, 8]] = 8I$! Su traza es $24 \implies b_3 = \frac{24}{3} = 8$, $\det(A) = (-1)^2 b_3 = 8$, y $A^{-1} = \frac{1}{8}(B_2 - 15I) = [[-0.5, 0.25, 0.5], [0.25, -0.875, 0.25], [0.5, 0.25, -0.5]]$.",
            r"**Factorización del polinomio y espectro:** $P(\lambda) = -\lambda^3 + 6\lambda^2 + 15\lambda + 8 = 0$. Probando raíces enteras $(\lambda - 8)(\lambda + 1)^2 = 0$, obtenemos $\lambda_1 = 8$, $\lambda_2 = \lambda_3 = -1$, con autovector normalizado $[-1, -0.5, -1]^T$ (o unitario paralelo) para $\lambda = 8$.",
        ],
    },
    "interpolacion_lineal": {
        "deduccion_teorica": [
            r"**Ecuación de la recta por dos puntos consecutivos:** Dados $(x_0, y_0)$ y $(x_1, y_1)$ con $h = x_1 - x_0 \ne 0$ y $\Delta y_0 = y_1 - y_0$, la razón de cambio constante iguala las pendientes entre $(x_0, y_0)$, el punto intermedio $(x, y)$ y $(x_1, y_1)$: $\frac{y - y_0}{x - x_0} = \frac{y_1 - y_0}{x_1 - x_0}$.",
            r"**Fórmula de interpolación lineal directa:** Despejando $y$ como función de $x$ obtenemos el polinomio de grado $1$: $P_1(x) = y_0 + \frac{\Delta y_0}{h}(x - x_0)$.",
            r"**Fórmula de interpolación lineal inversa:** Si en vez de conocer $x$ disponemos de una ordenada objetivo $y^*$, invertimos algebraicamente la ecuación anterior multiplicando por $\frac{h}{\Delta y_0}$: $x = x_0 + \frac{(y^* - y_0)\,h}{\Delta y_0}$.",
            r"**Cota del error por truncamiento lineal:** Expandiendo $f(x) - P_1(x)$ mediante el teorema de Rolle reiterado se obtiene $E(x) = \frac{f''(\xi)}{2}(x - x_0)(x - x_1)$, cuyo módulo alcanza su máximo en el punto medio $x_0 + \frac{h}{2}$ con valor $\frac{h^2}{8}\max|f''|$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Selección del par de nodos vecinos:** Dada la curva $\ln(x)$ en $x_0 = 1$ ($y_0 = 0$) y $x_1 = 4$ ($y_1 = \ln 4 \approx 1.38629$), el paso vale $h = 4 - 1 = 3$ y $\Delta y_0 = 1.38629$.",
            r"**Armado de la recta secante $P_1(x)$:** Sustituyendo obtenemos $P_1(x) = 0 + \frac{1.38629}{3}(x - 1) = 0.46210(x - 1)$.",
            r"**Evaluación intermedia y comparación:** En $x = 2$, $P_1(2) = 0.46210$ frente al exacto $\ln(2) = 0.69315$ ($E_a = 0.231$, debido a la fuerte concavidad $\Delta^2 y$ en el tramo ancho $h = 3$).",
            r"**Efecto de reducir $h$:** Si el intervalo se reduce a $[1, 2]$ ($h = 1$), por depender de $h^2$ el error cae aproximadamente $9$ veces, validando el uso lineal en tablas densas.",
        ],
        "practica_pasos_deduccion": [
            r"**Localización del intervalo que contiene $y^* = 1.702$:** En la tabla $[x, y]$, $y_0 = 1.792$ (a $x_0 = 0\ ^\circ\text{C}$) y $y_1 = 1.519$ (a $x_1 = 5\ ^\circ\text{C}$). Como $1.519 < 1.702 < 1.792$, el objetivo está entre $0$ y $5$.",
            r"**Cálculo del paso y primera diferencia:** $h = x_1 - x_0 = 5 - 0 = 5$, $\Delta y_0 = y_1 - y_0 = 1.519 - 1.792 = -0.273$, y la brecha buscada es $y^* - y_0 = 1.702 - 1.792 = -0.090$.",
            r"**Sustitución en la ecuación inversa:** $x = x_0 + \frac{(y^* - y_0)\,h}{\Delta y_0} = 0 + \frac{(-0.090)(5)}{-0.273} = \frac{0.45}{0.273}$.",
            r"**Resultado e interpretación:** Efectuando la división resulta $x \approx 1.64835\ ^\circ\text{C}$ (reportado truncado en el apunte como $1.64\ ^\circ\text{C}$); la pendiente negativa refleja que la viscosidad disminuye al subir la temperatura.",
        ],
    },
    "cuadratica_inversa": {
        "deduccion_teorica": [
            r"**Polinomio de Newton-Gregory de grado $2$ con paso uniforme $h$:** Considerando tres nodos equidistantes $x_0, x_1 = x_0+h, x_2 = x_0+2h$, el polinomio interpolante cuadrático es $P_2(x) = y_0 + \frac{\Delta y_0}{h}(x - x_0) + \frac{\Delta^2 y_0}{2!h^2}(x - x_0)(x - x_1)$.",
            r"**Cambio de variable $u = x - x_0$:** Como $x - x_1 = (x - x_0) - h = u - h$, reescribimos $(x - x_0)(x - x_1) = u^2 - h u$ e igualamos $P_2(x) = y^*$.",
            r"**Agrupamiento en ecuación cuadrática estándar $au^2 + bu + c = 0$:** Expandiendo y ordenando potencias de $u$: $\left(\frac{\Delta^2 y_0}{2h^2}\right)u^2 + \left(\frac{\Delta y_0}{h} - \frac{\Delta^2 y_0}{2h}\right)u + (y_0 - y^*) = 0$.",
            r"**Resolución y criterio de selección de raíz:** Aplicando la resolvente $u_{1,2} = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ obtenemos dos candidatos $x_{1,2} = x_0 + u_{1,2}$: una sola raíz pertenece físicamente al intervalo tabular $[x_0, x_2]$, mientras la otra es una rama parabólica externa que debe descartarse.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Cálculo de diferencias finitas sobre tres nodos equidistantes:** Calculamos $\Delta y_0 = y_1 - y_0$, $\Delta y_1 = y_2 - y_1$ y la diferencia segunda $\Delta^2 y_0 = \Delta y_1 - \Delta y_0$ que mide la curvatura.",
            r"**Construcción de coeficientes $(a, b, c)$:** Evaluamos $a = \frac{\Delta^2 y_0}{2h^2}$, $b = \frac{\Delta y_0}{h} - \frac{\Delta^2 y_0}{2h}$ y $c = y_0 - y^*$ sobre el parámetro desplazado $u = x - x_0$.",
            r"**Cálculo del discriminante $\Delta = b^2 - 4ac$:** Verificamos $\Delta \ge 0$ (si $\Delta < 0$, el valor $y^*$ no es cortado por la parábola real ajustada).",
            r"**Filtrado al dominio del tramo:** De las dos raíces $x_0 + u_1$ y $x_0 + u_2$, seleccionamos únicamente la que satisface $x_0 \le x \le x_2$.",
        ],
        "practica_pasos_deduccion": [
            r"**Diferencias en los $3$ primeros nodos ($x = 0, 5, 10$, $h = 5$):** $\Delta y_0 = 1.519 - 1.792 = -0.273$; $\Delta y_1 = 1.308 - 1.519 = -0.211$; diferencia segunda $\Delta^2 y_0 = -0.211 - (-0.273) = +0.062$.",
            r"**Sustitución en el polinomio cuadrático igualado a $y^* = 1.702$:** Como $x_0 = 0$, $u = x$. Así $a = \frac{0.062}{2(25)} = 0.00124$, $b = \frac{-0.273}{5} - \frac{0.062}{10} = -0.0546 - 0.0062 = -0.0608$, y $c = 1.792 - 1.702 = +0.090$. Queda $0.00124 x^2 - 0.0608 x + 0.09 = 0$.",
            r"**Resolvente cuadrática:** Discriminante $\Delta = (-0.0608)^2 - 4(0.00124)(0.09) = 0.00369664 - 0.0004464 = 0.00325024 \implies \sqrt{\Delta} \approx 0.057011$. Obtenemos $x_1 = \frac{0.0608 + 0.057011}{0.00248} \approx 47.50$ y $x_2 = \frac{0.0608 - 0.057011}{0.00248} \approx 1.5278\ ^\circ\text{C}$ (apunte: $1.52\ ^\circ\text{C}$).",
            r"**Selección y mejora frente a lineal:** Descartamos $47.50$ (muy fuera de $[0, 10]$) y elegimos $x \approx 1.53\ ^\circ\text{C}$, que corrige la concavidad respecto al $1.65\ ^\circ\text{C}$ lineal.",
        ],
    },
    "lagrange": {
        "deduccion_teorica": [
            r"**Objetivo de interpolación sobre $n+1$ nodos cualesquiera $(x_k, y_k)$:** Buscamos el único polinomio $P_n(x)$ de grado $\le n$ tal que $P_n(x_i) = y_i$ para todo $i = 0 \dots n$, incluso cuando los pasos $x_{i+1} - x_i$ son variables.",
            r"**Construcción de funciones base delta de Kronecker $L_k(x)$:** Descomponemos $P_n(x) = \sum_{k=0}^n y_k L_k(x)$ exigiendo que cada polinomio de grado $n$ cumpla $L_k(x_j) = 0$ si $j \ne k$ y $L_k(x_k) = 1$.",
            r"**Raíces forzadas y normalización del denominador:** Para anularse en todos los nodos $j \ne k$, el numerador de $L_k(x)$ debe ser $\prod_{j \ne k}(x - x_j)$. Al evaluar ese producto en $x = x_k$ vale $\prod_{j \ne k}(x_k - x_j) \ne 0$, por lo que dividiendo por esa constante garantizamos $L_k(x_k) = 1$: $L_k(x) = \prod_{\substack{j=0\\ j \ne k}}^n \frac{x - x_j}{x_k - x_j}$.",
            r"**Unicidad y error del polinomio interpolante:** Si existieran dos polinomios $P_n, Q_n$ que pasan por los mismos $n+1$ puntos, su diferencia $P_n - Q_n$ tendría grado $\le n$ con $n+1$ raíces, obligándola a ser idénticamente nula ($P_n \equiv Q_n$). El error cumple $E(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}\prod_{i=0}^n (x - x_i)$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Nodos desiguales de $\ln(x)$:** $x = [1, 4, 6]$ (pasos $3$ y $2$, no equidistante) con $y = [0, 1.386294, 1.791759]$ y objetivo $x^* = 2$.",
            r"**Evaluación de los $3$ pesos base $L_k(2)$:** $L_0(2) = \frac{(2-4)(2-6)}{(1-4)(1-6)} = \frac{8}{15} \approx 0.533333$; $L_1(2) = \frac{(2-1)(2-6)}{(4-1)(4-6)} = \frac{-4}{-6} \approx 0.666667$; $L_2(2) = \frac{(2-1)(2-4)}{(6-1)(6-4)} = \frac{-2}{10} = -0.200000$ (nótese que suman exactamente $1$).",
            r"**Combinación lineal ponderada:** $P_2(2) = 0 \cdot L_0(2) + 1.386294 \cdot (0.666667) + 1.791759 \cdot (-0.2) = 0.924196 - 0.358352$.",
            r"**Resultado final:** $P_2(2) = 0.565844$, obteniendo el único polinomio de grado $2$ que interpola esos tres puntos.",
        ],
        "practica_pasos_deduccion": [
            r"**Verificación de nodos no equidistantes:** Las temperaturas son $T = [100, 120, 200, 220]$ (pasos $20, 80, 20$), luego no aplican fórmulas de paso constante y es obligatorio usar Lagrange para $T^* = 180$.",
            r"**Producto global $\Pi(x) = \prod (180 - T_j)$:** $(180-100)(180-120)(180-200)(180-220) = (80)(60)(-20)(-40) = 3\,840\,000$.",
            r"**Bases y aportes $P_k L_k(180)$ término a término:** $k=0$: $\frac{3840000}{80(-20)(-100)(-120)} = -0.2 \implies 1.01(-0.2) = -0.202$; $k=1$: $\frac{3840000}{60(20)(-80)(-100)} = 0.4 \implies 3.61(0.4) = 1.444$; $k=2$: $\frac{3840000}{-20(100)(80)(-20)} = 1.2 \implies 15.5(1.2) = 18.6$; $k=3$: $\frac{3840000}{-40(120)(100)(20)} = -0.4 \implies 23.2(-0.4) = -9.28$.",
            r"**Suma total:** $P(180) = -0.202 + 1.444 + 18.600 - 9.280 = 10.562\ \text{Pa}$ (exactamente la presión interpolada de grado $3$).",
        ],
    },
    "newton_gregory_asc": {
        "deduccion_teorica": [
            r"**Base polinomial incremental de Newton en nodos equidistantes $x_k = x_0 + kh$:** Planteamos $P_n(x) = a_0 + a_1(x - x_0) + a_2(x - x_0)(x - x_1) + \dots + a_n \prod_{j=0}^{n-1}(x - x_j)$, de modo que agregar un punto solo añade un término nuevo sin alterar los coeficientes anteriores.",
            r"**Identificación de $a_k$ mediante diferencias finitas progresivas $\Delta^k y_0$:** Evaluando sucesivamente en $x_0, x_1, x_2$: $a_0 = y_0$; $a_1 = \frac{y_1 - y_0}{h} = \frac{\Delta y_0}{h}$; $a_2 = \frac{y_2 - 2y_1 + y_0}{2!h^2} = \frac{\Delta^2 y_0}{2!h^2}$, y por inducción $a_k = \frac{\Delta^k y_0}{k!\,h^k}$ donde $\Delta^{k}y_i = \Delta^{k-1}y_{i+1} - \Delta^{k-1}y_i$.",
            r"**Adimensionalización mediante $u = \frac{x - x_0}{h}$:** Sustituyendo $x - x_j = (x - x_0) - jh = h(u - j)$ en cada producto $\prod_{j=0}^{k-1}(x - x_j) = h^k \prod_{j=0}^{k-1}(u - j)$, los factores $h^k$ del numerador y denominador se cancelan.",
            r"**Fórmula final de Newton-Gregory Ascendente y error:** Queda $P_n(x) = y_0 + \sum_{k=1}^n \binom{u}{k} \Delta^k y_0 = y_0 + u\Delta y_0 + \frac{u(u-1)}{2!}\Delta^2 y_0 + \dots$, usando la diagonal superior de la tabla de diferencias; el error al truncar en grado $k$ es $E_k \approx \binom{u}{k+1} h^{k+1} f^{(k+1)}(\xi)$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Tabla de muestra para $y = x^2$ en $x = [0, 1, 2, 3, 4]$ ($h = 1$):** Las ordenadas iniciales son $y = [0, 1, 4, 9, 16]$.",
            r"**Columna 1 ($\Delta y_i = y_{i+1} - y_i$):** $[1-0, 4-1, 9-4, 16-9] = [1, 3, 5, 7]$.",
            r"**Columna 2 ($\Delta^2 y_i = \Delta y_{i+1} - \Delta y_i$):** $[3-1, 5-3, 7-5] = [2, 2, 2]$ (constante no nula).",
            r"**Columna 3 ($\Delta^3 y$) y conclusión:** $[2-2, 2-2] = [0, 0]$. Que $\Delta^{k+1}y = 0$ demuestra que los datos provienen exactamente de un polinomio de grado $k = 2$ (sirve además en la práctica para detectar errores de tipeo en una fila).",
        ],
        "practica_pasos_deduccion": [
            r"**Validación de equidistancia y parámetro $u$:** Los nodos $x = [0.2, 0.5, 0.8, 1.1, 1.4]$ tienen paso constante $h = 0.3$. Al anclar en $x_0 = 0.2$ para $x^* = 0.35$, obtenemos $u = \frac{0.35 - 0.2}{0.3} = 0.5$.",
            r"**Tabla de diferencias progresivas $\Delta^k y_0$ (diagonal superior):** $y_0 = 0.258$; $\Delta y_0 = 2.535 - 0.258 = 2.277$; $\Delta^2 y_0 = 2.367 - 2.277 = 0.090$; $\Delta^3 y_0 = -0.547 - 0.090 = -0.637$; $\Delta^4 y_0 = 2.709 - (-0.637) = 3.346$.",
            r"**Cálculo de coeficientes binomiales en $u = 0.5$:** $k=1$: $0.5$; $k=2$: $\frac{0.5(-0.5)}{2} = -0.125$; $k=3$: $\frac{-0.125(-1.5)}{3} = 0.0625$; $k=4$: $\frac{0.0625(-2.5)}{4} = -0.0390625$.",
            r"**Suma término a término:** $P_4(0.35) = 0.258 + (0.5)(2.277) + (-0.125)(0.090) + (0.0625)(-0.637) + (-0.0390625)(3.346) = 0.258 + 1.1385 - 0.01125 - 0.0398125 - 0.130703125 = 1.214734375$.",
        ],
    },
    "newton_gregory_desc": {
        "deduccion_teorica": [
            r"**Reordenamiento desde el extremo final $x_n$:** Cuando $x^*$ está próximo al final de una tabla equidistante $x_n = x_0 + nh$, planteamos el polinomio incremental retrocediendo hacia atrás: $P_n(x) = b_0 + b_1(x - x_n) + b_2(x - x_n)(x - x_{n-1}) + \dots + b_n \prod_{j=0}^{n-1}(x - x_{n-j})$.",
            r"**Operador de diferencias regresivas (retrógradas) $\nabla$:** Definido como $\nabla y_i = y_i - y_{i-1}$ y en orden superior $\nabla^k y_i = \nabla^{k-1} y_i - \nabla^{k-1} y_{i-1}$. Coincide numéricamente con la diagonal inferior de la misma tabla triangular $\Delta$. Evaluando en $x_n, x_{n-1}, \dots$ resulta $b_k = \frac{\nabla^k y_n}{k!\,h^k}$.",
            r"**Parámetro adimensional $v = \frac{x - x_n}{h}$:** Como $x_{n-j} = x_n - jh$, cada factor cumple $x - x_{n-j} = (x - x_n) + jh = h(v + j)$, cancelando nuevamente $h^k$ entre numerador y denominador.",
            r"**Fórmula final y Teorema de Unicidad:** Obtenemos $P_n(x) = y_n + v\nabla y_n + \frac{v(v+1)}{2!}\nabla^2 y_n + \frac{v(v+1)(v+2)}{3!}\nabla^3 y_n + \dots$. En grado completo produce algebraicamente el *mismo* polinomio que Newton-Gregory Ascendente y Lagrange, pero al truncar concentra la precisión en la cola de la tabla.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Elección de ancla en el borde superior de la serie:** Tomamos $x_n$ como referencia de modo que el punto objetivo $x^* \le x_n$ mantenga $|v| = \left|\frac{x^* - x_n}{h}\right| < 1$.",
            r"**Lectura de la diagonal inferior de diferencias:** Tomamos la última entrada no nula de cada columna ($\nabla y_n, \nabla^2 y_n, \nabla^3 y_n, \dots$), que involucra únicamente los últimos nodos de la tabla.",
            r"**Evaluación con signos positivos en los factores de $v$:** Aunque $v$ suele ser negativo ($x^* < x_n$), los desplazamientos dentro de los productos son $+1, +2, \dots$: $v(v+1)(v+2)/3!$.",
            r"**Truncamiento y equivalencia global:** Si sumamos todos los términos hasta grado $n$, coincidimos con la variante Ascendente; si truncamos a grado $2$ o $3$ cerca de $x_n$, el error residual $\binom{v+k}{k+1}h^{k+1}f^{(k+1)}(\xi)$ es mínimo por ser $|v|$ pequeño.",
        ],
        "practica_pasos_deduccion": [
            r"**Parámetro $v$ para $x^* = 1.37$ ($x_n = 1.4$, $h = 0.3$):** Calculamos $v = \frac{1.37 - 1.4}{0.3} = \frac{-0.03}{0.3} = -0.1$ (muy pequeño, ideal para anclar al final).",
            r"**Diagonal inferior retrógrada $\nabla^k y_n$:** $y_4 = 10.704$; $\nabla y_4 = 10.704 - 6.722 = 3.982$; $\nabla^2 y_4 = 3.982 - 1.820 = 2.162$; $\nabla^3 y_4 = 2.162 - (-0.547) = 2.709$; $\nabla^4 y_4 = 2.709 - (-0.637) = 3.346$.",
            r"**Factores combinatorios en $v = -0.1$:** $k=1$: $-0.1$; $k=2$: $\frac{(-0.1)(0.9)}{2} = -0.045$; $k=3$: $\frac{-0.045(1.9)}{3} = -0.0285$; $k=4$: $\frac{-0.0285(2.9)}{4} = -0.0206625$.",
            r"**Suma progresiva hacia el valor final:** $P_4(1.37) = 10.704 - 0.3982 - 0.09729 - 0.0772065 - 0.069136725 = 10.06216678$.",
        ],
    },
    "formula_gauss": {
        "deduccion_teorica": [
            r"**Anclaje central para minimizar $|u|$:** Para un $x^*$ en el centro de una tabla equidistante, anclarse en $x_0$ o $x_n$ hace $|u|$ o $|v|$ grande ($\ge 1.5$), amplificando el error al truncar. Si reindexamos los nodos alrededor de un nodo central $x_0^{\mathrm{c}}$ ($\dots, x_{-1}, x_0, x_1, \dots$) y definimos $u = \frac{x - x_0^{\mathrm{c}}}{h}$, logramos $|u| \le 0.5$.",
            r"**Criterio de recorrido en zig-zag (Gauss Progresiva vs Regresiva):** Por Unicidad de Newton podemos elegir en cada grado qué nodo nuevo incorporar: **Gauss Progresiva** ($u \ge 0$) agrega alternadamente hacia la derecha e izquierda ($x_0, x_1, x_{-1}, x_2, x_{-2}, \dots$), haciendo que en la tabla escalonada $\Delta$ baje primero ($\Delta f(x_0)$) y luego suba/baje en zig-zag. **Gauss Regresiva** ($u < 0$) incorpora ($x_0, x_{-1}, x_1, x_{-2}, x_2, \dots$), subiendo primero ($\Delta f(x_0 - h)$).",
            r"**Deducción de los coeficientes polinomiales por unicidad nodal:** Al agregar secuencialmente $x_0, x_1, x_{-1}, x_2, x_{-2}$, los factores de Newton en $x$ son $(x-x_0)$, $(x-x_0)(x-x_1)$, $(x-x_0)(x-x_1)(x-x_{-1})$, etc. En función de $u$ quedan $u$, $u(u-1)$, $(u-1)u(u+1)$, $(u-2)(u-1)u(u+1)$, correspondiendo respectivamente a $\Delta f(x_0)$, $\Delta^2 f(x_0-h)$, $\Delta^3 f(x_0-h)$ y $\Delta^4 f(x_0-2h)$.",
            r"**Ventaja decisiva al truncar antes del grado máximo:** Porque la sucesión de diferencias centrales rodea simétricamente a $x^*$, el resto $|R_k| \propto \prod_{j=0}^{k} |u - j_{\text{zig-zag}}|$ resulta muchísimo menor que el de Gregory Ascendente o Descendente para igual grado $k < n - 1$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Tabla de 5 nodos reindexados alrededor del centro:** Con $5$ puntos ($i = 0, 1, 2, 3, 4$), el ancla central de Gauss Progresiva es $x_0^{\mathrm{c}} = x_2$.",
            r"**Ruta en zig-zag de Gauss Progresiva ($0 \le u \le 0.5$):** Toma $f(x_2)$ en fila $2$, baja a $\Delta f(x_2)$ en fila $2$, sube a $\Delta^2 f(x_1)$ en fila $1$, permanece en $\Delta^3 f(x_1)$ en fila $1$ y cierra en $\Delta^4 f(x_0)$ en fila $0$ (trayectoria de filas $2, 2, 1, 1, 0$).",
            r"**Ruta simétrica de Gauss Regresiva ($-0.5 \le u < 0$):** Parte de $f(x_2)$ en fila $2$, sube inmediatamente a $\Delta f(x_1)$ en fila $1$, sigue horizontal en $\Delta^2 f(x_1)$ (fila $1$), sube a $\Delta^3 f(x_0)$ (fila $0$) y cierra en $\Delta^4 f(x_0)$ (filas $2, 1, 1, 0, 0$).",
            r"**Unicidad completa vs ventaja truncada:** Al evaluar los $5$ términos ($k=4$) ambas coinciden con Lagrange y Newton-Gregory; al cortar en $k=2$ o $k=3$, el truncamiento central logra el mínimo residuo.",
        ],
        "practica_pasos_deduccion": [
            r"**Elección del ancla central y cálculo de $u$:** Para $x = [0.2, 0.5, 0.8, 1.1, 1.4]$ ($h = 0.3$) y $x^* = 0.90$, el nodo central es $x_2 = 0.8$ ($y_2 = 4.902$). Calculamos $u = \frac{0.90 - 0.8}{0.3} = \frac{1}{3} \approx +0.333333 \in [0, 0.5]$, indicando **Gauss Progresiva**.",
            r"**Extracción del camino zig-zag $[2, 2, 1, 1, 0]$ en la tabla $\Delta$:** $f(0.8) = 4.902$; $\Delta f(0.8) = 6.722 - 4.902 = 1.820$; $\Delta^2 f(0.5) = 1.820 - 2.367 = -0.547$; $\Delta^3 f(0.5) = 2.162 - (-0.547) = 2.709$; $\Delta^4 f(0.2) = 3.346$.",
            r"**Factores polinomiales en $u = 1/3$:** $k=1$: $u = 1/3$; $k=2$: $\frac{u(u-1)}{2!} = \frac{(1/3)(-2/3)}{2} = -\frac{1}{9}$; $k=3$: $\frac{(u-1)u(u+1)}{3!} = \frac{(-2/3)(1/3)(4/3)}{6} = -\frac{4}{81}$; $k=4$: $\frac{(u-2)(u-1)u(u+1)}{4!} = \frac{(-5/3)(-2/3)(1/3)(4/3)}{24} = +\frac{5}{243}$.",
            r"**Suma acumulada término a término:** $4.902 + \frac{1.820}{3} - \frac{-0.547}{9} - \frac{4(2.709)}{81} + \frac{5(3.346)}{243} = 4.902 + 0.606667 + 0.060778 - 0.133778 + 0.068848 = 5.50451440$, idéntico en grado $4$ a las demás fórmulas.",
        ],
    },
    "diferenciacion_numerica": {
        "deduccion_teorica": [
            r"**Derivación directa del polinomio interpolante:** Dado que no poseemos $f(x)$ analítica sino una tabla equidistante, derivamos el polinomio de Newton-Gregory $P_n(u) = y_0 + u\Delta y_0 + \frac{u^2-u}{2}\Delta^2 y_0 + \frac{u^3-3u^2+2u}{6}\Delta^3 y_0 + \frac{u^4-6u^3+11u^2-6u}{24}\Delta^4 y_0 + \dots$ respecto a $x$.",
            r"**Aplicación de la regla de la cadena ($\frac{du}{dx} = \frac{1}{h}$):** Como $u = \frac{x - x_0}{h}$, obtenemos $\frac{dP}{dx} = \frac{1}{h}\frac{dP}{du}$ y $\frac{d^2P}{dx^2} = \frac{1}{h^2}\frac{d^2P}{du^2}$.",
            r"**Derivadas generales en función de $u$ (y especialización en el nodo $u = 0$):** Derivando término a término sale $f'(x) \approx \frac{1}{h}\left[\Delta y_0 + \frac{2u-1}{2}\Delta^2 y_0 + \frac{3u^2-6u+2}{6}\Delta^3 y_0 + \frac{4u^3-18u^2+22u-6}{24}\Delta^4 y_0\right]$; evaluando en $u = 0$ recuperamos los coeficientes $(1, -\frac{1}{2}, \frac{1}{3}, -\frac{1}{4})$. Análogamente para $v = \frac{x-x_n}{h}$ con diferencias retrógradas $\nabla^k y_n$.",
            r"**Compromiso entre error de truncamiento y cancelación numérica:** Reducir $h$ disminuye el error analítico ($\mathcal{O}(h^k)$), pero como la resta $y_{i+1}-y_i$ cancela dígitos significativos al acercar nodos y luego dividimos por $h$ o $h^2$, el error de redondeo escala como $\mathcal{O}(\varepsilon_{\text{maq}}/h)$, causando divergencia si $h$ es demasiado pequeño.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Planteo del experimento $\frac{e^{1+h} - e^1}{h}$ frente al exacto $e \approx 2.718281828$:** Aproximamos la derivada por cociente incremental progresivo variando el paso $h$ desde $10^{-1}$ hasta $10^{-14}$.",
            r"**Régimen dominado por truncamiento ($h \in [10^{-1}, 10^{-8}]$):** Al achicar $h$, el error cae linealmente en $\mathcal{O}(h)$ pasando de $0.14$ (en $h=0.1$) hasta alcanzar su mínimo en $h \approx 10^{-8}$ con error $5.10 \times 10^{-9}$.",
            r"**Aparición de la cancelación catastrófica ($h \le 10^{-9}$):** Con `float64` (~16 dígitos), $e^{1+10^{-14}}$ coincide en $14$ dígitos con $e^1$: al restarlos quedan apenas $2$ dígitos válidos de ruido.",
            r"**Amplificación al dividir por $h$:** Dividir ese ruido residual por $10^{-14}$ dispara el error a $0.0537$, ilustrando la barrera física del punto flotante.",
        ],
        "practica_pasos_deduccion": [
            r"**Construcción de diferencias en la tabla $x = [12 \dots 17]$ ($h = 1$):** Para $x_0 = 12$ ($y_0 = 0.96262$), $\Delta y_0 = -0.01544$, $\Delta^2 y_0 = -0.00066$, $\Delta^3 y_0 = 0.00789$, $\Delta^4 y_0 = -0.01494$, $\Delta^5 y_0 = 0.02116$.",
            r"**Primera derivada en el nodo inicial $x = 12$ ($u = 0$):** Aplicando $f'(12) = \frac{1}{1}\left[-0.01544 - \frac{1}{2}(-0.00066) + \frac{1}{3}(0.00789) - \frac{1}{4}(-0.01494) + \frac{1}{5}(0.02116)\right] = -0.0083175$ (con las diferencias hasta orden $4$ o $5$).",
            r"**Derivada en punto no tabular $x^* = 12.4$ ($u = 0.4$):** Sustituimos $u = 0.4$ en la fórmula general: coefs $(1, \frac{0.8-1}{2}=-0.1, \frac{0.48-2.4+2}{6}=0.01333, \frac{0.256-2.88+8.8-6}{24}=0.00733, \dots)$, resultando $f'(12.4) \approx -0.0153909$.",
            r"**Segunda derivada descendente en $x = 17$ ($v = 0$):** Usando la diagonal inferior retrógrada $\nabla^2 y_5 = -0.00010, \nabla^3 y_5 = -0.00163, \nabla^4 y_5 = 0.00024, \nabla^5 y_5 = 0.02116$ en $\frac{1}{1^2}[\nabla^2 y_5 + \nabla^3 y_5 + \frac{11}{12}\nabla^4 y_5 + \dots]$ se deduce $f''(17) \approx 0.011254$ (según orden truncado de cátedra).",
        ],
    },
    "trapecio": {
        "deduccion_teorica": [
            r"**Aproximación lineal en cada subintervalo $[x_i, x_{i+1}]$:** Reemplazando $f(x)$ por la recta secante $P_1(x) = y_i + \frac{y_{i+1}-y_i}{h}(x - x_i)$ e integrando en $[x_i, x_{i+1}]$ de longitud $h$: $\int_{x_i}^{x_{i+1}} P_1(x)\,dx = \frac{h}{2}(y_i + y_{i+1})$ (fórmula geométrica del área de un trapecio).",
            r"**Regla compuesta sumando $n$ trapecios adyacentes:** $A \approx \sum_{i=0}^{n-1} \frac{h}{2}(y_i + y_{i+1}) = \frac{h}{2}\left[(y_0 + y_1) + (y_1 + y_2) + \dots + (y_{n-1} + y_n)\right]$.",
            r"**Agrupación en notación de cátedra ($E, P, I$):** Cada nodo interior aparece dos veces (como techo derecho de un trapecio y techo izquierdo del siguiente), mientras los extremos aparecen una vez: $A = \frac{h}{2}[E + 2P + 2I]$, donde $E = y_0 + y_n$, $P = \sum y_{\text{pares int.}}$ e $I = \sum y_{\text{impares int.}}$.",
            r"**Deducción de la cota del error global $\mathcal{O}(h^2)$:** Integrando el error lineal $-\frac{1}{2}f''(\xi)(x-x_i)(x_{i+1}-x)$ sobre cada subintervalo da error local $-\frac{h^3}{12}f''(\xi_i)$. Sumando $n = \frac{b-a}{h}$ tramos se obtiene $E_{\text{global}} = -\frac{(b-a)h^2}{12}f''(\bar{\xi}) \implies |E_a| \le \frac{(b-a)^3}{12n^2}\max|f''(x)|$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Evaluación en los $n + 1 = 3$ nodos ($h = 0.4$):** $f(0) = 0.2$, $f(0.4) = 2.456$ y $f(0.8) = 0.232$.",
            r"**Agrupamiento de extremos y nodo interior:** $E = 0.2 + 0.232 = 0.432$ y término interior duplicado $2(2.456) = 4.912$.",
            r"**Aplicación con $n = 2$:** $A = \frac{0.4}{2}(0.432 + 4.912) = 0.2(5.344) = 1.0688$ (error relativo $34.9\%$ frente a $1.640533$).",
            r"**Efecto de refinar a $n = 10$ ($h$ dividido por $5$):** Por la dependencia $\mathcal{O}(1/n^2)$, el error se divide aproximadamente por $25$, bajando a menos del $1.5\%$.",
        ],
        "practica_pasos_deduccion": [
            r"**Identificación de parámetros y partición de nodos:** Con $x = [0.5, 5.5, 10.5, 15.5, 20.5, 25.5]$ e $y = [0.05, 6.05, 22.05, 48.05, 84.05, 130.05]$, hay $n = 5$ subintervalos de ancho constante $h = 5$.",
            r"**Clasificación de ordenadas en $E, P, I$:** Extremos $E = y_0 + y_5 = 0.05 + 130.05 = 130.10$; nodos interiores pares $P = y_2 + y_4 = 22.05 + 84.05 = 106.10$ (o según índice $1..n-1$ con $I = y_1 + y_3 = 6.05 + 48.05 = 54.10$, sumando $P+I = 160.20$).",
            r"**Sustitución en la fórmula compuesta del Trapecio:** $A = \frac{5}{2}[130.10 + 2(160.20)] = 2.5 \times [130.10 + 320.40] = 2.5 \times 450.50 = 1126.25$.",
            r"**Análisis del error frente a la integral exacta $\int_{0.5}^{25.5} \frac{x^2}{5}\,dx = \left[\frac{x^3}{15}\right]_{0.5}^{25.5} = 1105.4167$:** El desvío es $+20.8333$, coincidiendo exactamente con la fórmula teórica pues $f''(x) = \frac{2}{5}$ constante da $-\frac{(25)(5^2)}{12}(-2/5)$ con signo cambiado: $\frac{25 \cdot 25 \cdot 0.4}{12} = 20.8333$.",
        ],
    },
    "simpson_1_3": {
        "deduccion_teorica": [
            r"**Polinomio cuadrático sobre $3$ puntos equidistantes $[-h, 0, h]$:** Tomamos $P_2(x) = y_1 + \frac{y_2 - y_0}{2h}x + \frac{y_0 - 2y_1 + y_2}{2h^2}x^2$ (trasladando el origen al nodo medio $x_1 = 0$) para aprovechar la simetría de potencias impares.",
            r"**Integración simétrica exacta:** $\int_{-h}^{h} P_2(x)\,dx = \left[y_1 x + 0 + \frac{y_0 - 2y_1 + y_2}{6h^2}x^3\right]_{-h}^{h} = 2h y_1 + \frac{h}{3}(y_0 - 2y_1 + y_2) = \frac{h}{3}(y_0 + 4y_1 + y_2)$.",
            r"**Extensión compuesta ($n$ par) en notación $E, I, P$:** Encadenando $n/2$ parejas parabólicas, los extremos ($y_0, y_n$) aparecen $1$ vez, los nodos centrales impares ($y_1, y_3, \dots$) tienen peso $4$, y los nodos pares compartidos ($y_2, y_4, \dots$) suman $1+1 = 2$: $A = \frac{h}{3}[E + 4I + 2P]$.",
            r"**Por qué alcanza orden cuarto $\mathcal{O}(h^4)$:** Al integrar un término cúbico $x^3$ sobre el intervalo simétrico $[-h, h]$ su integral se anula idénticamente! Por ello el primer término de resto no nulo proviene de $f^{(4)}$, dando error local $-\frac{h^5}{90}f^{(4)}(\xi)$ y cota global $|E_a| \le \frac{(b-a)^5}{180\,n^4}\max|f^{(4)}|$ (exacta para cualquier polinomio hasta grado $3$).",
        ],
        "ejemplo_teorico_pasos": [
            r"**Evaluación sobre un par de intervalos $[0, h, 2h]$:** Centrando el eje en el nodo medio $x_1$ entre $[-h, h]$, integramos directamente una parábola por los tres puntos obteniendo los pesos $\frac{h}{3}(1, 4, 1)$.",
            r"**Prueba de grado de exactitud con $f(x) = x^3$ en $[-h, h]$:** La integral exacta de $x^3$ en $[-h, h]$ es $0$. Evaluando la fórmula de Simpson: $\frac{h}{3}[(-h)^3 + 4(0)^3 + h^3] = 0$, coincidiendo exactamente.",
            r"**Escalado de error ante refinamiento $\mathcal{O}(h^4)$:** Si dividimos el paso $h$ a la mitad (duplicando $n$), el denominador $n^4$ aumenta $2^4 = 16$ veces.",
            r"**Conclusión práctica:** Con el mismo costo de nodos que Trapecio, Simpson 1/3 gana dos órdenes extra de precisión sobre funciones suaves con $n$ par.",
        ],
        "practica_pasos_deduccion": [
            r"**Verificación de condiciones sobre $[0.5, 20.5]$:** Tenemos $5$ puntos ($x = [0.5, 5.5, 10.5, 15.5, 20.5]$), lo cual define $n = 4$ subintervalos (número **par**, habilita Simpson 1/3 puro) de paso constante $h = 5$.",
            r"**Cálculo de grupos $E, I, P$:** Extremos $E = y_0 + y_4 = 0.05 + 84.05 = 84.10$; posiciones impares interiores $I = y_1 + y_3 = 6.05 + 48.05 = 54.10$; posiciones pares interiores $P = y_2 = 22.05$.",
            r"**Sustitución en $A = \frac{h}{3}[E + 4I + 2P]$:** $A = \frac{5}{3}[84.10 + 4(54.10) + 2(22.05)] = \frac{5}{3}[84.10 + 216.40 + 44.10] = \frac{5}{3}(344.60) = 574.3333\dots$",
            r"**Confirmación de exactitud:** La integral analítica $\int_{0.5}^{20.5} \frac{x^2}{5}\,dx = \frac{20.5^3 - 0.5^3}{15} = \frac{8615.125 - 0.125}{15} = \frac{8615}{15} = 574.3333\dots$, con error $0$ pues $f^{(4)}(x) \equiv 0$.",
        ],
    },
    "simpson_3_8_combinado": {
        "deduccion_teorica": [
            r"**Polinomio cúbico de Lagrange/Gregory sobre $4$ nodos ($3$ subintervalos):** Tomamos $[x_0, x_1, x_2, x_3]$ con paso $h$ e integramos el polinomio de Newton-Gregory Ascendente $P_3(u) = y_0 + u\Delta y_0 + \frac{u^2-u}{2}\Delta^2 y_0 + \frac{u^3-3u^2+2u}{6}\Delta^3 y_0$ desde $u = 0$ hasta $u = 3$ (usando $dx = h\,du$).",
            r"**Integración de cada término en $[0, 3]$:** $\int_0^3 du = 3$; $\int_0^3 u\,du = \frac{9}{2}$; $\int_0^3 \frac{u^2-u}{2}\,du = \frac{9}{4}$; $\int_0^3 \frac{u^3-3u^2+2u}{6}\,du = \frac{3}{8}$.",
            r"**Expansión en términos de $y_0, y_1, y_2, y_3$:** Reemplazando $\Delta y_0 = y_1-y_0$, $\Delta^2 y_0 = y_2 - 2y_1 + y_0$ y $\Delta^3 y_0 = y_3 - 3y_2 + 3y_1 - y_0$ y simplificando coeficientes se obtiene directamente $A_{3/8} = \frac{3h}{8}(y_0 + 3y_1 + 3y_2 + y_3)$.",
            r"**Estrategia combinada para $n$ impar manteniendo orden $\mathcal{O}(h^4)$:** Su error local es $-\frac{3h^5}{80}f^{(4)}(\xi)$ (mismo orden $\mathcal{O}(h^4)$ que Simpson 1/3!). Cuando $n \ge 5$ es impar y no múltiplo de $3$, asignar $3$ intervalos a Simpson $3/8$ deja un resto $n - 3$ **par**, que se resuelve con Simpson $1/3$ sin degradar la precisión global al $\mathcal{O}(h^2)$ de Trapecio.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Diagnóstico de paridad cuando $n = 5$:** $n = 5$ no es par (falla Simpson $1/3$ puro) ni múltiplo de $3$ (falla Simpson $3/8$ puro).",
            r"**Partición en sub-bloques compatibles:** Separamos los primeros $3$ intervalos $[x_0, x_3]$ (donde $n_1 = 3$ sí cumple $3/8$) y los últimos $2$ intervalos $[x_3, x_5]$ (donde $n_2 = 2$ es par y aplica $1/3$).",
            r"**Cálculo independiente y continuidad en el nodo empalme $x_3$:** El valor $y_3$ actúa simultáneamente como límite derecho del bloque $3/8$ y límite izquierdo del bloque $1/3$.",
            r"**Aditividad integral y preservación de orden:** Sumando $A_{\text{total}} = A_{3/8} + A_{1/3}$, todo el dominio se integra con polinomios de grado $\ge 2$ y orden global $\mathcal{O}(h^4)$.",
        ],
        "practica_pasos_deduccion": [
            r"**Análisis de la tabla completa TP6 ($6$ puntos, $n = 5$, $h = 5$):** $[0.5, 5.5, 10.5, 15.5, 20.5, 25.5]$ posee $n = 5$ intervalos (impar), por lo que combinamos $3$ intervalos con Simpson 3/8 más $2$ intervalos con Simpson 1/3.",
            r"**Primer tramo $[0.5, 15.5]$ por Simpson 3/8:** Con $y_0 = 0.05, y_1 = 6.05, y_2 = 22.05, y_3 = 48.05$: $A_1 = \frac{3(5)}{8}[0.05 + 3(6.05) + 3(22.05) + 48.05] = 1.875 \times 132.40 = 248.25$.",
            r"**Segundo tramo $[15.5, 25.5]$ por Simpson 1/3:** Con $y_3 = 48.05, y_4 = 84.05, y_5 = 130.05$: $A_2 = \frac{5}{3}[48.05 + 4(84.05) + 130.05] = \frac{5}{3}(514.30) = 857.16667$.",
            r"**Suma total y contraste exacto:** $A_{\text{total}} = 248.25 + 857.16667 = 1105.41667$, coincidiendo al decimal con la integral exacta $\int_{0.5}^{25.5}\frac{x^2}{5}\,dx = 1105.41667$ (a diferencia de Trapecio que daba $1126.25$).",
        ],
    },
    "euler_heun": {
        "deduccion_teorica": [
            r"**Deducción de Euler mediante Taylor de primer orden:** Expandiendo la solución real $y(x_{i+1}) = y(x_i + h) = y_i + h\,y'(x_i) + \frac{h^2}{2}y''(\xi)$ y truncando en el término lineal ($y'(x_i) = f(x_i, y_i)$) se obtiene $y_{i+1} = y_i + h f(x_i, y_i)$, con error local $\mathcal{O}(h^2)$ y global $\mathcal{O}(h)$.",
            r"**Deducción geométrica de Heun (Regla Trapezoidal Implícita):** Integrando exactamente $y' = f(x, y)$ entre $x_i$ y $x_{i+1}$ resulta $y(x_{i+1}) - y(x_i) = \int_{x_i}^{x_{i+1}} f(x, y(x))\,dx$. Aproximando la integral mediante la regla del Trapecio obtenemos $y_{i+1} = y_i + \frac{h}{2}[f(x_i, y_i) + f(x_{i+1}, y_{i+1})]$.",
            r"**Esquema Predictor-Corrector iterado:** Como $y_{i+1}$ aparece también dentro de $f(x_{i+1}, y_{i+1})$, **predecimos** un valor inicial con Euler: $P(y_{i+1}) = y_i + h f(x_i, y_i)$, evaluamos la pendiente derecha en ese punto y la sustituimos en el **Corrector** $C^{(m)} = y_i + \frac{h}{2}[f(x_i, y_i) + f(x_{i+1}, C^{(m-1)})]$.",
            r"**Convergencia del corrector y orden global:** Cada nueva iteración del corrector reduce el error de punto fijo mientras $\frac{h}{2}|\partial f / \partial y| < 1$, deteniéndose al verificar $|C^{(m)} - C^{(m-1)}| < E$, logrando orden global $\mathcal{O}(h^2)$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Planteo de la EDO discreta $\frac{dv}{dt} = f(t, v) = g - \frac{c}{m}v$:** Dado el estado actual $v(t_i)$ y paso temporal $\Delta t = h$, calculamos la aceleración inicial $k_1 = f(t_i, v_i)$.",
            r"**Paso Predictor de Euler:** Proyectamos una velocidad tentativa lineal $P(v_{i+1}) = v_i + h\,k_1$, suponiendo aceleración constante en todo el intervalo.",
            r"**Evaluación en el destino y promedio trapezoidal (Corrector):** Medimos la nueva aceleración en el punto predicho $k_2 = f(t_i + h, P(v_{i+1}))$ y actualizamos $C^{(1)} = v_i + \frac{h}{2}(k_1 + k_2)$.",
            r"**Refinamiento iterativo hasta tolerancia:** Si $|C^{(1)} - P| \ge E$, recalculamos $k_2 = f(t_{i+1}, C^{(1)})$ y $C^{(2)}$ hasta estabilizar, eliminando el sesgo unilateral de Euler.",
        ],
        "practica_pasos_deduccion": [
            r"**Datos iniciales TP7 Ej. 1 ($y' = \frac{6x^2}{y}$, $y(4) = 16$, $h = 0.1$):** Pendiente inicial en $(4.0, 16.0)$: $k_1 = \frac{6(16)}{16} = 6.0$. Exacta analítica: $y(x) = 2x^{3/2}$.",
            r"**Primer paso ($x_1 = 4.1$) — Predictor de Euler:** $y_{\text{Euler}}(4.1) = P(y_1) = 16 + 0.1(6.0) = 16.600000$ (frente al exacto $2(4.1)^{1.5} \approx 16.603614$).",
            r"**Corrección iterada de Heun en $x_1 = 4.1$:** Pendiente predicha $k_2^{(0)} = \frac{6(4.1)^2}{16.6} \approx 6.075904 \implies C^{(1)} = 16 + \frac{0.1}{2}(6.0 + 6.075904) = 16.603795$. Recalculando con $C^{(1)}$: $k_2^{(1)} = \frac{6(4.1)^2}{16.603795} \approx 6.074515 \implies C^{(2)} = 16.603726$ (ya difiere menos de $0.0001$, error de solo $0.00011$).",
            r"**Avance hasta $x_f = 4.4$ (4 pasos):** Euler acumula deriva lineal terminando en $18.44514$ (error $0.01390$), mientras Heun se pega a la curva exacta concluyendo en $18.45901$ frente al exacto $18.459036$ (error de apenas $2.6 \times 10^{-5}$).",
        ],
    },
    "runge_kutta": {
        "deduccion_teorica": [
            r"**Idea de los métodos Runge-Kutta:** En lugar de derivar analíticamente $f(x, y)$ para usar series de Taylor de orden superior, los métodos RK logran el mismo orden evaluando $f$ en múltiples puntos intermedios dentro $[x_i, x_i + h]$ y combinándolos con pesos adecuados.",
            r"**Deducción de RK2 haciendo coincidir con Taylor de orden 2:** Planteamos $y_{i+1} = y_i + a k_1 + b k_2$ con $k_1 = h f(x_i, y_i)$ y $k_2 = h f(x_i + \alpha h, y_i + \beta k_1)$. Expandiendo $k_2$ en Taylor multivariable $f(x_i + \alpha h, y_i + \beta h f) \approx f + \alpha h f_x + \beta h f_y f$ e igualando término a término con $y(x_i + h) = y_i + h f + \frac{h^2}{2}(f_x + f_y f)$ resultan $a + b = 1$, $b\alpha = \frac{1}{2}$ y $b\beta = \frac{1}{2}$ (eligiendo $a=b=\frac{1}{2}, \alpha=\beta=1$ surge Heun/RK2).",
            r"**Construcción de RK4 Clásico (matching hasta Taylor de grado 4):** Usando cuatro pendientes exploratorias ($k_1$ al inicio, $k_2$ y $k_3$ en el punto medio $x_i + h/2$, $k_4$ al final $x_i + h$), la combinación simétrica $y_{i+1} = y_i + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$ satisface algebraicamente todas las ecuaciones de orden hasta $h^4$.",
            r"**Conexión exacta con Simpson 1/3 y orden de error:** Cuando $f(x, y)$ depende solo de $x$, $k_2 = k_3 = h f(x_i + h/2)$, reduciéndose RK4 a $\frac{h}{6}[f(x_i) + 4f(x_i + h/2) + f(x_i+h)]$ (Simpson $1/3$ sobre $[x_i, x_i+h]$!). Su error local es $\mathcal{O}(h^5)$ y el error acumulado global es $\mathcal{O}(h^4)$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Planteo de independencia en $y$:** Tomamos la EDO integrable directa $y' = g(x)$, donde la solución exacta cumple $y_{i+1} - y_i = \int_{x_i}^{x_i+h} g(x)\,dx$.",
            r"**Evaluación de las $4$ pendientes $k_i$:** $k_1 = h g(x_i)$, $k_2 = h g(x_i + h/2)$, $k_3 = h g(x_i + h/2)$ (igual a $k_2$) y $k_4 = h g(x_i + h)$.",
            r"**Reemplazo en la combinación ponderada de RK4:** $y_{i+1} = y_i + \frac{1}{6}(k_1 + 2k_2 + 2k_2 + k_4) = y_i + \frac{h/2}{3}\left[g(x_i) + 4g\!\left(x_i + \frac{h}{2}\right) + g(x_i + h)\right]$.",
            r"**Identidad demostrada:** La expresión resultante es idéntica a la Regla de Simpson 1/3 con paso $\tilde{h} = h/2$, heredando inmediatamente su precisión de cuarto orden global $\mathcal{O}(h^4)$.",
        ],
        "practica_pasos_deduccion": [
            r"**Paso 1 ($x_0 = 4.0 \to x_1 = 4.1$, $h = 0.1$) para $RK2$:** $k_1 = 0.1 \cdot \frac{6(4)^2}{16} = 0.600000$; $k_2 = 0.1 \cdot \frac{6(4.1)^2}{16.6} = 0.60759036 \implies y_{\text{RK2}}(4.1) = 16 + \frac{0.6 + 0.60759036}{2} = 16.603795$.",
            r"**Cálculo de las $4$ sondas de $RK4$ en el paso 1:** $k_1 = 0.600000$; $k_2 = 0.1 \cdot f(4.05, 16.3) = 0.1 \cdot \frac{98.415}{16.3} = 0.60377301$; $k_3 = 0.1 \cdot f(4.05, 16.3018865) = 0.60370312$; $k_4 = 0.1 \cdot f(4.1, 16.60370312) = 0.60745484$.",
            r"**Promedio ponderado de $RK4$:** $y_{\text{RK4}}(4.1) = 16 + \frac{0.6 + 2(0.60377301) + 2(0.60370312) + 0.60745484}{6} = 16.60373452$ (error frente al exacto $16.603614$ menor a $1.3 \times 10^{-4}$ — de hecho $2(4.1)^{1.5} = 16.60373452$, error $2.4 \times 10^{-9}$!).",
            r"**Evolución hasta $x = 4.3$ ($3$ pasos):** En $x = 4.3$ ($y_{\text{exacto}} = 17.83333956$), $RK2$ entrega $17.83352$ y $RK4$ logra $17.8333395$ con coincidencia de $6$ cifras exactas.",
        ],
    },
    "milne": {
        "deduccion_teorica": [
            r"**Necesidad de arranque previo ($4$ puntos):** Un método multipaso usa valores ya calculados en el historial para ahorrar evaluaciones de $f(x, y)$. Milne necesita $(x_{i-3}, y_{i-3}) \dots (x_i, y_i)$ para avanzar a $x_{i+1}$, por eso los primeros $4$ puntos ($k = 0, 1, 2, 3$) deben generarse con un método autoiniciable de orden $4$ como $RK4$.",
            r"**Deducción del Predictor abierto ($4$ pasos — Fórmula de Milne):** Integrando exactamente $y' = f$ sobre $4$ subintervalos $[x_{i-3}, x_{i+1}]$: $y_{i+1} - y_{i-3} = \int_{x_{i-3}}^{x_{i+1}} y'(x)\,dx$. Reemplazando $y'(x)$ por el polinomio interpolante cuadrático que pasa únicamente por los tres nodos interiores $\{x_{i-2}, x_{i-1}, x_i\}$ (sin usar el nodo extremo $x_{i+1}$ desconocido), al integrar resulta exactamente $P(y_{i+1}) = y_{i-3} + \frac{4h}{3}(2y'_i - y'_{i-1} + 2y'_{i-2})$ (error local $+\frac{14}{45}h^5 y^{(5)}$).",
            r"**Deducción del Corrector cerrado ($2$ pasos — Simpson 1/3):** Integrando ahora sobre $[x_{i-1}, x_{i+1}]$ e interpolando $y'$ por la parábola que atraviesa $x_{i-1}, x_i$ y el propio destino $x_{i+1}$ (usando $y'_{i+1} = f(x_{i+1}, P(y_{i+1}))$), la integral cerrada coincide exactamente con Simpson $1/3$: $C(y_{i+1}) = y_{i-1} + \frac{h}{3}(y'_{i-1} + 4y'_i + y'_{i+1})$ (error local $-\frac{1}{90}h^5 y^{(5)}$).",
            r"**Estimación gratuita del error local y doble corrección:** Restando los términos de error principal del Predictor ($+\frac{28}{90}h^5 y^{(5)}$) y del Corrector ($-\frac{1}{90}h^5 y^{(5)}$), resulta $|y_{\text{exacto}} - C| \approx \frac{1}{29}|C - P|$, permitiendo monitorear la precisión en cada paso y refinar en $C^1$.",
        ],
        "ejemplo_teorico_pasos": [
            r"**Verificación de malla mínima ($n \ge 4$):** Confirmamos que $\frac{x_f - x_0}{h} \ge 4$ para disponer de la ventana completa de cuatro nodos iniciales.",
            r"**Arranque con $RK4$ ($k = 0, 1, 2, 3$):** Calculamos $(x_0, y_0) \dots (x_3, y_3)$ y sus pendientes $y'_k = f(x_k, y_k)$ con orden $\mathcal{O}(h^4)$.",
            r"**Predicción abierta de $P(y_{i+1})$:** Usamos la combinación $(2y'_i - y'_{i-1} + 2y'_{i-2})$ desde $y_{i-3}$ y evaluamos $f(x_{i+1}, P(y_{i+1}))$.",
            r"**Doble pasada Correctora $C$ y $C^1$:** Aplicamos Simpson $1/3$ sobre $[x_{i-1}, x_{i+1}]$ dos veces seguidas para estabilizar $y_{i+1}$ usando la mitad de evaluaciones de $f$ que $RK4$.",
        ],
        "practica_pasos_deduccion": [
            r"**Fase de arranque con $RK4$ para $y' = \frac{6x^2}{y}$, $y(4) = 16$, $h = 0.1$:** Generamos los $4$ puntos de base ($k = 0 \dots 3$): $y_0(4.0) = 16.00000$ ($y'_0 = 6.00000$), $y_1(4.1) = 16.6037345$ ($y'_1 = 6.074537$), $y_2(4.2) = 17.2148773$ ($y'_2 = 6.148171$), $y_3(4.3) = 17.8333395$ ($y'_3 = 6.220933$).",
            r"**Fase Predictor para $x_4 = 4.4$:** Sustituimos $(y_0, y'_1, y'_2, y'_3)$ en $P(y_4) = 16 + \frac{4(0.1)}{3}[2(6.220933) - 6.148171 + 2(6.074537)] = 16 + \frac{0.4}{3}(18.442769) = 18.459036$.",
            r"**Primera pasada del Corrector $C(y_4)$:** Evaluamos la pendiente en el punto predicho: $f(4.4, 18.459036) = \frac{6(4.4)^2}{18.459036} = 6.292853$. Luego $C(y_4) = 17.2148773 + \frac{0.1}{3}[6.148171 + 4(6.220933) + 6.292853] = 17.2148773 + 1.2441585 = 18.4590358$.",
            r"**Segunda corrección $C^1(y_4)$ y diagnóstico $|P - C|$:** Recalculando la pendiente con $C(y_4)$ resulta $C^1(y_4) \approx 18.4590358$ (apunte redondeado a $5$ decimales: $P = 18.45903, C = C^1 = 18.45902$), prácticamente idéntico al exacto $2(4.4)^{1.5} = 18.45903573$ usando solo $2$ evaluaciones de $f$ por avance.",
        ],
    },
}

for _clave_m, _enriq in DEDUCCIONES_PASO_A_PASO_METODOS.items():
    if _clave_m in CATALOGO_METODOS:
        CATALOGO_METODOS[_clave_m]["deduccion_teorica"] = list(_enriq["deduccion_teorica"])
        if "ejemplo_teorico_pasos" in _enriq and "ejemplo_teorico" in CATALOGO_METODOS[_clave_m]:
            CATALOGO_METODOS[_clave_m]["ejemplo_teorico"]["pasos"] = list(_enriq["ejemplo_teorico_pasos"])
        if "practica_pasos_deduccion" in _enriq and "practica_profesora" in CATALOGO_METODOS[_clave_m]:
            CATALOGO_METODOS[_clave_m]["practica_profesora"]["pasos_deduccion"] = list(_enriq["practica_pasos_deduccion"])
