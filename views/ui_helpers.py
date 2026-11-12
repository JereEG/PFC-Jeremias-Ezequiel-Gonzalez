"""
views/ui_helpers.py
===================
Utilidades visuales y componentes reutilizables para el Santo Grial de Métodos Numéricos.
- Visor embebido de documentos PDF en base64 con iframe nativo.
- Placeholder educativo cuando el PDF aún no ha sido cargado en assets/teoria/.
- Bloque estandarizado de 'Material de Estudio' (Teoría + Práctica).
"""

from __future__ import annotations

import base64
import os
import streamlit as st


def renderizar_pdf(ruta_relativa: str, titulo: str = "Documento PDF", altura: int = 500) -> None:
    """
    Incrusta un archivo PDF local mediante base64 dentro de un iframe HTML responsivo.
    Si el archivo no existe, despliega un placeholder informativo con opción de carga directa.
    """
    ruta_completa = os.path.abspath(ruta_relativa)

    if os.path.exists(ruta_completa) and os.path.isfile(ruta_completa):
        try:
            with open(ruta_completa, "rb") as f:
                pdf_bytes = f.read()

            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_display = f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; margin-top: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
                <iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=1&navpanes=0" 
                        width="100%" height="{altura}px" type="application/pdf" style="border: none;">
                </iframe>
            </div>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.download_button(
                label=f"📥 Descargar {titulo}",
                data=pdf_bytes,
                file_name=os.path.basename(ruta_relativa),
                mime="application/pdf",
                key=f"dl_{ruta_relativa}",
            )
        except Exception as e:
            st.warning(f"No se pudo renderizar el PDF: {e}")
    else:
        nombre_esperado = os.path.basename(ruta_relativa)
        st.markdown(
            f"""
            <div style="border: 2px dashed #b0bec5; border-radius: 8px; padding: 20px; text-align: center; background-color: #f8f9fa; margin: 10px 0;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📄</div>
                <div style="font-weight: 600; color: #37474f; font-size: 1.05rem;">{titulo}</div>
                <p style="color: #607d8b; font-size: 0.9rem; margin: 6px 0 12px 0;">
                    Archivo esperado: <code>assets/teoria/{nombre_esperado}</code>
                </p>
                <div style="font-size: 0.85rem; color: #78909c;">
                    Coloque el archivo PDF correspondiente en la carpeta <code>assets/teoria/</code> para visualizarlo automáticamente aquí.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def bloque_material_estudio(
    pdf_teoria: str,
    pdf_practica: str,
    titulo_metodo: str,
    abierto_por_defecto: bool = False,
) -> None:
    """
    Genera el bloque estándar de Material de Estudio para un método numérico,
    estructurado con 2 pestañas: Ejemplo Teórico y Ejercicio Resuelto en Clase.
    """
    with st.expander(f"📚 Material de Estudio y Notas de Cátedra — {titulo_metodo}", expanded=abierto_por_defecto):
        tab_teo, tab_prac = st.tabs([
            "📖 Fundamento Teórico y Deducción (PDF)",
            "📝 Ejercicio Resuelto en Clase (PDF)",
        ])

        with tab_teo:
            st.caption(f"Notas teóricas formales y demostración matemática de: **{titulo_metodo}**.")
            renderizar_pdf(
                os.path.join("assets", "teoria", pdf_teoria),
                titulo=f"Teoría: {titulo_metodo}",
            )

        with tab_prac:
            st.caption(f"Desarrollo paso a paso del ejercicio oficial de la cátedra de: **{titulo_metodo}**.")
            renderizar_pdf(
                os.path.join("assets", "teoria", pdf_practica),
                titulo=f"Práctica Resuelta: {titulo_metodo}",
            )
