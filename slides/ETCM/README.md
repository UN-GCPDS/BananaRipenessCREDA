# Presentación IEEE ETCM 2026 · UNAL & UPSE

Plantilla académica/institucional híbrida en LaTeX Beamer (16:9) para la conferencia **IEEE ETCM 2026** (Ecuador Technical Chapters Meeting), desarrollada en colaboración entre la **Universidad Nacional de Colombia (Sede Manizales)** y la **Universidad Estatal Península de Santa Elena (UPSE)**.

**Ponencia:** *De los datos al despliegue: clasificación robusta de la madurez del banano ante variaciones de iluminación con CREDA*  
**Autores:** Prof. Lucas Iturriago Salas, Ph.D. (UNAL) & Prof. Luis Chuquimarca Jiménez, Ph.D. (UPSE)

---

## 📁 Contenido del Paquete

* **`main.tex`**: Código fuente de la presentación (Beamer, formato panorámico `aspectratio=169`, 8 diapositivas).
* **`beamerthemeETCMHybrid.sty`**: Paquete modular del tema de presentación:
  * **Isotipo IEEE ETCM 2026 vectorial 100% TikZ** (`\etcmLogo`): modelado isométrico fiel con los tres rombos (amarillo, azul y rojo), barra vertical y tipografía apilada.
  * **Paleta cromática oficial**: Azul IEEE (`#0B4A75`), Celeste ETCM (`#008FD5`), Amarillo Lima (`#EEF341`), Azul Real (`#274198`), Rojo Isotipo (`#C31E0A`), Dorado (`#F9B91B`) y Azul UNAL (`#28669E`).
  * **Tarjetas y bloques modernos (`tcolorbox`)**: `modernblock` (azul), `successblock` (celeste), `alertmodernblock` (rojo) y `goldblock` (dorado).
  * **Encabezado institucional (`frametitle`)**: pestaña angular lateral con acentos cromáticos y compensación vertical para un centrado equilibrado del contenido.
  * **Pie de página estilizado (`footline`)**: línea quebrada angular con créditos institucionales (`Iturriago & Chuquimarca | UNAL & UPSE`), título corto y paginación.
  * **Plantilla de portada (`\makeETCMtitlepage`)**: diseño con doble escudo institucional (UNAL y UPSE), badge de la edición y zócalo inferior balanceado.
  * **Plantilla de cierre (`\makeETCMclosingpage`)**: diapositiva de agradecimiento con ambos escudos y datos de contacto de ambos investigadores.
* **Recursos gráficos**:
  * `EscudoUNAL.png`: Escudo oficial de la Universidad Nacional de Colombia.
  * `EscudoUPSE.png`: Escudo oficial de la Universidad Estatal Península de Santa Elena.
  * `Maduracion.png`: Figura explicativa de los estados de maduración y variabilidad de iluminación.
  * `Clasificacion.png`: Diagrama de la arquitectura del modelo CREDA, espacio latente (UMAP) y desempeño.
* **`main.pdf`**: Documento compilado de referencia listo para presentación.

---

## 📑 Estructura de las Diapositivas

1. **Portada Híbrida**: Título completo, subtítulo, autores (UNAL & UPSE), filiaciones y doble escudo institucional.
2. **Agenda de la Presentación**: Organización en bloques de los tres ejes temáticos de la ponencia.
3. **El Problema: Estimación de Maduración (Visual)**: Diagrama descriptivo de fruta climatérica y cambio de apariencia.
4. **El Problema: Desafíos y Pipeline**: Fenómeno biológico, desafíos en visión por computador, pipeline y objetivos.
5. **Evaluación Experimental: Metodología CREDA**: Diagrama arquitectural del modelo propuesto y adaptación de dominio.
6. **Evaluación Experimental: Resultados Cuantitativos**: Tabla comparativa con baselines (SVM, ResNet-18, MobileNetV3) y formulación de la función de pérdida con penalización de vecindad.
7. **Conclusiones y Proyección**: Aportes principales e impacto en agroindustria e inspección de calidad.
8. **Cierre y Contacto**: Mensaje de agradecimiento con logos institucionales, correos de contacto y dependencias.

---

## ⚙️ Instrucciones de Compilación

La plantilla utiliza coordenadas absolutas y relativas en TikZ (`remember picture, overlay`). Para que los elementos angulares y decorativos se sincronicen adecuadamente con el archivo `main.aux`, se requieren **dos pasadas consecutivas** del compilador:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### Paquetes LaTeX requeridos
* `tikz` (librerías: `calc`, `shapes.geometric`, `shadows`)
* `tcolorbox` (librerías: `skins`, `breakable`)
* `graphicx`
* `booktabs`
* `colortbl`
* `amsmath`, `amssymb`
* `babel` (con opción `spanish`)
