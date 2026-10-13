# Marco de trabajo y reglas obligatorias — Pregunta 2, Kaggle: licitaciones desiertas

**Versión:** 1.0  
**Aplicación:** `pregunta2.ipynb`, sus futuras revisiones y el archivo de predicciones.  
**Carácter:** normativa técnica del proyecto. Las reglas marcadas **OBLIGATORIO** son condiciones de aceptación, no sugerencias.

## 1. Objetivo y alcance

Desarrollar un clasificador binario que prediga si una licitación pública chilena terminará **desierta** (`y_desierta = 1`), utilizando exclusivamente información que estaba disponible **al momento de su publicación**. El problema corresponde a datos de Mercado Público/ChileCompra (estándar OCDS), con **152.553 licitaciones de 2024** etiquetadas para desarrollo y **21.829 licitaciones del primer trimestre de 2025** sin etiquetas para predicción.

- Entrada: `train.csv`, `test.csv` y `sample_submission.csv`.
- Salida: `submission_pregunta2.csv` con columnas `id` y `y_desierta`, en el orden exigido por la plantilla.
- Métrica objetivo de la competencia: **F1 para la clase positiva**, siempre que así lo confirme la pauta oficial. Reportar adicionalmente precisión, recall, Average Precision (AP), ROC AUC, matriz de confusión y, cuando corresponda, Brier Score.
- Prioridad absoluta: **validez temporal y ausencia de fuga de información antes que maximizar una métrica**.
- La validez de las características al publicar y la disponibilidad de las etiquetas para entrenar en una fecha histórica son **dos verificaciones diferentes**.

## 2. Jerarquía de evidencia y política de dudas

1. **Pauta oficial y diccionario de datos** para la definición del objetivo, momento de predicción y métrica.
2. **Esquema y valores reales de los CSV** para verificar columnas, tipos, nulos y fechas.
3. **Código ejecutado y resultados reproducibles** para fundamentar conclusiones.
4. Hipótesis de ingeniería de características, **siempre marcadas como tales**, nunca como hechos comprobados.

**OBLIGATORIO — Política de fallo seguro (fail closed):** si no es posible demostrar que una variable existía en el instante de predicción, **no usarla** como predictor. Si no se conoce cuándo se hizo público un desenlace, **no usar etiquetas anteriores para construir características históricas**. No inventar fechas, tasas, costos ni resultados. Documentar cualquier limitación no solucionable.

## 3. Contrato de datos y disponibilidad temporal

El notebook actual recibe estas columnas en `train.csv` (más `y_desierta`) y en `test.csv` (sin objetivo):

- Identificador y publicación: `id`, `pub_fecha`.
- Texto: `pub_titulo`, `pub_descripcion`.
- Presupuesto: `pub_monto_estimado`, `pub_moneda`.
- Procedimiento: `pub_metodo`, `pub_modalidad`.
- Estructura: `pub_n_items`, `pub_cantidad_total`, `pub_categoria`, `pub_categoria_id`, `pub_n_categorias`.
- Fechas/plazos: `pub_inicio`, `pub_cierre`, `pub_dias_postulacion`, `pub_tiene_consultas`, `pub_dias_consultas`.
- Comprador y localización: `comprador`, `comprador_id`, `comprador_rut`, `region`, `direccion`.

**OBLIGATORIO:** para cada columna candidata, comprobar que representa el valor *de publicación*, no una versión actualizada después del cierre o adjudicación. Los nombres con prefijo `pub_` **no bastan por sí solos** como prueba. En particular, verificar semántica y disponibilidad de plazos, consultas, montos, texto y categorías. Si no es demostrable, excluir y dejar constancia.

Prohibiciones:

- `y_desierta` **nunca** entra en `X`, ni directa ni indirectamente mediante tasas, codificaciones, promedios o texto de desenlace.
- El `y_desierta` incluido en `sample_submission.csv` es **solo un ejemplo de formato**; no es la verdad de 2025 ni una fuente de etiquetas, prevalencia o hiperparámetros.
- `id` se utiliza para alineación, auditoría y entrega, **no como predictor**. Verificar su unicidad en cada conjunto y que no hay desajustes al escribir las predicciones.
- No utilizar estados posteriores a publicación, adjudicaciones, número real de ofertas, cancelaciones, resultados definitivos ni campos derivados de ellos.
- No inferir que un dato retrospectivo del CSV era visible al publicar sin comprobación adicional.

## 4. Definición estricta de *data leakage*

Existe fuga de información si una característica, estadística, transformación o decisión utiliza información que **no habría estado disponible** al predecir, o si las etiquetas del período de evaluación influyen en la selección o ajuste del modelo que se evalúa.

**Reglas inviolables:**

1. **Disponibilidad causal:** para una licitación publicada en `t`, cada predictor debe poder calcularse con información conocida en `t` o antes, según la definición de publicación. Para una característica basada en otras licitaciones, usar registros con `pub_fecha < t`; **nunca** `<= t` si no se demuestra un orden real de disponibilidad.
2. **Sin etiquetas en ingeniería histórica:** por defecto, prohibir tasas, sumas, medias, proporciones, target encoding o suavizados que empleen `y_desierta`, aunque se implemente `cumsum() - y_actual`. La fecha de publicación de una licitación anterior **no demuestra** que su resultado ya se conocía.
3. **Sin prevalencia futura:** prohibir `global_rate = X[TARGET].mean()` para elaborar características por fila. Las tasas calculadas con etiquetas de todo un bloque también violan el requisito de causalidad para filas tempranas.
4. **Sin estadísticas futuras:** medianas, escaladores, vocabulario TF-IDF, One-Hot, selección de características y cualquier transformación con `fit` deben aprenderse **solo del conjunto de entrenamiento permitido en cada etapa**; nunca de validación, evaluación o test.
5. **Sin filtración en selección:** la partición reservada para evaluación final no puede elegir modelos, columnas, regularización, pesos, umbrales, calibración ni reglas post hoc.
6. **Sin dependencia del leaderboard:** el resultado público de Kaggle puede informarse, pero no debe usarse para reclamar que la evaluación temporal interna es independiente si se hicieron ajustes tras verlo.
7. **Sin uso del futuro dentro de un mismo instante:** licitaciones con igual timestamp no deben contarse unas a otras como anteriores. Usar comparación estricta de timestamps y pruebas de empate.
8. **Sin transformación supervisada antes de separar:** cualquier selección de columnas, ranking, imputación supervisada o balanceo basado en `y` debe ajustarse dentro de la etapa de entrenamiento correspondiente.

### Restricción especial sobre disponibilidad de las etiquetas

Una validación por `pub_fecha` **no demuestra por sí sola** una simulación operacional completa: el desenlace de una licitación publicada en julio pudo conocerse después del corte del 1 de agosto. Si la fuente **no incluye la fecha en que el resultado estuvo disponible**, no se puede certificar que todas las etiquetas del bloque de entrenamiento fueran conocidas al comienzo de la validación siguiente.

**Consecuencia obligatoria:** llamar al procedimiento «validación retrospectiva por fecha de publicación», con esta limitación explícita. No afirmar «100% libre de fugas de disponibilidad operacional» sin una auditoría de fechas de desenlace. Un *gap* temporal puede disminuir el riesgo, **pero no sustituye la demostración** de disponibilidad de etiquetas.

## 5. Ingeniería de características permitida

### 5.1 Características publicadas o derivadas de una sola fila

Permitidas **solo tras verificar su disponibilidad inicial**:

- Longitudes del título/descripción y conteo de palabras.
- Mes y día de la semana extraídos de `pub_fecha` (fecha parseada de manera explícita y consistente en UTC).
- Modalidad, método, moneda, región y categoría principal.
- Número de ítems, cantidades, número de categorías y plazos de postulación/consultas.
- Indicadores `monto_nulo` y `moneda_nula`.
- Texto del título y descripción publicado inicialmente, sin incorporar correcciones o comentarios posteriores.

**Montos:** no tratar CLP, USD, EUR y UF como una magnitud única sin conversión verificable a una moneda común usando la referencia temporal correcta. Si no hay conversión confiable, omitir el valor monetario bruto, conservar moneda/indicadores y documentar la pérdida potencial de señal. No asignar CLP a monedas desconocidas por defecto.

### 5.2 Historial permitido, con condiciones estrictas

Se permiten **conteos de publicaciones anteriores**, sin consultar resultados:

- `hist_comprador_n`: publicaciones previas del comprador.
- `hist_categoria_n`: publicaciones previas de la categoría.
- `hist_comprador_categoria_n`: publicaciones previas de la combinación.
- Indicadores `comprador_nuevo` y `categoria_nueva`, derivados de esos conteos.

Calcularlos exclusivamente con `pub_fecha_referencia < pub_fecha_fila`. Para entrenamiento, utilizar conteos acumulados con tratamiento explícito de timestamps idénticos; para validación/test, utilizar solo el bloque histórico anterior al instante de cada fila. Se permite `log1p(conteo)` porque no requiere conocer el desenlace.

**PROHIBIDO por defecto:** `hist_comprador_tasa`, `hist_categoria_tasa`, `hist_comprador_categoria_tasa`, sumas de `y_desierta`, y cualquier suavizado con tasa global. Solo podrían habilitarse después de acreditar una **fecha de disponibilidad del desenlace** para cada evento, una construcción *as-of* verificable y pruebas que garanticen que ninguna etiqueta futura ingresa en los resultados.

**Importante:** comparar un modelo sin historial con otro de **conteos seguros**, no con tasas de desenlaces de disponibilidad desconocida.

## 6. Particiones temporales y derechos de uso

Se conserva, salvo justificación documentada antes de ver resultados, la partición del notebook:

| Bloque | Fechas de publicación, UTC | Uso autorizado |
|---|---|---|
| A — Desarrollo | Antes de 2024-08-01 | Ajustar modelos candidatos y transformadores |
| B — Hiperparámetros | 2024-08-01 a 2024-09-30 | Seleccionar algoritmo, características e hiperparámetros según AP u otra métrica fijada de antemano |
| C — Umbral | 2024-10-01 a 2024-11-14 | Fijar umbral que optimice F1, **sin** reajustarlo luego con D |
| D — Evaluación final | Desde 2024-11-15 hasta fin de 2024 | Medir desempeño **una sola vez** de las configuraciones previamente cerradas |
| E — Kaggle | Primer trimestre de 2025 | Predicción exclusivamente; sin etiquetas conocidas |

**OBLIGATORIO:** no mezclar registros de estos períodos; comprobar exclusividad, cobertura, monotonía y ausencia de fechas inválidas. Los cuatro bloques A–D deben sumar **152.553 registros** si el CSV es el esperado y ninguna fecha es inválida. El bloque E debe contener **21.829 registros** si corresponde al archivo entregado.

**Secuencia:** (1) ajustar candidatos en A y compararlos en B; (2) fijar el modelo/configuración con B; (3) reentrenar en A+B y elegir umbral en C; (4) reentrenar en A+B+C y evaluar una sola vez en D; (5) congelar decisiones; (6) reentrenar para la submission con todo 2024; (7) predecir E con el umbral elegido previamente. En cada ajuste, volver a calcular desde cero los transformadores y conteos causales correspondientes.

No usar D para escoger entre dos alternativas después de observar su F1. Si se muestran resultados de varios modelos en D, sus configuraciones y reglas de reporte deben haber sido fijadas **antes** de consultarlo. Tras modificar el modelo a partir de D, **D deja de ser una evaluación final independiente**.

## 7. Modelos, optimización y experimentos

Mantener comparaciones controladas entre:

1. Regresión logística con características estructurales.
2. Regresión logística con características estructurales + TF-IDF.
3. Regresión logística con características estructurales + TF-IDF + **conteos históricos seguros**.
4. Árbol o bosque como *baselines* opcionales, sin atribuir a texto la diferencia entre familias de modelos.

Para medir el aporte del texto o los conteos, mantener iguales los restantes componentes y parámetros en las comparaciones de ablación. La selección de hiperparámetros de cada candidato solo puede usar B. La elección de umbral de cada candidato que se vaya a informar debe usar C. La métrica principal, criterios de desempate, grillas y regla final de selección deben definirse **antes de consultar D**.

- AP evalúa ordenamiento sin elegir umbral; F1 evalúa etiquetas con un umbral especificado. Explicar la relación entre ambas métricas.
- No suponer que `class_weight="balanced"` mejora necesariamente el F1.
- No reportar F1 de entrenamiento como estimación de generalización.
- Respetar una semilla explícita para reproducibilidad y registrar versiones de las bibliotecas cuando sea útil.
- Si se incorpora remuestreo (por ejemplo, SMOTE), aplicar **exclusivamente al conjunto de entrenamiento del ajuste**, nunca antes de la partición temporal.

## 8. Preprocesamiento sin fuga

**OBLIGATORIO:** encapsular `SimpleImputer`, `StandardScaler`, `OneHotEncoder` y `TfidfVectorizer` dentro de `Pipeline`/`ColumnTransformer` y ajustarlos únicamente mediante `fit(X_train, y_train)` del bloque permitido.

- Medianas y vocabulario de TF-IDF de A **no** deben conocer B, C, D o E en la etapa de selección inicial.
- En cada reentrenamiento se permite reajustar transformadores con la nueva ventana histórica autorizada, sin usar sus períodos futuros.
- Transformar B, C, D y E únicamente con `.transform()`/`.predict_proba()` de un pipeline ya ajustado al historial permitido.
- Las categorías desconocidas deben tratarse de forma estable (`handle_unknown="ignore"`), sin explorar etiquetas futuras para decidir cómo reagruparlas.
- TF-IDF debe usar solo texto cuya versión corresponde al momento de publicación.
- El conjunto E **no** se usa en `fit`, `fit_transform`, búsqueda de términos, selección de características ni estimación supervisada.

## 9. Estabilidad, calibración e interpretación

Evaluar cambios de distribución entre A, B, C, D y E en: porcentajes de nulos, moneda, monto sin mezclar unidades, modalidad, categoría, región, plazos y longitudes de texto. Estas comparaciones **no supervisadas** pueden incluir E para diagnóstico, pero **no autorizan ajustar el modelo o umbral con etiquetas de E**.

Registrar en B/C/D tasas reales y predichas, F1, precisión, recall, AP y matrices de confusión. En E registrar solo distribución de probabilidades y porcentaje de positivos **predichos**, nunca inventar tasa real de desiertas.

- Una tasa de positivos predichos alta respecto de 2024 **no demuestra por sí sola** descalibración; el umbral F1 puede ser bajo.
- En D reportar Brier Score y, si procede, curva de calibración **solo como evaluación descriptiva**. No calibrar usando D si se pretende conservarlo como evaluación independiente.
- Si se desea calibración para desplegar el modelo, reservar otro período anterior a D o emplear un protocolo temporal con predicciones fuera de muestra y sin contaminar la evaluación.
- No equiparar F1 con costos operativos ni afirmar que un tipo de error es más costoso sin cuantificar supuestos.

## 10. Pruebas automáticas obligatorias antes de aprobar cambios

Cada modificación relevante debe pasar estas verificaciones y dejar evidencia (assertions/tests o resultados visibles):

1. **Integridad:** `train` contiene `y_desierta` binaria; `test` no la contiene; IDs únicos; dimensiones y orden de submission correctos.
2. **Particiones:** ningún `id` aparece en dos bloques; no hay fechas inválidas; todos los cortes respetan el orden cronológico.
3. **Lista blanca:** las columnas usadas por el clasificador son explícitas y no incluyen `id`, `y_desierta`, campos de resultados posteriores ni tasas derivadas del objetivo.
4. **Invariancia a etiquetas futuras:** cambiar aleatoriamente etiquetas en una fecha posterior **no puede cambiar** las características de ninguna fila anterior. Lo recomendable es que ninguna característica dependa de `y_desierta`.
5. **Invariancia a filas futuras:** agregar o modificar publicaciones futuras no cambia las características de una fila pasada; tampoco cambia los conteos de filas con el mismo timestamp.
6. **Auditoría de conteos:** para filas seleccionadas aleatoriamente, comparar el conteo calculado con un conteo directo filtrando por `fecha_referencia < fecha_fila` y la llave correspondiente.
7. **Aislamiento de `fit`:** registrar/verificar qué índices y fechas recibió cada paso de ajuste de preprocesamiento, selección, entrenamiento y calibración. Ninguno puede recibir datos posteriores al bloque autorizado.
8. **Independencia de D:** demostrar que hiperparámetros, umbral y regla de selección se congelaron antes de evaluar D. No leer D para iterar después.
9. **Independencia de E:** editar los valores de ejemplo de `sample_submission` no puede modificar características ni predicciones; no usar `y_desierta` de la plantilla.
10. **Reproducibilidad:** ejecutar el notebook desde un kernel limpio en orden, sin depender de variables residuales; métricas, figuras y CSV deben corresponder al código actual.

**Si falla cualquier prueba crítica, detener la generación de la entrega, corregir y volver a probar.** No ocultar errores con `try/except`, rellenos arbitrarios o filtrados silenciosos.

## 11. Criterios finales de aceptación y entrega

Un cambio se considera aceptado solo si:

- [ ] Todas las características respetan su tiempo de disponibilidad, o fueron excluidas cuando no se pudo demostrar.
- [ ] No hay ninguna característica histórica basada en desenlaces sin fecha de disponibilidad comprobable.
- [ ] No hay dependencia de `global_rate` derivada de etiquetas futuras.
- [ ] Los conteos históricos están calculados con `pub_fecha_referencia < pub_fecha_fila`, incluidos los empates de timestamp.
- [ ] Las transformaciones y la búsqueda se ajustan solo en el bloque de entrenamiento autorizado.
- [ ] El modelo, parámetros, métrica de selección y umbral se fijan antes de consultar D.
- [ ] Se informa un baseline sin historial y la ablación del texto/conteos.
- [ ] Se documentan los cambios de distribución y se evalúa la calibración sin reajustar con D.
- [ ] La evidencia experimental coincide con el código ejecutado de principio a fin.
- [ ] `submission_pregunta2.csv` conserva los `id` de `test.csv` en el mismo orden, contiene 21.829 filas y `y_desierta ∈ {0,1}`.
- [ ] Se diferencia explícitamente entre **ausencia de fuga en características** y **limitaciones de simulación histórica por falta de fechas de disponibilidad de etiquetas**.

## 12. Instrucciones operativas para asistentes de programación

Al recibir una tarea sobre este proyecto:

1. Leer primero este archivo y el notebook completo. Respetar la pauta oficial si estuviera adjunta.
2. Identificar columnas, funciones, dependencias y períodos afectados **antes** de editar.
3. Aplicar el cambio mínimo que resuelva el problema, priorizando simplicidad, legibilidad y trazabilidad.
4. No introducir nuevas variables objetivo derivadas, *look-ahead*, target encoding, transformaciones globales, ajustes con D/E ni selección posterior a la evaluación final.
5. Reescribir comentarios Markdown para que expresen solo conclusiones justificadas por el código y sus salidas.
6. Ejecutar las pruebas de la sección 10. Si faltan los CSV o no es posible ejecutar, **no inventar** métricas ni declarar éxito; indicar exactamente qué quedó sin verificar.
7. Entregar un resumen breve: **qué se modificó, por qué, pruebas realizadas, métricas realmente obtenidas y limitaciones pendientes**.
8. Si una instrucción futura contradice las reglas de fuga temporal, **señalar el conflicto y proponer una alternativa segura**; no sacrificar validez por F1.

> **Principio rector:** ningún resultado de Kaggle ni mejora aparente de F1 compensa el uso de información que no existía al momento de predicción. Si la disponibilidad de un dato es incierta, se excluye del modelo y se documenta la limitación.
