# Respuestas a Ejercicios de Procesamiento de Lenguaje Natural

## Ejercicio 1: Word2Vec - Entrenamiento desde Cero

### Resultados Obtenidos:
Las palabras más similares a "good" fueron:
- bad (0.859)
- nice (0.830)
- funny (0.804)
- great (0.790)
- here (0.725)

### Respuestas a las Preguntas:

**1. ¿Qué significa que "bad" sea similar a "good" en el espacio vectorial?**
Esto es normal en Word2Vec porque el modelo aprende que estas palabras aparecen en contextos similares. Tanto "good" como "bad" se usan para describir películas, aparecen con palabras como "movie", "film", "acting", etc. El modelo no entiende que son opuestos, solo que se usan de manera similar.

**2. ¿Cómo interpreta Word2Vec el contexto de las palabras?**
Word2Vec mira las palabras que están cerca de cada palabra objetivo (ventana de contexto). Si dos palabras aparecen frecuentemente con las mismas palabras vecinas, el modelo las considera similares. Por ejemplo, "good movie" y "bad movie" tienen contextos parecidos.

**3. ¿Qué tan precisos fueron los embeddings generados?**
Los resultados son buenos para un corpus pequeño como movie_reviews. Las palabras "nice" y "great" siendo similares a "good" tiene sentido semántico. La presencia de "bad" muestra que el modelo está capturando patrones de uso más que significado.

**4. ¿Cómo optimizarías el modelo para obtener mejores representaciones?**
- Usar un corpus más grande y diverso
- Aumentar las dimensiones del vector (100, 200, 300)
- Ajustar el tamaño de la ventana de contexto
- Usar más épocas de entrenamiento
- Filtrar palabras muy raras o muy comunes

**5. ¿Qué ventajas tiene entrenar tu propio modelo vs. usar embeddings preentrenados?**
**Ventajas del modelo propio:**
- Se adapta específicamente a tu dominio
- Incluye vocabulario específico de tu área
- Control total sobre los parámetros

**Ventajas de modelos preentrenados:**
- Entrenados con millones de documentos
- Mejor calidad general
- Ahorro de tiempo y recursos computacionales

---

## Ejercicio 2: GloVe - Similitud Semántica

### Resultados Obtenidos:
- Similitud king-queen: 0.751
- Similitud cat-banana: 0.274

### Respuestas a las Preguntas:

**1. ¿Por qué la similitud king-queen es mayor que cat-banana?**
King y queen están relacionados semánticamente (ambos son roles de realeza, aparecen en contextos similares sobre monarquía, poder, etc.). Cat y banana son conceptos completamente diferentes (animal vs. fruta) que raramente aparecen juntos en textos.

**2. ¿Qué tipo de relaciones captura GloVe mejor?**
GloVe captura muy bien:
- Relaciones semánticas (sinónimos, categorías)
- Relaciones funcionales (king-queen, man-woman)
- Analogías (madrid-spain :: paris-france)
- Contextos de co-ocurrencia frecuente

**3. ¿Cómo se comparan estos resultados con Word2Vec?**
GloVe tiende a ser mejor para analogías y relaciones semánticas porque:
- Usa estadísticas globales del corpus completo
- Word2Vec solo usa ventanas locales
- GloVe preserva mejor las relaciones lineales entre vectores

**4. ¿Qué desafíos enfrentarías al usar embeddings en idiomas distintos al inglés?**
- Menos modelos preentrenados disponibles
- Corpus de entrenamiento más pequeños
- Diferencias en estructura gramatical
- Palabras con múltiples significados (polisemia)
- Necesidad de más preprocesamiento (acentos, etc.)

**5. ¿En qué escenarios preferirías GloVe sobre otros métodos?**
- Cuando necesitas analogías precisas
- Tareas que requieren relaciones semánticas fuertes
- Análisis de co-ocurrencias
- Cuando tienes suficientes recursos computacionales
- Aplicaciones que necesitan vectores estables

---

## Ejercicio 3: Embeddings Personalizados con Corpus Propio

### Resultados Obtenidos:
**Palabras similares a 'intelligence':**
- intelligent-based (0.832)
- networks (0.741)
- artificial (0.741)
- neural (0.738)

**Palabras similares a 'artificial':**
- networks (0.807)
- intelligence (0.741)
- augmented (0.744)

**Palabras similares a 'algorithm':**
- approach (0.906)
- function (0.897)
- tool (0.894)
- program (0.839)

### Respuestas a las Preguntas:

**1. ¿Qué tan bien capturó el modelo las relaciones semánticas de tu dominio específico?**
Muy bien. El modelo capturó correctamente que:
- "intelligence" se relaciona con "artificial" y "neural"
- "algorithm" se conecta con términos técnicos como "function", "program"
- Las relaciones son coherentes con el dominio tecnológico

**2. ¿Qué diferencias notas comparado con embeddings preentrenados generales?**
- Vocabulario más especializado en tecnología
- Relaciones más específicas del dominio
- Menos diversidad semántica general
- Mejor para tareas específicas de tecnología
- Peor para conceptos generales no técnicos

**3. ¿Cómo podrías mejorar la calidad de los embeddings personalizados?**
- Usar más artículos (actualmente solo 1000)
- Limpiar mejor el texto (eliminar HTML, símbolos raros)
- Ajustar parámetros (dimensiones, ventana)
- Balancear el corpus con diferentes temas tecnológicos
- Más épocas de entrenamiento

**4. ¿En qué casos sería mejor usar embeddings personalizados vs. preentrenados?**
**Embeddings personalizados cuando:**
- Dominio muy específico (medicina, legal, técnico)
- Vocabulario especializado no cubierto en modelos generales
- Necesitas control total sobre el proceso
- Tienes suficientes datos del dominio

**Embeddings preentrenados cuando:**
- Aplicación general
- Recursos limitados
- Necesitas arrancar rápido
- Tu dominio está bien cubierto por datos generales

**5. ¿Qué aplicaciones prácticas le darías a estos embeddings en tu campo?**
- Búsqueda semántica en documentación técnica
- Recomendación de artículos relacionados
- Clasificación automática de papers por tema
- Detección de plagio conceptual
- Análisis de tendencias tecnológicas
- Chatbots especializados en tecnología

---

## Ejercicio 4: Clasificación de Texto con BERT

### Resultados Obtenidos:
**ERROR:** El código falló debido a un parámetro obsoleto: `evaluation_strategy` debería ser `eval_strategy`.

### Análisis del Error:
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```

### Respuestas a las Preguntas (Basadas en el Conocimiento Teórico):

**1. ¿Qué ventajas ofrece BERT sobre enfoques tradicionales de clasificación?**
- Comprende contexto bidireccional (lee hacia adelante y atrás)
- Captura relaciones complejas entre palabras
- Preentrenado en enormes cantidades de texto
- Transfer learning efectivo con pocos datos
- Maneja mejor la ambigüedad y sarcasmo

**2. ¿Cómo mejora el fine-tuning el rendimiento del modelo?**
- Adapta los pesos preentrenados al problema específico
- Mantiene el conocimiento general y agrega conocimiento específico
- Converge más rápido que entrenar desde cero
- Necesita menos datos para buenos resultados
- Preserva representaciones ricas del lenguaje

**3. ¿Qué métricas de evaluación consideras más importantes y por qué?**
- **Accuracy:** Fácil de interpretar, buena para datasets balanceados
- **F1-score:** Mejor para datasets desbalanceados
- **Precision/Recall:** Importantes cuando los falsos positivos/negativos tienen costos diferentes
- **AUC-ROC:** Buena para evaluar discriminación del modelo

**4. ¿Cómo manejarías el desbalance de clases en este dataset?**
- Usar métricas balanceadas (F1, AUC)
- Aplicar pesos de clase en la función de pérdida
- Técnicas de sobremuestreo/submuestreo
- Usar `class_weight='balanced'` en el entrenamiento
- Validación estratificada

**5. ¿Qué optimizaciones aplicarías para mejorar el rendimiento?**
- Ajustar learning rate y scheduler
- Aumentar épocas con early stopping
- Usar diferentes capas de BERT para fine-tuning
- Aplicar data augmentation
- Ensemble de múltiples modelos
- Optimizar batch size según memoria disponible

---

## Ejercicio 9: Pipeline Integrado de NLP

### Resultados Obtenidos:
**ERROR:** El código falló porque `summarizer` no estaba definido correctamente.

### Análisis del Error:
```
NameError: name 'summarizer' is not defined
```

### Respuestas a las Preguntas (Basadas en el Diseño Esperado):

**1. ¿Qué tarea resultó más precisa: el resumen, la clasificación o el análisis de sentimiento?**
**Expectativa basada en modelos típicos:**
- **Análisis de sentimiento:** Generalmente más preciso (modelos simples, tarea binaria)
- **Clasificación temática:** Precisión media (depende de qué tan diferentes sean las categorías)
- **Resumen:** Más subjetivo, difícil de evaluar automáticamente

**2. ¿Qué tan bien se adaptaron los modelos preentrenados a tu texto personalizado?**
Los modelos preentrenados en inglés funcionan bien para:
- Texto formal y técnico
- Contenido similar a sus datos de entrenamiento
- Conceptos universales

Pueden tener problemas con:
- Jerga muy específica
- Texto en español (si usas modelos en inglés)
- Contexto cultural específico

**3. ¿Cómo integrarías este pipeline en una aplicación web real?**
- **API REST:** Flask/FastAPI para servir el pipeline
- **Microservicios:** Cada tarea como servicio independiente
- **Cola de tareas:** Celery/Redis para procesamiento asíncrono
- **Cache:** Redis para resultados frecuentes
- **Base de datos:** Para almacenar resultados y métricas
- **Monitoreo:** Logging y métricas de rendimiento

**4. ¿Qué parte del pipeline automatizarías o optimizarías con otra herramienta?**
- **Preprocesamiento:** spaCy para limpieza avanzada
- **Modelo único:** Un modelo multiTask que haga todo
- **Batch processing:** Para múltiples textos simultáneamente
- **GPU optimization:** Para acelerar inferencia
- **Model serving:** TensorFlow Serving o TorchServe

**5. ¿Qué mejoras podrías hacer si el texto estuviera en otro idioma o jerga regional?**
- Usar modelos multilingües (mBERT, xlm-roberta)
- Entrenar modelos específicos del idioma/región
- Traducir primero al inglés (si es aceptable la pérdida)
- Crear diccionarios de jerga específica
- Usar modelos locales preentrenados en ese idioma
- Ajustar preprocesamiento para características del idioma

---

## Conclusiones Generales

### Ejercicios Exitosos:
1. **Ejercicio 1:** Word2Vec funcionó correctamente, mostró relaciones esperadas
2. **Ejercicio 2:** GloVe demostró buena similitud semántica
3. **Ejercicio 3:** Embeddings personalizados capturaron bien el dominio técnico

### Ejercicios con Errores:
1. **Ejercicio 4:** Error de compatibilidad de versión en Transformers
2. **Ejercicio 9:** Error de definición de variables

### Lecciones Aprendidas:
- Los modelos preentrenados ofrecen buenos resultados rápidamente
- Los embeddings personalizados son valiosos para dominios específicos
- La compatibilidad de versiones es crucial en ML
- Los pipelines integrados requieren manejo cuidadoso de dependencias

### Recomendaciones:
1. Siempre verificar compatibilidad de versiones de librerías
2. Implementar manejo de errores robusto
3. Usar entornos virtuales específicos por proyecto
4. Documentar versiones exactas de dependencias
5. Hacer pruebas unitarias para cada componente del pipeline