# 🏦 Predictor de Riesgo Crediticio con Red Neuronal

Proyecto de machine learning que predice si un cliente bancario representa un riesgo crediticio alto o bajo, es decir, si pagará o no un crédito solicitado.

---

## ¿En qué consiste?

El sistema analiza las características financieras y personales de un cliente — como su edad, historial crediticio, nivel de ahorros, estabilidad laboral y el monto solicitado — y devuelve una probabilidad de pago junto con una decisión: **APROBAR** o **RECHAZAR**.

Por debajo, una red neuronal feedforward entrenada con datos de 1,000 clientes aprende los patrones que distinguen a un buen pagador de uno riesgoso. El modelo pasa por tres etapas:

1. **Preprocesamiento** — las variables categóricas se codifican numéricamente y las numéricas se normalizan para que la red pueda procesarlas correctamente.
2. **Entrenamiento** — la red ajusta sus pesos durante 60 épocas usando el optimizador Adam y una función de pérdida binaria, con Dropout y BatchNorm para evitar sobreajuste.
3. **Evaluación** — se mide el rendimiento con métricas estándar de la industria financiera: accuracy, AUC-ROC, matriz de confusión y curva ROC.

Además incluye un dashboard interactivo donde cualquier persona puede ingresar los datos de un cliente desde el navegador y obtener la predicción al instante, sin necesidad de escribir código.

---

## Tecnologías utilizadas

- **PyTorch** — construcción y entrenamiento de la red neuronal
- **Scikit-learn** — preprocesamiento de datos y métricas de evaluación
- **Pandas / NumPy** — manipulación y generación del dataset
- **Streamlit** — dashboard interactivo en el navegador
- **Matplotlib / Seaborn** — visualización de curvas de entrenamiento y resultados

---

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | ~68% |
| AUC-ROC | 0.722 |

Un AUC de 0.72 indica que el modelo distingue correctamente entre buenos y malos pagadores en el 72% de los casos, lo cual es considerado aceptable en modelos de scoring crediticio reales.
