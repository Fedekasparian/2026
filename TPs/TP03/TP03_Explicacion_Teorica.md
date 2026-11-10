# TP03 — Árboles de decisión: explicación teórica

**Bases de Datos Masivas (11088)** — UNLu — 2° Cuatrimestre 2026
**Alumno:** Federico Kasparian

Este documento acompaña al notebook `TP03_Arboles_Respuesta.ipynb`. Para cada consigna explica **qué** se hizo,
**por qué** y **qué conceptos teóricos** hay detrás. Los números que aparecen salen de la ejecución del notebook
(`random_state = 42`).

---

## 0. ¿Qué es un árbol de decisión?

Un árbol de decisión es un clasificador que **divide recursivamente el espacio de atributos** en regiones cada
vez más homogéneas respecto de la clase:

- cada **nodo interno** es una pregunta sobre un atributo (`ASISTENCIA = media?`, `IDH1 <= 0.5?`);
- cada **rama** es una respuesta posible;
- cada **hoja** asigna una clase (la mayoritaria de los registros que llegan a ella).

Para clasificar un registro nuevo se recorre el árbol desde la raíz hasta una hoja. Su gran ventaja es la
**interpretabilidad**: cada camino raíz→hoja es una regla del tipo *SI … Y … ENTONCES clase*.

El algoritmo de construcción es **voraz** (*greedy*) y *top-down* (Hunt / ID3 / C4.5 / CART):

1. Si todos los registros del nodo son de la misma clase (nodo **puro**) o se cumple algún criterio de parada → hoja.
2. Si no, se evalúan todos los atributos (y puntos de corte) candidatos con una **medida de impureza** y se elige el
   que deja hijos más puros.
3. Se particiona el conjunto y se repite en cada hijo.

Lo que distingue a los algoritmos es la medida de impureza, el tipo de división y la forma de podar.

| Algoritmo | Medida | Divisiones | Atributos | Poda |
|---|---|---|---|---|
| ID3 (Quinlan, 1986) | Ganancia de información (entropía) | Multivía (una rama por valor) | Categóricos | No |
| C4.5 | *Gain ratio* | Multivía + umbrales numéricos | Ambos, maneja faltantes | Post-poda por error |
| CART (Breiman et al., 1984) — **Scikit-Learn** | Gini (o entropía) | **Binarias** | Numéricos | Costo-complejidad (`ccp_alpha`) |

---

## 1. Entropía, ganancia de información e ID3 (`parcial.csv`)

### 1.1 Entropía

La entropía de Shannon mide la **incertidumbre** (o impureza) de la distribución de clases de un conjunto $S$:

$$H(S) = -\sum_{i=1}^{k} p_i \log_2 p_i$$

donde $p_i$ es la proporción de registros de la clase $i$. Se mide en **bits**:

- $H = 0$: el conjunto es puro (todos de una clase), no hay incertidumbre.
- $H = 1$ (con 2 clases): mitad y mitad, la máxima incertidumbre.
- Por convención $0 \cdot \log_2 0 = 0$.

Interpretación: es la cantidad promedio de bits necesarios para codificar la clase de un registro elegido al azar.

### 1.2 Ganancia de información

Al particionar $S$ según un atributo $A$ con valores $v$, la incertidumbre que queda es el **promedio ponderado** de
la entropía de cada partición (entropía condicional):

$$H(S \mid A) = \sum_{v \in \text{valores}(A)} \frac{|S_v|}{|S|}\, H(S_v)$$

La **ganancia de información** es cuánto se reduce la entropía al conocer $A$:

$$G(S, A) = H(S) - H(S \mid A)$$

ID3 elige en cada nodo el atributo de **mayor ganancia**. Las funciones implementadas son justamente estas
tres fórmulas (`entropia`, `entropia_condicional`, `ganancia_informacion`).

> **Sesgo de la ganancia.** Favorece a los atributos con muchos valores (en el extremo, un ID único tiene ganancia
> máxima porque deja todas las particiones puras). C4.5 lo corrige con el *gain ratio*
> $GR(A) = G(A) / \text{SplitInfo}(A)$, con $\text{SplitInfo}(A) = -\sum_v \frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}$.
> Aquí no afecta porque todos los atributos tienen 2 o 3 valores.

### 1.3 Cálculo paso a paso

**Nodo raíz** — 14 registros: 8 *si*, 6 *no*.

$$H(S) = -\tfrac{8}{14}\log_2\tfrac{8}{14} - \tfrac{6}{14}\log_2\tfrac{6}{14} = 0{,}4613 + 0{,}5239 = \mathbf{0{,}9852}$$

| Atributo | Valor | n | si / no | $H(S_v)$ |
|---|---|---|---|---|
| ASISTENCIA | alta | 4 | 4 / 0 | 0 |
| | baja | 4 | 0 / 4 | 0 |
| | media | 6 | 4 / 2 | 0,9183 |
| TPS | completos | 7 | 5 / 2 | 0,8631 |
| | parciales | 7 | 3 / 4 | 0,9852 |
| TRABAJA | no | 6 | 4 / 2 | 0,9183 |
| | si | 8 | 4 / 4 | 1,0000 |
| MODALIDAD | presencial | 7 | 4 / 3 | 0,9852 |
| | virtual | 7 | 4 / 3 | 0,9852 |

- $G(\text{ASISTENCIA}) = 0{,}9852 - \tfrac{6}{14}\cdot 0{,}9183 = 0{,}9852 - 0{,}3936 = \mathbf{0{,}5917}$
- $G(\text{TPS}) = 0{,}9852 - (\tfrac{7}{14}\cdot 0{,}8631 + \tfrac{7}{14}\cdot 0{,}9852) = 0{,}9852 - 0{,}9242 = 0{,}0611$
- $G(\text{TRABAJA}) = 0{,}9852 - (\tfrac{6}{14}\cdot 0{,}9183 + \tfrac{8}{14}\cdot 1) = 0{,}9852 - 0{,}9650 = 0{,}0202$
- $G(\text{MODALIDAD}) = 0{,}9852 - 0{,}9852 = 0$ → la modalidad no dice nada sobre la aprobación
  (en ambas modalidades la proporción es la misma, 4/3).

→ Se divide por **ASISTENCIA**. Las ramas *alta* y *baja* son puras y se convierten en hojas.

**Nodo ASISTENCIA = media** — 6 registros: 4 *si*, 2 *no*, $H = 0{,}9183$.

| Atributo | Particiones | $H(S\mid A)$ | Ganancia |
|---|---|---|---|
| TPS | completos 3/0 (H=0) · parciales 1/2 (H=0,9183) | $\tfrac36\cdot 0{,}9183 = 0{,}4591$ | **0,4591** |
| TRABAJA | no 2/0 (H=0) · si 2/2 (H=1) | $\tfrac46 = 0{,}6667$ | 0,2516 |
| MODALIDAD | presencial 2/1 · virtual 2/1 | 0,9183 | 0 |

→ Se divide por **TPS**. *completos* es puro (si).

**Nodo ASISTENCIA = media ∧ TPS = parciales** — 3 registros: 1 *si*, 2 *no*, $H = 0{,}9183$.

| Atributo | Particiones | Ganancia |
|---|---|---|
| TRABAJA | no 1/0 · si 0/2 (ambas puras) | **0,9183** (toda la entropía) |
| MODALIDAD | presencial 1/1 · virtual 0/1 | 0,2516 |

→ Se divide por **TRABAJA** y quedan dos hojas puras. El algoritmo termina.

### 1.4 Árbol resultante

```
ASISTENCIA
├── alta  → APRUEBA = si
├── baja  → APRUEBA = no
└── media → TPS
            ├── completos → si
            └── parciales → TRABAJA
                            ├── no → si
                            └── si → no
```

Reglas: *asistencia alta ⇒ aprueba*; *asistencia baja ⇒ no aprueba*; *asistencia media ⇒ aprueba si entregó
todos los TPs, o si entregó TPs parciales pero no trabaja*. Clasifica correctamente el 100 % de las filas.
Es esperable: ID3 sin poda sigue dividiendo hasta que las hojas son puras, lo cual siempre es posible si no hay
dos registros con iguales atributos y distinta clase. Con datos reales esto es **sobreajuste** (punto 2 y 3).

### 1.5 Comparación con Scikit-Learn

Scikit-Learn implementa **CART optimizado**, que sólo acepta atributos numéricos y hace **cortes binarios**
(`x_j <= umbral`). Por eso los categóricos se codifican con **one-hot** (una columna 0/1 por valor). Con
`criterion="entropy"` usa la misma medida, pero el árbol cambia de forma: en vez de una rama por valor pregunta
`ASISTENCIA_baja <= 0.5` (baja vs. resto), y un mismo atributo puede volver a aparecer más abajo
(`ASISTENCIA_alta`). Las reglas que codifica son equivalentes.

---

## 2. Gliomas (`gliomas.csv`)

### 2.a Tema

El dataset de UCI *Glioma Grading Clinical and Mutation Features* (Tasci et al., 2022) toma 839 pacientes del
proyecto **TCGA** (*The Cancer Genome Atlas*). Los **gliomas** son los tumores primarios más comunes del cerebro y
se gradúan en:

- **LGG** — *Lower-Grade Glioma* (grados II y III), de crecimiento más lento: 487 pacientes (58 %).
- **GBM** — *Glioblastoma Multiforme* (grado IV), el más agresivo y de peor pronóstico: 352 pacientes (42 %).

El objetivo es **predecir el grado** a partir de 3 variables clínicas (sexo, edad al diagnóstico, raza) y del
estado de mutación (0/1) de los **20 genes más mutados** en gliomas. La motivación médica es encontrar un
**subconjunto pequeño de genes** que alcance para graduar el tumor, lo que abarata la secuenciación molecular.
Desde la teoría es una **clasificación binaria supervisada**, con clases moderadamente balanceadas y sin faltantes.

Ya en el análisis descriptivo se ve la señal principal: el 78 % de los LGG tiene IDH1 mutado contra sólo el 7 % de los
GBM, y la edad media es 44 años en LGG contra 61 en GBM.

### 2.b Árbol y muestreo 80/20

**Preprocesamiento.** `sexo` y `raza` son **nominales** → *one-hot* (no tienen orden, codificarlas como 0,1,2,3
inventaría una distancia falsa). Los genes ya son binarios y la edad es numérica. Los árboles **no necesitan
normalizar**: los cortes dependen sólo del orden de los valores, no de su escala.

**Hold-out estratificado.** Se separa 80 % para entrenar (671) y 20 % para test (168) con `stratify=y`, así ambas
partes mantienen la proporción 58/42. Sin estratificar, en datasets chicos la proporción de clases del test puede
desviarse y sesgar la evaluación. El test simula datos "nunca vistos" y sólo se usa para la evaluación final.

**Métricas** (clase positiva = GBM):

| Métrica | Fórmula | Qué responde |
|---|---|---|
| Accuracy | (VP+VN)/total | ¿Qué fracción acierta? |
| Precision | VP/(VP+FP) | De los que predigo GBM, ¿cuántos lo son? |
| Recall (sensibilidad) | VP/(VP+FN) | De los GBM reales, ¿cuántos detecto? |
| F1 | 2·P·R/(P+R) | Balance entre P y R |
| Balanced accuracy | (recall₊ + recall₋)/2 | Accuracy que no se infla por la clase mayoritaria |
| AUC-ROC | Área bajo la curva TPR vs. FPR | ¿Qué tan bien **ordena** las probabilidades? |

En un problema médico el **recall de GBM** es crítico: un falso negativo (decir "bajo grado" a un glioblastoma)
es mucho más grave que un falso positivo.

**Resultado del árbol por defecto** (`gini`, sin límites): profundidad 23, 126 hojas,
**accuracy train = 100 %, test = 79,2 %**, recall GBM = 0,74, AUC = 0,785.

Esto es **sobreajuste** (*overfitting*): el árbol sigue dividiendo hasta que cada hoja es pura, y en los niveles
profundos las divisiones se basan en 2–3 pacientes, es decir, en **ruido** del conjunto de entrenamiento que no
se repite en datos nuevos. Teoría detrás — **dilema sesgo-varianza**:

- un árbol muy profundo tiene **bajo sesgo** (puede representar cualquier frontera) pero **alta varianza**
  (cambia mucho si cambian los datos de entrenamiento);
- un árbol muy chico tiene alto sesgo (subajuste) y baja varianza;
- el error de generalización es mínimo en un punto intermedio, que hay que buscar.

### 2.c Búsqueda de un árbol "mejor"

Para controlar la complejidad hay dos familias de técnicas:

**Pre-poda** (detener el crecimiento antes): `max_depth`, `min_samples_split`, `min_samples_leaf`,
`max_leaf_nodes`, `min_impurity_decrease`. Es barata, pero miope: puede frenar un corte que parece malo y que
habilitaría cortes muy buenos debajo (efecto horizonte).

**Post-poda** (dejar crecer el árbol completo y después recortarlo). Scikit-Learn implementa la **poda por
costo-complejidad** de CART: para cada subárbol $T$ se define

$$R_\alpha(T) = R(T) + \alpha\,|\tilde T|$$

donde $R(T)$ es la impureza total de las hojas y $|\tilde T|$ la cantidad de hojas. $\alpha$ (`ccp_alpha`) penaliza
cada hoja extra: con $\alpha = 0$ queda el árbol completo y al aumentar $\alpha$ se van eliminando las ramas cuya
reducción de impureza no compensa su costo. `cost_complexity_pruning_path` devuelve los valores de $\alpha$ en los
que cambia el árbol óptimo.

**Selección de hiperparámetros sin mirar el test.** Si eligiera el modelo por su resultado en test, el test
dejaría de ser independiente y la estimación sería **optimista** (fuga de información). Por eso todo el ajuste se hace
con **validación cruzada de 5 folds sobre el conjunto de entrenamiento** (`GridSearchCV` / `cross_validate`) y el
test se usa una sola vez al final.

**Qué se encontró:**

1. **Curva de complejidad.** La accuracy de validación (CV) es máxima con `max_depth = 2` (0,869) y 3 (0,866), y
   baja de forma sostenida hasta ≈ 0,80 con árboles profundos, mientras la de entrenamiento sube hasta 1,00. Es la
   curva típica de sesgo-varianza.
2. **Grid search** (`criterion` × `max_depth` × `min_samples_leaf` × `class_weight`, 140 combinaciones): 17
   empatan en 0,869 de accuracy CV y dan exactamente las mismas predicciones que un árbol de dos niveles. Es decir,
   **una vez acotada la profundidad, gini vs. entropía y los demás parámetros casi no influyen**. Gini y entropía
   suelen elegir los mismos cortes porque ambas son funciones cóncavas de $p$ con el mismo máximo en $p = 0{,}5$.
3. **Poda por costo-complejidad**: el mejor `ccp_alpha` (0,0092) deja un árbol de 3 hojas con la misma accuracy CV.

**Comparación en test:**

| Modelo | Prof. | Hojas | Acc. train | Acc. test | Recall GBM | F1 | AUC |
|---|---|---|---|---|---|---|---|
| Por defecto | 23 | 126 | 1,000 | 0,792 | 0,743 | 0,748 | 0,785 |
| `max_depth=3` | 3 | 8 | 0,872 | 0,869 | 0,943 | 0,857 | 0,909 |
| Grid search (`max_depth=2`) | 2 | 4 | 0,869 | **0,869** | **0,943** | 0,857 | 0,888 |
| `ccp_alpha=0,0092` | 2 | 3 | 0,869 | 0,869 | 0,943 | 0,857 | 0,879 |

El árbol elegido:

```
IDH1 = 0 (no mutado)
├── IDH2 = 0  → GBM   (261 GBM / 67 LGG en train)
└── IDH2 = 1  → LGG   (2 / 19)
IDH1 = 1 (mutado)
├── TP53 = 0 → LGG   (1 / 139)
└── TP53 = 1 → LGG   (18 / 164)
```

En la práctica es una sola regla: **si ni IDH1 ni IDH2 están mutados ⇒ GBM**. Coincide con la clasificación de
la OMS (2021), donde el estado de IDH es el criterio central: los glioblastomas son, por definición, *IDH wild-type*.

**Importancia de atributos.** `feature_importances_` es la **reducción total de impureza (MDI)** que aporta cada
atributo, ponderada por la cantidad de muestras y normalizada a 1. En el árbol elegido IDH1 = 0,89,
IDH2 = 0,11. En el árbol por defecto la **edad** tenía 0,23: el MDI está **sesgado hacia los atributos continuos**
porque ofrecen muchos umbrales posibles y el árbol los usa para "memorizar" pacientes en los niveles profundos.
Cuando se poda, la edad desaparece. Esto muestra que la importancia de un árbol sobreajustado puede engañar.

**Estabilidad.** Con 168 registros de test, un punto de accuracy son menos de 2 pacientes, así que una sola
partición puede dar una conclusión casual. Repitiendo el hold-out con 30 semillas:

| Modelo | Accuracy test (media ± desvío) | Recall GBM |
|---|---|---|
| Por defecto | 0,805 ± 0,021 | 0,758 ± 0,047 |
| Seleccionado | **0,867 ± 0,022** | **0,924 ± 0,035** |

La mejora es **sistemática** (≈ 6 puntos de accuracy y 17 de recall), no depende de la partición.

**Conclusión.** "Mejor" no significa más complejo: el árbol de 2 niveles generaliza mejor, detecta más
glioblastomas y es interpretable por un médico. El dataset tiene una estructura muy simple dominada por IDH1/IDH2.

---

## 3. Informalidad laboral — EPH (`asalariados_eph.csv`)

### 3.a Tema

13.938 **asalariados** (obreros/empleados, incluido el servicio doméstico) de la **Encuesta Permanente de Hogares**
(INDEC, 1er trimestre 2026, 32 aglomerados urbanos). El tema es la **informalidad laboral asalariada**: el INDEC
considera **no registrado (informal)** al asalariado al que **no le hacen el descuento jubilatorio** (pregunta
`PP07H`). Los atributos describen a la persona (región, sexo, edad, nivel educativo) y al puesto (sector,
rama, calificación, tamaño del establecimiento, lugar, antigüedad, temporalidad, horas e ingreso).

Las filas **no están expandidas por el ponderador** (`PONDERA`). La EPH es una muestra probabilística en la que cada
persona representa a un número distinto de habitantes, así que las proporciones del dataset describen la muestra,
no a la población. Para clasificar no es un problema, pero el 37,6 % de informalidad no debe leerse como la tasa
oficial.

### 3.b Variable objetivo y balance

Se modela **`informal`** (si/no): **clasificación binaria**.

| informal | n | % |
|---|---|---|
| no | 8.694 | 62,4 |
| si | 5.244 | 37,6 |

Ratio 1,66 : 1 → **desbalance moderado**. No es extremo (como fraude, 99/1), pero tiene consecuencias:

- El **clasificador trivial** que siempre dice "no" ya obtiene **62,4 % de accuracy**; ese es el piso con el que comparar.
- La accuracy sola es engañosa; se reportan **precision, recall y F1 de la clase *si*** (la de interés),
  **balanced accuracy** y **AUC**.
- La partición y los folds se hacen **estratificados**.
- Si hiciera falta, se puede compensar con `class_weight="balanced"` (pesa cada clase en proporción inversa a su
  frecuencia) o con re-muestreo (*over/undersampling*, SMOTE) **sólo en entrenamiento**.

El análisis descriptivo ya muestra fuertes diferencias: el servicio doméstico tiene ≈ 79 % de informalidad contra ≈ 33 %
del resto. En general la informalidad baja al subir el nivel educativo (≈ 62 % sin instrucción o con primario incompleto,
14 % con superior completo), aunque no de forma estrictamente monótona.

### Preprocesamiento: codificación y faltantes

Scikit-Learn necesita atributos numéricos. La codificación debe respetar el **tipo de escala**:

- **Ordinales** (tienen orden): `nivel_educativo` (0 = Sin instrucción … 6 = Superior completo) y `antiguedad`
  (0 = Menos de 1 mes … 5 = Más de 5 años) → **enteros en orden**. Así el árbol puede cortar "nivel ≥ secundario
  completo" en un único corte. `tamanio_establecimiento` ya viene en tramos ordenados 1–12.
- **Nominales** (sin orden): región, sexo, sector, rama, calificación, lugar de trabajo, empleo temporario →
  **one-hot**. Codificarlos como 1, 2, 3… impondría un orden que no existe.

**Faltantes.** Hay faltantes de dos tipos:

- **Estructurales**: el cuestionario no le hace al **servicio doméstico** las preguntas del bloque "ocupación
  principal" (tamaño, lugar, antigüedad, temporalidad). No son datos perdidos al azar (**MNAR**): el faltante
  **dice** algo (que es servicio doméstico, con alta informalidad). Por eso, en las nominales se agrega una categoría
  `_nan` en el one-hot.
- **No respuesta**: sobre todo el ingreso (18,6 %).

En los numéricos se dejan los `NaN` porque `DecisionTreeClassifier` (desde la versión 1.3) **los maneja de forma
nativa**: en cada corte evalúa mandar los faltantes a la rama izquierda y a la derecha y elige lo que más reduce la
impureza. Imputar con la media o la mediana sería peor, porque ocultaría la información del faltante.
Todas las transformaciones son **mapeos fijos** (no se calculan con los datos), así que aplicarlas antes del
split no produce **fuga de información**. Si se imputara con estadísticas como la media, habría que calcularlas
sólo con el entrenamiento (p. ej. dentro de un `Pipeline`).

### 3.c Árbol con entropía, hold-out 80/20

`criterion="entropy"` hace que CART use la misma ganancia de información del punto 1 para elegir los cortes.
Train = 11.150, test = 2.788, ambos con 37,6 % de informales.

| Prof. | Hojas | Acc. train | Acc. test | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|
| 40 | 1.528 | **1,000** | **0,813** | 0,754 | 0,746 | 0,750 | 0,800 |

**Evaluación.** El modelo supera al trivial (81 % vs. 62 %), pero:

- **sobreajusta**: 19 puntos de brecha train–test, 1.528 hojas (en promedio ≈ 7 personas por hoja);
- su **AUC es baja** (0,80) para su accuracy: al ser hojas casi siempre puras, sus "probabilidades" valen 0 o 1 y no
  sirven para ordenar los casos por riesgo. Los árboles podados tienen hojas con proporciones intermedias (p. ej.
  0,73) y por eso logran AUC mucho más altos.

### 3.d Hiperparámetros (hold-out)

Los cuatro hiperparámetros son formas de **pre-poda**:

| Parámetro | Qué controla | Efecto al hacerlo más restrictivo |
|---|---|---|
| `max_depth` | Longitud máxima de un camino raíz → hoja | Menos interacciones entre atributos; reglas más cortas |
| `min_samples_split` | Mínimo de muestras para **intentar** dividir un nodo | Los nodos chicos no se dividen |
| `min_samples_leaf` | Mínimo de muestras en **cada hijo** | Prohíbe cortes que aíslen pocos casos; suaviza las probabilidades |
| `max_leaf_nodes` | Cantidad máxima de hojas | El árbol crece *best-first* (primero los cortes que más reducen la impureza) hasta el límite |

`min_samples_leaf` suele ser más efectivo que `min_samples_split`: el split=50 todavía permite partir un nodo de
50 en 49 + 1, mientras que leaf=50 garantiza que **ninguna** hoja tenga menos de 50 casos.

| # | max_depth | min_split | min_leaf | max_leaves | Prof. | Hojas | Acc. train | Acc. test | F1 | AUC | Brecha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | None | 2 | 1 | None | 40 | 1528 | 1,000 | 0,813 | 0,750 | 0,800 | 0,187 |
| 2 | 3 | 2 | 1 | None | 3 | 8 | 0,832 | 0,828 | 0,769 | 0,875 | 0,004 |
| 3 | 6 | 2 | 1 | None | 6 | 55 | 0,852 | 0,844 | 0,781 | 0,909 | 0,008 |
| 4 | 10 | 2 | 1 | None | 10 | 358 | 0,899 | 0,845 | 0,792 | 0,891 | 0,053 |
| 5 | 15 | 2 | 1 | None | 15 | 963 | 0,954 | 0,829 | 0,767 | 0,841 | 0,126 |
| 6 | None | 50 | 1 | None | 27 | 361 | 0,891 | 0,832 | 0,772 | 0,901 | 0,059 |
| 7 | None | 300 | 1 | None | 14 | 80 | 0,854 | 0,844 | 0,784 | 0,913 | 0,010 |
| 8 | None | 2 | 10 | None | 21 | 523 | 0,892 | 0,837 | 0,780 | 0,896 | 0,055 |
| 9 | None | 2 | 50 | None | 14 | 138 | 0,860 | 0,842 | 0,789 | 0,914 | 0,018 |
| 10 | None | 2 | 200 | None | 9 | 38 | 0,845 | 0,838 | 0,779 | 0,910 | 0,007 |
| 11 | None | 2 | 1 | 16 | 6 | 16 | 0,844 | 0,837 | 0,783 | 0,905 | 0,007 |
| 12 | None | 2 | 1 | 64 | 10 | 64 | 0,866 | **0,851** | **0,798** | 0,914 | 0,015 |
| 13 | 10 | 100 | 30 | None | 10 | 122 | 0,864 | 0,841 | 0,789 | 0,914 | 0,023 |
| 14 | 12 | 2 | 20 | 100 | 12 | 100 | 0,867 | 0,841 | 0,787 | 0,914 | 0,025 |
| 15 | 8 | 60 | 25 | 50 | 8 | 50 | 0,860 | 0,846 | 0,790 | 0,912 | 0,013 |

Lectura:

- **Brecha train–test** (indicador de sobreajuste): va de 0,187 (sin límites) a < 0,02 en los árboles regularizados.
- **Subajuste**: `max_depth=3` (#2) casi no tiene brecha, pero rinde menos (AUC 0,875): es demasiado simple.
- **Sobreajuste**: al subir `max_depth` de 10 a 15 la accuracy de test baja (0,845 → 0,829).
- Con entre ~40 y ~140 hojas los resultados son casi iguales: hay una meseta amplia de buenas configuraciones.

### 3.e Validación cruzada de 5 folds

**Hold-out vs. k-fold.** El hold-out da **una** estimación que depende de qué registros cayeron en el test (varianza
de la estimación). En **k-fold CV** el conjunto se divide en $k$ partes; se entrena $k$ veces con $k-1$ partes y se
valida con la restante; se promedian las $k$ métricas. Cada registro se usa para validar exactamente una vez, y además
se obtiene el **desvío**, que mide la incertidumbre. Con *stratified* k-fold cada fold mantiene la proporción 62/38.

Se aplicó CV sobre el 80 % de entrenamiento, para mantener el test independiente de la selección:

| # | Config. resumida | Acc. CV | F1 CV | AUC CV ± desvío | Acc. test (hold-out) |
|---|---|---|---|---|---|
| 1 | sin límites | 0,814 | 0,753 | 0,802 ± 0,007 | 0,813 |
| 2 | depth 3 | 0,833 | 0,774 | 0,875 ± 0,007 | 0,828 |
| 3 | depth 6 | 0,844 | 0,786 | 0,910 ± 0,006 | 0,844 |
| 4 | depth 10 | 0,844 | 0,791 | 0,889 ± 0,006 | 0,845 |
| 5 | depth 15 | 0,825 | 0,766 | 0,832 ± 0,004 | 0,829 |
| 6 | split 50 | 0,847 | 0,794 | 0,899 ± 0,004 | 0,832 |
| 7 | split 300 | 0,841 | 0,785 | 0,913 ± 0,008 | 0,844 |
| 8 | leaf 10 | 0,840 | 0,783 | 0,899 ± 0,006 | 0,837 |
| 9 | **leaf 50** | 0,849 | 0,798 | **0,916 ± 0,006** | 0,842 |
| 10 | leaf 200 | 0,840 | 0,785 | 0,910 ± 0,004 | 0,838 |
| 11 | leaves 16 | 0,839 | 0,789 | 0,908 ± 0,005 | 0,837 |
| 12 | **leaves 64** | 0,850 | **0,799** | 0,914 ± 0,005 | 0,851 |
| 13 | depth 10, split 100, leaf 30 | 0,849 | 0,796 | 0,916 ± 0,006 | 0,841 |
| 14 | depth 12, leaf 20, leaves 100 | 0,849 | 0,799 | 0,915 ± 0,008 | 0,841 |
| 15 | depth 8, split 60, leaf 25, leaves 50 | 0,850 | 0,798 | 0,915 ± 0,008 | 0,846 |

Observaciones:

- El **ranking es el mismo** que el del hold-out: los mismos extremos fallan y las mismas configuraciones ganan.
- Los **desvíos son chicos** (± 0,004–0,008 de AUC) porque el dataset es grande: con 11.150 registros, el hold-out ya era
  bastante confiable. En datasets chicos (como gliomas) la CV es mucho más necesaria.
- Las diferencias entre las mejores (#9, #12–#15) son menores que un desvío: son **estadísticamente equivalentes**.
  Con el criterio de **parsimonia** (navaja de Occam), entre modelos equivalentes conviene el más simple.
- La configuración #6 muestra una ventaja de la CV: en hold-out parecía peor (0,832) que en CV (0,847), así que esa
  partición en particular la perjudicaba.

**Modelo elegido** (mejor AUC CV): `min_samples_leaf=50`. Evaluado una sola vez en test: accuracy 0,842, F1 0,789,
AUC 0,914, valores casi idénticos a los de CV, lo que confirma que la estimación no fue optimista.

### 3.f Conclusiones

1. **Sobreajuste.** Un árbol sin restricciones memoriza el entrenamiento (100 %) y generaliza peor (81 %). Es el
   comportamiento esperado de un algoritmo voraz que divide hasta tener hojas puras.
2. **Regularización.** Cualquiera de los cuatro hiperparámetros mejora la generalización. Con árboles de 40–140 hojas
   (10 a 40 veces más chicos) se gana ≈ 3–4 puntos de accuracy y 0,11 de AUC. Los que limitan **el tamaño de las
   hojas** (`min_samples_leaf`) o **su cantidad** (`max_leaf_nodes`) funcionan mejor que limitar la profundidad,
   porque permiten ramas profundas donde hay muchos datos y cortas donde hay pocos.
3. **Hay que evitar los dos extremos**: muy poco (depth 3) subajusta; demasiado (depth 15) sobreajusta.
4. **Validación cruzada.** Da estimaciones más robustas, con su incertidumbre, y permite elegir sin tocar el test. Con
   un dataset de este tamaño confirma al hold-out, pero muestra que las mejores configuraciones están empatadas.
5. **Interpretación** (importancias del modelo elegido):
   - `empleo_temporario = no` (44 %): es el corte raíz. Separa a los **empleos permanentes** de los **temporarios**
     y del **servicio doméstico** (cuyo valor es faltante), los dos grupos con más informalidad (81 % y 84 %,
     contra 22 % en los permanentes).
   - `tamanio_establecimiento` (21 %): 74 % de informalidad en establecimientos de hasta 5 personas contra 14 % en
     los de más de 10.
   - `ingreso_mensual` (16 %): los ingresos bajos se asocian con no estar registrado.
   - `sector_Estatal` (8 %) y `antiguedad` (4 %): el empleo público casi siempre está registrado (10 % de
     informalidad contra 49 % en el privado), y lo mismo pasa con los empleos de mayor antigüedad.

   La **rama** casi no aparece aunque el servicio doméstico sea muy informal: su información ya la recogen el
   patrón de faltantes y el tamaño del establecimiento. Es el efecto de **atributos correlacionados**: el árbol usa
   uno y el otro pierde importancia, lo que **no** significa que no tenga relación con la clase.
6. **Desbalance.** Con `class_weight="balanced"` el recall de los informales sube de 0,79 a 0,85 y la precisión
   baja de 0,81 a 0,76, con F1 y AUC casi iguales. No hay un modelo "mejor" en abstracto: depende del costo
   de cada error. Si el objetivo fuera, por ejemplo, orientar inspecciones laborales, convendría priorizar el recall.
7. **Limitaciones.** Los datos no están ponderados, así que el modelo describe la muestra y no la población. Un solo
   árbol es inestable (pequeños cambios en los datos cambian los cortes superiores); los métodos de *ensemble*
   (Random Forest, Gradient Boosting) reducen esa varianza a costa de la interpretabilidad.

---

## Referencias

- Han, J., Kamber, M. & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3ª ed.), cap. 8.
- Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*, 1, 81–106.
- Breiman, L., Friedman, J., Olshen, R. & Stone, C. (1984). *Classification and Regression Trees*.
- Scikit-Learn — Decision Trees: https://scikit-learn.org/stable/modules/tree.html
- Tasci, E., Zhang, Y., Tasci, M. & Gonzalez, D. (2022). Glioma Grading Clinical and Mutation Features. UCI ML Repository.
- INDEC — EPH, base usuaria 1T 2026 y registro de diseño.
