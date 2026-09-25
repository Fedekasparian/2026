# TP02 — Apache Hop (paradigma visual/orquestado)

Mismo ETL que `../TP02_ETL_Respuesta.ipynb`, resuelto con [Apache Hop](https://hop.apache.org/)
en vez de pandas. Usa el mismo motor dockerizado que `dockers/hop/` (Hop Web 2.19.0), pero en un
contenedor propio para no interferir con el proyecto de clase.

## 1. Levantar

```bash
cd TPs/TP02/hop
docker compose up
```

Primera vez: baja la imagen (~2 GB). Cuando esté listo, abrir <http://localhost:8091> — el
proyecto `tp02` (esta carpeta) ya queda seleccionado por las variables de entorno del
`docker-compose.yml`.

## 2. Qué hay adentro

```
hop/
├── project-config.json
├── pipelines/
│   ├── 01-paises.hpl                      -> salida/paises.csv
│   ├── 02-preguntas.hpl                   -> salida/preguntas.csv
│   └── 03-educacion-primaria-2023.hpl     -> salida/educacion_primaria_2023.csv
├── workflows/
│   └── etl-tp02.hwf                       -> corre los tres pipelines en orden
└── salida/                                (se genera al ejecutar; se versiona como evidencia)
```

Los tres pipelines leen `../API_4_DS2_en_csv_v2_3434744.csv` (`${PROJECT_HOME}/../...`), el CSV
del Banco Mundial que ya está en el repo un nivel arriba — no se duplica.

## 3. Los pipelines

| Pipeline | Qué hace |
| --- | --- |
| `01-paises.hpl` | *CSV file input* (69 columnas) → *Select values* (deja 2) → *String operations* (trim) → *Filter rows* (nulos/vacíos, con salida de rechazos) → *Sort rows* + *Unique rows* (dedup) → *Add sequence* (`id`) → `paises.csv`. |
| `02-preguntas.hpl` | *Select values* (`indicador`) → *Replace in string* con RegEx ON (`\s*\([^)]*\)` → `""`, quita **todo** contenido entre paréntesis) → *String operations* (trim) → *Sort rows* + *Unique rows* (consolida las que quedan iguales) → *Add sequence* → `preguntas.csv`. |
| `03-educacion-primaria-2023.hpl` | *Filter rows* por indicador (`Primary education, duration (years)`) → *Filter rows* por dato 2023 no nulo (el rechazo, con los 11 países sin dato, queda en `salida/03-sin-dato-2023.csv` — no se descarta en silencio) → *Select values* (tipar a Integer) → *Sort rows* descendente → `educacion_primaria_2023.csv`. |

`workflows/etl-tp02.hwf` los encadena en secuencia (igual que `etl-alumnos.hwf` en el proyecto de
clase): si un pipeline falla, corta en vez de seguir con datos incompletos.

## 4. Correrlos sin abrir la GUI (`hop-run`)

Con el contenedor levantado (el proyecto `tp02` tiene que estar registrado una vez con
`hop-conf.sh -pc -p tp02 -ph /project`; `docker compose up` ya lo deja seleccionado por variables
de entorno, pero `hop-run` en un contenedor nuevo necesita que exista el registro):

```bash
docker compose exec hop-tp02 bash -c \
  "cd /usr/local/tomcat/webapps/ROOT && bash hop-run.sh -j tp02 -r local -l Basic \
   -f /project/workflows/etl-tp02.hwf"
```

(o apuntando `-f` a un `.hpl` puntual para correr un solo pipeline).

## 5. Verificación

Los tres pipelines y el workflow se corrieron de punta a punta con `hop-run` (motor local, sin
abrir la GUI) contra el dataset real. Resultado, reconciliado fila a fila contra
`../TP02_ETL_Respuesta.ipynb`:

| Pipeline | Filas escritas | Rechazos |
| --- | --- | --- |
| `01-paises.hpl` | 266 (`paises.csv`) | 0 (`01-paises-rechazados.csv`, solo header) |
| `02-preguntas.hpl` | 151 (`preguntas.csv`) | — |
| `03-educacion-primaria-2023.hpl` | 255 (`educacion_primaria_2023.csv`) | 11 (`03-sin-dato-2023.csv`) |

Los CSV de `salida/` en este repo son el resultado real de esa corrida (no se regeneraron a
mano), y quedan versionados como evidencia de que el proyecto de Hop es funcional, no solo un
diseño visual sin ejecutar.
