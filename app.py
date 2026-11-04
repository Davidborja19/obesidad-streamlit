"""
Dashboard en Streamlit - Estimación de niveles de obesidad
Mismo preprocesamiento y mismos 4 modelos del notebook estimacion_niveles_obesidad.ipynb
(Random Forest, Árbol de decisión, SVM y Regresión logística).

Ejecutar con:   streamlit run app.py
Tema oscuro:    ver .streamlit/config.toml
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Niveles de obesidad", layout="wide")

# ----------------------------------------------------------------------------
# 1. CONSTANTES (las mismas del notebook)
# ----------------------------------------------------------------------------
ORDEN_TARGET = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
                'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
NOMBRES_ES = {'Insufficient_Weight': 'Peso insuficiente', 'Normal_Weight': 'Peso normal',
              'Overweight_Level_I': 'Sobrepeso nivel I', 'Overweight_Level_II': 'Sobrepeso nivel II',
              'Obesity_Type_I': 'Obesidad tipo I', 'Obesity_Type_II': 'Obesidad tipo II',
              'Obesity_Type_III': 'Obesidad tipo III'}
ETIQUETAS = [NOMBRES_ES[c] for c in ORDEN_TARGET]          # etiquetas en español, en orden 0..6
# Un color por nivel, de más frío (bajo peso) a más cálido/oscuro (obesidad III)
COLORES = ['#6F9BD1', '#5FB38A', '#D6C25A', '#E2A23B', '#DE7A3C', '#D04E40', '#A62F3F']
NUM_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
BINARIAS = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
FRECUENCIA = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
RUTA_CSV = "obesity.csv"   # opcional: si existe junto a app.py se usa en vez de descargar de UCI

CONJUNTOS = {
    'completo': 'Completo (con peso y estatura)',
    'habitos': 'Solo hábitos (sin peso ni estatura)',
}

MODELOS = {
    'Random Forest':       lambda: RandomForestClassifier(n_estimators=400, random_state=42),
    'Árbol de decisión':   lambda: DecisionTreeClassifier(random_state=42),
    'SVM (RBF)':           lambda: SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'Regresión logística': lambda: LogisticRegression(max_iter=1000, random_state=42),
}

# ----------------------------------------------------------------------------
# 2. ESTILO (CSS + plantilla de gráficos)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stSidebar"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 500 !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1280px; }
footer { visibility: hidden; }

.eyebrow { font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; color: #4FB3A5; font-weight: 600; }
.titulo { font-family: 'Fraunces', Georgia, serif; font-size: 3.1rem; line-height: 1.05; font-weight: 500;
          letter-spacing: -0.02em; margin: .25rem 0 .5rem 0; color: #F0ECE4; }
.subtitulo { color: #8B909C; font-size: 1rem; max-width: 46rem; margin-bottom: 1.6rem; }

.rotulo { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: #8B909C;
          font-weight: 600; margin: 0 0 .5rem 0; padding-bottom: .4rem; border-bottom: 1px solid #262B36; }

.hero { background: #161A22; border: 1px solid #262B36; border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: .8rem; }
.hero .k { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: #8B909C; font-weight: 600; }
.hero .nivel { font-family: 'Fraunces', Georgia, serif; font-size: 2.7rem; line-height: 1.1; margin-top: .25rem; }

.stat { background: #161A22; border: 1px solid #262B36; border-radius: 12px; padding: .85rem 1.1rem; height: 100%; }
.stat .k { font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: #8B909C; font-weight: 600; }
.stat .v { font-family: 'Fraunces', Georgia, serif; font-size: 1.85rem; line-height: 1.15; margin-top: .15rem; }
.stat .s { font-size: .8rem; color: #8B909C; }

[data-testid="stMetric"] { background: #161A22; border: 1px solid #262B36; border-radius: 12px; padding: .85rem 1.1rem; }
[data-testid="stMetricLabel"] p { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: #8B909C; }
[data-testid="stMetricValue"] { font-family: 'Fraunces', Georgia, serif; }

[data-testid="stSidebar"] { border-right: 1px solid #262B36; }
[data-testid="stExpander"] { border: 1px solid #262B36; border-radius: 12px; background: #12151B; }
.nota { font-size: .82rem; color: #8B909C; border-left: 2px solid #4FB3A5; padding-left: .8rem; margin-top: 1rem; line-height: 1.5; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def estilo(fig, alto=None):
    """Aplica el aspecto oscuro común a todos los gráficos de Plotly."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#CFCBC2", size=13),
        margin=dict(l=10, r=10, t=30, b=10), height=alto,
    )
    fig.update_xaxes(gridcolor="#232832", zerolinecolor="#232832")
    fig.update_yaxes(gridcolor="#232832", zerolinecolor="#232832")
    return fig


def mostrar(fig):
    st.plotly_chart(fig, config={"displayModeBar": False})


def rotulo(texto):
    st.markdown(f'<div class="rotulo">{texto}</div>', unsafe_allow_html=True)


def tarjeta(titulo, valor, sub="", color="#F0ECE4"):
    return (f'<div class="stat"><div class="k">{titulo}</div>'
            f'<div class="v" style="color:{color}">{valor}</div><div class="s">{sub}</div></div>')


# ----------------------------------------------------------------------------
# 3. DATOS Y ENTRENAMIENTO (se ejecuta UNA sola vez gracias a la caché)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos():
    """Lee obesity.csv si existe; si no, descarga el dataset #544 de UCI (igual que el notebook)."""
    if os.path.exists(RUTA_CSV):
        return pd.read_csv(RUTA_CSV)
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=544)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def preparar(df):
    """Mismo preprocesamiento del notebook (Paso 4 y 5): duplicados, binarias, ordinales, One-Hot, target."""
    df = df.drop_duplicates().reset_index(drop=True)
    crudo = df.copy()                                           # copia legible para el análisis exploratorio
    for col in BINARIAS:
        df[col] = df[col].map({'no': 0, 'yes': 1})
    for col in ['CAEC', 'CALC']:
        df[col] = df[col].map(FRECUENCIA)
    df = pd.get_dummies(df, columns=['Gender', 'MTRANS'], drop_first=True, dtype=int)
    y = df['NObeyesdad'].map({c: i for i, c in enumerate(ORDEN_TARGET)})
    X = df.drop(columns='NObeyesdad')
    return X, y, crudo


def construir_pipeline(modelo, columnas):
    """Escala solo las numéricas continuas y luego aplica el modelo (igual que el notebook)."""
    num = [c for c in NUM_COLS if c in columnas]
    prep = ColumnTransformer([('escalado', StandardScaler(), num)], remainder='passthrough')
    return Pipeline([('prep', prep), ('modelo', modelo)])


@st.cache_resource(show_spinner="Entrenando los 4 modelos (solo la primera vez)...")
def entrenar():
    X_total, y, crudo = preparar(cargar_datos())
    cols = {'completo': list(X_total.columns),
            'habitos': [c for c in X_total.columns if c not in ('Weight', 'Height')]}

    # Misma partición del notebook: 80/20, estratificada, random_state=42
    idx_train, idx_test = train_test_split(X_total.index, test_size=0.2, stratify=y, random_state=42)
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    modelos, preds, filas = {}, {}, {}
    for conjunto, columnas in cols.items():
        filas[conjunto] = {}
        for nombre, fabrica in MODELOS.items():
            pipe = construir_pipeline(fabrica(), columnas).fit(X_total.loc[idx_train, columnas], y_train)
            pred = pipe.predict(X_total.loc[idx_test, columnas])
            modelos[(conjunto, nombre)] = pipe
            preds[(conjunto, nombre)] = pred
            filas[conjunto][nombre] = {
                'Accuracy': accuracy_score(y_test, pred),
                'Recall macro': recall_score(y_test, pred, average='macro'),
                'F1 macro': f1_score(y_test, pred, average='macro'),
            }
    tablas = {k: pd.DataFrame(v).T.round(3) for k, v in filas.items()}
    rangos = {c: (float(crudo[c].min()), float(crudo[c].max())) for c in ['Age', 'Height', 'Weight']}
    return {'modelos': modelos, 'preds': preds, 'tablas': tablas, 'cols': cols,
            'y_test': y_test, 'crudo': crudo, 'rangos': rangos}


# ----------------------------------------------------------------------------
# 4. FUNCIONES AUXILIARES
# ----------------------------------------------------------------------------
def clasificar_imc_oms(imc):
    """Categoría estándar de la OMS según el IMC (solo como referencia)."""
    if imc < 18.5:
        return "Bajo peso"
    if imc < 25:
        return "Peso normal"
    if imc < 30:
        return "Sobrepeso"
    if imc < 35:
        return "Obesidad tipo I"
    if imc < 40:
        return "Obesidad tipo II"
    return "Obesidad tipo III"


def construir_fila(d, columnas):
    """Convierte lo que escribió el usuario en UNA fila con las mismas columnas que usó el entrenamiento."""
    fila = {
        'Age': d['edad'], 'Height': d['estatura'], 'Weight': d['peso'],
        'family_history_with_overweight': int(d['antecedentes'] == 'Sí'),
        'FAVC': int(d['favc'] == 'Sí'), 'FCVC': d['fcvc'], 'NCP': d['ncp'],
        'CAEC': d['caec'], 'SMOKE': int(d['smoke'] == 'Sí'), 'CH2O': d['ch2o'],
        'SCC': int(d['scc'] == 'Sí'), 'FAF': d['faf'], 'TUE': d['tue'], 'CALC': d['calc'],
        'Gender_Male': int(d['genero'] == 'Masculino'),
        'MTRANS_Bike': int(d['mtrans'] == 'Bike'),
        'MTRANS_Motorbike': int(d['mtrans'] == 'Motorbike'),
        'MTRANS_Public_Transportation': int(d['mtrans'] == 'Public_Transportation'),
        'MTRANS_Walking': int(d['mtrans'] == 'Walking'),
    }
    return pd.DataFrame([fila])[columnas]


def grafico_imc(imc):
    """Medidor semicircular del IMC con los rangos de la OMS."""
    tramos = [(12, 18.5, COLORES[0]), (18.5, 25, COLORES[1]), (25, 30, COLORES[3]),
              (30, 35, COLORES[4]), (35, 40, COLORES[5]), (40, 50, COLORES[6])]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(imc, 1),
        number=dict(font=dict(family="Fraunces, Georgia, serif", size=46, color="#F0ECE4")),
        gauge=dict(
            axis=dict(range=[12, 50], tickvals=[18.5, 25, 30, 35, 40], tickcolor="#5B6170",
                      tickfont=dict(color="#8B909C", size=11)),
            bar=dict(color="#F0ECE4", thickness=0.2), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[a, b], color=c) for a, b, c in tramos]),
    ))
    return estilo(fig, 230)


def grafico_proba(clases, probs, clase_pred):
    """Barras horizontales con la probabilidad de cada nivel; el predicho resalta."""
    fig = go.Figure(go.Bar(
        x=probs * 100, y=[ETIQUETAS[int(c)] for c in clases], orientation='h',
        marker=dict(color=[COLORES[int(c)] for c in clases],
                    opacity=[1 if int(c) == clase_pred else 0.35 for c in clases]),
        text=[f"{p * 100:.1f}%" for p in probs], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(range=[0, 112], showticklabels=False, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return estilo(fig, 300)


# ----------------------------------------------------------------------------
# 5. INTERFAZ
# ----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Proyecto de Machine Learning · UCI #544</div>'
            '<div class="titulo">Estimación de niveles de obesidad</div>'
            '<div class="subtitulo">Clasificación multiclase en 7 niveles a partir de medidas corporales, '
            'hábitos alimentarios y condición física.</div>', unsafe_allow_html=True)

try:
    art = entrenar()
except Exception as e:  # sin internet o sin el CSV local
    st.error("No se pudieron cargar los datos. Verifica la conexión a internet o coloca el archivo "
             f"`{RUTA_CSV}` (CSV del dataset) junto a `app.py`.\n\nDetalle: {e}")
    st.stop()

# ----- Barra lateral -----
with st.sidebar:
    st.markdown('<div class="rotulo">Configuración</div>', unsafe_allow_html=True)
    nombre_modelo = st.selectbox("Modelo de predicción", list(MODELOS.keys()),
                                 help="Elige el algoritmo que hará la predicción.")
    conjunto = st.radio("Variables que usa el modelo", list(CONJUNTOS.keys()),
                        format_func=lambda k: CONJUNTOS[k],
                        help="'Completo' usa peso y estatura (equivale a calcular con el IMC). "
                             "'Solo hábitos' ignora peso y estatura.")
    st.markdown('<div class="nota">En el notebook se detectó que el nivel de obesidad se calculó a partir '
                'del IMC (peso/estatura²). Por eso el modelo <b>completo</b> acierta más (~95 %) pero casi '
                're-aprende esa fórmula. El modelo <b>solo hábitos</b> (~83 %) mide el efecto real de los '
                'hábitos.</div>', unsafe_allow_html=True)

tab_pred, tab_modelos, tab_datos = st.tabs(["Predicción", "Resultados de los modelos", "Exploración de datos"])

# ============================ TAB 1: PREDICCIÓN ============================
with tab_pred:
    izq, der = st.columns([5, 6], gap="large")

    # ---------- Columna izquierda: datos de entrada ----------
    with izq:
        rotulo("Datos personales")
        a, b = st.columns(2)
        genero = a.selectbox("Género", ["Femenino", "Masculino"])
        edad = b.number_input("Edad (años)", min_value=10, max_value=100, value=25, step=1)
        estatura = a.number_input("Estatura (m)", min_value=1.20, max_value=2.30, value=1.70, step=0.01, format="%.2f")
        peso = b.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5, format="%.1f")
        imc = peso / estatura ** 2

        with st.expander("Hábitos y condición física", expanded=True):
            h1, h2 = st.columns(2)
            antecedentes = h1.selectbox("Antecedentes familiares de sobrepeso", ["Sí", "No"])
            favc = h2.selectbox("Come alimentos muy calóricos con frecuencia", ["Sí", "No"])
            fcvc = h1.selectbox("Vegetales en las comidas", [1, 2, 3], index=1,
                                format_func=lambda v: {1: "Nunca", 2: "A veces", 3: "Siempre"}[v])
            ncp = h2.slider("Comidas principales al día", 1, 4, 3)
            caec = h1.selectbox("Come entre comidas", [0, 1, 2, 3], index=1,
                                format_func=lambda v: {0: "No", 1: "A veces", 2: "Frecuentemente", 3: "Siempre"}[v])
            ch2o = h2.selectbox("Agua diaria", [1, 2, 3], index=1,
                                format_func=lambda v: {1: "Menos de 1 L", 2: "Entre 1 y 2 L", 3: "Más de 2 L"}[v])
            smoke = h1.selectbox("Fuma", ["No", "Sí"])
            scc = h2.selectbox("Controla las calorías que consume", ["No", "Sí"])
            faf = h1.selectbox("Actividad física semanal", [0, 1, 2, 3], index=1,
                               format_func=lambda v: {0: "Ninguna", 1: "1-2 días", 2: "2-4 días", 3: "4-5 días"}[v])
            tue = h2.selectbox("Uso diario de dispositivos", [0, 1, 2],
                               format_func=lambda v: {0: "0-2 horas", 1: "3-5 horas", 2: "Más de 5 horas"}[v])
            calc = h1.selectbox("Consumo de alcohol", [0, 1, 2, 3],
                                format_func=lambda v: {0: "No", 1: "A veces", 2: "Frecuentemente", 3: "Siempre"}[v])
            mtrans = h2.selectbox("Transporte habitual",
                                  ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"],
                                  format_func=lambda v: {"Public_Transportation": "Transporte público",
                                                         "Automobile": "Automóvil", "Walking": "Caminando",
                                                         "Bike": "Bicicleta", "Motorbike": "Motocicleta"}[v])

        datos = dict(genero=genero, edad=edad, estatura=estatura, peso=peso, antecedentes=antecedentes, favc=favc,
                     fcvc=fcvc, ncp=ncp, caec=caec, smoke=smoke, ch2o=ch2o, scc=scc, faf=faf, tue=tue,
                     calc=calc, mtrans=mtrans)

        # Aviso si los valores están fuera de lo que el modelo vio al entrenar
        r = art['rangos']
        fuera = [n for n, v, k in [("Edad", edad, "Age"), ("Estatura", estatura, "Height"), ("Peso", peso, "Weight")]
                 if not r[k][0] <= v <= r[k][1]]
        if fuera:
            st.warning(f"{', '.join(fuera)} fuera del rango de los datos de entrenamiento: "
                       "la predicción puede ser poco fiable.")

    # ---------- Predicción ----------
    pipe = art['modelos'][(conjunto, nombre_modelo)]
    X_nuevo = construir_fila(datos, art['cols'][conjunto])
    clase = int(pipe.predict(X_nuevo)[0])
    proba = pipe.predict_proba(X_nuevo)[0]
    p_clase = float(proba[list(pipe.classes_).index(clase)])
    hay_obesidad = clase >= 4
    color = COLORES[clase]

    # ---------- Columna derecha: resultados ----------
    with der:
        rotulo(f"Resultado · {nombre_modelo}")
        st.markdown(f'<div class="hero" style="border-left:4px solid {color}">'
                    f'<div class="k">Nivel estimado</div>'
                    f'<div class="nivel" style="color:{color}">{ETIQUETAS[clase]}</div></div>',
                    unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        s1.markdown(tarjeta("IMC", f"{imc:.1f}", clasificar_imc_oms(imc)), unsafe_allow_html=True)
        s2.markdown(tarjeta("¿Obesidad?", "Sí" if hay_obesidad else "No",
                            "Tipo I, II o III" if hay_obesidad else ("Con sobrepeso" if clase >= 2 else "Sin sobrepeso"),
                            "#E8604F" if hay_obesidad else "#5FB38A"), unsafe_allow_html=True)
        s3.markdown(tarjeta("Seguridad", f"{p_clase * 100:.0f}%", "del modelo en ese nivel"),
                    unsafe_allow_html=True)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        rotulo("IMC en la escala de la OMS")
        mostrar(grafico_imc(imc))
        rotulo("Probabilidad por nivel")
        mostrar(grafico_proba(pipe.classes_, proba, clase))
        if nombre_modelo == 'SVM (RBF)':
            st.caption("En la SVM las probabilidades son una estimación aparte, por lo que a veces no coinciden "
                       "exactamente con la clase predicha.")
        st.caption("Herramienta académica. No reemplaza un diagnóstico médico.")

# ======================== TAB 2: RESULTADOS DE MODELOS ========================
with tab_modelos:
    tabla = art['tablas'][conjunto]
    rotulo(f"{nombre_modelo} · {CONJUNTOS[conjunto]}")
    k1, k2, k3 = st.columns(3)
    k1.metric("Accuracy", f"{tabla.loc[nombre_modelo, 'Accuracy']:.3f}")
    k2.metric("Recall macro", f"{tabla.loc[nombre_modelo, 'Recall macro']:.3f}")
    k3.metric("F1 macro", f"{tabla.loc[nombre_modelo, 'F1 macro']:.3f}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns([6, 5], gap="large")
    with g1:
        rotulo("Comparación de los 4 modelos (conjunto de prueba)")
        largo = tabla.rename_axis('Modelo').reset_index().melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
        fig = px.bar(largo, x='Modelo', y='Valor', color='Métrica', barmode='group', text_auto='.2f',
                     color_discrete_sequence=['#4FB3A5', '#C9A24D', '#7F8798'])
        fig.update_yaxes(range=[0, 1.05])
        fig.update_xaxes(title=None)
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, title=None))
        mostrar(estilo(fig, 380))
        st.caption("Partición 80/20 estratificada (random_state=42), igual que en el notebook.")
    with g2:
        rotulo(f"Matriz de confusión · {nombre_modelo}")
        cm = confusion_matrix(art['y_test'], art['preds'][(conjunto, nombre_modelo)], labels=list(range(7)))
        fig = px.imshow(cm, x=ETIQUETAS, y=ETIQUETAS, text_auto=True, aspect="auto",
                        color_continuous_scale=[[0, "#14171D"], [1, "#4FB3A5"]])
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Predicción", yaxis_title="Real")
        fig.update_xaxes(tickangle=-40)
        mostrar(estilo(fig, 380))

    rotulo("Tabla de métricas")
    st.dataframe(tabla.style.highlight_max(axis=0, props="background-color: #1F4A43; color: #FFFFFF;")
                 .format("{:.3f}"))

# ========================== TAB 3: EXPLORACIÓN ==========================
with tab_datos:
    crudo = art['crudo']
    c1, c2 = st.columns(2, gap="large")
    with c1:
        rotulo("Distribución de los niveles de obesidad")
        conteo = crudo['NObeyesdad'].map(NOMBRES_ES).value_counts().reindex(ETIQUETAS)
        fig = go.Figure(go.Bar(x=ETIQUETAS, y=conteo.values, marker_color=COLORES,
                               text=conteo.values, textposition="outside", cliponaxis=False))
        fig.update_xaxes(tickangle=-30)
        fig.update_yaxes(showgrid=True)
        mostrar(estilo(fig, 360))
    with c2:
        rotulo("IMC por nivel de obesidad")
        d = crudo.assign(IMC=crudo['Weight'] / crudo['Height'] ** 2, Nivel=crudo['NObeyesdad'].map(NOMBRES_ES))
        fig = go.Figure()
        for i, et in enumerate(ETIQUETAS):
            fig.add_trace(go.Box(y=d.loc[d['Nivel'] == et, 'IMC'], name=et, marker_color=COLORES[i],
                                 line=dict(width=1.5), showlegend=False))
        fig.update_xaxes(tickangle=-30)
        mostrar(estilo(fig, 360))
    st.caption("Cada nivel ocupa un tramo casi propio de IMC: el nivel se calculó a partir del IMC "
               "(la fuga de datos que se explica en el notebook).")

    rotulo("Vista previa de los datos")
    st.dataframe(crudo.head(50))
    st.caption(f"{len(crudo)} registros después de eliminar duplicados.")
