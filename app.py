"""
Dashboard en Streamlit - Estimación de niveles de obesidad
Replica el notebook estimacion_niveles_obesidad.ipynb: mismo preprocesamiento, misma partición 80/20
(random_state=42) y los 4 modelos con los hiperparámetros AJUSTADOS del Paso 9 (GridSearchCV).

Ejecutar con:   streamlit run app.py
Tema oscuro:    ver .streamlit/config.toml
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             recall_score)
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
MODELO_FINAL = 'Random Forest'   # elegido en el Paso 10 del notebook

CONJUNTOS = {
    'completo': 'Completo (con peso y estatura)',
    'habitos': 'Solo hábitos (sin peso ni estatura)',
}

# Modelos con los hiperparámetros AJUSTADOS (Paso 9 del notebook, GridSearchCV)
MODELOS = {
    'Random Forest':       lambda: RandomForestClassifier(n_estimators=400, bootstrap=False, max_features='sqrt',
                                                          min_samples_leaf=2, criterion='gini', random_state=42),
    'Árbol de decisión':   lambda: DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=1,
                                                          random_state=42),
    'SVM (RBF)':           lambda: SVC(kernel='rbf', C=10, gamma=0.3, probability=True, random_state=42),
    'Regresión logística': lambda: LogisticRegression(C=100, max_iter=1000, random_state=42),
}
# Los mismos modelos con valores POR DEFECTO (Paso 7), solo para comparar "antes vs después"
MODELOS_DEFECTO = {
    'Random Forest':       lambda: RandomForestClassifier(n_estimators=400, random_state=42),
    'Árbol de decisión':   lambda: DecisionTreeClassifier(random_state=42),
    'SVM (RBF)':           lambda: SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'Regresión logística': lambda: LogisticRegression(max_iter=1000, random_state=42),
}
# Hiperparámetros que se ajustaron: (valor por defecto, valor ajustado)
HIPERPARAMETROS = {
    'Random Forest': {'bootstrap': (True, False), 'max_features': ('sqrt', 'sqrt'),
                      'min_samples_leaf': (1, 2), 'criterion': ('gini', 'gini')},
    'Árbol de decisión': {'criterion': ('gini', 'entropy'), 'max_depth': (None, 15), 'min_samples_leaf': (1, 1)},
    'SVM (RBF)': {'C': (1, 10), 'gamma': ('scale', 0.3)},
    'Regresión logística': {'C': (1, 100)},
}

# Resultados del notebook que no se recalculan en la app (valores fijos con random_state=42)
_MODELOS_ORDEN = ['Random Forest', 'Árbol de decisión', 'SVM (RBF)', 'Regresión logística']
# Paso 8.1: validación cruzada de 5 pliegues, hiperparámetros por defecto, solo hábitos
CV_DEFECTO = pd.DataFrame({
    'Accuracy (media)':     [0.854, 0.723, 0.745, 0.575],
    'Recall macro (media)': [0.851, 0.717, 0.741, 0.564],
    'F1 macro (media)':     [0.851, 0.717, 0.733, 0.547],
    'F1 macro (desv.)':     [0.021, 0.019, 0.016, 0.024],
}, index=_MODELOS_ORDEN)
# Paso 9: GridSearchCV (F1 macro en validación cruzada, antes y después del ajuste)
RESUMEN_BUSQUEDA = pd.DataFrame({
    'F1 CV por defecto':      [0.851, 0.717, 0.733, 0.547],
    'F1 CV ajustado':         [0.853, 0.734, 0.795, 0.556],
    'Mejora (puntos)':        [0.2, 1.7, 6.2, 1.0],
    'Combinaciones probadas': [24, 48, 30, 6],
}, index=_MODELOS_ORDEN)
# Paso 11: importancia por permutación del modelo final (caída media del F1 macro, desviación estándar)
IMPORTANCIA_PERMUTACION = [
    ('Gender_Male', 0.215, 0.017), ('Age', 0.189, 0.019), ('FCVC', 0.184, 0.014),
    ('family_history_with_overweight', 0.103, 0.011), ('NCP', 0.070, 0.009), ('CALC', 0.054, 0.016),
    ('CAEC', 0.044, 0.007), ('CH2O', 0.038, 0.011), ('TUE', 0.033, 0.007), ('FAF', 0.022, 0.008),
    ('MTRANS_Public_Transportation', 0.014, 0.005), ('FAVC', 0.010, 0.004), ('SCC', 0.009, 0.002),
    ('MTRANS_Walking', 0.002, 0.001), ('SMOKE', 0.000, 0.000), ('MTRANS_Bike', 0.000, 0.000),
    ('MTRANS_Motorbike', 0.000, 0.001),
]

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
          font-weight: 600; margin: 1.1rem 0 .5rem 0; padding-bottom: .4rem; border-bottom: 1px solid #262B36; }

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
.creditos { font-size: .74rem; color: #6B7080; margin-top: 1.6rem; line-height: 1.6; }
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
    """Mismo preprocesamiento del notebook (Pasos 4 y 5): duplicados, binarias, ordinales, One-Hot, target."""
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


def metricas(y_real, y_pred):
    """Las 3 métricas del proyecto."""
    return {'Accuracy': accuracy_score(y_real, y_pred),
            'Recall macro': recall_score(y_real, y_pred, average='macro'),
            'F1 macro': f1_score(y_real, y_pred, average='macro')}


@st.cache_resource(show_spinner="Entrenando los modelos (solo la primera vez, puede tardar un poco)...")
def entrenar():
    df_original = cargar_datos()
    X_total, y, crudo = preparar(df_original)
    cols = {'completo': list(X_total.columns),
            'habitos': [c for c in X_total.columns if c not in ('Weight', 'Height')]}

    # Misma partición del notebook: 80/20, estratificada, random_state=42
    idx_train, idx_test = train_test_split(X_total.index, test_size=0.2, stratify=y, random_state=42)
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    X_train = {k: X_total.loc[idx_train, c] for k, c in cols.items()}
    X_test = {k: X_total.loc[idx_test, c] for k, c in cols.items()}

    modelos, preds, filas, f1_train = {}, {}, {}, {}
    for conjunto, columnas in cols.items():
        filas[conjunto] = {}
        for nombre, fabrica in MODELOS.items():
            pipe = construir_pipeline(fabrica(), columnas).fit(X_train[conjunto], y_train)
            pred = pipe.predict(X_test[conjunto])
            modelos[(conjunto, nombre)] = pipe
            preds[(conjunto, nombre)] = pred
            filas[conjunto][nombre] = metricas(y_test, pred)
            f1_train[(conjunto, nombre)] = f1_score(y_train, pipe.predict(X_train[conjunto]), average='macro')
    tablas = {k: pd.DataFrame(v).T.round(3) for k, v in filas.items()}

    # Valores por defecto (solo conjunto de hábitos, como en el notebook) y línea base
    filas_def = {}
    for nombre, fabrica in MODELOS_DEFECTO.items():
        pipe = construir_pipeline(fabrica(), cols['habitos']).fit(X_train['habitos'], y_train)
        filas_def[nombre] = metricas(y_test, pipe.predict(X_test['habitos']))
    base = DummyClassifier(strategy='most_frequent').fit(X_train['habitos'], y_train)
    baseline = metricas(y_test, base.predict(X_test['habitos']))

    rangos = {c: (float(crudo[c].min()), float(crudo[c].max())) for c in ['Age', 'Height', 'Weight']}
    return {'modelos': modelos, 'preds': preds, 'tablas': tablas, 'cols': cols, 'f1_train': f1_train,
            'tabla_def': pd.DataFrame(filas_def).T.round(3), 'baseline': baseline,
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'crudo': crudo, 'rangos': rangos, 'n_original': len(df_original)}


def metricas_rapidas(y, p):
    """Accuracy, Recall macro y F1 macro calculados con la matriz de confusión (rápido, para el bootstrap)."""
    cm = np.bincount(7 * y + p, minlength=49).reshape(7, 7)
    tp = np.diag(cm)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return tp.sum() / cm.sum(), rec.mean(), f1.mean()


@st.cache_data(show_spinner=False)
def ic_bootstrap(conjunto, nombre):
    """Intervalo de confianza del 95 % por bootstrap (1000 remuestreos del conjunto de prueba), como en el Paso 10.2."""
    art = entrenar()
    y = art['y_test'].to_numpy().astype(int)
    p = np.asarray(art['preds'][(conjunto, nombre)]).astype(int)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        boot.append(metricas_rapidas(y[i], p[i]))
    boot = np.array(boot)
    valor = metricas_rapidas(y, p)
    return pd.DataFrame({'Valor en prueba': valor,
                         'IC 95 % inferior': np.percentile(boot, 2.5, axis=0),
                         'IC 95 % superior': np.percentile(boot, 97.5, axis=0)},
                        index=['Accuracy', 'Recall macro', 'F1 macro']).round(3)


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


def nombre_modelo_visible(n):
    return f"{n} · modelo final" if n == MODELO_FINAL else n


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
    st.markdown('<div class="rotulo" style="margin-top:0">Configuración</div>', unsafe_allow_html=True)
    nombre_modelo = st.selectbox("Modelo de predicción", list(MODELOS.keys()),
                                 format_func=nombre_modelo_visible,
                                 help="Elige el algoritmo. Todos usan los hiperparámetros ajustados en el Paso 9.")
    conjunto = st.radio("Variables que usa el modelo", list(CONJUNTOS.keys()),
                        format_func=lambda k: CONJUNTOS[k],
                        help="'Completo' usa peso y estatura (equivale a calcular con el IMC). "
                             "'Solo hábitos' ignora peso y estatura.")
    st.markdown('<div class="nota">En el notebook se detectó que el nivel de obesidad se calculó a partir '
                'del IMC (peso/estatura²). Por eso el modelo <b>completo</b> acierta más (~95 %) pero casi '
                're-aprende esa fórmula. El modelo <b>solo hábitos</b> (~83 %) mide el efecto real de los '
                'hábitos y es el resultado oficial del proyecto.</div>', unsafe_allow_html=True)
    st.markdown('<div class="creditos">Curso: Introducción a los Sistemas Inteligentes<br>'
                'Noviembre 2026<br>'
                'Integrantes: David Borja Rivero, Roni Mendiz Barrios, Maicol Anaya Lucas<br>'
                'Docente: Sergio Antonio Sánchez Hernández</div>', unsafe_allow_html=True)

tab_pred, tab_modelos, tab_cv, tab_datos = st.tabs(
    ["Predicción", "Resultados de los modelos", "Validación y análisis", "Exploración de datos"])

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
        rotulo(f"Resultado · {nombre_modelo_visible(nombre_modelo)}")
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
        if conjunto == 'habitos':
            st.caption("Este modelo no usa peso ni estatura: el IMC se muestra solo como referencia.")
        st.caption("Herramienta académica. No reemplaza un diagnóstico médico.")

# ======================== TAB 2: RESULTADOS DE MODELOS ========================
with tab_modelos:
    tabla = art['tablas'][conjunto]
    pred_sel = art['preds'][(conjunto, nombre_modelo)]
    base = art['baseline']
    rotulo(f"{nombre_modelo_visible(nombre_modelo)} · {CONJUNTOS[conjunto]} · conjunto de prueba (418 registros)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{tabla.loc[nombre_modelo, 'Accuracy']:.3f}")
    k2.metric("Recall macro", f"{tabla.loc[nombre_modelo, 'Recall macro']:.3f}")
    k3.metric("F1 macro", f"{tabla.loc[nombre_modelo, 'F1 macro']:.3f}")
    k4.metric("Veces mejor que el baseline", f"{tabla.loc[nombre_modelo, 'Accuracy'] / base['Accuracy']:.1f}×",
              help="El baseline siempre predice la clase más frecuente (Accuracy 0.167).")

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns([6, 5], gap="large")
    with g1:
        rotulo("Comparación de los 4 modelos ajustados y el baseline")
        tabla_cmp = pd.concat([pd.DataFrame({'Baseline': base}).T.round(3), tabla])
        largo = tabla_cmp.rename_axis('Modelo').reset_index().melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
        fig = px.bar(largo, x='Modelo', y='Valor', color='Métrica', barmode='group', text_auto='.2f',
                     color_discrete_sequence=['#4FB3A5', '#C9A24D', '#7F8798'])
        fig.update_yaxes(range=[0, 1.05])
        fig.update_xaxes(title=None)
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, title=None))
        mostrar(estilo(fig, 380))
        st.caption("Partición 80/20 estratificada (random_state=42), igual que en el notebook.")
    with g2:
        rotulo(f"Matriz de confusión · {nombre_modelo}")
        modo = st.radio("Vista", ["Conteos", "Normalizada por fila"], horizontal=True, label_visibility="collapsed")
        cm = confusion_matrix(art['y_test'], pred_sel, labels=list(range(7)))
        if modo == "Conteos":
            z, fmt = cm, True
        else:
            z, fmt = cm / cm.sum(axis=1, keepdims=True), '.0%'
        fig = px.imshow(z, x=ETIQUETAS, y=ETIQUETAS, text_auto=fmt, aspect="auto",
                        color_continuous_scale=[[0, "#14171D"], [1, "#4FB3A5"]])
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Predicción", yaxis_title="Real")
        fig.update_xaxes(tickangle=-40)
        mostrar(estilo(fig, 350))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        rotulo(f"Desempeño por clase · {nombre_modelo}")
        rep = classification_report(art['y_test'], pred_sel, labels=list(range(7)), target_names=ETIQUETAS,
                                    output_dict=True, zero_division=0)
        df_rep = pd.DataFrame(rep).T.loc[ETIQUETAS, ['precision', 'recall', 'f1-score', 'support']]
        fig = go.Figure(go.Bar(x=ETIQUETAS, y=df_rep['f1-score'], marker_color=COLORES,
                               text=[f"{v:.2f}" for v in df_rep['f1-score']], textposition="outside",
                               cliponaxis=False))
        fig.update_yaxes(range=[0, 1.1], title="F1 por clase")
        fig.update_xaxes(tickangle=-30)
        mostrar(estilo(fig, 300))
        st.dataframe(df_rep.style.format({'precision': '{:.3f}', 'recall': '{:.3f}', 'f1-score': '{:.3f}',
                                          'support': '{:.0f}'}))
    with c2:
        rotulo(f"Métricas con intervalo de confianza del 95 % · {nombre_modelo}")
        st.dataframe(ic_bootstrap(conjunto, nombre_modelo).style.format("{:.3f}"))
        st.caption("Bootstrap con 1000 remuestreos del conjunto de prueba (Paso 10.2): indica cuánto podría "
                   "variar el resultado solo por el azar de qué registros cayeron en la prueba.")

        rotulo("Tamaño de los errores")
        y_t = art['y_test'].to_numpy().astype(int)
        y_p = np.asarray(pred_sel).astype(int)
        err = y_t != y_p
        dist = np.abs(y_t - y_p)[err]
        n_err = int(err.sum())
        e1, e2, e3 = st.columns(3)
        e1.metric("Errores", f"{n_err} de {len(y_t)}")
        e2.metric("A clase vecina", f"{(dist == 1).sum() / n_err:.0%}" if n_err else "—")
        e3.metric("Acierto ±1 clase", f"{(np.abs(y_t - y_p) <= 1).mean():.1%}")
        st.caption("Como las clases tienen orden, confundir clases vecinas es un error leve; confundir clases "
                   "lejanas es grave.")

    d1, d2 = st.columns(2, gap="large")
    with d1:
        rotulo("Efecto del ajuste de hiperparámetros (solo hábitos, F1 macro en prueba)")
        dv = (pd.DataFrame({'Por defecto': art['tabla_def']['F1 macro'],
                            'Ajustado': art['tablas']['habitos']['F1 macro']})
              .rename_axis('Modelo').reset_index()
              .melt(id_vars='Modelo', var_name='Configuración', value_name='F1 macro'))
        fig = px.bar(dv, x='Modelo', y='F1 macro', color='Configuración', barmode='group', text_auto='.3f',
                     color_discrete_sequence=['#7F8798', '#4FB3A5'])
        fig.update_yaxes(range=[0, 1.05])
        fig.update_xaxes(title=None)
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, title=None))
        mostrar(estilo(fig, 330))
        st.caption("La SVM es la gran beneficiada del ajuste; el Random Forest ya estaba cerca del óptimo.")
    with d2:
        rotulo("Sobreajuste: entrenamiento vs. prueba")
        sob = pd.DataFrame({'F1 entrenamiento': {n: art['f1_train'][(conjunto, n)] for n in MODELOS},
                            'F1 prueba': tabla['F1 macro']})
        sob['Brecha'] = sob['F1 entrenamiento'] - sob['F1 prueba']
        st.dataframe(sob.style.format("{:.3f}"))
        rotulo(f"Hiperparámetros ajustados · {nombre_modelo}")
        hp = pd.DataFrame([{'Hiperparámetro': k, 'Por defecto': str(v[0]), 'Ajustado': str(v[1])}
                           for k, v in HIPERPARAMETROS[nombre_modelo].items()])
        st.dataframe(hp, hide_index=True)

# ======================= TAB 3: VALIDACIÓN Y ANÁLISIS =======================
with tab_cv:
    st.caption("Resultados de los Pasos 8, 9 y 11 del notebook, obtenidos con la misma partición del proyecto "
               "(random_state=42). Son valores fijos: no se recalculan en la app.")
    v1, v2 = st.columns([5, 6], gap="large")
    with v1:
        rotulo("Validación cruzada de 5 pliegues · hiperparámetros por defecto")
        st.dataframe(CV_DEFECTO.style.format("{:.3f}"))
        st.caption("Solo hábitos, sobre el conjunto de entrenamiento. Una desviación pequeña indica un modelo estable.")
        st.markdown(tarjeta("Mejor modelo en validación cruzada", "Random Forest",
                            "gana a la SVM en 5 de 5 pliegues (diferencia media de F1 macro: +0.058)", COLORES[1]),
                    unsafe_allow_html=True)
    with v2:
        rotulo("Antes y después del ajuste de hiperparámetros")
        dv = (RESUMEN_BUSQUEDA[['F1 CV por defecto', 'F1 CV ajustado']].rename_axis('Modelo').reset_index()
              .melt(id_vars='Modelo', var_name='Configuración', value_name='F1 macro (validación cruzada)'))
        fig = px.bar(dv, x='Modelo', y='F1 macro (validación cruzada)', color='Configuración', barmode='group',
                     text_auto='.3f', color_discrete_sequence=['#7F8798', '#4FB3A5'])
        fig.update_yaxes(range=[0, 1.05])
        fig.update_xaxes(title=None)
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, title=None))
        mostrar(estilo(fig, 320))
        st.dataframe(RESUMEN_BUSQUEDA.style.format({'F1 CV por defecto': '{:.3f}', 'F1 CV ajustado': '{:.3f}',
                                                    'Mejora (puntos)': '{:.1f}', 'Combinaciones probadas': '{:.0f}'}))

    rotulo("Importancia por permutación · Random Forest ajustado (solo hábitos)")
    imp = pd.DataFrame(IMPORTANCIA_PERMUTACION, columns=['Variable', 'Caída media del F1', 'Desv. est.'])
    fig = go.Figure(go.Bar(x=imp['Caída media del F1'], y=imp['Variable'], orientation='h',
                           error_x=dict(type='data', array=imp['Desv. est.'], color='#8B909C'),
                           marker_color='#4FB3A5'))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Caída del F1 macro al barajar la variable (mayor = más importante)")
    mostrar(estilo(fig, 520))
    st.caption("Cuidado al interpretarla: el peso de Gender_Male, Age y FCVC está inflado por el origen "
               "sintético del dataset (SMOTE). Describe asociaciones en estos datos, no causas.")

# ========================== TAB 4: EXPLORACIÓN ==========================
with tab_datos:
    crudo = art['crudo']
    n_dup = art['n_original'] - len(crudo)
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown(tarjeta("Registros", f"{len(crudo):,}".replace(",", "."), f"tras quitar {n_dup} duplicados"),
                unsafe_allow_html=True)
    t2.markdown(tarjeta("Variables predictoras", "16", "más la variable objetivo"), unsafe_allow_html=True)
    t3.markdown(tarjeta("Clases", "7", "balanceadas (13 % a 17 % c/u)"), unsafe_allow_html=True)
    t4.markdown(tarjeta("Datos sintéticos", "≈ 77 %", "generados con SMOTE"), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        rotulo("Distribución de los niveles de obesidad")
        conteo = crudo['NObeyesdad'].map(NOMBRES_ES).value_counts().reindex(ETIQUETAS)
        fig = go.Figure(go.Bar(x=ETIQUETAS, y=conteo.values, marker_color=COLORES,
                               text=conteo.values, textposition="outside", cliponaxis=False))
        fig.update_xaxes(tickangle=-30)
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
               "(la fuga de datos que se explica en el notebook, Paso 5.5).")

    e1, e2 = st.columns([5, 6], gap="large")
    with e1:
        rotulo("IMC mínimo, máximo y promedio por clase")
        tabla_imc = (d.groupby('Nivel')['IMC'].agg(['min', 'max', 'mean']).loc[ETIQUETAS]
                     .rename(columns={'min': 'IMC mínimo', 'max': 'IMC máximo', 'mean': 'IMC promedio'}))
        st.dataframe(tabla_imc.style.format("{:.1f}"))
    with e2:
        rotulo("Correlación entre variables numéricas")
        corr = crudo[NUM_COLS].corr()
        fig = px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale=[[0, "#D04E40"], [0.5, "#14171D"], [1, "#4FB3A5"]])
        fig.update_layout(coloraxis_showscale=False)
        mostrar(estilo(fig, 360))

    rotulo("Distribución de una variable numérica")
    var = st.selectbox("Variable", NUM_COLS, label_visibility="collapsed")
    fig = px.histogram(crudo, x=var, nbins=20, color_discrete_sequence=['#4FB3A5'])
    fig.update_traces(marker_line_color="#0E1014", marker_line_width=1)
    fig.update_yaxes(title="Registros")
    mostrar(estilo(fig, 300))

    rotulo("Vista previa de los datos")
    st.dataframe(crudo.head(50))
    st.caption(f"{len(crudo)} registros después de eliminar duplicados.")
