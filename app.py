"""
Dashboard en Streamlit - Estimación de niveles de obesidad
Usa el mismo preprocesamiento y los mismos 4 modelos del notebook
estimacion_niveles_obesidad.ipynb (Random Forest, Árbol de decisión, SVM y Regresión logística).

Ejecutar con:   streamlit run app.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

st.set_page_config(page_title="Niveles de obesidad", page_icon="⚖️", layout="wide")

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
NUM_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
BINARIAS = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
FRECUENCIA = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
RUTA_CSV = "obesity.csv"   # opcional: si existe este archivo junto a app.py, se usa en vez de descargar de UCI

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
# 2. DATOS Y ENTRENAMIENTO (se ejecuta UNA sola vez gracias a la caché)
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
# 3. FUNCIONES AUXILIARES
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


# ----------------------------------------------------------------------------
# 4. INTERFAZ
# ----------------------------------------------------------------------------
st.title("⚖️ Estimación de niveles de obesidad")
st.caption("Proyecto de Machine Learning · Dataset UCI #544 · Clasificación multiclase (7 niveles)")

try:
    art = entrenar()
except Exception as e:  # sin internet o sin el CSV local
    st.error("No se pudieron cargar los datos. Verifica tu conexión a internet o coloca el archivo "
             f"`{RUTA_CSV}` (CSV del dataset) junto a `app.py`.\n\nDetalle: {e}")
    st.stop()

# ----- Barra lateral: selección del modelo -----
with st.sidebar:
    st.header("⚙️ Configuración")
    nombre_modelo = st.selectbox("Modelo de predicción", list(MODELOS.keys()),
                                 help="Elige el algoritmo que hará la predicción.")
    conjunto = st.radio("Variables que usa el modelo", list(CONJUNTOS.keys()),
                        format_func=lambda k: CONJUNTOS[k],
                        help="'Completo' usa peso y estatura (equivale a calcular con el IMC). "
                             "'Solo hábitos' ignora peso y estatura.")
    st.info("**Nota:** en el notebook se detectó que el nivel de obesidad se calculó a partir del IMC "
            "(peso/estatura²). Por eso el modelo *completo* acierta más (~95 %), pero casi está "
            "re-aprendiendo la fórmula del IMC. El modelo *solo hábitos* (~83 %) mide el efecto real de los hábitos.")

tab_pred, tab_modelos, tab_datos = st.tabs(["🔮 Predicción", "📊 Resultados de los modelos", "🔎 Exploración de datos"])

# ============================ TAB 1: PREDICCIÓN ============================
with tab_pred:
    st.subheader("Ingresa los datos de la persona")

    c1, c2, c3, c4 = st.columns(4)
    genero = c1.selectbox("Género", ["Femenino", "Masculino"])
    edad = c2.number_input("Edad (años)", min_value=10, max_value=100, value=25, step=1)
    estatura = c3.number_input("Estatura (m)", min_value=1.20, max_value=2.30, value=1.70, step=0.01, format="%.2f")
    peso = c4.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5, format="%.1f")

    imc = peso / estatura ** 2
    st.markdown(f"**IMC calculado:** `{imc:.1f} kg/m²`  ·  Categoría OMS: **{clasificar_imc_oms(imc)}**")

    with st.expander("Hábitos y condición física", expanded=True):
        h1, h2, h3 = st.columns(3)
        antecedentes = h1.selectbox("¿Antecedentes familiares de sobrepeso?", ["Sí", "No"])
        favc = h1.selectbox("¿Come con frecuencia alimentos muy calóricos?", ["Sí", "No"])
        fcvc = h1.selectbox("Consumo de vegetales en las comidas", [1, 2, 3], index=1,
                            format_func=lambda v: {1: "Nunca", 2: "A veces", 3: "Siempre"}[v])
        ncp = h2.slider("Comidas principales al día", 1, 4, 3)
        caec = h2.selectbox("Come entre comidas", [0, 1, 2, 3], index=1,
                            format_func=lambda v: {0: "No", 1: "A veces", 2: "Frecuentemente", 3: "Siempre"}[v])
        ch2o = h2.selectbox("Agua diaria", [1, 2, 3], index=1,
                            format_func=lambda v: {1: "Menos de 1 L", 2: "Entre 1 y 2 L", 3: "Más de 2 L"}[v])
        smoke = h3.selectbox("¿Fuma?", ["No", "Sí"])
        scc = h3.selectbox("¿Controla las calorías que consume?", ["No", "Sí"])
        faf = h3.selectbox("Actividad física semanal", [0, 1, 2, 3], index=1,
                           format_func=lambda v: {0: "Ninguna", 1: "1-2 días", 2: "2-4 días", 3: "4-5 días"}[v])
        h4, h5, h6 = st.columns(3)
        tue = h4.selectbox("Uso diario de dispositivos (tecnología)", [0, 1, 2],
                           format_func=lambda v: {0: "0-2 horas", 1: "3-5 horas", 2: "Más de 5 horas"}[v])
        calc = h5.selectbox("Consumo de alcohol", [0, 1, 2, 3],
                            format_func=lambda v: {0: "No", 1: "A veces", 2: "Frecuentemente", 3: "Siempre"}[v])
        mtrans = h6.selectbox("Medio de transporte habitual",
                              ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"],
                              format_func=lambda v: {"Public_Transportation": "Transporte público",
                                                     "Automobile": "Automóvil", "Walking": "Caminando",
                                                     "Bike": "Bicicleta", "Motorbike": "Motocicleta"}[v])

    datos = dict(genero=genero, edad=edad, estatura=estatura, peso=peso, antecedentes=antecedentes, favc=favc,
                 fcvc=fcvc, ncp=ncp, caec=caec, smoke=smoke, ch2o=ch2o, scc=scc, faf=faf, tue=tue,
                 calc=calc, mtrans=mtrans)

    # Aviso si los valores están fuera de lo que el modelo vio al entrenar
    fuera = [n for n, v in [("Edad", edad), ("Estatura", estatura), ("Peso", peso)]
             if not art['rangos'][{"Edad": "Age", "Estatura": "Height", "Peso": "Weight"}[n]][0] <= v
             <= art['rangos'][{"Edad": "Age", "Estatura": "Height", "Peso": "Weight"}[n]][1]]
    if fuera:
        st.warning(f"{', '.join(fuera)} fuera del rango de los datos de entrenamiento: "
                   "la predicción puede ser poco fiable.")

    # ----- Predicción -----
    pipe = art['modelos'][(conjunto, nombre_modelo)]
    X_nuevo = construir_fila(datos, art['cols'][conjunto])
    clase = int(pipe.predict(X_nuevo)[0])
    proba = pipe.predict_proba(X_nuevo)[0]

    st.divider()
    st.subheader(f"Resultado · {nombre_modelo}")
    r1, r2, r3 = st.columns(3)
    r1.metric("IMC", f"{imc:.1f}", clasificar_imc_oms(imc), delta_color="off")
    r2.metric("¿Tiene obesidad?", "SÍ" if clase >= 4 else "NO")
    r3.metric("Nivel estimado", ETIQUETAS[clase])

    if clase >= 4:
        st.error(f"El modelo clasifica a esta persona con **{ETIQUETAS[clase]}**.")
    elif clase >= 2:
        st.warning(f"No hay obesidad, pero el modelo detecta **{ETIQUETAS[clase]}**.")
    else:
        st.success(f"El modelo clasifica a esta persona en **{ETIQUETAS[clase]}**: sin obesidad ni sobrepeso.")

    st.markdown("**Probabilidad asignada a cada nivel**")
    tabla_proba = pd.DataFrame({"Nivel": [ETIQUETAS[int(c)] for c in pipe.classes_],
                                "Probabilidad (%)": (proba * 100).round(1)})
    st.dataframe(tabla_proba, hide_index=True,
                 column_config={"Probabilidad (%)": st.column_config.ProgressColumn(
                     "Probabilidad (%)", min_value=0, max_value=100, format="%.1f%%")})
    if nombre_modelo == 'SVM (RBF)':
        st.caption("En la SVM las probabilidades son una estimación aparte, por lo que a veces no coinciden "
                   "exactamente con la clase predicha.")
    st.caption("⚠️ Herramienta académica. No reemplaza un diagnóstico médico.")

# ======================== TAB 2: RESULTADOS DE MODELOS ========================
with tab_modelos:
    st.subheader(f"Métricas en el conjunto de prueba · {CONJUNTOS[conjunto]}")
    tabla = art['tablas'][conjunto]
    st.dataframe(tabla.style.highlight_max(axis=0, color="#c6efce").format("{:.3f}"))
    st.bar_chart(tabla)
    st.caption("Partición 80/20 estratificada (`random_state=42`), igual que en el notebook. "
               "Accuracy, Recall macro y F1 macro.")

    st.subheader(f"Matriz de confusión · {nombre_modelo}")
    cm = confusion_matrix(art['y_test'], art['preds'][(conjunto, nombre_modelo)], labels=list(range(7)))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=ETIQUETAS, yticklabels=ETIQUETAS)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ========================== TAB 3: EXPLORACIÓN ==========================
with tab_datos:
    crudo = art['crudo']
    st.subheader("Distribución de los niveles de obesidad")
    conteo = crudo['NObeyesdad'].map(NOMBRES_ES).value_counts().reindex(ETIQUETAS)
    st.bar_chart(conteo)

    st.subheader("IMC por nivel de obesidad")
    d = crudo.assign(IMC=crudo['Weight'] / crudo['Height'] ** 2,
                     Nivel=crudo['NObeyesdad'].map(NOMBRES_ES))
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=d, x='Nivel', y='IMC', order=ETIQUETAS, color='skyblue', ax=ax2)
    ax2.set_xlabel("")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("Cada nivel ocupa un tramo casi propio de IMC: por eso el nivel se calculó a partir del IMC "
               "(la 'fuga de datos' que se explica en el notebook).")

    st.subheader("Vista previa de los datos")
    st.dataframe(crudo.head(50))
    st.caption(f"{len(crudo)} registros después de eliminar duplicados.")
