# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from features import convert_commas_to_dots, make_features, split_X_y

st.set_page_config(page_title="Gallstone Predictor", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    bundle = joblib.load("model.joblib")
    return bundle["pipeline"], bundle["threshold"]

def predict_proba(pipeline, df_features: pd.DataFrame) -> np.ndarray:
    X = df_features.to_numpy()
    proba = pipeline.predict_proba(X)[:, 1]
    return proba

st.title("🩺 Gallstone Status – Logistic Regression App")
st.write("Mini‑app para predecir **Gallstone Status** con tu modelo de Regresión Logística.")

pipeline, default_threshold = load_model()

tabs = st.tabs(["📄 Predicción individual", "📥 Batch por CSV"])

# ====== TAB 1: FORM ======
with tabs[0]:
    st.subheader("Completa los campos mínimos (o sube un CSV en la otra pestaña)")

    st.write("**Datos demográficos básicos:**")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender (0=female, 1=male)", [0, 1], index=0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
    with col2:
        comorbidity = st.selectbox("Comorbidity (0/1)", [0, 1], index=0)
        cad = st.selectbox("Coronary Artery Disease (CAD) (0/1)", [0, 1], index=0)
        hypothyroidism = st.selectbox("Hypothyroidism (0/1)", [0, 1], index=0)
        hyperlipidemia = st.selectbox("Hyperlipidemia (0/1)", [0, 1], index=0)
        dm = st.selectbox("Diabetes Mellitus (DM) (0/1)", [0, 1], index=0)

    st.write("**Composición corporal:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        bmi = st.number_input("Body Mass Index (BMI)", min_value=5.0, max_value=80.0, value=27.0, step=0.1)
        tbw = st.number_input("Total Body Water (TBW)", min_value=0.0, max_value=100.0, value=52.9, step=0.1)
        ecw = st.number_input("Extracellular Water (ECW)", min_value=0.0, max_value=50.0, value=21.2, step=0.1)
        icw = st.number_input("Intracellular Water (ICW)", min_value=0.0, max_value=80.0, value=31.7, step=0.1)
        ecf_tbw = st.number_input("Extracellular Fluid/Total Body Water (ECF/TBW)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
    with col2:
        tbfr = st.number_input("Total Body Fat Ratio (TBFR) (%)", min_value=0.0, max_value=80.0, value=19.2, step=0.1)
        lm = st.number_input("Lean Mass (LM) (%)", min_value=0.0, max_value=100.0, value=80.84, step=0.1)
        protein = st.number_input("Body Protein Content (Protein) (%)", min_value=0.0, max_value=30.0, value=18.88, step=0.1)
        vfr = st.number_input("Visceral Fat Rating (VFR)", min_value=0.0, max_value=30.0, value=9.0, step=0.1)
        bm = st.number_input("Bone Mass (BM)", min_value=0.0, max_value=10.0, value=3.7, step=0.1)
    with col3:
        mm = st.number_input("Muscle Mass (MM)", min_value=0.0, max_value=100.0, value=71.4, step=0.1)
        obesity = st.number_input("Obesity (%)", min_value=0.0, max_value=100.0, value=23.4, step=0.1)
        tfc = st.number_input("Total Fat Content (TFC)", min_value=0.0, max_value=100.0, value=17.8, step=0.1)
        vfa = st.number_input("Visceral Fat Area (VFA)", min_value=0.0, max_value=1000.0, value=39.7, step=0.1)
        vma = st.number_input("Visceral Muscle Area (VMA) (Kg)", min_value=0.0, max_value=100.0, value=10.6, step=0.1)
        hfa = st.number_input("Hepatic Fat Accumulation (HFA)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

    st.write("**Parámetros bioquímicos:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        glucose = st.number_input("Glucose", min_value=40.0, max_value=800.0, value=102.0, step=1.0)
        tc = st.number_input("Total Cholesterol (TC)", min_value=50.0, max_value=1000.0, value=250.0, step=1.0)
        ldl = st.number_input("Low Density Lipoprotein (LDL)", min_value=10.0, max_value=600.0, value=175.0, step=1.0)
        hdl = st.number_input("High Density Lipoprotein (HDL)", min_value=5.0, max_value=150.0, value=40.0, step=1.0)
        tg = st.number_input("Triglyceride", min_value=20.0, max_value=2000.0, value=134.0, step=1.0)
    with col2:
        ast = st.number_input("Aspartat Aminotransferaz (AST)", min_value=1.0, max_value=1000.0, value=20.0, step=1.0)
        alt = st.number_input("Alanin Aminotransferaz (ALT)", min_value=1.0, max_value=1000.0, value=22.0, step=1.0)
        alp = st.number_input("Alkaline Phosphatase (ALP)", min_value=1.0, max_value=1000.0, value=87.0, step=1.0)
        creatinine = st.number_input("Creatinine", min_value=0.1, max_value=10.0, value=0.82, step=0.01)
        gfr = st.number_input("Glomerular Filtration Rate (GFR)", min_value=1.0, max_value=200.0, value=112.47, step=0.1)
    with col3:
        crp = st.number_input("C-Reactive Protein (CRP)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        hgb = st.number_input("Hemoglobin (HGB)", min_value=5.0, max_value=20.0, value=16.0, step=0.1)
        vitd = st.number_input("Vitamin D", min_value=0.0, max_value=200.0, value=33.0, step=1.0)

    # Construimos un DF con TODAS las columnas como en el dataset original
    input_row = {
        "Age": age,
        "Gender": gender,
        "Comorbidity": comorbidity,
        "Coronary Artery Disease (CAD)": cad,
        "Hypothyroidism": hypothyroidism,
        "Hyperlipidemia": hyperlipidemia,
        "Diabetes Mellitus (DM)": dm,
        "Height": height,
        "Weight": weight,
        "Body Mass Index (BMI)": bmi,
        "Total Body Water (TBW)": tbw,
        "Extracellular Water (ECW)": ecw,
        "Intracellular Water (ICW)": icw,
        "Extracellular Fluid/Total Body Water (ECF/TBW)": ecf_tbw,
        "Total Body Fat Ratio (TBFR) (%)": tbfr,
        "Lean Mass (LM) (%)": lm,
        "Body Protein Content (Protein) (%)": protein,
        "Visceral Fat Rating (VFR)": vfr,
        "Bone Mass (BM)": bm,
        "Muscle Mass (MM)": mm,
        "Obesity (%)": obesity,
        "Total Fat Content (TFC)": tfc,
        "Visceral Fat Area (VFA)": vfa,
        "Visceral Muscle Area (VMA) (Kg)": vma,
        "Hepatic Fat Accumulation (HFA)": hfa,
        "Glucose": glucose,
        "Total Cholesterol (TC)": tc,
        "Low Density Lipoprotein (LDL)": ldl,
        "High Density Lipoprotein (HDL)": hdl,
        "Triglyceride": tg,
        "Aspartat Aminotransferaz (AST)": ast,
        "Alanin Aminotransferaz (ALT)": alt,
        "Alkaline Phosphatase (ALP)": alp,
        "Creatinine": creatinine,
        "Glomerular Filtration Rate (GFR)": gfr,
        "C-Reactive Protein (CRP)": crp,
        "Hemoglobin (HGB)": hgb,
        "Vitamin D": vitd
    }
    df_single = pd.DataFrame([input_row])

    # Aplicar mismo FE
    df_single = convert_commas_to_dots(df_single)  # inerte aquí, por consistencia
    df_feat = make_features(df_single)

    thr = st.slider("Threshold", 0.10, 0.90, float(default_threshold), 0.01,
                    help="Umbral para convertir probabilidad en clase (1=riesgo)")

    if st.button("Predecir"):
        proba = predict_proba(pipeline, df_feat)[0]
        pred = int(proba >= thr)

        st.metric("Probabilidad de Gallstone (1)", f"{proba*100:.1f}%")
        st.write(f"**Predicción:** {pred}  (0 = No, 1 = Sí)")
        st.info("Recuerda: esta app es con fines educativos/demostrativos. No sustituye criterio médico.")

# ====== TAB 2: CSV ======
with tabs[1]:
    st.subheader("Sube un CSV con las mismas columnas originales (incluye o no el target)")
    file = st.file_uploader("CSV", type=["csv"])
    thr = st.slider("Threshold (batch)", 0.10, 0.90, float(default_threshold), 0.01)

    if file is not None:
        df = pd.read_csv(file)
        # Convertir decimales con comas si aplica
        df = convert_commas_to_dots(df)
        # Guardamos si trae target para comparar
        y_true = df["Gallstone Status"].values if "Gallstone Status" in df.columns else None

        df_feat = make_features(df)
        proba = predict_proba(pipeline, df_feat)
        pred = (proba >= thr).astype(int)

        out = df.copy()
        out["proba_1"] = proba
        out["pred"] = pred

        st.write("Vista previa de resultados:")
        st.dataframe(out.head(20))

        if y_true is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            # Para ROC AUC, si solo hay una clase presente, manejamos excepción
            try:
                auc = roc_auc_score(y_true, proba)
            except Exception:
                auc = np.nan
            acc = accuracy_score(y_true, pred)
            prec = precision_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred, zero_division=0)
            f1 = f1_score(y_true, pred, zero_division=0)

            st.markdown("**Métricas (comparado contra tu columna Gallstone Status):**")
            st.write(f"- Accuracy:  {acc:.3f}")
            st.write(f"- Precision: {prec:.3f}")
            st.write(f"- Recall:    {rec:.3f}")
            st.write(f"- F1-score:  {f1:.3f}")
            st.write(f"- ROC-AUC:   {auc:.3f if not np.isnan(auc) else 'NA'}")
