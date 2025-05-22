import streamlit as st
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("ATHENA-DS: Dashboard Interativo para Descoberta de Fármacos")
st.markdown("Este painel utiliza dados reais do ChEMBL para treinar modelos preditivos com explicações SHAP.")

# Etapa 1: Buscar ChEMBL ID
target_name = st.text_input("Nome do alvo (ex: NPC1):", "NPC1")

if st.button("Buscar ChEMBL ID"):
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={target_name}"
    response = requests.get(url)
    data = response.json()
    for target in data["targets"]:
        st.write(f"ChEMBL ID: {target['target_chembl_id']} | Nome: {target['pref_name']} | Tipo: {target['target_type']}")

# Etapa 2: Escolher ChEMBL ID
target_id = st.text_input("ChEMBL ID do alvo:", "CHEMBL1293277")

if st.button("Buscar Compostos Bioativos"):
    # Buscar atividades
    activities_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_id}&limit=1000"
    response = requests.get(activities_url)
    data = response.json()

    activities_list = []
    for a in data["activities"]:
        activities_list.append({
            "molecule_chembl_id": a["molecule_chembl_id"],
            "standard_type": a.get("standard_type"),
            "standard_value": a.get("standard_value"),
            "standard_units": a.get("standard_units"),
            "pchembl_value": a.get("pchembl_value")
        })

    activities_df = pd.DataFrame(activities_list)
    st.subheader("Bioatividades coletadas")
    st.dataframe(activities_df)

    # Buscar SMILES
    compound_list = []
    for mol_id in activities_df["molecule_chembl_id"].dropna().unique()[:100]:
        r = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json")
        if r.status_code == 200:
            mol_data = r.json()
            if mol_data.get("molecule_structures"):
                smiles = mol_data["molecule_structures"].get("canonical_smiles")
                if smiles:
                    compound_list.append({
                        "molecule_chembl_id": mol_id,
                        "smiles": smiles
                    })

    compounds_df = pd.DataFrame(compound_list)
    st.subheader("Compostos e SMILES")
    st.dataframe(compounds_df)

    # Gerar descritores
    def extract_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RingCount': Descriptors.RingCount(mol)
        }

    descriptors_list = []
    for _, row in compounds_df.iterrows():
        try:
            d = extract_descriptors(row["smiles"])
            d["molecule_chembl_id"] = row["molecule_chembl_id"]
            descriptors_list.append(d)
        except:
            continue

    desc_df = pd.DataFrame(descriptors_list)
    merged_df = pd.merge(activities_df, desc_df, on="molecule_chembl_id")
    merged_df = merged_df.dropna(subset=["pchembl_value"])
    merged_df["active"] = merged_df["pchembl_value"].astype(float).apply(lambda x: 1 if x > 6 else 0)

    st.subheader("Distribuição de Ativos/Inativos")
    st.bar_chart(merged_df["active"].value_counts())

    # Balanceamento
    actives = merged_df[merged_df["active"] == 1].sample(n=10, random_state=42)
    inactives = merged_df[merged_df["active"] == 0].sample(n=10, random_state=42)
    balanced_df = pd.concat([actives, inactives])

    features = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "RingCount"]
    X = balanced_df[features]
    y = balanced_df["active"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Acurácia no conjunto de teste: {acc:.2f}")
    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, y_pred))

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    st.subheader("Importância das Variáveis (SHAP)")
    fig1 = plt.figure()
    shap.summary_plot(shap_values[1], X_train, plot_type="bar", show=False)
    st.pyplot(fig1)

    fig2 = plt.figure()
    shap.summary_plot(shap_values[1], X_train, show=False)
    st.pyplot(fig2)

    st.download_button("Baixar Resultados (CSV)", data=balanced_df.to_csv(index=False), file_name="resultados_modelo.csv")
