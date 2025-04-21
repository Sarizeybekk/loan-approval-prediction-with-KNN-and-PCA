import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc
)

# Başlık
st.title("Kredi Onay Tahmini - KNN + PCA + GridSearchCV")

# Veri yükleme
uploaded_file = st.file_uploader("Lütfen veri setini yükleyin (.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, skipinitialspace=True)
    st.subheader("Veri Setinin İlk 5 Satırı")
    st.dataframe(df.head())

    # Eksik veri kontrolü
    st.write("Eksik veri:", df.isnull().sum())

    # Kategorik kolonları encode etme
    categorical_columns = ["education", "self_employed", "loan_status"]
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # Eğitim ve test verisi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Pipeline ve GridSearch
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'pca__n_components': [2, 3, 5, 7],
        'knn__n_neighbors': list(range(1, 21)),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }

    st.write("Model eğitiliyor, lütfen bekleyin...")
    grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_final = best_model.predict(X_test)

    # Sonuçlar
    st.subheader("En İyi Parametreler")
    st.json(grid_search.best_params_)

    st.subheader("Test Doğruluğu")
    acc = accuracy_score(y_test, y_pred_final)
    st.write(f"Test doğruluğu: {acc:.2f}")

    st.subheader("Sınıflandırma Raporu")
    st.text(classification_report(y_test, y_pred_final))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_final)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Tahmin")
    ax_cm.set_ylabel("Gerçek")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    if len(set(y_test)) == 2:
        y_pred_proba = best_model.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        st.pyplot(fig_roc)

    # PCA Görselleştirme
    pca_vis = PCA(n_components=2)
    X_test_scaled = StandardScaler().fit_transform(X_test)  # Düzgün transform
    X_pca = pca_vis.fit_transform(X_test_scaled)

    st.subheader("PCA 2D Görselleştirme")
    fig_pca, ax_pca = plt.subplots()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_test, palette='Set1', alpha=0.7, ax=ax_pca)
    ax_pca.set_title("PCA Bileşenlerine Göre Test Verisi Sınıf Dağılımı")
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    st.pyplot(fig_pca)
else:
    st.warning("Lütfen bir CSV dosyası yükleyin.")