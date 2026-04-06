"""
Machine Learning Model for Books Classification
Predicts if a book will be well-rated (>= 4.0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score)


def prepare_ml_data(df):
    """
    Prepare data for ML model
    
    Args:
        df: DataFrame with book data
        
    Returns:
        tuple: (features, target, le_language, le_publisher)
    """
    df_ml = df.copy()
    df_ml["is_well_rated"] = (df_ml["average_rating"] >= 4.0).astype(int)
    
    # Encodage des variables catégoriques
    le_language = LabelEncoder()
    le_publisher = LabelEncoder()
    df_ml["language_encoded"] = le_language.fit_transform(df_ml["language_code"].fillna("unknown"))
    df_ml["publisher_encoded"] = le_publisher.fit_transform(df_ml["publisher"].fillna("unknown"))
    
    # Sélection des features
    features = df_ml[["num_pages", "ratings_count", "text_reviews_count", "language_encoded", 
                      "publisher_encoded", "publication_year"]]
    features = features.fillna(0)
    target = df_ml["is_well_rated"]
    
    # Suppression des valeurs NaN
    valid_idx = ~(features.isna().any(axis=1) | target.isna())
    features = features[valid_idx].copy()
    target = target[valid_idx].copy()
    
    return features, target, le_language, le_publisher


def train_model(features, target, test_size=0.2, random_state=42):
    """
    Train Random Forest model
    
    Args:
        features: Feature DataFrame
        target: Target Series
        test_size: Test set proportion
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba)
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba


def get_model_metrics(y_test, y_pred, y_pred_proba, target):
    """
    Calculate model metrics
    
    Returns:
        dict: Dictionary with metrics
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    baseline = max(target.mean(), 1 - target.mean())
    
    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba
    }


def plot_confusion_matrix_and_roc(metrics):
    """
    Plot confusion matrix and ROC curve
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matrice de confusion
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Mal noté", "Bien noté"],
                yticklabels=["Mal noté", "Bien noté"],
                ax=axes[0], cbar_kws={"label": "Nombre de prédictions"})
    axes[0].set_title("Matrice de Confusion — Random Forest", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Valeur Réelle")
    axes[0].set_xlabel("Prédiction")
    
    # Courbe ROC
    axes[1].plot(metrics["fpr"], metrics["tpr"], color="#4d96ff", lw=2.5, 
                 label=f'Courbe ROC (AUC = {metrics["roc_auc"]:.3f})')
    axes[1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Classifieur aléatoire')
    axes[1].set_xlabel('Taux Faux Positif')
    axes[1].set_ylabel('Taux Vrai Positif')
    axes[1].set_title('Courbe ROC', fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("confusion_roc_books.png", dpi=150, bbox_inches="tight")
    
    return fig


def plot_feature_importance(model, features):
    """
    Plot feature importance
    """
    labels_fr = {
        "num_pages": "Nombre de pages",
        "ratings_count": "Nombre d'évaluations",
        "text_reviews_count": "Nombre d'avis textuels",
        "language_encoded": "Langue",
        "publisher_encoded": "Éditeur",
        "publication_year": "Année de publication"
    }
    
    imp_df = pd.DataFrame({
        "Feature": features.columns,
        "Importance": model.feature_importances_
    })
    imp_df["Feature_FR"] = imp_df["Feature"].map(labels_fr)
    imp_df = imp_df.sort_values("Importance", ascending=True)
    
    colors = ["#ff4d6d" if v > 0.20 else ("#ff9500" if v > 0.10 else "#4f8ef7") 
              for v in imp_df["Importance"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(imp_df["Feature_FR"], imp_df["Importance"], color=colors, 
                   edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, imp_df["Importance"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", fontsize=10, fontweight="bold")
    
    ax.set_title("Importance des Variables — Modèle Random Forest", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Relative (%)")
    ax.set_xlim([0, max(imp_df["Importance"]) * 1.15])
    
    plt.tight_layout()
    plt.savefig("feature_importance_books.png", dpi=150, bbox_inches="tight")
    
    return fig, imp_df


def print_model_report(features, target, model, metrics, imp_df, df):
    """
    Print detailed model report and insights
    """
    print("=" * 80)
    print("RAPPORT DE MODÈLE - CLASSIFICATION DES LIVRES")
    print("=" * 80)
    
    # Préparation des données
    print("\nPRÉPARATION DES DONNÉES :")
    print(f"  • Nombre d'exemples: {len(features)}")
    print(f"  • Livres bien notés (>=4.0): {target.sum()} ({target.mean()*100:.1f}%)")
    print(f"  • Livres mal notés (<4.0): {(1-target).sum()} ({(1-target).mean()*100:.1f}%)")
    print(f"  • Nombre de variables: {len(features.columns)}")
    
    # Performance du modèle
    print("\nPERFORMANCE DU MODÈLE :")
    print(f"  • Précision (Accuracy): {metrics['accuracy']*100:.2f}%")
    print(f"  • Baseline (classe majoritaire): {metrics['baseline']*100:.2f}%")
    print(f"  • Amélioration vs baseline: +{(metrics['accuracy'] - metrics['baseline'])*100:.2f}%")
    print(f"  • AUC-ROC: {metrics['roc_auc']:.3f}")
    print(f"  • Sensibilité (Rappel): {metrics['sensitivity']:.3f}")
    print(f"  • Spécificité: {metrics['specificity']:.3f}")
    print(f"  • Précision: {metrics['precision']:.3f}")
    
    # Variable importance
    print("\nFACTEURS PRÉDICTIFS (Importance des Variables) :")
    for idx, row in imp_df.sort_values("Importance", ascending=False).iterrows():
        print(f"  • {row['Feature_FR']}: {row['Importance']*100:.1f}%")
    
    # Insights
    print("\nINSIGHTS CLÉS :")
    stats = df.groupby("rating_category")[["average_rating", "ratings_count"]].mean()
    eng_count = df[df['language_code']=='eng'].shape[0]
    fre_count = df[df['language_code']=='fre'].shape[0]
    
    print(f"""
  1. DISTRIBUTION DES ÉVALUATIONS
     • {(df['average_rating'] >= 4.0).sum()} livres ({(df['average_rating'] >= 4.0).mean()*100:.1f}%) sont bien notés
     • Note moyenne: {df['average_rating'].mean():.2f}/5.0

  2. LONGUEUR OPTIMALE
     • Les livres de 200-400 pages sont les mieux notés
     • {df[df['length_category']=='Moyen'].shape[0]} livres dans cette catégorie

  3. ENGAGEMENT DES LECTEURS
     • Corrélation forte: meilleure note = plus d'évaluations
     • Livres Excellents: {df[df['rating_category']=='Excellent']['ratings_count'].mean():.0f} évaluations en moyenne

  4. TENDANCES TEMPORELLES
     • {df[df['publication_year']>=2000].shape[0]} livres publiés après 2000 ({df[df['publication_year']>=2000].shape[0]/df.shape[0]*100:.1f}%)

  5. LANGUES
     • Anglais domine: {eng_count} livres ({eng_count/df.shape[0]*100:.1f}%)
     • Français: {fre_count} livres ({fre_count/df.shape[0]*100:.1f}%)
    """)
    
    print("\n" + "=" * 80)


def run_full_ml_pipeline(df, verbose=True):
    """
    Run complete ML pipeline
    
    Returns:
        dict: Dictionary with model, metrics, and visualizations
    """
    # Prepare data
    features, target, le_language, le_publisher = prepare_ml_data(df)
    
    # Train model
    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_model(features, target)
    
    # Get metrics
    metrics = get_model_metrics(y_test, y_pred, y_pred_proba, target)
    
    # Get feature importance
    fig_importance, imp_df = plot_feature_importance(model, features)
    
    # Plot confusion matrix and ROC
    fig_confusion = plot_confusion_matrix_and_roc(metrics)
    
    # Print report
    if verbose:
        print_model_report(features, target, model, metrics, imp_df, df)
    
    return {
        "model": model,
        "features": features,
        "target": target,
        "metrics": metrics,
        "imp_df": imp_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "le_language": le_language,
        "le_publisher": le_publisher,
        "fig_importance": fig_importance,
        "fig_confusion": fig_confusion
    }
