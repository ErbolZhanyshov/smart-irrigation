"""
Smart Irrigation System - Machine Learning Modeling Module
Modellerin eğitimi, TimeSeriesSplit doğrulaması, test değerlendirmesi ve model kaydetme fonksiyonları.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


FEATURE_COLUMNS = [
    "Soil Moisture Resistance (Ohm)",
    "Ambient Temperature (deg. C)",
    "Atmospheric Humidity (%)",
]

TARGET_COLUMN = "DC Water pump Status"


def get_project_root() -> Path:
    """Proje kök dizinini güvenli ve dinamik olarak döner."""
    return Path(__file__).resolve().parent.parent


def get_chronological_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Zaman serisi verilerini veri sızıntısını (data leakage) önlemek için kronolojik olarak böler.
    Rastgele train_test_split KULLANILMAZ.

    Parametreler:
        df: Sıralı zaman serisi veri çerçevesi.
        test_ratio: Test kümesi oranı (varsayılan: 0.2 = %20).

    Dönen Değer:
        X_train, X_test, y_train, y_test, train_df, test_df
    """
    if not df["date"].is_monotonic_increasing:
        df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    return X_train, X_test, y_train, y_test, train_df, test_df


def build_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Karşılaştırılacak 6 sınıflandırma algoritması için StandardScaler içeren Pipeline'ları kurar.
    Veri sızıntısını önlemek için ölçekleme Pipeline içinde uygulanır.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", C=1.0, random_state=random_state))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=random_state))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=random_state))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=random_state))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
    }
    return models


def evaluate_models_cv(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Yalnızca eğitim kümesi üzerinde TimeSeriesSplit (expanding window) ile çapraz doğrulama yapar.
    Test verisi bu aşamada ASLA kullanılmaz.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_records = []
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=tscv, scoring=scoring)
        cv_records.append({
            "Model": name,
            "CV_Accuracy_Mean": float(scores["test_accuracy"].mean()),
            "CV_Accuracy_Std": float(scores["test_accuracy"].std()),
            "CV_Precision_Mean": float(scores["test_precision"].mean()),
            "CV_Recall_Mean": float(scores["test_recall"].mean()),
            "CV_F1_Mean": float(scores["test_f1"].mean()),
            "CV_F1_Std": float(scores["test_f1"].std()),
            "CV_ROC_AUC_Mean": float(scores["test_roc_auc"].mean()),
            "CV_ROC_AUC_Std": float(scores["test_roc_auc"].std()),
        })

    return pd.DataFrame(cv_records)


def train_and_evaluate_test(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, Any]]:
    """
    Modelleri tam eğitim kümesinde eğitir ve test kümesi üzerinde değerlendirir.

    Dönen Değer:
        results_df: Model metrikleri karşılaştırma tablosu.
        trained_models: Eğitilmiş pipeline sözlüğü.
        detailed_eval: Confusion matrix ve classification report sözlükleri.
    """
    results = []
    trained_models = {}
    detailed_eval = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.5
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": roc,
            "True_Negatives": cm[0][0],
            "False_Positives": cm[0][1],
            "False_Negatives": cm[1][0],
            "True_Positives": cm[1][1],
        })

        detailed_eval[name] = {
            "confusion_matrix": cm,
            "classification_report": cr,
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist() if y_prob is not None else None,
        }

    results_df = pd.DataFrame(results).sort_values("F1_Score", ascending=False).reset_index(drop=True)
    return results_df, trained_models, detailed_eval


def run_sensitivity_analysis(
    df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42
) -> pd.DataFrame:
    """
    3.4 °C olan şüpheli kayıtların eğitimden çıkarılması durumundaki duyarlılık analizi.
    Ham CSV dosyasını değiştirmez; bellek içinde filtreleme yapar.
    """
    split_idx = int(len(df) * (1.0 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Filtrelenmiş eğitim kümesi (3.4 °C içeren 3 satır çıkarılır)
    train_clean = train_df[train_df["Ambient Temperature (deg. C)"] > 10.0].copy()

    X_train_full = train_df[FEATURE_COLUMNS]
    y_train_full = train_df[TARGET_COLUMN]

    X_train_clean = train_clean[FEATURE_COLUMNS]
    y_train_clean = train_clean[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    models_full = build_models(random_state=random_state)
    models_clean = build_models(random_state=random_state)

    comparison_records = []
    for name in models_full.keys():
        # Tam veriyle eğitim
        models_full[name].fit(X_train_full, y_train_full)
        pred_full = models_full[name].predict(X_test)
        acc_full = accuracy_score(y_test, pred_full)
        f1_full = f1_score(y_test, pred_full, zero_division=0)

        # Temiz veriyle eğitim
        models_clean[name].fit(X_train_clean, y_train_clean)
        pred_clean = models_clean[name].predict(X_test)
        acc_clean = accuracy_score(y_test, pred_clean)
        f1_clean = f1_score(y_test, pred_clean, zero_division=0)

        comparison_records.append({
            "Model": name,
            "Accuracy_Original_Train": acc_full,
            "Accuracy_Cleaned_Train": acc_clean,
            "Accuracy_Delta": acc_clean - acc_full,
            "F1_Original_Train": f1_full,
            "F1_Cleaned_Train": f1_clean,
            "F1_Delta": f1_clean - f1_full,
        })

    return pd.DataFrame(comparison_records)


def run_feature_ablation(
    df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42
) -> pd.DataFrame:
    """
    Toprak nem direnci tek başına kullanıldığında vs 3 özellik birlikte kullanıldığında
    elde edilen model başarılarını karşılaştırır.
    """
    split_idx = int(len(df) * (1.0 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    single_feat = ["Soil Moisture Resistance (Ohm)"]
    all_feats = FEATURE_COLUMNS
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    models_all = build_models(random_state=random_state)
    records = []

    for name in models_all.keys():
        # 1. Tek özellik (Sadece Direnç)
        pipe_single = build_models(random_state=random_state)[name]
        pipe_single.fit(train_df[single_feat], y_train)
        pred_single = pipe_single.predict(test_df[single_feat])
        acc_single = accuracy_score(y_test, pred_single)
        f1_single = f1_score(y_test, pred_single, zero_division=0)

        # 2. Tüm 3 özellik
        pipe_all = build_models(random_state=random_state)[name]
        pipe_all.fit(train_df[all_feats], y_train)
        pred_all = pipe_all.predict(test_df[all_feats])
        acc_all = accuracy_score(y_test, pred_all)
        f1_all = f1_score(y_test, pred_all, zero_division=0)

        records.append({
            "Model": name,
            "Accuracy_Single_Feature (Resistance Only)": acc_single,
            "Accuracy_All_3_Features": acc_all,
            "Accuracy_Diff": acc_all - acc_single,
            "F1_Single_Feature": f1_single,
            "F1_All_3_Features": f1_all,
            "F1_Diff": f1_all - f1_single,
        })

    return pd.DataFrame(records)


def save_artifacts(
    trained_models: Dict[str, Pipeline],
    test_results: pd.DataFrame,
    cv_results: pd.DataFrame,
    sensitivity_results: pd.DataFrame,
    ablation_results: pd.DataFrame,
) -> Dict[str, Path]:
    """
    Eğitilen modelleri 'models/' ve raporları 'reports/' dizinlerine kaydeder.
    """
    root = get_project_root()
    models_dir = root / "models"
    reports_dir = root / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    # En iyi modelin belirlenmesi (En yüksek test F1 skoru, eşitlikte Decision Tree)
    best_model_name = test_results.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, best_path)
    saved_paths["best_model"] = best_path

    # Tüm modelleri kaydetme
    for name, model in trained_models.items():
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        m_path = models_dir / f"{clean_name}.joblib"
        joblib.dump(model, m_path)
        saved_paths[f"model_{clean_name}"] = m_path

    # CSV Raporlarını kaydetme
    test_path = reports_dir / "model_comparison.csv"
    test_results.to_csv(test_path, index=False)
    saved_paths["test_comparison"] = test_path

    cv_path = reports_dir / "cv_scores.csv"
    cv_results.to_csv(cv_path, index=False)
    saved_paths["cv_scores"] = cv_path

    sens_path = reports_dir / "sensitivity_analysis.csv"
    sensitivity_results.to_csv(sens_path, index=False)
    saved_paths["sensitivity_analysis"] = sens_path

    ablation_path = reports_dir / "feature_ablation.csv"
    ablation_results.to_csv(ablation_path, index=False)
    saved_paths["feature_ablation"] = ablation_path

    return saved_paths


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_data

    print("=" * 70)
    print("SMART IRRIGATION SYSTEM - MACHINE LEARNING PIPELINE EXECUTION")
    print("=" * 70)

    data = load_data()
    X_tr, X_te, y_tr, y_te, tr_df, te_df = get_chronological_split(data, test_ratio=0.2)
    print(f"✓ Chronological split complete: Train={len(tr_df)}, Test={len(te_df)}")

    models = build_models()

    print("\n--- 1. TimeSeriesSplit (n_splits=5) Cross-Validation on Training Set ---")
    cv_df = evaluate_models_cv(models, X_tr, y_tr, n_splits=5)
    print(cv_df[["Model", "CV_Accuracy_Mean", "CV_F1_Mean", "CV_ROC_AUC_Mean"]].to_string())

    print("\n--- 2. Final Test Set Evaluation (Unseen 20% Chronological Split) ---")
    test_df_res, trained_clfs, eval_details = train_and_evaluate_test(models, X_tr, y_tr, X_te, y_te)
    print(test_df_res[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]].to_string())

    print("\n--- 3. Sensitivity Analysis (3.4 °C Records Removed from Training Set) ---")
    sens_df = run_sensitivity_analysis(data)
    print(sens_df[["Model", "Accuracy_Original_Train", "Accuracy_Cleaned_Train", "Accuracy_Delta"]].to_string())

    print("\n--- 4. Single Feature vs All Features Ablation Analysis ---")
    ablation_df = run_feature_ablation(data)
    print(ablation_df.to_string())

    print("\n--- 5. Saving Artifacts to models/ and reports/ ---")
    saved = save_artifacts(trained_clfs, test_df_res, cv_df, sens_df, ablation_df)
    for k, p in saved.items():
        print(f"  ✓ Saved: {k} -> {p.name}")
