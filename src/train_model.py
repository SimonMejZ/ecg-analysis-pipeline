import argparse
import os
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'ptb_features.csv')
DEFAULT_REPORTS = os.path.join(_PROJECT_ROOT, 'reports')
DEFAULT_MODEL = os.path.join(_PROJECT_ROOT, 'models', 'rf_combined.joblib')

POSITIVE_LABEL = 'MI'
NEGATIVE_LABEL = 'Normal'

# Columns that describe the record or its quality rather than the patient's heart.
# They are used for filtering but never as model inputs (n_beats and duration
# would leak recording length, which differs between acquisition protocols).
METADATA_COLUMNS = ['record', 'patient', 'diagnosis', 'label', 'age', 'sex', 'duration_s',
                    'n_beats', 'nn_fraction', 'template_corr']

HRV_FEATURES = ['mean_rr', 'sdnn', 'rmssd', 'pnn50', 'mean_hr', 'sd_hr', 'cv_rr',
                'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_nu', 'hf_nu',
                'sd1', 'sd2', 'sd1_sd2_ratio', 'sample_entropy']

# drop recordings where many beats were rejected or beat shapes are inconsistent.
MIN_NN_FRACTION = 0.8
MIN_TEMPLATE_CORR = 0.8

# Small grid: the dataset has ~200 patients, so tune only the regularising parameters.
PARAM_GRID = {
    'max_depth': [None, 8],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 0.3],
}

RANDOM_STATE = 42


def load_dataset(path: str, quality_filter: bool = True) -> pd.DataFrame:
    """
    Loads the feature table, keeps the binary MI vs Normal task and applies the quality gate.
    """
    df = pd.read_csv(path)
    df = df[df['label'].isin([POSITIVE_LABEL, NEGATIVE_LABEL])].reset_index(drop=True)
    print(f"Loaded {len(df)} MI/Normal records from {df['patient'].nunique()} patients.")

    if quality_filter:
        good = (df['nn_fraction'] >= MIN_NN_FRACTION) & (df['template_corr'] >= MIN_TEMPLATE_CORR)
        print(f"Quality filter removed {(~good).sum()} records.")
        df = df[good].reset_index(drop=True)

    print(df['label'].value_counts().to_string())
    return df


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    The feature groups compared in the experiment, plus an age-only baseline that
    shows how much of the signal could be explained by the cohorts' age difference - see report for more info.
    """
    morphology = [c for c in df.columns if c not in METADATA_COLUMNS and c not in HRV_FEATURES]
    return {
        'HRV': HRV_FEATURES,
        'Morphology': morphology,
        'HRV + Morphology': HRV_FEATURES + morphology,
        'Age only (baseline)': ['age'],
    }


def make_model(tune: bool, n_splits_inner: int = 3):
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                random_state=RANDOM_STATE, n_jobs=-1)
    if not tune:
        return rf
    inner_cv = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(rf, PARAM_GRID, scoring='roc_auc', cv=inner_cv, n_jobs=1)


def cross_validate(df: pd.DataFrame, features: List[str], tune: bool, n_splits: int = 5,
                   importance: bool = False):
    """
    Patient-grouped cross-validation: all recordings of a patient fall in the same
    fold, so the model is always tested on people it has never seen.
    Hyperparameters are tuned inside each training fold (nested CV).
    Returns out-of-fold MI probabilities and, optionally, permutation importances.
    """
    X = df[features]
    y = (df['label'] == POSITIVE_LABEL).astype(int).to_numpy()
    groups = df['patient'].to_numpy()

    outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(df))
    importances = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        model = make_model(tune)
        fit_kwargs = {'groups': groups[train_idx]} if tune else {}
        model.fit(X.iloc[train_idx], y[train_idx], **fit_kwargs)
        oof_proba[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

        if importance:
            estimator = model.best_estimator_ if tune else model
            result = permutation_importance(estimator, X.iloc[test_idx], y[test_idx], scoring='roc_auc',
                                            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
            importances.append(result.importances_mean)

    mean_importance = pd.Series(np.mean(importances, axis=0), index=features) if importance else None
    return y, oof_proba, mean_importance


def summarise(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        'roc_auc': roc_auc_score(y, proba),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'sensitivity': recall_score(y, pred),                 # MI recall
        'specificity': recall_score(y, pred, pos_label=0),    # Normal recall
        'precision': precision_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred),
    }


def plot_roc_curves(results: Dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y, proba) in results.items():
        fpr, tpr, _ = roc_curve(y, proba)
        style = ':' if 'baseline' in name else '-'
        ax.plot(fpr, tpr, style, label=f"{name} (AUC = {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], color='grey', lw=0.8, ls='--')
    ax.set_xlabel('False positive rate (1 - specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_title('MI vs Normal - patient-grouped CV')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y: np.ndarray, proba: np.ndarray, title: str, path: str) -> None:
    cm = confusion_matrix(y, (proba >= 0.5).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, path: str, top_n: int = 20) -> None:
    top = importance.sort_values(ascending=True).tail(top_n)
    colors = ['tab:orange' if f in HRV_FEATURES else 'tab:blue' for f in top.index]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel('Mean drop in ROC AUC when permuted (held-out folds)')
    ax.set_title(f'Top {top_n} features - permutation importance\n(orange = HRV, blue = morphology)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(data_path: str, reports_dir: str, model_path: str, tune: bool, quality_filter: bool) -> None:
    df = load_dataset(data_path, quality_filter)
    os.makedirs(reports_dir, exist_ok=True)

    metrics, curves = {}, {}
    combined_importance = None
    for name, features in feature_sets(df).items():
        print(f"\n--- {name}: {len(features)} features ---")
        data = df.dropna(subset=['age']) if name.startswith('Age') else df
        is_combined = name == 'HRV + Morphology'
        y, proba, importance = cross_validate(data, features, tune, importance=is_combined)

        metrics[name] = summarise(y, proba)
        curves[name] = (y, proba)
        print(pd.Series(metrics[name]).round(3).to_string())

        if is_combined:
            combined_importance = importance
            combined_y, combined_proba = y, proba

    # Age-confound check: re-score the combined model's out-of-fold predictions only
    # on patients aged 40+, where the MI and Normal age distributions overlap.
    older = (df['age'] >= 40).to_numpy()
    metrics['HRV + Morphology (age >= 40 subset)'] = summarise(combined_y[older], combined_proba[older])
    n_older = pd.Series(combined_y[older]).value_counts()
    print(f"\n--- HRV + Morphology on age >= 40 subset "
          f"({n_older.get(1, 0)} MI / {n_older.get(0, 0)} Normal records) ---")
    print(pd.Series(metrics['HRV + Morphology (age >= 40 subset)']).round(3).to_string())

    metrics_df = pd.DataFrame(metrics).T.round(3)
    metrics_df.to_csv(os.path.join(reports_dir, 'metrics.csv'))
    combined_importance.sort_values(ascending=False).to_csv(
        os.path.join(reports_dir, 'feature_importance.csv'), header=['importance'])

    plot_roc_curves(curves, os.path.join(reports_dir, 'roc_curves.png'))
    plot_confusion_matrix(combined_y, combined_proba, 'HRV + Morphology (out-of-fold)',
                          os.path.join(reports_dir, 'confusion_matrix.png'))
    plot_feature_importance(combined_importance, os.path.join(reports_dir, 'feature_importance.png'))

    # Final model: tuned on all data, for use on new recordings.
    features = feature_sets(df)['HRV + Morphology']
    final = make_model(tune)
    y_all = (df['label'] == POSITIVE_LABEL).astype(int)
    final.fit(df[features], y_all, **({'groups': df['patient']} if tune else {}))
    if tune:
        print(f"\nBest hyperparameters (all data): {final.best_params_}")
        final = final.best_estimator_
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': final, 'features': features}, model_path)

    print("\n=== Summary (patient-grouped 5-fold CV) ===")
    print(metrics_df.to_string())
    print(f"\nReports saved to {reports_dir}")
    print(f"Final model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the MI vs Normal Random Forest.')
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--reports', default=DEFAULT_REPORTS)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--no-tune', action='store_true', help='Skip nested hyperparameter search.')
    parser.add_argument('--no-quality-filter', action='store_true', help='Keep low-quality recordings.')
    args = parser.parse_args()

    main(args.data, args.reports, args.model, tune=not args.no_tune, quality_filter=not args.no_quality_filter)
