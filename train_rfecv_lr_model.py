import warnings
import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, clone
import datetime
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
})

TARGET_COLUMN_CANDIDATES = ("Outcome", "Label", "label")


def clean_name(name):
    if name is None:
        return name
    value = str(name).replace('\t', ' ')
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def read_table(data_path):
    if str(data_path).lower().endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path, sheet_name='Sheet1')
    df.columns = [clean_name(col) for col in df.columns]
    return df


def resolve_target_column(df):
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    if len(df.columns) >= 69:
        return str(df.columns[-1])
    raise ValueError(f"Target column not found. Tried: {TARGET_COLUMN_CANDIDATES} and final-column fallback.")


class CustomRFECV:
    def __init__(self, estimator, min_features_to_select=5, step=1, cv=5, scoring='roc_auc', random_state=42):
        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
    def fit(self, X, y):
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.feature_names_ = X.columns.tolist()
        n_features = X.shape[1]
        self.cv_scores_ = []
        self.n_features_history_ = []
        self.support_history_ = []
        self.ranking_ = np.ones(n_features, dtype=int) * (n_features + 1)
        current_support = np.ones(n_features, dtype=bool)
        current_ranking = 1
        print("Starting RFECV...")
        while np.sum(current_support) >= self.min_features_to_select:
            n_current_features = np.sum(current_support)
            cv_scores = self._cross_validate(X.iloc[:, current_support], y)
            mean_score = np.mean(cv_scores)
            self.cv_scores_.append(mean_score)
            self.n_features_history_.append(n_current_features)
            self.support_history_.append(current_support.copy())
            eliminated_indices = ~current_support & (self.ranking_ > current_ranking)
            self.ranking_[eliminated_indices] = current_ranking
            if n_current_features <= self.min_features_to_select:
                break
            feature_importance = self._get_feature_importance(
                X.iloc[:, current_support], y, current_support
            )
            n_features_to_eliminate = min(self.step, n_current_features - self.min_features_to_select)
            if n_features_to_eliminate > 0:
                least_important_indices = np.argsort(feature_importance)[:n_features_to_eliminate]
                current_support_indices = np.where(current_support)[0]
                indices_to_eliminate = current_support_indices[least_important_indices]
                current_support[indices_to_eliminate] = False
                current_ranking += 1
            else:
                break
        remaining_indices = current_support & (self.ranking_ > current_ranking)
        self.ranking_[remaining_indices] = current_ranking
        if len(self.cv_scores_) > 0:
            self.best_score_ = max(self.cv_scores_)
            self.best_idx_ = self.cv_scores_.index(self.best_score_)
            self.n_features_opt_ = self.n_features_history_[self.best_idx_]
            self.support_opt_ = self.support_history_[self.best_idx_]
            self.support_ = self.support_opt_
            self.n_features_ = self.n_features_opt_
            print(f"RFECV Done. Best features: {self.n_features_opt_}, Score: {self.best_score_:.6f}")
        else:
            print("Error: RFECV failed.")
        return self
    def _cross_validate(self, X, y):
        cv_scores = []
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            estimator_clone = clone(self.estimator)
            estimator_clone.fit(X_train, y_train)
            if self.scoring == 'accuracy':
                y_pred = estimator_clone.predict(X_test)
                score = accuracy_score(y_test, y_pred)
            elif self.scoring == 'roc_auc':
                y_pred_proba = estimator_clone.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred_proba)
            elif self.scoring == 'f1':
                y_pred = estimator_clone.predict(X_test)
                score = f1_score(y_test, y_pred)
            else:
                y_pred = estimator_clone.predict(X_test)
                score = accuracy_score(y_test, y_pred)
            cv_scores.append(score)
        return cv_scores
    def _get_feature_importance(self, X, y, current_support):
        estimator_clone = clone(self.estimator)
        estimator_clone.fit(X, y)
        if hasattr(estimator_clone, 'coef_'):
            importance = np.abs(estimator_clone.coef_[0])
        else:
            importance = np.ones(X.shape[1])
        return importance
    def get_selected_features(self):
        if hasattr(self, 'support_'):
            return [name for name, selected in zip(self.feature_names_, self.support_) if selected]
        return []
    def plot_results(self):
        if hasattr(self, 'n_features_history_') and hasattr(self, 'cv_scores_'):
            n_features_data = self.n_features_history_
            cv_scores_data = self.cv_scores_
        else:
            n_features_data = getattr(self, 'n_features_', [])
            cv_scores_data = getattr(self, 'cv_scores_', [])
        if not isinstance(n_features_data, (list, np.ndarray)):
            n_features_data = [n_features_data]
        if not isinstance(cv_scores_data, (list, np.ndarray)):
            cv_scores_data = [cv_scores_data]
        if len(n_features_data) != len(cv_scores_data):
            min_len = min(len(n_features_data), len(cv_scores_data))
            n_features_data = n_features_data[:min_len]
            cv_scores_data = cv_scores_data[:min_len]
        if len(n_features_data) == 0:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(n_features_data, cv_scores_data, 'bo-')
        plt.axvline(x=self.n_features_opt_, color='r', linestyle='--',
                    label=f'Optimal: {self.n_features_opt_}')
        plt.xlabel('Number of Features')
        if self.scoring == 'roc_auc':
            ylabel = 'CV AUC'
        elif self.scoring == 'f1':
            ylabel = 'CV F1'
        else:
            ylabel = 'CV Accuracy'
        plt.ylabel(ylabel)
        plt.title('RFECV Selection Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('feature_select.svg', dpi=600, format='svg')
        plt.show()
class PreprocessingClassifier(BaseEstimator):
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features
        self.scaler = StandardScaler()
    def fit(self, X, y):
        X_processed = X.copy()
        categorical_features = self.categorical_features or []
        numerical_features = [col for col in X_processed.columns
                              if col not in categorical_features]
        for col in numerical_features:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        for col in categorical_features:
            if col in X_processed.columns:
                mode_val = X_processed[col].mode()
                if not mode_val.empty:
                    X_processed[col] = X_processed[col].fillna(mode_val.iloc[0])
                else:
                    X_processed[col] = X_processed[col].fillna(0)
        if len(numerical_features) > 0:
            X_processed[numerical_features] = self.scaler.fit_transform(X_processed[numerical_features])
        self.model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        self.model.fit(X_processed, y)
        return self
    def predict(self, X):
        X_processed = self._preprocess(X)
        return self.model.predict(X_processed)
    def predict_proba(self, X):
        X_processed = self._preprocess(X)
        return self.model.predict_proba(X_processed)
    @property
    def coef_(self):
        return self.model.coef_
    @property
    def intercept_(self):
        return self.model.intercept_
    def _preprocess(self, X):
        X_processed = X.copy()
        categorical_features = self.categorical_features or []
        numerical_features = [col for col in X_processed.columns
                              if col not in categorical_features]
        for col in numerical_features:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        for col in categorical_features:
            if col in X_processed.columns:
                mode_val = X_processed[col].mode()
                if not mode_val.empty:
                    X_processed[col] = X_processed[col].fillna(mode_val.iloc[0])
                else:
                    X_processed[col] = X_processed[col].fillna(0)
        if hasattr(self, 'scaler') and len(numerical_features) > 0:
            X_processed[numerical_features] = self.scaler.transform(X_processed[numerical_features])
        return X_processed
def custom_recursive_feature_elimination_lr(X, y, categorical_features=None, cv_folds=5, step=1, scoring='roc_auc'):
    estimator = PreprocessingClassifier(categorical_features)
    custom_rfecv = CustomRFECV(
        estimator=estimator,
        min_features_to_select=5,
        step=step,
        cv=cv_folds,
        scoring=scoring,
        random_state=42
    )
    custom_rfecv.fit(X, y)
    custom_rfecv.plot_results()
    selected_features = custom_rfecv.get_selected_features()
    ranking = custom_rfecv.ranking_
    support = custom_rfecv.support_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Selected': support,
        'Ranking': ranking
    }).sort_values('Ranking')
    return feature_importance_df, selected_features, custom_rfecv
def build_and_evaluate_logistic_model(X, y, selected_features, categorical_features=None, cv_folds=5, random_state=42):
    print("Evaluating Model (5-fold CV)...")
    X_selected = X[selected_features].copy()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_accuracies = []
    fold_f1_scores = []
    fold_aucs = []
    fold_sensitivities = []
    fold_specificities = []
    fold_predictions = []
    fold_true_labels = []
    fold_models = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_selected, y), 1):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = PreprocessingClassifier(categorical_features)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1)
        fold_aucs.append(auc)
        fold_sensitivities.append(sensitivity)
        fold_specificities.append(specificity)
        fold_predictions.extend(y_pred_proba)
        fold_true_labels.extend(y_test)
        fold_models.append(model)
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_sensitivity = np.mean(fold_sensitivities)
    std_sensitivity = np.std(fold_sensitivities)
    mean_specificity = np.mean(fold_specificities)
    std_specificity = np.std(fold_specificities)
    print("Finalizing Model...")
    final_model = PreprocessingClassifier(categorical_features)
    final_model.fit(X_selected, y)
    fpr, tpr, _ = roc_curve(fold_true_labels, fold_predictions)
    evaluation_results = {
        'accuracy': mean_accuracy,
        'accuracy_std': std_accuracy,
        'f1_score': mean_f1,
        'f1_score_std': std_f1,
        'auc': mean_auc,
        'auc_std': std_auc,
        'sensitivity': mean_sensitivity,
        'sensitivity_std': std_sensitivity,
        'specificity': mean_specificity,
        'specificity_std': std_specificity,
        'fold_accuracies': fold_accuracies,
        'fold_f1_scores': fold_f1_scores,
        'fold_aucs': fold_aucs,
        'fold_sensitivities': fold_sensitivities,
        'fold_specificities': fold_specificities,
        'roc_curve': (fpr, tpr),
        'y_true': fold_true_labels,
        'y_pred_proba': fold_predictions,
        'selected_features': selected_features,
        'final_model': final_model
    }
    print(f"Results: Acc={mean_accuracy:.4f}, F1={mean_f1:.4f}, AUC={mean_auc:.4f}")
    thresholds, net_benefits = plot_decision_curve_analysis(
        np.array(fold_true_labels),
        np.array(fold_predictions),
        "LR Model"
    )
    evaluation_results['dca_thresholds'] = thresholds
    evaluation_results['dca_net_benefits'] = net_benefits
    mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, aucs = plot_roc_curve_with_cv(
        fold_true_labels, fold_predictions, cv_folds
    )
    y_pred_all = [1 if prob > 0.5 else 0 for prob in fold_predictions]
    cm = plot_confusion_matrix_with_percentages(fold_true_labels, y_pred_all)
    evaluation_results.update({
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr,
        'fold_aucs': aucs
    })
    return evaluation_results, final_model
def plot_roc_curve_with_cv(fold_true_labels, fold_predictions, cv_folds=5):
    fprs = []
    tprs = []
    aucs = []
    for i in range(cv_folds):
        start_idx = i * len(fold_true_labels) // cv_folds
        end_idx = (i + 1) * len(fold_true_labels) // cv_folds
        y_true_fold = fold_true_labels[start_idx:end_idx]
        y_pred_proba_fold = fold_predictions[start_idx:end_idx]
        fpr, tpr, _ = roc_curve(y_true_fold, y_pred_proba_fold)
        roc_auc = roc_auc_score(y_true_fold, y_pred_proba_fold)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tprs = []
    for i in range(cv_folds):
        interp_tpr = np.interp(mean_fpr, fprs[i], tprs[i])
        interp_tpr[0] = 0.0
        mean_tprs.append(interp_tpr)
    mean_tpr = np.mean(mean_tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(mean_tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.figure(figsize=(10, 8))
    for i in range(cv_folds):
        plt.plot(fprs[i], tprs[i], alpha=0.3, linewidth=1,
                 label=f'Fold {i + 1} (AUC = {aucs[i]:.3f})')
    plt.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
             label=f'Mean ROC (AUC = {mean_auc:.3f} + {std_auc:.3f})')
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                     label='+1 std. dev.')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with CV')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.svg', dpi=600, format='svg')
    plt.show()
    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, aucs
def plot_decision_curve_analysis(y_true, y_pred_proba, model_name="Our Model"):
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefits = []
    none_treat = []
    n = len(y_true)
    prevalence = np.mean(y_true)
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)
        none_treat.append(0)
    all_treat = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(thresholds, net_benefits, label=model_name, linewidth=2)
    ax1.plot(thresholds, all_treat, label='Treat All', linestyle='--', color='red')
    ax1.plot(thresholds, none_treat, label='Treat None', linestyle='--', color='green')
    ax1.set_xlabel('Threshold Probability')
    ax1.set_ylabel('Net Benefit')
    ax1.set_title('DCA (Full Range)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_xticks(np.arange(0, 1.01, 0.1))
    ax1.set_ylim(-0.1, 0.8)
    clinical_thresholds = thresholds[:70]
    ax2.plot(clinical_thresholds, net_benefits[:70], label=model_name, linewidth=2)
    ax2.plot(clinical_thresholds, all_treat[:70], label='Treat All', linestyle='--', color='red')
    ax2.plot(clinical_thresholds, none_treat[:70], label='Treat None', linestyle='--', color='green')
    ax2.set_xlabel('Threshold Probability')
    ax2.set_ylabel('Net Benefit')
    ax2.set_title('DCA (Clinical Range)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.7)
    all_values = net_benefits[:70] + all_treat[:70] + none_treat[:70]
    valid_values = [v for v in all_values if not np.isnan(v) and not np.isinf(v)]
    if valid_values:
        y_min = min(valid_values)
        y_max = max(valid_values)
        ax2.set_ylim(y_min - 0.02, y_max + 0.02)
    plt.tight_layout()
    plt.show()
    return thresholds, net_benefits
def plot_feature_importance(model, feature_names, top_k=None):
    coefficients = np.abs(model.model.coef_[0])
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Coefficient': np.abs(coefficients)
    }).sort_values('Absolute_Coefficient', ascending=False)
    if top_k is not None:
        feature_importance_df = feature_importance_df.head(top_k)
    plt.figure(figsize=(10, 8))
    colors = ['blue' for coef in feature_importance_df['Coefficient']]
    y_pos = np.arange(len(feature_importance_df))
    plt.barh(y_pos, feature_importance_df['Absolute_Coefficient'], color=colors, alpha=0.7)
    plt.yticks(y_pos, feature_importance_df['Feature'])
    plt.xlabel('Importance (Absolute Coefficients)')
    plt.title('Feature Importance')
    for i, v in enumerate(feature_importance_df['Coefficient']):
        plt.text(feature_importance_df['Absolute_Coefficient'].iloc[i] + 0.01, i,
                 f'{v:.4f}', va='center', fontsize=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.svg', dpi=600, format='svg')
    plt.show()
    return feature_importance_df
def perform_statistical_tests(results, baseline_model='RFECV+LR'):
    p_values = {}
    metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
    for metric in metrics:
        p_values[metric] = {}
        baseline_scores = results[metric][baseline_model]
        for model_name in results[metric].keys():
            if model_name != baseline_model:
                other_scores = results[metric][model_name]
                t_stat, p_val = stats.ttest_rel(baseline_scores, other_scores)
                p_values[metric][model_name] = p_val
    return p_values
def plot_comparison_results(results, p_values=None):
    metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
    metric_names = ['Accuracy', 'F1-Score', 'AUC', 'Sensitivity', 'Specificity']
    models = list(results['accuracy'].keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        means = [np.mean(results[metric][model]) for model in models]
        stds = [np.std(results[metric][model]) for model in models]
        bars = axes[i].bar(range(len(models)), means, yerr=stds,
                           capsize=5, alpha=0.7, color=['red' if m == 'RFECV+LR' else 'blue' for m in models])
        if p_values and metric in p_values:
            for j, model in enumerate(models):
                if model != 'RFECV+LR' and model in p_values[metric]:
                    p_val = p_values[metric][model]
                    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    axes[i].text(j, means[j] + stds[j] + 0.02, sig_text,
                                 ha='center', va='bottom', fontweight='bold')
        axes[i].set_title(f'{metric_name} Comparison')
        axes[i].set_xticks(range(len(models)))
        axes[i].set_xticklabels(models, rotation=45)
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True, alpha=0.3)
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig('comparison_results.svg', dpi=600, format='svg')
    plt.show()
def save_comparison_results(comparison_results, p_values, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Model Comparison & Statistical Tests\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
        metric_names = ['Accuracy', 'F1', 'AUC', 'Sensitivity', 'Specificity']
        models = list(comparison_results['accuracy'].keys())
        for metric, metric_name in zip(metrics, metric_names):
            f.write(f"\n{metric_name}:\n")
            f.write("Model\t\tMean +/- STD\n")
            for model in models:
                mean_val = np.mean(comparison_results[metric][model])
                std_val = np.std(comparison_results[metric][model])
                f.write(f"{model:<12}\t{mean_val:.4f} +/- {std_val:.4f}\n")
        f.write("\n\nP-Values (vs RFECV+LR):\n")
        f.write("Model\t\tAcc_P\tF1_P\tAUC_P\tSens_P\tSpec_P\n")
        for model in models:
            if model != 'RFECV+LR':
                f.write(f"{model:<12}")
                for metric in metrics:
                    p_val = p_values[metric].get(model, 1.0)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    f.write(f"\t{p_val:.4f}{sig}")
                f.write("\n")
def save_results_to_txt(selected_features, evaluation_results, model, data_info, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Model Analysis Results (5-fold CV)\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Data Summary\n")
        f.write(f"N = {data_info['total_samples']}\n")
        f.write(f"Class 0/1: {data_info['class_0']}/{data_info['class_1']}\n")
        f.write(f"Features: {data_info['original_features']} -> {len(selected_features)}\n\n")
        f.write("Selected Features\n")
        for i, feature in enumerate(selected_features, 1):
            f.write(f"{i}. {feature}\n")
        f.write("\nEvaluation Results\n")
        f.write(f"Accuracy: {evaluation_results['accuracy']:.4f} +/- {evaluation_results['accuracy_std']:.4f}\n")
        f.write(f"F1: {evaluation_results['f1_score']:.4f} +/- {evaluation_results['f1_score_std']:.4f}\n")
        f.write(f"AUC: {evaluation_results['auc']:.4f} +/- {evaluation_results['auc_std']:.4f}\n")
        f.write(f"Sens: {evaluation_results['sensitivity']:.4f} +/- {evaluation_results['sensitivity_std']:.4f}\n")
        f.write(f"Spec: {evaluation_results['specificity']:.4f} +/- {evaluation_results['specificity_std']:.4f}\n\n")
def main():
    print("Loading Data...")
    data_path = 'data/raw/original_data_p6e.xlsx'
    if not os.path.exists(data_path):
        fallback_path = 'data/raw/Data_Process.xlsx'
        if os.path.exists(fallback_path):
            data_path = fallback_path
    df = read_table(data_path)
    target_column = resolve_target_column(df)
    categorical_columns = ['Gender', 'Primary Disease', 'Surgical Procedure']
    feature_columns = [c for c in df.columns[1:68] if c != target_column]
    X = df[feature_columns].copy()
    y = df[target_column].fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    feature_results, selected_features, custom_rfecv = custom_recursive_feature_elimination_lr(
        X, y, categorical_columns, cv_folds=5, scoring='accuracy'
    )
    evaluation_results, model = build_and_evaluate_logistic_model(
        X, y, selected_features, categorical_columns, cv_folds=5
    )
    X_processed = X.copy()
    for col in X_processed.columns:
        if col in categorical_columns:
            X_processed[col] = X_processed[col].fillna(X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 0)
        else:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
    comparison_results, comparison_models = run_comparison_experiment(
        X_processed, y, selected_features, categorical_columns, cv_folds=5
    )
    p_values = perform_statistical_tests(comparison_results, 'RFECV+LR')
    plot_comparison_results(comparison_results, p_values)
    feature_importance_df = plot_feature_importance(model, selected_features, top_k=30)
    data_info = {
        'total_samples': len(df),
        'class_0': (y == 0).sum(),
        'class_1': (y == 1).sum(),
        'class_ratio': f"{(y == 0).sum()}:{(y == 1).sum()}",
        'original_features': len(feature_columns)
    }
    output_dir = 'results/hand_training'
    os.makedirs(output_dir, exist_ok=True)
    save_results_to_txt(selected_features, evaluation_results, model, data_info, os.path.join(output_dir, 'model_results.txt'))
    save_comparison_results(comparison_results, p_values, os.path.join(output_dir, 'comparison_results.txt'))
    print(f"Process Complete. Results saved to {output_dir}")
    return selected_features, evaluation_results, model, feature_importance_df
def run_comparison_experiment(X, y, selected_features, categorical_features=None, cv_folds=5, random_state=42):
    models = {
        'RFECV+LR': LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'),
        'LR': LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'),
        'BP': MLPClassifier(random_state=random_state, max_iter=1000, hidden_layer_sizes=(100,)),
        'RF': RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight='balanced_subsample'),
        'DT': DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
    }
    results = {m: {name: [] for name in models.keys()} for m in ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']}
    results['all_true_labels'] = {name: [] for name in models.keys()}
    results['all_pred_probas'] = {name: [] for name in models.keys()}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        for name, model in models.items():
            X_curr = X[selected_features] if name == 'RFECV+LR' else X
            X_tr, X_te = X_curr.iloc[train_idx], X_curr.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if name in ['LR', 'BP', 'RFECV+LR']:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
            model.fit(X_tr, y_tr)
            y_p = model.predict(X_te)
            y_pp = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else y_p
            results['accuracy'][name].append(accuracy_score(y_te, y_p))
            results['f1'][name].append(f1_score(y_te, y_p))
            results['auc'][name].append(roc_auc_score(y_te, y_pp))
            tn, fp, fn, tp = confusion_matrix(y_te, y_p).ravel()
            results['sensitivity'][name].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            results['specificity'][name].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    return results, models
def plot_confusion_matrix_with_percentages(y_true, y_pred, model_name='LR', save_path='results/'):
    os.makedirs(save_path, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_row = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_col = cm.astype('float') / cm.sum(axis=0) * 100
    display_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            display_text[i, j] = f'{cm[i, j]}\n({cm_row[i, j]:.1f}%)\n({cm_col[i, j]:.1f}%)'
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=display_text, fmt='', cmap='Blues', square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(save_path, f'cm_{model_name}.svg'), format='svg')
    plt.show()
    return cm
if __name__ == "__main__":
    main()
