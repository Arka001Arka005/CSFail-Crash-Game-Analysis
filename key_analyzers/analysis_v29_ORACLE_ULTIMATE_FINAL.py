import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from scipy.stats import skew, entropy
from tqdm import tqdm
import warnings
import gc
import multiprocessing as mp
from typing import Tuple, Dict, List, Optional
import os

warnings.filterwarnings("ignore")

# Optimization settings
N_JOBS = min(12, mp.cpu_count())  # Use all available cores
os.environ['OMP_NUM_THREADS'] = str(N_JOBS)

# Main settings
DATASET = "analytics_dataset_v3_players_41k.csv"
TARGET_STATE = 2
N_SPLITS_CV = 3
CONTAMINATION = 0.001  # More conservative contamination rate
SAMPLE_FRAC = 0.3  # Increased sample for better statistics

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific features for crash game analysis.
    """
    print("[INFO] Creating advanced domain features...")
    df = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['crashed_at', 'hidden_state']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARNING] Missing required columns: {missing_cols}")
        return df
    
    # Crash analysis features
    if 'crashed_at' in df.columns:
        # Basic crash statistics
        df['crash_volatility'] = df['crashed_at'].rolling(window=10, min_periods=1).std().fillna(0)
        df['crash_momentum'] = df['crashed_at'].diff().rolling(window=5, min_periods=1).mean().fillna(0)
        df['crash_zscore'] = (df['crashed_at'] - df['crashed_at'].rolling(50).mean()) / df['crashed_at'].rolling(50).std()
        df['crash_zscore'] = df['crash_zscore'].fillna(0)
        
        # Crash regime detection
        df['is_low_crash_regime'] = (df['crashed_at'] < df['crashed_at'].quantile(0.2)).astype(int)
        df['is_high_crash_regime'] = (df['crashed_at'] > df['crashed_at'].quantile(0.8)).astype(int)
        
        # Crash streaks
        df['low_crash_streak'] = df['is_low_crash_regime'].groupby((df['is_low_crash_regime'] != df['is_low_crash_regime'].shift()).cumsum()).cumcount()
        df['high_crash_streak'] = df['is_high_crash_regime'].groupby((df['is_high_crash_regime'] != df['is_high_crash_regime'].shift()).cumsum()).cumcount()
        
        # Time since last extreme event
        extreme_crashes = df['crashed_at'] > df['crashed_at'].quantile(0.95)
        df['rounds_since_extreme'] = np.arange(len(df)) - pd.Series(np.where(extreme_crashes)[0]).reindex(df.index, method='ffill').fillna(0)
    
    # Player behavior analysis
    bet_cols = [col for col in df.columns if 'bet' in col.lower() and df[col].dtype in [np.float64, np.int64]]
    if bet_cols:
        print(f"[INFO] Found {len(bet_cols)} betting columns")
        
        # Betting statistics
        df['total_bet_volume'] = df[bet_cols].sum(axis=1)
        df['active_players'] = (df[bet_cols] > 0).sum(axis=1)
        df['avg_bet_size'] = df['total_bet_volume'] / (df['active_players'] + 1e-10)
        
        # Betting concentration and diversity
        bet_probs = df[bet_cols].div(df[bet_cols].sum(axis=1) + 1e-10, axis=0)
        df['bet_entropy'] = -np.sum(bet_probs * np.log2(bet_probs + 1e-10), axis=1)
        df['bet_gini'] = 1 - np.sum(bet_probs**2, axis=1)  # Gini coefficient approximation
        
        # Whale detection
        df['max_single_bet'] = df[bet_cols].max(axis=1)
        df['whale_dominance'] = df['max_single_bet'] / (df['total_bet_volume'] + 1e-10)
        df['is_whale_round'] = (df['whale_dominance'] > 0.5).astype(int)
        
        # Betting momentum and trends
        df['bet_volume_ma5'] = df['total_bet_volume'].rolling(5).mean()
        df['bet_volume_ma20'] = df['total_bet_volume'].rolling(20).mean()
        df['bet_trend'] = (df['bet_volume_ma5'] / (df['bet_volume_ma20'] + 1e-10)).fillna(1)
        
        # Player engagement metrics
        if 'users_count' in df.columns:
            df['bet_per_user'] = df['total_bet_volume'] / (df['users_count'] + 1e-10)
            df['participation_rate'] = df['active_players'] / (df['users_count'] + 1e-10)
    
    # Market microstructure features
    if 'users_count' in df.columns:
        df['user_volatility'] = df['users_count'].rolling(10).std().fillna(0)
        df['user_growth'] = df['users_count'].pct_change().fillna(0)
    
    # State transition features
    if 'hidden_state' in df.columns:
        df['state_changed'] = (df['hidden_state'] != df['hidden_state'].shift(1)).astype(int)
        df['rounds_in_current_state'] = df.groupby((df['hidden_state'] != df['hidden_state'].shift()).cumsum()).cumcount()
        
        # Time since target state
        target_state_mask = df['hidden_state'] == TARGET_STATE
        if target_state_mask.any():
            target_indices = np.where(target_state_mask)[0]
            df['rounds_since_target'] = np.inf
            for i in range(len(df)):
                past_targets = target_indices[target_indices < i]
                if len(past_targets) > 0:
                    df.iloc[i, df.columns.get_loc('rounds_since_target')] = i - past_targets[-1]
            df['rounds_since_target'] = df['rounds_since_target'].replace(np.inf, len(df))
    
    # Technical indicators
    if 'crashed_at' in df.columns:
        # Bollinger Bands
        crash_mean = df['crashed_at'].rolling(20).mean()
        crash_std = df['crashed_at'].rolling(20).std()
        df['crash_bb_upper'] = crash_mean + 2 * crash_std
        df['crash_bb_lower'] = crash_mean - 2 * crash_std
        df['crash_bb_position'] = (df['crashed_at'] - crash_mean) / (2 * crash_std + 1e-10)
        
        # RSI-like indicator
        crash_delta = df['crashed_at'].diff()
        gain = crash_delta.where(crash_delta > 0, 0).rolling(14).mean()
        loss = (-crash_delta.where(crash_delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['crash_rsi'] = 100 - (100 / (1 + rs))
        df['crash_rsi'] = df['crash_rsi'].fillna(50)
    
    # Fill any remaining NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    print(f"[INFO] Created {len(df.columns)} total features")
    return df

def prepare_data_robust(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Robust data preparation with proper handling of edge cases.
    """
    print("[INFO] Preparing data with robust preprocessing...")
    
    # Sample data if too large
    original_size = len(df)
    if len(df) > 25000:
        df = df.sample(frac=SAMPLE_FRAC, random_state=42).sort_index()
        print(f"[INFO] Sampled {len(df):,} rows from {original_size:,} for efficiency")
    
    # Identify columns to drop
    drop_cols = ['round_id', 'start_at', 'crashed_at', 'is_low_crash', 'hidden_state']
    available_drops = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=available_drops, errors='ignore')
    y_true = (df['hidden_state'] == TARGET_STATE).astype(int) if 'hidden_state' in df.columns else np.zeros(len(df))
    
    # Select only numeric columns with sufficient variance
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns with too many NaN or zero variance
    valid_cols = []
    for col in numeric_cols:
        if X[col].isna().sum() < len(X) * 0.9:  # Less than 90% NaN
            if X[col].std() > 1e-10:  # Has some variance
                valid_cols.append(col)
    
    print(f"[INFO] Selected {len(valid_cols)} valid numeric features")
    
    if not valid_cols:
        raise ValueError("No valid numeric columns found!")
    
    X = X[valid_cols]
    
    # Robust imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Robust scaling (less sensitive to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Remove any remaining invalid values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"[INFO] Final data shape: {X_scaled.shape}")
    print(f"[INFO] True anomalies: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.4f}%)")
    
    return X_scaled, y_true, valid_cols, df

def run_anomaly_detection_suite(X: np.ndarray, y_true: np.ndarray) -> Dict[str, Dict]:
    """
    Run comprehensive anomaly detection with multiple algorithms.
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION SUITE")
    print("="*60)
    
    results = {}
    
    # Adjust contamination based on true anomaly rate
    true_anomaly_rate = y_true.mean()
    adaptive_contamination = min(0.01, max(0.0005, true_anomaly_rate * 2))
    print(f"[INFO] Using adaptive contamination: {adaptive_contamination:.6f}")
    
    # 1. Isolation Forest
    print("\n--- Isolation Forest ---")
    try:
        iso_forest = IsolationForest(
            contamination=adaptive_contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=N_JOBS,
            behaviour='new'  # Use new behavior for consistency
        )
        y_pred_iso = (iso_forest.fit_predict(X) == -1).astype(int)
        results['isolation_forest'] = evaluate_anomaly_performance(y_true, y_pred_iso, "Isolation Forest")
    except Exception as e:
        print(f"[ERROR] Isolation Forest failed: {e}")
        results['isolation_forest'] = {'precision': 0, 'recall': 0, 'f1': 0}
    
    # 2. One-Class SVM
    print("\n--- One-Class SVM ---")
    try:
        svm = OneClassSVM(
            nu=adaptive_contamination * 5,  # More conservative
            kernel='rbf',
            gamma='scale'
        )
        y_pred_svm = (svm.fit_predict(X) == -1).astype(int)
        results['one_class_svm'] = evaluate_anomaly_performance(y_true, y_pred_svm, "One-Class SVM")
    except Exception as e:
        print(f"[ERROR] One-Class SVM failed: {e}")
        results['one_class_svm'] = {'precision': 0, 'recall': 0, 'f1': 0}
    
    # 3. DBSCAN (outlier detection)
    print("\n--- DBSCAN Outlier Detection ---")
    try:
        # Use more conservative parameters for DBSCAN
        dbscan = DBSCAN(eps=0.8, min_samples=10, n_jobs=N_JOBS)
        clusters = dbscan.fit_predict(X)
        y_pred_dbscan = (clusters == -1).astype(int)
        results['dbscan'] = evaluate_anomaly_performance(y_true, y_pred_dbscan, "DBSCAN")
    except Exception as e:
        print(f"[ERROR] DBSCAN failed: {e}")
        results['dbscan'] = {'precision': 0, 'recall': 0, 'f1': 0}
    
    return results

def evaluate_anomaly_performance(y_true: np.ndarray, y_pred: np.ndarray, method_name: str) -> Dict:
    """
    Comprehensive evaluation of anomaly detection performance.
    """
    n_predicted_anomalies = y_pred.sum()
    n_true_anomalies = y_true.sum()
    
    if n_predicted_anomalies == 0:
        print(f"[WARNING] {method_name}: No anomalies detected")
        return {'precision': 0, 'recall': 0, 'f1': 0, 'detected': 0, 'true_positives': 0}
    
    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Detected anomalies: {n_predicted_anomalies}")
    print(f"True positives: {tp}/{n_true_anomalies}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detected': n_predicted_anomalies,
        'true_positives': tp
    }

def run_classification_analysis(df: pd.DataFrame) -> Dict[int, Optional[float]]:
    """
    Run classification analysis for different states.
    """
    print("\n" + "="*60)
    print("CLASSIFICATION ANALYSIS")
    print("="*60)
    
    if 'hidden_state' not in df.columns:
        print("[ERROR] No hidden_state column found")
        return {}
    
    # Analyze state distribution
    state_counts = df['hidden_state'].value_counts()
    print("State distribution:")
    for state, count in state_counts.items():
        print(f"  State {state}: {count:,} ({count/len(df)*100:.2f}%)")
    
    results = {}
    
    # Try to classify each state with sufficient samples
    for state in state_counts.index:
        if state_counts[state] >= 20:  # Minimum samples for classification
            print(f"\n--- Classifying State {state} ---")
            try:
                auc = classify_state(df, state)
                results[state] = auc
            except Exception as e:
                print(f"[ERROR] Classification of state {state} failed: {e}")
                results[state] = None
        else:
            print(f"\n--- Skipping State {state} (insufficient samples: {state_counts[state]}) ---")
            results[state] = None
    
    return results

def classify_state(df: pd.DataFrame, target_state: int) -> Optional[float]:
    """
    Classify a specific state using time series cross-validation.
    """
    # Prepare features
    drop_cols = ['round_id', 'start_at', 'crashed_at', 'is_low_crash', 'hidden_state']
    available_drops = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=available_drops, errors='ignore')
    y = (df['hidden_state'] == target_state).astype(int)
    
    # Select numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    auc_scores = []
    
    try:
        # Use LightGBM if available, otherwise sklearn
        try:
            from lightgbm import LGBMClassifier
            model_class = LGBMClassifier
            model_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': N_JOBS
            }
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
            model_params = {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': N_JOBS
            }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if y_train.sum() < 5 or y_test.sum() < 2:
                print(f"  Fold {fold+1}: Insufficient events, skipping...")
                continue
            
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.predict(X_test)
            
            if len(np.unique(y_test)) > 1:  # Need both classes for AUC
                auc = roc_auc_score(y_test, y_pred_proba)
                auc_scores.append(auc)
                print(f"  Fold {fold+1}: AUC = {auc:.4f}")
            else:
                print(f"  Fold {fold+1}: Only one class present, skipping AUC calculation")
        
        if auc_scores:
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            return mean_auc
        else:
            print("No valid AUC scores computed")
            return None
            
    except Exception as e:
        print(f"Classification error: {e}")
        return None

def analyze_feature_importance(df: pd.DataFrame, feature_names: List[str]) -> None:
    """
    Analyze feature importance and correlations.
    """
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    # Select numeric features that exist in the dataframe
    available_features = [f for f in feature_names if f in df.columns]
    if not available_features:
        print("[WARNING] No features available for analysis")
        return
    
    # Basic statistics
    feature_stats = df[available_features].describe()
    print("\nTop features by variance:")
    variances = df[available_features].var().sort_values(ascending=False)
    print(variances.head(10))
    
    # Correlation with target (if available)
    if 'hidden_state' in df.columns:
        target_corrs = df[available_features + ['hidden_state']].corr()['hidden_state'].abs().sort_values(ascending=False)
        print(f"\nTop features correlated with hidden_state:")
        print(target_corrs.head(10))

def main():
    """
    Main execution function.
    """
    print("="*80)
    print("ORACLE GODMODE LEGEND - COMPREHENSIVE ANOMALY ANALYSIS")
    print("="*80)
    print(f"Using {N_JOBS} CPU cores for parallel processing")
    
    # Load data
    try:
        df = pd.read_csv(DATASET)
        print(f"[INFO] Loaded {len(df):,} rounds from {DATASET}")
    except FileNotFoundError:
        print(f"[ERROR] Dataset {DATASET} not found!")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return
    
    # Basic data info
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Create advanced features
    try:
        df_enhanced = create_advanced_features(df)
        print(f"[INFO] Enhanced dataset shape: {df_enhanced.shape}")
    except Exception as e:
        print(f"[ERROR] Feature creation failed: {e}")
        df_enhanced = df
    
    # Prepare data for analysis
    try:
        X, y_true, feature_names, sample_df = prepare_data_robust(df_enhanced)
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {e}")
        return
    
    # Run anomaly detection
    anomaly_results = run_anomaly_detection_suite(X, y_true)
    
    # Run classification analysis
    classification_results = run_classification_analysis(sample_df)
    
    # Feature analysis
    analyze_feature_importance(sample_df, feature_names)
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    # Evaluate anomaly detection success
    best_anomaly_f1 = 0
    best_anomaly_method = None
    
    for method, results in anomaly_results.items():
        if results['f1'] > best_anomaly_f1:
            best_anomaly_f1 = results['f1']
            best_anomaly_method = method
    
    print(f"\nAnomaly Detection Results:")
    if best_anomaly_f1 > 0.1:
        print(f"SUCCESS! Best method: {best_anomaly_method}")
        print(f"  F1-Score: {best_anomaly_f1:.4f}")
        print(f"  True positives: {anomaly_results[best_anomaly_method]['true_positives']}")
        print("  Recommendation: Anomalies are detectable - investigate patterns!")
    elif best_anomaly_f1 > 0.01:
        print(f"WEAK SIGNAL detected with {best_anomaly_method}")
        print(f"  F1-Score: {best_anomaly_f1:.4f}")
        print("  Recommendation: Try ensemble methods or more data")
    else:
        print("NO SIGNIFICANT ANOMALY DETECTION")
        print("  Recommendation: Redefine anomalies or collect more targeted data")
    
    # Evaluate classification results
    print(f"\nClassification Results:")
    successful_classifications = 0
    for state, auc in classification_results.items():
        if auc is not None:
            if auc > 0.7:
                print(f"  State {state}: STRONG SIGNAL (AUC = {auc:.4f})")
                successful_classifications += 1
            elif auc > 0.6:
                print(f"  State {state}: MODERATE SIGNAL (AUC = {auc:.4f})")
            else:
                print(f"  State {state}: WEAK/NO SIGNAL (AUC = {auc:.4f})")
        else:
            print(f"  State {state}: INSUFFICIENT DATA")
    
    # Overall recommendation
    print(f"\nOVERALL ASSESSMENT:")
    if best_anomaly_f1 > 0.1 or successful_classifications > 0:
        print("ACTIONABLE SIGNALS DETECTED!")
        print("Next steps:")
        print("1. Focus on the best-performing method")
        print("2. Investigate feature patterns in detected anomalies")
        print("3. Develop real-time detection system")
        print("4. Collect more data to improve model stability")
    else:
        print("LIMITED PREDICTIVE POWER DETECTED")
        print("Recommendations:")
        print("1. Redefine what constitutes an 'anomaly'")
        print("2. Collect different types of features (player behavior, external factors)")
        print("3. Try unsupervised pattern mining")
        print("4. Consider that the target events may be truly random")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()