import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew, entropy
import pickle

# Try advanced libraries
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    print("⚠️ LightGBM not available")
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("⚠️ XGBoost not available") 
    HAS_XGB = False

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except ImportError:
    print("⚠️ TensorFlow not available")
    HAS_TF = False

warnings.filterwarnings("ignore")

class UltimateEverythingAnalyzer:
    def __init__(self):
        self.scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler()
        }
        self.player_clusters = None
        
    def load_all_data(self):
        """Load and merge ALL available data"""
        print("🔄 LOADING ALL DATA SOURCES...")
        
        # Main dataset
        df_main = pd.read_csv("analytics_dataset_v3_players_41k.csv")
        print(f"✅ Main dataset: {len(df_main):,} rounds")
        
        # Rounds data
        try:
            df_rounds = pd.read_csv("rounds.csv")
            df_main = df_main.merge(df_rounds, on='round_id', how='left', suffixes=('', '_rounds'))
            print(f"✅ Merged rounds data: {len(df_rounds):,} rounds")
        except:
            print("⚠️ rounds.csv not found")
            
        # BETS DATA - THE GAME CHANGER
        try:
            print("📊 Loading MASSIVE bets dataset...")
            df_bets = pd.read_csv("bets.csv")
            print(f"✅ Loaded bets: {len(df_bets):,} bets")
            
            # Aggregate bets per round
            bets_agg = self.create_betting_features(df_bets)
            df_main = df_main.merge(bets_agg, on='round_id', how='left')
            print(f"✅ Merged betting features")
            
        except Exception as e:
            print(f"⚠️ bets.csv not available: {e}")
            
        return df_main.sort_values('round_id').reset_index(drop=True)
    
    def create_betting_features(self, df_bets):
        """Create advanced betting behavior features"""
        print("🧠 CREATING BETTING BEHAVIOR FEATURES...")
        
        # Basic aggregations per round
        agg_features = df_bets.groupby('round_id').agg({
            'bet_amount_usd': ['count', 'sum', 'mean', 'std', 'min', 'max', 'median'],
            'cashout_ratio': ['mean', 'std', 'min', 'max', 'median', 'count'],
            'user_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = ['round_id'] + [f'bets_{col[0]}_{col[1]}' for col in agg_features.columns[1:]]
        
        # Advanced betting patterns
        for round_id in df_bets['round_id'].unique():
            round_bets = df_bets[df_bets['round_id'] == round_id]
            
            if len(round_bets) > 0:
                # Whale detection
                total_volume = round_bets['bet_amount_usd'].sum()
                top_10_pct = round_bets['bet_amount_usd'].quantile(0.9)
                whale_volume = round_bets[round_bets['bet_amount_usd'] >= top_10_pct]['bet_amount_usd'].sum()
                
                agg_features.loc[agg_features['round_id'] == round_id, 'whale_dominance'] = whale_volume / (total_volume + 1e-10)
                
                # Risk appetite  
                high_risk_bets = len(round_bets[round_bets['bet_amount_usd'] > round_bets['bet_amount_usd'].median()])
                agg_features.loc[agg_features['round_id'] == round_id, 'risk_appetite'] = high_risk_bets / len(round_bets)
                
                # Bet distribution entropy
                bet_bins = pd.cut(round_bets['bet_amount_usd'], bins=5, labels=False)
                bet_entropy = entropy(pd.Series(bet_bins).value_counts().values + 1e-10)
                agg_features.loc[agg_features['round_id'] == round_id, 'bet_entropy'] = bet_entropy
        
        return agg_features
    
    def create_ultimate_features(self, df):
        """Create the most advanced features possible"""
        print("🛠️ CREATING ULTIMATE FEATURES...")
        df = df.copy()
        
        # ============ HISTORICAL CRASH ANALYSIS ============
        if 'crashed_at' in df.columns:
            print("📈 Advanced crash time series analysis...")
            
            # Basic lags (safe)
            for lag in [1, 2, 3, 5, 10, 15, 20, 30]:
                df[f'crash_lag_{lag}'] = df['crashed_at'].shift(lag)
                
            # Rolling statistics on past crashes
            past_crashes = df['crashed_at'].shift(1)
            for window in [3, 5, 10, 15, 20, 30, 50]:
                df[f'past_crash_mean_{window}'] = past_crashes.rolling(window).mean()
                df[f'past_crash_std_{window}'] = past_crashes.rolling(window).std()
                df[f'past_crash_min_{window}'] = past_crashes.rolling(window).min()
                df[f'past_crash_max_{window}'] = past_crashes.rolling(window).max()
                df[f'past_crash_median_{window}'] = past_crashes.rolling(window).median()
                df[f'past_crash_q25_{window}'] = past_crashes.rolling(window).quantile(0.25)
                df[f'past_crash_q75_{window}'] = past_crashes.rolling(window).quantile(0.75)
                
            # Advanced statistics
            for window in [10, 20, 30]:
                rolling = past_crashes.rolling(window)
                df[f'past_crash_skew_{window}'] = rolling.skew()
                df[f'past_crash_kurtosis_{window}'] = rolling.apply(lambda x: kurtosis(x, nan_policy='omit'))
                df[f'past_crash_range_{window}'] = df[f'past_crash_max_{window}'] - df[f'past_crash_min_{window}']
                df[f'past_crash_iqr_{window}'] = df[f'past_crash_q75_{window}'] - df[f'past_crash_q25_{window}']
                
            # Fourier Analysis - detect hidden periodicities
            print("🌊 Fourier Transform analysis...")
            for window in [50, 100]:
                def fourier_features(series):
                    if len(series) < window or series.isna().all():
                        return [0, 0, 0]
                    
                    # Remove NaN and ensure we have enough data
                    clean_series = series.dropna()
                    if len(clean_series) < 10:
                        return [0, 0, 0]
                    
                    # Apply FFT
                    fft_vals = fft(clean_series.values)
                    freqs = fftfreq(len(clean_series))
                    
                    # Get dominant frequencies
                    power = np.abs(fft_vals) ** 2
                    dominant_freq_idx = np.argsort(power)[-3:]  # Top 3 frequencies
                    
                    return [
                        np.mean(freqs[dominant_freq_idx]),  # Mean dominant frequency
                        np.max(power[1:len(power)//2]),     # Max power (excluding DC)
                        np.sum(power[1:len(power)//4]) / np.sum(power[1:len(power)//2])  # Low freq ratio
                    ]
                
                fourier_results = past_crashes.rolling(window).apply(
                    lambda x: fourier_features(x)[0], raw=False
                )
                df[f'fourier_dom_freq_{window}'] = fourier_results
                
            # Volatility regimes
            for window in [10, 20, 50]:
                vol = past_crashes.rolling(window).std()
                df[f'volatility_{window}'] = vol
                df[f'high_vol_regime_{window}'] = (vol > vol.rolling(100).quantile(0.8)).astype(int)
                df[f'low_vol_regime_{window}'] = (vol < vol.rolling(100).quantile(0.2)).astype(int)
                
        # ============ BETTING BEHAVIOR ANALYSIS ============
        betting_cols = [col for col in df.columns if col.startswith('bets_')]
        if betting_cols:
            print("💰 Advanced betting behavior analysis...")
            
            # Lag betting features (safe to use)
            for col in betting_cols:
                for lag in [1, 2, 3, 5]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
            # Betting momentum
            for col in ['bets_bet_amount_usd_sum', 'bets_bet_amount_usd_count']:
                if col in df.columns:
                    df[f'{col}_momentum'] = df[col].pct_change().fillna(0)
                    df[f'{col}_acceleration'] = df[f'{col}_momentum'].diff().fillna(0)
                    
            # Betting pressure indicators
            if 'whale_dominance' in df.columns:
                for lag in [1, 2, 3]:
                    df[f'whale_dominance_lag_{lag}'] = df['whale_dominance'].shift(lag)
                    
        # ============ TIME SERIES DECOMPOSITION ============
        print("📊 Time series decomposition...")
        
        # Trend extraction using different methods
        if 'crashed_at' in df.columns:
            # Simple linear trend
            df['time_index'] = range(len(df))
            for window in [50, 100, 200]:
                df[f'crash_trend_{window}'] = past_crashes.rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window and not np.isnan(x).all() else 0,
                    raw=False
                )
                
        # ============ TEMPORAL FEATURES ============
        if 'start_at' in df.columns:
            print("⏰ Advanced temporal features...")
            df['start_at'] = pd.to_datetime(df['start_at'], errors='coerce')
            
            # Basic time features
            df['hour'] = df['start_at'].dt.hour
            df['minute'] = df['start_at'].dt.minute
            df['second'] = df['start_at'].dt.second
            df['day_of_week'] = df['start_at'].dt.dayofweek
            df['day_of_month'] = df['start_at'].dt.day
            df['month'] = df['start_at'].dt.month
            df['quarter'] = df['start_at'].dt.quarter
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Time-based flags
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_prime_time'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)
            df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
        # ============ SEQUENCE PATTERNS ============
        print("🔢 Sequence pattern analysis...")
        
        # Round-based features
        df['round_mod_10'] = df.index % 10
        df['round_mod_100'] = df.index % 100
        df['round_mod_1000'] = df.index % 1000
        
        # Fibonacci-based features
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        for fib in fib_numbers:
            df[f'round_mod_fib_{fib}'] = df.index % fib
            
        # ============ MARKET MICROSTRUCTURE ============
        if any(col.startswith('bets_') for col in df.columns):
            print("📈 Market microstructure analysis...")
            
            # Order flow imbalance (if we have betting data)
            if 'bets_bet_amount_usd_sum' in df.columns:
                total_volume = df['bets_bet_amount_usd_sum']
                for lag in [1, 2, 3]:
                    df[f'volume_imbalance_lag_{lag}'] = (total_volume - total_volume.shift(lag)).fillna(0)
                    
        # Remove rows with insufficient history
        df = df.iloc[100:].copy()  # Need more history for advanced features
        
        # Clean all data
        print("🧹 Final data cleaning...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace infinite values
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
        # Fill NaN values intelligently
        for col in numeric_cols:
            if col != 'round_id':
                # Use forward fill for time series, then backward fill, then zero
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
        # Clip extreme values
        for col in numeric_cols:
            if col != 'round_id' and not col.startswith('round_mod'):
                q01, q99 = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(q01, q99)
                
        print(f"✅ ULTIMATE FEATURES CREATED: {len(df.columns)} total features, {len(df)} rows")
        return df
        
    def create_prediction_targets(self, df):
        """Create comprehensive prediction targets"""
        print("🎯 CREATING PREDICTION TARGETS...")
        
        targets = {}
        
        if 'crashed_at' in df.columns:
            # Crash level targets
            targets['next_ultra_low'] = (df['crashed_at'] <= 1.05).astype(int)
            targets['next_extreme_low'] = (df['crashed_at'] <= 1.1).astype(int)
            targets['next_low'] = (df['crashed_at'] <= 1.5).astype(int)
            targets['next_medium_low'] = ((df['crashed_at'] > 1.5) & (df['crashed_at'] <= 2.0)).astype(int)
            targets['next_medium'] = ((df['crashed_at'] > 2.0) & (df['crashed_at'] < 5.0)).astype(int)
            targets['next_high'] = (df['crashed_at'] >= 5.0).astype(int)
            targets['next_extreme_high'] = (df['crashed_at'] >= 10.0).astype(int)
            targets['next_mega_high'] = (df['crashed_at'] >= 20.0).astype(int)
            targets['next_ultra_high'] = (df['crashed_at'] >= 50.0).astype(int)
            
            # Volatility targets
            vol_5 = df['crashed_at'].rolling(5).std()
            targets['next_high_volatility'] = (vol_5 > vol_5.quantile(0.8)).astype(int)
            targets['next_low_volatility'] = (vol_5 < vol_5.quantile(0.2)).astype(int)
            
        for name, target in targets.items():
            count = target.sum()
            rate = target.mean() * 100
            print(f"📊 {name}: {count:,} events ({rate:.2f}%)")
            
        return targets
    
    def build_ultimate_models(self, X, y, target_name):
        """Build ensemble of ALL possible models"""
        print(f"🤖 BUILDING ULTIMATE MODEL ENSEMBLE FOR {target_name}...")
        
        models = {}
        model_scores = {}
        
        # Classical ML models
        models['rf'] = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        models['et'] = ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42)
        models['gb'] = GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42)
        models['mlp'] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        
        # Advanced models if available
        if HAS_LGB:
            models['lgb'] = lgb.LGBMClassifier(n_estimators=200, max_depth=10, random_state=42, verbosity=-1)
        if HAS_XGB:
            models['xgb'] = xgb.XGBClassifier(n_estimators=200, max_depth=10, random_state=42, verbosity=0)
            
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models.items():
            try:
                scores = []
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    if y_train.sum() < 5 or y_test.sum() < 2:
                        continue
                        
                    # Scale data
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train and predict
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Score
                    if len(np.unique(y_test)) > 1:
                        auc = roc_auc_score(y_test, y_pred)
                        scores.append(auc)
                
                if scores:
                    avg_score = np.mean(scores)
                    std_score = np.std(scores)
                    model_scores[name] = avg_score
                    print(f"   {name}: AUC = {avg_score:.4f} ± {std_score:.4f}")
                else:
                    model_scores[name] = 0.5
                    
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
                model_scores[name] = 0.5
        
        return model_scores
    
    def ultimate_walk_forward_validation(self, df, targets):
        """The most rigorous validation possible"""
        print("\n🧠 ULTIMATE WALK-FORWARD VALIDATION")
        print("="*70)
        
        # Get absolutely safe features only
        safe_features = []
        forbidden_patterns = ['crashed_at', 'pnl', 'profit_loss', 'winning_players', 'hidden_state']
        
        for col in df.columns:
            # Skip non-numeric columns
            if df[col].dtype == 'object' or df[col].dtype.name.startswith('datetime'):
                continue
                
            is_safe = True
            for pattern in forbidden_patterns:
                if pattern in col.lower() and not col.startswith(('crash_lag_', 'past_')):
                    is_safe = False
                    break
            if is_safe and col not in ['round_id', 'start_at'] and col not in targets.keys():
                safe_features.append(col)
                
        print(f"🔒 SAFE FEATURES: {len(safe_features)}")
        print(f"📋 Feature categories: {len([f for f in safe_features if f.startswith('past_')])} historical, " + 
              f"{len([f for f in safe_features if f.startswith('bets_')])} betting, " +
              f"{len([f for f in safe_features if any(p in f for p in ['hour', 'day', 'weekend'])])} temporal")
        
        X = df[safe_features]
        
        # Ultra-clean data
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Final validation - ensure all columns are numeric
        print("🔍 Final data type validation...")
        numeric_features = []
        for col in X.columns:
            try:
                # Try to convert to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if not X[col].isna().all():  # Skip columns that are all NaN after conversion
                    numeric_features.append(col)
            except:
                print(f"   ⚠️ Skipping non-numeric column: {col}")
                
        X = X[numeric_features].fillna(0)
        print(f"✅ Final numeric features: {len(numeric_features)}")
        
        results = {}
        
        for target_name, y in targets.items():
            if y.sum() < 100:  # Need sufficient samples
                print(f"\n⚠️ SKIPPING {target_name} - insufficient events ({y.sum()})")
                continue
                
            print(f"\n🎯 ULTIMATE VALIDATION: {target_name}")
            print("-" * 50)
            
            # Build and evaluate models
            model_scores = self.build_ultimate_models(X, y, target_name)
            
            if model_scores:
                best_score = max(model_scores.values())
                best_model = max(model_scores, key=model_scores.get)
                
                results[target_name] = {
                    'best_score': best_score,
                    'best_model': best_model,
                    'all_scores': model_scores,
                    'features_used': safe_features
                }
                
                print(f"🏆 BEST: {best_model} with AUC = {best_score:.4f}")
            
        return results

def main():
    print("🌟" * 35)
    print("🌟 ULTIMATE EVERYTHING ANALYZER v34 🌟")
    print("🌟" * 35)
    print("🎯 Target: MAXIMUM POSSIBLE ANALYSIS POWER")
    print("🧠 ALL data + ALL models + ALL features")
    print("📊 If this doesn't find signals, nothing will!")
    print("="*70)
    
    analyzer = UltimateEverythingAnalyzer()
    
    # Load ALL data
    df = analyzer.load_all_data()
    
    # Create ultimate features
    df_ultimate = analyzer.create_ultimate_features(df)
    
    # Create prediction targets
    targets = analyzer.create_prediction_targets(df_ultimate)
    
    # Ultimate validation
    results = analyzer.ultimate_walk_forward_validation(df_ultimate, targets)
    
    # Results summary
    print("\n🏆 ULTIMATE RESULTS")
    print("="*70)
    
    if not results:
        print("❌ NO VALID RESULTS FOUND")
        print("🎲 Game confirmed to be completely random")
        return
    
    # Sort by best score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_score'], reverse=True)
    
    print("📊 ULTIMATE RANKING:")
    ultimate_best_score = 0
    ultimate_best_target = None
    
    for target, data in sorted_results:
        score = data['best_score']
        model = data['best_model']
        
        if score > ultimate_best_score:
            ultimate_best_score = score
            ultimate_best_target = target
        
        if score >= 0.7:
            status = "🚀 GODLIKE"
        elif score >= 0.65:
            status = "🔥 EXCELLENT" 
        elif score >= 0.6:
            status = "⚡ STRONG"
        elif score >= 0.55:
            status = "📈 DECENT"
        else:
            status = "🎲 RANDOM"
            
        print(f"   {status} {target}: {score:.4f} ({model})")
    
    # Final verdict
    print(f"\n🎉 ULTIMATE ANALYSIS COMPLETE!")
    print(f"🏆 ABSOLUTE BEST: {ultimate_best_target} = {ultimate_best_score:.4f}")
    
    if ultimate_best_score >= 0.6:
        print(f"💥 BREAKTHROUGH! Found genuine predictive signal!")
        print(f"💰 This could be profitable for trading!")
    elif ultimate_best_score >= 0.55:
        print(f"🤔 Weak signal detected - might be worth investigating")
    else:
        print(f"🎲 FINAL VERDICT: Game is completely random")
        print(f"✅ No amount of sophistication can predict it")
        print(f"🧠 We tried EVERYTHING possible!")

if __name__ == "__main__":
    main()
