import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SmartCrashAnalyzer:
    def __init__(self):
        print("🧠 SMART CRASH ANALYZER ЗАПУЩЕН!")
        print("🎯 УМНЫЙ ПОДХОД К АНАЛИЗУ КРАШ-ИГРЫ")
        print("💡 УЧИТЫВАЕТ ДИСБАЛАНС КЛАССОВ И РЕАЛЬНОСТЬ")
        print("=" * 60)
    
    def load_and_analyze(self):
        """Загрузка и первичный анализ"""
        print("📊 Загрузка данных...")
        self.df = pd.read_csv('analytics_dataset_v3_players_41k.csv')
        print(f"✅ Загружено: {len(self.df):,} записей")
        
        # Анализ распределения
        hidden_counts = self.df['hidden_state'].value_counts().sort_index()
        print(f"\n🎯 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
        for state, count in hidden_counts.items():
            percent = count / len(self.df) * 100
            print(f"  State {state}: {count:,} ({percent:.3f}%)")
        
        # Определяем стратегию
        state_2_count = hidden_counts.get(2, 0)
        if state_2_count < 100:
            print(f"\n💡 УМНАЯ СТРАТЕГИЯ:")
            print(f"  State 2 слишком редкий ({state_2_count} примеров)")
            print(f"  ✅ Переходим к анализу State 0 vs State 1")
            print(f"  ✅ Дополнительно: поиск аномальных крашей")
            return "binary_with_anomaly_detection"
        else:
            return "multiclass"
    
    def create_smart_features(self):
        """Создает умные признаки без утечек"""
        print("\n🔧 СОЗДАНИЕ УМНЫХ ПРИЗНАКОВ...")
        
        df = self.df.copy()
        
        # Временные признаки
        try:
            df['start_at'] = pd.to_datetime(df['start_at'], format='mixed')
        except:
            df['start_at'] = pd.to_datetime(df['start_at'], errors='coerce')
        
        df['hour'] = df['start_at'].dt.hour
        df['day_of_week'] = df['start_at'].dt.dayofweek
        
        # ЧЕСТНЫЕ лаговые признаки
        for lag in [1, 2, 3, 5]:
            df[f'crash_lag_{lag}'] = df['crashed_at'].shift(lag)
            df[f'users_lag_{lag}'] = df['users_count'].shift(lag)
            df[f'bank_lag_{lag}'] = df['total_bank_usd'].shift(lag)
        
        # ЧЕСТНЫЕ rolling статистики (только прошлые данные)
        for window in [5, 10]:
            df[f'crash_mean_{window}'] = df['crashed_at'].shift(1).rolling(window).mean()
            df[f'crash_std_{window}'] = df['crashed_at'].shift(1).rolling(window).std()
            df[f'users_mean_{window}'] = df['users_count'].shift(1).rolling(window).mean()
        
        # Исторические пороги (expanding, только прошлые)
        df['historical_q25'] = df['crashed_at'].shift(1).expanding().quantile(0.25)
        df['historical_q75'] = df['crashed_at'].shift(1).expanding().quantile(0.75)
        df['historical_q90'] = df['crashed_at'].shift(1).expanding().quantile(0.90)
        
        # Дополнительные честные фичи
        df['profit_per_user'] = df['profit_loss'] / (df['users_count'] + 1)
        df['bank_per_user'] = df['total_bank_usd'] / (df['users_count'] + 1)
        
        # Удаляем NaN
        df_clean = df.dropna()
        print(f"✅ Создано {len(df_clean.columns)} признаков, {len(df_clean):,} строк")
        
        return df_clean
    
    def binary_analysis(self, df):
        """Бинарный анализ State 0 vs State 1"""
        print("\n🎯 БИНАРНЫЙ АНАЛИЗ: State 0 vs State 1")
        
        # Фильтруем только State 0 и 1
        binary_df = df[df['hidden_state'].isin([0, 1])].copy()
        print(f"📊 Данные для анализа: {len(binary_df):,} записей")
        
        # Подготовка данных
        feature_cols = [col for col in binary_df.columns 
                       if col not in ['hidden_state', 'round_id', 'start_at']]
        
        X = binary_df[feature_cols]
        y = binary_df['hidden_state']
        
        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Модель
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf.fit(X_train_scaled, y_train)
        
        # Оценка
        y_pred = rf.predict(X_test_scaled)
        y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📈 РЕЗУЛЬТАТЫ БИНАРНОЙ КЛАССИФИКАЦИИ:")
        print(f"🎯 AUC: {auc:.4f}")
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
        for i, (idx, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"  {i:2}. {row['feature']}: {row['importance']:.4f}")
        
        return auc, feature_importance
    
    def anomaly_detection_analysis(self, df):
        """Анализ аномалий с помощью Isolation Forest"""
        print("\n🔍 АНАЛИЗ АНОМАЛЬНЫХ КРАШЕЙ:")
        
        # Используем только краши для поиска аномалий
        crash_features = ['crashed_at', 'total_bank_usd', 'users_count', 'profit_loss']
        X_anomaly = df[crash_features].dropna()
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.05,  # 5% аномалий
            random_state=42,
            n_jobs=-1
        )
        
        anomaly_pred = iso_forest.fit_predict(X_anomaly)
        anomaly_scores = iso_forest.decision_function(X_anomaly)
        
        # Анализ найденных аномалий
        anomalies = X_anomaly[anomaly_pred == -1]
        normal = X_anomaly[anomaly_pred == 1]
        
        print(f"🔍 Найдено аномалий: {len(anomalies):,} из {len(X_anomaly):,}")
        print(f"📊 Процент аномалий: {len(anomalies)/len(X_anomaly)*100:.2f}%")
        
        print(f"\n📈 СТАТИСТИКА АНОМАЛЬНЫХ КРАШЕЙ:")
        print(f"  Средний краш (норма): {normal['crashed_at'].mean():.2f}x")
        print(f"  Средний краш (аномалии): {anomalies['crashed_at'].mean():.2f}x")
        print(f"  Медиана (норма): {normal['crashed_at'].median():.2f}x")
        print(f"  Медиана (аномалии): {anomalies['crashed_at'].median():.2f}x")
        
        # Топ аномальных крашей
        anomaly_df = df.iloc[X_anomaly.index[anomaly_pred == -1]].copy()
        anomaly_df['anomaly_score'] = anomaly_scores[anomaly_pred == -1]
        top_anomalies = anomaly_df.nsmallest(10, 'anomaly_score')
        
        print(f"\n🏆 ТОП-10 САМЫХ АНОМАЛЬНЫХ РАУНДОВ:")
        for i, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
            print(f"  {i:2}. Round {row['round_id']}: {row['crashed_at']:.2f}x "
                  f"(score: {row['anomaly_score']:.3f})")
        
        return anomalies, normal, top_anomalies
    
    def crash_pattern_analysis(self, df):
        """Анализ паттернов крашей"""
        print("\n📊 АНАЛИЗ ПАТТЕРНОВ КРАШЕЙ:")
        
        crashes = df['crashed_at']
        
        # Основные статистики
        print(f"📈 ОСНОВНЫЕ СТАТИСТИКИ:")
        print(f"  Среднее: {crashes.mean():.2f}x")
        print(f"  Медиана: {crashes.median():.2f}x")
        print(f"  Стандартное отклонение: {crashes.std():.2f}x")
        print(f"  Минимум: {crashes.min():.2f}x")
        print(f"  Максимум: {crashes.max():.2f}x")
        
        # Квантили
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        print(f"\n📊 КВАНТИЛИ:")
        for q in quantiles:
            value = crashes.quantile(q)
            print(f"  Q{q*100:2.0f}: {value:.2f}x")
        
        # Распределение по диапазонам
        bins = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, float('inf')]
        labels = ['1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0', '5.0-10.0', '10.0-25.0', '25.0+']
        
        crash_ranges = pd.cut(crashes, bins=bins, labels=labels, right=False)
        range_counts = crash_ranges.value_counts().sort_index()
        
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО ДИАПАЗОНАМ:")
        for range_name, count in range_counts.items():
            percent = count / len(crashes) * 100
            print(f"  {range_name}: {count:,} ({percent:.1f}%)")
    
    def run_complete_analysis(self):
        """Запускает полный умный анализ"""
        print("🚀 ЗАПУСК ПОЛНОГО УМНОГО АНАЛИЗА...")
        
        # 1. Определяем стратегию
        strategy = self.load_and_analyze()
        
        # 2. Создаем умные фичи
        df_features = self.create_smart_features()
        
        # 3. Анализ паттернов крашей
        self.crash_pattern_analysis(df_features)
        
        # 4. Бинарный анализ
        auc, feature_importance = self.binary_analysis(df_features)
        
        # 5. Поиск аномалий
        anomalies, normal, top_anomalies = self.anomaly_detection_analysis(df_features)
        
        # 6. Итоговые выводы
        print(f"\n🎊 ИТОГОВЫЕ ВЫВОДЫ:")
        print(f"📊 Бинарная классификация AUC: {auc:.4f}")
        if auc < 0.55:
            print(f"  ✅ Результат близок к случайности - игра честная!")
        elif auc < 0.65:
            print(f"  🤔 Слабая предсказуемость - возможны скрытые паттерны")
        else:
            print(f"  ⚠️  Высокая предсказуемость - проверьте на утечки данных!")
        
        print(f"🔍 Найдено аномальных крашей: {len(anomalies):,}")
        print(f"📈 Средний аномальный краш: {anomalies['crashed_at'].mean():.2f}x")
        
        # Сохранение результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_importance.to_csv(f'smart_analysis_features_{timestamp}.csv', index=False)
        top_anomalies.to_csv(f'top_anomalies_{timestamp}.csv', index=False)
        
        print(f"\n💾 Результаты сохранены с timestamp: {timestamp}")
        
        return {
            'auc': auc,
            'anomalies_count': len(anomalies),
            'feature_importance': feature_importance,
            'top_anomalies': top_anomalies
        }

def main():
    print("🧠" * 30)
    print("🧠 SMART CRASH ANALYZER")
    print("🧠" * 30)
    print("💡 ОСОБЕННОСТИ:")
    print("  ✅ Учитывает дисбаланс классов")
    print("  ✅ Умная стратегия анализа")
    print("  ✅ Честная оценка без утечек")
    print("  ✅ Поиск аномальных крашей")
    print("  ✅ Детальный анализ паттернов")
    print("  ✅ Практичные выводы")
    print("=" * 60)
    
    analyzer = SmartCrashAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\n🎯 АНАЛИЗ ЗАВЕРШЕН!")
    return results

if __name__ == "__main__":
    main()
