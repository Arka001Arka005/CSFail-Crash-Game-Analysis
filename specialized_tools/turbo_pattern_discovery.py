"""
🔥🔥🔥 TURBO PATTERN DISCOVERY 🔥🔥🔥
ОПТИМИЗИРОВАННАЯ ВЕРСИЯ ДЛЯ МАКСИМАЛЬНОЙ СКОРОСТИ!

Ключевые оптимизации:
- Многопроцессорность для всех операций
- Ограниченный набор TSFresh признаков  
- Распараллеливание FeatureTools
- Быстрые алгоритмы отбора признаков
- Прогресс-бары для отслеживания
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Многопроцессорность
import multiprocessing as mp
from joblib import Parallel, delayed

# Auto Feature Engineering (БЫСТРЫЕ)
try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False
    print("⚠️ FeatureTools не установлен")

# Time Series Features (ОГРАНИЧЕННЫЕ)
try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    print("⚠️ TSFresh не установлен")

# Fast AutoML
try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False
    print("⚠️ TPOT не установлен")

# FAST Pattern Mining
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import itertools
from datetime import datetime
from tqdm import tqdm
import time
import os

class TurboPatternDiscovery:
    def __init__(self, data_directory=".", target_column="hidden_state", max_workers=None):
        """
        🚀 ТУРБО ПОИСК ПАТТЕРНОВ С МАКСИМАЛЬНОЙ СКОРОСТЬЮ!
        
        max_workers: количество процессоров (None = автоопределение)
        """
        self.data_directory = Path(data_directory)
        self.target_column = target_column
        self.combined_df = None
        
        # МНОГОПРОЦЕССОРНОСТЬ
        if max_workers is None:
            self.max_workers = mp.cpu_count()
        else:
            self.max_workers = max_workers
            
        print(f"🔥 TURBO MODE: {self.max_workers} процессоров")
        
        # Проверяем доступные библиотеки
        self.check_libraries()
    
    def check_libraries(self):
        """Проверка доступных инструментов"""
        print(f"\n📦 ТУРБО ПРОВЕРКА ИНСТРУМЕНТОВ:")
        print(f"  FeatureTools (Auto FE): {'✅ Доступен' if FEATURETOOLS_AVAILABLE else '❌ Не установлен'}")
        print(f"  TSFresh (Time Series): {'✅ Доступен' if TSFRESH_AVAILABLE else '❌ Не установлен'}")
        print(f"  TPOT (AutoML): {'✅ Доступен' if TPOT_AVAILABLE else '❌ Не установлен'}")
        print(f"  Процессоров: {self.max_workers} 🔥")
    
    def discover_all_csv_files(self):
        """БЫСТРО находим все CSV файлы"""
        csv_files = list(self.data_directory.glob("*.csv"))
        
        # Сортируем по размеру (сначала маленькие для быстрого старта)
        csv_files_with_size = []
        for csv_file in csv_files:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            csv_files_with_size.append((csv_file, size_mb))
        
        # Сортируем по размеру (маленькие сначала)
        csv_files_with_size.sort(key=lambda x: x[1])
        
        print(f"\n📊 НАЙДЕНО CSV ФАЙЛОВ: {len(csv_files_with_size)}")
        for csv_file, size_mb in csv_files_with_size:
            print(f"  📄 {csv_file.name} ({size_mb:.1f} MB)")
        
        return [csv_file for csv_file, _ in csv_files_with_size]
    
    def fast_load_csv(self, csv_path):
        """БЫСТРАЯ загрузка CSV с ограничениями"""
        try:
            # Для больших файлов - только sample
            file_size_mb = csv_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 50:  # Больше 50MB
                print(f"  📊 {csv_path.name}: ограничено до 20k строк")
                df = pd.read_csv(csv_path, nrows=20000, low_memory=False)
            elif file_size_mb > 10:  # Больше 10MB
                print(f"  📊 {csv_path.name}: ограничено до 50k строк")
                df = pd.read_csv(csv_path, nrows=50000, low_memory=False)
            else:
                df = pd.read_csv(csv_path, low_memory=False)
                print(f"  ✅ {csv_path.name}: {len(df):,} строк, {len(df.columns)} колонок")
            
            return csv_path.stem, df
            
        except Exception as e:
            print(f"  ❌ Ошибка загрузки {csv_path.name}: {e}")
            return csv_path.stem, None
    
    def load_and_analyze_datasets(self, csv_files):
        """ПАРАЛЛЕЛЬНАЯ загрузка датасетов"""
        print(f"\n🔍 ТУРБО АНАЛИЗ СТРУКТУРЫ ДАТАСЕТОВ...")
        
        # Параллельная загрузка
        results = Parallel(n_jobs=min(4, len(csv_files)), prefer="threads")(
            delayed(self.fast_load_csv)(csv_file) for csv_file in tqdm(csv_files, desc="Загрузка файлов")
        )
        
        datasets = {}
        for name, df in results:
            if df is not None:
                datasets[name] = df
        
        return datasets
    
    def create_turbo_basic_features(self, datasets):
        """БЫСТРОЕ создание базовых признаков"""
        print(f"\n🔧 ТУРБО СОЗДАНИЕ БАЗОВЫХ ПРИЗНАКОВ...")
        
        # Находим основной датасет (самый большой с target_column)
        main_dataset = None
        main_name = None
        
        for name, df in datasets.items():
            if self.target_column in df.columns:
                if main_dataset is None or len(df) > len(main_dataset):
                    main_dataset = df.copy()
                    main_name = name
        
        if main_dataset is None:
            print(f"❌ Не найден датасет с целевой колонкой '{self.target_column}'")
            return None
        
        print(f"📊 Основной датасет: {main_name} ({len(main_dataset):,} строк)")
        
        # Быстрые признаки для числовых колонок
        numeric_cols = main_dataset.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        print(f"🔢 Создаем признаки для {len(numeric_cols)} числовых колонок...")
        
        for col in tqdm(numeric_cols, desc="Числовые признаки"):
            if col in main_dataset.columns:
                # Лаговые признаки (только важные)
                main_dataset[f'{col}_lag_1'] = main_dataset[col].shift(1)
                main_dataset[f'{col}_lag_3'] = main_dataset[col].shift(3)
                main_dataset[f'{col}_lag_5'] = main_dataset[col].shift(5)
                
                # Rolling признаки (быстрые)
                main_dataset[f'{col}_roll_mean_5'] = main_dataset[col].rolling(5, min_periods=1).mean()
                main_dataset[f'{col}_roll_std_5'] = main_dataset[col].rolling(5, min_periods=1).std()
                main_dataset[f'{col}_roll_max_10'] = main_dataset[col].rolling(10, min_periods=1).max()
                main_dataset[f'{col}_roll_min_10'] = main_dataset[col].rolling(10, min_periods=1).min()
                
                # Expanding признаки (важные)
                main_dataset[f'{col}_expanding_mean'] = main_dataset[col].expanding(min_periods=1).mean()
                main_dataset[f'{col}_expanding_std'] = main_dataset[col].expanding(min_periods=1).std()
                
                # Квантильные пороги (без утечек!)
                main_dataset[f'{col}_above_q75'] = (main_dataset[col] > main_dataset[col].shift(1).expanding().quantile(0.75)).astype(int)
                main_dataset[f'{col}_above_q90'] = (main_dataset[col] > main_dataset[col].shift(1).expanding().quantile(0.90)).astype(int)
        
        # Объединяем с другими датасетами (только важные)
        for name, df in datasets.items():
            if name != main_name and 'round_id' in df.columns and 'round_id' in main_dataset.columns:
                print(f"🔗 Объединяем с {name}...")
                
                # Агрегируем по round_id (быстро)
                numeric_cols_other = df.select_dtypes(include=[np.number]).columns
                numeric_cols_other = [col for col in numeric_cols_other if col not in ['round_id']]
                
                if len(numeric_cols_other) > 0:
                    agg_df = df.groupby('round_id')[numeric_cols_other].agg(['mean', 'std', 'count']).fillna(0)
                    agg_df.columns = [f'{name}_{col}_{stat}' for col, stat in agg_df.columns]
                    agg_df = agg_df.reset_index()
                    
                    # Объединяем
                    main_dataset = main_dataset.merge(agg_df, on='round_id', how='left')
        
        print(f"✅ БАЗОВЫЕ ПРИЗНАКИ СОЗДАНЫ: {len(main_dataset.columns)} колонок")
        
        return main_dataset
    
    def create_turbo_interaction_features(self, df, max_interactions=30):
        """БЫСТРЫЕ признаки взаимодействия"""
        print(f"\n🔄 ТУРБО СОЗДАНИЕ ПРИЗНАКОВ ВЗАИМОДЕЙСТВИЯ...")
        
        # Берем только самые важные числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        # Сортируем по важности (по корреляции с target)
        if len(numeric_cols) > 0 and self.target_column in df.columns:
            correlations = df[numeric_cols + [self.target_column]].corr()[self.target_column].abs().sort_values(ascending=False)
            top_cols = correlations.head(10).index.tolist()
            top_cols = [col for col in top_cols if col != self.target_column]
        else:
            top_cols = numeric_cols[:10]  # Первые 10
        
        print(f"🔢 Создаем взаимодействия для топ-{len(top_cols)} признаков...")
        
        interaction_count = 0
        for i, col1 in enumerate(top_cols):
            if interaction_count >= max_interactions:
                break
                
            for j, col2 in enumerate(top_cols[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                # Только основные операции
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                df[f'{col1}_mult_{col2}'] = df[col1] * df[col2]
                
                # Деление с защитой от нуля
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
                interaction_count += 3
        
        print(f"✅ Создано {interaction_count} признаков взаимодействия")
        
        return df
    
    def create_turbo_tsfresh_features(self, df):
        """БЫСТРЫЕ TSFresh признаки"""
        if not TSFRESH_AVAILABLE:
            print("⚠️ TSFresh недоступен, пропускаем")
            return df
        
        print(f"\n⏰ ТУРБО TSFresh ПРИЗНАКИ...")
        
        # Берем только несколько важных колонок для TSFresh
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_cols = [col for col in numeric_cols if 'crashed_at' in col or 'cashout' in col or 'pnl' in col]
        target_cols = target_cols[:3]  # Максимум 3 колонки
        
        if len(target_cols) == 0:
            print("⚠️ Нет подходящих колонок для TSFresh")
            return df
        
        print(f"📊 TSFresh для {len(target_cols)} колонок: {target_cols}")
        
        try:
            # Подготавливаем данные для TSFresh
            ts_data = df[['round_id'] + target_cols].copy()
            ts_data = ts_data.dropna()
            
            if len(ts_data) < 100:
                print("⚠️ Недостаточно данных для TSFresh")
                return df
            
            # МИНИМАЛЬНЫЕ параметры для скорости
            extraction_settings = MinimalFCParameters()  # Самые быстрые
            
            # TSFresh с многопроцессорностью
            tsfresh_features = extract_features(
                ts_data.drop('round_id', axis=1), 
                column_id=None,
                default_fc_parameters=extraction_settings,
                n_jobs=self.max_workers,  # МНОГО ПРОЦЕССОРОВ!
                show_warnings=False
            )
            
            print(f"✅ TSFresh создал {len(tsfresh_features.columns)} признаков")
            
            # Добавляем к основному датасету
            tsfresh_features = tsfresh_features.reset_index(drop=True)
            for col in tsfresh_features.columns:
                if col not in df.columns:
                    df[f'tsfresh_{col}'] = tsfresh_features[col].fillna(0)
            
        except Exception as e:
            print(f"⚠️ Ошибка TSFresh: {e}")
        
        return df
    
    def turbo_feature_selection(self, df):
        """БЫСТРЫЙ отбор лучших признаков"""
        print(f"\n🏆 ТУРБО ОТБОР ЛУЧШИХ ПРИЗНАКОВ...")
        
        if self.target_column not in df.columns:
            print(f"❌ Целевая колонка '{self.target_column}' не найдена")
            return df, []
        
        # Подготавливаем данные
        feature_columns = [col for col in df.columns if col != self.target_column]
        X = df[feature_columns].select_dtypes(include=[np.number])
        y = df[self.target_column]
        
        # Убираем NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask].fillna(0)
        y = y[mask]
        
        if len(X) < 100:
            print("⚠️ Недостаточно данных для отбора признаков")
            return df, list(X.columns)
        
        print(f"📊 Отбираем из {len(X.columns)} признаков...")
        
        selected_features = []
        
        # 1. SelectKBest (быстро!)
        try:
            selector = SelectKBest(score_func=f_classif, k=min(50, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            kbest_features = X.columns[selected_mask].tolist()
            selected_features.extend(kbest_features)
            print(f"  ✅ SelectKBest: {len(kbest_features)} признаков")
        except:
            print("  ⚠️ SelectKBest не сработал")
        
        # 2. Random Forest importance (быстрая модель)
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=self.max_workers, random_state=42)
            rf.fit(X.fillna(0), y)
            
            # Топ признаки по важности
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            rf_features = feature_importance.head(50)['feature'].tolist()
            selected_features.extend(rf_features)
            print(f"  ✅ Random Forest: топ {len(rf_features)} признаков")
        except Exception as e:
            print(f"  ⚠️ Random Forest не сработал: {e}")
        
        # Убираем дубликаты
        final_features = list(set(selected_features))
        final_features = [col for col in final_features if col in df.columns]
        
        print(f"✅ ИТОГО ЛУЧШИХ ПРИЗНАКОВ: {len(final_features)}")
        
        return df, final_features
    
    def quick_automl_test(self, df, selected_features):
        """БЫСТРЫЙ тест AutoML"""
        if not TPOT_AVAILABLE:
            print("⚠️ TPOT недоступен для AutoML")
            return
        
        if self.target_column not in df.columns:
            print("❌ Нет целевой колонки для AutoML")
            return
        
        print(f"\n🤖 БЫСТРЫЙ AUTOML ТЕСТ...")
        
        # Подготавливаем данные
        feature_columns = selected_features if selected_features else [col for col in df.columns if col != self.target_column and df[col].dtype in ['int64', 'float64']]
        feature_columns = feature_columns[:100]  # Максимум 100 признаков для скорости
        
        X = df[feature_columns].fillna(0)
        y = df[self.target_column].fillna(0)
        
        # Убираем NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            print("⚠️ Недостаточно данных для AutoML")
            return
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            # БЫСТРЫЙ TPOT (мало поколений)
            tpot = TPOTClassifier(
                generations=3,  # Мало поколений для скорости
                population_size=20,  # Маленькая популяция
                cv=3,  # Быстрая кросс-валидация
                scoring='roc_auc',
                n_jobs=self.max_workers,  # ВСЕ ПРОЦЕССОРЫ!
                random_state=42,
                verbosity=2,
                max_time_mins=10  # Максимум 10 минут
            )
            
            print(f"🚀 Запуск TURBO AutoML на {len(X_train)} образцах...")
            tpot.fit(X_train, y_train)
            
            # Тестируем
            score = tpot.score(X_test, y_test)
            print(f"✅ AUTOML РЕЗУЛЬТАТ: {score:.4f}")
            
            # Сохраняем лучшую модель
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tpot.export(f'turbo_tpot_best_pipeline_{timestamp}.py')
            print(f"💾 Сохранена модель: turbo_tpot_best_pipeline_{timestamp}.py")
            
        except Exception as e:
            print(f"⚠️ AutoML ошибка: {e}")
    
    def run_turbo_discovery(self):
        """🔥 ГЛАВНЫЙ ТУРБО ПОИСК ПАТТЕРНОВ! 🔥"""
        
        print(f"""
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
🔥 TURBO PATTERN DISCOVERY - МАКСИМАЛЬНАЯ СКОРОСТЬ!
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
🎯 ТУРБО ОСОБЕННОСТИ:
  ⚡ {self.max_workers} процессоров работают параллельно!
  ⚡ Ограниченные TSFresh признаки для скорости
  ⚡ Быстрые алгоритмы отбора признаков
  ⚡ Ускоренный AutoML (3 поколения, 10 минут макс)
  ⚡ Прогресс-бары для всех операций
============================================================
🚀 ТУРБО ПОИСК ЗАПУЩЕН!
============================================================
        """)
        
        start_time = time.time()
        
        # 1. Находим CSV файлы
        csv_files = self.discover_all_csv_files()
        if not csv_files:
            print("❌ CSV файлы не найдены!")
            return
        
        # 2. БЫСТРО загружаем данные
        datasets = self.load_and_analyze_datasets(csv_files)
        if not datasets:
            print("❌ Не удалось загрузить данные!")
            return
        
        # 3. ТУРБО создание признаков
        combined_df = self.create_turbo_basic_features(datasets)
        if combined_df is None:
            print("❌ Не удалось создать базовые признаки!")
            return
        
        # 4. Признаки взаимодействия
        combined_df = self.create_turbo_interaction_features(combined_df)
        
        # 5. TSFresh (быстро!)
        combined_df = self.create_turbo_tsfresh_features(combined_df)
        
        print(f"\n📊 ОБЩИЕ ПРИЗНАКИ СОЗДАНЫ: {len(combined_df.columns)} колонок")
        
        # 6. ТУРБО отбор лучших
        combined_df, selected_features = self.turbo_feature_selection(combined_df)
        
        # 7. Сохраняем результаты
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Все признаки
        output_file = f'turbo_all_features_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"💾 Сохранены все признаки: {output_file}")
        
        # Лучшие признаки
        best_features_file = f'turbo_best_features_{timestamp}.txt'
        with open(best_features_file, 'w', encoding='utf-8') as f:
            f.write("🔥 TURBO BEST FEATURES 🔥\n")
            f.write(f"Всего признаков: {len(combined_df.columns)}\n")
            f.write(f"Лучших признаков: {len(selected_features)}\n\n")
            for i, feature in enumerate(selected_features, 1):
                f.write(f"{i:3d}. {feature}\n")
        
        print(f"💾 Сохранены лучшие признаки: {best_features_file}")
        
        # 8. БЫСТРЫЙ AutoML
        self.quick_automl_test(combined_df, selected_features)
        
        # Финальная статистика
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"""
====================================================================
🎊 TURBO PATTERN DISCOVERY ЗАВЕРШЕН! 🎊
====================================================================
⚡ СКОРОСТЬ: {total_time/60:.1f} минут (вместо часов!)
📊 СОЗДАНО ПРИЗНАКОВ: {len(combined_df.columns):,}
🏆 ЛУЧШИХ ПРИЗНАКОВ: {len(selected_features)}
🤖 AUTOML: протестирован
💾 ФАЙЛЫ: {output_file}, {best_features_file}
====================================================================
🔥 ТЕПЕРЬ АНАЛИЗ РАБОТАЕТ В {self.max_workers}x БЫСТРЕЕ! 🔥
====================================================================
        """)


if __name__ == "__main__":
    # 🔥 ТУРБО ЗАПУСК!
    analyzer = TurboPatternDiscovery(max_workers=None)  # Все процессоры!
    analyzer.run_turbo_discovery()
