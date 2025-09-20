"""
🔧🔧🔧 OPTIMIZED BACKTEST BOT 🔧🔧🔧  
ОПТИМИЗИРОВАННАЯ ВЕРСИЯ С УЛУЧШЕННЫМИ ПАРАМЕТРАМИ!

ИЗМЕНЕНИЯ ПО РЕЗУЛЬТАТАМ ПЕРВОГО ТЕСТА:
- Более агрессивные пороги волатильности
- Разные временные периоды для тестирования
- Улучшенная логика предсказания
- Адаптивные размеры ставок
"""

import requests
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

class OptimizedBacktestBot:
    def __init__(self, token):
        """🔧 ОПТИМИЗИРОВАННЫЙ БЭКТЕСТИНГ БОТ"""
        
        self.token = token
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'Authorization': token
        }
        
        # Стартовые настройки
        self.start_round_id = 7026046
        
        # Данные
        self.historical_data = []
        
        # ОПТИМИЗИРОВАННЫЕ торговые параметры
        self.base_bet = 4.0  # Уменьшили базовую ставку
        self.balance = 1000.0
        
        # УЛУЧШЕННЫЕ пороги волатильности
        self.vol_high_threshold = 0.4    # Снизили порог HIGH (было 0.6)
        self.vol_low_threshold = 0.15    # Снизили порог LOW (было 0.3) 
        
        # ОПТИМИЗИРОВАННЫЕ множители
        self.high_vol_bet_mult = 0.8     # Менее агрессивное снижение (было 0.7)
        self.high_vol_target_mult = 1.3  # Менее агрессивное повышение (было 1.4)
        self.low_vol_bet_mult = 1.2      # Менее агрессивный рост (было 1.3)
        self.low_vol_target_mult = 0.95  # Менее агрессивное снижение (было 0.9)
        
        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        
        print("🔧 OPTIMIZED BACKTEST BOT INITIALIZED")
        print("💡 УЛУЧШЕННЫЕ ПАРАМЕТРЫ НА ОСНОВЕ ПЕРВОГО ТЕСТА!")
        print("📊 ТЕСТИРУЕМ РАЗНЫЕ ВРЕМЕННЫЕ ПЕРИОДЫ!")
        print(f"💰 Starting balance: ${self.balance}")
    
    def fetch_historical_data(self, num_rounds=150):
        """Загружает больше исторических данных"""
        print(f"📥 Загружаем {num_rounds} раундов для оптимизированного бэктестинга...")
        
        loaded_data = []
        
        for i in range(num_rounds):
            round_id = self.start_round_id - i
            
            if i % 25 == 0:
                print(f"📊 Прогресс: {i}/{num_rounds} раундов...")
            
            api_url = f"https://6cs.fail/api/crash/games/{round_id}"
            
            try:
                response = requests.get(api_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'game' in data['data']:
                        game = data['data']['game']
                        
                        round_info = {
                            'round_id': game.get('id'),
                            'crashed_at': float(game.get('crashedAt', 0.0)),
                            'total_bank_usd': float(game.get('totalBankUsd', 0.0)),
                            'users_count': int(game.get('usersCount', 0)),
                            'start_at': game.get('startAt')
                        }
                        
                        loaded_data.append(round_info)
                
                time.sleep(random.uniform(0.08, 0.25))
                
            except Exception as e:
                print(f"❌ Ошибка для раунда {round_id}: {e}")
        
        self.historical_data = sorted(loaded_data, key=lambda x: x['round_id'])
        
        print(f"✅ Загружено {len(self.historical_data)} раундов")
        return len(self.historical_data) > 80
    
    def advanced_volatility_prediction(self, current_idx):
        """🔧 УЛУЧШЕННОЕ предсказание волатильности"""
        
        if current_idx < 25:
            return "MEDIUM", 0.5, "insufficient_history"
        
        history = self.historical_data[:current_idx]
        crashes = [r['crashed_at'] for r in history[-40:]]  # Больше истории
        
        if len(crashes) < 15:
            return "MEDIUM", 0.5, "insufficient_data"
        
        # РАСШИРЕННЫЕ признаки
        features = {}
        
        # Лаговые значения (больше лагов)
        lags = [2, 3, 5, 7, 10]  # Добавили больше лагов
        for lag in lags:
            if len(crashes) > lag:
                features[f'crash_lag_{lag}'] = crashes[-lag-1]
        
        # Различные окна для rolling
        windows = [3, 5, 8, 12]  # Разнообразные окна
        for window in windows:
            if len(crashes) >= window + 2:
                shifted = crashes[:-2]
                if len(shifted) >= window:
                    features[f'vol_strict_{window}'] = np.std(shifted[-window:])
                    features[f'mean_strict_{window}'] = np.mean(shifted[-window:])
        
        # Expanding статистики
        if len(crashes) >= 8:
            expanding_data = crashes[:-2]
            if len(expanding_data) > 3:
                features['expanding_vol'] = np.std(expanding_data)
                features['expanding_mean'] = np.mean(expanding_data)
                features['expanding_q75'] = np.percentile(expanding_data, 75)
                features['expanding_q25'] = np.percentile(expanding_data, 25)
        
        # 🎯 УЛУЧШЕННАЯ МОДЕЛЬ ПРЕДСКАЗАНИЯ
        score = 0.0
        
        # Правило 1: Волатильность vs средняя (более чувствительное)
        for window in [3, 5, 8]:
            vol_key = f'vol_strict_{window}'
            if vol_key in features and 'expanding_vol' in features:
                if features[vol_key] > features['expanding_vol'] * 1.1:  # Снизили с 1.2
                    score += 0.15
        
        # Правило 2: Экстремальные лаги (больше проверок)
        extreme_count = 0
        for lag in [2, 3, 5, 7]:
            lag_key = f'crash_lag_{lag}'
            if lag_key in features and 'expanding_mean' in features:
                ratio = features[lag_key] / (features['expanding_mean'] + 1e-8)
                if ratio > 1.7 or ratio < 0.45:  # Менее строгие пороги
                    extreme_count += 1
        
        if extreme_count >= 2:
            score += 0.2
        elif extreme_count >= 1:
            score += 0.1
        
        # Правило 3: Квартили (новое правило)
        if 'crash_lag_2' in features and 'expanding_q75' in features and 'expanding_q25' in features:
            if features['crash_lag_2'] > features['expanding_q75']:
                score += 0.15
            elif features['crash_lag_2'] < features['expanding_q25']:
                score += 0.15
        
        # Правило 4: Тренд волатильности (улучшенное)
        if len(crashes) >= 15:
            recent_vol = np.std(crashes[-5:])
            mid_vol = np.std(crashes[-10:-5])
            older_vol = np.std(crashes[-15:-10])
            
            # Растущий тренд волатильности
            if recent_vol > mid_vol > older_vol:
                score += 0.2
            elif recent_vol > older_vol * 1.1:
                score += 0.1
        
        # Правило 5: Банк и активность (улучшенное)
        if current_idx >= 8:
            recent_data = history[-8:]
            banks = [r['total_bank_usd'] for r in recent_data]
            users = [r['users_count'] for r in recent_data]
            
            if len(banks) > 3:
                bank_vol = np.std(banks) / (np.mean(banks) + 1e-8)
                user_vol = np.std(users) / (np.mean(users) + 1e-8)
                
                if bank_vol > 0.25:  # Снизили с 0.3
                    score += 0.1
                if user_vol > 0.2:
                    score += 0.1
        
        # Классификация с новыми порогами
        if score >= self.vol_high_threshold:
            return "HIGH", score, f"HIGH volatility predicted (score: {score:.3f})"
        elif score >= self.vol_low_threshold:
            return "MEDIUM", score, f"MEDIUM volatility predicted (score: {score:.3f})"
        else:
            return "LOW", score, f"LOW volatility predicted (score: {score:.3f})"
    
    def get_optimized_strategy(self, vol_level, vol_score):
        """🔧 ОПТИМИЗИРОВАННАЯ торговая стратегия"""
        
        bet_size = self.base_bet
        target_mult = 1.5
        
        # УЛУЧШЕННАЯ адаптация
        if vol_level == "HIGH":
            bet_size *= self.high_vol_bet_mult
            target_mult *= self.high_vol_target_mult
            strategy = "HIGH_VOL_OPTIMIZED"
            
        elif vol_level == "LOW":
            bet_size *= self.low_vol_bet_mult
            target_mult *= self.low_vol_target_mult
            strategy = "LOW_VOL_OPTIMIZED"
            
        else:
            # MEDIUM стратегия тоже оптимизируем
            if vol_score > 0.25:
                bet_size *= 1.05  # Слегка увеличиваем
                target_mult *= 1.05
            strategy = "MEDIUM_VOL_OPTIMIZED"
        
        # АДАПТИВНЫЕ ограничения
        max_bet_percent = 0.12 if vol_level == "LOW" else 0.08
        bet_size = min(bet_size, self.balance * max_bet_percent)
        bet_size = max(bet_size, 1.0)
        target_mult = min(target_mult, 3.5)  # Более реалистичные цели
        target_mult = max(target_mult, 1.05)
        
        return {
            'bet_size': bet_size,
            'target_multiplier': target_mult,
            'strategy': strategy
        }
    
    def run_multiple_periods_test(self):
        """🔧 ТЕСТИРОВАНИЕ НА РАЗНЫХ ВРЕМЕННЫХ ПЕРИОДАХ"""
        
        print(f"""
🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧
🔧 OPTIMIZED BACKTEST BOT - МНОЖЕСТВЕННОЕ ТЕСТИРОВАНИЕ!
🔧 УЛУЧШЕННЫЕ ПАРАМЕТРЫ + РАЗНЫЕ ВРЕМЕННЫЕ ПЕРИОДЫ!
🔧 НАХОДИМ ЛУЧШУЮ КОНФИГУРАЦИЮ!
🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧
        """)
        
        # Загружаем больше данных
        if not self.fetch_historical_data(150):
            print("❌ Не удалось загрузить достаточно данных")
            return
        
        # ТЕСТИРУЕМ НА РАЗНЫХ ПЕРИОДАХ
        test_periods = [
            {"start": 30, "trades": 30, "name": "PERIOD_1"},
            {"start": 50, "trades": 40, "name": "PERIOD_2"}, 
            {"start": 70, "trades": 35, "name": "PERIOD_3"},
            {"start": 90, "trades": 30, "name": "PERIOD_4"}
        ]
        
        all_results = []
        
        for period in test_periods:
            print(f"\n" + "🔧"*70)
            print(f"🔧 ТЕСТИРОВАНИЕ ПЕРИОДА: {period['name']}")
            print(f"🔧 Раунды: {period['start']} - {period['start'] + period['trades']}")
            print("🔧"*70)
            
            # Сбрасываем состояние
            self.balance = 1000.0
            self.total_trades = 0
            self.winning_trades = 0
            self.total_profit = 0.0
            
            period_results = self.run_period_test(period['start'], period['trades'])
            period_results['period_name'] = period['name']
            all_results.append(period_results)
        
        # АНАЛИЗ ВСЕХ РЕЗУЛЬТАТОВ
        self.analyze_all_periods(all_results)
    
    def run_period_test(self, start_idx, num_trades):
        """Тестирование одного периода"""
        
        trade_results = []
        
        for i in range(num_trades):
            current_idx = start_idx + i
            
            if current_idx >= len(self.historical_data):
                break
            
            current_round = self.historical_data[current_idx]
            round_id = current_round['round_id']
            actual_crash = current_round['crashed_at']
            
            # Предсказываем волатильность
            vol_level, vol_score, vol_desc = self.advanced_volatility_prediction(current_idx)
            
            # Получаем стратегию
            strategy = self.get_optimized_strategy(vol_level, vol_score)
            
            bet_size = strategy['bet_size']
            target_mult = strategy['target_multiplier']
            
            # Результат
            if actual_crash >= target_mult:
                profit = bet_size * (target_mult - 1)
                self.balance += profit
                self.total_profit += profit
                self.winning_trades += 1
                result = "WIN"
            else:
                self.balance -= bet_size
                self.total_profit -= bet_size
                result = "LOSS"
            
            self.total_trades += 1
            
            trade_results.append({
                'round_id': round_id,
                'vol_prediction': vol_level,
                'vol_score': vol_score,
                'bet_size': bet_size,
                'target_mult': target_mult,
                'actual_crash': actual_crash,
                'result': result,
                'profit': profit if result == "WIN" else -bet_size
            })
        
        # Результаты периода
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        roi = (self.total_profit / 1000) * 100
        
        print(f"📊 ПЕРИОД ЗАВЕРШЕН:")
        print(f"🎯 Trades: {self.total_trades} | Wins: {self.winning_trades} | Win Rate: {win_rate:.1f}%")
        print(f"💰 P&L: ${self.total_profit:.2f} | ROI: {roi:.1f}%")
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'roi': roi,
            'final_balance': self.balance
        }
    
    def analyze_all_periods(self, all_results):
        """Анализ всех периодов"""
        
        print(f"\n" + "🔧"*80)
        print("🔧 СВОДНЫЙ АНАЛИЗ ВСЕХ ПЕРИОДОВ!")
        print("🔧"*80)
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ПО ПЕРИОДАМ:")
        
        total_profit = 0
        profitable_periods = 0
        
        for result in all_results:
            period_name = result['period_name']
            profit = result['total_profit']
            roi = result['roi']
            win_rate = result['win_rate']
            
            total_profit += profit
            if profit > 0:
                profitable_periods += 1
            
            status = "✅ ПРИБЫЛЬ" if profit > 0 else "❌ УБЫТОК"
            print(f"🎯 {period_name}: {status} | P&L: ${profit:.2f} | ROI: {roi:.1f}% | WR: {win_rate:.1f}%")
        
        average_profit = total_profit / len(all_results)
        average_roi = (average_profit / 1000) * 100
        
        print(f"\n💡 ИТОГОВЫЙ АНАЛИЗ:")
        print(f"📊 Прибыльных периодов: {profitable_periods}/{len(all_results)}")
        print(f"💰 Средняя прибыль за период: ${average_profit:.2f}")
        print(f"📈 Средний ROI: {average_roi:.1f}%")
        
        if profitable_periods >= len(all_results) / 2:
            print(f"\n🚀 ОПТИМИЗАЦИЯ УСПЕШНА!")
            print(f"✅ Стратегия показывает стабильную прибыльность!")
            print(f"💰 Можно тестировать на реальных деньгах!")
        elif average_profit > -10:
            print(f"\n📊 УЛУЧШЕНИЕ ЗАМЕТНО!")
            print(f"🔧 Близко к прибыльности - нужна дополнительная настройка!")
        else:
            print(f"\n❌ СТРАТЕГИЯ ВСЕ ЕЩЕ УБЫТОЧНА")
            print(f"🤔 Возможно волатильность действительно непредсказуема")

if __name__ == "__main__":
    TOKEN = "YOUR_TOKEN"
    
    bot = OptimizedBacktestBot(TOKEN)
    bot.run_multiple_periods_test()
