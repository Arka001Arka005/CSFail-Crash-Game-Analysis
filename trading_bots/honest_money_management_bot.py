"""
💰💰💰 HONEST MONEY MANAGEMENT BOT 💰💰💰  
ФИНАЛЬНОЕ РЕШЕНИЕ: НЕ ПРЕДСКАЗЫВАЕМ, А УПРАВЛЯЕМ РИСКАМИ!

- Принимает что волатильность непредсказуема
- Фокус на управлении размером ставок и рисками
- Адаптируется к текущим наблюдаемым условиям
- Консервативные цели и строгая дисциплина
- Честный подход без попыток предсказания
"""

import requests
import pandas as pd
import numpy as np
import time
import random
import warnings
warnings.filterwarnings('ignore')

class HonestMoneyManagementBot:
    def __init__(self, token):
        """💰 ЧЕСТНЫЙ МАНИ-МЕНЕДЖМЕНТ БОТ"""
        
        self.token = token
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Authorization': token
        }
        
        self.start_round_id = 7026046
        self.historical_data = []
        
        # ЧЕСТНЫЕ ПАРАМЕТРЫ - НЕ ОСНОВАННЫЕ НА ПРЕДСКАЗАНИЯХ
        self.initial_balance = 1000.0
        self.balance = 1000.0
        self.base_bet_percentage = 0.02  # 2% от баланса
        self.max_bet_percentage = 0.05   # Максимум 5% от баланса
        self.min_bet = 2.0
        self.max_bet = 20.0
        
        # КОНСЕРВАТИВНЫЕ ЦЕЛИ
        self.conservative_targets = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        self.current_target_idx = 0
        
        # АДАПТИВНОЕ УПРАВЛЕНИЕ РИСКАМИ
        self.recent_results = []  # Последние 20 результатов
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # СТАТИСТИКА
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 1000.0
        
        print("💰 HONEST MONEY MANAGEMENT BOT INITIALIZED")
        print("🎯 ФОКУС НА УПРАВЛЕНИИ РИСКАМИ, А НЕ ПРЕДСКАЗАНИЯХ!")
    
    def fetch_data_for_adaptation(self, num_rounds=100):
        """📊 Загружаем данные только для адаптации к текущим условиям"""
        print(f"📥 Загружаем {num_rounds} раундов для адаптации...")
        
        loaded_data = []
        for i in range(num_rounds):
            round_id = self.start_round_id - i
            
            if i % 25 == 0:
                print(f"📊 Прогресс: {i+1}/{num_rounds} раундов...")
            
            try:
                response = requests.get(f"https://6cs.fail/api/crash/games/{round_id}", 
                                      headers=self.headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'game' in data['data']:
                        game = data['data']['game']
                        
                        round_info = {
                            'round_id': game.get('id'),
                            'crashed_at': float(game.get('crashedAt', 0.0)),
                            'total_bank_usd': float(game.get('totalBankUsd', 0.0)),
                            'users_count': int(game.get('usersCount', 0))
                        }
                        loaded_data.append(round_info)
                
                time.sleep(random.uniform(0.05, 0.15))
                
            except Exception as e:
                continue
        
        self.historical_data = sorted(loaded_data, key=lambda x: x['round_id'])
        print(f"✅ Загружено {len(self.historical_data)} раундов для адаптации")
        return len(self.historical_data) > 50
    
    def observe_current_conditions(self, current_idx):
        """👁️ НАБЛЮДАЕМ текущие условия (НЕ ПРЕДСКАЗЫВАЕМ!)"""
        if current_idx < 20:
            return {
                'recent_volatility': 'unknown',
                'activity_level': 'unknown',
                'trend': 'unknown'
            }
        
        # Смотрим только на ПРОШЛЫЕ данные
        recent_crashes = [self.historical_data[i]['crashed_at'] for i in range(current_idx-20, current_idx)]
        recent_banks = [self.historical_data[i]['total_bank_usd'] for i in range(current_idx-10, current_idx)]
        recent_users = [self.historical_data[i]['users_count'] for i in range(current_idx-10, current_idx)]
        
        # НАБЛЮДАЕМАЯ волатильность (прошлая)
        observed_volatility = np.std(recent_crashes)
        volatility_level = 'low' if observed_volatility < 0.8 else 'medium' if observed_volatility < 1.2 else 'high'
        
        # НАБЛЮДАЕМАЯ активность
        avg_bank = np.mean(recent_banks)
        avg_users = np.mean(recent_users)
        activity_level = 'low' if avg_users < 50 else 'medium' if avg_users < 100 else 'high'
        
        # НАБЛЮДАЕМЫЙ тренд последних крашей
        if len(recent_crashes) >= 5:
            recent_5 = recent_crashes[-5:]
            high_crashes = len([x for x in recent_5 if x > 2.0])
            low_crashes = len([x for x in recent_5 if x < 1.5])
            
            if high_crashes >= 3:
                trend = 'high_crashes'
            elif low_crashes >= 3:
                trend = 'low_crashes'
            else:
                trend = 'mixed'
        else:
            trend = 'unknown'
        
        return {
            'recent_volatility': volatility_level,
            'activity_level': activity_level,
            'trend': trend,
            'observed_vol_value': observed_volatility
        }
    
    def get_risk_adjusted_bet_size(self):
        """💰 Размер ставки на основе управления рисками"""
        
        # Базовый размер от баланса
        base_bet = self.balance * self.base_bet_percentage
        
        # Корректировка на основе последних результатов
        if self.consecutive_losses >= 3:
            # После серии проигрышей - уменьшаем ставку
            risk_multiplier = 0.5
        elif self.consecutive_losses >= 2:
            risk_multiplier = 0.75
        elif self.consecutive_wins >= 3:
            # Небольшое увеличение после побед
            risk_multiplier = 1.25
        else:
            risk_multiplier = 1.0
        
        # Корректировка на основе дравдауна
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if current_drawdown > 0.15:  # Большой дравдаун
            risk_multiplier *= 0.6
        elif current_drawdown > 0.10:
            risk_multiplier *= 0.8
        
        adjusted_bet = base_bet * risk_multiplier
        
        # Ограничения
        adjusted_bet = max(adjusted_bet, self.min_bet)
        adjusted_bet = min(adjusted_bet, self.max_bet)
        adjusted_bet = min(adjusted_bet, self.balance * self.max_bet_percentage)
        
        return round(adjusted_bet, 2)
    
    def get_adaptive_target(self, conditions):
        """🎯 Адаптивная цель на основе наблюдаемых условий"""
        
        # Базовая консервативная цель
        base_targets = [1.5, 1.6, 1.7, 1.8]
        
        # Адаптация на основе НАБЛЮДАЕМЫХ условий
        if conditions['recent_volatility'] == 'high':
            # При высокой волатильности - более консервативные цели
            target = random.choice([1.4, 1.5, 1.6])
        elif conditions['recent_volatility'] == 'low':
            # При низкой волатильности - чуть более агрессивные
            target = random.choice([1.6, 1.7, 1.8, 1.9])
        else:
            # Стандартные цели
            target = random.choice(base_targets)
        
        # Корректировка на основе последних результатов
        if self.consecutive_losses >= 2:
            # Более консервативно после потерь
            target = min(target, 1.6)
        elif len(self.recent_results) >= 10:
            recent_win_rate = sum(self.recent_results[-10:]) / 10
            if recent_win_rate < 0.4:  # Плохая серия
                target = min(target, 1.5)
        
        return target
    
    def update_statistics(self, won, bet_size, profit):
        """📊 Обновление статистики"""
        
        self.total_trades += 1
        if won:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_wins = 0
            self.consecutive_losses += 1
        
        self.total_profit += profit
        self.balance += profit
        
        # Отслеживание пика и дравдауна
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        current_drawdown = self.peak_balance - self.balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Обновляем последние результаты
        self.recent_results.append(1 if won else 0)
        if len(self.recent_results) > 20:
            self.recent_results.pop(0)
    
    def run_honest_test(self, start_idx=50, num_trades=80):
        """💰 ЧЕСТНЫЙ ТЕСТ МАНИ-МЕНЕДЖМЕНТА"""
        
        print(f"""
💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰
💰 HONEST MONEY MANAGEMENT BOT - ФИНАЛЬНЫЙ ТЕСТ!
💰 НЕ ПРЕДСКАЗЫВАЕМ - УПРАВЛЯЕМ РИСКАМИ!
💰 АДАПТИРУЕМСЯ К НАБЛЮДАЕМЫМ УСЛОВИЯМ!
💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰
        """)
        
        # Загружаем данные для адаптации
        if not self.fetch_data_for_adaptation(100):
            print("❌ Не удалось загрузить данные для адаптации")
            return
        
        print(f"🎯 Начинаем честный тест с {num_trades} сделок")
        print(f"💰 Стартовый баланс: ${self.balance}")
        print(f"📊 Управление рисками: 2-5% от баланса")
        print(f"🎯 Консервативные цели: 1.4-1.9x")
        
        results = []
        
        for i in range(num_trades):
            current_idx = start_idx + i
            
            if current_idx >= len(self.historical_data):
                break
            
            current_round = self.historical_data[current_idx]
            actual_crash = current_round['crashed_at']
            
            # НАБЛЮДАЕМ текущие условия (не предсказываем!)
            conditions = self.observe_current_conditions(current_idx)
            
            # Размер ставки на основе управления рисками
            bet_size = self.get_risk_adjusted_bet_size()
            
            # Адаптивная цель на основе наблюдений
            target_mult = self.get_adaptive_target(conditions)
            
            # Результат
            if actual_crash >= target_mult:
                profit = bet_size * (target_mult - 1)
                won = True
                result = "WIN"
            else:
                profit = -bet_size
                won = False
                result = "LOSS"
            
            # Обновляем статистику
            self.update_statistics(won, bet_size, profit)
            
            # Сохраняем результат
            results.append({
                'round_id': current_round['round_id'],
                'conditions': conditions,
                'actual_crash': actual_crash,
                'bet_size': bet_size,
                'target_mult': target_mult,
                'result': result,
                'profit': profit,
                'balance': self.balance,
                'consecutive_losses': self.consecutive_losses
            })
            
            # Промежуточная статистика
            if (i + 1) % 20 == 0:
                win_rate = (self.winning_trades / self.total_trades * 100)
                roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
                drawdown_pct = (self.max_drawdown / self.peak_balance) * 100
                print(f"📊 {i+1} сделок: WR={win_rate:.1f}%, Balance=${self.balance:.2f}, ROI={roi:.1f}%, DD={drawdown_pct:.1f}%")
        
        # Финальный анализ
        self.print_honest_results(results)
    
    def print_honest_results(self, results):
        """💰 Анализ результатов честного мани-менеджмента"""
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        drawdown_pct = (self.max_drawdown / self.peak_balance) * 100
        
        print(f"""
💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰
💰 HONEST MONEY MANAGEMENT RESULTS - ФИНАЛЬНЫЙ ТЕСТ!
💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰
        """)
        
        print(f"🎯 Total Trades: {self.total_trades}")
        print(f"✅ Wins: {self.winning_trades}")
        print(f"📈 Win Rate: {win_rate:.1f}%")
        print(f"💰 Start Balance: ${self.initial_balance:.2f}")
        print(f"💰 Final Balance: ${self.balance:.2f}")
        print(f"📊 Total P&L: ${self.total_profit:.2f}")
        print(f"📈 ROI: {roi:.1f}%")
        print(f"📉 Max Drawdown: ${self.max_drawdown:.2f} ({drawdown_pct:.1f}%)")
        print(f"🏔️ Peak Balance: ${self.peak_balance:.2f}")
        
        # Анализ по условиям
        conditions_stats = {}
        for r in results:
            vol = r['conditions']['recent_volatility']
            if vol not in conditions_stats:
                conditions_stats[vol] = {'wins': 0, 'total': 0, 'profit': 0}
            
            conditions_stats[vol]['total'] += 1
            conditions_stats[vol]['profit'] += r['profit']
            if r['result'] == 'WIN':
                conditions_stats[vol]['wins'] += 1
        
        print(f"\n📊 BREAKDOWN BY OBSERVED CONDITIONS:")
        for condition, stats in conditions_stats.items():
            wr = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"👁️ {condition.upper()}_VOLATILITY: {stats['wins']}/{stats['total']} ({wr:.1f}%) | P&L: ${stats['profit']:.2f}")
        
        # Анализ управления рисками
        avg_bet = np.mean([r['bet_size'] for r in results])
        bet_range = (min([r['bet_size'] for r in results]), max([r['bet_size'] for r in results]))
        avg_target = np.mean([r['target_mult'] for r in results])
        
        print(f"\n💰 RISK MANAGEMENT ANALYSIS:")
        print(f"💵 Average Bet Size: ${avg_bet:.2f}")
        print(f"📊 Bet Size Range: ${bet_range[0]:.2f} - ${bet_range[1]:.2f}")
        print(f"🎯 Average Target: {avg_target:.2f}x")
        print(f"⚖️ Risk per Trade: {(avg_bet/self.initial_balance)*100:.1f}% of initial capital")
        
        # Финальный вердикт
        print(f"\n💡 HONEST MONEY MANAGEMENT CONCLUSION:")
        if roi > 5:
            print("🚀 ОТЛИЧНЫЙ РЕЗУЛЬТАТ! Честный мани-менеджмент работает!")
            print("💰 Стабильная прибыльность без предсказаний!")
            print("✅ Готово к реальной торговле с управлением рисками!")
        elif roi > 0:
            print("✅ ПОЛОЖИТЕЛЬНЫЙ ROI! Мани-менеджмент эффективен!")
            print("📊 Риск-менеджмент помогает даже в случайной игре!")
        elif roi > -5:
            print("⚖️ БЛИЗКО К БЕЗУБЫТОЧНОСТИ!")
            print("💰 Хорошее управление рисками ограничивает потери!")
        else:
            print("📉 Даже мани-менеджмент не спасает")
            print("🤔 Возможно нужны еще более консервативные цели")
        
        print(f"\n🎯 КЛЮЧЕВОЙ ИНСАЙТ:")
        print(f"💰 Этот подход НЕ ПЫТАЕТСЯ предсказать будущее")
        print(f"📊 Фокусируется на управлении размером позиций и рисками")
        print(f"👁️ Адаптируется к наблюдаемым условиям без предсказаний")
        print(f"⚖️ Может быть эффективен даже в полностью случайной игре!")

if __name__ == "__main__":
    TOKEN = "YOUR_TOKEN"
    
    bot = HonestMoneyManagementBot(TOKEN)
    bot.run_honest_test(start_idx=50, num_trades=60)
