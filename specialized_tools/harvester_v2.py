import requests
import json
import pandas as pd
import time
import os
import random

# --- ГЛАВНЫЕ НАСТРОЙКИ ---
START_ROUND_ID = 7005265
ROUNDS_TO_FETCH = 50000

# --- ВАШ КЛЮЧ ОТ АККАУНТА ---
AUTHORIZATION_TOKEN = 'YOUR_TOKEN'

# --- НАСТРОЙКИ БЕЗОПАСНОСТИ ---
MIN_DELAY_SECONDS = 0.1
MAX_DELAY_SECONDS = 0.9

# --- НАЗВАНИЯ ФАЙЛОВ ---
ROUNDS_FILE = "rounds.csv"
BETS_FILE = "bets.csv"

def fetch_single_round(round_id, token):
    """Скачивает и возвращает полные данные по одному раунду."""
    api_url = f"https://6cs.fail/api/crash/games/{round_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Authorization': token
    }
    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  - Предупреждение: Раунд #{round_id} не найден (Код {response.status_code}).")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  - ОШИБКА СЕТИ при запросе раунда #{round_id}: {e}")
        return None

def calculate_profit_loss(game_data):
    """Рассчитывает прибыль/убыток казино за раунд."""
    if not game_data or 'data' not in game_data:
        return 0.0

    bets = game_data['data'].get('bets', [])
    crashed_at = float(game_data['data']['game'].get('crashedAt', 0.0))
    
    total_bets_value = sum(float(bet.get('itemsTotal', 0.0)) for bet in bets)
    total_payouts = 0
    
    for bet in bets:
        cashout_ratio = bet.get('cashoutRatio')
        if cashout_ratio:
            cashout_ratio = float(cashout_ratio)
            if cashout_ratio <= crashed_at:
                # Игрок успел забрать выигрыш
                bet_value = float(bet.get('itemsTotal', 0.0))
                total_payouts += bet_value * cashout_ratio
                
    return round(total_bets_value - total_payouts, 4)

def save_data(round_info, bets_info):
    """Дописывает данные в CSV файлы."""
    # Сохраняем информацию о раунде
    df_round = pd.DataFrame([round_info])
    df_round.to_csv(ROUNDS_FILE, mode='a', header=not os.path.exists(ROUNDS_FILE), index=False)
    
    # Сохраняем информацию о ставках
    if bets_info:
        df_bets = pd.DataFrame(bets_info)
        df_bets.to_csv(BETS_FILE, mode='a', header=not os.path.exists(BETS_FILE), index=False)

def main():
    print("--- Запуск Сборщика Урожая v2.0 ---")
    if 'СЮДА_ВСТАВИТЬ' in AUTHORIZATION_TOKEN:
        print("ОШИБКА: Вы не вставили токен авторизации.")
        return

    processed_ids = set()
    if os.path.exists(ROUNDS_FILE):
        try:
            df_existing = pd.read_csv(ROUNDS_FILE)
            if not df_existing.empty:
                processed_ids = set(df_existing['round_id'])
            print(f"Найден файл {ROUNDS_FILE}. Загружено {len(processed_ids)} уже скачанных ID.")
        except (pd.errors.EmptyDataError, FileNotFoundError):
             print(f"Файл {ROUNDS_FILE} найден, но он пустой или поврежден.")

    for i in range(ROUNDS_TO_FETCH):
        current_round_id = START_ROUND_ID - i
        
        if current_round_id in processed_ids:
            print(f"Раунд #{current_round_id} уже в базе. Пропускаем.")
            continue

        print(f"Запрашиваем раунд #{current_round_id} ({i+1}/{ROUNDS_TO_FETCH})...")
        raw_data = fetch_single_round(current_round_id, AUTHORIZATION_TOKEN)
        
        if raw_data and 'data' in raw_data and 'game' in raw_data['data']:
            game = raw_data['data']['game']
            bets = raw_data['data']['bets']
            
            # 1. Готовим данные для rounds.csv
            profit = calculate_profit_loss(raw_data)
            round_info = {
                'round_id': game.get('id'),
                'crashed_at': game.get('crashedAt'),
                'total_bank_usd': game.get('totalBankUsd'),
                'users_count': game.get('usersCount'),
                'start_at': game.get('startAt'),
                'profit_loss': profit
            }
            
            # 2. Готовим данные для bets.csv
            bets_info = []
            for bet in bets:
                bets_info.append({
                    'bet_id': bet.get('id'),
                    'round_id': game.get('id'),
                    'user_id': bet.get('user', {}).get('id'),
                    'bet_amount_usd': bet.get('itemsTotal'),
                    'cashout_ratio': bet.get('cashoutRatio')
                })
            
            # 3. Сохраняем всё
            save_data(round_info, bets_info)
            processed_ids.add(current_round_id)
            print(f"  ... Раунд #{current_round_id} успешно обработан и сохранен.")

        delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
        print(f"  ... Задержка {delay:.1f} сек.")
        time.sleep(delay)
        
    print(f"\n--- Сбор Завершен ---")
    print(f"Проверено {ROUNDS_TO_FETCH} раундов. Новые данные добавлены в {ROUNDS_FILE} и {BETS_FILE}.")

if __name__ == "__main__":
    main()