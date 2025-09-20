import pandas as pd
import requests
import time
import os
import random
from tqdm import tqdm

# --- НАСТРОЙКИ ---
ROUNDS_FILE = "rounds.csv"
BETS_FILE = "bets.csv"
AUTHORIZATION_TOKEN = 'YOUR_TOKEN'
MIN_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 1.5

# --- Копируем функции из нашего "Сборщика" ---
# (Код функций fetch_single_round и calculate_profit_loss остается без изменений)
def fetch_single_round(round_id, token):
    api_url = f"https://6cs.fail/api/crash/games/{round_id}"
    headers = {'User-Agent': 'Mozilla/5.0...', 'Authorization': token}
    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        if response.status_code == 200: return response.json()
    except requests.exceptions.RequestException: pass
    return None

def calculate_profit_loss(game_data):
    if not game_data or 'data' not in game_data: return 0.0
    bets = game_data['data'].get('bets', [])
    crashed_at = float(game_data['data']['game'].get('crashedAt', 0.0))
    total_bets_value = sum(float(bet.get('itemsTotal', 0.0)) for bet in bets)
    total_payouts = 0
    for bet in bets:
        cashout_ratio = bet.get('cashoutRatio')
        # Проверяем, что cashout_ratio не None и конвертируется в float
        if cashout_ratio and float(cashout_ratio) <= crashed_at:
            total_payouts += float(bet.get('itemsTotal', 0.0)) * float(cashout_ratio)
    return round(total_bets_value - total_payouts, 4)

# --- Основная функция ---
def main():
    print("--- Запуск Операции 'Аудит и Восполнение' v2.0 ---")
    
    # --- Шаг 1: Загрузка и первичный аудит ---
    try:
        df_rounds = pd.read_csv(ROUNDS_FILE)
        df_bets = pd.read_csv(BETS_FILE, low_memory=False)
        print(f"\n[INFO] Загружено {len(df_rounds)} раундов и {len(df_bets)} ставок.")
    except FileNotFoundError as e:
        print(f"[ERROR] Файл не найден: {e}.")
        return

    # Удаление дубликатов
    initial_rounds = len(df_rounds)
    df_rounds.drop_duplicates(subset=['round_id'], inplace=True)
    if initial_rounds > len(df_rounds):
        print(f"[INFO] Удалено {initial_rounds - len(df_rounds)} дубликатов раундов.")
    
    initial_bets = len(df_bets)
    df_bets.drop_duplicates(subset=['bet_id'], inplace=True)
    if initial_bets > len(df_bets):
        print(f"[INFO] Удалено {initial_bets - len(df_bets)} дубликатов ставок.")

    # --- Шаг 2: Поиск пропущенных раундов ---
    min_id = df_rounds['round_id'].min()
    max_id = df_rounds['round_id'].max()
    print(f"[INFO] Диапазон раундов: от {min_id} до {max_id}.")
    
    # Создаем полный ожидаемый диапазон ID
    expected_ids = set(range(min_id, max_id + 1))
    actual_ids = set(df_rounds['round_id'])
    missing_ids = sorted(list(expected_ids - actual_ids))
    
    if not missing_ids:
        # **НОВАЯ ПРОВЕРКА ЦЕЛОСТНОСТИ**
        df_rounds_sorted = df_rounds.sort_values('round_id').reset_index(drop=True)
        id_diffs = df_rounds_sorted['round_id'].diff().dropna()
        if (id_diffs != 1).any():
             print("[WARNING] Обнаружены разрывы в последовательности ID, даже если нет пропущенных ID в диапазоне. Пересчитываем...")
             # Этот блок может сработать, если у вас несколько несвязанных кусков истории.
             # Мы уже обработали это в `expected_ids`, но это дополнительная проверка.
        else:
            print("[SUCCESS] Проверка целостности пройдена. Все раунды идут по порядку.")

    print(f"[INFO] Найдено {len(missing_ids)} пропущенных раундов в диапазоне.")

    # --- Шаг 3: Восполнение данных ---
    if missing_ids:
        if 'СЮДА_ВСТАВИТЬ' in AUTHORIZATION_TOKEN:
            print("[ERROR] Найдены пропуски, но вы не вставили токен авторизации.")
            return
            
        print(f"\n[INFO] Начинаем докачивать {len(missing_ids)} пропущенных раундов...")
        new_rounds_data = []
        new_bets_data = []
        
        for round_id in tqdm(missing_ids, desc="Докачивание раундов"):
            raw_data = fetch_single_round(round_id, AUTHORIZATION_TOKEN)
            if raw_data and 'data' in raw_data and 'game' in raw_data['data']:
                game = raw_data['data']['game']
                bets = raw_data['data']['bets']
                profit = calculate_profit_loss(raw_data)
                
                new_rounds_data.append({
                    'round_id': game.get('id'), 'crashed_at': game.get('crashedAt'),
                    'total_bank_usd': game.get('totalBankUsd'), 'users_count': game.get('usersCount'),
                    'start_at': game.get('startAt'), 'profit_loss': profit
                })
                for bet in bets:
                    new_bets_data.append({
                        'bet_id': bet.get('id'), 'round_id': game.get('id'),
                        'user_id': bet.get('user', {}).get('id'),
                        'bet_amount_usd': bet.get('itemsTotal'), 'cashout_ratio': bet.get('cashoutRatio')
                    })
            time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))

        if new_rounds_data:
            df_new_rounds = pd.DataFrame(new_rounds_data)
            df_rounds = pd.concat([df_rounds, df_new_rounds], ignore_index=True)
            print(f"[INFO] Успешно докачано и добавлено {len(new_rounds_data)} новых раундов.")
        if new_bets_data:
            df_new_bets = pd.DataFrame(new_bets_data)
            df_bets = pd.concat([df_bets, df_new_bets], ignore_index=True)
            
    # --- Шаг 4: Финальная сортировка и сохранение ---
    print("\n[INFO] Финальная сортировка и сохранение данных для гарантии целостности...")
    df_rounds = df_rounds.sort_values('round_id').reset_index(drop=True)
    
    # Оставляем только те ставки, для которых у нас есть информация о раунде
    df_bets = df_bets[df_bets['round_id'].isin(df_rounds['round_id'])]
    df_bets = df_bets.sort_values(['round_id', 'bet_id']).reset_index(drop=True)
    
    df_rounds.to_csv(ROUNDS_FILE, index=False)
    df_bets.to_csv(BETS_FILE, index=False)
    
    print(f"\n[SUCCESS] Аудит и восполнение завершены!")
    print(f"Итоговое количество раундов в {ROUNDS_FILE}: {len(df_rounds)}")
    print(f"Итоговое количество ставок в {BETS_FILE}: {len(df_bets)}")
    
if __name__ == "__main__":

    main()
