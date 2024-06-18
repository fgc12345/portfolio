import pandas as pd
from binance.client import Client
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from ta.utils import dropna
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import datetime
import time
import traceback
from BinanceTradingBot import BinanceTrading
from binance_simulation import Simulation

# 바이낸스 API 키 설정
api_key = 'your_api_key'
api_secret = 'your_secret_key'
client = Client(api_key, api_secret)

# 거래 설정
LEVERAGE = 7
SYMBOL = 'HIGHUSDT'
BALANCE_PERCENTAGE = 0.8
THRESHOLD = 0.0017  # 임계값 (%)
RETRAIN_INTERVAL = 12 * 60 * 60  # 재학습 간격 (12시간)
LOOKBACK_HOURS = 3  # 최근 데이터 조회 시간 (3시간)
학습할기간 = 4
최소atr값 = 0.005


# 로깅 설정
log_path = 'trading_log.txt'




def get_historical_klines(symbol, interval, start_str, end_str=None):
    print(f"Fetching data from {start_str} to {end_str} for {symbol} with interval {interval}")
    start_ts = int(datetime.datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    end_ts = int(datetime.datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000) if end_str else None

    if end_ts:
        klines = client.get_historical_klines(symbol, interval, start_ts, end_ts)
    else:
        klines = client.get_historical_klines(symbol, interval, start_ts)

    print(f"Number of klines fetched: {len(klines)}")
    if len(klines) == 0:
        print("No data fetched. Check the start and end times.")

    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data = data.astype(float)
    return data[['open', 'high', 'low', 'close', 'volume']]


def calculate_indicators(data):
    data['sma'] = SMAIndicator(data['close'], window=14).sma_indicator()
    data['ema'] = EMAIndicator(data['close'], window=14).ema_indicator()
    data['macd'] = MACD(data['close']).macd()
    data['rsi'] = RSIIndicator(data['close']).rsi()
    data['stoch'] = StochasticOscillator(data['high'], data['low'], data['close']).stoch()
    bb = BollingerBands(data['close'])
    data['bb_hband'] = bb.bollinger_hband()
    data['bb_lband'] = bb.bollinger_lband()
    data['obv'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    data['atr'] = AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
    return data


def train_models(data):
    # 기술적 지표 추가
    data = calculate_indicators(data)

    # 목표 변수 추가 (1분, 2분, 3분, 4분, 5분, 6분, 7분, 8분, 9분, 10분 뒤의 종가)
    data['target_1min'] = data['close'].shift(-1)
    data['target_2min'] = data['close'].shift(-2)
    data['target_3min'] = data['close'].shift(-3)
    data['target_4min'] = data['close'].shift(-4)
    data['target_5min'] = data['close'].shift(-5)
    data['target_6min'] = data['close'].shift(-6)
    data['target_7min'] = data['close'].shift(-7)
    data['target_8min'] = data['close'].shift(-8)
    data['target_9min'] = data['close'].shift(-9)
    data['target_10min'] = data['close'].shift(-10)

    # 결측치 제거
    data.dropna(inplace=True)

    # 피처와 타겟 분리
    features = data.drop(['target_1min', 'target_2min', 'target_3min', 'target_4min', 'target_5min',
                          'target_6min', 'target_7min', 'target_8min', 'target_9min', 'target_10min'], axis=1)
    target_1min = data['target_1min']
    target_2min = data['target_2min']
    target_3min = data['target_3min']
    target_4min = data['target_4min']
    target_5min = data['target_5min']
    target_6min = data['target_6min']
    target_7min = data['target_7min']
    target_8min = data['target_8min']
    target_9min = data['target_9min']
    target_10min = data['target_10min']

    # 데이터 분리
    X_train_1min, X_test_1min, y_train_1min, y_test_1min = train_test_split(features, target_1min, test_size=0.2, shuffle=True)
    X_train_2min, X_test_2min, y_train_2min, y_test_2min = train_test_split(features, target_2min, test_size=0.2, shuffle=True)
    X_train_3min, X_test_3min, y_train_3min, y_test_3min = train_test_split(features, target_3min, test_size=0.2, shuffle=True)
    X_train_4min, X_test_4min, y_train_4min, y_test_4min = train_test_split(features, target_4min, test_size=0.2, shuffle=True)
    X_train_5min, X_test_5min, y_train_5min, y_test_5min = train_test_split(features, target_5min, test_size=0.2, shuffle=True)
    X_train_6min, X_test_6min, y_train_6min, y_test_6min = train_test_split(features, target_6min, test_size=0.2, shuffle=True)
    X_train_7min, X_test_7min, y_train_7min, y_test_7min = train_test_split(features, target_7min, test_size=0.2, shuffle=True)
    X_train_8min, X_test_8min, y_train_8min, y_test_8min = train_test_split(features, target_8min, test_size=0.2, shuffle=True)
    X_train_9min, X_test_9min, y_train_9min, y_test_9min = train_test_split(features, target_9min, test_size=0.2, shuffle=True)
    X_train_10min, X_test_10min, y_train_10min, y_test_10min = train_test_split(features, target_10min, test_size=0.2, shuffle=True)

    # 선형 회귀 모델 학습
    linear_model_1min = LinearRegression().fit(X_train_1min, y_train_1min)
    linear_model_2min = LinearRegression().fit(X_train_2min, y_train_2min)
    linear_model_3min = LinearRegression().fit(X_train_3min, y_train_3min)
    linear_model_4min = LinearRegression().fit(X_train_4min, y_train_4min)
    linear_model_5min = LinearRegression().fit(X_train_5min, y_train_5min)
    linear_model_6min = LinearRegression().fit(X_train_6min, y_train_6min)
    linear_model_7min = LinearRegression().fit(X_train_7min, y_train_7min)
    linear_model_8min = LinearRegression().fit(X_train_8min, y_train_8min)
    linear_model_9min = LinearRegression().fit(X_train_9min, y_train_9min)
    linear_model_10min = LinearRegression().fit(X_train_10min, y_train_10min)

    # 랜덤 포레스트 모델 학습
    rf_model_1min = RandomForestRegressor(n_estimators=100).fit(X_train_1min, y_train_1min)
    rf_model_2min = RandomForestRegressor(n_estimators=100).fit(X_train_2min, y_train_2min)
    rf_model_3min = RandomForestRegressor(n_estimators=100).fit(X_train_3min, y_train_3min)
    rf_model_4min = RandomForestRegressor(n_estimators=100).fit(X_train_4min, y_train_4min)
    rf_model_5min = RandomForestRegressor(n_estimators=100).fit(X_train_5min, y_train_5min)
    rf_model_6min = RandomForestRegressor(n_estimators=100).fit(X_train_6min, y_train_6min)
    rf_model_7min = RandomForestRegressor(n_estimators=100).fit(X_train_7min, y_train_7min)
    rf_model_8min = RandomForestRegressor(n_estimators=100).fit(X_train_8min, y_train_8min)
    rf_model_9min = RandomForestRegressor(n_estimators=100).fit(X_train_9min, y_train_9min)
    rf_model_10min = RandomForestRegressor(n_estimators=100).fit(X_train_10min, y_train_10min)

    return (linear_model_1min, linear_model_2min, linear_model_3min, linear_model_4min, linear_model_5min, linear_model_6min, linear_model_7min, linear_model_8min, linear_model_9min, linear_model_10min,
            rf_model_1min, rf_model_2min, rf_model_3min, rf_model_4min, rf_model_5min, rf_model_6min, rf_model_7min, rf_model_8min, rf_model_9min, rf_model_10min)


def main():
    bot = BinanceTrading(client, SYMBOL, LEVERAGE, BALANCE_PERCENTAGE)
    simul = Simulation(client, SYMBOL, LEVERAGE, BALANCE_PERCENTAGE,100000)

    # 모델 학습
    start_str = (datetime.datetime.now() - datetime.timedelta(days=학습할기간)).strftime('%Y-%m-%d %H:%M:%S')
    end_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = get_historical_klines(SYMBOL, Client.KLINE_INTERVAL_1MINUTE, start_str, end_str)
    data = dropna(data)
    (linear_model_1min, linear_model_2min, linear_model_3min, linear_model_4min, linear_model_5min, linear_model_6min, linear_model_7min, linear_model_8min, linear_model_9min, linear_model_10min,
     rf_model_1min, rf_model_2min, rf_model_3min, rf_model_4min, rf_model_5min, rf_model_6min, rf_model_7min, rf_model_8min, rf_model_9min, rf_model_10min) = train_models(data)
    last_train_time = time.time()

    while True:
        try:
            # 일정 시간 주기로 모델 재학습
            if time.time() - last_train_time > RETRAIN_INTERVAL:
                print("Retraining models...")
                bot.close_position()  # 포지션 종료 후 재학습
                simul.stop_position()
                start_str = (datetime.datetime.now() - datetime.timedelta(days=학습할기간)).strftime('%Y-%m-%d %H:%M:%S')
                end_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data = get_historical_klines(SYMBOL, Client.KLINE_INTERVAL_1MINUTE, start_str, end_str)
                data = dropna(data)
                (linear_model_1min, linear_model_2min, linear_model_3min, linear_model_4min, linear_model_5min, linear_model_6min, linear_model_7min, linear_model_8min, linear_model_9min, linear_model_10min,
                 rf_model_1min, rf_model_2min, rf_model_3min, rf_model_4min, rf_model_5min, rf_model_6min, rf_model_7min, rf_model_8min, rf_model_9min, rf_model_10min) = train_models(data)
                last_train_time = time.time()

            # 최근 데이터 가져오기
            end_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            start_str_recent = (datetime.datetime.now() - datetime.timedelta(hours=LOOKBACK_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
            recent_data = get_historical_klines(SYMBOL, Client.KLINE_INTERVAL_1MINUTE, start_str_recent, end_str)

            # 기술적 지표 추가
            recent_data = calculate_indicators(recent_data)

            # 결측치 제거
            recent_data.dropna(inplace=True)

            # 피처 추출
            recent_features = recent_data.tail(1).drop(['target_1min', 'target_2min', 'target_3min', 'target_4min', 'target_5min', 'target_6min', 'target_7min', 'target_8min', 'target_9min', 'target_10min'], axis=1, errors='ignore')

            if not recent_features.empty:
                pred_1min_linear = linear_model_1min.predict(recent_features)
                pred_2min_linear = linear_model_2min.predict(recent_features)
                pred_3min_linear = linear_model_3min.predict(recent_features)
                pred_4min_linear = linear_model_4min.predict(recent_features)
                pred_5min_linear = linear_model_5min.predict(recent_features)
                pred_6min_linear = linear_model_6min.predict(recent_features)
                pred_7min_linear = linear_model_7min.predict(recent_features)
                pred_8min_linear = linear_model_8min.predict(recent_features)
                pred_9min_linear = linear_model_9min.predict(recent_features)
                pred_10min_linear = linear_model_10min.predict(recent_features)

                pred_1min_rf = rf_model_1min.predict(recent_features)
                pred_2min_rf = rf_model_2min.predict(recent_features)
                pred_3min_rf = rf_model_3min.predict(recent_features)
                pred_4min_rf = rf_model_4min.predict(recent_features)
                pred_5min_rf = rf_model_5min.predict(recent_features)
                pred_6min_rf = rf_model_6min.predict(recent_features)
                pred_7min_rf = rf_model_7min.predict(recent_features)
                pred_8min_rf = rf_model_8min.predict(recent_features)
                pred_9min_rf = rf_model_9min.predict(recent_features)
                pred_10min_rf = rf_model_10min.predict(recent_features)

                # 예측값 평균 계산
                avg_pred_1min = (pred_1min_linear + pred_1min_rf) / 2
                avg_pred_2min = (pred_2min_linear + pred_2min_rf) / 2
                avg_pred_3min = (pred_3min_linear + pred_3min_rf) / 2
                avg_pred_4min = (pred_4min_linear + pred_4min_rf) / 2
                avg_pred_5min = (pred_5min_linear + pred_5min_rf) / 2
                avg_pred_6min = (pred_6min_linear + pred_6min_rf) / 2
                avg_pred_7min = (pred_7min_linear + pred_7min_rf) / 2
                avg_pred_8min = (pred_8min_linear + pred_8min_rf) / 2
                avg_pred_9min = (pred_9min_linear + pred_9min_rf) / 2
                avg_pred_10min = (pred_10min_linear + pred_10min_rf) / 2

                # 현재 가격 가져오기
                current_price = float(client.futures_symbol_ticker(symbol=SYMBOL)['price'])
                print(f"Current Price: {current_price}")
                print(f"1 min Prediction: {avg_pred_1min[-1]}")
                print(f"2 min Prediction: {avg_pred_2min[-1]}")
                print(f"3 min Prediction: {avg_pred_3min[-1]}")
                print(f"4 min Prediction: {avg_pred_4min[-1]}")
                print(f"5 min Prediction: {avg_pred_5min[-1]}")
                print(f"6 min Prediction: {avg_pred_6min[-1]}")
                print(f"7 min Prediction: {avg_pred_7min[-1]}")
                print(f"8 min Prediction: {avg_pred_8min[-1]}")
                print(f"9 min Prediction: {avg_pred_9min[-1]}")
                print(f"10 min Prediction: {avg_pred_10min[-1]}")

                # 포지션 진입 및 종료 로직
                if bot.current_position is None:
                    롱 = avg_pred_10min[-1] > current_price * (1 + THRESHOLD / 100)
                    숏 = avg_pred_10min[-1] < current_price * (1 + THRESHOLD / 100)

                    print("long : ", avg_pred_10min[-1] > current_price * (1 + THRESHOLD / 100))
                    print("short : ", avg_pred_10min[-1] < current_price * (1 + THRESHOLD / 100))
                    if 롱 and (recent_data['atr'].iloc[-1] > 최소atr값):
                        print("long if 문 진입")
                        bot.long_position()
                        simul.long_position()
                        simul.now_price()
                        bot.initial_pred = avg_pred_10min[-1]  # 초기 예측 값 저장
                    elif 숏 and (recent_data['atr'].iloc[-1] > 최소atr값):
                        print("short if 문 진입")
                        bot.short_position()
                        simul.short_position()
                        simul.now_price()
                        bot.initial_pred = avg_pred_10min[-1]  # 초기 예측 값 저장
                    else:
                        print(f" Atr 임계값({최소atr값})을 넘기지 못함. 현재 값: {recent_data['atr'].iloc[-1]}")
                else:
                    bot.remaining_minutes -= 1
                    if bot.remaining_minutes <= 0:
                        bot.close_position()
                        simul.stop_position()
                        continue

                    if bot.remaining_minutes == 9:
                        current_remaining_pred = avg_pred_9min
                    elif bot.remaining_minutes == 8:
                        current_remaining_pred = avg_pred_8min
                    elif bot.remaining_minutes == 7:
                        current_remaining_pred = avg_pred_7min
                    elif bot.remaining_minutes == 6:
                        current_remaining_pred = avg_pred_6min
                    elif bot.remaining_minutes == 5:
                        current_remaining_pred = avg_pred_5min
                    elif bot.remaining_minutes == 4:
                        current_remaining_pred = avg_pred_4min
                    elif bot.remaining_minutes == 3:
                        current_remaining_pred = avg_pred_3min
                    elif bot.remaining_minutes == 2:
                        current_remaining_pred = avg_pred_2min
                    elif bot.remaining_minutes == 1:
                        current_remaining_pred = avg_pred_1min

                    print(f"Initial 10 min Prediction: {bot.initial_pred}")
                    print(f"Current Prediction for {bot.remaining_minutes + 1} min ahead: {current_remaining_pred[-1]}")
                    print("현재 오차율: ", -(bot.initial_pred - current_remaining_pred[-1]) / bot.initial_pred if bot.current_position == 'buy' else (bot.initial_pred - current_remaining_pred[-1]) / bot.initial_pred)
                    print("현재 ATR값: ", recent_data['atr'].iloc[-1])
                    simul.now_price() #시뮬레이션 잔고 표시

                    if bot.current_position == 'BUY' and -(bot.initial_pred - current_remaining_pred[-1]) / bot.initial_pred < -THRESHOLD:
                        bot.close_position()
                        simul.stop_position()
                        print("3분 대기")
                        time.sleep(180)  # 3분 대기
                        continue
                    elif bot.current_position == 'SELL' and (bot.initial_pred - current_remaining_pred[-1]) / bot.initial_pred < -THRESHOLD:
                        bot.close_position()
                        simul.stop_position()
                        print("3분 대기")
                        time.sleep(180)  # 3분 대기
                        continue

            # 1분 대기 후 다시 예측
            time.sleep(60)
        except Exception as e:
            print(f"Error in main loop: {e}")
            print(traceback.format_exc())
            time.sleep(60)


if __name__ == "__main__":
    main()
