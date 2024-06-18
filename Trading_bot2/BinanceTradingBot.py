import time
import traceback

# 로깅 설정
log_path = 'trading_log.txt'

class BinanceTrading:
    def __init__(self, client, symbol, leverage, balance_percentage):
        self.client = client
        self.symbol = symbol
        self.leverage = leverage
        self.balance_percentage = balance_percentage
        self.client = client
        self.current_position = None
        self.entry_price = None
        self.initial_pred = None  # 진입 시점의 예측 값 저장 변수
        self.remaining_minutes = 10  # 초기 10분 후 예측

    def get_balance(self):
        try:
            balance_info = self.client.futures_account_balance()
            for balance in balance_info:
                if balance['asset'] == 'USDT':
                    print(f"Balance fetched: {balance['balance']}")
                    return float(balance['balance'])
        except Exception as e:
            print(f"Error fetching balance: {e}")
        return 0.0

    def get_symbol_info(self):
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == self.symbol:
                    print(f"Symbol info: {s}")
                    return s
        except Exception as e:
            print(f"Error fetching symbol info: {e}")
        return None

    def enter_position(self, side, amount):
        print("enter_position 진입")
        symbol_info = self.get_symbol_info()
        if symbol_info:
            print(f"Symbol info: {symbol_info}")
            step_size = float(symbol_info['filters'][2]['stepSize'])
            min_notional = None
            for f in symbol_info['filters']:
                if 'minNotional' in f:
                    min_notional = float(f['minNotional'])
                elif 'notional' in f:
                    min_notional = float(f['notional'])

            print(f"Step size: {step_size}, Min notional: {min_notional}")

            amount = round(amount - (amount % step_size), 8)
            price = float(self.client.futures_symbol_ticker(symbol=self.symbol)['price'])
            required_margin = amount * price / self.leverage

            print(f"Amount: {amount}, Price: {price}, Required margin: {required_margin}")

            if min_notional and amount * price < min_notional:
                print(f"Order amount is below the minimum notional value of {min_notional}")
                return None

            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            try:
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type='MARKET',
                    quantity=amount
                )
                print(f"Order placed: {order}")
                order_id = order['orderId']
                start_time = time.time()

                while True:
                    order_status = self.client.futures_get_order(symbol=self.symbol, orderId=order_id)
                    print(f"Order status: {order_status}")
                    if order_status['status'] == 'FILLED':
                        self.entry_price = float(self.client.futures_symbol_ticker(symbol=self.symbol)['price'])  # 진입가 저장
                        self.current_position = side  # 현재 포지션 저장
                        print(f"Position entered: {side} at {self.entry_price}")
                        return order
                    if time.time() - start_time > 3:
                        self.client.futures_cancel_order(symbol=self.symbol, orderId=order_id)
                        print("Order not filled within 3 seconds, retrying...")
                        raise Exception("Order not filled within 3 seconds")
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error entering position: {e}")
                print(traceback.format_exc())
                return None

        return None

    def close_position(self):
        try:
            positions = self.client.futures_position_information()
            for position in positions:
                if position['symbol'] == self.symbol:
                    if float(position['positionAmt']) != 0:
                        side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
                        amount = abs(float(position['positionAmt']))
                        try:
                            order = self.client.futures_create_order(
                                symbol=self.symbol,
                                side=side,
                                type='MARKET',
                                quantity=amount
                            )
                            self.current_position = None
                            self.initial_pred = None  # 초기 예측 값 리셋
                            self.remaining_minutes = 5  # 예측 시간 초기화
                            self.log_trade(self.entry_price,
                                           float(self.client.futures_symbol_ticker(symbol=self.symbol)['price']))
                            return order
                        except Exception as e:
                            print(f"Error closing position: {e}")
                            print(traceback.format_exc())
                            return None
        except Exception as e:
            print(f"Error fetching positions: {e}")
            print(traceback.format_exc())
            return None

    def log_trade(self, entry_price, exit_price):
        try:
            with open(log_path, 'a') as log_file:
                log_file.write(f"Entry Price: {entry_price}\n")
                log_file.write(f"Exit Price: {exit_price}\n")
                profit = (exit_price - entry_price) * 100 * self.leverage / entry_price  # 백분율 계산
                log_file.write(f"Profit: {profit:.2f}%\n\n")
        except Exception as e:
            print(f"Error logging trade: {e}")
            print(traceback.format_exc())

    def long_position(self):
        print("long_position 진입")
        balance = self.get_balance()
        amount = round((balance * self.balance_percentage * self.leverage) / float(self.client.futures_symbol_ticker(symbol=self.symbol)['price']), 3)
        print(f"Trying to enter long position with amount: {amount}")
        self.enter_position('BUY', amount)

    def short_position(self):
        print("short_position 진입")
        balance = self.get_balance()
        amount = round((balance * self.balance_percentage * self.leverage) / float(self.client.futures_symbol_ticker(symbol=self.symbol)['price']), 3)
        print(f"Trying to enter short position with amount: {amount}")
        self.enter_position('SELL', amount)
