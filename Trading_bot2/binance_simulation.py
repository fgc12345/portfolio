import ccxt

class Simulation:
    def __init__(self, client, symbol, leverage, balance_percentage, initial_balance):
        self.client = client
        self.symbol = symbol
        self.leverage = leverage
        self.balance_percentage = balance_percentage
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = None
        self.entry_price = None

    def get_market_price(self):
        try:
            balance_info = self.client.futures_account_balance()
            for balance in balance_info:
                if balance['asset'] == 'USDT':
                    print(f"Balance fetched: {balance['balance']}")
                    return float(balance['balance'])
        except Exception as e:
            print(f"Error fetching balance: {e}")
        return None

    def calculate_position_size(self):
        position_size = (self.balance * self.balance_percentage) / 100
        return position_size * self.leverage

    def long_position(self):
        if self.position is not None:
            print("이미 포지션이 존재합니다.")
            return
        market_price = self.get_market_price()
        if market_price is not None:
            position_size = self.calculate_position_size()
            self.position = 'long'
            self.entry_price = market_price
            print(f"롱 포지션 진입: 크기={position_size}, 진입가={market_price}, 현재 잔고={self.balance} ")

    def short_position(self):
        if self.position is not None:
            print("이미 포지션이 존재합니다.")
            return
        market_price = self.get_market_price()
        if market_price is not None:
            position_size = self.calculate_position_size()
            self.position = 'short'
            self.entry_price = market_price
            print(f"숏 포지션 진입: 크기={position_size}, 진입가={market_price}, 현재 잔고={self.balance}")

    def stop_position(self):
        if self.position is None:
            print("현재 포지션이 없습니다.")
            return
        market_price = self.get_market_price()
        if market_price is not None:
            pnl = 0
            position_size = self.calculate_position_size()
            if self.position == 'long':
                pnl = (market_price - self.entry_price) * position_size
            elif self.position == 'short':
                pnl = (self.entry_price - market_price) * position_size

            self.balance += pnl
            print(f"포지션 종료: 종료가={market_price}, PnL={pnl}, 현재 잔고={self.balance}")
            self.position = None
            self.entry_price = None

    def now_price(self):
        pnl = self.balance - self.initial_balance
        print(f" 현재 잔고: {self.balance}, PnL : {pnl}")