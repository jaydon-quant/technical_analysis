"""
기본적으로 helper function 으로 사용하는 부분들 호출하기
"""

from .data_load import DataLoad
from .data_load import DBLoad
from .technical_indicators import Indicator
import pandas as pd
import numpy as np


class TechDataLoad:
    def __init__(self):
        return

    @staticmethod
    def get_tech_indicator(chart_info):

        o_price = chart_info['open']
        h_price = chart_info['high']
        l_price = chart_info['low']
        c_price = chart_info['close']

        ## MACD
        macd_test = Indicator.macd(c_price)
        chart_info['macd-gap'] = macd_test['macd'] - macd_test['signal']
        ## 이평선
        ma_5 = Indicator.moving_average(c_price, 5)
        ma_20 = Indicator.moving_average(c_price, 20)
        ma_60 = Indicator.moving_average(c_price, 60)
        chart_info['ma-20-60'] = np.array(ma_20) - np.array(ma_60)
        chart_info['ma-5-20'] = np.array(ma_5) - np.array(ma_20)
        ## RSI
        rsi_test = Indicator.rsi(c_price)
        chart_info['rsi'] = rsi_test['rsi']

        ## OSCILLATOR
        oscillator = Indicator.stochastic_oscillator(c_price)
        chart_info['per_k'] = oscillator['per_k']
        chart_info['per_d'] = oscillator['per_d']
        chart_info['stochastic'] = np.array(oscillator['per_k']) - np.array(oscillator['per_d'])

        ## DMI
        dmi = Indicator.dmi(o_price, h_price, l_price, c_price)
        chart_info['dmi_signal'] = np.array(dmi['di_plus']) - np.array(dmi['di_minus'])

        ## Bollinger band
        bollinger = Indicator.bollinger_band(c_price)
        chart_info['bollinger_up'] = bollinger['b_upper']
        chart_info['bollinger_middle'] = bollinger['b_middle']
        chart_info['bollinger_down'] = bollinger['b_lower']
        # bollinger_band 포지션 위치
        chart_info['bollinger_position'] = bollinger['b_position']

        #future_5 = np.append(c_price[5:], np.zeros(5))
        #chart_info['future-5'] = future_5
        #future_20 = np.append(c_price[20:], np.zeros(20))
        #chart_info['future-20'] = future_20
        #future_60 = np.append(c_price[60:], np.zeros(60))
        #chart_info['future-60'] = future_60

        #chart_info_fixed = chart_info[100:].reset_index(drop=True)
        #chart_info_fixed['yld_5'] = 1 - (chart_info_fixed['future-5'] / chart_info_fixed['close'])
        #chart_info_fixed['yld_20'] = 1 - (chart_info_fixed['future-20'] / chart_info_fixed['close'])
        #chart_info_fixed['yld_60'] = 1 - (chart_info_fixed['future-60'] / chart_info_fixed['close'])

        return chart_info

    """
    date : 데이터 최초 시작일
    """
    @staticmethod
    def load_dataset(code, N=4200, mode='KR_API', date=None):
        # 코드 추출 #
        if mode == 'KR_API':
            stock_info = DataLoad(code=code, candle_type='1d', total_candle_length=N)
            daily_info = pd.DataFrame(data=stock_info.data_by_category)[::-1].reset_index(drop=True)

        elif mode == 'KR_quantDB':
            daily_info = DBLoad.kr_quant_db(code, date)
            daily_info = daily_info.rename(columns={"open_prc": "open", "high_prc": "high",
                                                    "low_prc": "low", "cls_prc": "close",
                                                    "trd_dt": "datetime"})
        elif mode == 'US':
            daily_info = DBLoad.us_market_db(code)
            daily_info = daily_info.rename(columns={"opening_price": "open", "high_price": "high",
                                                    "low_price": "low", "trade_price": "close",
                                                    "date": "datetime"})

        daily_info_fixed = TechDataLoad.get_tech_indicator(daily_info)

        return daily_info_fixed



########################


