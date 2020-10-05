# coding=utf-8
"""
Created by Jayden.jeon @ Dunamu datavalue lab

다음 스크립트에서 포함하고 있는 기술적 지표는 다음과 같다.

대부분의 계산에서 평균을 계산할때 사용하는 period-length 는 변수로 남겨두었다.
(default setting 의 경우 trading view 의 세팅과 동일하게 해놓음)


########################################################
########################################################
########################################################
Stochastic Oscillator :

reference
https://www.investopedia.com/terms/s/stochasticoscillator.asp#:~:text=A%20stochastic%20oscillator%20is%20a,moving%20average%20of%20the%20result.

스토캐스틱 지표

input
closing_price (list):  종가의 리스트
K (int): Slow stochastic 추출 파라미터
D (int): Fast stochastic 추출 파라미터

output

dictionary
{
    'l_k': K 기간동안의 최저가
    'h_k': K 기간동안의 최고가
    'per_k': Stochastic indicator (slow)
    'per_d': Stochastic indicator (fast)
}

########################################################
########################################################
########################################################
RSI :

reference : https://www.macroption.com/rsi-calculation/

α = 1 / N
and therefore 1 – α = ( N – 1 ) / N
where, N = RSI period

For example, if N is 14, RSI formula for average up move is:
AvgUt = 1/14 * Ut + 13/14 * AvgUt-1


input
closing_price (list):  종가의 리스트
n (int): RSI period
avg_method (function): 평균 계산 법

output

dictionary
{
    'avg_u': 평균 상승
    'avg_d': 평균 하락
    'rsi': rsi value
}



########################################################
########################################################
########################################################
MACD :

reference : https://www.investopedia.com/terms/m/macd.asp

MACD 계산법 : ema_fast_row - ema_slow_row
signal 계산법 : macd_row 의 ema

input
closing_price (list): 종가의 리스트
fast (int): fast_ema period
slow (int): slow_ema period
signal (int): signal_ema period

output

dictionary
{
    'ema_fast': ema_fast_row,
    'ema_slow': ema_slow_row,
    'macd': macd_row,
    'signal': signal_row
}


########################################################
########################################################
########################################################
moving average :

reference : https://www.investopedia.com/terms/m/macd.asp

MACD 계산법 : ema_fast_row - ema_slow_row
signal 계산법 : macd_row 의 ema

input
price_row (list): 가격의 리스트
n (int): period

output
average_row (list) : 평균가격 리스트


########################################################
########################################################
########################################################
dmi (Directional Movement Index) :

reference :
https://coolbingle.blogspot.com/2019/03/using-spreadsheet-to-produce-dmi.html
https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/dmi


input
price_open (list): 시가 리스트
price_high (list): 고가 리스트
price_low (list): 저가 리스트
price_close (list): 종가 리스트
n (int): di_length parameter
m (int): adx_smoothing parameter

output
dictionary
{
    'di_plus': di_p_row,
    'di_minus': di_m_row,
    'avg_dx': avg_dx_row
}


########################################################
########################################################
########################################################
Bollinger (Bollinger band) :

reference :
https://www.investopedia.com/terms/b/bollingerbands.asp

input
price_row (list): 가격 리스트
length (int): used to calculate ma
stdev (int): bollinger band range

output
dictionary
{
    'b_upper': bollinger_upper,
    'b_middle': moving_average,
    'b_lower': bollinger_lower,
}

"""

import numpy as np
from . import zigzag
from . import strategy

class Helper:
    def __init__(self):
        return

    # simple moving average
    @staticmethod
    def sma(row, n):
        if len(row) == 0: return 0
        return np.sum(row) / n

    # exponential moving average
    @staticmethod
    def ema(prev_avg, incoming_val, n):
        alpha = 2 / (n + 1)
        return_val = alpha * incoming_val + (1 - alpha) * prev_avg
        return return_val

    @staticmethod
    def get_ema_row(row, n):
        ema_row = []
        for i, val in enumerate(row):
            if i < n - 1:
                ema_row.append(-1)
            elif i == n - 1:
                input_row = row[i - n + 1:i + 1]
                ema_row.append(np.average(input_row))

            else:
                ema_val = Helper.ema(ema_row[-1], val, n)
                ema_row.append(ema_val)

        return ema_row


    # wilder's smoothing method
    @staticmethod
    def wilder(row, n):
        avg_t = 0
        if len(row) == 0: return 0
        alpha = 1 / n
        for i, value in enumerate(row):
            if i == 0:
                avg_t = value
            else:
                avg_t = alpha * value + (1 - alpha) * avg_t
        return avg_t





class Indicator:
    def __init__(self):
        return

    @staticmethod
    def stochastic_oscillator(closing_price, k=14, d=3):
        # 가장 최근의(마지막 trade_price)
        l_k = []
        h_k = []
        per_k = []
        per_d = []
        for i, val in enumerate(closing_price):
            if i < k - 1:
                l_k.append(0)
                h_k.append(0)
                per_k.append(0)
            else:
                l_val = np.min(closing_price[i - k + 1:i + 1])
                h_val = np.max(closing_price[i - k + 1:i + 1])
                l_k.append(l_val)
                h_k.append(h_val)
                per_k_val = 100 * (val - l_val) / (h_val - l_val)
                per_k.append(per_k_val)

        for i, val in enumerate(per_k):
            if i < k + d - 1:
                per_d.append(0)
            else:
                per_d.append(np.average(per_k[i - d:i + 1]))
        return {
            'l_k': l_k,
            'h_k': h_k,
            'per_k': per_k,
            'per_d': per_d,
        }


    @staticmethod
    def rsi(closing_price, n=14, avg_method=Helper.sma):
        # rsi step1
        # 100 - (100/1+ avg_gain/avg_loss)
        # get n days info
        date_val = closing_price[1:].values
        pre_date_val = closing_price[:-1].values
        change_row = np.append(0, date_val - pre_date_val)

        avg_u = []
        avg_d = []
        rsi_row = []

        for i, val in enumerate(closing_price):
            if i < n - 1:
                avg_u.append(0)
                avg_d.append(0)
                rsi_row.append(0)
            else:
                temp_row = change_row[i - n:i + 1]
                u_row = temp_row[temp_row >= 0]
                d_row = np.abs(temp_row[temp_row < 0])
                avg_u_val = avg_method(u_row, n)
                avg_d_val = avg_method(d_row, n)
                if avg_d_val == 0:
                    rsi = 1
                else:
                    rs = avg_u_val / avg_d_val
                    rsi = 100 * (1 - 1 / (1 + rs))
                avg_u.append(avg_u_val)
                avg_d.append(avg_d_val)
                rsi_row.append(rsi)

        return {
            'avg_u': avg_u,
            'avg_d': avg_d,
            'rsi': rsi_row
        }



    @staticmethod
    def macd(closing_price, fast=12, slow=26, sig=9):
        if slow <= fast:
            print('ERRRR')
            return None
        ema_fast_row = np.array(Helper.get_ema_row(closing_price, fast))
        ema_slow_row = np.array(Helper.get_ema_row(closing_price, slow))
        macd_row = ema_fast_row - ema_slow_row
        macd_row[:slow] = 0
        signal_row = np.array(Helper.get_ema_row(macd_row, sig))

        return {
            'ema_fast': ema_fast_row,
            'ema_slow': ema_slow_row,
            'macd': macd_row,
            'signal': signal_row
        }



    @staticmethod
    def moving_average(price_row, n):
        average_row = []
        for i, value in enumerate(price_row):
            if i < n - 1:
                average_row.append(0)
            else:
                input_row = price_row[i - n + 1:i + 1]
                average_row.append(np.average(input_row))
        return average_row


    @staticmethod
    def dmi(price_open, price_high, price_low, price_close, n=14, m=14):
        true_range_row = []
        dm_plus_row = []
        dm_minus_row = []

        for i, val in enumerate(zip(price_open, price_high, price_low, price_close)):
            if i == 0:
                true_range_row.append(0)
                dm_plus_row.append(0)
                dm_minus_row.append(0)
                continue

            (o, h, l, c) = val
            prev_c = price_close[i - 1]
            prev_h = price_high[i - 1]
            prev_l = price_low[i - 1]

            candidate_1 = h - l
            candidate_2 = abs(h - prev_c)
            candidate_3 = abs(l - prev_c)

            true_range = max(candidate_1, candidate_2, candidate_3)

            if h - prev_h > prev_l - l:
                dm_plus0 = max(h - prev_h, 0)
            else:
                dm_plus0 = 0

            if prev_l - l > h - prev_h:
                dm_minus0 = max(prev_l - l, 0)
            else:
                dm_minus0 = 0

            if dm_plus0 > dm_minus0:
                dm_plus = dm_plus0
                dm_minus = 0

            elif dm_plus0 < dm_minus0:
                dm_minus = dm_minus0
                dm_plus = 0
            else:
                dm_minus, dm_plus = 0, 0

            true_range_row.append(true_range)
            dm_plus_row.append(dm_plus)
            dm_minus_row.append(dm_minus)

        avg_true_range_row = []
        avg_dm_plus_row = []
        avg_dm_minus_row = []

        di_p_row = []
        di_m_row = []
        dx_row = []

        for i, val in enumerate(zip(true_range_row, dm_plus_row, dm_minus_row)):
            (tr, dm_p, dm_m) = val

            if i < n - 1:
                avg_true_range_row.append(0)
                avg_dm_plus_row.append(0)
                avg_dm_minus_row.append(0)

                di_p_row.append(0)
                di_m_row.append(0)
                dx_row.append(0)
                continue

            elif i == n - 1:
                avg_tr = np.average(true_range_row[i - (n - 1):i + 1])
                avg_dm_p = np.average(dm_plus_row[i - (n - 1):i + 1])
                avg_dm_m = np.average(dm_minus_row[i - (n - 1):i + 1])

            else:
                avg_tr_p = avg_true_range_row[-1]
                avg_dm_p_p = avg_dm_plus_row[-1]
                avg_dm_m_p = avg_dm_minus_row[-1]

                avg_tr = avg_tr_p + (tr - avg_tr_p) / n
                avg_dm_p = avg_dm_p_p + (dm_p - avg_dm_p_p) / n
                avg_dm_m = avg_dm_m_p + (dm_m - avg_dm_m_p) / n


            di_p = avg_dm_p / avg_tr
            di_m = avg_dm_m / avg_tr
            dx = 100 * abs(avg_dm_p - avg_dm_m) / (avg_dm_p + avg_dm_m)

            di_p_row.append(di_p)
            di_m_row.append(di_m)
            dx_row.append(dx)

            avg_true_range_row.append(avg_tr)
            avg_dm_plus_row.append(avg_dm_p)
            avg_dm_minus_row.append(avg_dm_m)




        avg_dx_row = []

        for i, dx in enumerate(dx_row):
            if (dx == 0) or (i < m - 1):
                avg_dx_row.append(0)
                continue
            # 맨 처음 들어오는 value
            if avg_dx_row[-1] == 0:
                avg_dx_row.append(dx)

            else:
                adx_p = avg_dx_row[-1]
                adx = adx_p + (dx - adx_p) / m
                avg_dx_row.append(adx)

        return {
            'di_plus': di_p_row,
            'di_minus': di_m_row,
            'avg_dx': avg_dx_row
        }


    @staticmethod
    def bollinger_band(price_row, length=20, stdev=2):
        average_row = []
        upper_row = []
        lower_row = []
        position_row = []

        for i, value in enumerate(price_row):
            if i < length - 1:
                average_row.append(0)
                upper_row.append(0)
                lower_row.append(0)
                position_row.append(0)
            else:
                input_row = price_row[i - length + 1:i + 1]
                average_row.append(np.average(input_row))
                stdev_val = np.std(input_row)
                upper_row.append(np.average(input_row)+stdev*stdev_val)
                lower_row.append(np.average(input_row)-stdev*stdev_val)
                # 현재 가격이 볼린저밴드에서 위치한 곳
                position = (value-np.average(input_row))/stdev_val
                position_row.append(position)

        return {
            'b_upper': upper_row,
            'b_middle': average_row,
            'b_lower': lower_row,
            'b_position': position_row
        }


