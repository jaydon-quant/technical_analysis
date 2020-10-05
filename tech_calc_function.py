# coding=utf-8
import numpy as np





def stochastic_oscillator(closing_price,k,d):
    #가장 최근의(마지막 trade_price)
    l_k = []
    h_k = []
    per_k = []
    per_d = []
    for i, val in enumerate(closing_price):
        if i < k-1:
            l_k.append(0)
            h_k.append(0)
            per_k.append(0)
        else:
            l_val = np.min(closing_price[i-k+1:i+1])
            h_val = np.max(closing_price[i-k+1:i+1])
            l_k.append(l_val)
            h_k.append(h_val)
            per_k_val = 100*(val-l_val)/(h_val-l_val)
            per_k.append(per_k_val)

    for i, val in enumerate(per_k):
        if i < k+d-1:
            per_d.append(0)
        else:
            per_d.append(np.average(per_k[i-d:i+1]))
    return {
        'l_k':l_k,
        'h_k':h_k,
        'per_k':per_k,
        'per_d':per_d,
    }


"""
reference : https://www.macroption.com/rsi-calculation/

α = 1 / N

and therefore 1 – α = ( N – 1 ) / N

N = RSI period

For example, for RSI 14 the formula for average up move is:

AvgUt = 1/14 * Ut + 13/14 * AvgUt-1
"""


# simple moving average
def sma(row, n):
    if len(row) == 0: return 0
    return np.sum(row) / n


# exponential moving average
def ema(prev_avg, incoming_val, n):
    alpha = 2 / (n + 1)
    return_val = alpha * incoming_val + (1 - alpha) * prev_avg
    return return_val


def get_ema_row(row, n):
    ema_row = []
    for i, val in enumerate(row):
        if i < n-1:
            ema_row.append(-1)
        elif i == n-1:
            input_row = row[i - n + 1:i + 1]
            ema_row.append(np.average(input_row))

        else:
            ema_val = ema(ema_row[-1], val, n)
            ema_row.append(ema_val)

    return ema_row



# wilder's smoothing method
def wilder(row, n):
    if len(row) == 0: return 0
    alpha = 1 / n
    for i, value in enumerate(row):
        if i == 0:
            avg_t = value
        else:
            avg_t = alpha * value + (1 - alpha) * avg_t
    return avg_t


def rsi(closing_price, n=14, avg_method=sma):
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


# formula for MACD : 12 period

# MACD : 12-period ema - 26 period ema
def macd(closing_price, fast=12, slow=26, sig=9):
    if slow <= fast:
        print('ERRRR')
        return None
    ema_fast_row = np.array(get_ema_row(closing_price, fast))
    ema_slow_row = np.array(get_ema_row(closing_price, slow))
    macd_row = ema_fast_row - ema_slow_row
    macd_row[:slow] = 0
    signal_row = np.array(get_ema_row(macd_row, sig))

    return {
        'ema_fast': ema_fast_row,
        'ema_slow': ema_slow_row,
        'macd': macd_row,
        'signal': signal_row
    }


def moving_average(price_row, n):
    average_row = []
    for i, value in enumerate(price_row):
        if i < n-1: average_row.append(0)
        else:
            input_row = price_row[i - n + 1:i + 1]
            average_row.append(np.average(input_row))
    return average_row


# reference : https://coolbingle.blogspot.com/2019/03/using-spreadsheet-to-produce-dmi.html
# https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/dmi

"""
n : di_length
m : adx smoothing
이렇게 두개에 사용하는 파라미터
"""
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

        elif i == n - 1:
            avg_true_range_row.append(np.average(true_range_row[i - (n - 1):i + 1]))
            avg_dm_plus_row.append(np.average(dm_plus_row[i - (n - 1):i + 1]))
            avg_dm_minus_row.append(np.average(dm_minus_row[i - (n - 1):i + 1]))
        else:
            avg_tr_p = avg_true_range_row[-1]
            avg_dm_p_p = avg_dm_plus_row[-1]
            avg_dm_m_p = avg_dm_minus_row[-1]

            avg_tr = avg_tr_p + (tr - avg_tr_p) / n
            avg_dm_p = avg_dm_p_p + (dm_p - avg_dm_p_p) / n
            avg_dm_m = avg_dm_m_p + (dm_m - avg_dm_m_p) / n

            avg_true_range_row.append(avg_tr)
            avg_dm_plus_row.append(avg_dm_p)
            avg_dm_minus_row.append(avg_dm_m)

            di_p = avg_dm_p / avg_tr
            di_m = avg_dm_m / avg_tr
            dx = 100 * abs(avg_dm_p - avg_dm_m) / (avg_dm_p + avg_dm_m)

            di_p_row.append(di_p)
            di_m_row.append(di_m)
            dx_row.append(dx)

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



