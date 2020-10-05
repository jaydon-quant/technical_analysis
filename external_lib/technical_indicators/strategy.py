# coding=utf-8

"""
STRATEGY

투자 전략 Code들

기본적인 투자 전략에에 관련된 code

Strategy 1 : 볼린저밴드 + MACD

Strategy 2 : RSI Divergence

Strategy 3 : 지지선 / 저항

"""


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
from scipy import stats
import matplotlib.pyplot as plt
import pwlf
import datetime
import seaborn as sns
from scipy.spatial import distance
import external_lib
import matplotlib
from pandas.plotting import register_matplotlib_converters
import external_lib.technical_indicators as tech
from sklearn.model_selection import train_test_split
import xgboost
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix

import itertools
from sklearn.linear_model import LinearRegression


class Strategy:
    def __init__(self):
        return


    """
    input : price_high, price_low, bollinger_up, bollinger_down, macd_gap
    각각 고가 / 저가 / 볼린저 상단 / 하단 / macd gap (히스토그램 값)    
    """
    @staticmethod
    def macd_bollinger(price_high, price_low, bollinger_up, bollinger_down, macd_gap):
        # 해당 주식의 정보를 loading
        total_len = len(price_high)
        index_row = np.linspace(0, total_len - 1, total_len, dtype=int)

        # bollinger band가 높은 경우
        # condition 1) macd >0  / 2) macd peaked max (-> decreasing)
        high_index = index_row[bollinger_up <= price_high]
        high_index_modified = []
        counted_index_h = []

        for index in high_index:
            if index in counted_index_h:
                continue
            checker = False
            i = 0
            while (not checker):

                counted_index_h.append(index + i)
                pre_index = index + i - 1
                current_index = index + i

                if current_index > total_len - 1:
                    break

                if i > 0:
                    macd_pre = macd_gap[pre_index]
                    macd = macd_gap[current_index]
                    if macd_pre > macd:
                        high_index_modified.append(index + i)
                        checker = True
                i += 1

        # bollinger band가 낮은 경우
        # condition 2) macd <0  / 2) macd peaked min (-> increasing)
        low_index = index_row[bollinger_down >= price_low]
        low_index_modified = []
        counted_index = []

        for index in low_index:
            if index in counted_index:
                continue
            checker2 = False
            i = 0
            while not checker2:
                counted_index.append(index + i)
                pre_index = index + i - 1
                current_index = index + i

                if current_index > total_len - 1:
                    break

                if i > 0:
                    macd_pre = macd_gap[pre_index]
                    macd = macd_gap[current_index]
                    if macd_pre < macd:
                        low_index_modified.append(index + i)
                        checker2 = True
                i += 1

        status = np.zeros(total_len)
        status[high_index_modified] = 1
        status[low_index_modified] = -1

        return status


    # x_values : x point들 / price_arr : 전체 price_array
    # 조합들중 가장 rsq + 포함된 점의 개수의 비율점수가 높은 것들을 return 한다
    @staticmethod
    def find_resistance_line_single_run(x_values, price_arr, pt_num=3):
        # 점들의 combination 을 계산한다.
        pt_combs = list(itertools.combinations(x_values, pt_num))

        rsq_scores = []
        for comb in pt_combs:
            x_points = np.array(comb)
            x_points_fixed = x_points.reshape((-1, 1))
            y_points = price_arr[x_points]
            # Linear regresssion 만들기
            reg = LinearRegression().fit(x_points_fixed, y_points)
            r_sq = reg.score(x_points_fixed, y_points)
            rsq_scores.append(r_sq)

        # pd_format 으로 치환 뒤 정리
        pd_format = pd.DataFrame(data={'comb': pt_combs, 'rsq': rsq_scores})
        pd_format = pd_format.sort_values(by='rsq')[::-1].reset_index(drop=True)[:5]

        return pd_format.to_numpy()

    @staticmethod
    def get_resistance_lines(high_price, low_price, close_price):
        zigzag_info = tech.zigzag.ZigZag(high_price, low_price, deviation=0.25)
        pts = zigzag_info.pivot_points

        # 시작점이 max인지 min인지 확인하기
        if close_price[pts].values[0] > close_price[pts].values[1]:
            start_max = True
        else:
            start_max = False

        if start_max:
            up_counter = 0
        else:
            up_counter = 1

        # 우선적으로 상승 - 하락 점들을 확인
        up_pts, down_pts = [], []

        for i, pt in enumerate(pts):
            if i % 2 == up_counter:
                up_pts.append(pt)
            else:
                down_pts.append(pt)

        line_down = Strategy.find_resistance_line_single_run(down_pts, close_price, 5)
        line_up = Strategy.find_resistance_line_single_run(up_pts, close_price, 5)

        return_info = {
            'up': line_up,
            'down': line_down
        }
        return return_info





