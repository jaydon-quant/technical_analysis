# coding=utf-8

"""
ZigZag

Deviation 과 / Depth 에 따라서 패턴을 분류한다
중요 : 항상 퍼센트의 변화를 따질때는 (고-저)/저  로 계산한다.

zigzag :

single
하나의 Array 로만 계산할때 (예 : 종가로만 계산)
input array low -> 종가 insert

double
두개의 Array 로 계산할때 ( 고가/저가)
input_array_low -> 저가 insert
input_array_low -> 고가 insert

"""

import numpy as np


class ZigZag:

    def __init__(self, array1, array2=None, deviation=0.05, depth=5):

        # 인풋에 대해서는 자동으로 numpy 로 바꿔주기
        self.array1 = np.array(array1)
        if array2 is None:
            self.array2 = None
        else:
            self.array2 = np.array(array2)

        self.deviation = deviation
        self.depth = depth
        self.pivot_points, self.pivot_prices, self.pivot_stats = None, None, None
        self.high, self.low = None, None
        # High 와 Low 잡아주기
        self.find_high_low()
        self.run_zigzag()

    # with assumption array2 is not None
    def find_high_low(self):
        try:
            test = self.array1-self.array2
        except:
            return
        # test1 : 고가 / test2 : 저가인 경우
        if len(test[test >= 0])==len(test):
            self.high, self.low = self.array1, self.array2

        # test2 : 저가 / test2 : 고가인 경우
        elif len(test[test<=0])==len(test):
            self.low, self.high = self.array1, self.array2

        # Array 2개가 low_high 의 관계가 아닐경우
        else:
            raise ValueError('고가-저가 데이터에 대해서 다시한번 확인해주세요')
        return

    def run_zigzag(self):
        if self.array2 is None:
            self.pivot_points, self.pivot_prices, self.pivot_stats = self.zigzag_single(self.array1,
                                                                                        self.deviation, self.depth)
        else:
            self.pivot_points, self.pivot_prices, self.pivot_stats = self.zigzag_double(self.low, self.high,
                                                                                        self.deviation, self.depth)
        return

    @staticmethod
    def zigzag_single(input_array, deviation=0.05, depth=5):

        # 맨 처음점은 무조건 포함하기
        key_points = []
        min_price, max_price, start_price = input_array[0], input_array[0], input_array[0]
        min_index, max_index, start_index = 0, 0, 0

        stats = []
        # 맨처음에는 Direction 이 안정해져있음
        current_direction = 0

        for index, price in enumerate(input_array):

            if min_price > price:
                min_price = price
                min_index = index

                if current_direction == -1:
                    max_candidate = index

            if max_price < price:
                max_price = price
                max_index = index

                if current_direction == 1:
                    min_candidate = index

            if current_direction == 0:

                if (start_price - min_price) / min_price >= deviation and min_index - start_index >= depth:
                    if (price - min_price) / min_price >= deviation and index - min_index >= depth:
                        # 시작점 추가
                        key_points.append(max_index)
                        # 시작점을, 최소점으로 변경
                        start_price, start_index = min_price, min_index
                        # 최대점만 현재점으로 변경해준다
                        max_price, max_index = price, index
                        min_price, min_index = price, index
                        # 방향 뒤집기
                        stats.append(-1)
                        current_direction = 1

                elif (max_price - start_price) / start_price >= deviation and max_index - start_index >= depth:
                    if (max_price - price) / price >= deviation and index - max_index >= depth:
                        # 시작점 추가
                        key_points.append(min_index)
                        # 시작점을, 최대점으로 변경
                        start_price, start_index = max_price, max_index
                        # 최소점만 현재점으로 변경해준다
                        min_price, min_index = price, index
                        max_price, max_index = price, index
                        # 방향 뒤집기
                        stats.append(1)
                        current_direction = -1

            # 만약 현재 방향의 극점과, 현재값의 ratio 가 반대방향으로 key_dev 를 넘을시,
            # 극점을 추가해주고, search 방향을 반대로 돌린다고 생각하면 된다.
            # 예: 현재 - 방향 서치중인데, (-) 방향으로 key_dev 를 넘고, 극점에서 (+) 로 key_dev 돌파시 추가해준다
            if current_direction == -1:
                if (start_price - min_price) / min_price >= deviation and min_index - start_index >= depth:
                    if (price - min_price) / min_price >= deviation and index - min_index >= depth:
                        # 시작점 추가
                        key_points.append(start_index)
                        # 시작점을, 최소점으로 변경
                        start_price, start_index = min_price, min_index
                        # 최대점만 현재점으로 변경해준다
                        max_price, max_index = price, index
                        min_price, min_index = price, index
                        # 방향 뒤집기
                        stats.append(-1)
                        current_direction = 1

            elif current_direction == 1:
                if (max_price - start_price) / start_price >= deviation and max_index - start_index >= depth:
                    if (max_price - price) / price >= deviation and index - max_index >= depth:
                        # 시작점 추가
                        key_points.append(start_index)
                        # 시작점을, 최대점으로 변경
                        start_price, start_index = max_price, max_index
                        # 최소점만 현재점으로 변경해준다
                        min_price, min_index = price, index
                        max_price, max_index = price, index
                        # 방향 뒤집기
                        stats.append(1)
                        current_direction = -1

        key_points.append(start_index)
        stats.append(current_direction)

        pivot_points, direction_stats = np.array(key_points), np.array(stats)
        pivot_prices = input_array[key_points]

        # Direction_stat -> 상승방향을 측정 할때 +1 / 하락방향을 측정할때 -1
        # Pivot_stat -> 시작점이 상승방향일때는 저점 / 하락방향일때는 고점
        pivot_stats = -1 * direction_stats

        return pivot_points, pivot_prices, pivot_stats

    @staticmethod
    def zigzag_double(input_array_low, input_array_high, deviation=0.05, depth=5):
        # 맨 처음점은 무조건 포함하기
        key_points = []
        min_price = input_array_low[0]
        max_price = input_array_high[0]
        start_price = (min_price + max_price) / 2
        min_index, max_index, start_index = 0, 0, 0
        stats = []

        # 맨처음에는 Direction이 안정해져있음
        current_direction = 0

        for index, prices in enumerate(zip(input_array_low, input_array_high)):

            low_price, high_price = prices

            if min_price > low_price:
                min_price = low_price
                min_index = index

                if current_direction == -1:
                    max_candidate = index

            if max_price < high_price:
                max_price = high_price
                max_index = index

                if current_direction == 1:
                    min_candidate = index

            # 맨 처음이기 때문에, 처음으로 deviation 을 만족하는 방향으로 설정해준다.
            if current_direction == 0:
                # Min 방향이라고 설정
                if (start_price - min_price) / min_price >= deviation and min_index - start_index >= depth:
                    if (high_price - min_price) / min_price >= deviation and index - min_index >= depth:
                        # 시작점 추가
                        key_points.append(max_index)
                        # 시작점을, 최소점으로 변경
                        start_price, start_index = min_price, min_index
                        # 최대점만 현재점으로 변경해준다
                        max_price, max_index = high_price, index
                        min_price, min_index = low_price, index
                        # 방향 뒤집기
                        current_direction = 1
                        stats.append(-1)

                # Max 방향이라고 설정
                elif (max_price - start_price) / start_price >= deviation and max_index - start_index >= depth:
                    if (max_price - low_price) / low_price >= deviation and index - max_index >= depth:
                        # 시작점 추가
                        key_points.append(min_index)
                        # 시작점을, 최대점으로 변경
                        start_price, start_index = max_price, max_index
                        # 최소점만 현재점으로 변경해준다
                        min_price, min_index = low_price, index
                        max_price, max_index = high_price, index
                        # 방향 뒤집기
                        current_direction = -1
                        stats.append(1)

            # 만약 현재 방향의 극점과, 현재값의 ratio 가 반대방향으로 key_dev 를 넘을시,
            # 극점을 추가해주고, search 방향을 반대로 돌린다고 생각하면 된다.
            # 예: 현재 - 방향 서치중인데, (-) 방향으로 key_dev 를 넘고, 극점에서 (+) 로 key_dev 돌파시 추가해준다
            if current_direction == -1:
                if (start_price - min_price) / min_price >= deviation and min_index - start_index >= depth:
                    if (high_price - min_price) / min_price >= deviation and index - min_index >= depth:
                        # 시작점 추가
                        key_points.append(start_index)
                        # 시작점을, 최소점으로 변경
                        start_price, start_index = min_price, min_index
                        # 최대점만 현재점으로 변경해준다
                        max_price, max_index = high_price, index
                        min_price, min_index = low_price, index
                        # 방향 뒤집기
                        current_direction = 1
                        stats.append(-1)

            elif current_direction == 1:
                if (max_price - start_price) / start_price >= deviation and max_index - start_index >= depth:
                    if (max_price - low_price) / low_price >= deviation and index - max_index >= depth:
                        # 시작점 추가
                        key_points.append(start_index)
                        # 시작점을, 최대점으로 변경
                        start_price, start_index = max_price, max_index
                        # 최소점만 현재점으로 변경해준다
                        min_price, min_index = low_price, index
                        max_price, max_index = high_price, index
                        # 방향 뒤집기
                        current_direction = -1
                        stats.append(1)

        key_points.append(start_index)
        stats.append(current_direction)
        key_points, stats = np.array(key_points), np.array(stats)

        # key_prices 적용하기 #
        key_prices = []
        for i, stat in enumerate(stats):
            if stat == 1:
                key_prices.append(input_array_low[key_points[i]])
            else:
                key_prices.append(input_array_high[key_points[i]])

        pivot_points, direction_stats = np.array(key_points), np.array(stats)
        pivot_prices = np.array(key_prices)
        # Direction_stat -> 상승방향을 측정 할때 +1 / 하락방향을 측정할때 -1
        # Pivot_stat -> 시작점이 상승방향일때는 저점 / 하락방향일때는 고점
        pivot_stats = -1 * direction_stats

        return pivot_points, pivot_prices, pivot_stats

