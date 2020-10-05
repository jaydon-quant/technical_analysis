# coding=utf-8

# Last_fixed_Feb_25_2020
# VERSION 1.0.0

import datetime, json
import urllib
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import config
from sqlalchemy import create_engine





class DbAccess:
    def __init__(self):
        return

    # AWS quant 팀
    @staticmethod
    def get_conn_aws_quant():
        engine = create_engine(config.DbConfig.AWS_QUANT_DB_CONNECTION_STRING, pool_recycle=300)
        return engine
    # 시세 DB
    @staticmethod
    def get_idc():
        engine = create_engine(config.DbConfig.STOCKPLUS_CONNECTION_STRING, pool_recycle=300)
        return engine
    # Topic DB
    @staticmethod
    def get_topic():
        engine = create_engine(config.DbConfig.KAKAOSTOCK_TOPIC_CONNECTION_STRING, pool_recycle=300)
        return engine




class Etcs:
    def __init__(self):
        return
    # 회귀분석 시 정규화를 위해 최솟값은 1, 최댓값은 2로 리스트 내의 값들을 변환하는 함수
    @staticmethod
    def scale_y_axis(price_array):
        min_value = np.min(price_array)
        max_value = np.max(price_array)

        if max_value == min_value:
            alpha = 0
            beta = 1
        else:
            alpha = 1 / (max_value - min_value)
            beta = 1 - min_value * alpha

        scaled_price_array = price_array * alpha + beta

        return scaled_price_array


#############################################################
#################  데이터 로딩 Function!!!!    #################
#############################################################

# 캔들 데이터를 조회하여 load_master 함수를 통해 데이터를 분석에 용이한 구조(data_by_date, data_by_category)로 변환하는 클래스
# load_master_dict
# data_by_date에는 리스트의 한 요소마다 날짜별 ohlc 및 날짜 데이터가 포함됨(Ex) [{'datetime': datetime1, 'open': o1, 'high': h1, 'low': l1, 'close': c1}, ...])
# data_by_category는 딕셔너리 형식이며 ohlc별로 리스트가 들어감(Ex) {'datetime': [datetime1, datetime2, ...], 'open': [o1, o2, ...], ...}
#
#
#
# load_master_pd
# data 를 그냥 Pandas format 으로 리턴
#
class Dunamu_API_DataLoader:
    def __init__(self, code, candle_type, total_candle_length):
        self.code = code
        self.candle_type = candle_type
        self.total_candle_length = total_candle_length
        self.splitted_candle_type = self.split_candle_type(self.candle_type)



    @staticmethod
    def get_api_data(url):
        request_headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36',
        }
        raw_response = requests.get(url, headers=request_headers)
        soup = BeautifulSoup(raw_response.text, 'html.parser')
        soup_fix = json.loads(str(soup))
        return soup_fix

    # string 형식으로 들어온 캔들 타입(1d, 1m, 5m 등)을 쪼개서 리턴하는 함수
    def split_candle_type(self, candle_type):
        try:
            splitted_candle_type = [candle_type[:-1], candle_type[-1]]
            return splitted_candle_type
        except Exception as e:
            return str(e) + "\ncandle type 형식 오류가 발생하였습니다. 형식에 맞게 candle type을 넣어주세요."

    # 'yyyy-mm-dd HH:MM:SS.0' string을 datetime 형식으로 바꾸는 함수def string_to_datetime(datetime_string):
    def string_to_datetime(self, datetime_string):
        candle_ymd = datetime_string.split(' ')[0].split('-')
        candle_hms = datetime_string.split(' ')[1].split('.')[0].split(':')

        formatted_datetime = datetime.datetime(int(candle_ymd[0]), int(candle_ymd[1]), int(candle_ymd[2]),
                                               int(candle_hms[0]), int(candle_hms[1]), int(candle_hms[2]))

        return formatted_datetime





    def get_d_candles_rawdata(self, gicode, t1, day_count=500):
        loop_cnt = int(np.ceil(day_count / 500))
        start_datetime = t1
        total_data = []
        for idx in range(loop_cnt):
            t = start_datetime.strftime("%Y-%m-%d 16:00:00")
            query = {'to': t, 'count': day_count, 'shortCode': gicode, 'adjusted': True}
            url = 'http://quotation-api.dunamu.com/v1/candle/gdays?' + urllib.parse.urlencode(query)
            data = self.get_api_data(url)
            total_data += data
            if len(data) == 500:
                start_datetime = datetime.datetime.strptime(data[-1]['candleTime'],
                                                            '%Y-%m-%d %H:%M:%S.0') - datetime.timedelta(days=1)
            else:
                break
        return total_data

    def get_m_candles_rawdata(self, gicode, t1, day_count=500):
        loop_cnt = int(np.ceil(day_count / 500))
        start_datetime = t1
        total_data = []
        for idx in range(loop_cnt):
            t = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
            query = {'to': t, 'count': day_count, 'shortCode': gicode, 'adjusted': True}
            url = 'http://quotation-api.dunamu.com/v1/candle/minutes/%s?' % (self.splitted_candle_type[0]) + \
                  urllib.parse.urlencode(query)
            data = self.get_api_data(url)
            total_data += data
            if len(data) == 500:
                start_datetime = datetime.datetime.strptime(data[-1]['candleTime'], '%Y-%m-%d %H:%M:%S.0')
            else:
                break
        return total_data


    def get_week_candles_rawdata(self, gicode, t1, day_count=500):
        loop_cnt = int(np.ceil(day_count / 500))
        start_datetime = t1
        total_data = []
        for idx in range(loop_cnt):
            t = start_datetime.strftime("%Y-%m-%d 16:00:00")
            query = {'to': t, 'count': day_count, 'shortCode': gicode, 'adjusted': True}
            url = 'http://quotation-api.dunamu.com/v1/candle/weeks?' + urllib.parse.urlencode(query)
            data = self.get_api_data(url)
            total_data += data
            if len(data) == 500:
                start_datetime = datetime.datetime.strptime(data[-1]['candleTime'],
                                                            '%Y-%m-%d %H:%M:%S.0') - datetime.timedelta(days=1)
            else:
                break
        return total_data

    def get_month_candles_rawdata(self, gicode, t1, day_count=500):
        loop_cnt = int(np.ceil(day_count / 500))
        start_datetime = t1
        total_data = []
        for idx in range(loop_cnt):
            t = start_datetime.strftime("%Y-%m-%d 16:00:00")
            query = {'to': t, 'count': day_count, 'shortCode': gicode, 'adjusted': True}
            url = 'http://quotation-api.dunamu.com/v1/candle/months?' + urllib.parse.urlencode(query)
            data = self.get_api_data(url)
            total_data += data
            if len(data) == 500:
                start_datetime = datetime.datetime.strptime(data[-1]['candleTime'],
                                                            '%Y-%m-%d %H:%M:%S.0') - datetime.timedelta(days=1)
            else:
                break
        return total_data


    def load_master_dict(self):
        candle_raw_data = None
        data_by_date = []
        data_by_category = {
            'datetime': np.array([]),
            'open': np.array([]),
            'high': np.array([]),
            'low': np.array([]),
            'close': np.array([])
        }

        if self.splitted_candle_type[1] == 'm':
            last_candle_time = datetime.datetime.now()
            candle_raw_data = self.get_m_candles_rawdata(self.code, last_candle_time,
                                                         day_count=self.total_candle_length)

        elif self.splitted_candle_type[1] == 'd':
            # 현재는 API로 캔들 데이터를 받아와 전일 일봉까지밖에 데이터를 받지 못해 last_candle_time을 전일 일봉으로 설정함.
            # 추후 실시간 일봉 데이터를 받게 되면 last_candle_time을 오늘로 수정 필요
            last_candle_time = datetime.datetime.now() - datetime.timedelta(days=1)
            candle_raw_data = self.get_d_candles_rawdata(self.code, last_candle_time,
                                                         day_count=self.total_candle_length)

        # 주봉
        elif self.splitted_candle_type[1] == 'w':
            last_candle_time = datetime.datetime.now()
            candle_raw_data = self.get_week_candles_rawdata(self.code, last_candle_time,
                                                         day_count=self.total_candle_length)

        # 월봉
        elif self.splitted_candle_type[1] == 'M':
            last_candle_time = datetime.datetime.now()
            candle_raw_data = self.get_month_candles_rawdata(self.code, last_candle_time,
                                                         day_count=self.total_candle_length)
        # 숫자 맞춰주기
        if len(candle_raw_data)!=self.total_candle_length:
            candle_raw_data = candle_raw_data[:self.total_candle_length]

        for item in candle_raw_data:
            data_by_date.append({
                'datetime': self.string_to_datetime(item['candleTime']),
                'open': item['openingPrice'],
                'high': item['highPrice'],
                'low': item['lowPrice'],
                'close': item['tradePrice']
            })

        for item in data_by_date:
            data_by_category['datetime'] = np.append(data_by_category['datetime'], item['datetime'])
            data_by_category['open'] = np.append(data_by_category['open'], item['open'])
            data_by_category['high'] = np.append(data_by_category['high'], item['high'])
            data_by_category['low'] = np.append(data_by_category['low'], item['low'])
            data_by_category['close'] = np.append(data_by_category['close'], item['close'])

        return data_by_date, data_by_category


    def load_master_pd(self):
        candle_raw_data = None
        data_by_date = []
        data_by_category = {
            'datetime': np.array([]),
            'open': np.array([]),
            'high': np.array([]),
            'low': np.array([]),
            'close': np.array([])
        }

        if self.splitted_candle_type[1] == 'm':
            last_candle_time = datetime.datetime.now()
            candle_raw_data = self.get_m_candles_rawdata(self.code, last_candle_time,
                                                         day_count=self.total_candle_length)

        elif self.splitted_candle_type[1] == 'd':
            # 현재는 API로 캔들 데이터를 받아와 전일 일봉까지밖에 데이터를 받지 못해 last_candle_time을 전일 일봉으로 설정함.
            # 추후 실시간 일봉 데이터를 받게 되면 last_candle_time을 오늘로 수정 필요
            last_candle_time = datetime.datetime.now() - datetime.timedelta(days=1)
            candle_raw_data = self.get_d_candles_rawdata(self.code, last_candle_time,
                                                         day_count=self.total_candle_length)

        # 주봉
        elif self.splitted_candle_type[1] == 'w':
            return

        # 월봉
        elif self.splitted_candle_type[1] == 'M':
            return

        # Pandas Format 으로 변환
        pd_format = pd.DataFrame(candle_raw_data)

        return pd_format

    #### 아래는 신용매매 / 공매도 / 투자자 동향 관련 로딩 코드이다

    def credits(self, long_code, t1, day_count=500):

        loop_cnt = int(np.ceil(day_count / 100))
        start_datetime = t1
        total_data = []
        for idx in range(loop_cnt):
            t = start_datetime.strftime("%Y-%m-%d")
            query = {'code': long_code, 'type': type, 'to': t, 'count': day_count}
            url = 'http://quotation-api.dunamu.com/v1/credits?' + urllib.parse.urlencode(query)
            data = self.get_api_data(url)
            total_data += data

            if len(data) == 100:
                start_datetime = datetime.datetime.strptime(data[-1]['date'],
                                                            '%Y-%m-%d %H:%M:%S.0') - datetime.timedelta(days=1)
            else:
                break
        return total_data
##