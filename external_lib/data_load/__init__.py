# coding=utf-8

from . import helpers as helper
import pandas.io.sql as psql
import pandas as pd
import numpy as np


"""
리서치할때 사용하는 Loading Module
Service 에서는 사용 불필요 !!
"""
class DataLoad:
    def __init__(self, code, candle_type, total_candle_length, candle_gap=0):

        self.code = code
        self.candle_type = candle_type
        self.total_candle_length = total_candle_length
        self.dataloader = helper.Dunamu_API_DataLoader(code=self.code, candle_type=self.candle_type,
                                                       total_candle_length = self.total_candle_length + candle_gap)

        self.data_by_date, self.data_by_category = self.dataloader.load_master_dict()

        self.original_yld = 0
        # 분석 대상 캔들 패턴을 소수의 회귀선으로 표현한 정보
        self.model_control = None
        # 미래 예측 캔들 구간 길
        self.time_measure = 0

        # 데이터를 'N일 전으로 설정'
        self.candle_gap = candle_gap

        # candle_gap 맞춰주기
        self.data_by_date = self.data_by_date[self.candle_gap:]

        for arr in self.data_by_category:
            self.data_by_category[arr] = self.data_by_category[arr][self.candle_gap:]

        self.data_load()

    ###########################################################
    # Step 1 : 데이터 불러오기
    ###########################################################
    def data_load(self):

        self.data = ((self.data_by_category['close'] + self.data_by_category['open']) / 2)[::-1]

        # 불러낸 데이터에 대해서 scaling 해주기 #
        self.data_scaled = helper.Etcs.scale_y_axis(self.data)

        # 중요 !! -> 데이터의 가장 마지막 날
        self.end_date = self.data_by_category['datetime'][0]

        return


class DBLoad:
    def __init__(self):
        return

    @staticmethod
    def call_db():
        connector_quant = helper.DbAccess.get_conn_aws_quant()
        sql_text = """
            select * from quant.ks_stk_master
            where security_group = 'STOCK' and division = 'COMMON';
        """
        stocks = psql.read_sql(sql_text, connector_quant)
        return stocks

    @staticmethod
    def market_cap_db():
        connector_quant = helper.DbAccess.get_conn_aws_quant()
        sql_text = """
            select gicode, korean_name, market_capitalization from quant.ks_stk_quant_base_data
            where trd_dt = '2020-08-31'
            order by market_capitalization desc;
        """
        stocks = psql.read_sql(sql_text, connector_quant)
        return stocks

    @staticmethod
    def us_market_db(code):
        connector_sec = helper.DbAccess.get_idc()
        sql_text = """
            select code, date, opening_price, high_price, low_price, trade_price from day_candle
            where code = '%s';       
        """%code
        stocks_info = psql.read_sql(sql_text, connector_sec)
        return stocks_info


    @staticmethod
    def kr_quant_db(code, date=None):
        connector_quant = helper.DbAccess.get_conn_aws_quant()
        if date==None:
            sql_text = """
                select trd_dt, open_prc, high_prc, low_prc, cls_prc, trd_vol from quant.ks_stk_jd
                where gicode = '%s';
            """%code
        else:
            sql_text = """
                select trd_dt, open_prc, high_prc, low_prc, cls_prc, trd_vol from quant.ks_stk_jd
                where gicode = '%s' and trd_dt >= '%s';
            """%(code, date)
        stocks_info = psql.read_sql(sql_text, connector_quant)
        return stocks_info


