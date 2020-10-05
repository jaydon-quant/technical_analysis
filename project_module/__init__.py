from . import slack_manage
import external_lib.technical_indicators as tech
from external_lib import TechDataLoad as tech_load
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import external_lib
import pickle
import datetime
import pandas.io.sql as psql

"""

check_date : 삼성전자 데이터를 로딩, 금일자 데이터가 들어왔는지 확인


"""
class ProjectHelpers:

    def __init__(self):
        return

    @staticmethod
    def review_history_helper(code, end_date):

        connector_quant = external_lib.data_load.helpers.DbAccess.get_conn_aws_quant()

        sql_text = """
            select trd_dt, open_prc, high_prc, low_prc, cls_prc, trd_vol from quant.ks_stk_jd
            where gicode = '%s' and trd_dt > '%s';
        """ % (code, end_date)

        candle_info = psql.read_sql(sql_text, connector_quant)
        candle_info = candle_info.rename(columns={"open_prc": "open", "high_prc": "high",
                                                  "low_prc": "low", "cls_prc": "close",
                                                  "trd_dt": "datetime"})
        candle_info = tech_load.get_tech_indicator(candle_info)
        return candle_info

    @staticmethod
    def review_history():

        connector_quant = external_lib.data_load.helpers.DbAccess.get_conn_aws_quant()

        # 우선 여태까지의 기록을 Loading 한다

        sql_text = """
            select * from quant.ks_stk_tech_analysis_history
        """
        record = psql.read_sql(sql_text, connector_quant)

        result = {}

        for date in record['post_date'].unique():
            date_set = record[record['post_date'] == date]

            day_list = []

            for i, stock in date_set.iterrows():
                                
                price_hist = ProjectHelpers.review_history_helper(stock['gicode'], date - datetime.timedelta(1))
                rec_price = price_hist[price_hist['datetime'] == date]['close'].values[0]
                last_price = price_hist.iloc[-1]['close']
                price_diff = 100 * (last_price - rec_price) / rec_price

                small_result = {
                    'name': stock['name'],
                    'rec_date': date,
                    'start_price': rec_price,
                    'end_price': last_price,
                    'price_diff': price_diff,
                    'status': stock['status']
                }
                
                day_list.append(small_result)

            result[date] = day_list

        return result




    @staticmethod
    def check_data_up_to_date():

        connector_quant = external_lib.data_load.helpers.DbAccess.get_conn_aws_quant()

        # Step 1 : 오늘이 거래일인가유 ??
        sql_text = """
            select * from quant.ks_calendar
            where TRD_DT_PDAY='%s'
        """ % (datetime.date.today())
        day_check = psql.read_sql(sql_text, connector_quant)

        if len(day_check) == 0:
            # 거래일이 아니에유 ~
            return -1

        # Step 2 : 거래일이긴 한데 가장 최근 데이터가 업데이트가 되 있나유 ~ ??
        # 삼성전자 데이터에 업데이트가 되어있는지 확인한다.
        test_data = tech_load.load_dataset('A005930', N=500, mode='KR_quantDB', date='2020-09-01')
        last_day_update = test_data.iloc[-1]['datetime']

        if last_day_update == datetime.date.today():
            # 완벽해유 ~
            return 1

        # 데이터가 없어유 !
        return -2




    @staticmethod
    def save_to_quantDB_helper(df, update_time, today, status):

        #Pandas 형식으로 만들어주기
        pd_format = pd.DataFrame(data=df).transpose().reset_index(drop=False)
        pd_format = pd_format.rename(columns={'index': 'gicode'})
        pd_format['status'] = status
        pd_format['post_date'] = today
        pd_format['created_at'] = update_time

        # 수정사항 1 : 'SUM' float64 -> float
        sum_arr = []
        for i in pd_format['sum'].values:
            sum_arr.append(float(i))

        pd_format['sum'] = sum_arr

        # 수정사항 2 : 'value' numpy arr -> str
        value_arr = []
        for arr in pd_format['value'].values:
            val_str = ''
            for i in arr:
                val_str += str(i)
            value_arr.append(val_str)

        pd_format['value'] = value_arr

        return pd_format



    @staticmethod
    def save_to_quantDB(over_bought, over_sold):

        # Pandas 형식으로 만들어주기
        update_time = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        today = datetime.date.today()

        # Overbought
        df_over_bought = ProjectHelpers.save_to_quantDB_helper(over_bought, update_time, today, 'over_bought')
        # Oversold
        df_over_sold = ProjectHelpers.save_to_quantDB_helper(over_sold, update_time, today, 'over_sold')
        pd_format = pd.concat([df_over_bought, df_over_sold]).reset_index(drop=True)

        #Quant_db 호출
        quant_db = external_lib.data_load.helper.DbAccess.get_conn_aws_quant()

        # Upload 해주기
        pd_format.to_sql('ks_stk_tech_analysis_history', con=quant_db, if_exists='append', chunksize=1000, index=0)

        return


    @staticmethod
    def pre_process(df):
        # Normalized 정보 추가시키기
        df['macd_norm'] = df['macd-gap'] / df['close']
        df['ma-20-60_norm'] = df['ma-20-60'] / df['close']
        df['ma-5-20_norm'] = df['ma-5-20'] / df['close']

        # column 선택하기
        modified = df[['datetime', 'close', 'rsi', 'stochastic',
                       'dmi_signal', 'bollinger_position', 'macd_norm', 'ma-20-60_norm',
                       'ma-5-20_norm']]
        return modified

    @staticmethod
    def add_ending_info(df):

        # 정보 선택하기
        length = df['orig_index'][1:].values - df['orig_index'][:-1].values
        bef_pr = df['close'][:-1].values
        aft_pr = df['close'][1:].values
        yld = (aft_pr - bef_pr) / (bef_pr)
        modified = df.iloc[:-1]
        modified['after_length'] = length
        modified['after_yld'] = yld
        return modified

    @staticmethod
    def peak_point_info(df, len_input=5, dev_input=0.25):

        #zigzag_info = tech.zigzag.ZigZag(df['high'], df['low'],
        #                                 deviation=dev_input, depth=len_input)
        zigzag_info = tech.zigzag.ZigZag(df['close'], deviation=dev_input, depth=len_input)

        pts = zigzag_info.pivot_points
        # 중요 지점 추출
        key_values = df.iloc[pts]
        index_row = key_values.index.values
        key_values = key_values.reset_index(drop=True)

        # min_max marking
        position_row = []

        if len(key_values)>1:

            if key_values['close'].values[0] > key_values['close'].values[1]:
                start_max = True
            else:
                start_max = False

            if start_max:
                up_counter = 0
            else:
                up_counter = 1

            for i, pt in enumerate(pts):
                if i % 2 == up_counter:
                    position_row.append(1)
                else:
                    position_row.append(-1)

            modified = ProjectHelpers.pre_process(key_values)
            modified['orig_index'] = index_row
            modified = ProjectHelpers.add_ending_info(modified)
            modified['position'] = position_row[:-1]

        else:
            modified = ProjectHelpers.peak_point_info(df, len_input, dev_input-0.01)

        return modified

    @staticmethod
    def get_score_helper(train_info, dev=0.1):
        df = ProjectHelpers.peak_point_info(train_info, dev_input=dev)
        up_pts = df[df['position'] == 1]['orig_index'].values
        down_pts = df[df['position'] == -1]['orig_index'].values
        score = np.zeros(len(train_info))
        score[up_pts] = 1
        score[down_pts] = -1
        return score

    @staticmethod
    def get_score(train_info):
        # 4개의 다른 level 에 대해서 확인해보기
        ## LV1
        score1 = ProjectHelpers.get_score_helper(train_info, dev=0.1)
        ## LV2
        score2 = ProjectHelpers.get_score_helper(train_info, dev=0.15)
        ## LV3
        score3 = ProjectHelpers.get_score_helper(train_info, dev=0.2)
        ## LV4
        score4 = ProjectHelpers.get_score_helper(train_info, dev=0.25)

        score_total = (score1 + score2 + score3 + score4) / 4

        return score_total


    @staticmethod
    def train_process(code, test_length=300):
        # 제일먼저 로딩!!
        stock_info = tech_load.load_dataset(code, 4000, mode='KR_quantDB')

        test_bars = test_length

        ## 맨 처음에 시작할때 Test Set 을 분리시킨다 (맨 마지막 100 거래일)
        test_set = stock_info.iloc[-test_bars:]

        # 나머지는 (train + validatin set)
        train_valid_set_raw = stock_info.iloc[:-test_bars]
        score_row = ProjectHelpers.get_score(train_valid_set_raw)

        # pre_process 된 상태로 만들어주기
        train_valid_set_processed = ProjectHelpers.pre_process(train_valid_set_raw)
        train_valid_set_processed['score'] = score_row

        # Score 가 적힌 index 추출하기
        index_list = np.argwhere(score_row != 0)
        index_list = index_list.reshape(len(index_list))

        train_valid_set = train_valid_set_processed.iloc[index_list].reset_index(drop=True)

        # train set 그리고 validation set 으로 분리시키기 (15 %)

        test_ratio = 0.15
        divide_num = int((1 - test_ratio) * len(train_valid_set))

        train_sample = train_valid_set.iloc[:divide_num]
        valid_sample = train_valid_set.iloc[divide_num:]

        features = ['bollinger_position', 'ma-20-60_norm', 'ma-5-20_norm', 'rsi', 'stochastic', 'dmi_signal',
                    'macd_norm']

        # train set / validation set 정리하기
        x_train = train_sample[features]
        x_valid = valid_sample[features]

        y_train = train_sample['score'].values
        y_valid = valid_sample['score'].values

        SEED = 0
        params = {'iterations': 5000,
                  'learning_rate': 0.01,
                  'depth': 3,
                  'verbose': 200,
                  'od_type': "Iter",  # overfit detector
                  'od_wait': 500,  # most recent best iteration to wait before stopping
                  'random_seed': SEED
                  }

        # 모델 training
        cat_model = CatBoostRegressor(**params)
        cat_model.fit(x_train, y_train,
                      eval_set=(x_valid, y_valid),
                      use_best_model=True,
                      # True if we don't want to save trees created after iteration with the best validation score
                      plot=False
                      )

        # 테스팅 해보기
        test_sample = ProjectHelpers.pre_process(test_set)[features]
        dval_predictions = cat_model.predict(test_sample)

        ## Output 형식으로
        date_row = stock_info['datetime'][-test_bars:].values
        predict_row = dval_predictions
        predict_df = pd.DataFrame(data={'date': date_row, code: predict_row})


        result_format = {
            'predict_df':predict_df,
            'cat_model':cat_model,
            'test_x':test_sample,
            'test_y':dval_predictions
        }
        return result_format




class ProjectRun:

    def __init__(self):
        return

    # 코스피 200 종목 추출
    @staticmethod
    def load_kospi200():
        stock_info = external_lib.DBLoad.call_db()
        # '스팩' 들어간 종목들 전부 제외하기
        stock_info = stock_info[~stock_info['itemabbrnm'].str.contains('스팩')].reset_index(drop=True)
        # 특정 column 추출하기
        stock_info = stock_info[['code', 'gicode', 'itemabbrnm', 'kospi_200_sector']]
        kospi_200 = stock_info[stock_info['kospi_200_sector'] != 'NONE'].reset_index(drop=True)

        return kospi_200

    @staticmethod
    def get_stock_name(kospi_200, code):
        stock_name = kospi_200[kospi_200['gicode'].isin([code])]['itemabbrnm'].values[0]
        return stock_name

    @staticmethod
    def run(test_length=300, count=5, test_mode=False):

        kospi_200 = ProjectRun.load_kospi200()

        # 맨 처음 삼성전자로 확인하기 (맨 마지막 날짜)
        code = 'A005930'
        stock_info = tech_load.load_dataset(code, 500, mode='KR_quantDB')
        last_date = stock_info['datetime'].iloc[-1]

        result = [0]

        if test_mode:
            kospi_200_fixed = kospi_200.iloc[:8]
        else:
            kospi_200_fixed = kospi_200

        for i, row in kospi_200_fixed.iterrows():
            code = row['gicode']
            # 버그 처리해주기
            df_format = ProjectHelpers.train_process(code, test_length)['predict_df']
            # 맨 마지막 날짜 동일한지 확인하기
            if last_date != df_format['date'].iloc[-1]:
                continue
            # DF format 추가해주기
            if len(result) == 1:
                result = df_format
            else:
                result = pd.merge(result, df_format, on='date')

        # 결과 추출하기
        l = test_length-1
        result_format = result.transpose().drop(['date']).sort_values(by=l, ascending=False)

        top_vals = result_format[l][:count].values
        top_codes = result_format[l][:count].index.values
        top_names = [ProjectRun.get_stock_name(kospi_200, code) for code in top_codes]

        top_info = pd.DataFrame(data = {'name':top_names, 'code':top_codes, 'value':top_vals})

        bot_vals = result_format[l][-count:].values
        bot_codes = result_format[l][-count:].index.values
        bot_names = [ProjectRun.get_stock_name(kospi_200, code) for code in bot_codes]

        bot_info = pd.DataFrame(data = {'name':bot_names, 'code':bot_codes, 'value':bot_vals})

        return [top_info, bot_info]


    # df_info -> list of codes

    @staticmethod
    def post_analysis_helper(df_info, stock_list_info, cat_model_load=None):

        df_info_fixed = {}


        for i, c in enumerate(df_info):

            stock_name = stock_list_info[stock_list_info['gicode'] == c]['itemabbrnm'].values[0]

            if cat_model_load== None:
                result_check = ProjectHelpers.train_process(c)
                cat_model = result_check['cat_model']
                test_x = result_check['test_x']
                test_y = result_check['test_y']
            else:
                # cat_model 로딩한것 사용
                cat_model = cat_model_load[c]
                # 주식 데이터 바로 로딩
                stock_info = tech_load.load_dataset(c, 4000, date='2019-02-01', mode='KR_quantDB')
                test_set = stock_info.iloc[-100:]
                features = ['bollinger_position', 'ma-20-60_norm', 'ma-5-20_norm', 'rsi', 'stochastic', 'dmi_signal',
                            'macd_norm']
                test_x = ProjectHelpers.pre_process(test_set)[features]
                test_y = cat_model.predict(test_x)

            # 결과에 대해서 Shap Value 계산
            shap_values = cat_model.get_feature_importance(Pool(test_x, label=test_y), type="ShapValues")
            # features + bias(expected_value)
            feature_values = shap_values[-1, :]

            df_info_fixed[c] = {
                'name': stock_name,
                'sum': np.sum(feature_values),
                'value': feature_values
            }

        return df_info_fixed


    @staticmethod
    def post_analysis(top_info, bot_info, cat_model_load=False):

        if cat_model_load:
            with open('/Users/jayden.jeon/dev/technical_analysis/cat_model.pkl', 'rb') as f:
                cat_model_dict = pickle.load(f)
        else:
            cat_model_dict = None

        stock_list_info = external_lib.DBLoad.call_db()

        top_info_fixed = ProjectRun.post_analysis_helper(top_info, stock_list_info, cat_model_dict)
        bot_info_fixed = ProjectRun.post_analysis_helper(bot_info, stock_list_info, cat_model_dict)

        column_names = ['bollinger_position', 'ma-20-60_norm', 'ma-5-20_norm', 'rsi',
                        'stochastic', 'dmi_signal', 'macd_norm']

        return top_info_fixed, bot_info_fixed, column_names

