"""

run_batch : weekly update
1주일에 한번씩 (주말에) 모델을 업데이트한다.

우선 적용이 되는 종목들은
1. 거래정지에 적용이 안되는 종목 (is_trading_suspeded)
2. 행정적인 이슈가 있는 종목 (is_administrative_issue)
3. ~ 스팩 이라는 이름이 들어가는 종목. (예 : 유안타제3호스팩)
를 제외한 전 종목들이다

계산시간 : 약 2시간 내외 소요

"""


import external_lib
from project_module import ProjectHelpers
import pickle


class WeeklyUpdate:
    def __init__(self):
        return

    @staticmethod
    def run_batch():
        stock_info = external_lib.DBLoad.call_db()
        # ~ 스팩 들어가는 종목 제외
        stock_info = stock_info[~stock_info['itemabbrnm'].str.contains('스팩')].reset_index(drop=True)
        # 관리종목은 제거한다 & 거래정지 종목 또한 포함
        stock_info = stock_info[
            ~((stock_info['is_administrative_issue'].isin([1])) | stock_info['is_trading_suspended'] == 1)].reset_index(
            drop=True)

        # UPDATING PROCEDURE
        cat_model_dict = {}
        for i, row in stock_info.iterrows():
            try:
                test_result = ProjectHelpers.train_process(row['gicode'])
            except:
                continue
            cat_model = test_result['cat_model']
            cat_model_dict[row['gicode']] = cat_model

        with open('cat_model.pkl', 'wb') as f:
            pickle.dump(cat_model_dict, f)

        return