"""

Daily Evaluation : Daily(real-time) update
로컬에 저장되어있는 모델을 로딩, 현재 데이터를 적용

Process 는 다음과 같다.

1. 모델 로딩
2. 주식의 데이터 계속 로딩 (for loop)
3. 데이터를 모델에 적용, 오늘의 고점-저점 지수 계산


계산시간 : 약 15분정도 소요

"""

from external_lib import TechDataLoad as tech_load
import numpy as np
import pandas as pd
from project_module import ProjectHelpers
import pickle
import project_module as project
from project_module.slack_manage import SlackManager



class DailyEval:
    def __init__(self):
        return

    # 데이터 적용하기
    @staticmethod
    def apply_cat_model(code, cat_model_dict):
        # 현재 데이터를 불러와서 비교한다 (Quant_db 호출)
        current_data = tech_load.load_dataset(code, 4000, mode='KR_quantDB', date='2019-02-01')
        # Apply Normalization
        features = ['bollinger_position', 'ma-20-60_norm', 'ma-5-20_norm', 'rsi', 'stochastic', 'dmi_signal',
                    'macd_norm']
        cat_model = cat_model_dict[code]
        data_processed = ProjectHelpers.pre_process(current_data)[features].iloc[-1:]
        prediction_val = cat_model.predict(data_processed)
        return prediction_val[0]


    @staticmethod
    def daily_evaluation():

        # Dictionary Loading
        with open('cat_model.pkl', 'rb') as f:
            cat_model_dict = pickle.load(f)

        code_arr = []
        val_arr = []
        for code in cat_model_dict.keys():
            # 우선은 임시방편으로 해당코드를 다음과 같이 cut_off 를 설정함
            # 종목번호 A220000 이상인 것들은 전부 제외시키는 방법으로 적용
            #
            if int(code[1:]) >= 220000:

                continue
            try:
                value = DailyEval.apply_cat_model(code, cat_model_dict)
                code_arr.append(code)
                val_arr.append(value)
            except:
                continue

        # pandas 형식으로 변경
        pandas_format = pd.DataFrame(data={'code': code_arr, 'value': val_arr})
        pandas_format = pandas_format.sort_values(by='value', ascending=False).reset_index(drop=True)

        # dicitonary 로 바로 제공하면 문제가 발생해서 다시 pd_format 으로 만들어야

        result_format = {
            'top': pandas_format.iloc[:5]['code'].values,
            'bot': pandas_format.iloc[-5:]['code'].values
        }

        return result_format



    """
    실행 후, quant_push / private channel 에 종목을 미리 보여주기
    
    
    """
    @staticmethod
    def pre_run():

        daily_result = DailyEval.daily_evaluation()
        over_bought, over_sold = daily_result['top'], daily_result['bot']

        # 종목 추천의 결과 추출하기
        [over_bought_fix, over_sold_fix, _] = project.ProjectRun.post_analysis(over_bought, over_sold,
                                                                               cat_model_load=True)

        # 저장한 후에 pkl 로 저장해놓기
        daily_result_fix = {
            'over_bought': over_bought_fix,
            'over_sold': over_sold_fix
        }

        with open('daily_report.pkl', 'wb') as f:
            pickle.dump(daily_result_fix, f)

        over_sold_num = len(over_sold_fix)
        over_bought_num = len(over_bought_fix)

        if (over_sold_num == 0) or (over_bought_num == 0):
            send_msg = '<@jayden.jeon> quant 종목추천 애러 발생 : 고점 종목 개수 : %s, 저점 종목 개수 : %s / 백업모델로 재실행 필요!' \
                       % (over_sold_num, over_bought_num)
            SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_push')

        else:
            # 인트로 정보 보내주기
            service_channel = '#quant_test_jayden'
            SlackManager.send_intro_msg(channel=service_channel)
            slk_up = SlackManager(over_bought_fix, 'overbought', channel=service_channel)
            slk_down = SlackManager(over_sold_fix, 'oversold', channel=service_channel)
            SlackManager.send_msg_directly('\n', attachment=None, channel=service_channel)
            slk_up.run()
            slk_down.run()

            send_msg = '<@jayden.jeon> 오늘의 종목추천 정보 업로드 완료, 확인 바람'
            SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_push')

            # Quant_DB에 적재하기
            project.ProjectHelpers.save_to_quantDB(over_bought_fix, over_sold_fix)

        return


    """
    Service channel 에 확인 이후 보여주기
    """
    @staticmethod
    def run():

        # Dictionary Loading
        with open('daily_report.pkl', 'rb') as f:
            daily_result_fix = pickle.load(f)

        over_bought_fix = daily_result_fix['over_bought']
        over_sold_fix = daily_result_fix['over_sold']

        service_channel = '#quant_종목추천'
        SlackManager.send_intro_msg(channel=service_channel)
        slk_up = SlackManager(over_bought_fix, 'overbought', channel=service_channel)
        slk_down = SlackManager(over_sold_fix, 'oversold', channel=service_channel)
        SlackManager.send_msg_directly('\n', attachment=None, channel=service_channel)
        slk_up.run()
        slk_down.run()

        send_msg = '<@jayden.jeon> quant_종목추천 메세지 PUSH 완료'
        SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_push')

        return


