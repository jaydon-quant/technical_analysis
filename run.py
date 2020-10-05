# -*- coding: utf-8 -*-

# common libs
import sys
import numpy as np
import project_module as project
from project_module.slack_manage import SlackManager
import time
import external_lib
import pickle
from project_module.batch_module import WeeklyUpdate
from project_module.daily_evaluation import DailyEval
import datetime

# action list (수행 리스트)
_DAILY_STOCK_RECOMMEND = "daily_report"
_TESTMODE = 'test'
_WEEKLY_BATCH = 'weekly_update'
_PRERUN_CHECK = 'pre_check'
_HOT_FIX = 'hot_fix'

if __name__ == "__main__":

    action = sys.argv[1]

    if action == _HOT_FIX:
        send_msg = '시장의 regime 변경에 따른 모델 재학습 필요 (재학습까지 4시간 소요)'
        SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_test_jayden')

    if action == _WEEKLY_BATCH:
        WeeklyUpdate.run_batch()
        send_msg = '<@jayden.jeon> Weekly Model Update 완료'
        SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_test_jayden')

    if action == _PRERUN_CHECK:
        pre_check = project.ProjectHelpers.check_data_up_to_date()
        if pre_check == -2:
            send_msg = '<@jayden.jeon> 가장 마지막 영업일 데이터 업로드 아직 안되어있습니다!'
            SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_test_jayden')

    # 당일 주식별 점수 추출
    if action == _DAILY_STOCK_RECOMMEND:

        # 데이터가 다 업데이트 되어있는지 확인한다.
        pre_check = project.ProjectHelpers.check_data_up_to_date()

        if pre_check == -1:
            send_msg = '<@jayden.jeon> 종목추천 메세지 메세지 PUSH 보류 : 오늘(%s)은 영업일이 아닙니다' % datetime.date.today()
            SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_push')

        elif pre_check == -2:
            send_msg = '<@jayden.jeon> 종목추천 메세지 메세지 PUSH 보류 : 가장 최근 데이터 필요'
            SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_push')

        elif pre_check == 1:
            daily_result = DailyEval.daily_evaluation()
            over_bought, over_sold = daily_result['top'], daily_result['bot']

            # 종목 추천의 결과 추출하기
            [over_bought_fix, over_sold_fix, column_names] = project.ProjectRun.post_analysis(over_bought, over_sold,
                                                                                              cat_model_load=True)

            over_sold_num = len(over_sold_fix)
            over_bought_num = len(over_bought_fix)

            if (over_sold_num == 0) or (over_bought_num == 0):
                send_msg = '<@jayden.jeon> quant 종목추천 애러 발생 : 고점 종목 개수 : %s, 저점 종목 개수 : %s / 백업모델로 재실행 필요!'\
                           % (over_sold_num, over_bought_num)
                SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_push')

            else:
                # 인트로 정보 보내주기
                service_channel = '#quant_종목추천'
                SlackManager.send_intro_msg(channel=service_channel)
                slk_up = SlackManager(over_bought_fix, 'overbought', channel=service_channel)
                slk_down = SlackManager(over_sold_fix, 'oversold', channel=service_channel)
                SlackManager.send_msg_directly('\n', attachment=None, channel=service_channel)
                slk_up.run()
                slk_down.run()

                send_msg = '<@jayden.jeon> quant_종목추천 메세지 PUSH 완료'
                SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_push')

                # Quant_DB에 적재하기
                project.ProjectHelpers.save_to_quantDB(over_bought_fix, over_sold_fix)




    if action == _TESTMODE:
        print('test')
        test_channel = '#quant_test_jayden'
        # 데이터가 다 업데이트 되어있는지 확인한다.
        if not project.ProjectHelpers.check_data_up_to_date():
            send_msg = '<@jayden.jeon> Quant_db에 오늘자(%s) 데이터가 업로드 안되어있음, 장 휴일인지 확인 바람'%datetime.date.today()
            SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel=test_channel)

        # 인트로 정보 보내주기
        SlackManager.send_intro_msg(channel=test_channel)

        with open('test_variable.pkl', 'rb') as f:
            over_bought_fix, over_sold_fix = pickle.load(f)


        slk_up = SlackManager(over_bought_fix, 'overbought', channel=test_channel)
        slk_down = SlackManager(over_sold_fix, 'oversold', channel=test_channel)
        SlackManager.send_msg_directly('\n', attachment=None, channel=test_channel)
        slk_up.run()
        slk_down.run()
        print('전송완료')




