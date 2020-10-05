"""
포트폴리오 생성 코드


"""
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

if __name__ == "__main__":

    action = sys.argv[1]

    if action == _WEEKLY_BATCH:
        WeeklyUpdate.run_batch()
        send_msg = '<@jayden.jeon> FINISH!'
        SlackManager.send_msg_directly(msg=send_msg, attachment=None, channel='#quant_test_jayden')


    # 당일 주식별 점수 추출
    if action == _DAILY_STOCK_RECOMMEND:
        msg  = 'hi'























