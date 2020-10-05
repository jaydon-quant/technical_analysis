# -*- coding: utf-8 -*-

import sys, datetime, config
from slacker import Slacker
import json
#reload(sys)
#sys.setdefaultencoding('utf-8')
import datetime

class SlackManager:

    def __init__(self, data, status, channel='#quant_종목추천'):
        self._token = 'xoxb-3264352708-1373542005569-hbx4PUYGECve2P0ACKck09XE'
        self.data = data
        self.status = status
        self.channel = channel

    @staticmethod
    def send_msg_directly(msg, attachment, channel='#quant_종목추천'):

        #channel = '#quant_test_jayden'

        token = 'xoxb-3264352708-1373542005569-hbx4PUYGECve2P0ACKck09XE'
        slack = Slacker(token)
        slack.chat.post_message(channel=channel, text=msg, attachments=attachment)
        return

    "맨 처음 소개하는 문장 써주기"
    @staticmethod
    def send_intro_msg(channel='#quant_종목추천'):
        head_text = '고점 - 저점 탐색기'
        explain_text1 = '*%s* 오늘의 *고점* / *저점* 주식 정보는 다음과 같습니다.\n'\
                        '수치 설명 : -1에 가까울수록 저점, 1에 가까울수록 고점 입니다.\n' \
                        '약 2000개의 종목에 대해서 실행한 결과입니다.\n' % str(datetime.date.today())
        explain_text2 = "<https://www.notion.so/dunamu/b4bf5f66de5e48a091c06c2f403d3e60| " \
                        "자세한 계산 방식은 *Notion 링크* 를 참조 해주세요>"

        attachment_intro = json.dumps([{
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": head_text,
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": explain_text1,
                    }
                },

                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": explain_text2,
                    }
                }
            ]}])
        SlackManager.send_msg_directly('', attachment_intro, channel)
        return


    def send_msg(self, msg):
        channel = '#quant_종목추천'
        slack = Slacker(self._token)
        slack.chat.post_message(channel=channel, text=msg)
        return


    ## input 값 Dictionary
    """
        df_info_fixed[c] = {
        'name': row['name'],
        'sum': row['value'],
        'value': feature_values
    }
    """
    def single_stock_msg(self, name, code, sum, feature_row):

        column_names = ['bollinger_position', 'ma-20-60_norm', 'ma-5-20_norm', 'rsi', 'stochastic', 'dmi_signal',
                    'macd_norm', 'BIAS']

        attachment = json.dumps([{

            "blocks": [

            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "<https://finance.daum.net/quotes/%s#home|*%s (%s) * : %.3f>" %
                                (code, code, name, sum),

                    }]
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "plain_text",
                        "text": '%s : %.3f'%(column_names[0], feature_row[0])
                    },
                    {
                        "type": "plain_text",
                        "text": '%s : %.3f'%(column_names[1], feature_row[1]),
                    },
                    {
                        "type": "plain_text",
                        "text":  '%s : %.3f'%(column_names[2], feature_row[2]),
                    },
                    {
                        "type": "plain_text",
                        "text":  '%s : %.3f'%(column_names[3], feature_row[3]),
                    },
                    {
                        "type": "plain_text",
                        "text":  '%s : %.3f'%(column_names[4], feature_row[4]),
                    },
                    {
                        "type": "plain_text",
                        "text":  '%s : %.3f'%(column_names[5], feature_row[5]),
                    },
                    {
                        "type": "plain_text",
                        "text": '%s : %.3f' % (column_names[6], feature_row[6]),
                    },
                    {
                        "type": "plain_text",
                        "text": '%s : %.3f' % (column_names[7], feature_row[7]),
                    }
                ]
            },
            {
                "type": "image",
                "title": {
                    "type": "plain_text",
                    "text": "%s 최근 100일 종가" % name,
                },
                "image_url": "https://ssl.pstatic.net/imgfinance/chart/item/candle/day/%s.png" % (code[1:]),
                "alt_text": "%s" % name
            },
            {
                "type": "divider"
            }
            ]
        }])

        SlackManager.send_msg_directly('', attachment, self.channel)
        return

    def run(self):

        if self.status == 'overbought':
            head_text = '고점으로 추측되는 종목들입니다.'

        elif self.status == 'oversold':
            head_text = '저점으로 추측되는 종목들입니다.'

        attachment_head = json.dumps([{
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": head_text,
                    }
                }
            ]}])
        SlackManager.send_msg_directly('', attachment_head, self.channel)

        for code in self.data.keys():
            info = self.data[code]
            stock_name = info['name']
            score_sum = info['sum']
            feature_row = info['value']
            self.single_stock_msg(stock_name, code, score_sum, feature_row)

        return

    @staticmethod
    def test():
        token = 'xoxb-3264352708-1373542005569-hbx4PUYGECve2P0ACKck09XE'
        channel = '#quant_종목추천'
        msg = 'tech_test 1 2 3'
        slack = Slacker(token)
        slack.chat.post_message(channel=channel, text=msg)