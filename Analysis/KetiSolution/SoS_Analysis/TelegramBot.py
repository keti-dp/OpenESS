import telegram
from telegram.ext import Updater
from pprint import pprint


class TelegramBot:
    def __init__(self):
        token = "" #token info
        name = ""  #bot name
        self.core = telegram.Bot(token)
        self.updater = Updater(token)
        self.id = "" # id
        self.name = name
        self.sendMessage("챗봇을 생성 및 시작합니다.")
    
    def sendMessage(self, text):
        self.core.sendMessage(chat_id=self.id, text=text)

    def stop(self):
        self.updater.start_polling()
        self.updater.dispatcher.stop()
        self.updater.job_queue.stop()
        self.updater.stop()

    def showText(self):
        updates = self.core.getUpdates()
        for m in updates:
            pprint(m.message)

    def addHandler(self, cmd, func):
        self.updater.dispatcher.add_handler(commandHandler(cmd, cunc,))

    def start(self):
        self.updater.start_polling()
        self.updater.idle()


