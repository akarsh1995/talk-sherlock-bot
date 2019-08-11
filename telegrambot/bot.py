from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters
from telegram.ext import CommandHandler
import logging

from model_interface.predict import pred
import os

updater = Updater(token=os.environ.get('TELEGRAM_BOT_TOKEN'), use_context=True)
dispatcher = updater.dispatcher
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def start(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="This bot will let you grow")


start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)


def prediction(update, context):
    reply = ' '.join(pred(update.message.text))
    context.bot.send_message(chat_id=update.message.chat_id, text=reply)


echo_handler = MessageHandler(Filters.text, prediction)
dispatcher.add_handler(echo_handler)


def main():
    updater.start_polling()
