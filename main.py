import os

if __name__ == "__main__":
    runfile('create_dataset.py', wdir=os.getcwd())
    runfile('word2vec.py', wdir=os.getcwd())
    runfile('seq2seq.py', wdir=os.getcwd())
    from telegrambot import bot
    bot.main()
