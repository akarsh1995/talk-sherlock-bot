# Talk Sherlock Telegram Bot.

This bot is made for Sherlock lovers@221-B.

The bot is trained on the subtitles of the popular TV series Sherlock Holmes. It talks to the person in Sherlock Style. I bet you're gonna love it.

The bot is trained using Language modeling technique using tensorflow LSTM seq2seq encoder-decoder framework. Taking sequence of "words" into account.

You've two options to get it up and running:

> before your run the script you have to obtain a telegram bot token from telegram.

Use this documentation to obtain your telegram bot token.  https://core.telegram.org/bots#6-botfather
*After obtaining the tokes set it as environment variable using the shell command below.*
```sh
export TELEGRAM_BOT_TOKEN="{your token}"
````
Now :
1. You can download the pretrained model from https://drive.google.com/drive/folders/1pSN2da3WckuWqgsLVos7WkrrjYhs_FJd?usp=sharing and put it inside `./data` and run `python main.py`
2. You can train your own models by running. 

```sh
python create_dataset.py
python word2vec.py
python seq2seq.py
python main.py
```