from whatsapp_chatbot_python import GreenAPIBot, Notification
from configuration import conf
import requests

bot = GreenAPIBot(
    conf.id, conf.token
)


@bot.router.message()
def message_handler(notification: Notification) -> None:
    answer = requests.get(f'http://{conf.host}:8080/answer/{notification.message_text}')
    notification.answer(answer.json()['answer'])


bot.run_forever()