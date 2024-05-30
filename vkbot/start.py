from vkbottle.bot import Bot, Message
import aiohttp
from configuration import conf

bot = Bot(token=conf.token)


async def url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            result = await resp.json()
            return result


@bot.on.message()
async def handler(message: Message):
    answer = await url(f"http://{conf.host}:8080/answer/{message.text}")
    return answer['answer']


bot.run_forever()
