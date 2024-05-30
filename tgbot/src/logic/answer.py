from aiogram import Router, types
import aiohttp

from tgbot.configuration import conf
from aiogram import F

answer_router = Router(name='answer')


async def url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            result = await resp.json()
            return result


@answer_router.message(F.text)
async def answer_handler(message: types.Message):
    answer = await url(f"http://{conf.host}:8080/answer/{message.text}")
    await message.answer(answer['answer'])
