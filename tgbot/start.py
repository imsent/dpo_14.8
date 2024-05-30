import asyncio
import logging

from aiogram import Bot

from src.dispatcher import get_dispatcher
from configuration import conf


async def start_bot():
    bot = Bot(token=conf.token)
    dp = get_dispatcher()

    await dp.start_polling(
        bot
    )


if __name__ == '__main__':
    logging.basicConfig(level=conf.logging_level)
    asyncio.run(start_bot())
