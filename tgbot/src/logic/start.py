from aiogram import Router, types
from aiogram.filters import CommandStart

start_router = Router(name='start')


@start_router.message(CommandStart())
async def start_handler(message: types.Message):
    return await message.answer('Добро пожаловать! Я помощник, готовый помочь вам ответить на ваши вопросы.')
