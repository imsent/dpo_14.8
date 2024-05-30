import logging
from dataclasses import dataclass
from environs import Env

env = Env()
env.read_env(".env")


@dataclass
class Configuration:
    debug = bool(env('DEBUG',0))
    logging_level = int(env('LOGGING_LEVEL', logging.INFO))
    token: str = env('BOT_TOKEN')
    host: str = env('HOST')


conf = Configuration()
