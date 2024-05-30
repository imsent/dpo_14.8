from dataclasses import dataclass
from environs import Env

env = Env()
env.read_env(".env")


@dataclass
class Configuration:
    token: str = env('BOT_TOKEN')
    host: str = env('HOST')


conf = Configuration()
