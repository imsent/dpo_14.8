from dataclasses import dataclass
from environs import Env

env = Env()
env.read_env(".env")


@dataclass
class Configuration:
    token: str = env('API_TOKEN')
    id: str = env('ID')
    host: str = env('HOST')


conf = Configuration()
