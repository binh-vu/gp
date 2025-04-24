import os


class AppConfig:
    instance = None

    @staticmethod
    def get_instance():
        if AppConfig.instance is None:
            AppConfig.instance = AppConfig()
        return AppConfig.instance

    @property
    def is_canrank_verbose(self) -> bool:
        return os.environ.get("CANRANK_VERBOSE") == "1"

    @property
    def is_cache_enable(self) -> bool:
        return os.environ.get("ENABLE_CACHE") == "1"

    @property
    def is_ray_disable(self) -> bool:
        return os.environ.get("DISABLE_RAY") == "1"
