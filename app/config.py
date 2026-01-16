from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Security
    api_key: str = "dev-key-change-me"

    # Execution limits
    execution_timeout: int = 30  # seconds
    max_memory_mb: int = 512

    # Forbidden patterns for code execution
    forbidden_imports: list[str] = [
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "pickle",
        "shelve",
        "marshal",
        "importlib",
        "__import__",
    ]

    forbidden_builtins: list[str] = [
        "open",
        "eval",
        "exec",
        "compile",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    ]

    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()
