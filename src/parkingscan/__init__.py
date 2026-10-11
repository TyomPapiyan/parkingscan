from importlib.metadata import version

# версия берётся из метаданных установленного пакета, источник истины - pyproject.toml
__version__ = version("parkingscan")
