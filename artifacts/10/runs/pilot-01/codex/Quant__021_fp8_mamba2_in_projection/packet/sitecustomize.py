import enum as _enum


if not hasattr(_enum, "StrEnum"):
    class StrEnum(str, _enum.Enum):
        pass

    _enum.StrEnum = StrEnum
