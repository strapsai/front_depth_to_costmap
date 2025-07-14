from .sequence_inn import SequenceINN
from .all_in_one_block import AllInOneBlock

# 사용 편의를 위해 FrEIA 처럼 alias 제공
framework = __import__(__name__)
modules = __import__(__name__)

__all__ = ["SequenceINN", "AllInOneBlock"]