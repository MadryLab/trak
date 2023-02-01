import pytest
from trak.traker import TRAKer

def test_class_init():
    traker = TRAKer(save_dir='.')