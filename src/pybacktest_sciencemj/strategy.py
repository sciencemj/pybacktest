from pydantic import BaseModel
from pybacktest_sciencemj.models import Action, Stock, Portfolio
import pandas as pd
from typing import Callable, List

class Strategy:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func
    
    def apply(self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp) -> List[Action]:
        return self.func(portfolio, stocks, date)
    
    def get_name(self) -> str:
        return self.name