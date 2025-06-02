from abc import ABC

class OrderBookData(ABC):
    
    def __init__(self):
        pass
    
    def get_books_from_range(start_time : float, end_time : float):
        raise NotImplementedError
    
class Strategy(ABC):
    # todo
    pass