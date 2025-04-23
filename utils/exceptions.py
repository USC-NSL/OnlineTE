

class Unreachable(Exception):
    def __init__(self):
        super().__init__('This line should never be executed!')
