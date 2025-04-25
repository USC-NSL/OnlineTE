

class Unreachable(Exception):
    """Used when a line gets executed when it really shouldn't be reachable"""
    def __init__(self):
        super().__init__('This line should never be executed!')


class SolutionInterrupted(Exception):
    """Used when solution is interrupted. Must stop and output whatever we have"""
    def __init__(self, *args):
        super().__init__('Solution was interrupted')
