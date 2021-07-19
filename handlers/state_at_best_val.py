from typing import Any, Callable, Dict, Optional

from ignite.engine import Engine


class StateAtBestVal:
    def __init__(self, score_function: Callable, state_function: Callable):
        self.score_function = score_function
        self.state_function = state_function

        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict[str, Any]] = None

    def __call__(self, engine: Engine):
        score = self.score_function()
        if self.best_score is None or self.best_score < score:
            self.best_score = score
            self.best_state = self.state_function()
