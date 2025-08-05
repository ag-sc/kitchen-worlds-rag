from typing import List


class CurrentScenario:
    def __init__(self, goal: str, arms: int, objects: List[str], obsv_objects: List[str], is_nl=True):
        self._goal = goal
        self._no_arms = arms
        self._objects = objects
        self._observed_objects = obsv_objects
        self._is_nl = is_nl
        self._actions = None

    def get_goal(self) -> str:
        return self._goal

    def get_no_arms(self) -> int:
        return self._no_arms

    def get_objects(self) -> List[str]:
        return self._objects

    def get_observed_objects(self) -> List[str]:
        return self._observed_objects

    def is_natural_language(self) -> bool:
        return self._is_nl

    def get_actions(self) -> str:
        return self._actions

    def set_actions(self, actions: str):
        self._actions = actions
        self._is_nl = False
