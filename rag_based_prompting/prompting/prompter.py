from typing import List

from RAG4Robots.src.manager import RAGManager


class Prompter:
    def __init__(self, model: str, rag: RAGManager, temp=0.0):
        self._model_name = model
        self._rag_manager = rag
        self._temperature = temp
        self._curr_goal = None
        self._last_system_message = None
        self._last_user_message = None

    def get_model_name(self) -> str:
        return self._model_name

    def get_temperature(self) -> float:
        return self._temperature

    def get_current_goal(self) -> str:
        return self._curr_goal

    def set_current_goal(self, goal: str):
        self._curr_goal = goal

    def get_last_system_message(self) -> str:
        return self._last_system_message

    def get_last_user_message(self) -> str:
        return self._last_user_message

    def get_context_through_rag(self, query: str) -> List[str]:
        cont_rank = self._rag_manager.query_all_dbs(query)
        contexts = []
        for cont in cont_rank:
            contexts.append(cont[0])
        return contexts

    def prompt_model(self, system_msg: str, user_msg: str) -> str:
        pass
