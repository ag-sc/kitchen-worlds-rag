from RAG4Robots.src.manager import RAGManager
from rag_based_prompting.prompting.prompts import *
from rag_based_prompting.scenario import CurrentScenario


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

    def prompt_model_for_scenario(self, scen: CurrentScenario) -> str:
        self.set_current_goal(scen.get_goal())
        contexts = self.get_context_through_rag(self.get_current_goal())
        if scen.is_natural_language():
            self._last_system_message = build_sys_msg_nl(scen.get_no_arms(), self.get_current_goal())
            self._last_user_message = build_user_msg_nl(scen.get_objects(), scen.get_observed_objects(), contexts)
        else:
            self._last_system_message = build_sys_msg_fl()
            self._last_user_message = build_user_msg_fl(scen.get_actions(), scen.get_objects(), contexts)
        return self.prompt_model(self.get_last_system_message(), self.get_last_user_message())

    def prompt_model(self, system_msg: str, user_msg: str) -> str:
        pass
