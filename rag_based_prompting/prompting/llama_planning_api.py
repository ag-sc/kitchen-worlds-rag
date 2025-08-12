import re

from RAG4Robots.src.utils.enums import ResourceType
from pybullet_planning.vlm_tools.vlm_planning_api import LLAMPApi
from pybullet_planning.vlm_tools.vlm_utils import add_prompt_answer_to_chat, process_test_for_html, add_answer_to_chat, \
    parse_subgoals
from rag_based_prompting.prompting.llama_api import *
from rag_based_prompting.prompting.prompts import *


class LlamaLocalPlanningApi(LLAMPApi):
    def __init__(self, open_goal, vlm_kwargs=dict(), **kwargs):
        super(LlamaLocalPlanningApi, self).__init__(open_goal, **kwargs)
        resources = kwargs.get('res', [(ResourceType.RECIPES, 0.5)])
        rag = RAGManager(resources)
        self.llm = LlamaLocalApi(rag, **vlm_kwargs)

    def parse_lines_into_lists_fn(self, string: str, **kwargs):
        return parse_lines_into_lists_llama(string, **kwargs)

    def _query_actions(self, goal, world, world_args, temperature: float = 0.0, first_query_kwargs=dict(),
                       include_preconditions=True, objects=None):
        return query_llama_for_actions(self, goal, world, world_args, temperature, first_query_kwargs,
                                       include_preconditions, objects)


class LlamaClusterPlanningApi(LLAMPApi):
    def __init__(self, open_goal, vlm_kwargs=dict(), **kwargs):
        super(LlamaClusterPlanningApi, self).__init__(open_goal, **kwargs)
        self.llm = LlamaClusterApi(**vlm_kwargs)

    def parse_lines_into_lists_fn(self, string: str, **kwargs):
        return parse_lines_into_lists_llama(string, **kwargs)

    def _query_actions(self, goal, world, world_args, temperature: float = 0.0, first_query_kwargs=dict(),
                       include_preconditions=True, objects=None):
        return query_llama_for_actions(self, goal, world, world_args, temperature, first_query_kwargs,
                                       include_preconditions, objects)


def parse_lines_into_lists_llama(string, n=1, planning_mode='actions'):
    actions = []
    for line in string.strip().split("\n"):
        match = re.match(r"^\s*\d+\.\s*(.+)$", line.strip())
        if match:
            actions.append(match.group(1))
    return actions


def query_llama_for_actions(api: LLAMPApi, goal, world, world_args, temperature: float = 0.0, first_query_kwargs=dict(),
                            include_preconditions=True, objects=None):
    reprompted = len(world_args['history']) > 0
    ## ------------- question 1: actions in english -------------
    prompt_english_to_actions = build_user_msg_nl(contexts=api.llm.rag_manager.query_all_dbs(goal)).format(
        **world_args)
    sys_msg_english_to_actions = system_prompt_nl.format(goal=goal, **world_args)

    actions_english, actions_string, responses = api.load_llm_answers()
    first_prompt_name = 'actions_english'
    if actions_english is None or reprompted:
        first_prompt_name += api.suffix
        actions_english = api.ask(prompt_english_to_actions, prompt_name=first_prompt_name, temperature=temperature,
                                  **dict(first_query_kwargs, sys_msg=sys_msg_english_to_actions, image_name=None))

        ## ------------- question 2: actions in predicates -------------
        from pybullet_planning.vlm_tools.prompts_gpt4v import list_of_actions_with_preconditions, list_of_actions
        actions = list_of_actions_with_preconditions if include_preconditions else list_of_actions
        description = world.get_objects_by_type(objects)
        prompt_actions_to_fl = build_user_msg_fl().format(actions=actions_english, set_of_actions=actions,
                                                          objects=description)
        if actions_string is None or reprompted:
            actions_string = api.ask(prompt_actions_to_fl, prompt_name=f'actions_string{api.suffix}',
                                     temperature=temperature,
                                     **dict(first_query_kwargs, sys_msg=system_prompt_fl, image_name=None))

    print(actions_string)
    print('-' * 40 + '\n')
    chat = []
    add_prompt_answer_to_chat(chat, api.llm.memory[first_prompt_name])
    chat += [('prompt', process_test_for_html(prompt_actions_to_fl))]
    add_answer_to_chat(chat, actions_string)

    ## ------------- process the generated response -------------
    actions, actions_to_print = parse_subgoals(world, actions_string, api.parse_lines_into_lists_fn,
                                               planning_mode=api.planning_mode)
    for one in actions_to_print:
        chat += [('processed', '<br>'.join([str(s) for s in one]))]
    return actions, actions_to_print, chat
