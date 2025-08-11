from RAG4Robots.src.manager import RAGManager
from RAG4Robots.src.utils.enums import ResourceType
from pybullet_planning.vlm_tools.prompts_gpt4v import list_of_actions_with_preconditions
from rag_based_prompting.prompting.llama_api import LlamaLocalApi
from rag_based_prompting.prompting.prompts import *


def test_action_sequence_gen(task="make chicken soup"):
    rags = [(ResourceType.RECIPES, 0.5)]
    rag = RAGManager(rags)
    prompter = LlamaLocalApi(rag)

    objects = ['bottle', 'pan', 'pot', 'lid', 'chicken broth', 'spoon']
    prompt = build_user_msg_nl(contexts=prompter.rag_manager.query_all_dbs(task)).format(objects=str(objects),
                                                                                         observed=str(objects))
    sys_msg = system_prompt_nl.format(goal=task, n_arms=2)
    res = prompter._ask(prompt, prompt_name='test_action_sequence_gen', **dict(sys_msg=sys_msg, image_name=None))
    print(res)


def test_sequence_translation(task="make chicken soup"):
    # Example output from 'test_action_sequence_gen("make chicken soup")'
    actions = """1. Open the pot. 
    2. Pick up the pot.
    3. Put the pot on the stove.
    4. Open the lid.
    5. Pour the chicken broth into the pot.
    6. Add the chicken to the pot.
    7. Put the spoon in the pot.
    8. Open the pan.
    9. Pick up the spoon.
    10. Pour the chicken broth and chicken into the pan.
    11. Add the spoon to the pan.
    12. Open the lid of the pan.
    13. Put the spoon in the pan.
    14. Open the bottle.
    15. Pour the chicken broth into the bottle.
    16. Put the bottle on the counter.
    17. Pick up the bottle.
    18. Open the pot.
    19. Put the spoon in the pot.
    20. Open the lid of the pot.
    21. Pour the chicken broth and chicken into the pot.
    22. Add the spoon to the pot.
    23. Put the spoon in the pot.
    24. Open the pot.
    25. Put the spoon in the pot."""
    rags = [(ResourceType.RECIPES, 0.5)]
    rag = RAGManager(rags)
    prompter = LlamaLocalApi(rag)

    objects = ['bottle', 'pan', 'pot', 'lid', 'chicken broth', 'spoon']
    prompt = build_user_msg_fl(contexts=prompter.rag_manager.query_all_dbs(task)).format(actions=actions,
                                                                                         set_of_actions=list_of_actions_with_preconditions,
                                                                                         objects=str(objects))
    res = prompter._ask(prompt, prompt_name='test_action_sequence_gen',
                        **dict(sys_msg=system_prompt_fl, image_name=None))
    print(res)


if __name__ == '__main__':
    test_action_sequence_gen()
    test_sequence_translation()
