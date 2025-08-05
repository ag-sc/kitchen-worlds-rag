from RAG4Robots.src.manager import RAGManager
from RAG4Robots.src.utils.enums import ResourceType
from rag_based_prompting.prompting.llama_prompter import LlamaLocalPrompter
from rag_based_prompting.scenario import CurrentScenario


def test_action_sequence_gen(task="make chicken soup"):
    rags = [(ResourceType.RECIPES, 0.5)]
    rag = RAGManager(rags)
    prompter = LlamaLocalPrompter(rag)

    scen = CurrentScenario(task, 2, ['bottle', 'pan', 'pot', 'lid', 'chicken broth', 'spoon'],
                           ['bottle', 'pan', 'pot', 'lid', 'chicken broth', 'spoon'])
    res = prompter.prompt_model_for_scenario(scen)
    print(res)


def test_sequence_translation():
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
    prompter = LlamaLocalPrompter(rag)

    scen = CurrentScenario("make chicken soup", 2, ['bottle', 'pan', 'pot', 'lid', 'chicken broth', 'spoon'],
                           ['bottle', 'pan', 'pot', 'lid', 'chicken broth', 'spoon'])
    scen.set_actions(actions)
    res = prompter.prompt_model_for_scenario(scen)
    print(res)


if __name__ == '__main__':
    test_action_sequence_gen()
    test_sequence_translation()
