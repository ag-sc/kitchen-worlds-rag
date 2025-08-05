from typing import List


def build_sys_msg_nl(no_arms: int, goal: str) -> str:
    return f"""Imagine you are a robot with {no_arms} arms interacting in a household environment. 
    You are tasked with executing the goal ''{goal}'' by providing a short sequence of actions in natural language."""


def build_user_msg_nl(objects: List[str], obs_objs: List[str], contexts=None, add_rules=True) -> str:
    prompt = f"""Respond with detailed but simple instructions in English. Each line must consists of only one action, where the mentioned objects must be items from the following list: {objects}.\n
    Currently, you can see the following objects: ``{obs_objs}''"""
    if contexts is not None:
        prompt += f"Use the following contexts for creating your action sequence: {contexts}\n"
    if add_rules:
        prompt += f"""You must obey the following commonsense rules:\n
        1. You must have at least one empty hand before you can pick up an object or open or close a joint.\n
        2. When you sprinkle or pour something into a container, there must not be objects placed on top of the container.\n
        3. You can only take actions on objects that you can see.\n
        4. If you cannot see an object, it may be behind a door or inside drawer.\n
        5. If you cannot see the inside of a space, you must open its door or drawer before you can pick objects from it or place objects inside it."""
    return prompt


def build_sys_msg_fl() -> str:
    return f"""Translate the following list of actions written in natural language into a formal language defined by the provided primitive actions.\n
    Each action in natural language may correspond to multiple primitive actions. Return the actions in a list and give no explanation."""


def build_user_msg_fl(actions: List[str], objects: List[str], contexts=None) -> str:
    prompt = f"""Actions: {actions}\nPrimitives: {get_primitives()}\n
    The primitive actions include argument types. Please use these objects in the respective types: {objects}
    Note that:\n
    1. The arguments shouldn't include robot parts, e.g., 'arm', 'gripper'.\n
    2. If a new object not mentioned in the set of objects is used as arguments, please name it with the same as the object that constitutes it the most and exists in the given list of objects.\n
    3. If one action cannot be translated into the above set, skip that action.
    """
    if contexts is not None:
        prompt += f"Use the following contexts for creating your action sequence: {contexts}\n"
    return prompt


def get_primitives() -> str:
    return """'pick(<obj>)': it contains one argument. The robot must have an empty hand to pick up an object.\n
        'place(<obj>, <surface>)': it contains two arguments.\n
        'sprinkle(<obj>, <container>)': it contains two arguments. There must be no object placed on top of the container.\n
        'stir(<container>, <obj>)': it contains two arguments. There must be no object placed on top of the container.\n
        'chop(<obj>, <utensil>)': it contains two arguments.\n
        'open(<joint>)': it contains one argument.\n
        'close(<joint>)': it contains one argument.\n
        'nudge(<door>)': it contains one argument. This action is applied to open a door even wider.\n
        'turn-on(<knob>)': turning on a stove knob, it contains one argument.\n
        'turn-off(<knob>)': turning off a stove knob, it contains one argument.\n
        'turn-on(<handle>)': turning on a faucet handle, it contains one argument.\n
        'turn-off(<handle>)': turning off a faucet handle, it contains one argument."""
