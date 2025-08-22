system_prompt_nl = """Imagine you are a robot with {n_arms} arms interacting in a household environment. 
    You are tasked with executing the goal ''{goal}'' by providing a short sequence of actions in natural language."""

system_prompt_fl = """Translate the following list of actions written in natural language into a formal language defined by the provided primitive actions.\n
    Each action in natural language may correspond to multiple primitive actions. The order of actions is important, keep it."""


def build_user_msg_nl(contexts=None, add_rules=False) -> str:
    prompt = """Respond with detailed but simple instructions in English. Each line must consists of only one action, where the mentioned objects must be items from the following list: {objects}.\n
    Currently, you can see the following objects: ``{observed}''"""
    if contexts is not None and len(contexts) > 0:
        prompt += "\nUse the following contexts for creating your action sequence:\n"
        for cont in contexts:
            prompt += f"{cont[0]}\n"
    if add_rules:
        prompt += f"""\nYou must obey the following commonsense rules:
        1. You must have at least one empty hand before you can pick up an object or open or close a joint.
        2. When you sprinkle or pour something into a container, there must not be objects placed on top of the container.
        3. You can only take actions on objects that you can see.
        4. If you cannot see an object, it may be behind a door or inside drawer.
        5. If you cannot see the inside of a space, you must open its door or drawer before you can pick objects from it or place objects inside it."""
    return prompt


def build_user_msg_fl(contexts=None) -> str:
    prompt = """Actions: {actions}\nPrimitives: {set_of_actions}\n
    The primitive actions include argument types. Please use these objects in the respective types: {objects}
    Note that:\n
    1. The arguments shouldn't include robot parts, e.g., 'arm', 'gripper'.
    2. If a new object not mentioned in the set of objects is used as arguments, please name it with the same as the object that constitutes it the most and exists in the given list of objects.
    3. If one action cannot be translated into the above set, skip that action.\n
    """
    if contexts is not None and len(contexts) > 0:
        prompt += "Use the following contexts for creating your action sequence:\n"
        for cont in contexts:
            prompt += f"{cont[0]}\n"
    prompt += "Return the actions in a sequential list and give no explanation. Don't repeat the primitive actions and available objects."
    return prompt
