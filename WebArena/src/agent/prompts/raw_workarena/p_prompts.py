import argparse
import os
import json
from browsergym.core.action.highlevel import HighLevelActionSet

exploration_prompt_outline = {
    "intro": """You are an autonomous intelligent agent tasked with performing tasks on a web interface. Your objective is to simulate a task that a person might request, by interacting with the interface through the use of specific actions.

Here's the information you'll have:
{information}

You can perform the following actions:
{action_space}

If you are done exploring, you can issue the stop action: ```stop```

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12. In summary, the next action I will perform is ```click("12")```
"

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation.
2. You should only issue one action at a time.
3. You should reason step by step and then issue the next action.
4. Make sure to wrap your action in a code block using triple backticks.
5. The DOM / Accessibility Tree only shows the visible part of the webpage. If you need to interact with elements that are not visible, you can scroll to them using the scroll action. Often submit buttons are not visible and are at the bottom of the page. To scroll to the bottom of the page, use the scroll action with a large value for the y coordinate.
6. To generate an interesting task, make sure you issue atleast 4 actions before stopping. More interesting tasks typically involve more interactions with the browser.
7. You can issue atmost 20 actions before stopping, but feel free to output the stop action early if you want to stop exploring. Don't generate anything after stop.""",
    "examples": [],
    "template": """{information_template}""",
    "meta_data": {
        "observation": "accessibility_tree",
        "action_type": "id_accessibility_tree",
        "keywords": [
            "url",
            "observation",
            "previous_action",
            "trajectory",
            "person_description",
        ],
        "prompt_constructor": "CoTPromptConstructor",
        "answer_phrase": "In summary, the next action I will perform is",
        "action_splitter": "```",
    },
}
policy_prompt_outline = {
    "intro": """You are an autonomous intelligent agent tasked with performing tasks on a web interface. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
{information}

You can perform the following actions:
{action_space}

If you are done with the task, you can issue the stop action: ```stop```

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. Make sure to wrap your action in a code block using triple backticks.
5. The DOM / Accessibility Tree only shows the visible part of the webpage. If you need to interact with elements that are not visible, you can scroll to them using the scroll action. Often submit buttons are not visible and are at the bottom of the page. To scroll to the bottom of the page, use the scroll action with a large value for the y coordinate.
6. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.""",
    "examples": [],
    "template": """{information_template}""",
    "meta_data": {
        "observation": "accessibility_tree",
        "action_type": "id_accessibility_tree",
        "keywords": ["url", "objective", "observation", "previous_action"],
        "prompt_constructor": "CoTPromptConstructor",
        "answer_phrase": "In summary, the next action I will perform is",
        "action_splitter": "```",
    },
}
reward_prompt_outline = {
    "intro": """An autonomous intelligent agent navigating a web-based interface is given a task by a user. Your objective is to give a score to the agent based on how well it completed its task. Your score must be on the scale of 1 to 5. Give a score of 5 only when there are no errors.

To do this task you are provided with the following information:
Instruction: This is the natural language instruction given to the agent.
Trajectory: This is a sequence of natural language descriptions of the agent's interaction with the web-browser.

To be successful, it is very important to follow the following rules:
1. Explictly think about what is needed to follow the instruction correctly on the website and if the trajectory reflects these steps.
2. Start by thinking by outputing Thought: <your-reasoning>.3. End your answer by strictly following the format \"Reward: <your-answer>\" for your output.
""",
    "examples": [],
    "template": "Instruction:\n{instruction}\n\nTrajectory:\n\n{trajectory}",
    "meta_data": {
        "observation": "accessibility_tree",
        "action_type": "id_accessibility_tree",
        "keywords": ["instruction", "trajectory"],
        "prompt_constructor": "PassivePromptConstructor",
        "answer_phrase": "Reward: ",
        "action_splitter": ":",
    },
}
instruction_relabel_prompt_outline = {
    "intro": """Given a task from a user, an autonomous intelligent agent carries out a sequence of actions on a web-interface. The actions the agent can take fall under the following categories:
{action_space} 

Your objective is to guess the instruction the user gave, given the following information:
Trajectory: This is a sequence of natural language descriptions of the agent's interaction with the web-browser.

To be successful, it is very important to follow the following rules:
1. Explictly think about how the trajectory is a valid way to achieve the instruction, before outputing the instruction.
2. Start by thinking by outputing Thought: <your-reasoning>.
3. End your answer by strictly following the format \"Instruction: <your-answer>\" for your output.""",
    "examples": [],
    "template": "Trajectory:\n\n{trajectory}",
    "meta_data": {
        "observation": "accessibility_tree",
        "action_type": "id_accessibility_tree",
        "keywords": ["trajectory"],
        "prompt_constructor": "PassivePromptConstructor",
        "answer_phrase": "Instruction: ",
        "action_splitter": ":",
    },
}
state_changelog_prompt_outline = {
    "intro": """You are given the output of an action taken by an autonomous intelligent agent navigating a web-interface to fulfill a task given by a user.  Your objective is to produce a description of the changes made to the state of the browser.
    
Here's the information you'll have:
{information}
The action taken by the agent: This is the action taken by the agent to change the state of the browser. The actions the agent can take come from the following categories:
{action_space}

To be successful, it is very important to follow the following rules:
1. Explictly think about the various features on the website and how the interaction with the website changed these features
2. Provide the description of changes in one or two sentences.
3. Strictly follow the format \"State change: <your-answer>\" for your output""",
    "examples": [],
    "template": "Initial state:\n{init_observation}\nFinal state:\n{final_observation}\nAction: {action}",
    "meta_data": {
        "observation": "accessibility_tree",
        "action_type": "id_accessibility_tree",
        "keywords": ["init_observation", "final_observation", "action"],
        "prompt_constructor": "PassivePromptConstructor",
        "answer_phrase": "State change: ",
        "action_splitter": ":",
    },
}


def process_policy_prompt(args):
    if args.action_space_type == "basic":
        action_set = HighLevelActionSet(
            strict=False, demo_mode="off", multiaction=False, subsets=["bid"]
        )
        action_description = action_set.describe(
            with_long_description=True, with_examples=True
        )
    elif args.action_space_type == "servicenow":
        action_set = HighLevelActionSet(
            strict=False,
            demo_mode="off",
            multiaction=False,
            subsets=["chat", "bid"],
        )
        action_description = action_set.describe(
            with_long_description=True, with_examples=True
        )
    else:
        raise NotImplementedError("Action space type not implemented")
    observation_str = ""
    if args.observation_type == "basic":
        observation_str += "Current Accessibility Tree: This is a simplified representation of the current webpage, providing key information\n"
        observation_str += (
            "The user's objective: This is the task you're trying to complete.\n"
        )
        observation_str += "The previous action: This is the action you just performed. It may be helpful to track your progress.\n"

        policy_prompt_outline["intro"] = policy_prompt_outline["intro"].format(
            information=observation_str, action_space=action_description
        )
        policy_prompt_outline["template"] = (
            "CURRENT ACCESSIBILITY TREE:\n{observation}\nOBJECTIVE: {objective}\nPREVIOUS ACTION: {previous_action}"
        )
        policy_prompt_outline["meta_data"]["keywords"] = [
            "observation",
            "objective",
            "previous_action",
        ]
    elif args.observation_type == "dom":
        observation_str += "DOM Representation: This is the current webpage's Document Object Model (DOM) representation as a string.\n"
        observation_str += (
            "The user's objective: This is the task you're trying to complete.\n"
        )
        if args.policy_with_history:
            observation_str += "Trajectory: This is a sequence of natural language descriptions of the agent's interaction with the web-browser.\n"
        observation_str += "The previous action: This is the action you just performed. It may be helpful to track your progress.\n"

        policy_prompt_outline["intro"] = policy_prompt_outline["intro"].format(
            information=observation_str, action_space=action_description
        )
        policy_prompt_outline["template"] = (
            "DOM Representation:\n{observation}\nOBJECTIVE: {objective}\nPREVIOUS ACTION: {previous_action}"
        )
        if args.policy_with_history:
            policy_prompt_outline["template"] += "\nTRAJECTORY:\n{trajectory}"
            policy_prompt_outline["meta_data"]["keywords"] = [
                "observation",
                "objective",
                "previous_action",
                "trajectory",
            ]
        else:
            policy_prompt_outline["meta_data"]["keywords"] = [
                "observation",
                "objective",
                "previous_action",
            ]
    else:
        raise NotImplementedError("Observation type not implemented")
    return


def process_exploration_prompt(args):
    if args.action_space_type == "basic":
        action_set = HighLevelActionSet(
            strict=False, demo_mode="off", multiaction=False, subsets=["bid"]
        )
        action_description = action_set.describe(
            with_long_description=False, with_examples=True
        )
    elif args.action_space_type == "servicenow":
        action_set = HighLevelActionSet(
            strict=False,
            demo_mode="off",
            multiaction=False,
            subsets=["chat", "bid"],
        )
        action_description = action_set.describe(
            with_long_description=True, with_examples=True
        )
    else:
        raise NotImplementedError("Action space type not implemented")
    observation_str = ""
    if args.observation_type == "basic":
        observation_str += "Current Accessibility Tree: This is a simplified representation of the current webpage, providing key information\n"
        observation_str += "The previous action: This is the action you just performed. It may be helpful to track your progress.\n"
        observation_str += "Trajectory: This is a sequence of natural language descriptions of the agent's interaction with the web-browser.\n"
        if args.with_persona:
            observation_str += "Person Description: The description of a specific kind of person whose task you are supposed to simulate.\n"
        exploration_prompt_outline["intro"] = exploration_prompt_outline[
            "intro"
        ].format(information=observation_str, action_space=action_description)
        if args.with_persona:
            exploration_prompt_outline["template"] = (
                "CURRENT ACCESSIBILITY TREE:\n{observation}\nPerson Description:\n{person_description}\nTRAJECTORY:\n{trajectory}\nPREVIOUS ACTION: {previous_action}"
            )
            exploration_prompt_outline["meta_data"]["keywords"] = [
                "observation",
                "previous_action",
                "trajectory",
                "person_description",
            ]
        else:
            exploration_prompt_outline["template"] = (
                "CURRENT ACCESSIBILITY TREE:\n{observation}\nTRAJECTORY:\n{trajectory}\nPREVIOUS ACTION:\n{previous_action}"
            )
            exploration_prompt_outline["meta_data"]["keywords"] = [
                "observation",
                "previous_action",
                "trajectory",
            ]
    elif args.observation_type == "dom":
        observation_str += "DOM Representation: This is the current webpage's Document Object Model (DOM) representation as a string.\n"
        observation_str += "The previous action: This is the action you just performed. It may be helpful to track your progress.\n"
        observation_str += "Trajectory: This is a sequence of natural language descriptions of the agent's interaction with the web-browser.\n"
        if args.with_persona:
            observation_str += "Person Description: The description of a specific kind of person whose task you are supposed to simulate.\n"
        exploration_prompt_outline["intro"] = exploration_prompt_outline[
            "intro"
        ].format(information=observation_str, action_space=action_description)
        if args.with_persona:
            exploration_prompt_outline["template"] = (
                "DOM Representation:\n{observation}\nPerson Description:\n{person_description}\nTRAJECTORY:\n{trajectory}\nPREVIOUS ACTION: {previous_action}"
            )
            exploration_prompt_outline["meta_data"]["keywords"] = [
                "observation",
                "previous_action",
                "trajectory",
                "person_description",
            ]
        else:
            exploration_prompt_outline["template"] = (
                "DOM Representation:\n{observation}\nTRAJECTORY:\n{trajectory}\nPREVIOUS ACTION: {previous_action}"
            )
            exploration_prompt_outline["meta_data"]["keywords"] = [
                "observation",
                "previous_action",
                "trajectory",
            ]
    else:
        raise NotImplementedError("Observation type not implemented")
    return


def process_reward_prompt(args):
    if args.action_space_type == "basic":
        action_set = HighLevelActionSet(
            strict=False, demo_mode="off", multiaction=False, subsets=["bid"]
        )
        action_description = action_set.describe(
            with_long_description=False, with_examples=True
        )
    elif args.action_space_type == "servicenow":
        action_set = HighLevelActionSet(
            strict=False,
            demo_mode="off",
            multiaction=False,
            subsets=["chat", "bid"],
        )
        action_description = action_set.describe(
            with_long_description=True, with_examples=True
        )
    else:
        raise NotImplementedError("Action space type not implemented")
    reward_prompt_outline["intro"] = reward_prompt_outline["intro"].format(
        action_space=action_description
    )
    return


def process_instruction_relabel_prompt(args):
    if args.action_space_type == "basic":
        action_set = HighLevelActionSet(
            strict=False, demo_mode="off", multiaction=False, subsets=["bid"]
        )
        action_description = action_set.describe(
            with_long_description=False, with_examples=True
        )
    elif args.action_space_type == "servicenow":
        action_set = HighLevelActionSet(
            strict=False,
            demo_mode="off",
            multiaction=False,
            subsets=["chat", "bid"],
        )
        action_description = action_set.describe(
            with_long_description=True, with_examples=True
        )
    else:
        raise NotImplementedError("Action space type not implemented")
    instruction_relabel_prompt_outline["intro"] = instruction_relabel_prompt_outline[
        "intro"
    ].format(action_space=action_description)
    return


def process_state_changelog_prompt(args):
    if args.observation_type == "basic":
        observation = "Initial state of the browser as an accessibility tree: This is a simplified representation of the webpage, providing key information.\n"
        observation += "Final state of the browser as an accessibility tree: This is the accessibility tree representation after the agent took the action.\n"
    elif args.observation_type == "dom":
        observation = "Initial state of the browser as a DOM representation: This is the webpage's Document Object Model (DOM) representation as a string.\n"
        observation += "Final state of the browser as a DOM representation: This is the DOM representation after the agent took the action.\n"
    else:
        raise NotImplementedError("Observation type not implemented")

    if args.action_space_type == "basic":
        action_set = HighLevelActionSet(
            strict=False, demo_mode="off", multiaction=False, subsets=["bid"]
        )
        action_description = action_set.describe(
            with_long_description=False, with_examples=True
        )
    elif args.action_space_type == "servicenow":
        action_set = HighLevelActionSet(
            strict=False,
            demo_mode="off",
            multiaction=False,
            subsets=["chat", "bid"],
        )
        action_description = action_set.describe(
            with_long_description=True, with_examples=True
        )
    else:
        raise NotImplementedError("Action space type not implemented")
    state_changelog_prompt_outline["intro"] = state_changelog_prompt_outline[
        "intro"
    ].format(action_space=action_description, information=observation)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Outlines")
    parser.add_argument("--observation_type", type=str, default="basic")
    parser.add_argument("--action_space_type", type=str, default="basic")
    parser.add_argument("--with_persona", action="store_true")
    parser.add_argument("--policy_with_history", action="store_true")
    args = parser.parse_args()

    process_policy_prompt(args)
    process_exploration_prompt(args)
    process_reward_prompt(args)
    process_state_changelog_prompt(args)
    process_instruction_relabel_prompt(args)
    dir_name = "agent/prompts/jsons_servicenow"
    os.makedirs(dir_name, exist_ok=True)

    prompt_name2outline = {
        "policy": policy_prompt_outline,
        "reward": reward_prompt_outline,
        "instruction_relabel": instruction_relabel_prompt_outline,
        "state_changelog": state_changelog_prompt_outline,
    }

    if args.policy_with_history:
        prompt_name2outline["policy_with_history"] = policy_prompt_outline
        del prompt_name2outline["policy"]

    if args.with_persona:
        prompt_name2outline["cot_exploration_with_history_persona"] = (
            exploration_prompt_outline
        )
    else:
        prompt_name2outline["cot_exploration_with_history"] = exploration_prompt_outline

    for prompt_name, prompt in prompt_name2outline.items():
        with open(f"{dir_name}/p_{prompt_name}.json", "w+") as f:
            json.dump(prompt, f, indent=2)
    print(f"Done convert python files to json")
