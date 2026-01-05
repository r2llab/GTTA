import json

reward_lenient_system_prompt = """
An autonomous intelligent agent navigating a web browser is given an instruction by a user. Your objective is to give a score to the agent based on how well it completed its task. Your score must be on the scale of 1 to 5. Give a score of 5 only when there are no errors. To do this task you are provided with the following information:\n\nInstruction: This is the natural language instruction given to the agent.\nTrajectory: This is a sequence of natural language descriptions of the agent's interaction with the web-browser.\nPrevious sub-task accomplished: Optionally you will be given the previous sub-task the model has accomplished.\n\nTo be successful, it is very important to follow the following rules:\n1. Explictly think about what is needed to follow the instruction correctly on the website and if the trajectory reflects these steps.\n2. Give a score of 4 if there are minor errors, or if the task was more than 70% completed. For tasks that require inputting multiple values into a form (such as booking a hotel room or finding flights), give a score of 4, if the agent is still interacting with the form, and making meaningful progress. Give a score of 3 (or below) if the model made very little progress towards the given instruction. Additionalliy if the current instruction is not different from the previously accomplished sub-task, give a score of 3 (or below), because this implies that the model did wasteful interactions after accomplishing the sub-task. Your output should be structured as follows:\n\n[think]\nYour reasoning for the score\n[think]\n\n[output]\nyour score\n[output].\n\nHere are some random eaxmples and corresponding outputs:\n### First example\nInstruction:\nFind the cheapest flight ticket from SFO to NYC\n\nTrajectory\n\n1. Agent types 'SFO' in the 'From' field\n2. Agent types 'NYC' in the 'To' field\n3. Agent clicks on the 'Search' button\n4. Agent clicks on the 'Sort by Price' button\n5. Agent clicks on the 'Book' button\n\nPrevious Sub-task:\nAgent types 'SFO' in the 'From' field\n\nOutput:\n[think]\nLet's think step-by-step. The instruction is to find the cheapest flight ticket from SFO to NYC. The agent has already typed 'SFO' in the 'From' field. The agent then types 'NYC' in the 'To' field, clicks on the 'Search' button, clicks on the 'Sort by Price' button, and finally clicks on the 'Book' button. The agent has successfully completed the task.\n[think]\n\n[output]\n5\n[output]\n\n### Second example\nInstruction:\nFind out who won the 2020 NBA Finals\n\nTrajectory\n\n1. Agent types '2020 NBA Finals winner' in the search bar\n2. Agent clicks on the first search result\n3. Agent reads the article\n\nPrevious Sub-task:\nAgent types '2020 NBA Finals winner' in the search bar\n\nOutput:\n[think]\nLet's think step-by-step. The instruction is to find out who won the 2020 NBA Finals. The agent has already typed '2020 NBA Finals winner' in the search bar. The agent then clicks on the first search result and reads the article. The agent has successfully completed the task.\n[think]\n\n[output]\n5\n[output]\n\n### Third example\nInstruction:\nFind information about CS classes at Stanford\n\nTrajectory\n\n1. Agent looks up math classes at UW.\n2 Agent types 'CS classes at Stanford' in the search bar\n3. Agent cancels the search\n\nPrevious Sub-task:\nAgent types 'math classes at UW' in the search bar\n\nOutput:\n[think]\nLet's think step-by-step. The instruction is to find information about CS classes at Stanford. The agent has already typed 'math classes at UW' in the search bar. The agent then types 'CS classes at Stanford' in the search bar and cancels the search. The agent has not successfully completed the task.\n[think]\n\n[output]\n3\n[output]\n\n### Fourth example\nInstruction:\n Check out the transformers paper on arxiv, and go to the results section\n\nTrajectory\n\n1. Agent types 'transformers paper arxiv' in the search bar\n2. Agent clicks on the first search result\n3. Agent reads the abstract\n\nPrevious Sub-task:\nAgent types 'transformers paper arxiv' in the search bar\n\nOutput:\n[think]\nLet's think step-by-step. The instruction is to check out the transformers paper on arxiv and then go to the results section. The agent has already typed 'transformers paper arxiv' in the search bar. The agent then clicks on the first search result and reads the abstract. The agent has not gone to the results section. The agent has not successfully completed the task.\n[think]\n\n[output]\n3\n[output]
"""
examples = []
template = "Instruction:\n{instruction}\n\nTrajectory:\n\n{trajectory}\n\nPrevious Sub-task:\n{previous_subtask}"

meta_data = {
    "observation": "accessibility_tree",
    "action_type": "id_accessibility_tree",
    "keywords": ["instruction", "trajectory", "previous_subtask"],
    "prompt_constructor": "StructuredPassivePromptConstructor",
    "answer_phrase": "Reward: ",
    "action_splitter": ":",
}

json_data = {
    "intro": reward_lenient_system_prompt,
    "examples": examples,
    "template": template,
    "meta_data": meta_data,
}

with open("p_reward_lenient.json", "w") as f:
    json.dump(json_data, f, indent=4)
