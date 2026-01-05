"""
    Same as postprocess_outputs.py but with the addition of the agentlab specific postprocessing functions.
"""

import os
import argparse
import glob
import pickle
import gzip
from tqdm import tqdm
from agent.prompts import *
import copy
import re

from browser_env.helper_functions import (
    get_action_description_bgym,
)


def extract_action_from_response(response: str) -> str:
    # find the first occurence of action
    action_splitter = "```"
    pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError(f"Could not find action in response: {response}")


def set_paths():
    """
    If we do not do this, then webarena can complain.
    Does not even matter what the paths are, as long as they are set.
    TODO(smurty): fix this hack later
    """
    os.environ["GITLAB"] = "PASS"
    os.environ["SHOPPING"] = "PASS"
    os.environ["SHOPPING_ADMIN"] = "PASS"
    os.environ["REDDIT"] = "PASS"
    os.environ["GITLAB"] = "PASS"
    os.environ["MAP"] = "PASS"
    os.environ["WIKIPEDIA"] = "PASS"
    os.environ["HOMEPAGE"] = "PASS"
    return


def get_action_description_helper(obs, action):
    # get all bids
    all_bids = [o for o in obs["axtree_object"]["nodes"] if "browsergym_id" in o]
    bid_dict = {}
    for o in all_bids:
        if "name" in o:
            bid_dict[o["browsergym_id"]] = o["name"]["value"]
        else:
            bid_dict[o["browsergym_id"]] = ""
    action_str = get_action_description_bgym(
        action,
        bid_dict,
        action_set_tag="id_accessibility_tree",
        action_splitter="```",
    )

    return action_str


def get_all_data(exp_dir):
    previous_steps = [
        pickle.load(gzip.open(f, "rb")) for f in glob.glob(f"{exp_dir}/step_*.gz")
    ]
    step2action = {s.step: s for s in previous_steps}
    total_actions = len(previous_steps)
    action_descriptions = []
    obs = []
    all_data = []
    all_actions = []
    all_bgym_actions = []
    chat_messages = []
    if len(step2action) == 0:
        return all_data
    elif "url" not in step2action[0].obs:
        return all_data
    site = step2action[0].obs["url"]
    for i in range(total_actions):
        if (
            step2action[i].agent_info == {}
            or not step2action[i].agent_info.chat_messages
        ):
            return all_data
        chat_messages.append(step2action[i].agent_info.chat_messages)
        obs.append(step2action[i].obs["axtree_txt"])
        messages = [
            m
            for m in step2action[i].agent_info.chat_messages
            if m["role"] == "assistant"
        ]
        try:
            all_actions.append(extract_action_from_response(messages[-1]["content"]))
        except Exception as e:
            print(len(messages))
            print(
                "Error in extracting action from response. This must be a parsing error",
                e,
            )
            return all_data
        all_bgym_actions.append(step2action[i].action)
        webarena_action_ds = step2action[i].agent_info.extra_info["webarena_action_ds"]
        if webarena_action_ds:
            action_descriptions.append(
                get_action_description_helper(
                    step2action[i].obs,
                    webarena_action_ds,
                )
            )
        else:
            action_descriptions.append("None")
        if i > 0 and i % 4 == 0:
            try:
                extra_info = step2action[i].agent_info.extra_info
                if "reward" in extra_info and int(extra_info["reward"]["answer"]) >= 4:
                    instruction = extra_info["trajectory_label"]["answer"]
                    print(exp_dir, i, instruction)
                    all_data.append(
                        {
                            "intent": instruction,
                            "obs": copy.deepcopy(obs),
                            "chat_messages": copy.deepcopy(chat_messages),
                            "action_descriptions": copy.deepcopy(action_descriptions),
                            "parsed_actions": copy.deepcopy(all_actions),
                            "bgym_actions": copy.deepcopy(all_bgym_actions),
                            "task_id": exp_dir,
                            "sites": [site],
                            "summary": extra_info["summary"],
                        }
                    )

            except Exception as e:
                print(f"Error at {i} in {exp_dir}: {e}")
                continue
    return all_data


def postprocess_outputs(orig_dir):
    """
    We will also save all the action descriptions
    """
    all_episodes = [
        exp_dir for exp_dir in glob.glob(f"{orig_dir}/*") if os.path.isdir(exp_dir)
    ]
    all_data = []
    for exp_dir in tqdm(all_episodes):
        try:
            curr_data = get_all_data(exp_dir)
            all_data += curr_data
        except Exception as e:
            print(f"Error in {exp_dir}: {e}")
            continue

    print(f"Found {len(all_data)} trajectories...")
    with open(f"{orig_dir}/filtered_parsed.json", "w") as f:
        json.dump(all_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_dir", type=str, default="")
    args = parser.parse_args()
    orig_dir = args.orig_dir
    postprocess_outputs(orig_dir)
