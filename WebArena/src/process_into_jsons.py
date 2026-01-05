"""
    Script to process the counterfactual data into a format for SFT or preference tuning
"""

import argparse
import pickle
import glob, gzip
import os

from collections import defaultdict as ddict
from tqdm import tqdm
from agentlab.llm.llm_utils import count_tokens
import json

from dataclasses import is_dataclass
from browsergym.core.action.parsers import highlevel_action_parser
from agents.judge_prompts import (
    SystemPromptJudge,
    JudgePrompt,
    SystemPromptLMCriticSelector,
    LMCriticSelector,
)


from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.chat_api import make_system_message, make_user_message


def dataclass_to_dict(obj):
    """
    Thanks: https://github.com/ServiceNow/ui-copilot/blob/main/finetuning/src/finetuning/utils/utils.py#L278
    """
    if is_dataclass(obj):
        result = {}
        for key, value in obj.__dict__.items():
            if is_dataclass(value):
                result[key] = dataclass_to_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    dataclass_to_dict(item) if is_dataclass(item) else item for item in value
                ]
            elif isinstance(value, dict):
                result[key] = {
                    k: dataclass_to_dict(v) if is_dataclass(v) else v for k, v in value.items()
                }
            else:
                result[key] = value
        return result
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) if is_dataclass(item) else item for item in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) if is_dataclass(v) else v for k, v in obj.items()}
    else:
        return obj


def get_actions(directory):
    previous_steps = [pickle.load(gzip.open(f, "rb")) for f in glob.glob(f"{directory}/step_*.gz")]
    step2action = {s.step: s for s in previous_steps}
    total_actions = len(previous_steps) - 1
    actions = [step2action[i].action for i in range(total_actions)]

    thoughts = [step2action[i].agent_info.think for i in range(total_actions)]
    chat_messages = [step2action[i].agent_info.chat_messages for i in range(total_actions)]
    return chat_messages, actions, thoughts


def get_counterfactual_and_branch_step(directory_name):
    exp_args = pickle.load(open(os.path.join(directory_name, "exp_args.pkl"), "rb"))
    # this is the pairwise reward that compares the original trajectory with the counterfactual trajectory
    reward = pickle.load(open(os.path.join(directory_name, "pairwise_reward.pkl"), "rb"))
    return exp_args.branch_time_step, exp_args.candidate_action, reward["reward"]


def get_reason_and_action(step):
    try:
        action = step.action
        chat_messages = step.agent_info.chat_messages
        if "agent_thoughts" in step.agent_info.extra_info:
            # backward compatibility
            all_reasons = step.agent_info.extra_info["agent_thoughts"]
        else:
            all_reasons = step.agent_info.extra_info["candidate_thoughts"]
        all_actions = step.agent_info.extra_info["candidate_actions"]
        return chat_messages, all_actions, all_reasons, action
    except:
        return None, None, None, None


def sanity_checks(counterfactual_file):
    if not os.path.exists(f"{counterfactual_file}/summary.pkl"):
        return False
    exp_args = pickle.load(open(f"{counterfactual_file}/exp_args.pkl", "rb"))
    branch_point = exp_args.branch_time_step
    total_actions = len(pickle.load(open(f"{counterfactual_file}/summary.pkl", "rb")))

    # the branch_point is k such that it is less than equal to total_actions -1

    if branch_point > total_actions - 1:
        return False

    total_steps = len([f for f in os.listdir(counterfactual_file) if f.startswith("step_")])
    # total_steps should be one greater than total_actions
    if total_steps != total_actions + 1:
        return False

    return True


def get_all_trajectories(args, all_files):
    """
    Get all the trajectories
    """
    trajectories = {}
    for f in tqdm(all_files):
        try:
            if not os.path.exists(f"{f}/reward.pkl"):
                continue
            exp_args = pickle.load(open(os.path.join(f, "exp_args.pkl"), "rb"))
            task_name = exp_args.env_args.task_name
            task_id = int(task_name.split("_")[-1])
            if task_id < args.task_start:
                continue
            reward = pickle.load(open(f"{f}/reward.pkl", "rb"))
            if int(reward["reward"]) < 4:
                continue

            chat_messages, all_actions, all_thoughts = get_actions(f)

            trajectories[f] = (chat_messages, all_actions, all_thoughts, reward["reward"])
        except Exception as e:
            print(e)
            continue
    return trajectories


def get_all_data(args, all_files, with_sanity_check=True):
    rewards = {}
    reward_file = args.reward_file
    for f in tqdm(all_files):
        # get the summary pickle file
        if with_sanity_check and not sanity_checks(f):
            continue
        else:
            exp_args = pickle.load(open(f"{f}/exp_args.pkl", "rb"))
            orig_dir = exp_args.orig_exp_dir
            # get the branch time step
            branch_point = exp_args.branch_time_step
            # get the action at the branch time step in the original directory
            if not os.path.exists(f"{orig_dir}/step_{branch_point}.pkl.gz"):
                continue
            step_original = pickle.load(gzip.open(f"{orig_dir}/step_{branch_point}.pkl.gz", "rb"))
            chat_messages, all_action, all_reasons, action_original = get_reason_and_action(
                step_original
            )
            action_counterfactual = exp_args.candidate_action
            if all_action is None:
                print("Error")
                continue
            if os.path.exists(f"{f}/{reward_file}"):
                reward = pickle.load(open(f"{f}/{reward_file}", "rb"))
                rewards[f] = (
                    chat_messages,
                    reward["reward"],
                    action_original,
                    action_counterfactual,
                    all_action,
                    all_reasons,
                )
            else:
                continue

    processed_rewards = {}
    for f, (
        chat_messages,
        reward,
        action_original,
        action_counterfactual,
        all_action,
        all_reasons,
    ) in rewards.items():
        # print(len(all_action), len(all_reasons))
        all_actions_parsed = [
            highlevel_action_parser.parse_string(action).as_list()[0] for action in all_action
        ]
        orig_action_parsed = highlevel_action_parser.parse_string(action_original).as_list()[0]
        counterfactual_action_parsed = highlevel_action_parser.parse_string(
            action_counterfactual
        ).as_list()[0]

        assert (orig_action_parsed in all_actions_parsed) and (
            counterfactual_action_parsed in all_actions_parsed
        )

        # get the index of the original action
        orig_action_index = all_actions_parsed.index(orig_action_parsed)
        counterfactual_action_index = all_actions_parsed.index(counterfactual_action_parsed)

        # get the thoughts at the original action index
        processed_rewards[f] = (
            chat_messages,
            reward,
            action_original,
            action_counterfactual,
            all_reasons[orig_action_index],
            all_reasons[counterfactual_action_index],
        )

    return processed_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, help="Directory containing traces")
    parser.add_argument("--model_name", type=str, default="'meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--exp_name", type=str, default="sft")

    parser.add_argument(
        "--experiment_mode",
        type=str,
        choices=["q_judge", "rejection_tuning", "q_judge_with_future", "q_judge_reranker"],
        help="What kind of experiment are we running. 1. q_judge: train q-judge without future, 2. rejection_tuning: train with RFT, 3. q_judge_with_future: train q-judge with future. 4. q_judge_reranker: train q-judge as an action re-ranker",
    )
    parser.add_argument(
        "--reward_file",
        type=str,
        default="pairwise_reward.pkl",
        help="The file containing the reward",
    )
    parser.add_argument(
        "--task_start",
        type=int,
        default=0,
        help="The task id from which to start processing",
    )

    parser.add_argument(
        "--myopic_id",
        type=int,
        default=-1,
        help="if non-zero then pick out labels from a myopic judge",
    )
    args = parser.parse_args()

    counterfactual_dir = args.trace_dir
    trace2dir_counterfactual = ddict(list)

    task2actions = {}
    all_files = []
    for d1 in os.listdir(args.trace_dir):
        if os.path.isdir(os.path.join(args.trace_dir, d1)):
            for d2 in os.listdir(os.path.join(args.trace_dir, d1)):
                all_files.append(os.path.join(args.trace_dir, d1, d2))

    if args.experiment_mode == "q_judge_with_future":
        all_task_names = []
        outputs = []
        for f in tqdm(all_files):
            if not os.path.exists(f"{f}/judge_out.pkl"):
                continue
            if not sanity_checks(f):
                continue

            q_judge = pickle.load(open(f"{f}/judge_out.pkl", "rb"))
            exp_args = pickle.load(open(f"{f}/exp_args.pkl", "rb"))

            task_name = exp_args.env_args.task_name
            orig_exp_dir = exp_args.orig_exp_dir
            if "chat_messages" not in q_judge:
                continue
            chat_messages = q_judge["chat_messages"]
            full_prompt = "{}\n{}".format(chat_messages[0]["content"], chat_messages[1]["content"])
            if "message" in chat_messages[-1]:
                chat_messages[-1]["content"] = chat_messages[-1]["message"]
                del chat_messages[-1]["message"]
            n_tokens = count_tokens(full_prompt, args.model_name)
            q_judge["n_tokens"] = n_tokens
            q_judge["task_name"] = task_name
            q_judge["prompt"] = full_prompt
            q_judge["output"] = chat_messages[-1]["content"]
            q_judge["messages"] = chat_messages
            del q_judge["chat_messages"]
            all_task_names.append(task_name)
            outputs.append(q_judge)
    elif args.experiment_mode == "q_judge":
        all_task_names = []
        outputs = []
        system_prompt = {"role": "system", "content": SystemPromptJudge().prompt}
        FLAGS_GPT_4o_webarena = GenericPromptFlags(
            obs=dp.ObsFlags(
                use_html=False,
                use_ax_tree=True,
                use_focused_element=True,
                use_error_logs=True,
                use_history=True,
                use_past_error_logs=False,
                use_action_history=True,
                use_think_history=False,
                use_diff=False,
                html_type="pruned_html",
                use_screenshot=False,
                use_som=False,
                extract_visible_tag=True,
                extract_clickable_tag=True,
                extract_coords="False",
                filter_visible_elements_only=False,
            ),
            action=dp.ActionFlags(
                multi_actions=False,
                action_set="webarena",
                long_description=False,
                individual_examples=False,
            ),
            use_plan=False,
            use_criticise=False,
            use_thinking=True,
            use_memory=False,
            use_concrete_example=True,
            use_abstract_example=True,
            use_hints=True,
            enable_chat=False,
            max_prompt_tokens=None,
            be_cautious=True,
            extra_instructions=None,
        )
        for f in tqdm(all_files):
            exp_args = pickle.load(open(f"{f}/exp_args.pkl", "rb"))
            orig_exp_dir = exp_args.orig_exp_dir
            time_step = exp_args.branch_time_step
            if not os.path.exists(f"{f}/judge_out.pkl"):
                continue
            if not sanity_checks(f):
                # not a good counterfactual
                continue
            q_judge_out = pickle.load(open(f"{f}/judge_out.pkl", "rb"))
            if "chat_messages" not in q_judge_out:
                continue

            orig_trajectory = pickle.load(open(f"{orig_exp_dir}/summary.pkl", "rb"))
            branch_point = exp_args.branch_time_step
            action_counterfactual = exp_args.candidate_action
            # everything before branch point is the original trajectory
            previous_changes = [s for s in orig_trajectory[:branch_point]]
            orig_step = pickle.load(gzip.open(f"{orig_exp_dir}/step_{branch_point}.pkl.gz", "rb"))
            orig_action = orig_step.action
            orig_observation = orig_step.obs["axtree_txt"]
            instruction = pickle.load(gzip.open(f"{orig_exp_dir}/goal_object.pkl.gz", "rb"))[0][
                "text"
            ]

            # decide what went first in the prompt
            orig_reward = q_judge_out["reward"]
            orig_choice = q_judge_out["choice"]
            if orig_reward == "orig":
                if orig_choice == "1":
                    action_1 = orig_action
                    action_2 = action_counterfactual
                else:
                    action_1 = action_counterfactual
                    action_2 = orig_action
            elif orig_reward == "new":
                if orig_choice == "1":
                    action_1 = action_counterfactual
                    action_2 = orig_action
                else:
                    action_1 = orig_action
                    action_2 = action_counterfactual
            else:
                raise ValueError("Invalid reward")

            # make the prompt
            prompt = JudgePrompt(
                instruction,
                orig_observation,
                previous_changes,
                action_1,
                action_2,
                FLAGS_GPT_4o_webarena,
            ).prompt
            chat_messages = [system_prompt, {"role": "user", "content": prompt}] + [
                {"role": "assistant", "content": q_judge_out["chat_messages"][-1]["message"]}
            ]

            full_prompt = "{}\n{}".format(chat_messages[0]["content"], chat_messages[1]["content"])
            n_tokens = count_tokens(full_prompt, args.model_name)
            output_curr = {
                "dataset": f"webarena_{args.exp_name}",
                "id": exp_args.env_args.task_name,
                "messages": chat_messages,
                "output": q_judge_out["chat_messages"][-1]["message"],
                "task_name": exp_args.env_args.task_name,
                "agent_args": dataclass_to_dict(exp_args.agent_args),
                "prompt": full_prompt,
                "n_tokens": n_tokens,
            }
            all_task_names.append(output_curr["task_name"])
            outputs.append(output_curr)
    elif args.experiment_mode == "rejection_tuning":
        # we are going to load all data and keep the ones for which outcome reward is > 4
        all_files = [f for f in glob.glob("{}/*".format(args.trace_dir)) if os.path.isdir(f)]
        rewards = get_all_trajectories(args, all_files)
        outputs = []
        all_task_names = []
        for fname in rewards:
            exp_args = pickle.load(open(os.path.join(fname, "exp_args.pkl"), "rb"))
            task_name = exp_args.env_args.task_name
            chat_messages, all_actions, all_thoughts, reward = rewards[fname]
            for chat_message, action, thought in zip(chat_messages, all_actions, all_thoughts):
                output = "<think>\n{}\n</think>\n\n<action>\n{}\n</action>".format(thought, action)
                full_prompt = "{}\n{}".format(
                    chat_message[0]["content"], chat_message[1]["content"]
                )
                n_tokens = count_tokens(full_prompt, args.model_name)
                output_curr = {
                    "dataset": f"webarena_{args.exp_name}",
                    "id": task_name,
                    "messages": chat_message + [{"role": "assistant", "content": output}],
                    "output": output,
                    "task_name": task_name,
                    "agent_args": dataclass_to_dict(exp_args.agent_args),
                    "prompt": full_prompt,
                    "n_tokens": n_tokens,
                }
                all_task_names.append(task_name)
                outputs.append(output_curr)
    elif args.experiment_mode == "q_judge_reranker":
        FLAGS_GPT_4o_webarena = GenericPromptFlags(
            obs=dp.ObsFlags(
                use_html=False,
                use_ax_tree=True,
                use_focused_element=True,
                use_error_logs=True,
                use_history=True,
                use_past_error_logs=False,
                use_action_history=True,
                use_think_history=False,
                use_diff=False,
                html_type="pruned_html",
                use_screenshot=False,
                use_som=False,
                extract_visible_tag=True,
                extract_clickable_tag=True,
                extract_coords="False",
                filter_visible_elements_only=False,
            ),
            action=dp.ActionFlags(
                multi_actions=False,
                action_set="webarena",
                long_description=False,
                individual_examples=False,
            ),
            use_plan=False,
            use_criticise=False,
            use_thinking=True,
            use_memory=False,
            use_concrete_example=True,
            use_abstract_example=True,
            use_hints=True,
            enable_chat=False,
            max_prompt_tokens=None,
            be_cautious=True,
            extra_instructions=None,
        )
        all_task_names = []
        outputs = []
        all_files = []
        for d1 in os.listdir(args.trace_dir):
            if os.path.isdir(os.path.join(args.trace_dir, d1)):
                all_files.append(os.path.join(args.trace_dir, d1))

        for f in tqdm(all_files):
            task_name = f.split("/")[-1]
            if args.myopic_id != -1:
                all_input_dicts = [
                    fname for fname in glob.glob(f"{f}/MYOPIC_{args.myopic_id}_input_dict_*.pkl")
                ]
                all_rewards = [
                    fname
                    for fname in glob.glob(
                        f"{f}/MYOPIC_{args.myopic_id}_q_function_with_outcomes_reward_*.pkl"
                    )
                ]
            else:
                all_input_dicts = [fname for fname in glob.glob(f"{f}/input_dict_*.pkl")]
                all_rewards = [
                    fname for fname in glob.glob(f"{f}/q_function_with_outcomes_reward_*.pkl")
                ]

            all_input_dicts = sorted(
                all_input_dicts, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            all_rewards = sorted(all_rewards, key=lambda x: int(x.split("_")[-1].split(".")[0]))

            if len(all_input_dicts) == 0 or len(all_rewards) == 0:
                continue

            for input_dict_file, reward_file in zip(all_input_dicts, all_rewards):
                input_info = pickle.load(open(input_dict_file, "rb"))
                reward_info = pickle.load(open(reward_file, "rb"))
                all_actions = input_info["all_candidate_actions"]
                try:
                    chosen_action = all_actions[int(reward_info["choice"]) - 1]
                except Exception as e:
                    print(e)
                    continue

                output = "<think>\n{}\n</think>\n\n<action>\n{}\n</action>".format(
                    reward_info["think"], chosen_action
                )

                system_prompt_critic = SystemPromptLMCriticSelector(
                    num_actions=len(all_actions),
                    flags=FLAGS_GPT_4o_webarena,
                    use_past_transitions=True,
                ).prompt

                main_prompt_critic = LMCriticSelector(
                    obs_history=[input_info["observation"]],
                    action_set=dp.make_action_set(FLAGS_GPT_4o_webarena.action),
                    actions=None,
                    candidate_actions=all_actions,
                    flags=FLAGS_GPT_4o_webarena,
                    past_transitions=input_info["previous_changes"],
                ).prompt
                full_prompt = "{}\n{}".format(system_prompt_critic, main_prompt_critic)
                chat_message = [
                    make_system_message(system_prompt_critic),
                    make_user_message(main_prompt_critic),
                ]
                n_tokens = count_tokens(full_prompt, args.model_name)
                output_curr = {
                    "dataset": f"webarena_{args.exp_name}",
                    "id": task_name,
                    "messages": chat_message + [{"role": "assistant", "content": output}],
                    "output": output,
                    "task_name": task_name,
                    "prompt": full_prompt,
                    "n_tokens": n_tokens,
                }
                all_task_names.append(task_name)
                outputs.append(output_curr)

    else:
        raise ValueError("Invalid experiment mode")

    print("Found {} tasks".format(len(all_task_names)))
    print("Found {} outputs".format(len(outputs)))
    all_task_names = list(set(all_task_names))
    # split into train / val / test with a 80/10/10 split
    train_size = int(0.8 * len(all_task_names))
    val_size = int(0.1 * len(all_task_names))
    test_size = len(all_task_names) - train_size - val_size
    train_tasks = set(all_task_names[:train_size])
    val_tasks = set(all_task_names[train_size : train_size + val_size])
    test_tasks = set(all_task_names[train_size + val_size :])

    train_outputs = [o for o in outputs if o["task_name"] in train_tasks]
    val_outputs = [o for o in outputs if o["task_name"] in val_tasks]
    test_outputs = [o for o in outputs if o["task_name"] in test_tasks]

    os.makedirs(f"DATA_DUMP/{args.exp_name}", exist_ok=True)
    with open(f"DATA_DUMP/{args.exp_name}/train.json", "w") as f:
        json.dump(train_outputs, f)
    with open(f"DATA_DUMP/{args.exp_name}/val.json", "w") as f:
        json.dump(val_outputs, f)
    with open(f"DATA_DUMP/{args.exp_name}/test.json", "w") as f:
        json.dump(test_outputs, f)

    # take the val and test tasks and write them to eval_tasks.txt
    with open(f"DATA_DUMP/{args.exp_name}/eval_tasks.txt", "w") as f:
        for task in val_tasks:
            f.write(task + "\n")
        for task in test_tasks:
            f.write(task + "\n")

    # also write the full data
    with open(f"DATA_DUMP/{args.exp_name}/full.json", "w") as f:
        json.dump(outputs, f)
