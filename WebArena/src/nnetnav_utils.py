"""
    Main code for NNetscape Navigator.
"""

import glob
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict as ddict

from tqdm import tqdm
import openai

import gymnasium as gym
from bs4 import BeautifulSoup

from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from browsergym.core.action.highlevel import HighLevelActionSet

from agent import PromptAgent
from agent.prompts import *
from agent import PromptAgent, InstructionGenerator

from browser_env import ActionTypes, Action, create_stop_action
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    get_action_description,
    get_action_description_bgym,
)
from browser_env.auto_login import get_site_comb_from_filepath
from distributed import LocalCluster, Client


def get_changelog_model(args, only_path=False):
    if args.environment_type in ["webarena", "openweb"]:
        state_changelog_prompt = "src/agent/prompts/jsons/p_state_changelog.json"
    elif args.environment_type == "miniwob":
        state_changelog_prompt = (
            "src/agent/prompts/jsons_miniwob/p_state_changelog.json"
        )
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    llm_config = lm_config.construct_llm_config(args)
    with open(state_changelog_prompt, "r") as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        state_changelog_prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    if only_path:
        return state_changelog_prompt
    else:
        changelog_model = InstructionGenerator(
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )

        return changelog_model


def get_trajectory_relabeler(args, only_path=False):
    if args.environment_type in ["webarena", "openweb"]:
        relabeling_prompt = "src/agent/prompts/jsons/p_instruction_relabel.json"
    elif args.environment_type == "miniwob":
        relabeling_prompt = "src/agent/prompts/jsons_miniwob/p_instruction_relabel.json"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    llm_config = lm_config.construct_llm_config(args)
    tokenizer = Tokenizer(args.provider, args.model)
    with open(relabeling_prompt, "r") as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    prompt_constructor = eval(constructor_type)(
        relabeling_prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    if only_path:
        return relabeling_prompt
    else:
        relabeling_model = InstructionGenerator(
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )
        return relabeling_model


def get_reward_model_full(args, only_path=False):
    llm_config = lm_config.construct_llm_config(args)
    if args.environment_type in ["webarena", "openweb"]:
        reward_prompt = "src/agent/prompts/jsons/p_reward_detailed.json"
    elif args.environment_type == "miniwob":
        reward_prompt = "src/agent/prompts/jsons_miniwob/p_reward_lenient.json"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    if only_path:
        return reward_prompt
    with open(reward_prompt) as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        reward_prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    reward_model = InstructionGenerator(
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
    )
    return reward_model


def get_reward_model(args, only_path=False):
    llm_config = lm_config.construct_llm_config(args)
    if args.environment_type == "webarena":
        reward_prompt = "src/agent/prompts/jsons/p_reward_lenient.json"
    elif args.environment_type == "openweb":
        reward_prompt = "src/agent/prompts/jsons_openweb/p_reward_lenient.json"
    elif args.environment_type == "miniwob":
        reward_prompt = "src/agent/prompts/jsons_miniwob/p_reward_lenient.json"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    if only_path:
        return reward_prompt
    with open(reward_prompt) as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        reward_prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    reward_model = InstructionGenerator(
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
    )
    return reward_model


def get_exploration_policy(args, only_path=False, use_llama=False):
    llm_config = lm_config.construct_llm_config(args)
    if args.environment_type == "webarena":
        json_dir = "src/agent/prompts/jsons_bgym"
    elif args.environment_type == "openweb":
        json_dir = "src/agent/prompts/jsons_openweb"
    elif args.environment_type == "miniwob":
        json_dir = "src/agent/prompts/jsons_miniwob"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")

    if use_llama:
        if args.use_personas:
            zero_shot_policy_prompt = (
                "{}/p_cot_exploration_llama_with_history_persona.json".format(json_dir)
            )
        else:
            zero_shot_policy_prompt = (
                "{}/p_cot_exploration_llama_with_history.json".format(json_dir)
            )

    else:
        if args.use_personas:
            zero_shot_policy_prompt = (
                "{}/p_cot_exploration_with_history_persona.json".format(json_dir)
            )
        else:
            zero_shot_policy_prompt = "{}/p_cot_exploration_with_history.json".format(
                json_dir
            )
    with open(zero_shot_policy_prompt) as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        zero_shot_policy_prompt, lm_config=llm_config, tokenizer=tokenizer
    )

    if only_path:
        return zero_shot_policy_prompt
    else:
        base_agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )

        return base_agent


def get_url(url_name):
    ## get the environment variable corresponding to url_name
    return os.getenv(url_name)


def make_dask_client(n_worker):
    """Create a Dask client with a LocalCluster backend.

    Thanks: https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/experiments/graph_execution_dask.py
    Args:
        n_worker: int
            Number of workers to create.

    Returns:
        A Dask client object.
    """
    cluster = LocalCluster(
        n_workers=n_worker,
        processes=True,
        threads_per_worker=1,
    )

    return Client(cluster)


def setup_config(env, config_file):
    with open(config_file) as f:
        _c = json.load(f)

    # automatically login
    if _c["storage_state"]:
        cookie_file_name = os.path.basename(_c["storage_state"])
        comb = get_site_comb_from_filepath(cookie_file_name)
        temp_dir = tempfile.mkdtemp()
        # subprocess to renew the cookie
        subprocess.run(
            [
                "python",
                "src/browser_env/auto_login.py",
                "--auth_folder",
                temp_dir,
                "--site_list",
                *comb,
            ]
        )
        _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
        assert os.path.exists(_c["storage_state"])
        # update the config file
        config_file_out = f"{temp_dir}/{os.path.basename(config_file)}"
        with open(config_file_out, "w") as f:
            json.dump(_c, f)
    else:
        config_file_out = config_file
    return config_file_out


def convert_to_description(changelogs):
    """
    returns a natural language description of all the changes to states
    """
    i = 0
    descriptions = []
    for log in changelogs:
        cstr = "Step: " + str(i) + "\n"
        cstr += log
        descriptions.append(cstr)
        i += 1
    return "\n\n".join(descriptions)


def log_message(logger, message):
    if logger:
        logger.info(message)
    return


def early_stop(trajectory, max_steps, thresholds):
    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [action["action_type"] == ActionTypes.NONE for action in last_k_actions]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all([is_equivalent(action, last_action) for action in last_k_actions]):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
            return True, f"Same typing action for {k} times"

    return False, ""


def convert_html_to_jsons(result_folder: str, config_json: str, logger=None) -> None:
    all_data = {}
    template_to_id = ddict(lambda: len(template_to_id))

    with open(config_json, "r") as f:
        data_configs = json.load(f)
        data_configs = {int(item["task_id"]): item for item in data_configs}
        for k, v in data_configs.items():
            if "require_login" in v:
                v.pop("require_login")
            if "storage_state" in v:
                v.pop("storage_state")
            if "start_url" in v:
                v.pop("start_url")
            if "geolocation" in v:
                v.pop("geolocation")
            if "require_reset" in v:
                v.pop("require_reset")
            if "intent_template_id" in v:
                v.pop("intent_template_id")
                v["intent_template_id"] = template_to_id[v["intent_template"]]
            if "eval" in v:
                v["eval_types"] = v["eval"].pop("eval_types")
                if v["eval"]["reference_answers"]:
                    v["reference_answers"] = v["eval"].pop("reference_answers")
                if v["eval"]["reference_url"]:
                    v["reference_url"] = v["eval"].pop("reference_url")
                v.pop("eval")
                if v.get("reference_answers", {}).get("exact_match", "") == "N/A":
                    v["achievable"] = False
                else:
                    v["achievable"] = True

    files = list(glob.glob(f"{result_folder}/render_*.html"))
    files = [x for x in files if os.path.exists(x)]
    log_message(logger, f"Total number of files: {len(files)}")

    for render_file in tqdm(files):
        task_id = int(render_file.split("_")[-1].split(".")[0])
        with open(render_file, "r") as f:
            try:
                content = f.read()
                soup = BeautifulSoup(content, "html.parser")
                observations = [
                    obv.find("pre").text
                    for obv in soup.find_all("div", {"class": "state_obv"})
                ]
                base64_images = [
                    img["src"].split(",")[1] for img in soup.find_all("img")
                ]
                image_observations = []
                # save image to file and change the value to be path
                image_folder = f"images/{os.path.basename(result_folder)}"
                os.makedirs(image_folder, exist_ok=True)
                for i, image in enumerate(base64_images):
                    # smurty: hack to make this script fast. comment it out if you want image data
                    # image_data = base64.b64decode(image)
                    # filename = f"{image_folder}/image_{task_id}_{i}.png"
                    filename = "[not stored]"
                    # with open(filename, "wb") as f:  # type: ignore[assignment]
                    #    f.write(image_data)  # type: ignore[arg-type]
                    image_observations.append(filename)
                urls = [url.get_text() for url in soup.find_all("h3", {"class": "url"})]
                actions = [
                    action.get_text()
                    for action in soup.find_all(
                        "div", {"class": "raw_parsed_prediction"}
                    )
                ]
                parsed_actions = [
                    action.get_text()
                    for action in soup.find_all("div", {"class": "parsed_action"})
                ]
                # fill action with parsed action if action is empty
                for i in range(len(actions)):
                    if actions[i] == "":
                        actions[i] = parsed_actions[i]

                messages = []
                for o, u, a, image in zip(
                    observations, urls, actions, image_observations
                ):
                    messages.append(
                        {
                            "user": f"{u}\n\nobservation:\n{o}",
                            "image": image,
                        }
                    )
                    messages.append({"assistant": a})

                all_data[f"example_{task_id}"] = {
                    **data_configs[task_id],
                    "messages": messages,
                    "success": False,
                }

            except Exception as e:
                log_message(logger, f"Error in {render_file}")

    with open(f"{result_folder}/json_dump.json", "w+") as f:
        json.dump(all_data, f, indent=4)
    return all_data


def get_dom(observation):
    """
    Get the DOM from the observation.
    """
    return flatten_dom_to_str(
        observation["dom_object"],
        filter_visible_only=True,
        extra_properties=observation["extra_element_properties"],
        with_clickable=True,
    )


def get_axtree(observation):
    """
    Get the accessibility tree from the observation.
    """
    return flatten_axtree_to_str(
        observation["axtree_object"],
        filter_visible_only=True,
        extra_properties=observation["extra_element_properties"],
    )


class TrajectoryLabeler:
    """
    Label a trajectory with a language model.
    """

    def __init__(self, base_model, changelog_model):
        self.labeling_model = base_model
        self.changelog_model = changelog_model

    def _post_process(self, instruction):
        last_sent = instruction.split("\n")[-1]
        keyword = "Instruction"
        if keyword not in last_sent:
            return "n/a"
        else:
            last_sent = re.sub(keyword, "", last_sent, count=1).strip()
            return last_sent

    def get_changelogs(self, trajectory_dict, logger=None):
        all_responses = {}
        for key in tqdm(trajectory_dict, desc="Getting state changelogs"):
            state_action_state_tuples = []
            state = None
            action = None
            ex = trajectory_dict[key]["messages"]
            try:
                for e in ex:
                    if "user" in e:
                        if state is not None and action is not None:
                            state_action_state_tuples.append(
                                {
                                    "init_observation": state,
                                    "action": action,
                                    "final_observation": e["user"],
                                }
                            )
                        state = "observation: ".join(
                            e["user"].split("observation: ")[1:]
                        )
                    if "assistant" in e:
                        action = e["assistant"]
                responses = []

                for i in range(len(state_action_state_tuples)):
                    response = self.changelog_model.generate(
                        state_action_state_tuples[i], meta_data=None
                    )
                    responses.append(response)
                all_responses[key] = responses
            except openai.OpenAIError as e:
                log_message(logger, f"[OpenAI Error] {repr(e)}")
                continue
        return all_responses

    def label_all_endpoints(self, trajectory_list, all_changelogs, curr_dir, logger):
        if not os.path.exists(
            "{}/relabeled_instructions_all_endpoints.json".format(curr_dir)
        ):
            all_labeled_instructions = {}
            all_changelogs_increments = {}
            for key in tqdm(all_changelogs, desc="labeling all trajectory prefixes"):
                responses = all_changelogs[key]
                if len(responses) < 4:
                    try:
                        i = len(responses)
                        instruction = self.labeling_model.generate(
                            {"trajectory": convert_to_description(responses[:i])},
                            meta_data=None,
                        )
                        all_labeled_instructions["{}:{}".format(key, i)] = {
                            "message": instruction,
                            "instruction": self._post_process(instruction),
                        }
                        print(instruction)
                        all_changelogs_increments["{}:{}".format(key, i)] = responses[
                            :i
                        ]
                    except openai.OpenAIError as e:
                        log_message(logger, f"[OpenAI Error] {repr(e)}")
                        continue
                else:
                    for i in range(4, len(responses), 4):
                        try:
                            instruction = self.labeling_model.generate(
                                {"trajectory": convert_to_description(responses[:i])},
                                meta_data=None,
                            )
                            all_labeled_instructions["{}:{}".format(key, i)] = {
                                "message": instruction,
                                "instruction": self._post_process(instruction),
                            }
                            print(instruction)
                            all_changelogs_increments["{}:{}".format(key, i)] = (
                                responses[:i]
                            )
                        except openai.OpenAIError as e:
                            log_message(logger, f"[OpenAI Error] {repr(e)}")
                            continue
            with open(
                curr_dir + "/relabeled_instructions_all_endpoints.json", "w"
            ) as f:
                json.dump(all_labeled_instructions, f, indent=4)
            with open(curr_dir + "/state_changelogs_all_endpoints.json", "w") as f:
                json.dump(all_changelogs_increments, f, indent=4)
        else:
            all_changelogs_increments = json.load(
                open(f"{curr_dir}/state_changelogs_all_endpoints.json")
            )
            all_labeled_instructions = json.load(
                open(f"{curr_dir}/relabeled_instructions_all_endpoints.json")
            )
        return all_labeled_instructions, all_changelogs_increments

    def __call__(
        self, trajectory_list, curr_dir, itr_num, logger=None, all_endpoints=False
    ):
        if not os.path.exists("{}/state_changelogs.json".format(curr_dir)):
            all_changelogs = self.get_changelogs(trajectory_list, logger=logger)
            with open(curr_dir + "/state_changelogs.json", "w") as f:
                json.dump(all_changelogs, f, indent=4)
        else:
            all_changelogs = json.load(open(f"{curr_dir}/state_changelogs.json"))

        return self.label_all_endpoints(
            trajectory_list, all_changelogs, curr_dir, logger
        )


class LanguagePruning:
    """
    Prune a trajectory based on a language model.
    """

    def __init__(self, reward_model, trajectory_labeler, best_reward=5):
        self.reward_model = reward_model
        self.trajectory_labeler = trajectory_labeler
        self.best_reward = best_reward

    def _post_process_reward_str(self, reward_string):
        last_sent = reward_string.split("\n")[-1]
        keyword = "Reward"
        if keyword not in last_sent:
            return 0
        else:
            last_sent = re.sub(keyword, "", last_sent, count=1).strip()
            try:
                reward = float(last_sent)
                return reward
            except ValueError:
                return 0

    def _post_process_instruction_str(self, instruction):
        last_sent = instruction.split("\n")[-1]
        keyword = "Instruction"
        if keyword not in last_sent:
            return "n/a"
        else:
            last_sent = re.sub(keyword, "", last_sent, count=1).strip()
            return last_sent

    def __call__(self, trajectory, logger=None):
        trajectory_description = convert_to_description(trajectory)
        instruction_str = self.trajectory_labeler.generate(
            {"trajectory": trajectory_description}, meta_data=None
        )
        labeled_instruction = self._post_process_instruction_str(instruction_str)
        reward_str = self.reward_model.generate(
            {
                "trajectory": trajectory_description,
                "instruction": labeled_instruction,
            },
            meta_data=None,
        )
        reward = self._post_process_reward_str(reward_str)
        log_message(logger, f"Instruction: {labeled_instruction}")
        log_message(logger, f"{reward_str}")
        return reward < (self.best_reward - 1), labeled_instruction


class RewardModelBatched:
    def __init__(self, reward_model):
        self.reward_model = reward_model

    def get_usage(self):
        return self.reward_model.total_usage

    def reset_usage(self):
        self.reward_model.total_usage = []

    def _post_process(self, reward_string):
        last_sent = reward_string.split("\n")[-1]
        keyword = "Reward"
        if keyword not in last_sent:
            return 0
        else:
            last_sent = re.sub(keyword, "", last_sent, count=1).strip()
            try:
                reward = float(last_sent)
                return reward
            except ValueError:
                return 0

    def __call__(
        self,
        state_changelogs,
        relabeled_instructions,
        curr_dir,
        logger=None,
        all_endpoints=False,
    ):
        if all_endpoints:
            reward_file = f"{curr_dir}/rewards_all_endpoints.json"
        else:
            reward_file = f"{curr_dir}/rewards.json"

        if os.path.exists(reward_file):
            all_rewards = json.load(open(reward_file, "r"))
            return all_rewards
        all_rewards = {}
        for key in state_changelogs:
            try:
                responses = convert_to_description(state_changelogs[key])
                intent = relabeled_instructions[key]["instruction"]
                log_message(logger, f"Generating reward for Instruction: {intent}\n")
                reward = self.reward_model.generate(
                    {"trajectory": responses, "instruction": intent}, meta_data=None
                )
                all_rewards[key] = {
                    "message": reward,
                    "reward": self._post_process(reward),
                }
                if logger:
                    logger.info(reward)
            except openai.OpenAIError as e:
                log_message(logger, f"[OpenAI Error] {repr(e)}")
                continue
            except Exception as e:
                log_message(logger, f"[Exception] {repr(e)}")
                continue
        with open(reward_file, "w") as f:
            json.dump(all_rewards, f, indent=4)

        return all_rewards


class NNetscapeNavigator:
    """
    Unroll an exploration policy to discover a long horizon plan.
    We do this by running the exploration policy for a maximum depth.
    Prune at various stages if trajectory so far does not correspond to a meaningful sub-plan.
    """

    def __init__(
        self,
        exploration_policy,
        prune_function,
        max_depth=40,
        prune_at=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
        action_set_tag="id_accessibility_tree",
    ):
        """
        Args:
            prune_function: function that takes in a trajectory and returns a boolean indicating whether to prune
            max_depth: maximum depth to run the exploration policy
            prune_at: list of depths at which to prune
        """
        self.prune_function = prune_function
        self.max_depth = max_depth
        self.prune_at = prune_at
        self.exploration_policy = exploration_policy
        self.action_set_tag = action_set_tag
        self.early_stop_thresholds = {
            "parsing_failure": 3,
            "repeating_action": 3,
        }

    def __call__(
        self,
        env,
        config_file,
        state_changelogger=None,
        persona_str="",
        render_helper=None,
        logger=None,
    ):
        """
        Args:
            env: environment to run the exploration policy in
            config_file: configuration file for the environment
            state_changelogger: a prompted LM that summarizes changes to environment state (to be used as extra features for the exploration policy)
            persona_str: persona string to be used by the exploration policy to simulate a candidate user
            render_helper: renderer to visualize the environment state: outputs an html file
        """
        config_json = json.load(open(config_file))
        config_file_dir = os.path.dirname(config_file)
        intent = config_json["intent"]
        env_type = config_json.get("env_type", "webarena")
        if env_type == "webarena":
            env_type = "webarena"
            config_file_tmp = setup_config(env, config_file)
            obs, info = env.reset(options={"config_file": config_file_tmp})
        elif env_type == "open_ended":
            assert (
                "start_url" in config_json
            ), "start_url is required for open_ended task"
            start_url = config_json["start_url"]
            env_type = config_json["env_type"]
            action_set = HighLevelActionSet("webarena")
            env = gym.make(
                "browsergym/openended",
                task_kwargs={"start_url": start_url},
                wait_for_user_message=False,
                action_mapping=action_set.to_python_code,
            )
            obs, info = env.reset()
            observation_constructor = get_axtree
            obs["text"] = observation_constructor(obs)
            info["observation_metadata"] = {
                "text": {"obs_nodes_info": obs["axtree_object"]}
            }
            config_file_tmp = config_file
        else:
            env = gym.make(
                "browsergym/{}".format(config_json["task"]),
            )
            obs, info = env.reset()
            # need to make workarena observation compatible with rest of the code
            # obs["text"] = flatten_axtree_to_str(obs["axtree_object"])
            if env_type == "miniwob":
                observation_constructor = get_dom
            else:
                observation_constructor = get_axtree
            obs["text"] = observation_constructor(obs)
            info["observation_metadata"] = {
                "text": {"obs_nodes_info": obs["axtree_object"]}
            }
            config_file_tmp = config_file

        self.exploration_policy.reset(config_file_tmp)
        trajectory = []
        state_info = {"observation": obs, "info": info}
        state_info["history"] = "None"

        trajectory.append(state_info)
        meta_data = {
            "action_history": ["None"],
            "person_description": persona_str,
            "env_type": env_type,
        }

        history_accum = []
        curr_depth = 0
        labeled_instruction_dict = {}
        while curr_depth <= self.max_depth:
            # check if early stop
            if env_type == "webarena":
                # for webarena there is a special early stop condition
                early_stop_flag, stop_info = early_stop(
                    trajectory, self.max_depth, self.early_stop_thresholds
                )
            else:
                early_stop_flag = len(trajectory) >= 2 * self.max_depth
                stop_info = f"Reach max depth {self.max_depth}"

            # now check if we can prune
            if curr_depth in self.prune_at:
                to_prune, labeled_instruction = self.prune_function(
                    history_accum, logger
                )

                if to_prune:
                    early_stop_flag = True
                    stop_info = f"Prune at depth {curr_depth}"
                else:
                    labeled_instruction_dict[curr_depth] = labeled_instruction

            if early_stop_flag:
                action = create_stop_action(f"Early stop: {stop_info}")
            else:
                try:
                    action = self.exploration_policy.next_action(
                        trajectory, intent, meta_data=meta_data
                    )
                except Exception as e:
                    # get the error message
                    action = create_stop_action(f"ERROR: {str(e)}")

            trajectory.append(action)

            init_observation = state_info["observation"]["text"]
            if env_type == "webarena":
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=self.action_set_tag,
                    prompt_constructor=(
                        self.exploration_policy.prompt_constructor
                        if isinstance(self.exploration_policy, PromptAgent)
                        else None
                    ),
                )
            elif env_type == "open_ended":
                action_str = None  # need to do something here
                # get all bids
                all_bids = [
                    o
                    for o in state_info["observation"]["axtree_object"]["nodes"]
                    if "browsergym_id" in o
                ]
                bid_dict = {}
                for o in all_bids:
                    if "name" in o:
                        bid_dict[o["browsergym_id"]] = o["name"]["value"]
                    else:
                        bid_dict[o["browsergym_id"]] = ""
                action_str = get_action_description_bgym(
                    action,
                    bid_dict,
                    action_set_tag=self.action_set_tag,
                    prompt_constructor=(
                        self.exploration_policy.prompt_constructor
                        if isinstance(self.exploration_policy, PromptAgent)
                        else None
                    ),
                )
                # a bit painful but we will need to convert actions into bgym format:
            else:
                action_str = None
            if action["action_type"] == ActionTypes.STOP:
                action["parsed_response"] = "stop"
            if render_helper is not None:
                render_helper.render(
                    action, state_info, meta_data, render_screenshot=True
                )
            if action["action_type"] == ActionTypes.STOP:
                break
            try:
                if env_type == "webarena":
                    obs, _, terminated, _, info = env.step(action)
                elif env_type == "open_ended":
                    obs, _, terminated, _, info = env.step(action["bgym_action"])
                else:
                    obs, _, terminated, _, info = env.step(action["parsed_response"])
            except Exception as e:
                trajectory.append(create_stop_action(f"ERROR: {str(e)}"))
                break

            if env_type != "webarena":
                if obs["last_action_error"]:
                    action_str = f"ERROR: {obs['last_action_error']}"
                else:
                    action_str = action["parsed_response"]
            meta_data["action_history"].append(action_str)
            log_message(logger, action_str)

            if env_type != "webarena":
                obs["text"] = observation_constructor(obs)
            state_info = {"observation": obs, "info": info}
            final_observation = state_info["observation"]["text"]
            if state_changelogger is not None:
                state_changelogger_inp = {
                    "init_observation": init_observation,
                    "action": action_str,
                    "final_observation": final_observation,
                }
                curr_changelog = state_changelogger.generate(
                    state_changelogger_inp, meta_data=None
                )
                log_message(logger, curr_changelog)
                history_accum.append(curr_changelog)
                state_info["history"] = convert_to_description(history_accum)
            trajectory.append(state_info)
            if terminated:
                # add a action place holder
                trajectory.append(create_stop_action(""))
                break
            curr_depth += 1
        return trajectory, history_accum, labeled_instruction_dict
