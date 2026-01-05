"""
    postprocess trajectories to add retroactive reasoning and stop action
"""

import argparse
import os
import json
import logging
import random
import time
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup

from agent import InstructionGenerator
from dataclasses import dataclass

from agent.prompts import *
import browsergym.miniwob  # register miniwob tasks as gym environments
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from nnetnav_utils import make_dask_client
from agentlab.llm.chat_api import (
    SelfHostedModelArgs,
    OpenRouterModelArgs,
    TogetherAIModelArgs,
)
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags

from agent import LMModule

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

default_oss_llms_args = {
    "n_retry_server": 4,
    "temperature": 0.01,
}


def parse_action(output):
    action_splitter = "```"
    pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def config():
    parser = argparse.ArgumentParser(
        description="Run Postprocessing for nnetnav trajectories"
    )
    parser.add_argument("--data_dir", type=str, default="")

    # agent config
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument("--agent_type", type=str, default="prompt")
    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    parser.add_argument(
        "--environment_type",
        type=str,
        default="webarena",
        choices=["webarena", "miniwob"],
    )
    parser.add_argument(
        "--script_mode",
        type=str,
        default="all",
        choices=["add_stop_action", "retroactive_reasoning", "all"],
    )
    parser.add_argument("--use_openrouter", action="store_true")
    parser.add_argument("--use_together_ai", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()
    return args


def get_retroactive_reasoner(args, chat_model, obs_flags):
    if args.environment_type == "webarena":
        prompt_folder = "src/agent/prompts/jsons"
    elif args.environment_type == "openweb":
        prompt_folder = "src/agent/prompts/jsons_openweb"
    elif args.environment_type == "miniwob":
        prompt_folder = "src/agent/prompts/jsons_miniwob"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")

    prompt = f"{prompt_folder}/p_forward_reasoning.json"

    llm_config = lm_config.construct_llm_config(args)
    with open(prompt, "r") as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    return LMModule(chat_model, obs_flags, prompt_constructor)


# def get_retroactive_reasoner(args, chat_model, obs_flags):
#     if args.environment_type == "webarena":
#         # use the openweb prompt for webarena
#         prompt_folder = "src/agent/prompts/jsons_openweb"
#     elif args.environment_type == "miniwob":
#         prompt_folder = "src/agent/prompts/jsons_miniwob"
#     else:
#         raise ValueError(f"Unknown environment type: {args.environment_type}")

#     prompt = f"{prompt_folder}/p_retroactive_reasoning.json"

#     llm_config = lm_config.construct_llm_config(args)
#     with open(prompt, "r") as f:
#         constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
#     tokenizer = Tokenizer(args.provider, args.model)
#     prompt_constructor = eval(constructor_type)(
#         prompt, lm_config=llm_config, tokenizer=tokenizer
#     )
#     return LMModule(chat_model, obs_flags, prompt_constructor)


def get_add_stop_action(args, chat_model, obs_flags):
    if args.environment_type == "webarena":
        # use the openweb prompt for webarena
        prompt = "src/agent/prompts/jsons_openweb/p_add_stop_action.json"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    llm_config = lm_config.construct_llm_config(args)
    with open(prompt, "r") as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    return LMModule(chat_model, obs_flags, prompt_constructor)


@dataclass
class ReasoningFunc:
    all_actions: list
    all_previous_actions: list
    all_observation: list
    instruction: str
    data_idx: int

    agent: LMModule

    def run(self):
        reasoner_agent = self.agent
        actions = self.all_actions
        states = self.all_observation
        instruction = self.instruction
        retroactive_reasoning = []
        for idx, state in enumerate(states):
            orig_action = actions[idx]
            previous_action_list = self.all_previous_actions[: idx + 1]
            previous_actions = "\n".join(
                [f"{i+1}. {a}" for i, a in enumerate(previous_action_list)]
            )
            action = orig_action
            output = reasoner_agent(
                {
                    # "action": action,
                    "previous_actions": previous_actions,
                    "state": state,
                    "instruction": instruction,
                }
            )
            if "raw_prediction" in output:
                retroactive_reasoning.append(output["raw_prediction"])
            else:
                retroactive_reasoning.append("")
        return {"data_idx": self.data_idx, "reasoning": retroactive_reasoning}


@dataclass
class StopActionFunc:
    last_observation: str
    instruction: str
    data_idx: int
    agent: LMModule

    def run(self):
        data_idx = self.data_idx
        stop_action_agent = self.agent
        last_observation = self.last_observation
        instruction = self.instruction

        stop_output = stop_action_agent(
            {"state": last_observation, "instruction": instruction}
        )
        return {"data_idx": data_idx, "stop_action": stop_output["answer"]}


def run_reasoning(args, chat_model, obs_flags):
    reasoner_agent = get_retroactive_reasoner(args, chat_model, obs_flags)

    orig_data = json.load(open(os.path.join(args.data_dir, "filtered_parsed.json")))

    all_reasoning_funcs = []
    all_data = []
    for i, data in tqdm(enumerate(orig_data), total=len(orig_data)):
        actions = data["parsed_actions"]
        states = [m["user"] for m in data["messages"] if "user" in m]
        instruction = data["intent"]
        with open(
            "{}/render_states/render_{}.html".format(args.data_dir, data["task_id"]),
            "r",
        ) as f:
            render_state = f.read()
            soup = BeautifulSoup(render_state, "html.parser")
            previous_actions = [
                obv.get_text() for obv in soup.find_all("div", {"class": "prev_action"})
            ]

        reasoning_func = ReasoningFunc(
            actions, previous_actions, states, instruction, i, reasoner_agent
        )
        all_data.append(data)
        all_reasoning_funcs.append(reasoning_func)

    if args.n_jobs == 1:
        all_reasoning_outputs = []
        for reasoning_func in all_reasoning_funcs:
            all_reasoning_outputs.append(reasoning_func.run())
    else:
        with ProgressBar():
            delayed_results = []
            for reasoning_func in all_reasoning_funcs:
                delayed_results.append(delayed(reasoning_func.run)())
            all_reasoning_outputs = compute(*delayed_results)

    for rout in all_reasoning_outputs:
        # might get jumbled up by dask so we need to index correctly
        all_data[rout["data_idx"]]["retroactive_reasoning"] = rout["reasoning"]

    out_data_path = os.path.join(
        args.data_dir, "filtered_parsed_with_retroactive_reasoning.json"
    )
    with open(out_data_path, "w") as f:
        json.dump(all_data, f)


def run_stop_action(args, chat_model, obs_flags):
    """
    WebArena requires models to output a stop action.
    We postprocess nnetnav trajectories to add stop action.
    """
    stop_action_agent = get_add_stop_action(args, chat_model, obs_flags)
    orig_data = json.load(
        open(
            os.path.join(
                args.data_dir, "filtered_parsed_with_retroactive_reasoning.json"
            )
        )
    )
    all_outputs = []
    all_stop_action_funcs = []
    all_data = []
    for i, data in tqdm(enumerate(orig_data), total=len(orig_data)):
        states = [m["user"] for m in data["messages"] if "user" in m]
        last_observation = states[-1]
        instruction = data["intent"]
        stop_action_func = StopActionFunc(
            last_observation, instruction, i, stop_action_agent
        )
        all_stop_action_funcs.append(stop_action_func)
        all_data.append(data)
    if args.n_jobs == 1:
        all_stop_outputs = []
        for stop_action_func in all_stop_action_funcs:
            all_outputs.append(stop_action_func.run())
    else:
        with ProgressBar():
            delayed_results = []
            for stop_action_func in all_stop_action_funcs:
                delayed_results.append(delayed(stop_action_func.run)())
            all_stop_outputs = compute(*delayed_results)

    for stop_output in all_stop_outputs:
        # might get jumbled up by dask so we need to index correctly
        all_data[stop_output["data_idx"]]["stop_action"] = stop_output["stop_action"]

    out_data_path = os.path.join(
        args.data_dir, "filtered_parsed_with_retroactive_stop_action.json"
    )
    with open(out_data_path, "w") as f:
        json.dump(all_data, f)


if __name__ == "__main__":
    args = config()
    if args.use_openrouter:
        chat_model = OpenRouterModelArgs(
            model_name=args.model,
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            temperature=0.01,
        ).make_model()
    elif args.use_together_ai:
        # use a reasonably high temperature to get multiple trajectories
        # TogetherAI uses Turbo postfix for quantized models
        chat_model = TogetherAIModelArgs(
            model_name=f"{args.model}-Turbo",
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            temperature=0.01,
        ).make_model()
    else:
        chat_model = SelfHostedModelArgs(
            model_name=args.model,
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            backend="vllm",
            **default_oss_llms_args,
        ).make_model()
    # not that these flags don't really matter because we are directly using the axtree_txt objects...
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

    if args.script_mode == "all":
        run_reasoning(args, chat_model, FLAGS_GPT_4o_webarena)
        run_stop_action(args, chat_model, FLAGS_GPT_4o_webarena)
    elif args.script_mode == "retroactive_reasoning":
        run_reasoning(args, chat_model, FLAGS_GPT_4o_webarena)
    else:
        assert os.path.exists(
            "{}/filtered_parsed_with_retroactive_reasoning.json".format(args.data_dir)
        ), "Retroactive reasoning must be run before stop action"
        run_stop_action(args, chat_model, FLAGS_GPT_4o_webarena)
