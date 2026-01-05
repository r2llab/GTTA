import json
import re
from pathlib import Path
from typing import Any, TypedDict

from browser_env import Action, ActionParsingError, Trajectory
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput
import os
from agentlab.llm.llm_utils import ParseError


def extract_tags(text, keys):
    """Extract the content within tags for a list of keys.

    All text and keys will be converted to lowercase before matching.

    Returns:
        dict: A dictionary mapping each key to a list of subset in `text` that match the key.
    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        # make a pattern that matches the text between [key] and [key]
        pattern = rf"\[{key}\](.*?)\[{key}\]"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def parse_tags_raise(text, keys=(), optional_keys=(), merge_multiple=False):
    """A version of parse_tags that raises an exception if the parsing is not successful."""
    content_dict, valid, retry_message = parse_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def parse_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """Satisfy the parse api, extracts 1 match per key and validates that all keys are present

    Args:
        text (str): The input string containing the tags.
        keys (list[str]): The tags to extract the content from.
        optional_keys (list[str]): The tags to extract the content from, but are optional.
        merge_multiple (bool): Whether to merge multiple instances of the same key.

    Returns:
        dict: A dictionary mapping each key to a subset of `text` that match the key.
        bool: Whether the parsing was successful.
        str: A message to be displayed to the agent if the parsing was not successful.

    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if not key in content_dict:
            if not key in optional_keys:
                retry_messages.append(f"Missing the key <{key}> in the answer.")
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. You should have only one of them."
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message


def create_bgym_action(action_str: str) -> str:
    """Convert from the webarena action space to the bgym action space. Lame, but needs to be done.
    The bgym action space is defined as follows:
    noop(wait_ms: float = 1000)
    report_infeasible(reason: str)
    send_msg_to_user(text: str)
    click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
    hover(bid: str)
    fill(bid: str, value: str)
    keyboard_press(key: str)
    scroll(delta_x: float, delta_y: float)
    tab_focus(index: int)
    new_tab()
    tab_close()
    go_back()
    go_forward()
    goto(url: str)
    """
    action_str = action_str.strip()
    action = (
        action_str.split("[")[0].strip()
        if "[" in action_str
        else action_str.split()[0].strip()
    )
    match action:
        case "click":
            match = re.search(r"click ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid click action {action_str}")
            element_id = match.group(1)
            return "click('{}')".format(element_id)
        case "hover":
            match = re.search(r"hover ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid hover action {action_str}")
            element_id = match.group(1)
            return "hover('{}')".format(element_id)
        case "type":
            # add default enter flag
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"

            match = re.search(r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid type action {action_str}")
            element_id, text, enter_flag = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if enter_flag == "1":
                return "fill('{}', '{}')\nkeyboard_press('Enter')".format(
                    element_id, text
                )
            else:
                return "fill('{}', '{}')".format(element_id, text)

            # deal with enter flag later
        case "press":
            match = re.search(r"press ?\[(.+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid press action {action_str}")
            key_comb = match.group(1)
            return "keyboard_press('{}')".format(key_comb)
        case "scroll":
            # up or down
            match = re.search(r"scroll ?\[?(up|down)\]?", action_str)
            if not match:
                raise ParseError(f"Invalid scroll action {action_str}")
            direction = match.group(1)
            return "scroll(0, 100)" if direction == "down" else "scroll(0, -100)"
        case "goto":
            match = re.search(r"goto ?\[(.+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid goto action {action_str}")
            url = match.group(1)
            return "goto('{}')".format(url)
        case "new_tab":
            return "new_tab()"
        case "go_back":
            return "go_back()"
        case "go_forward":
            return "go_forward()"
        case "tab_focus":
            match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid tab_focus action {action_str}")
            page_number = int(match.group(1))
            return "tab_focus({})".format(page_number)
        case "close_tab":
            return "tab_close()"
        case "stop":  # stop answer
            match = re.search(r"stop ?\[(.+)\]", action_str)
            if not match:  # some tasks don't require an answer
                answer = ""
            else:
                answer = match.group(1)
            return "send_msg_to_user('{}')".format(answer)

    raise ParseError(f"Invalid action {action_str}")


class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]


class PromptConstructor(object):
    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def __lt__(self, other):
        if not isinstance(other, PromptConstructor):
            return NotImplemented
        return self.instruction_path < other.instruction_path

    def get_lm_api_input(
        self, intro: str, examples: list[tuple[str, str]], current: str
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str

        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                if "mistral" in self.lm_config.model or "llama" in self.lm_config.model:
                    intro_message = intro
                    message = [{"role": "system", "content": intro_message}]
                else:
                    message = [{"role": "system", "content": intro}]
                    for x, y in examples:
                        message.append(
                            {
                                "role": "system",
                                "name": "example_user",
                                "content": x,
                            }
                        )
                        message.append(
                            {
                                "role": "system",
                                "name": "example_assistant",
                                "content": y,
                            }
                        )
                message.append({"role": "user", "content": current})
                return message
            elif self.lm_config.mode == "completion":
                message = f"{intro}\n\n"
                message += "Here are a few examples:\n"
                for example in examples:
                    message += f"Observation\n:{example[0]}\n\n"
                    message += f"Action: {example[1]}\n\n"
                message += "Now make prediction given the observation\n\n"
                message += f"Observation\n:{current}\n\n"
                message += "Action:"
                return message
            else:
                raise ValueError(
                    f"OpenAI models do not support mode {self.lm_config.mode}"
                )     
        elif "vllm" in self.lm_config.provider:
            # vLLM uses OpenAI-compatible chat/completion APIs with standard message format
            if self.lm_config.mode == "chat":
                # For vllm, use standard OpenAI chat format with proper user/assistant roles for examples
                # If using Llama/Mistral, use system message only, else use standard chat format for examples
                if "mistral" in self.lm_config.model or "llama" in self.lm_config.model:
                    intro_message = intro
                    message = [{"role": "system", "content": intro_message}]
                else:
                    message = [{"role": "system", "content": intro}]
                    for x, y in examples:
                        message.append({"role": "user", "content": x})
                        message.append({"role": "assistant", "content": y})
                message.append({"role": "user", "content": current})
                return message
            elif self.lm_config.mode == "completion":
                message = f"{intro}\n\n"
                message += "Here are a few examples:\n"
                for example in examples:
                    message += f"Observation\n:{example[0]}\n\n"
                    message += f"Action: {example[1]}\n\n"
                message += "Now make prediction given the observation\n\n"
                message += f"Observation\n:{current}\n\n"
                message += "Action:"
                return message
            else:
                raise ValueError(
                    f"vLLM models do not support mode {self.lm_config.mode}"
                )
        elif "huggingface" in self.lm_config.provider:
            # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L320
            if "Llama-2" in self.lm_config.model:
                if self.lm_config.mode == "chat":
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    BOS, EOS = "<s>", "</s>"
                    # adding the system message to be the starting of the first example
                    examples = [
                        (
                            B_SYS + intro + E_SYS + examples[0][0],
                            examples[0][1],
                        )
                    ] + examples[1:]
                    message = "".join(
                        [
                            f"{BOS}{B_INST} {x.strip()} {E_INST} {y.strip()} {EOS}"
                            for (x, y) in examples
                        ]
                    )
                    # add the current observation
                    message += f"{BOS}{B_INST} {current.strip()} {E_INST} {self.instruction['meta_data'].get('force_prefix', '')}"

                    return message
                else:
                    raise ValueError("Only chat mode is supported for Llama-2")
            else:
                raise ValueError(
                    f"Huggingface models do not support model_tag {self.lm_config.gen_config['model_tag']}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class DirectPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        if "page" in state_info["info"]:
            page = state_info["info"]["page"]
            url = page.url
        else:
            url = state_info["observation"]["url"]
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(f"Cannot parse action from response {response}")


class PassivePromptConstructor(PromptConstructor):
    """An LM that generates some output based on changes to the environment."""

    def __init__(self, instruction_path, lm_config, tokenizer):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(self, inputs, *args):
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]

        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            for key in inputs:
                inputs[key] = self.tokenizer.decode(self.tokenizer.encode(inputs[key])[:max_obs_length])  # type: ignore[arg-type]

        # replace each keyword in the template with inputs[keyword]
        inputs_for_keywords = {k: inputs[k] for k in keywords}

        current = template.format(**inputs_for_keywords)
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _parse_answer(self, response: str) -> dict:
        def _parse_json(response: str) -> dict:
            first_bracket_index = response.find("{")
            last_bracket_index = response.rfind("}")
            if first_bracket_index == -1 or last_bracket_index == -1:
                raise ParseError(f"Cannot find JSON in response {response}")
            json_str = response[first_bracket_index:last_bracket_index + 1]
            try:
                return json.loads(json_str)
            except Exception as e:
                raise ParseError(f"Cannot parse JSON from response {response}. Error: {e}")
        parsed_response = _parse_json(response)
        try:
            return parsed_response
        except Exception as e:
            raise ParseError(
                f"Cannot parse output from response {response}. Error: {e}"
            )
    # def _parse_answer(self, response: str) -> dict:
    #     return {"answer": response}


class StructuredPassivePromptConstructorBasic(PassivePromptConstructor):
    """
    The same as PassivePromptConstructor but the output is like this: [output] model_output [output]
    """

    def __init__(self, instruction_path, lm_config, tokenizer):
        super().__init__(instruction_path, lm_config, tokenizer)

    def _parse_answer(self, response):
        try:
            # first extract the stuff inside the think tag
            ans_dict = {}
            ans_dict["answer"] = parse_tags_raise(
                response, keys=["output"], merge_multiple=True
            )["output"]

            return {
                "raw_prediction": response,
                "answer": ans_dict["answer"],
            }
        except ParseError as e:
            raise ParseError(
                f"Error while parsing output\n: {e}\n"
                "Make sure your output is correctly formatted. In particular, make sure to wrap your output in the [output] tag."
            )


class StructuredPassivePromptConstructor(PassivePromptConstructor):
    """
    The same as PassivePromptConstructor but the output is like this: <think> reasoning step </think> <output> output </output>
    """

    def __init__(self, instruction_path, lm_config, tokenizer):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def _parse_answer(self, response):
        try:
            # first extract the stuff inside the think tag
            ans_dict = {}
            ans_dict["thought"] = parse_tags_raise(
                response, keys=["think"], merge_multiple=True
            )["think"]
            ans_dict["answer"] = parse_tags_raise(
                response, keys=["output"], merge_multiple=True
            )["output"]

            return {
                "output": "Thought: {}. Answer: {}".format(
                    ans_dict["thought"], ans_dict["answer"]
                ),
                "raw_prediction": response,
                "answer": ans_dict["answer"],
            }
        except ParseError as e:
            raise ParseError(
                f"Error while parsing output\n: {e}\n"
                "Make sure your output is correctly formatted."
            )


class CoTPromptConstructorBgym(PromptConstructor):
    """Same as a prompt constructor, but for bgym"""

    def __init__(self, instruction_path, lm_config, tokenizer):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]
        self.open_world = self.instruction["meta_data"].get("open_world", False)

    def construct(self, obs, meta_data) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        observation = obs["axtree_txt"]
        url = obs["url"]
        intent = obs["goal"]
        previous_action_str = meta_data["action_history"][-1]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        # tutorial = meta_data.get("tutorial", None)
        if max_obs_length:
            observation = self.tokenizer.decode(self.tokenizer.encode(observation)[:max_obs_length])  # type: ignore[arg-type]

        inputs_for_keywords = {
            "objective": intent,
            "url": url,
            "observation": observation,
            "previous_action": previous_action_str,
        }
        if os.environ.get("USE_ENV_DYNAMICS", "false") == "true":
            # this is just for task agent, not for exploration agent
            env_dynamics_str = meta_data.get("env_dynamics_str", None)
            if env_dynamics_str is None or env_dynamics_str == "":
                raise ValueError("Environmental dynamics is not provided")
            intro = intro + "\nHere are the list of environmental dynamics of this environment:\n" + env_dynamics_str
        if "person_description" in meta_data:
            inputs_for_keywords["person_description"] = meta_data["person_description"]
        if "person_description" in meta_data:
            inputs_for_keywords["person_description"] = meta_data["person_description"]
        if "trajectory" in meta_data:
            inputs_for_keywords["trajectory"] = meta_data["trajectory"]

        if "past_actions" in keywords:
            all_actions = meta_data["action_history"]
            all_actions_with_step_counts = [
                "{}: {}".format(i + 1, a) for i, a in enumerate(all_actions)
            ]
            inputs_for_keywords["past_actions"] = "\n".join(
                all_actions_with_step_counts
            )
        
        # if tutorial is not None:
        #     intro = intro + "\nThe provided tutorial can be used as a reference to help you complete the task. Here is the tutorial:\n" + tutorial
        
        if "env_dynamics_str" in keywords:
            # this is just for exploration agent, not for task agent
            inputs_for_keywords["env_dynamics_str"] = meta_data["env_dynamics_str"]

        inputs_for_keywords = {k: inputs_for_keywords[k] for k in keywords}
        current = template.format(**inputs_for_keywords)
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ParseError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        WA_REDDIT = os.getenv("WA_REDDIT", "")
        WA_SHOPPING = os.getenv("WA_SHOPPING", "")
        WA_SHOPPING_ADMIN = os.getenv("WA_SHOPPING_ADMIN", "")
        WA_GITLAB = os.getenv("WA_GITLAB", "")
        WA_WIKIPEDIA = os.getenv("WA_WIKIPEDIA", "")
        WA_MAP = os.getenv("WA_MAP", "")
        WA_HOMEPAGE = os.getenv("WA_HOMEPAGE", "")

        URL_MAPPINGS_BGYM = {
            WA_REDDIT: "http://reddit.com",
            WA_SHOPPING: "http://onestopmarket.com",
            WA_SHOPPING_ADMIN: "http://luma.com/admin",
            WA_GITLAB: "http://gitlab.com",
            WA_WIKIPEDIA: "http://wikipedia.org",
            WA_MAP: "http://openstreetmap.org",
            WA_HOMEPAGE: "http://homepage.com",
        }

        for i, j in URL_MAPPINGS_BGYM.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        return response

    def _parse_tutorial(self, response: str) -> str:
        import json
        """
        Parses a JSON response containing a tutorial with a topic and steps.
        Returns a formatted string with the topic and steps, or raises ParseError if parsing fails.
        """
        try:
            data = json.loads(response)
            topic = data.get("topic", "")
            steps = data.get("steps", [])
            if not topic or not isinstance(steps, list):
                raise ValueError("Missing topic or steps in tutorial JSON.")
            steps_str = "\n".join(steps)
            return f"Topic: {topic}\nSteps:\n{steps_str}"
        except Exception as e:
            raise ParseError(f"Cannot parse tutorial from response: {response}. Error: {e}")

    def _parse_answer(self, response: str) -> dict:
        try:
            parsed_response = self.extract_action(response)
            # for the bgym action we need to map goto links to the local links
            if self.open_world:
                # do not remap urls
                bgym_action = create_bgym_action(parsed_response)
            else:
                bgym_action = create_bgym_action(self.map_url_to_local(parsed_response))
            return {
                "action": parsed_response,
                "bgym_action": bgym_action,
                "raw_prediction": response,
            }
        except Exception as e:
            raise ParseError(
                f"Cannot parse action from response {response}. Error: {e}"
            )


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        if "page" in state_info["info"]:
            page = state_info["info"]["page"]
            url = page.url
        else:
            url = state_info["observation"]["url"]

        previous_action_str = meta_data["action_history"][-1]

        inputs_for_keywords = {
            "objective": intent,
            "url": url,
            "observation": obs,
            "previous_action": previous_action_str,
        }
        if "person_description" in meta_data:
            inputs_for_keywords["person_description"] = meta_data["person_description"]
        if "history" in trajectory[-1]:
            inputs_for_keywords["trajectory"] = trajectory[-1]["history"]

        if "past_actions" in keywords:
            all_actions = meta_data["action_history"]
            all_actions_with_step_counts = [
                "{}: {}".format(i + 1, a) for i, a in enumerate(all_actions)
            ]
            inputs_for_keywords["past_actions"] = "\n".join(
                all_actions_with_step_counts
            )

        inputs_for_keywords = {k: inputs_for_keywords[k] for k in keywords}
        current = template.format(**inputs_for_keywords)
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )
