"""
    All the LM modules for NNetNavExplorerAgent
"""

from agentlab.llm.llm_utils import Discussion, ParseError, retry


class LMModule:
    """
    A generic module class to instantiate various LM modules needed in NNetNav
    """

    def __init__(
        self, chat_llm, flags, prompt_constructor, max_retry=4, fail_message=""
    ):
        self.chat_llm = chat_llm
        self.flags = flags
        self.max_retry = max_retry
        self.prompt_constructor = prompt_constructor
        self.fail_message = fail_message

    def __call__(self, obs: dict) -> dict:
        prompt = self.prompt_constructor.construct(obs)
        try:
            chat_messages = Discussion(prompt)
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=self.prompt_constructor._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                answer=self.fail_message,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )
        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]
        return ans_dict
