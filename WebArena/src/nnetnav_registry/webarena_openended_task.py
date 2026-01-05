import json
import logging
import tempfile
import urllib.parse
from typing import Tuple

import numpy as np
import playwright.sync_api

from browsergym.core.task import AbstractBrowserTask
from browsergym.webarena.instance import WebArenaInstance

logger = logging.getLogger(__name__)


class NNetNavOpenEndedTask(AbstractBrowserTask):
    @classmethod
    def get_task_id(cls):
        return "nnetnav_openended"

    def __init__(self, seed: int, start_url: str, goal: str = None) -> None:
        """
        Args:
            seed: random seed.
            start_url: str, the url for the starting page.
            goal: str, the initial goal.

        """
        super().__init__(seed)
        self.start_url = start_url
        # bigger viewport so screenshot has more stuff
        self.viewport = {"width": 1280, "height": 1600}
        self.goal = goal

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        page.goto(self.start_url, timeout=20000) # we change timeout to 20 seconds to avoid timeout
        return self.goal, {}

    def teardown(self) -> None:
        pass

    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list[str]
    ) -> Tuple[float, bool, str, dict]:
        reward, done, msg, info = 0, False, "", {}
        if chat_messages and chat_messages[-1]["role"] == "assistant":
            done = True

        return reward, done, msg, info


class WebArenaOpenEnded(AbstractBrowserTask):
    """
    Base class for all WebArena tasks.

    """

    def __init__(self, seed: int, config_str: str, task_id: int) -> None:
        super().__init__(seed)

        # task properties, will be used to set up the browsergym environment
        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 1000  # ms
        self.timeout = 10000  # ms

        self.webarena_instance = WebArenaInstance()
        self.config_file: str = None
        self.task_id = task_id
        # substitute URLs
        config = json.loads(config_str)
        config_to_load = config.copy()
        start_url = config["start_url"]
        for pattern, url_key in {
            "__GITLAB__": "gitlab",
            "__REDDIT__": "reddit",
            "__SHOPPING__": "shopping",
            "__SHOPPING_ADMIN__": "shopping_admin",
            "__WIKIPEDIA__": "wikipedia",
            "__MAP__": "map",
        }.items():
            start_url = start_url.replace(pattern, self.webarena_instance.urls[url_key])

        config_to_load["start_url"] = start_url
        # load all task configs to JSON
        self.config = config_to_load

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:

        # pick a task at random

        # hack: dynamically build a config file to read from
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            json.dump(self.config, f)
            f.flush()
            self.config_file = f.name

        # authenticate
        for site in self.config["sites"]:
            self.webarena_instance.ui_login(site=site, page=page)

        # set geolocation
        page.context.set_geolocation(self.config["geolocation"])

        # navigate to the starting url(s) (might need several pages)
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/browser_env/envs.py#L150
        if self.config["start_url"]:
            start_urls = self.config["start_url"].split(" |AND| ")
            for i, url in enumerate(start_urls):
                page.goto(url)
                if i < len(start_urls) - 1:
                    page = page.context.new_page()

        # recover goal
        goal = self.config["intent"]
        return goal, {}

    def cheat(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> None:
        raise NotImplementedError

    @classmethod
    def get_task_id(cls):
        """
        Generic class for several task ids, this way of obtaining the task id is not compatible for now.
        """

    def teardown(self) -> None:
        # Nothing to be done here
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/browser_env/envs.py#L227
        pass

    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list[str]
    ) -> Tuple[float, bool, str, dict]:

        # safeguard: check that all open tabs are either blank or within the list of WebArena URLs
        authorized_locations = ["newtab", ""] + [
            urllib.parse.urlparse(url).netloc
            for url in [
                *self.webarena_instance.urls.values(),
                self.webarena_instance.home_url,
            ]
        ]
        for open_page in page.context.pages:
            page_location = urllib.parse.urlparse(open_page.url).netloc
            if not page_location in authorized_locations:
                logger.info("Unauthorized url, terminating task")
                return 0, True, "", {"error": "Unauthorized url, terminating task"}

        # import webarena dynamically
        from webarena.browser_env.actions import ActionTypes

        # if any, use the last assistant message as the stop answer for webarena
        if chat_messages and chat_messages[-1]["role"] == "assistant":
            last_action = {
                "action_type": ActionTypes.STOP,
                "answer": chat_messages[-1]["message"],
            }
        elif chat_messages and chat_messages[-1]["role"] == "infeasible":
            last_action = {"action_type": ActionTypes.STOP, "answer": "N/A"}
        else:
            last_action = {"action_type": ActionTypes.NONE, "answer": ""}
            # llm_fuzzy_match() bugfix
            last_action["answer"] = "whatever"

        if last_action["action_type"] == ActionTypes.STOP:
            return 1, True, "", {}
        else:
            return 0, False, "", {}
