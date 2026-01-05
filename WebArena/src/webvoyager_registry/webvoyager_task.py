import json
import logging
import tempfile
import urllib.parse
from typing import Tuple

import numpy as np
import playwright.sync_api

from browsergym.core.task import AbstractBrowserTask

logger = logging.getLogger(__name__)


class WebVoyagerTask(AbstractBrowserTask):
    @classmethod
    def get_task_id(cls):
        return "webvoyager_task"

    def __init__(
        self, seed: int, web_name: str, id: str, goal: str, start_url: str
    ) -> None:
        """
        Args:
            seed: random seed.
            web_name: name of the website.
            id: id for the task
            goal: goal for the agent
            start_url: url for the first website
        """
        super().__init__(seed)
        self.web_name = web_name
        # bigger viewport so screenshot has more stuff
        self.viewport = {"width": 1280, "height": 1600}
        self.id = id
        self.goal = goal
        self.start_url = start_url

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        page.goto(self.start_url, timeout=10000)
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
