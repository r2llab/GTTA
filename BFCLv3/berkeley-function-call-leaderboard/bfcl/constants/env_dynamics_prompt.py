MAXIMUM_STEP_LIMIT = 20

ENV_DYNAMICS_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """You are an expert in composing functions. You are given a question and a set of possible functions along with the environment dynamics that describes how the environment changes in response to function calls. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""

ENV_DYNAMICS_SYSTEM_PROMPT = (
    ENV_DYNAMICS_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
    + """
Here is a list of functions in JSON format that you can invoke.
{functions}
Here is a list of environmental dynamics in JSON format that describes how the environment changes in response to function calls. Each item consists of three fields: initial_state, action_taken, and environmental_dynamics. Initial_state is the state of the environment before the action is taken. action_taken is the function call that is made. Environmental_dynamics is the change in the environment as a result of the function call. Please refer to the environmental dynamics to understand and predict how the environment changes in response to function calls.
{environmental_dynamics}
"""
)
