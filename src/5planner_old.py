import asyncio
from typing import Annotated
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.core_plugins.text_plugin import TextPlugin
from semantic_kernel.core_plugins.math_plugin import MathPlugin
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.planners import SequentialPlanner
from semantic_kernel.planners.function_calling_stepwise_planner import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
    FunctionCallingStepwisePlannerResult,
)

###
# SequentialPlannerやFunctionCallingStepwisePlannerは、廃止されるみたい
###

# カーネルのロード
kernel = Kernel()
kernel.add_service(AzureChatCompletion(env_file_path='../env/.env.text'))
chat_completion = kernel.get_service(type=ChatCompletionClientBase)

# プロンプトの実行設定
execution_settings = AzureChatPromptExecutionSettings(
    max_tokens=4000,
    temperature=0.5,
    stream=True,
)

# プラグインの定義方法には、2種類ある
# 1. skprompt.txtを使用したプロンプトベースのプラグイン
# 2. classを使用したコードベースのプラグイン
# 3. importして使用する https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/core_plugins

# プラグインの定義1
plugins_directory = "../semantic-kernel/prompt_template_samples/"
summarize_plugin = kernel.add_plugin(plugin_name="SummarizePlugin", parent_directory=plugins_directory)
writer_plugin = kernel.add_plugin(plugin_name="WriterPlugin", parent_directory=plugins_directory)
text_plugin = kernel.add_plugin(plugin=TextPlugin(), plugin_name="TextPlugin")
math_plugin = kernel.add_plugin(plugin=MathPlugin(), plugin_name="MathPlugin")
time_plugin = kernel.add_plugin(plugin=TimePlugin(), plugin_name="TimePlugin")

# プラグインの定義2 (例)
class EmailPlugin:
    """
    Description: EmailPlugin provides a set of functions to send emails.

    Usage:
        kernel.import_plugin_from_object(EmailPlugin(), plugin_name="email")

    Examples:
        {{email.SendEmail}} => Sends an email with the provided subject and body.
    """

    @kernel_function(name="SendEmail", description="Given an e-mail and message body, send an e-email")
    def send_email(
        self,
        subject: Annotated[str, "the subject of the email"],
        body: Annotated[str, "the body of the email"],
    ) -> Annotated[str, "the output is a string"]:
        """Sends an email with the provided subject and body."""
        return f"Email sent with subject: {subject} and body: {body}"

    @kernel_function(name="GetEmailAddress", description="Given a name, find the email address")
    def get_email_address(
        self,
        input: Annotated[str, "the name of the person"],
    ):
        email = ""
        if input == "Jane":
            email = "janedoe4321@example.com"
        elif input == "Paul":
            email = "paulsmith5678@example.com"
        elif input == "Mary":
            email = "maryjones8765@example.com"
        else:
            input = "johndoe1234@example.com"
        return email
kernel.add_plugin(plugin_name="EmailPlugin", plugin=EmailPlugin())

# プランナーは生成AIが質問に対してstep by stepで考えて回答する
# いくつか種類がある
# 1. シーケンシャルプランナー (事前に定義された一連のステップを順番に実行する。メール送信や文書作成など)
# 2. ステップワイズプランナー (質問に応じて動的にステップを決定する。状況に応じてステップが異なっても良い場合)

# シーケンシャルプランナー
# planner = SequentialPlanner(kernel=kernel, service_id="default")

# ステップワイズプランナー
# question = "今日の日付を教えてください。今日の季節を春、夏、秋、冬から選んでください。その季節のことわざを3つお願いいたします"
questions = [
    "今日の日付を教えてください。",
    "今日の季節を春、夏、秋、冬から選んでください。",
    "今日の季節のことわざを1つお願いいたします。",
]
options = FunctionCallingStepwisePlannerOptions(
    max_iterations=10,
    max_tokens=4000,
    stream=True,
)
planner = FunctionCallingStepwisePlanner(service_id="default", options=options)


async def main():
    # ステップワイズプランナーの実行
    for question in questions:
        result = await planner.invoke(kernel=kernel, question=question)
        print(f"Q: {question}\nA: {result.final_answer}\n")
    # print(f"Chat history: {result.chat_history}\n")
if __name__ == "__main__":
    asyncio.run(main())