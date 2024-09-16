import asyncio
import random
import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.functions import kernel_function
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig

### 3plugin_skprompt.pyと似た内容

# カーネルのロード
kernel = sk.Kernel()
kernel.add_service(AzureChatCompletion(env_file_path='../env/.env.text'))
chat_completion = kernel.get_service(type=ChatCompletionClientBase)

# プロンプトの実行設定
execution_settings = AzureChatPromptExecutionSettings(
    max_tokens=4000,
    temperature=0.5,
    stream=True,
)

# ネイティブ関数の作成(inputに対して、コードで処理を行う。Web検索など)
class GenerateNumberPlugin:
    """
    Description: Generate a number between 3-x.
    """

    @kernel_function(description="Generate a random number between 3-x", name="GenerateNumberThreeOrHigher")
    def generate_number_three_or_higher(self, input: str) -> str:
        """
        Generate a number between 3-<input>
        Example:
            "8" => rand(3,8)
        Args:
            input -- The upper limit for the random number generation
        Returns:
            int value
        """
        try:
            return str(random.randint(3, int(input)))
        except ValueError as e:
            print(f"Invalid input {input}")
            raise e
generate_number_plugin = kernel.add_plugin(GenerateNumberPlugin(), "GenerateNumberPlugin")

# セマンティック関数の作成(inputに対して、自然言語でoutputを行う)
prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$input}} paragraphs long. It must be this length.
"""

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="story",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=execution_settings,
)

corgi_story = kernel.add_function(
    function_name="CorgiStory",
    plugin_name="CorgiPlugin",
    prompt_template_config=prompt_template_config,
)



async def main():
    # ネイティブ関数の実行
    number_generate = generate_number_plugin["GenerateNumberThreeOrHigher"]
    number_result = await number_generate(kernel, input=6)
    print(number_result.value)

    # セマンティック関数の実行
    story = await corgi_story.invoke(kernel, input=number_result.value)

    print(f"Generating a corgi story exactly {number_result.value} paragraphs long.")
    print("=====================================================")
    print(story)


if __name__ == "__main__":
    asyncio.run(main())