import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments


# カーネルのロード
kernel = sk.Kernel()
kernel.add_service(AzureChatCompletion(env_file_path='../env/.env.text'))
chat_completion = kernel.get_service(type=ChatCompletionClientBase)

# プラグインを設定
plugin = kernel.add_plugin(parent_directory="../prompt_template/", plugin_name="Plugin",)

# プロンプトの実行設定
execution_settings = AzureChatPromptExecutionSettings(
    max_tokens=4000,
    temperature=0.5,
    stream=True,
)

# KernelArgumentsは、skprompt.txtに渡す引数を指定する。skprompt.txtでは、{input}と{style}という引数を使用している。ファイルはShift-JISで保存すること
async def main():
    # プラグインの関数を呼び出す
    dayo_function = plugin["Dayo"]
    async for chunk in kernel.invoke_stream(
        dayo_function,
        KernelArguments(input="LangchainとSemantic Kernelの違いを教えてください", style="語尾にだよをつけて回答してください"),
    ):
        for item in chunk:
            if hasattr(item, "items") and item.items:
                print(item.items[0].text, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())