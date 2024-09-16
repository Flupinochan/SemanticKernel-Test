import asyncio
import os
import shutil
import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings

# カーネルのロード
kernel = sk.Kernel()
kernel.add_service(AzureChatCompletion(env_file_path='../env/.env.text'))
chat_completion = kernel.get_service(type=ChatCompletionClientBase)

# チャット履歴の作成
history = ChatHistory()
history.add_system_message("あなたは、優秀な生成AIアシスタントです")
history.add_user_message("LangchainとSemantic Kernelの違いを教えてください")

# プロンプトの実行設定
execution_settings = AzureChatPromptExecutionSettings(
    max_tokens=4000,
    temperature=0.5,
    stream=True,
    number_of_responses=3,
)

async def main():
    # ./multi_resultディレクトリがなければ作成する
    if not os.path.exists("./multi_result"):
        os.makedirs("./multi_result")
    else:
        # ./multi_resultディレクトリの中を空にする
        shutil.rmtree("./multi_result")
        os.makedirs("./multi_result")

    # プロンプトの実行
    async for chunk in chat_completion.get_streaming_chat_message_contents(
        settings=execution_settings,
        kernel=kernel,
        chat_history=history,
    ):
        for item in chunk:
            if hasattr(item, "items") and item.items:
                result = item.items[0].text
                # 3つの回答をそれぞれ3つのファイルに保存する
                with open(f"./multi_result/{item.choice_index+1}.txt", "a", encoding="utf-8") as f:
                    f.write(result)

if __name__ == "__main__":
    asyncio.run(main())