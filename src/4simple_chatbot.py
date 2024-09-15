import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.functions import KernelArguments

# カーネルのロード
kernel = sk.Kernel()
kernel.add_service(AzureChatCompletion(env_file_path='../env/.env.text'))
chat_completion = kernel.get_service(type=ChatCompletionClientBase)

# チャット履歴の作成
history = ChatHistory()
history.add_system_message("あなたは、優秀な生成AIアシスタントです")

# プロンプトの実行設定
execution_settings = AzureChatPromptExecutionSettings(
    max_tokens=4000,
    temperature=0.5,
    stream=True,
)

# プロンプトテンプレートの設定
prompt = """ 
 {{$history}} 
ユーザー: {{$user_input }} 
チャットボット: """ 

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="chat",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="user_input", description="ユーザーの入力", is_required=True),
        InputVariable(name="history", description="会話履歴", is_required=True),
    ],
    execution_settings=execution_settings,
)

# 関数の追加
chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="chatPlugin",
    prompt_template_config=prompt_template_config,
)

async def main():
    while True:
        # ユーザーの入力を対話形式で取得
        user_input = input("ユーザー: ")
        # 引数定義
        arguments = KernelArguments(
            user_input=user_input,
            history=history,
        )
        # プロンプトの実行
        print("チャットボット: ", end="")
        response = ""
        async for chunk in kernel.invoke_stream(
            function=chat_function,
            arguments=arguments,
        ):
            for item in chunk:
                if hasattr(item, "items") and item.items:
                    print(item.items[0].text, end="", flush=True)
                    response += item.items[0].text
        history.add_user_message(user_input) # ユーザーの入力をチャット履歴に追加
        history.add_assistant_message(response) # アシスタントの回答をチャット履歴に追加
        print()
        print()

if __name__ == "__main__":
    asyncio.run(main())