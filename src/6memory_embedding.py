import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.azure_text_embedding import AzureTextEmbedding
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.kernel import Kernel
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.functions import KernelFunction
from semantic_kernel.prompt_template import PromptTemplateConfig

# カーネルのロード
kernel = sk.Kernel()
azure_chat_service = AzureChatCompletion(service_id="text", env_file_path='../env/.env.text')
azure_embedding_service = AzureTextEmbedding(service_id="embedding", env_file_path='../env/.env.embedding')
kernel.add_service(azure_chat_service)
kernel.add_service(azure_embedding_service)
chat_completion = kernel.get_service(type=ChatCompletionClientBase)

# プロンプトの実行設定
execution_settings = AzureChatPromptExecutionSettings(
    max_tokens=4000,
    temperature=0.5,
    stream=True,
)

# メモリ設定
memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=azure_embedding_service)
kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")

# メモリにデータを追加1(save_information)
collection_id = "generic"
async def populate_memory(memory):
    await memory.save_information(collection=collection_id, id="info1", text="Your budget for 2024 is $100,000")
    await memory.save_information(collection=collection_id, id="info2", text="Your savings from 2023 are $50,000")
    await memory.save_information(collection=collection_id, id="info3", text="Your investments are $80,000")

async def search_memory_examples(memory):
    questions = [
        "What is my budget for 2024?",
        "What are my savings from 2023?",
        "What are my investments?",
    ]
    for question in questions:
        print(f"Question: {question}")
        result = await memory.search(collection_id, question)
        print(f"Answer: {result[0].text}\n")


async def setup_chat_with_memory(kernel: Kernel, service_id: str) -> KernelFunction:
    prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if
    it does not have an answer.

    Information about me, from previous conversations:
    - {{recall 'budget by year'}} What is my budget for 2024?
    - {{recall 'savings from previous year'}} What are my savings from 2023?
    - {{recall 'investments'}} What are my investments?

    {{$request}}
    """.strip()

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        execution_settings=execution_settings,
    )

    return kernel.add_function(
        function_name="chat_with_memory",
        plugin_name="chat",
        prompt_template_config=prompt_template_config,
    )
    
async def chat(user_input: str):
    chat_func = await setup_chat_with_memory(kernel=kernel, service_id="text")
    print(f"User: {user_input}")
    answer = await kernel.invoke(chat_func, request=user_input)
    print(f"ChatBot:> {answer}")

# メモリにデータを追加2(save_reference)
memory_collection_name = "SKGitHub"
github_files = {}
github_files["https://github.com/microsoft/semantic-kernel/blob/main/README.md"] = "README: Installation, getting started, and how to contribute"
github_files["https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/02-running-prompts-from-file.ipynb"] = "Jupyter notebook describing how to pass prompts from a file to a semantic plugin or function"
github_files["https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/00-getting-started.ipynb"] = "Jupyter notebook describing how to get started with the Semantic Kernel"
github_files["https://github.com/microsoft/semantic-kernel/tree/main/samples/plugins/ChatPlugin/ChatGPT"] = "Sample demonstrating how to create a chat plugin interfacing with ChatGPT"
github_files["https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src/SemanticKernel/Memory/Volatile/VolatileMemoryStore.cs"] = "C# class that defines a volatile embedding store"

async def add_github_files_to_memory(memory):
    for index, (entry, value) in enumerate(github_files.items()):
        await memory.save_reference(
            collection=memory_collection_name,
            description=value,
            text=value,
            external_id=entry,
            external_source_name="GitHub",
        )
    ask = "I love Jupyter notebooks, how should I get started?"
    question = "What is my budget for 2024?"
    memories = await memory.search(memory_collection_name, ask, limit=5, min_relevance_score=0.77)
    result = await memory.search(collection_id, question)
    print(f"Answer: {result[0].text}\n")
    for index, memory in enumerate(memories):
        print(f"Result {index}:")
        print("  URL:     : " + memory.id)
        print("  Title    : " + memory.description)
        print("  Relevance: " + str(memory.relevance))
        print()

async def main():
    await populate_memory(memory)
    # await search_memory_examples(memory)
    # await chat("What is my budget for 2024?")
    # await chat("talk to me about my finances")

if __name__ == "__main__":
    asyncio.run(populate_memory(memory))
    # asyncio.run(search_memory_examples(memory))
    # asyncio.run(main())
    asyncio.run(add_github_files_to_memory(memory))