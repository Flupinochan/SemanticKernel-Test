import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments

### ハルシネーション対策用プラグインの使用例
# ハルシネーション対策は、3stepで行う
# 1. エンティティを抽出する
# 2. 抽出したエンティティのグラウンディング(ハルシネーションのチェック)を行い、ハルシネーションされているものをリスト化する
# 3. ハルシネーションされているものを除去して回答する



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

# 元の文章
grounding_text = """
MetalMental is a Full-Stack and SRE engineer
フロントエンドからバックエンド、インフラまで手掛けるメタルなメンタルを持つエンジニアです
Currently working on generative AI

Skills
Frontendは、Figma、Next.js、Tailwind CSS、Reactを使用してモダンなWebアプリケーションを作成します
Backendは、Flask、GraphQL、Generative AIなどのテキスト生成や画像生成、RAGなどのトレンド技術を使用します
Infrastructureは、AWS、VMware、Docker、コンテナやサーバレス、IaCおよびCI/CDを用いた環境構築をします

About Me
経歴は、学生時代は、BlenderやMMDでアニメーション作成をしていました
卒業した後は、すき屋でワンオペを行い、気づいたらエンジニアになっていました

得意分野は、IaCによるCI/CD環境構築です
最近は、サーバレスや生成AIを中心に勉強しています

趣味は、アニメを見ながら、新サービスの検証やコードを書くこと!
自作アバターで、メタバースを探索すること!
"""

groundingSemanticFunctions = kernel.add_plugin(parent_directory="../semantic-kernel/prompt_template_samples/", plugin_name="GroundingPlugin")

entity_extraction = groundingSemanticFunctions["ExtractEntities"]
reference_check = groundingSemanticFunctions["ReferenceCheckEntities"]
entity_excision = groundingSemanticFunctions["ExciseEntities"]


# KernelArgumentsは、skprompt.txtに渡す引数を指定する。skprompt.txtでは、{input}と{style}という引数を使用している。ファイルはShift-JISで保存すること
async def main():
    # ハルシネーションされた要約文章
    summary_text = """
豆腐メンタルを持つエンジニアのMetalMentalです。
Frontendでは、Vue.jsを駆使してWebアプリケーションを作成しました。
Backendでは独自のLLMを開発して、テキスト生成や画像生成、RAGなどのトレンド技術を使用しています。
Azureと自作のロボットでInfrastructureを支える一方、学生時代はゲーム開発に熱中し、卒業後は異世界でエンジニアをしています。
    """
    print(summary_text)
    summary_text = summary_text.replace("\n", " ").replace("  ", " ")

    # ハルシネーションされた要約文章を対象にして、人や場所などのエンティティを抽出する
    # topicやexample_entitiesは、ハルシネーション対策したいものを指定する。指定しなければ全てチェックされる
    extraction_result = await kernel.invoke(
        entity_extraction,
        input=summary_text,
        topic="",
        example_entities="",
    )
    print(extraction_result)
    print("--------------------------------")
    
    # 抽出したエンティティのグラウンディング(ハルシネーションのチェック)を行い、ハルシネーションされているもの(元の文章に存在しない人や場所)をリスト化する
    grounding_result = await kernel.invoke(reference_check, input=extraction_result.value, reference_context=grounding_text)
    print(grounding_result)
    print("--------------------------------")
    # ハルシネーションを除去して回答する
    excision_result = await kernel.invoke(entity_excision, input=summary_text, ungrounded_entities=grounding_result.value)
    print(excision_result)

if __name__ == "__main__":
    asyncio.run(main())