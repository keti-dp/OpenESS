import kfp

from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import NamedTuple

#project_id = "KETI-IISRC"
#pipeline_root_path = 'https://console.cloud.google.com/vertex-ai/pipelines?hl=ko&project=keti-iisrc'
PIPELINE_ROOT = "gs://ess-bucket-1/pipeline_root/" # Pipeline 결과를 저장할 위치

@component(base_image="python:3.9", output_component_file="first-component.yaml") # 컴포넌트 정의
def product_name(text: str) -> str:
    return text

#product_name_component = kfp.components.load_component_from_file('./first-component.yaml')

@component(packages_to_install=["emoji"]) # 컴포넌트 정의
def emoji(
    text: str, # 입력 및 출력
) -> NamedTuple(
    "Outputs",
    [
        ("emoji_text", str),  # Return parameters
        ("emoji", str),
    ],
):

    import emoji # 함수 설정
    emoji_text = text
    emoji_str = emoji.emojize(':' + emoji_text + ':', use_aliases=True)
    print("output one: {}; output_two: {}".format(emoji_text, emoji_str))
    return (emoji_text, emoji_str)

@component
def build_sentence(
    product: str,
    emoji: str,
    emojitext: str
) -> str:
    print("We completed the pipeline, hooray!")

    end_str = product + " is "
    if len(emoji) > 0:
        end_str += emoji
    else:
        end_str += emojitext
    return(end_str)

@pipeline( # pipeline 설정
    name="hello-world",
    description="An intro pipeline",
    pipeline_root=PIPELINE_ROOT,
)
# You can change the `text` and `emoji_str` parameters here to update the pipeline output
def intro_pipeline(text: str = "Vertex Pipelines", emoji_str: str = "sparkles"): # 입력
    product_task = product_name(text) # product_name 함수의 task
    emoji_task = emoji(emoji_str) # emoji 함수의 task
    consumer_task = build_sentence( # product_task 및 emoji_task의 output 값을 build_sentence 입력 값으로 사용
        product_task.output,
        emoji_task.outputs["emoji"],
        emoji_task.outputs["emoji_text"],
    )

compiler.Compiler().compile( # 컴파일
    pipeline_func=intro_pipeline, package_path="intro_pipeline_job.json"
)