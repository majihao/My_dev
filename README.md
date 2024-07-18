# 模型
## 使用LLMs：
### 概念
LLM是LlamaIndex的核心组成部分。它们可以作为独立模块使用，也可以插入到其他核心LlamaIndex模块(索引、检索器、查询引擎）中。  

它们总是在响应合成步骤期间（例如在检索后）使用。根据所使用的索引类型，还可以在索引构造、插入和查询遍历过程中使用LLM。 

LlamaIndex 提供了一个统一的接口来定义 LLM 模块，无论是来自 OpenAI、Hugging Face 还是 LangChain，这样你 不必自己编写定义 LLM 接口的样板代码。此接口由以下内容组成（更多详细信息见下文）：  
- 支持**文本补全**和**聊天**接口（详情如下）  
- 支持**流式**和**非流式**接口  
- 支持**同步**和**异步**接口  
### 使用模式
以下代码展示了如何开始使用LLM。  
如果你还没有它，请安装您的LLM：  
``` 
pip install llama-index-llms-openai 
```
然后：  
```python
from llama_index.llms.openai import OpenAI

# non-streaming
resp = OpenAI().complete("Paul Graham is ")
print(resp)
```
查找有关**独立使用**或**自定义使用**的更多详细信息。  
### 关于分词的说明
默认情况下，LlamaIndex 使用全局的分词器进行所有的词元计数。默认的分词器是来自 tiktoken 的 `cl100k`，这与默认的 LLM `gpt-3.5-turbo` 相匹配。  

如果你更改了 LLM，可能需要更新分词器以确保准确的词元计数、分块和提示。  

分词器的唯一要求是它必须是一个可调用的函数，接受一个字符串并返回一个列表。  
您可以像这样设置全局分词器：
```python
from llama_index.core import Settings

# tiktoken
import tiktoken

Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode

# huggingface
from transformers import AutoTokenizer

Settings.tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta"
)
```
### LLM兼容性跟踪
虽然 LLM 很强大，但并不是每个 LLM 都易于设置。此外，即使设置正确，一些 LLM 在执行需要严格遵循指令的任务时也会遇到困难。  
  
LlamaIndex 提供了与几乎所有 LLM 的集成，但通常不清楚 LLM 是否可以开箱即用，或者是否需要进一步定制。  
## 使用LLM作为独立模块
您可以单独使用我们的 LLM 模块。  

### 文本补全示例
```python
from llama_index.llms.openai import OpenAI

# non-streaming
completion = OpenAI().complete("Paul Graham is ")
print(completion)

# using streaming endpoint
from llama_index.llms.openai import OpenAI

llm = OpenAI()
completions = llm.stream_complete("Paul Graham is ")
for completion in completions:
    print(completion.delta, end="")
```
Chat Example
```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = OpenAI().chat(messages)
print(resp)
```
## 在LlamaIndex抽象中自定义LLMs
你可以在LlamaIndex的其他模块中插入这些LLM抽象（索引、检索器、查询引擎、代理），这允许你构建高级的数据工作流程。  

默认情况下，我们使用OpenAI的`gpt-3.5-turbo`模型。但你也可以选择自定义所使用的底层LLM。  

下面，我们将展示一些LLM自定义的示例。这包括：
- 更改底层LLM
- 更改输出令牌的数量（针对OpenAI, Cohere或AI21）
- 对所有LLM的所有参数进行更细粒度的控制，从上下文窗口到块重叠
### 示例：更改底层LLM
下面展示了一个自定义所使用的LLM的代码片段。在这个例子中，我们使用`gpt-4`而不是`gpt-3.5-turbo`。可用模型包括`gpt-3.5-turbo`、`gpt-3`.`5-turbo-instruct`、`gpt-3`.`5-turbo-16k`、`gpt-4`、`gpt-4-32k`、`text-davinci-003`和`text-davinci-002`。  

请注意，你还可以在LlamaIndex中插入Langchain的LLM页面上显示的任何LLM。
```python
from llama_index.core import KeywordTableIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# alternatively
# from langchain.llms import ...

documents = SimpleDirectoryReader("data").load_data()

# define LLM
llm = OpenAI(temperature=0.1, model="gpt-4")

# build index
index = KeywordTableIndex.from_documents(documents, llm=llm)

# get response from query
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author do after his time at Y Combinator?"
)
```
### 示例：使用HuggingFace LLM
LlamaIndex 支持直接使用来自 HuggingFace 的 LLM。请注意，为了完全私密的体验，还需要设置一个本地嵌入模型。  

许多来自 HuggingFace 的开源模型在每个提示之前都需要一些前置文本，即 `system_prompt`。此外，查询本身可能需要在 `query_str` 周围添加额外的包装器。所有这些信息通常可以从您正在使用的模型的 HuggingFace 模型卡片中找到。  

下面的例子同时使用了 `system_prompt` 和 `query_wrapper_prompt`，使用了在这里找到的模型卡片中特定的提示。

```python
from llama_index.core import PromptTemplate


# Transform a string into input zephyr-specific input
def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


# Transform a list of chat messages into zephyr-specific input
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

Settings.llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)
```
如果将分词器中的所有键都传递给模型，则某些模型将引发错误。导致问题的常见分词器输出是 `token_type_ids`。下面是一个配置预测器以在将输入传递给模型之前删除此预测器的示例：
```python
HuggingFaceLLM(
    # ...
    tokenizer_outputs_to_remove=["token_type_ids"]
)
```
### 示例：使用自定义 LLM 模型 - Advanced 
要使用自定义 LLM 模型，您只需要实现 `LLM` 类（或 `CustomLLM` 为了更简单的界面） 您将负责将文本传递给模型并返回新生成的令牌。

这个实现可以是一些本地模型，甚至可以是你自己的 API 的包装器。

请注意，为了获得完全私密的体验，还应设置本地 嵌入模型 。

下面是一个小型样板示例： 

```python
from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings


class OurLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "custom"
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self.dummy_response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# define our LLM
Settings.llm = OurLLM()

# define embed model
Settings.embed_model = "local:BAAI/bge-base-en-v1.5"


# Load the your data
documents = SimpleDirectoryReader("./data").load_data()
index = SummaryIndex.from_documents(documents)

# Query and print response
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
print(response)
```
使用这种方法，您可以使用任何LLM。也许您在本地运行一个，或者在自己的服务器上运行。只要实现了类并返回了生成的令牌，就应该可以正常工作。需要注意的是，我们需要使用提示助手来自定义提示大小，因为每个模型的上下文长度都略有不同。

装饰器是可选的，但它通过回调提供对LLM调用的可观察性。  

请注意，您可能需要调整内部提示以获得良好的性能。即便如此，您应该使用足够大的LLM来确保它能够处理LlamaIndex内部使用的复杂查询，因此您的效果可能会有所不同。