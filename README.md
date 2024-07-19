
<<<<<<< HEAD
<<<<<<< HEAD
# 模型
## 使用LLMs：
<<<<<<< HEAD
**LLM**：是指像GPT-4、BERT、T5等大型语言模型，它们基于大量数据进行训练，能够理解和生成自然语言文本。LLM的主要功能是处理各种自然语言任务，如文本生成、翻译、问答等。
### 如何开始使用LLM
#### 1. 选择平台或服务
决定使用哪个LLM平台或服务。常见的有：
=======
### 概念
LLM是LlamaIndex的核心组成部分。它们可以作为独立模块使用，也可以插入到其他核心LlamaIndex模块(索引、检索器、查询引擎）中。  
=======
Llama_index 使用：我将记录的的心得：
>>>>>>> 2202cae (doc:2)
<<<<<<< HEAD
>>>>>>> 0dc4ad0 (doc:2)
=======
=======
Llama_index 使用：
>>>>>>> fdaa548 (Revert "doc:2")
>>>>>>> fdff41e (Revert "doc:2")

- OpenAI（提供GPT-4、ChatGPT等）
- Hugging Face（提供BERT、T5等多种模型）
- Google Cloud AI（提供BERT、T5等模型）
- Microsoft Azure AI（提供GPT、BERT等模型）
#### 2. 注册并获取API密钥
大多数LLM服务提供商要求你注册一个账号并获取API密钥，以便进行身份验证和使用模型。
#### 3.安装必要的软件
根据选择的服务，安装相应的软件包。例如，通过`Hugging Face`的`Transformers`库来访问和使用这些模型：
```
pip install transformers
```
#### 4. 编写调用代码
编写代码来调用LLM进行文本生成、问答等任务。以下是使用OpenAI GPT模型的示例：
```python
from transformers import pipeline

# 加载GPT-2模型
generator = pipeline('text-generation', model='gpt2')

# 生成文本
prompt = "介绍一下LLM和LlamaIndex的关系。"
result = generator(prompt, max_length=50)

print(result[0]['generated_text'])
```

#### 5. 处理模型输出
模型返回的输出通常需要进行处理，以适应你的具体需求。可以根据应用场景对输出文本进行进一步分析、清洗或格式化。
#### 6. 集成到你的应用
将LLM的功能集成到你的应用中，比如聊天机器人、内容生成工具、问答系统等。确保你的应用能够高效地调用模型，并处理返回的数据。
#### 7. 优化和调整
根据实际使用情况，可能需要调整模型参数，如`prompt`、`max_tokens`等，以获得最佳结果。此外，可以结合使用其它工具和库来优化性能和结果质量。
### A Note on Tokenization
在自然语言处理（NLP）中，**Tokenization**（分词）是将文本分解成较小单元（称为tokens）的过程。Tokens可以是单词、子单词、字符甚至句子。对于大型语言模型（LLM）来说，正确的分词非常重要，因为它直接影响模型的输入和输出表现。以下是关于Tokenization的一些关键点和注意事项：
#### 1. 分词方法：
不同的LLM使用不同的分词方法。常见的分词方法包括：
- **Word Tokenization**：将文本按空格或标点符号分开，得到单词作为tokens。这种方法简单但对处理未知单词和拼写错误的鲁棒性较差。

- **Subword Tokenization**：将文本分成子词单元，可以更好地处理未知单词和稀有词。常用的方法包括`Byte-Pair Encoding`（BPE）和`WordPiece`。

- **Character Tokenization**：将文本分解为单个字符，能最大程度地处理未知单词，但会生成大量的tokens，增加计算成本。
#### 2. 常用的分词工具
- **Hugging Face Transformers**：提供了多种预训练模型及其相应的分词器，如BERT、GPT-2、GPT-3等。
- **spaCy**：一个强大的NLP库，支持多种语言的分词。
- **NLTK**：一个经典的NLP库，提供基本的分词功能。
- **StanfordNLP**：提供了高质量的分词和其他NLP工具。
### 3. 示例代码
**使用Hugging Face的Tokenizer**
```python
from transformers import GPT2Tokenizer

# 加载GPT-2的分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 示例文本
text = "Tokenization is important for LLMs."

# 分词
tokens = tokenizer.tokenize(text)
print(tokens)

# 转换为模型输入的ID
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
```
## Using LLMs as standalone modules
### Text Completion
文本补全通常依赖于大型语言模型（LLM），如GPT-3和GPT-4，它们通过大量的文本数据进行训练，能够生成上下文相关的连续文本。这些模型使用概率统计和深度学习技术，预测下一个词或一系列词的可能性。  

下面是使用Hugging Face的Transformers库进行文本补全的示例：
```python
from transformers import pipeline

# 加载文本生成管道，使用GPT-2模型
generator = pipeline('text-generation', model='gpt2')

# 示例文本
prompt = "Text completion is a crucial aspect of"

# 生成补全文本
result = generator(prompt, max_length=50, num_return_sequences=1)

print(result[0]['generated_text'])
```
### Chat
使用Hugging Face的Transformers库构建简单的聊天机器人:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载预训练的模型和分词器
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 初始对话历史
chat_history_ids = None

def chatbot_response(user_input):
    global chat_history_ids

    # 对用户输入进行编码
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # 将新输入添加到对话历史中
    chat_history_ids = model.generate(new_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # 解码生成的响应
    response = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response

# 测试聊天机器人
print("Chatbot: 你好！请问有什么我可以帮忙的吗？")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: 再见！")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
```
## Customizing LLMs within LlamaIndex Abstractions
在LlamaIndex（之前称为GPT Index）中自定义大型语言模型（LLM）涉及到如何在LlamaIndex的抽象层中实现模型的个性化和优化。这通常包括调整模型的行为、响应风格或其他特定的应用需求。以下是如何在LlamaIndex抽象中自定义LLM的几个关键步骤：
### 1. 了解LLMs与LlamaIndex
LlamaIndex：是一个面向特定应用的索引系统或数据库，专门优化了对语言模型生成的数据进行存储和检索。LlamaIndex可以利用LLM生成的数据，进行高效的索引和查询，从而支持复杂的文本检索任务。
- LlamaIndex 依赖于 LLM 生成的数据来创建索引。例如，LLM可以生成大量的文本或文档，然后LlamaIndex对这些文档进行索引，使得用户可以高效地进行查询。
- LLM提供了自然语言处理和生成的能力，而LlamaIndex则提供了高效的数据管理和检索能力。两者结合可以实现高效的文本查询和分析。
### 2. 设置和加载自定义LLM
要在LlamaIndex中使用自定义LLM，你首先需要设置和加载适合你需求的LLM模型。可以选择微调已有模型或者使用专门训练的模型。

<<<<<<< HEAD
**选择或微调模型**  
- **选择模型**：选择一个预训练的LLM模型，如GPT-3、GPT-4、GPT-Neo等。
- **微调模型**：根据具体需求微调模型，以便它能够生成更符合要求的输出。微调通常涉及对模型进行再训练，以适应特定的任务或数据集
=======
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
>>>>>>> fdff41e (Revert "doc:2")
