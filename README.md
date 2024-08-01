# 模型
## LLMs
### 使用LLMs：
**LLM**：是指像GPT-4、BERT、T5等大型语言模型，它们基于大量数据进行训练，能够理解和生成自然语言文本。LLM的主要功能是处理各种自然语言任务，如文本生成、翻译、问答等。   

**1. 选择平台或服务**  
决定使用哪个LLM平台或服务。常见的有：

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
编写代码来调用LLM进行文本生成、问答等任务
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
#### 3. 示例代码
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
输出
```
['Token', 'ization', 'Ġis', 'Ġimportant', 'Ġfor', 'ĠLL', 'Ms', '.']
[30642, 1634, 318, 1593, 329, 27140, 10128, 13]
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

**选择或微调模型**  
- **选择模型**：选择一个预训练的LLM模型，如GPT-3、GPT-4、GPT-Neo等。
- **微调模型**：根据具体需求微调模型，以便它能够生成更符合要求的输出。微调通常涉及对模型进行再训练，以适应特定的任务或数据集

### Example: Changing the underlying LLM  
下面显示了自定义所使用的 LLM 的示例片段，这个例子使用的是`chatglm3-6b`模型。
```python
from llama_index.core import KeywordTableIndex, SimpleDirectoryReader
from transformers import AutoTokenizer, AutoModel

# 加载数据
documents = SimpleDirectoryReader("data").load_data()

# 初始化 chatglm3-6b 模型和分词器
model_name = "THUDM/chatglm3-6b"

# Load tokenizer and model with trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
# 定义索引
index = KeywordTableIndex.from_documents(documents, llm=model, tokenizer=tokenizer)

# 查询索引
query_engine = index.as_query_engine()
response = query_engine.query(
    "创始人在 Y Combinator 之后做了什么？"
)

# 打印或处理响应
print(response)
```
### Example: Changing the number of output tokens  
更改输出令牌数量通常涉及到调整模型生成文本的长度或详细程度，这在不同的LLM中可能有所不同。例如，对于OpenAI的`GPT-3.5 Turbo`模型，可以通过调整`max_length`参数来控制生成文本的长度。以下是一个示例代码，演示如何使用`Transformers`库来调整输出文本的长度：
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel, PretrainedConfig

# 加载模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 输入文本
input_text = "Your input text here."

# 生成文本的最大长度
max_output_length = 20

# 生成文本
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=max_output_length, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
```
输出：
```
Generated Text: Your input text here.

You can also use the following command to add a new line to
```
在上述示例中，`max_length`参数控制了生成文本的最大长度，您可以根据需要调整这个参数来改变输出文本的长度。
### Example:Explicitly configure `context_window` and `num_output`
在自然语言生成（NLG）任务中，特别是使用预训练语言模型（如GPT-2）时，`context_window` 和 `num_output` 是两个重要的参数，用于控制生成文本的行为和输出数量。  
- `context_window`指定了生成文本时考虑的上下文长度。这个上下文可以是输入的文本或者是模型自身在生成过程中保持的记忆。较大的 context_window 可以提供更多的上下文信息，有助于生成更连贯和相关的文本。然而，过大的上下文窗口可能会增加计算复杂度和生成时间。  
- `num_output`确定了生成器一次性输出的文本序列数量。在一些应用场景中，可能需要生成多个类似但不完全相同的文本序列，以增加多样性或者考虑不同的生成可能性。例如，对于对话系统或创意文本生成，多个输出序列有助于提供更多选择或者多样性。
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from llama_index.core import SimpleDirectoryReader

# 设置输入文件夹
data_directory = "data"

# 加载数据
documents = SimpleDirectoryReader(data_directory).load_data()

# 设置模型和tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name,trust_remote_code=True)
model = GPT2LMHeadModel.from_pretrained(model_name,trust_remote_code=True)

# 设置参数
context_window = 40
num_output = 1

# 使用模型生成文本
input_text = "您要生成的文本内容"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids, max_length=context_window, num_return_sequences=num_output)

# 处理生成的输出
for output in outputs:
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(generated_text)
```
### Example: Using a Custom LLM Model - Advanced
使用自定义的语言生成模型，你只需要实现LLM类（或者对于更简单的接口，实现CustomLLM类）
```python
import argparse  # 用于命令行参数解析
import os  # 系统操作相关
import os.path as osp  # 路径操作相关
from typing import Any  # 类型提示
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, ServiceContext,
    load_index_from_storage, StorageContext, Settings, Document
)
from llama_index.core.llms import (
    CustomLLM, LLMMetadata, CompletionResponse, CompletionResponseGen
)
# from llama_index.core.embeddings import HuggingFaceEmbedding
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import ReActAgent, FunctionCallingAgentWorker
from llama_index.agent.openai import OpenAIAgent
# from tool import read  # 假设存在这样的工具函数或方法
from transformers import AutoModel, AutoTokenizer

parser=argparse.ArgumentParser()
parser.add_argument("--model_name",type=str,default="chatglm3-6b")
parser.add_argument("--model_path",type=str,default="THUDM/chatglm3-6b")
parser.add_argument("--context_window",type=int,default=8192 ,help="上下文窗口大小")
parser.add_argument("--num_output",type=int,default=128 ,help="output tokren number")
parser.add_argument("--data_path",type=str,default='/data/chatglmindex/data',help="data_path")
parser.add_argument("--emb_path",type=str,default='/home/majihao/chatglmindex/m3e-base',help="emb model path")


parser.add_argument("--index_path",type=str,default='/data/chatglmindex/IndexJson',help="index path")
args=parser.parse_args()

#定义自定义LLM类
class ChatGML(CustomLLM): 
    context_window: int =8129   
    num_output: int =128  
    model_name: str = "chatglm3-6b"  
    tokenizers: object = None 
    models: object = None  

    def __init__(self,
                 tokenizer,
                 model,
                 model_name,
                 context_window,
                 num_output):
        super().__init__()
        self.tokenizers = tokenizer
        self.models = model
        self.context_window=context_window
        self.num_output=num_output
        self.model_name=model_name

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_length = len(prompt)
        # only return newly generated tokens
        text,_ = self.models.chat(self.tokenizers, prompt, history=[])
        return CompletionResponse(text=text)

    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError()

# 初始化模型和分词器
model_path=args.model_path
model_name=args.model_name
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 实例化 ChatGML 类
chat_gml = ChatGML(
    tokenizer=tokenizer,
    model=model,
    model_name=model_name,
    context_window=8129,
    num_output=128
)

# 提示文本
prompt = "今天天气怎么样？"

# 调用 complete 方法获取响应
response = chat_gml.complete(prompt)
print("生成的文本响应：", response.text)
```
## Embeddings
在LlamaIndex中，Embeddings用于以精细化的数值表示来表达您的文档。嵌入模型接受文本作为输入，并返回一个长列表的数字，用于捕捉文本的语义信息。这些预训练的嵌入模型被训练成以这种方式表示文本，并能够支持许多应用程序，包括搜索！
### Usage Pattern
在LlamaIndex中，通常会在`Settings`对象中指定嵌入模型，然后将其用于向量索引。嵌入模型用于对构建索引时使用的文档进行嵌入，以及后续使用查询引擎进行查询时嵌入任何查询文本。您还可以针对每个索引指定嵌入模型。
```python
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 假设使用HuggingFace的嵌入模型
from llama_index.core import VectorStoreIndex

# 设置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding(model_name="text-embedding-ada-002",trust_remote_code=True)

# 加载数据文档
documents = SimpleDirectoryReader("../data").load_data()

# 使用嵌入模型构建向量索引
index = VectorStoreIndex.from_documents(documents)

# 查询引擎
query_engine = index.as_query_engine()

# 执行查询
query_text = "关于狗的问题"
results = query_engine.query(query_text)

# 处理查询结果
for result in results:
    print(result)
```
>`VectorStoreIndex` 是 LlamaIndex 中的一个重要组件，用于管理和查询文档的向量表示

# Prompts
## 概念
"提示"在语言模型中指的是最初提供给模型的文本或查询，用于生成响应。它设定了模型在生成文本时的上下文或方向。提示的内容可以根据任务和期望的输出而大不相同。例如，在聊天机器人应用中，一个提示可以是用户的查询，比如"今天天气怎么样？"在文本生成中，提示可以是一个故事的开头或用户希望模型写作的特定主题。
## Usage Pattern
### Defining a custom prompt

通过 `PromptTemplate` 类，您可以根据具体需求定义和格式化不同类型的提示信息，以支持各种对话和问答场景的应用。
```python
from llama_index.core import PromptTemplate

# 定义模板字符串
template = (
    "We have provided context information below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based on this information, please answer the question: {query_str}\n"
)

# 创建 PromptTemplate 对象
qa_template = PromptTemplate(template)

# 格式化文本提示
context_info = "This is the context information."
query = "What is the main conclusion drawn from the data?"

prompt_text = qa_template.format(context_str=context_info, query_str=query)
print("Formatted prompt text:\n", prompt_text)

# 格式化消息提示（用于聊天 API）
def format_messages(context_str, query_str):
    messages = []
    messages.append(f"System: We have provided context information below:")
    messages.append(f"System: ---------------------")
    messages.append(f"System: {context_str}")
    messages.append(f"System: ---------------------")
    messages.append(f"System: Based on this information, please answer the question: {query_str}")
    return messages

messages = format_messages(context_info, query)
print("\nFormatted messages for chat:\n")
for msg in messages:
    print(msg)
```
输出
```
Formatted prompt text:
 We have provided context information below:
---------------------
This is the context information.
---------------------
Based on this information, please answer the question: What is the main conclusion drawn from the data?

Formatted messages for chat:

System: We have provided context information below:
System: ---------------------
System: This is the context information.
System: ---------------------
System: Based on this information, please answer the question: What is the main conclusion drawn from the data?
```
### Getting and Setting Custom Prompts
**常用提示**  

最常用的提示是 `text_qa_template` 和 `refine_template`.
- `ext_qa_template` - 用于使用检索到的节点获取查询的初始答案
- `refine_template` - 当检索到的文本不适合单个 LLM 调用时使用 `response_mode="compact"` （默认），或者当使用检索多个节点时 `response_mode="refine"`。 第一个查询的答案作为  
- `existing_answer`，并且法学硕士必须根据新的上下文更新或重复现有答案。

**访问提示**  

使用 `get_prompts` 方法来获取提示
```python
# 加载你的数据
documents = SimpleDirectoryReader("data").load_data()
index = SummaryIndex.from_documents(documents)

# 使用自定义的提示模板来设置查询引擎
query_engine = index.as_query_engine(response_mode="compact")

# 获取更新后的提示字典
prompts_dict = query_engine.get_prompts()

# 打印提示字典的键
print(list(prompts_dict.keys()))
```
这段代码从一个索引对象 `index` 中获取查询引擎，并以`compact`配置它。然后，通过查询引擎的 `get_prompts()` 方法获取预定义的提示信息，并打印出所有提示信息的键名列表。  

**修改查询引擎中使用的提示**  
可以通过调用`set_prompts`方法并传入新的`PromptTemplate`实例来实现。这将替换先前设置的任何提示模板。  
```python
from llama_index.core import PromptTemplate, SimpleDirectoryReader, SummaryIndex, Settings

# 定义一个新的提示模板
new_template = (
    "New context details:\n"
    "{context_str}\n"
    "Please provide your response to: {query_str}\n"
)
updated_template = PromptTemplate(new_template)

# 加载数据并创建索引
documents = SimpleDirectoryReader("data").load_data()
index = SummaryIndex.from_documents(documents)

# 获取查询引擎并更新提示模板
query_engine = index.as_query_engine(response_mode="compact")
query_engine.set_prompts(updated_template)

# 获取更新后的提示字典
prompts_dict = query_engine.get_prompts()

# 打印提示字典的键
print(list(prompts_dict.keys()))
```
**修改索引构建中使用的提示**

```python
from llama_index.core import SimpleDirectoryReader, SummaryIndex, PromptTemplate

# 1. 加载数据
documents = SimpleDirectoryReader("data").load_data()

# 2. 定义新的提示模板
new_template = (
    "We have provided context information below:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on this information, please answer the question: {query_str}\n"
)

# 3. 创建索引并设置提示模板
index = SummaryIndex.from_documents(documents)
index.set_prompt_template(new_template)

# 4. 重新构建索引
index.rebuild_index()

# 5. 创建查询引擎并获取提示字典
query_engine = index.as_query_engine(response_mode="compact")
prompts_dict = query_engine.get_prompts()

# 6. 打印提示字典的键
print(list(prompts_dict.keys()))
```
### Advanced Prompt Capabilities
**部分格式化**  

部分格式化提示，填写一些变量，同时留下其他变量稍后填写。
```python
from llama_index.core import PromptTemplate

# 定义一个模板，包含需要填充的部分和占位符
template = (
    "今天的天气是 {weather}。\n"
    "我觉得 {emotion} 很重要。\n"
    "请帮我理解 {query_str}。\n"
)

# 创建一个PromptTemplate对象
prompt_template = PromptTemplate(template)

# 部分填充模板，留下其他变量以便稍后填写
partial_prompt = prompt_template.format(
    weather="晴朗",
    emotion="心情",
    query_str="{query}"
)

# 输出部分填充后的提示
print(partial_prompt)
```
输出
```
今天的天气是 晴朗。
我觉得 心情 很重要。
请帮我理解 {query}。
```
**模板变量映射**  

如果想要将模板中的变量映射到具体的值，可以使用Python的字典来实现变量映射。
```python
from llama_index.core import PromptTemplate

# 定义一个模板，包含需要填充的变量
template = (
    "今天的天气是 {weather}。\n"
    "我觉得 {emotion} 很重要。\n"
    "请帮我理解 {query_str}。\n"
)

# 创建一个PromptTemplate对象
prompt_template = PromptTemplate(template)

# 定义一个字典，映射模板中的变量到具体的值
variable_mapping = {
    "weather": "晴朗",
    "emotion": "开心",
    "query_str": "这个问题"
}

# 使用字典进行变量映射，并生成最终的提示文本
final_prompt = prompt_template.format_map(variable_mapping)

# 输出最终的提示文本
print(final_prompt)
```
输出
```
今天的天气是 晴朗。
我觉得 开心 很重要。
请帮我理解 这个问题。
```
**函数映射**  

将函数作为模板变量而不是固定值传递。 
```python
from llama_index.core import PromptTemplate

# 定义模板字符串，包含需要填充的变量
qa_prompt_tmpl_str = (
    "We have provided context information below:\n\n"
    "{context_str}\n\n"
    "Given this information, please answer the question: {query_str}\n"
)

# 定义格式化函数，用于处理 context_str 变量
def format_context_fn(**kwargs):
    context_list = kwargs["context_str"].split("\n\n")
    fmtted_context = "\n\n".join([f"- {c}" for c in context_list])
    return fmtted_context

# 创建 PromptTemplate 对象，并传入格式化函数作为 function_mappings 参数
prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, function_mappings={"context_str": format_context_fn}
)

try:
    # 使用 format 方法填充模板变量，并生成最终的提示文本
    final_prompt = prompt_tmpl.format(context_str="Context 1\n\nContext 2", query_str="What is the answer?")
    # 输出最终的提示文本
    print(final_prompt)
except KeyError as e:
    print(f"发生了 KeyError 错误: {e}")
```
输出：
```
We have provided context information below:

- Context 1

- Context 2

Given this information, please answer the question: What is the answer?
```
# Loading
LlamaIndex 中数据摄取的关键是加载和转换。加载文档后，您可以通过转换和输出节点来处理它们。
## Documents and Nodes
在LlamaIndex中，Documents 和 Nodes 是两个关键的概念，Documents 表示实际的文本数据或文档对象，而 Nodes 则是索引结构中用来管理和组织这些文档数据的基本构建块。它们在索引和查询过程中起着重要作用： 

**1.Documents**:  
- 在 LlamaIndex 中，Documents 指的是索引中存储的实际文本数据或者文档对象。
- 每个 Document 包含了一篇文档的内容和相关的元数据信息，如标题、作者、创建时间等。
- 文档可以是各种形式的文本数据，例如文章、报告、新闻稿等，这些文档会被索引以便于后续的查询和检索。

**2.Nodes**:
- 在 LlamaIndex 的上下文中，Nodes 通常指的是索引结构中的节点，特别是在树形或图形数据结构中。
- 索引可以使用不同的数据结构来组织和管理文档数据，其中节点（Nodes）是组成这些结构的基本单元。
- 例如，如果索引使用了倒排索引（Inverted Index）或者是基于 B 树的结构，那么节点就代表这些数据结构中的一个单元，用来存储文档信息或者索引键值对。  

### Usage Pattern
**Documents**
```python
from llama_index.core import Document, VectorStoreIndex

text_list = [text1, text2, ...]
documents = [Document(text=t) for t in text_list]

# build index
index = VectorStoreIndex.from_documents(documents)
```
**Nodes**
```python
from llama_index.core.node_parser import SentenceSplitter

# load documents
...

# parse nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# build index
index = VectorStoreIndex(nodes)
```
>`SentenceSplitter` 是用于分割文本成句的工具组件
## SimpleDirectoryReader
`SimpleDirectoryReader` 是 LlamaIndex 中用于从指定目录加载文档数据的简单文件读取器  
### 功能
- **加载数据**：从指定的目录中加载文档数据，可以是文本文-件、HTML 文件或其他格式的文档。
- **预处理**：可选地进行数据预处理，例如文本清洗、格式转换等，以便后续索引或查询处理。
- **灵活性**：支持不同格式的文档数据加载，可以根据需要进行定制和扩展。  

### 用法
最基本的用法是传递一个 `directory_path` 它将加载该目录中所有受支持的文件。
```python
from llama_index.core import SimpleDirectoryReader

# Specify the directory containing documents
directory_path = "path/to/your/documents"

# Create a SimpleDirectoryReader instance and load documents
reader = SimpleDirectoryReader(directory_path)
documents = reader.load_data()

# Process or index the loaded documents
# Example: index = SummaryIndex.from_documents(documents)
```
**从子目录中读取**  

默认情况下， `SimpleDirectoryReader` 将仅读取目录顶层中的文件。要从子目录中读取，请将 `recursive=True`
```python
SimpleDirectoryReader(input_dir="path/to/directory", recursive=True)
```
**在文件加载时迭代文件**  

您还可以使用 `iter_data()` 在文件加载时循环访问和处理文件的方法
```python
reader = SimpleDirectoryReader(input_dir="path/to/directory", recursive=True)
all_docs = []
for docs in reader.iter_data():
    # <do something with the documents per file>
    all_docs.extend(docs)
```
**限制加载的文件**  

您可以传递文件路径列表，而不是所有文件
```python
SimpleDirectoryReader(input_files=["path/to/file1", "path/to/file2"])
```
或者，您可以使用 URL 传递要排除的文件路径列表 **exclude** `exclude`： 
```python
SimpleDirectoryReader(
    input_dir="path/to/directory", exclude=["path/to/file1", "path/to/file2"]
)
```
您还可以设置 `required_exts` 添加到文件扩展名列表以仅加载具有这些扩展名的文件：
```python
SimpleDirectoryReader(
    input_dir="path/to/directory", required_exts=[".pdf", ".docx"]
)
```
您可以设置要加载的最大文件数 `num_files_limit`
```python
SimpleDirectoryReader(input_dir="path/to/directory", num_files_limit=100)
```
**指定文件编码**  

`SimpleDirectoryReader` 期望文件为 `utf-8` 已编码，但您可以使用 `encoding` 参数：
```python
SimpleDirectoryReader(input_dir="path/to/directory", encoding="latin-1")
```  
**提取元数据**  

您可以指定一个函数，该函数将读取每个文件并提取附加到结果的元数据 `Document` 对象，通过将函数作为 `file_metadata`:
```python
def get_meta(file_path):
    return {"foo": "bar", "file_path": file_path}

SimpleDirectoryReader(input_dir="path/to/directory", file_metadata=get_meta)
```
该函数应接受单个参数（文件路径），并返回元数据字典  

**扩展到其他文件类型**  

您可以通过将文件扩展名与 `BaseReader` 的实例作为 `file_extractor` 参数传递给 `SimpleDirectoryReader`，来扩展其读取其他文件类型的能力。`BaseReader` 应该读取文件并返回一个文档列表。
```python
import pdfplumber
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

class PdfReader(BaseReader):
    def load_data(self, file, extra_info=None):
        documents = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                documents.append(Document(text=text, extra_info=extra_info or {}))
        return documents

# 使用SimpleDirectoryReader加载PDF文件
reader = SimpleDirectoryReader(input_dir="./pdf_files", file_extractor={".pdf": PdfReader()})
documents = reader.load_data()

# 打印文档内容
for doc in documents:
    print(doc.text)
```
## Data Connectors
数据连接器（aka `Reader`）从不同的数据源和数据格式中提取数据，将其转换为简单的`Document`表示形式（文本和简单元数据）
### Usage Pattern
每个数据加载器都包含一个显示如何使用该加载器的“用法”部分。在每个加载器的使用核心是一个`download_loader`函数，它将加载器文件下载到一个模块中，您可以在应用程序中使用该模块。

**Example**:使用 LlamaIndex 框架从 Google 文档中加载指定的文档数据，并将这些文档转换为文档向量存储索引
```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.google import GoogleDocsReader

# 假设要查询的 Google 文档 ID 列表
gdoc_ids = ["1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec"]

# 创建 GoogleDocsReader 实例，并加载数据
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)

# 创建文档向量存储索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 执行查询
query_result = query_engine.query("Where did the author go to school?")

# 打印查询结果
print("Query Result:")
for result in query_result:
    print(result)
```
## Node Parsers/Text Splitters
### Node Parser Usage Pattern

节点解析器是一种简单的抽象，它接收文档列表，并将它们分块成节点对象，使得每个节点都是父文档的特定片段。当文档被分解为节点时，所有属性都会继承给子节点（i.e. `metadata`, text and metadata templates, etc.）  

**Standalone Usage**  

Node parsers can be used on their own:
```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 定义要分割成长节点的文本。
text = "这是一段很长的文字，我们将把它分割成更小的部分以便处理。" * 100  # 重复句子来创建一段较长的文本。

# 初始化SentenceSplitter，并指定所需的块大小和重叠。
node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# 将文档分割成节点。
nodes = node_parser.get_nodes_from_documents(
    [Document(text=text)],  # 将文本作为Document对象传递。
    show_progress=False      # 禁用进度条以简化输出。
)

# 打印出关于生成节点的一些信息。
for i, node in enumerate(nodes):
    print(f"节点 {i + 1}:")
    print(node.get_text())
    print("-" * 50)
```
>`SentenceSplitter`用于将长文本分割成较小的块（称为节点），通常这些块是基于句子边界进行分割的。这样做的目的是为了更好地处理和理解文本，尤其是在构建索引和执行问答系统时。  

**Transformation Usage**  

节点分析器可以包含在具有引入管道的任何一组转换中
```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

# 加载目录中的所有文本文件
documents = SimpleDirectoryReader("./data").load_data()

# 定义节点解析器
node_parser = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)

# 创建数据摄入管道
pipeline = IngestionPipeline(transformations=[node_parser])

# 运行管道以处理文档
nodes = pipeline.run(documents=documents)

# 打印分割后的节点信息
for i, node in enumerate(nodes):
    print(f"节点 {i + 1}:")
    print(node.get_text())
    print("-" * 50)
```
>`IngestionPipeline` 在 `llama_index` 库中是一个非常有用的工具，它负责处理和转换文档，使其适合后续的自然语言处理任务，如构建索引、回答问题等。`IngestionPipeline` 可以包含一系列的转换步骤，这些步骤可以是预处理、文本分割、提取关键信息等操作。

**Index Usage**  

`Index`用于存储和组织文档中的信息，使得能够高效地进行查询和检索。
```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# 加载目录中的所有文本文件
documents = SimpleDirectoryReader("data").load_data()

# 使用 SentenceSplitter 分割文本
# 注意：这里我们可以选择使用全局设置或每个索引设置，下面的例子中我们采用每索引设置的方式

# 构建向量存储索引，并指定 SentenceSplitter 作为转换
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)],
)

# 创建查询引擎
query_engine = index.as_query_engine()

# 查询索引以获取相关信息
response = query_engine.query("这段文字的主要内容是什么？")

print(response)
```
### Node Parser Modules
[基于文件的节点解析器](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#file-based-node-parsers)
- SimpleFileNodeParser
- HTMLNodeParser
- JSONNodeParser
- MarkdownNodeParser  
- LangchainNodeParser
- SentenceWindowNodeParser
- SemanticSplitterNodeParser

[Text-Splitters](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#text-splitters)  
- CodeSplitter
- SentenceSplitter
- TokenTextSplitter
## Ingestion Pipeline
一个 `IngestionPipeline` 就像是一个处理数据的工厂。在这个工厂里，有一些步骤叫做“转换”，它们负责改变数据的样子。

当你把原始数据送进这个工厂时，这些转换会按照一定的顺序来加工数据。加工好的数据可以被直接输出或者保存到一个特殊的仓库（我们这里称它为向量数据库）里。

为了提高效率，工厂会记住每次加工的具体步骤和结果。这样，如果下次有同样的数据进来，工厂就不需要重新做一遍加工过程了，而是直接从记忆中拿出之前的结果来用，这样就能节省很多时间。  

`IngestionPipeline`简单的使用示例前面已经给出。

### Caching
在 `IngestionPipeline`中，每个节点加上转换的组合都会被哈希并缓存。这在使用相同数据的后续运行中节省了时间。 

**Local Cache Management**  
Once you have a pipeline, you may want to store and load the cache.
```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

# 创建示例文档
example_doc = Document(text="This is an example document for testing purposes.")

# 定义转换步骤
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
    ]
)

# 执行数据摄取
nodes = pipeline.run(documents=[example_doc])

# 保存管道的状态到文件系统
pipeline.persist("data")

# 加载并恢复状态
new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
    ],
)
new_pipeline.load("data")

# 由于缓存的存在，此操作将立即运行
cached_nodes = new_pipeline.run(documents=[example_doc])

# 输出节点列表
for node in cached_nodes:
    print(node)
```
If the cache becomes too large, you can clear it
```python
# delete all context of the cache
cache.clear()
```
**Remote Cache Management**  

在 `IngestionPipeline` 中处理远程缓存管理涉及将处理过的数据节点及其转换结果缓存在远程位置，以便在未来的处理中可以复用这些结果。这有助于减少重复工作，特别是在处理大量重复数据时。远程缓存可以存储在云服务中，例如数据库、对象存储服务或其他分布式缓存系统。  
We support multiple remote storage backends for caches
- `RedisCache`
- `MongoDBCache`
- `FirestoreCache`  

下面是一个示例代码，展示如何使用 `Redis` 作为远程缓存来管理 `IngestionPipeline` 中的缓存。
```python
import redis
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

# 初始化 Redis 客户端
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 定义转换步骤
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
    ]
)

# 创建示例文档
example_doc = Document(text="This is an example document for testing purposes.")

# 定义缓存键前缀
cache_prefix = "ingestion_pipeline_cache_"

def get_cache_key(node, transformation):
    # 生成缓存键
    return f"{cache_prefix}{hash(node)}_{hash(transformation)}"

def is_cached(node, transformation):
    # 检查数据是否已经被缓存
    cache_key = get_cache_key(node, transformation)
    return bool(redis_client.exists(cache_key))

def cache_result(node, transformation, result):
    # 缓存处理结果
    cache_key = get_cache_key(node, transformation)
    redis_client.set(cache_key, result)

def run_pipeline_with_remote_cache(documents):
    # 运行管道，使用远程缓存
    processed_nodes = []
    for doc in documents:
        if is_cached(doc, pipeline.transformations[0]):
            # 从缓存中获取结果
            cached_node = redis_client.get(get_cache_key(doc, pipeline.transformations[0]))
            processed_nodes.append(cached_node)
        else:
            # 如果结果不在缓存中，则运行转换
            nodes = pipeline.run(documents=[doc])
            processed_nodes.extend(nodes)
            # 缓存结果
            for node in nodes:
                cache_result(node, pipeline.transformations[0], node)
    return processed_nodes

# 执行数据摄取，使用远程缓存
nodes = run_pipeline_with_remote_cache([example_doc])

# 输出节点列表
for node in nodes:
    print(node)
```
通过这种方式，您可以有效地管理远程缓存，并在处理重复数据时利用缓存结果来提高性能。
### Async Support
Async Support是指软件系统中的一种编程模式，它允许程序在执行某个操作时不必等待该操作完成就可以继续执行其他任务。这种模式特别适合于处理耗时较长的操作，如网络请求、文件读写、数据库查询等，因为这些操作可能会阻塞程序的执行，导致程序响应变慢或无法响应用户输入。  

`IngestionPipeline` 还支持异步操作：
```python
nodes = await pipeline.arun(documents=documents)
```
完整代码：
```python
import asyncio
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

# 定义转换步骤
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
    ]
)

# 创建示例文档
example_doc = Document(text="This is an example document for testing purposes.")

# 定义异步版本的 run 方法
async def async_run_pipeline(documents):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, pipeline.run, [doc]) for doc in documents]
    results = await asyncio.gather(*tasks)
    return [result for sublist in results for result in sublist]

# 异步执行数据摄取
async def main():
    # 运行异步数据摄取
    nodes = await async_run_pipeline([example_doc])
    
    # 输出节点列表
    for node in nodes:
        print(node)

# 运行异步任务
asyncio.run(main())
```
### Document Management
文档管理是在软件系统中处理文档的过程，包括文档的创建、存储、检索、更新和删除等功能。在 `IngestionPipeline` 的上下文中，文档管理涉及处理文档流，将其转换为节点，并可能将这些节点存储在向量数据库中。  

**文档管理关键步骤**  
1. **文档加载**：加载文档，可以是从文件系统、数据库或其他数据源加载。
2. **文档处理**：将文档转换为节点，可能包括文本分割、标题提取等转换。
3. **节点存储**：将处理后的节点存储到向量数据库或其他存储系统中。
4. **文档检索**：根据查询检索相关文档或节点。
5. **文档更新**：更新文档内容或元数据。
6. **文档删除**：删除不再需要的文档或节点。  

将 `Docstore` 附加到 `IngestionPipeline` 上可以启用文档管理功能。  

使用文档的 `doc_id` 或节点的 `ref_doc_id` 作为基点，`IngestionPipeline` 将主动查找重复的文档。这样可以帮助您避免重复处理相同的文档，并确保文档及其节点的一致性和完整性。  

**Docstore的作用**
1. **文档存储**：存储文档的元数据和内容。
2. **节点到文档的映射**：维护节点与原始文档之间的映射关系。
3. **文档检索**：根据文档 ID 或节点引用检索文档。  

在 `IngestionPipeline` 中附加 `Docstore` 的工作原理涉及文档和节点的管理。`Docstore` 是一个文档存储系统，它负责存储文档的元数据和内容，并维护节点与原始文档之间的映射关系。
### Parallel Processing
在 `IngestionPipeline` 类中，`run` 方法使用 Python 的 `multiprocessing.Pool` 模块将数据处理任务分发给多个处理器以并行执行。这样可以加速数据处理过程，尤其是在处理大量数据时。

具体来说，`multiprocessing.Pool` 是一个进程池，它可以创建一组工作进程，并将任务分配给这些进程。当调用 `run` 方法时，它会将要处理的数据分成若干个小批次，然后将这些批次提交给进程池中的各个进程进行并行处理。
### Transformations
A Transformation 是一个接收节点列表作为输入，并返回节点列表的过程。实现 `Transformation` 基类的每个组件都具有同步的` __call__()` 方法定义以及异步的 `acall()` 方法定义。

**Usage Pattern**  

`Transformations`通常是与 `IngestionPipeline` 结合使用的，但它们也可以直接使用。变换组件的主要功能是从输入数据（在这里指节点列表）中提取、转换或生成新的数据，以便于进一步的处理或分析。
```python
import asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor

# 创建文档读取器
documents = SimpleDirectoryReader('data').load_data()

# 初始化 SentenceSplitter
node_parser = SentenceSplitter(chunk_size=512)

# 初始化 TitleExtractor
extractor = TitleExtractor()

# 使用 SentenceSplitter 直接处理文档
nodes = node_parser.split_text(documents)

# 使用 TitleExtractor 异步处理节点
async def process_nodes():
    # 异步调用 TitleExtractor
    transformed_nodes = await extractor.acall(nodes)
    return transformed_nodes

# 运行异步函数
transformed_nodes = asyncio.run(process_nodes())

# 输出结果
for node in transformed_nodes:
    print(f"节点标题: {node.metadata['title']}")
```
**Combining with An Index**  
`Transformations`可以传递给索引或全局设置，并将在调用索引的 `from_documents()` 或 `insert()` 方法时使用。
```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

# 定义转换器
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=128)
title_extractor = TitleExtractor(nodes=5)
qa_extractor = QuestionsAnsweredExtractor(questions=3)

# 创建索引并应用转换
documents = SimpleDirectoryReader('data').load_data()

# 创建 IngestionPipeline 并设置转换器
pipeline = IngestionPipeline(
    transformations=[text_splitter, title_extractor, qa_extractor]
)

# 使用 IngestionPipeline 处理文档
nodes = pipeline.run(documents)

# 创建索引
index = VectorStoreIndex.from_documents(nodes)

# 保存索引到磁盘
index.save_to_disk('index.json')

# 打印一些基本信息
print("索引已创建，包含的文档数量:", len(index.docstore.docs))
```
**Custom Transformations**  

在 llama_index 中，你可以创建自定义的Transformations来适应特定的需求。自定义变换可以让你实现更复杂的逻辑，或者针对特定领域的数据进行处理。  

**步骤**
1. 定义变换类：
    - 继承 `BaseTransform` 类。
    - 实现 `__call__` 方法来定义变换的逻辑。
    - 如果需要异步处理，还可以实现 `acall` 方法。

2. 使用变换：
    - 将自定义变换添加到 `IngestionPipeline`。
    - 在创建索引时使用 `IngestionPipeline`。

首先，让我们定义一个简单的自定义变换类，该类将对每个节点添加一个额外的元数据字段，例如一个标签。
```python
from typing import List
from llama_index.core.base import BaseTransform
from llama_index.core.node import Node

class CustomTagger(BaseTransform):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag

    def __call__(self, nodes: List[Node]) -> List[Node]:
        # 对每个节点添加标签
        tagged_nodes = [node.add_metadata({"tag": self.tag}) for node in nodes]
        return tagged_nodes

    async def acall(self, nodes: List[Node]) -> List[Node]:
        # 异步处理逻辑
        return [node.add_metadata({"tag": self.tag}) for node in nodes]
```
接下来，我们将创建一个索引，并使用自定义变换：
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor

# 定义转换器
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=128)
title_extractor = TitleExtractor(nodes=5)
custom_tagger = CustomTagger(tag="example")

# 创建 IngestionPipeline 并设置转换器
pipeline = IngestionPipeline(
    transformations=[text_splitter, title_extractor, custom_tagger]
)

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 使用 IngestionPipeline 处理文档
nodes = pipeline.run(documents)

# 创建索引
index = VectorStoreIndex.from_documents(nodes)

# 保存索引到磁盘
index.save_to_disk('index.json')

# 打印一些基本信息
print("索引已创建，包含的文档数量:", len(index.docstore.docs))
```
以上示例展示了如何创建一个简单的自定义变换，并将其集成到 `llama_index` 的索引创建流程中。你可以根据自己的需求扩展这个示例，实现更复杂的变换逻辑。
# Indexing
## Index Guide
### How Each Index Works
**1.Vector Store Index**
- **工作原理**：
    - **嵌入式向量**：将文档内容转换为向量表示。
    - **相似度搜索**：使用向量数据库（如 Pinecone、Qdrant 或 ChromaDB）来存储这些向量，并允许基于向量相似度进行查询。
    - **回答生成**：根据查询向量找到最相似的文档片段，并从中生成答案。
- **应用场景**：
    - 大量文档的近似最近邻搜索。
    - 需要快速响应时间的问答系统。
    - 文档内容的语义理解
- **查询**：查询向量存储索引涉及获取与查询最相似的前 k 个节点（top-k 最相似节点），并将这些节点传递给我们的响应合成模块。  

**2.Keyword Table Index**  
- **工作原理**：
    - **关键词提取**：从文档中提取关键词。
    - **表格构建**：构建一个关键词表，其中每个关键词指向文档中包含该关键词的部分。
    - **查询匹配**：根据查询关键词找到相关文档片段。
- **应用场景**：
    - 小型文档集合的精确匹配查询。
    - 快速查找文档中的特定关键词。
- **查询方式**：在查询时，我们从查询中提取相关的关键词，并将这些关键词与预先提取的节点关键词进行匹配以获取对应的节点。所提取的节点会被传递给我们的响应合成模块。  

**3.Tree Index**  
- **工作原理**：
    - **文档分割**：将文档分割成较小的段落或句子。
    - **树状结构**：构建一个层次结构，其中根节点包含文档的摘要，子节点包含更详细的段落。
    - **逐层细化**：查询时从根节点开始，逐步细化到更详细的节点，直到找到最相关的段落。
- **应用场景**：
    - 大型文档集合的层次化查询。
    - 需要层次结构来组织和检索文档内容。
- **查询**：查询一棵树索引涉及从根节点遍历到叶节点。默认情况下（`child_branch_factor=1`），一个查询在给定父节点的情况下选择一个子节点。如果 `child_branch_factor=2`，则每个层级的查询会选择两个子节点。  

**6.Graph Index**
- **工作原理**：
    - **图结构**：将文档分割成节点，并建立节点之间的关系。
    - **图查询**：使用图查询语言来查询文档。
- **应用场景**：
    - 需要探索节点间关系的场景。
    - 复杂的关系型数据。
- **查询**：检索工作通过使用多个子检索器并结合结果来实现。默认情况下，会使用关键词加同义词扩展以及向量检索（如果你的图已经进行了嵌入），来检索相关的三元组。  

**4.List Index**
- **工作原理**：
    - **简单列表**：将文档分割成节点，并将这些节点存储在一个简单的列表中。
    - **全文搜索**：提供全文搜索能力，但没有特定的索引结构。
- **应用场景**：
    - 小型文档集合的简单全文搜索。
    - 不需要复杂索引结构的场景。
- **查询**：在查询时，如果没有指定其他查询参数，LlamaIndex 将简单地加载列表中的所有节点到我们的响应合成模块中。

### Vector Store Index
`VectorStoreIndex` 主要用于处理非结构化的文本数据，通过将文本转换为向量表示，从而实现高效的相似性搜索。  

**步骤**
1. **加载文档**：
    - 使用 `SimpleDirectoryReader` 或其他方式加载你的文档。

2. **构建索引**：
    - 创建 `VectorStoreIndex` 实例，并传入文档。
    - 索引构建过程将把文档分割成较小的段落或句子，并为这些段落创建向量表示。

3. **查询索引**：
    - 使用 `as_query_engine` 方法创建查询引擎。
    - 查询引擎允许进行基于向量相似性的查询。
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.query_engine import RetrieverQueryEngine

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 构建 VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# 保存索引到磁盘
index.save_to_disk('index.json')

# 查询索引
query_engine = index.as_query_engine()

# 查询示例
response = query_engine.query("What is the main idea of this document?")
print(response)
```
### Using a Property Graph Index
Property Graph Index 是一种专门用于处理图数据的索引类型，在 `llama_index` 中用于存储和查询具有丰富属性和关系的数据结构。这种索引非常适合处理那些包含节点（代表实体）和边（代表实体之间的关系）的图数据，并且每个节点和边都可以拥有多个属性。

**Usage**  

Basic usage can be found by simply importing the class and using it:
```python
from llama_index import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 创建图存储
graph_store = SimpleGraphStore()

# 创建存储上下文
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# 构建 Property Graph Index
index = PropertyGraphIndex.from_documents(
    documents, storage_context=storage_context
)

# 保存索引到磁盘
index.save_to_disk('index.json')

# 查询索引
query_engine = index.as_query_engine()
response = query_engine.query("Find all entities related to 'entity_name'")
print(response)
```
**Construction**  

在 LlamaIndex 中构建属性图 (Property Graph) 的过程涉及以下几个关键步骤：  

**1.知识图谱提取器** (kg_extractors)：  
- **定义**：知识图谱提取器是一系列的工具或函数，用于从文本中提取实体和它们之间的关系。
- **工作方式**：在构建属性图索引时，会对每个文档块 (chunk) 应用这些提取器。
- **结果**：提取出的实体和关系会被作为元数据附加到每个 LlamaIndex 节点上。

**2.节点和元数据**：
- **节点**：在属性图中，每个文档块被视为一个节点。
- **元数据**：每个节点可以携带元数据，这些元数据可以是实体和它们之间的关系。

**3.使用多个提取器**：
- **灵活性**：你可以使用任意多个知识图谱提取器。
- 应用：所有的提取器都会被应用到每个文档块上。

**4.与数据摄入管道的兼容性**：
- **数据摄入管道**：LlamaIndex 提供了一个数据摄入管道 (ingestion pipeline)，用于处理文档，例如使用变换或元数据提取器。
- **兼容性**：知识图谱提取器与数据摄入管道兼容，意味着你可以在数据摄入过程中使用这些提取器。  

**示例说明**

假设你有一个文档集合，你想从中提取实体和它们之间的关系，并构建一个属性图索引来表示这些关系。你可以使用 LlamaIndex 的`kg_extractors`来实现这一目标。

**1.定义知识图谱提取器**：
- 你可以定义多个知识图谱提取器，每个提取器负责提取不同类型的实体和关系。

**2.构建属性图索引**：
- 在构建属性图索引的过程中，每个文档块都会经过一系列的知识图谱提取器。
- 提取出的实体和关系将作为元数据附加到每个节点上。

**3.使用数据摄入管道**：
- 如果你已经使用了数据摄入管道来处理文档，那么知识图谱提取器可以很容易地集成到这个流程中。  

下面是一个详细的示例代码，展示如何使用 `LlamaIndex` 构建 `Property Graph Index`。在这个示例中，我们将使用知识图谱提取器 (`kg_extractors`)` 来提取实体和关系，并将它们作为元数据附加到每个节点上。
```python
from llama_index import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.core.extractors import EntityExtractor, RelationExtractor

# 定义知识图谱提取器
entity_extractor = EntityExtractor()
relation_extractor = RelationExtractor()

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 创建图存储
graph_store = SimpleGraphStore()

# 创建存储上下文
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# 构建 Property Graph Index
index = PropertyGraphIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    kg_extractors=[entity_extractor, relation_extractor]
)

# 保存索引到磁盘
index.save_to_disk('index.json')
```

**示例文档结构**
```json
{
  "nodes": [
    {"id": "1", "name": "Alice", "type": "Person"},
    {"id": "2", "name": "Bob", "type": "Person"},
    {"id": "3", "name": "Company X", "type": "Organization"}
  ],
  "edges": [
    {"source": "1", "target": "2", "relationship": "FRIENDS_WITH"},
    {"source": "2", "target": "3", "relationship": "EMPLOYED_AT"}
  ]
}
```
这个 JSON 文件描述了两个人员实体和一个组织实体，以及它们之间的关系。通过使用知识图谱提取器，你可以将这些实体和关系提取出来，并附加到每个节点上作为元数据。
## Document Management
使用`VectorStoreIndex`对文档进行insertion, deletion, update, and refresh 操作

**insertion**
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 构建 VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# 新的文档
new_documents = SimpleDirectoryReader('path/to/new/documents').load_data()

# 将新文档插入索引
index.insert(new_documents)
```
**deletion**
```python
# 删除文档
# 假设我们知道要删除的文档 ID
doc_id = "document-id"
index.delete(doc_id)
```

**update**
```python
# 删除旧文档
old_doc_id = "old-document-id"
index.delete(old_doc_id)

# 新文档
updated_documents = SimpleDirectoryReader('path/to/updated/documents').load_data()

# 将新文档插入索引
index.insert(updated_documents)
```

**refresh**
```python
# 重新加载所有文档
all_documents = SimpleDirectoryReader('path/to/all/documents').load_data()

# 重新构建索引
index = VectorStoreIndex.from_documents(all_documents)
```
## Metadata Extraction
在 LlamaIndex 中，Metadata Extraction是一种重要的功能，它可以从文档中抽取有用的信息，并将其作为元数据附加到文档或索引中的节点上。元数据可以用来增强索引的能力，使其能够支持更复杂的查询和数据分析。  

**元数据的作用**

- **提高查询精度**：通过使用元数据，索引可以更好地理解文档的内容，从而提高查询结果的相关性和准确性。
- **支持复杂查询**：元数据使得索引能够支持基于元数据字段的查询，例如按日期排序、按作者筛选等。
- **增强索引功能**：元数据可以用于构建更高级的索引结构，例如属性图索引（`Property Graph Index`）。

在 LlamaIndex 中，可以使用内置的提取器来进行文档转换，这些提取器可以帮助从文档中提取有用的元数据和其他信息。以下是一个完整的示例代码，展示了如何使用这些内置提取器，并通过 `IngestionPipeline` 运行这些转换。
```python
from llama_index import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.ingestion import IngestionPipeline

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 定义转换器列表
transformations = [
    SentenceSplitter(),  # 分句器
    TitleExtractor(nodes=5),  # 提取标题
    QuestionsAnsweredExtractor(questions=3),  # 提取可以回答的问题
    SummaryExtractor(summaries=["prev", "self"]),  # 提取摘要
    KeywordExtractor(keywords=10),  # 提取关键词
    EntityExtractor(prediction_threshold=0.5),  # 实体提取
]

# 创建 IngestionPipeline 实例
pipeline = IngestionPipeline(transformations=transformations)

# 运行 IngestionPipeline
nodes = pipeline.run(documents=documents)

# 打印转换后的节点
for node in nodes:
    print(node.get_content())
    print(f"Metadata: {node.metadata}")
    print("-----")
```
### Custom Extractors
在 LlamaIndex 中，Custom Extractors允许开发者根据自己的需求定义特定的逻辑来提取文档中的元数据或其他信息。这些自定义提取器可以应用于节点解析器，从而在构建索引的过程中自动提取并附加这些元数据到每个节点上。
```python
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    IngestionPipeline,
    SimpleNodeParser,
    StorageContext,
    SimpleVectorStore
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

# 定义自定义元数据提取器
def custom_date_extractor(text):
    # 示例：从文本中提取日期
    if "2024" in text:
        return {"date": "2024-08-01"}
    return {}

def custom_author_extractor(text):
    # 示例：从文本中提取作者
    if "John Doe" in text:
        return {"author": "John Doe"}
    return {}

# 创建自定义元数据提取器实例
metadata_extractor = EntityExtractor(prediction_threshold=0.5)

# 创建节点解析器
node_parser = SimpleNodeParser(metadata_extractors=[
    custom_date_extractor,
    custom_author_extractor,
    metadata_extractor
])

# 创建向量存储
vector_store = SimpleVectorStore()

# 创建存储上下文
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 创建 Ingestion Pipeline
pipeline = IngestionPipeline(
    document_loaders=[SimpleDirectoryReader('path/to/your/documents')],
    node_parsers=[node_parser],
    index_constructors=[VectorStoreIndex],
    storage_context=storage_context
)

# 运行 Ingestion Pipeline
pipeline.ingest()

# 查询索引
index = pipeline.indexes[0]
query_engine = index.as_query_engine()
response = query_engine.query("What is the main idea of this document?")
print(response)
```
# Storing
在 LlamaIndex 中，一旦构建了索引并完成了元数据的提取，通常需要将索引和相关的元数据持久化存储起来。这有助于避免每次重新加载和处理文档时重复工作，并且可以在后续查询中快速访问索引。  

例如上节代码：
```python
# 创建向量存储
vector_store = SimpleVectorStore()

# 创建存储上下文
storage_context = StorageContext.from_defaults(vector_store=vector_store)
```
## Vector Stores
在 LlamaIndex 中，Vector Store 用于存储和检索文本片段的嵌入（embedding）。这些嵌入被用来执行相似性搜索，这对于语义搜索和检索任务至关重要。向量存储允许您有效地存储和查询大量的嵌入，使其成为索引过程中的关键组成部分

## Document Stores
在 LlamaIndex 中，Document Store 用于存储原始文档及其相关元数据。文档存储对于持久化和管理索引中的文档至关重要。
### Simple Document Store
默认情况下，`SimpleDocumentStore` 在内存中存储 `Node` 对象。可以通过调用 `docstore.persist()` 将它们持久化到磁盘（并通过调用 `SimpleDocumentStore`.`from_persist_path(...)` 从磁盘加载）  

**持久化文档节点**  

**1.保存文档节点到磁盘**：
- 调用 `docstore.persist()` 方法将文档节点保存到磁盘上。
- 这个方法通常在索引构建完成后调用，以确保索引中的文档节点不会因为程序重启而丢失。

**2.从磁盘加载文档节点**：
- 使用 `SimpleDocumentStore.from_persist_path(...)` 方法从磁盘上加载文档节点。
- 这个方法允许你在下次启动程序时恢复之前保存的文档节点状态。

**示例代码：**
```python
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    IngestionPipeline,
    SimpleNodeParser,
    StorageContext,
    SimpleDocumentStore,
    SimpleVectorStore
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

# 定义自定义元数据提取器
def custom_date_extractor(text):
    # 示例：从文本中提取日期
    if "2023" in text:
        return {"date": "2023-01-01"}
    return {}

def custom_author_extractor(text):
    # 示例：从文本中提取作者
    if "John Doe" in text:
        return {"author": "John Doe"}
    return {}

# 创建自定义元数据提取器实例
metadata_extractor = EntityExtractor(prediction_threshold=0.5)

# 创建节点解析器
node_parser = SimpleNodeParser(metadata_extractors=[
    custom_date_extractor,
    custom_author_extractor,
    metadata_extractor
])

# 创建文档存储
document_store = SimpleDocumentStore()

# 创建存储上下文
storage_context = StorageContext.from_defaults(document_store=document_store)

# 创建 Ingestion Pipeline
pipeline = IngestionPipeline(
    document_loaders=[SimpleDirectoryReader('path/to/your/documents')],
    node_parsers=[node_parser],
    index_constructors=[VectorStoreIndex],
    storage_context=storage_context
)

# 运行 Ingestion Pipeline
pipeline.ingest()

# 保存文档节点到磁盘
document_store.persist('docstore.json')

# 保存索引到磁盘
index = pipeline.indexes[0]
index.save_to_disk('index.json')

# 从磁盘加载文档节点
loaded_docstore = SimpleDocumentStore.from_persist_path('docstore.json')

# 从磁盘加载索引
loaded_index = VectorStoreIndex.load_from_disk('index.json', storage_context=StorageContext.from_defaults(document_store=loaded_docstore))
query_engine = loaded_index.as_query_engine()
response = query_engine.query("What is the main idea of this document?")
print(response)
```
通过使用文档存储的持久化功能，您可以确保即使在程序重启后也能保留索引中的文档节点状态。

## Index Stores
在 LlamaIndex 中，Index Store是用于存储和管理索引结构的组件。索引存储可以存储不同类型的索引，如向量索引、关键词索引等。这些索引用于加速文档的检索过程，使得用户可以快速地找到相关信息。

### Simple Index Store
默认情况下，LlamaIndex 使用一个由内存中的键值对存储支持的简单索引存储。可以通过调用 `index_store.persist()` 将它们持久化到磁盘（并通过调用 `SimpleIndexStore.from_persist_path(...)` 从磁盘加载）。  

在 LlamaIndex 中，默认的索引存储实现是 `SimpleIndexStore`，它使用内存中的键值对存储来保存索引的信息。为了确保索引数据的持久性，可以将索引存储中的数据持久化到磁盘上，并在需要时从磁盘上加载回来。
## Chat Stores
在 LlamaIndex 中，Chat Store 并不是直接提供的核心组件之一，但它可以被视为一种概念性的存储方案，用于记录和管理聊天会话的历史记录。虽然 LlamaIndex 主要关注于文档索引和检索，但在构建聊天应用时，跟踪和利用聊天历史记录对于提供上下文相关且连贯的回答是非常有用的。
### SimpleChatStore
在 LlamaIndex 中，`SimpleChatStore` 是一个用于存储聊天历史记录的简单实现。它可以帮助你跟踪聊天会话，这对于构建连续且有意义的对话非常重要。`SimpleChatStore` 默认使用内存中的数据结构来存储消息，但也可以将其持久化到磁盘上。

**示例代码：**
```python
from llama_index import SimpleChatStore

# 创建 SimpleChatStore 实例
chat_store = SimpleChatStore()

# 添加消息
chat_store.add_message("user1", "Hello!")
chat_store.add_message("user1", "How are you?")
chat_store.add_message("user1", "I'm good, thank you.")

# 获取消息
messages = chat_store.get_messages("user1")
print(messages)

# 保存聊天记录到磁盘
chat_store.persist('chatstore.json')

# 从磁盘加载聊天记录
loaded_chat_store = SimpleChatStore.from_persist_path('chatstore.json')

# 使用加载的聊天记录
loaded_messages = loaded_chat_store.get_messages("user1")
print(loaded_messages)
```

## Key-Value Stores
在 LlamaIndex 中，Key-Value Store 是一种简单而有效的数据存储方式，它使用键值对的形式来存储数据。键值存储在 LlamaIndex 的各个组件中扮演着重要的角色，尤其是在存储索引、文档和元数据等方面。键值存储提供了一种灵活的方式来存储和检索数据，特别适合于那些不需要复杂查询的数据集。  

下面是一个使用 `SimpleKeyValueStore` 的示例，展示了如何使用键值存储来存储和检索数据：
```python
from llama_index import SimpleKeyValueStore

# 创建 SimpleKeyValueStore 实例
kv_store = SimpleKeyValueStore()

# 添加键值对
kv_store.set("key1", "value1")
kv_store.set("key2", "value2")

# 获取值
value1 = kv_store.get("key1")
print(value1)  # 输出: value1

# 检查键是否存在
exists = kv_store.contains("key1")
print(exists)  # 输出: True

# 删除键值对
kv_store.delete("key1")

# 保存键值存储到磁盘
kv_store.persist('kvstore.json')

# 从磁盘加载键值存储
loaded_kv_store = SimpleKeyValueStore.from_persist_path('kvstore.json')

# 使用加载的键值存储
loaded_value = loaded_kv_store.get("key2")
print(loaded_value)  # 输出: value2
```

## Persisting & Loading Data
在 LlamaIndex 中，持久化和加载数据涉及到将索引和检索过程中的各种组件保存到磁盘上，以便以后能够恢复这些组件。这包括保存文档存储、索引存储以及其他相关组件，如聊天存储或键值存储。

# Querying
## Query Engine
查询引擎是一个通用的接口，允许您使用自然语言查询您的数据。查询引擎接收自然语言查询，并返回丰富的响应。它最常基于一个或多个索引构建而成，通过retrievers来实现。您可以组合多个查询引擎以实现更高级的功能。

### Usage Pattern

```python
# 创建查询引擎
query_engine = index.as_query_engine()

# 执行查询
response = query_engine.query("这篇文档的主要观点是什么？")
```

### Streaming
在 LlamaIndex 中，Streaming 通常指的是查询引擎在处理查询时能够逐步返回结果的能力。这种方式可以让用户在等待完整答案的同时看到部分结果，这对于长时间运行的查询尤其有用。

**如何启用流式传输**  
要使用低级 API 配置查询引擎以启用流式传输，您需要在构建`ResponseSynthesizer` 时传递 `streaming=True` 参数。下面是一个具体的示例代码：
```python
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorStoreRetriever
from llama_index.response_synthesizers import ResponseSynthesizer
from llama_index.response.schema import StreamingResponse

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 构建索引
service_context = ServiceContext.from_defaults()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 保存索引到磁盘
index.storage_context.persist(persist_dir='storage/index')

# 创建检索器
retriever = VectorStoreRetriever(
    storage_context=index.storage_context,
    similarity_top_k=2
)

# 创建响应合成器并启用流式传输
response_synthesizer = ResponseSynthesizer.from_args(
    service_context=service_context,
    streaming=True
)

# 创建查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

# 执行查询
query_str = "这篇文档的主要观点是什么？"
response = query_engine.query(query_str)

# 处理流式响应
if isinstance(response, StreamingResponse):
    for chunk in response.response_gen:
        print(chunk, end='')
else:
    print(response)
```
`StreamingResponse` 是一种特殊的响应类型，用于支持流式传输功能。当启用流式传输时，查询引擎可以在处理查询的过程中逐步返回结果，而不是等到整个处理过程完成后再返回所有结果。  

在执行查询并收到响应后，检查响应类型是否为 `StreamingResponse`。  
如果是 `StreamingResponse`，则可以迭代访问 `response_gen` 属性来逐步处理和显示结果。

## Chat Engine
构建一个Chat Engine可以让您创建一个交互式的聊天应用，用户可以通过自然语言与系统进行对话。聊天引擎通常结合了多种技术，包括检索、问答、对话管理和流式传输等特性。

### Usage Pattern
在 LlamaIndex 中，您可以直接从已构建的索引创建一个聊天引擎，这使得与索引中的数据进行交互变得更加简单。以下是使用 `as_chat_engine()` 方法构建聊天引擎并与其进行交互的示例：
```python
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
)

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 构建索引
service_context = ServiceContext.from_defaults()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 保存索引到磁盘
index.storage_context.persist(persist_dir='storage/index')

# 从索引构建聊天引擎
chat_engine = index.as_chat_engine()

# 与聊天引擎进行对话
response = chat_engine.chat("Tell me a joke.")
print(response)

# 重置聊天历史
chat_engine.reset()

# 进入交互式聊天 REPL
chat_engine.chat_repl()
```
## Retrieval
在 LlamaIndex 中，Retriever 的主要职责是从索引中找到与查询最相关的文档片段，这些片段随后会被用来合成最终的回答.

### Usage Pattern
您可以使用索引的 `as_retriever()` 方法来创建一个 `Retriever` 对象。这使得您可以直接从索引中检索与查询最相关的文档片段。
```python
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
)

# 加载文档
documents = SimpleDirectoryReader('path/to/your/documents').load_data()

# 构建索引
service_context = ServiceContext.from_defaults()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 保存索引到磁盘
index.storage_context.persist(persist_dir='storage/index')

# 从索引构建一个 Retriever
retriever = index.as_retriever(similarity_top_k=2)

# 使用 retriever 检索文档
nodes = retriever.retrieve("Who is Paul Graham?")

# 打印检索到的文档
for node in nodes:
    print(node.get_text())
```
## Node Postprocessor
`Node Postprocessor` 是一种用于处理从索引中检索到的文档节点的组件。它的主要作用是在检索到的节点被用于合成最终回答之前对其进行修改或过滤。这可以包括但不限于删除不相关的信息、合并相似的节点、提取关键信息等。

### Usage Pattern
在 LlamaIndex 中，`SimilarityPostprocessor` 和 CohereRerank`` 是两种不同的 `Node Postprocessor`，它们分别用于过滤低相似度的节点和重新排序节点以提高查询结果的质量。
```python
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore

nodes = [
    NodeWithScore(node=Node(text="text1"), score=0.7),
    NodeWithScore(node=Node(text="text2"), score=0.8),
]

# similarity postprocessor: filter nodes below 0.75 similarity score
processor = SimilarityPostprocessor(similarity_cutoff=0.75)
filtered_nodes = processor.postprocess_nodes(nodes)

# cohere rerank: rerank nodes given query using trained model
reranker = CohereRerank(api_key="<COHERE_API_KEY>", top_n=2)
reranked_nodes = reranker.postprocess_nodes(nodes, query_str="What is the main topic?")

# 打印过滤后的节点
for node in filtered_nodes:
    print(node.node.text)

# 打印重新排序后的节点
for node in reranked_nodes:
    print(node.node.text)
```
>`NodeWithScore` 是一个数据结构，用于表示从索引中检索到的文档节点及其相关性评分。当您从索引中检索文档片段时，`NodeWithScore` 通常会被返回，其中包含了原始的 `Node` 对象以及一个分数，该分数反映了该节点与查询的相关程度。
### Using with a Query Engine

```python
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.core.postprocessor import TimeWeightedPostprocessor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorStoreRetriever
from llama_index.response_synthesizers import ResponseSynthesizer

# 加载文档
documents = SimpleDirectoryReader('./data').load_data()

# 构建索引
service_context = ServiceContext.from_defaults()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 创建 VectorStoreRetriever
retriever = VectorStoreRetriever(
    storage_context=index.storage_context,
    similarity_top_k=2
)

# 创建 Response Synthesizer
response_synthesizer = ResponseSynthesizer.from_args(
    service_context=service_context
)

# 创建 TimeWeightedPostprocessor
node_postprocessor = TimeWeightedPostprocessor(
    time_decay=0.5,
    time_access_refresh=False,
    top_k=1
)

# 创建 Query Engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[node_postprocessor]
)

# 执行查询
query_string = "这个文档的主要主题是什么？"
response = query_engine.query(query_string)
print(response)
```
>`TimeWeightedPostprocessor` 是一种后处理器，它可以基于节点的时间戳来调整节点的权重，从而影响查询结果的相关性排序

## Response Synthesizer
`Response Synthesizer` 用于根据检索到的文档片段（节点）生成最终的回答。它负责将多个相关文档片段整合成一个连贯且有意义的回答，通常涉及到自然语言处理技术和机器学习模型的应用。

### Usage Pattern

**在 Query Engine 中使用**
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import get_response_synthesizer

# 加载文档
documents = SimpleDirectoryReader('./data').load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 创建 `Response Synthesizer`
response_synthesizer = get_response_synthesizer(
    response_mode='compact'
)

# 创建 `Query Engine` 并指定 `Response Synthesizer`
query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)

# 执行查询
response = query_engine.query("法国的首都是哪里？")
print(response)
```
## Routing
### Routers
在 LlamaIndex 的上下文中，Routers是一组模块，它们接收用户的查询和一组“选项”（由元数据定义），并返回一个或多个选定的选项。这些路由器可以独立使用（作为“选择器模块”），也可以用作查询引擎或检索器的一部分。  

**路由器的用途**

路由器模块非常灵活且强大，它们可以用于以下用途及其他更多：

- **选择合适的数据源**：从多种数据源中选择最合适的数据源来回答用户的查询。
- **决定使用哪种查询方式**：决定是否进行摘要（例如，使用摘要索引查询引擎）或语义搜索（例如，使用向量索引查询引擎）。
- **多路路由**：决定是否同时尝试多个选项并合并结果（使用多路路由能力）。

**路由器的核心模块**

核心的路由器模块以以下形式存在：
- **LLM 选择器**：将选项作为文本堆栈放入提示中，并使用 LLM 文本完成端点来做出决策。
- **Pydantic 选择器**：将选项作为 Pydantic 模型传递给函数调用端点，并返回 Pydantic 对象。  

在这个示例中，我们将创建一个查询引擎，它使用路由器来决定是否使用摘要索引查询引擎还是向量索引查询引擎来回答查询。
```python
from llama_index import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, ServiceContext
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import SummaryIndexRetriever, VectorStoreRetriever

# 加载文档
documents = SimpleDirectoryReader('./data').load_data()

# 构建摘要索引
summary_service_context = ServiceContext.from_defaults()
summary_index = SummaryIndex.from_documents(documents, service_context=summary_service_context)

# 构建向量索引
vector_service_context = ServiceContext.from_defaults()
vector_index = VectorStoreIndex.from_documents(documents, service_context=vector_service_context)

# 创建 SummaryIndexRetriever
summary_retriever = SummaryIndexRetriever(
    summary_index,
    similarity_top_k=2
)

# 创建 VectorStoreRetriever
vector_retriever = VectorStoreRetriever(
    vector_index.storage_context,
    similarity_top_k=2
)

# 创建 SummaryIndexQueryEngine
summary_query_engine = RetrieverQueryEngine(
    retriever=summary_retriever
)

# 创建 VectorIndexQueryEngine
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever
)

# 创建工具
list_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to the data source",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context related to the data source",
)

# 创建路由器
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

# 示例使用
query = "法国的首都是哪里？"

# 使用路由器处理查询
response = query_engine.query(query)
print(f"Response: {response}")
```

## Query Pipeline
Query Pipeline是指一系列处理步骤，这些步骤组合起来形成一个流程，用于处理用户的查询并生成最终的回答。

一个典型的查询管道可能包含以下几个部分：
- **检索器（Retriever）**：用于从索引中检索相关文档或信息。
- **选择器（Selector）**：根据查询和候选选项决定使用哪个查询引擎或处理逻辑。
- **合成器（Synthesizer）**：用于生成最终的回答，可能基于检索到的信息或多个查询引擎的结果。

`QueryPipeline` 是一个用于构建和管理查询流程的高级抽象。它允许您以声明式的方式定义查询流程，包括定义各个模块及其之间的依赖关系。下面是一个使用 `QueryPipeline` 的简化示例，展示了如何构建一个简单的查询管道：
```python
from llama_index import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, ServiceContext
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import SummaryIndexRetriever, VectorStoreRetriever

# 加载文档
documents = SimpleDirectoryReader('./data').load_data()

# 构建摘要索引
summary_service_context = ServiceContext.from_defaults()
summary_index = SummaryIndex.from_documents(documents, service_context=summary_service_context)

# 构建向量索引
vector_service_context = ServiceContext.from_defaults()
vector_index = VectorStoreIndex.from_documents(documents, service_context=vector_service_context)

# 创建 SummaryIndexRetriever
summary_retriever = SummaryIndexRetriever(
    summary_index,
    similarity_top_k=2
)

# 创建 VectorStoreRetriever
vector_retriever = VectorStoreRetriever(
    vector_index.storage_context,
    similarity_top_k=2
)

# 创建 SummaryIndexQueryEngine
summary_query_engine = RetrieverQueryEngine(
    retriever=summary_retriever
)

# 创建 VectorIndexQueryEngine
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever
)

# 创建工具
list_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to the data source",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context related to the data source",
)

# 创建路由器
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

# 创建查询管道
pipeline = QueryPipeline()
pipeline.add_modules({"summary_engine": summary_query_engine, "vector_engine": vector_query_engine})
pipeline.add_modules({"selector": query_engine})

# 定义模块间的链接
pipeline.add_link("summary_engine", "selector")
pipeline.add_link("vector_engine", "selector")

# 执行查询
query = "法国的首都是哪里？"

# 使用查询管道处理查询
response = pipeline.run(query)
print(f"Response: {response}")
```
## Structured Outputs
`Structured Outputs`是指查询管道能够返回格式化的、结构化的数据，而不是简单的文本响应。这使得开发人员能够更容易地解析和利用查询结果中的信息。
### Output Parsing Modules
输出解析模块是在 LlamaIndex 这样的大型语言模型 (LLMs) 上下文中非常重要的组成部分。这些模块有助于确保 LLM 生成的响应以一种可以被下游应用程序可靠解析和利用的结构化方式进行组织。

**输出解析模块的作用**
  
    输出解析模块是专门设计用于解析 LLM 产生的文本输出，并将其转换为结构化格式的模块。这些解析器通常位于 LLM 调用之前和之后，用于准备输入或解析输出。它们帮助确保 LLM 的输出可以被应用程序可靠地解析和使用。
**输出解析模块的分类**

输出解析模块可以分为几个类别：
- **Pydantic 程序**：这些模块接收输入提示并通过 LLM 生成结构化输出。它们可以使用函数调用 API 或者文本完成 API 加上输出解析器来生成结构化的 Pydantic 对象。这些程序可以灵活地集成到查询引擎中，以便根据输入提示生成预期的结构化输出。

- **预定义的 Pydantic 程序**：这些是预定义的 Pydantic 程序，将输入映射到特定的输出类型（如数据帧）。这些程序通常已经定义好了输入和输出的结构，可以直接使用。

- **输出解析器**：这些是在 LLM 文本完成端点之前和之后运行的模块。它们不用于 LLM 函数调用端点（因为那些端点本身就包含结构化输出）。输出解析器负责解析 LLM 产生的原始文本输出，并将其转换为结构化的数据格式，比如 JSON 或者其他格式。

我们可以创建一个 `PydanticOutputParser` 来解析LLM的输出并将其转换为该 `Pydantic` 模型的实例。
```python
from pydantic import BaseModel, Field
from llama_index.output_parsers.pydantic import PydanticOutputParser

# 定义 Pydantic 模型
class CustomOutput(BaseModel):
    answer: str = Field(description="The answer to the question.")
    confidence: float = Field(description="The confidence level of the answer.", ge=0, le=1)

# 创建输出解析器
output_parser = PydanticOutputParser(output_cls=CustomOutput)

# 示例 LLM 输出
llm_output = """
{
    "answer": "The capital of France is Paris.",
    "confidence": 0.95
}
"""

# 解析 LLM 输出
parsed_output = output_parser.parse(llm_output)

# 打印解析后的输出
print(parsed_output)
```
# Agents
在 LlamaIndex 中，Agents是一种基于大型语言模型 (LLMs) 的知识工作者，它们能够智能地执行各种任务，既可以在“读取”功能中检索数据，也可以在“写入”功能中修改数据。  

代理具备以下能力：
- **执行自动化搜索和检索**：能够在不同类型的资料中执行自动化搜索和检索，包括非结构化、半结构化和结构化数据。
- **调用外部服务 API**：以结构化的方式调用任何外部服务 API，并处理响应以及存储以便后续使用。

从这个意义上说，代理超越了我们的查询引擎，因为它们不仅可以从静态数据源中“读取”，还可以动态地从各种不同的工具中获取和修改数据。

构建一个数据代理需要以下核心组件：

- **推理循环**：代理使用推理循环来决定使用哪些工具、按照什么顺序使用这些工具，以及调用每个工具的参数。
- **工具抽象**：代理被初始化时配备了一系列 API 或工具，这些工具可以被代理调用以返回信息或修改状态。

## Usage Pattern
**如何构建一个数据代理：**  
**1. 定义工具**：创建工具，每个工具都包含一个查询引擎或其他功能。  
**2. 创建推理循环**：使用推理循环来决定使用哪些工具、按照什么顺序使用这些工具，以及调用每个工具所需的参数。  
**3. 执行任务**：使用数据代理处理任务并返回结果。
```python
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.agents import DataAgent
from llama_index.tools import QueryEngineTool
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorStoreRetriever

# 加载文档
documents = SimpleDirectoryReader('./data').load_data()

# 构建向量索引
service_context = ServiceContext.from_defaults()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 创建 VectorStoreRetriever
retriever = VectorStoreRetriever(
    index.storage_context,
    similarity_top_k=2
)

# 创建查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever
)

# 创建工具
tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    description="Useful for retrieving information from the data source",
)

# 创建数据代理
agent = DataAgent(tools=[tool])

# 执行任务
task = "法国的首都是哪里？"

# 使用数据代理处理任务
response = agent.execute(task)
print(f"Response: {response}")
```
>`QueryEngineTool` 是一个工具类，它封装了一个查询引擎 (`QueryEngine`)，并提供了一种方法来将查询引擎作为一个独立的工具使用。`QueryEngineTool` 可以被集成到更大的系统中，例如数据代理 (`DataAgent`) 或者其他更复杂的查询管道中。
## Tools
在 LlamaIndex 中，Tools就像是让Agents完成特定任务的小工具箱。这些工具可以是查询引擎、API 调用、外部服务调用或其他任何可以执行特定任务的功能。工具是构建更复杂系统的基础组件，如Data Agents和其他高级查询管道。

[LlamaIndex 提供了几种不同类型的工具](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/#tools)：
- FunctionTool：函数工具允许你轻松地将任何自定义函数转换为工具。它还可以自动推断函数的模式。
- QueryEngineTool：一个封装现有查询引擎的工具。这些工具可以用于检索数据。
- ToolSpec：定义围绕单一服务的一个或多个工具的工具规范。例如，可以有一个围绕 Gmail 的工具规范。
- Utility Tools：用于封装其他工具以处理从工具返回大量数据的情况。

# Evaluation
## Evaluating
在 LlamaIndex 中，Evaluating指对Agents、Query Engines或其他组件的性能进行评估的过程。这通常涉及衡量这些组件在处理任务时的准确性和效率。

**评估的步骤**  ：

**1.定义评估目标**：明确你要评估的内容，比如准确性、响应速度等。  
**2.准备数据集**：收集或创建用于评估的数据集。  
**3.实施评估方法**：根据评估目标选择合适的评估方法。  
**4.分析结果**：分析评估结果以确定组件的性能。  
**5.迭代改进**：根据评估结果对系统进行调整和优化。  

下面是一个简化的示例，展示了如何使用 LlamaIndex 对查询引擎进行评估:
```python
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorStoreRetriever
from llama_index.evaluation.query import QueryEvaluationTool

# 加载文档
documents = SimpleDirectoryReader('./data').load_data()

# 构建向量索引
service_context = ServiceContext.from_defaults()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 创建 VectorStoreRetriever
retriever = VectorStoreRetriever(
    index.storage_context,
    similarity_top_k=2
)

# 创建查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever
)

# 创建评估工具
evaluation_tool = QueryEvaluationTool()

# 准备评估数据
evaluation_data = [
    ("法国的首都是哪里？", "巴黎"),
    ("美国的首都是哪里？", "华盛顿"),
]

# 执行评估
for query, expected_answer in evaluation_data:
    # 使用查询引擎处理查询
    response = query_engine.query(query)
    
    # 评估结果
    evaluation_result = evaluation_tool.evaluate(response, expected_answer)
    print(f"Query: {query}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Actual Answer: {response}")
    print(f"Evaluation Result: {evaluation_result}")
    print("-" * 40)
```

# Observability
`Observability`是指监控和理解系统内部运作的能力。这通常涉及到收集、记录和分析系统运行时的数据，以便更好地理解系统的行为、性能和健康状况。

**可观测性的目的**

可观测性的主要目的是提高系统的透明度，使得开发人员和运维团队能够：
- 监控系统性能。
- 发现潜在的问题。
- 诊断错误。
- 改进系统的设计和实现。

**可观测性的组成部分**

可观测性通常包括以下几个组成部分：
- 日志记录（Logging）：记录系统运行时的信息，如事件、错误和警告。
- 指标收集（Metrics Collection）：收集关键性能指标（KPIs），如响应时间、请求吞吐量、错误率等。
- 追踪（Tracing）：跟踪请求或操作在整个系统中的流动路径，以了解各个组件之间的交互情况。

## Usage Patern
**如何实现可观测性**

在 LlamaIndex 中，实现可观测性通常涉及以下几个步骤：
- **配置日志记录**：设置日志记录框架，以记录关键的事件和信息。
- **配置指标收集**：设置指标收集框架，以定期收集和报告性能指标。
- **配置追踪**：设置追踪框架，以追踪请求或操作的整个生命周期。

下面是一个简化的示例，展示了如何使用 LlamaIndex 设置基本的日志记录：
```python
import logging
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorStoreRetriever

# 配置日志记录
logging.basicConfig(level=logging.INFO)

# 加载文档
documents = SimpleDirectoryReader('./data').load_data()

# 构建向量索引
service_context = ServiceContext.from_defaults()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 创建 VectorStoreRetriever
retriever = VectorStoreRetriever(
    index.storage_context,
    similarity_top_k=2
)

# 创建查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever
)

# 执行查询
query = "法国的首都是哪里？"
response = query_engine.query(query)

# 记录日志
logging.info(f"Query: {query}")
logging.info(f"Response: {response}")
```

## Instrumentation
Using the new instrumentation module involves 3 high-level steps.
1. Define a `dispatcher`：`dispatcher`是负责分发事件和跨度的核心组件。首先，你需要定义一个`dispatcher`实例。
2. (Optional) Define and attach your `EventHandler`'s to `dispatcher`：`EventHandler`用于处理来自 LlamaIndex 的事件。你可以定义自己的事件处理器，并将其附加到`dispatcher`上。
3. (Optional) Define and attach your `SpanHandler` to `dispatcher`：`SpanHandler` 用于处理来自LlamaIndex 的跨度。你可以定义自己的`SpanHandler`，并将其附加到`dispatcher`上。


通过这些步骤，你可以处理在 LlamaIndex 库及其扩展包中传输的事件和跨度。这有助于实现可观测性，即监控和理解系统内部运作的能力。
```python
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorStoreRetriever
from llama_index import set_global_service_context
from llama_index.instrumentation import (
    TraceLogger,
    Tracer,
    Dispatcher,
    EventHandler,
    SpanHandler,
)

# 定义一个调度器
dispatcher = Dispatcher()

# 定义一个事件处理器
class MyEventHandler(EventHandler):
    def handle_event(self, event_type, event_payload, **kwargs):
        print(f"Handling event: {event_type} - {event_payload}")

# 将事件处理器附加到调度器
dispatcher.attach(MyEventHandler())

# 定义一个跨度处理器
class MySpanHandler(SpanHandler):
    def handle_span(self, span_name, span_attributes, **kwargs):
        print(f"Handling span: {span_name} - {span_attributes}")

# 将跨度处理器附加到调度器
dispatcher.attach(MySpanHandler())

# 设置全局服务上下文
service_context = ServiceContext.from_defaults(dispatcher=dispatcher)
service_context = set_global_service_context(service_context)

# 加载文档
documents = SimpleDirectoryReader('./data').load_data()

# 构建向量索引
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 创建 VectorStoreRetriever
retriever = VectorStoreRetriever(
    index.storage_context,
    similarity_top_k=2
)

# 创建查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever
)

# 执行查询
query = "法国的首都是哪里？"
with dispatcher.start_as_current_span("query"):
    response = query_engine.query(query)

# 输出结果
print(f"Response: {response}")
```

# Settings
## Configuring Settings
`Settings`是 LlamaIndex 中用于索引和查询阶段的一组常用资源。你可以用它来设置全局配置，比如设置默认的LLM或嵌入模型。如果某个组件没有特别指定，就会使用 `Settings` 中的全局默认值。简单来说，`Settings`就像是一个工具箱，里面包含了索引和查询过程中需要用到的各种默认工具和资源。

[一些属性的配置](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/)