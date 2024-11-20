# imports
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from few_shot_data import few_shot_examples
HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')




# Initialize the embedding model
embed_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize Mixtral as a llm
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_new_tokens=512,
    temperature=0.01,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Create the vector store
vector_store = InMemoryVectorStore(embedding=embed_model)

# Step 2: Index examples list in vector store
example_selector = SemanticSimilarityExampleSelector.from_examples(
    few_shot_examples,  # list of examples to index
    embed_model,  # embedding model to use
    vector_store,  # vector store to use
    k=3,  # number of examples to retrieve
    input_keys=["input"],  # keys in the examples that contain the input text
)

# Step 3: Find most relevant examples and
few_shot_prompt = FewShotPromptTemplate(
    # uses semantic similarity to retrieve examples
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\n Assistant output: {output}"
    ),
    prefix="You are a customer support AI agent. Follow this examples of how to handle different tasks:",  # add instructions to the model
    suffix="",
)


# Step 4: Add examples to user prompt
full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


# Step 5: Generate Response
user_query = "Example user inquirie"
print(f"User inquirie: {user_query}")

prompt_val = full_prompt.invoke(
    {
        "input": user_query,
        "agent_scratchpad": [],
    }
)

print(f'Most relevant prompts: {prompt_val.to_string()}')
answer = llm.invoke(prompt_val)
print(answer)