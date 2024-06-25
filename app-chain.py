import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
import os

# Define the base path
base_path = "C:\\Users\\Ce PC\\AppData\\Local\\nomic.ai\\GPT4All"
# Define the model name
model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
# Combine the base path and model name to create the full path
model_path = os.path.join(base_path, model_name)

callbacks = [StreamingStdOutCallbackHandler()]

# Initialize the model without the 'device' parameter
llm = GPT4All(model=model_path, callbacks=callbacks, verbose=True)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)

######################################## fro the notebook 
# from gpt4all import GPT4All
# import os

# Define the base path
base_path = "C:\\Users\\Ce PC\\AppData\\Local\\nomic.ai\\GPT4All"
# Define the model name
model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
# Combine the base path and model name to create the full path
model_path = os.path.join(base_path, model_name)

model1 = GPT4All(model_path)
output = model1.generate("The name of the capital of France is ", max_tokens=3)
print(output)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
llm = GPT4All(model=model_path, callbacks=callbacks, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)
########################################

st.title("🦜🔗 GPT4ALL Y'All")
st.info('This is using the MPT model!')
prompt = st.text_input('Enter your prompt here!')

if prompt:
    response = llm_chain.run(prompt)
    st.write(response)
