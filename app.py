from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


# Define the base path
base_path = "C:\\Users\\Ce PC\\AppData\\Local\\nomic.ai\\GPT4All"
# Define the model name
model_name = "mistral-7b-openorca.Q4_0.gguf"
# Combine the base path and model name to create the full path
model_path = os.path.join(base_path, model_name)

model = GPT4All(model=model_path, n_threads=8)
# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
llm = GPT4All(model=model_path, callbacks=callbacks, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

# llm_chain.run(question)

# response = model.invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?")

st.title('🦜🔗 GPT For Y\'all')

prompt = st.text_input('Enter your prompt here!')

if prompt: 
    response = llm_chain.run(prompt)
    st.write(response)

