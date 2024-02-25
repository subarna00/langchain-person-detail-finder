## integrate code with openAI aAPI

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# streamlit framework

st.title("Langchain demo with openAI")
input_text = st.text_input("Search the topic you want")

# prompot template
prompt_template = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity ${name}",
   
)
prompt_template1 = PromptTemplate(
    input_variables=["person"],
    template="When was ${person} person?"
)
prompt_template2 = PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events happened in ${dob}"
)
# OpenAI LLMS
llm = OpenAI(temperature=0.8)

# memory
person_memory=ConversationBufferMemory(input_key="name",memory_key="chat_history")
dob_memory=ConversationBufferMemory(input_key="person",memory_key="chat_history")
events_memory=ConversationBufferMemory(input_key="dob",memory_key="chat_history")

# creating chains
chain = LLMChain(llm=llm,prompt=prompt_template,verbose=True, output_key="person",memory=person_memory)
chain1 = LLMChain(llm=llm,prompt=prompt_template1,verbose=True, output_key="dob",memory=dob_memory)
chain2 = LLMChain(llm=llm,prompt=prompt_template2,verbose=True, output_key="events",memory=events_memory)


parent_chain = SequentialChain(chains=[chain,chain1,chain2],input_variables=["name"],output_variables=["person","dob","events"], verbose=True)



if input_text:
    # st.write(llm(input_text))
    # st.write(parent_chain.run({"name":input_text}))
    parent_chain({"name":input_text})
    with st.expander("Person Name"):
        st.info(person_memory.buffer)
    with st.expander("DOB of person"):
        st.info(dob_memory.buffer)
    with st.expander("Major Events at dob"):
        st.info(events_memory.buffer)

