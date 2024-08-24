import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

def getLLamaresponse(input_text, no_words, blog_style):
    try:
        # Initialize LLama2 model
        llm = CTransformers(
            model='Model/llama-2-7b-chat.ggmlv3.q8_0.bin',
            model_type='llama',
            config={'max_new_tokens': 256, 'temperature': 0.01}
        )

        # Define prompt template
        template = """
            Write a blog for {blog_style} job profile for a topic {input_text}
            within {no_words} words.
        """

        # Generate prompt
        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", 'no_words'],
            template=template
        )

        # Get response from LLama2 model
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
        print(response)
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'),
        index=0
    )

submit = st.button("Generate")

# Generate final response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
