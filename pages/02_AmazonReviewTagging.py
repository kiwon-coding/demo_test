import pandas as pd
import json
import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from tqdm import tqdm 
import ast
import streamlit as st

@st.cache_data
def load_review_data():
    json_file_path = './data/amazon_fashion_simple.json'
    with open(json_file_path, 'r') as f:
        json_data = [json.loads(line) for line in f]

    return pd.DataFrame(json_data)
    
def parse_string_to_list(string_data):
    try:
        parsed_list = ast.literal_eval(string_data)
        return parsed_list
    except Exception as e:
        print(string_data)
        print("Error parsing string:", e)
        return []

def get_taggings(review_text, openai_api_key):
    chat = ChatOpenAI(openai_api_key = openai_api_key)

    template = "A tagging system that creates tags for use in an online shopping mall."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Create up to 5 tags for the given review. The result should be an python style array of strings: ```{text}```"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    res = chain.run(text = review_text)

    tags = parse_string_to_list(res)

    return tags

@st.cache_data
def load_review_tags():
    review_tags = pd.read_csv('./data/amazon_fashion_review_tags.csv')
    return review_tags

def show_reviews(item):
    custom_css = """
        <style>
            .review-card {
                border: 1px solid #e1e1e1;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 15px;
                background-color: #f9f9f9;
            }
            .reviewer-info {
                font-size: 14px;
                color: #4f4f4f;
            }
            .review-text {
                font-size: 16px;
                color: #000;
                margin-top: 10px;
            }
            .product-image {
                max-width: 150px;
                margin-top: 10px;
            }
        </style>
        """
    st.write(custom_css, unsafe_allow_html=True)

    for i, row in item.iterrows():
        st.markdown(f'<div class="review-card">', unsafe_allow_html=True)
        st.write(f"**Overall Score:** {row['overall']}")

        if row['verified']:
            st.write("Verified Purchase")
        else:
            st.write("Non-Verified Purchase")

        st.write(f"**Written Date:** {row['reviewTime']}")
        st.write(f"**Reviewer ID:** {row['reviewerID']}")
        st.write(f"**Product ID:** {row['asin']}")
        st.write(f"**Reviewer Name:** {row['reviewerName']}")

        st.markdown('<hr>', unsafe_allow_html=True)

        st.write(f"**Review Text:**")
        st.markdown(
            f'<div class="review-text">{row["reviewText"]}</div>', unsafe_allow_html=True)

        st.write(f"**Tags:** {row['tags']}")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", ""))
    api_key_button = st.button("Add")
    if api_key_button:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if openai_api_key:

        if not os.path.exists('./data/amazon_fashion_review_tags.csv'):
            reviews = load_review_data()

            tqdm.pandas()
            reviews['tags'] = reviews.progress_apply(lambda x: get_taggings(x['reviewText'], openai_api_key), axis=1)
            reviews.to_csv('./data/amazon_fashion_review_tags.csv', index=False)
        else:
            all_tags = {}
            review_tags = load_review_tags()
            tag_column_df = review_tags['tags'].apply(ast.literal_eval)
            for tags in tag_column_df:
                for tag in tags:
                    all_tags[tag] = all_tags.get(tag, 0) + 1
            # print(all_tags)

            # print(dict(sorted(all_tags.items())))
            # print(dict(sorted(all_tags.items(), key=lambda item: item[1])))

            sorted_tags = dict(sorted(all_tags.items(), key=lambda item: item[1], reverse=True))
            # st.write(f"\# of reviews: {len(review_tags)}")
            # print(sorted_tags.items())
            major_keywords = list(sorted_tags.keys())[:10]
            selected_tags = st.multiselect('Select tags to filter reviews', major_keywords)
            # st.write("You selected:", selected_tags)


            # all_tags = [tag for tags in tag_column_df for tag in tags]
            # print(tag_list)

            # st.write(review_tags.head())

            # show reviews containing selected tags
            # 1) find matching reviews (containing at least one tag in its tags)
            # 2) show the reviews (st.write(reviewText), st.divider())
            if len(selected_tags) > 0:
                selected_reviews = review_tags[review_tags['tags'].apply(lambda x: all(tag in x for tag in selected_tags))]
                # print(selected_reviews)
                show_reviews(selected_reviews)
