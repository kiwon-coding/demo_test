import streamlit as st
import openai
import os

if __name__ == '__main__':
    st.title("Product Description Generator")

    st.write("OK")
    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", ""))
    api_key_button = st.button("Add")
    if api_key_button:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key

        product_desc = st.text_input("Product Information", placeholder="Enter a short product information")
        if st.button("Generate"):
            prompt = f"write a product description based on the below information: {product_desc}"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )

            msg = ""
            msg_element = st.empty()
            for chunk in response:
                res = chunk.choices[0]['delta']
                if 'content' in res:
                    ai_output = res['content']
                    msg += ai_output

                    with msg_element.container():
                        st.write(msg)