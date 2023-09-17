import streamlit as st
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


if __name__ == '__main__':
    st.title("Product Description Generator")
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