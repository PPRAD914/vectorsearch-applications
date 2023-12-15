from tiktoken import get_encoding
from weaviate_interface import WeaviateClient
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (load_data, convert_seconds, generate_prompt_series, 
                          validate_token_threshold)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import css_templates
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
#root folder on Google Colab is: /content/
root_folder = '/content/drive/MyDrive/Corise/Vector_Search'
data_file = 'impact_theory_data.json'
data_path = os.path.join(root_folder, data_file)

# read env vars from local .env file
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
openai_api_key = os.environ['OPENAI_API_KEY']

#instantiate client
retriever = WeaviateClient(api_key, url, 'sentence-transformers/all-MiniLM-L6-v2', openai_api_key)

## RERANKER
reranker = ReRanker('cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM  --> openai_interface.py
llm = GPT_Turbo()

## ENCODING  --> tiktoken library
encodings = encoding_for_model('gpt-3.5-turbo-0613')
# = get_encoding('gpt-3.5-turbo-0613')

## INDEX NAME  --> name of your class on Weaviate cluster
class_name = "Impact_theory_minilmL6_256"

##############
#  END CODE  #
##############
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
    st.write(css_templates.load_css(), unsafe_allow_html=True)
    
    with st.sidebar:
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            ##############
            # START CODE #
            ##############

            st.write('Hmmm...this app does not seem to be working yet.  Please check back later.')
            if guest:
                st.write(f'However, it looks like you selected {guest} as a filter.')
            # make hybrid call to weaviate
            hybrid_response = retriever.hybrid_search(query, class_name)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response, query, 3)
            # validate token count is below threshold
            # valid_response = validate_token_threshold(ranked_response, 
                                                    #    question_answering_prompt_series, 
                                                    #    query=query,
                                                    #    tokenizer= # variable from ENCODING,
                                                    #    token_threshold=4000, 
                                                    #    verbose=True)
            ##############
            #  END CODE  #
            ##############

            # # generate LLM prompt
            # prompt = generate_prompt_series(base_prompt=question_answering_prompt_series, query=query, results=valid_response)
            
            # # prep for streaming response
            # st.subheader("Response from Impact Theory (context)")
            # with st.spinner('Generating Response...'):
            #     st.markdown("----")
            #     res_box = st.empty()
            #     report = []
            #     col_1, _ = st.columns([4, 3], gap='large')
                
            #     # execute chat call to LLM
            #                  ##############
            #                  # START CODE #
            #                  ##############
            #     

            #                  ##############
            #                  #  END CODE  #
            #                  ##############
            #         try:
            #             with res_box:
            #                 report.append(resp.choices[0].delta.get('content', '\n'))
            #                 result = "".join(report).strip()
            #                 res_box.markdown(f'*{result}*')

            #         except Exception as e:
            #             print(e)
            #             continue
            # ##############
            # # START CODE #
            # ##############
            # st.subheader("Search Results")
            # for i, hit in enumerate(valid_response):
            #     col1, col2 = st.columns([7, 3], gap='large')
            #     image = # get thumbnail_url
            #     episode_url = # get episode_url
            #     title = # get title
            #     show_length = # get length
            #     time_string = # convert show_length to readable time string
            # ##############
            # #  END CODE  #
            # ##############
            #     with col1:
            #         st.write(css_templates.search_result(i=i, 
            #                                         url=episode_url,
            #                                         episode_num=i,
            #                                         title=title,
            #                                         content=hit['content'], 
            #                                         length=time_string),
            #                 unsafe_allow_html=True)
            #         st.write('\n\n')
            #     with col2:
            #         # st.write(f"<a href={episode_url} <img src={image} width='200'></a>", 
            #         #             unsafe_allow_html=True)
            #         st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()
