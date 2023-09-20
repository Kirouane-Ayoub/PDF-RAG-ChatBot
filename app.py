from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import streamlit as st
st.set_page_config(page_title='RAG Chatbot')
def model(temperature ,top_p , model_id_name ) :
    tokenizer = LlamaTokenizer.from_pretrained(model_id_name)

    base_model = LlamaForCausalLM.from_pretrained(
        model_id_name,
        load_in_4bit=True,
        device_map='auto',
    )
    pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.15
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_model = HuggingFaceEmbeddings(model_name=embed_model_id)
    return llm , embed_model

st.header(":hand: Welcome To your Custom Open-Source RAG Chatbot : ")
st.write("""
1 / Input HF model ID and Click Entre To Submit.\n 
2 / You can change temperature and top_p values.\n
3 / Click START to start downloading your (model + tokenizer).\n
4 / When it finishes, you can Upload your pdf file as external knowledge for your chatbot and Start chating.
""")
st.sidebar.image("icon.png")
model_id_name = st.sidebar.text_input("Input your HF model ID")
temperature = st.sidebar.slider("Select your temperature value : " ,min_value=0.1 ,
                                 max_value=1.0 ,
                                   value=0.5)
top_p = st.sidebar.slider("Select your top_p value : " ,min_value=0.1 ,
                           max_value=1.0 , 
                           value=0.5)

k_n = st.number_input("Enter the number of top-ranked retriever Results:" ,
                             min_value=1 , max_value=5 , value=4)

if st.sidebar.button("START") : 
    if model_id_name  : 
        with st.spinner("downloading (model + tokenizer)..."):
            llm , embed_model = model(temperature ,top_p , model_id_name)

def load_pdf(pdf_path) : 
  # location of the pdf file/files. 
  doc_reader = PdfReader(pdf_path)
  # read data from the file and put them into a variable called raw_text
  raw_text = ''
  for i, page in enumerate(doc_reader.pages):
      text = page.extract_text()
      if text:
          raw_text += text
  # Splitting up the text into smaller chunks for indexing
  text_splitter = CharacterTextSplitter(        
          separator = "\n",
          chunk_size = 1000,
          chunk_overlap  = 200, #striding over the text
          length_function = len,
      )
  texts = text_splitter.split_text(raw_text)
  return texts

pdffile = st.file_uploader("Please upload your pdf file to start The conversation: " , type=['pdf'])
if pdffile and llm and embed_model : 
    with st.spinner("In progress...") :
        texts=load_pdf(pdffile.name)
        vectordb = Chroma.from_texts(texts, embedding=embed_model , persist_directory="DB")

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={"k": k_n}))

    def chatbot_response(input_text):
        return rag_pipeline(input_text)["result"]
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi :hand: Im Your open-source RAG chatbot, You can ask any thing about you pdf file content."}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
