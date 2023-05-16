# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

import os


#FastAPI setup
app = FastAPI()

class Message(BaseModel):
    text: str


#Setup OpemAI API Key
os.environ["OPENAI_API_KEY"] = ""


loader = DirectoryLoader('./Test_data_Excel-Words-Plaintext/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()



#splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(len(texts))
print(texts[0])


# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## using OpenAI embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# persiste the db to disk
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 2})


# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)


## Cite sources
def process_llm_response(llm_response):

    ans = ''
    ans += llm_response['result']
    ans += '\n\nSources:\n\n'

    for i,source in enumerate(llm_response["source_documents"]):
        ans += '- ' + source.metadata['source']
        ans += '\n\n'
    return ans




@app.post("/chat")
def chat(message: Message):
    query = message.text


    llm_response = qa_chain(query)


    print(llm_response)
    ans = process_llm_response(llm_response)
    response = ans
    # print(response)
    return {"message": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
