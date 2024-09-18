from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
   
    ####PDF from File

    path="AdvisorHandbook.pdf"
    
    ####Ask for the PDF
    #files = None
   
    ## Wait for the user to upload a file
    #while files == None:
    #    files = await cl.AskFileMessage(
    #        content="Please upload a PDF file to begin!",
    #        accept=["application/pdf"],
    #        max_size_mb=20,
    #        timeout=180,
    #    ).send()

    #file = files[0]
    #path=file.path

    #msg = cl.Message(content=f"Processing `{file.name}`...")
    #await msg.send()

    #msg = cl.Message(content=f"Processing `{path}`...")
    #await msg.send()

    loader = PyMuPDFLoader(path)
    loaded_pdf = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000 , chunk_overlap=1000)

    texts = text_splitter.split_documents(loaded_pdf)
    print("Chunking ready")

    #### Using Local Embeddings
    #embeddings = OpenAIEmbeddings(
    #    openai_api_base="http://localhost:1234/v1",
    #    disallowed_special=(),
    #    openai_api_key="no-needed",
    #    check_embedding_ctx_length=False,
    #)
    
    ### Using Google Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    docsearch = Chroma.from_documents(texts, embeddings)
    print("Embeddings Ready")

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    prompt = ChatPromptTemplate.from_messages( 
        [
            (
                "system",
                """"You are an academic advisor at ECT, your job is to provide informative answer to student questions about program regulations based on the following context. 
                ONLY USE THE INFORMATION IN THE CONTEXT. If you do not find the answer in the context, say "I do not know, ask a human advisor for that" 

                CONTEXT: 
                {context}

            """,
            ),
            ("human", "{question}"),
        ]

     )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatGroq(model="mixtral-8x7b-32768", streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    print("Chain ready")

    # Let the user know that the system is ready
    #msg.content = f"Processing `{path}` done. You can now ask questions!"
    #await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  

    source_elements_dict = {}
    source_elements = []
    
    if source_documents:
        for idx, source in enumerate(res["source_documents"]):
            title = source.metadata["title"]

            if title not in source_elements_dict:
                source_elements_dict[title] = {
                    "page_number": [source.metadata["page"]+1],
                    "url": source.metadata["title"],
                    "raw": source.page_content
                }

            else:
                source_elements_dict[title]["page_number"].append(source.metadata["page"]+1)

            # sort the page numbers
            source_elements_dict[title]["page_number"].sort()

        for title, source in source_elements_dict.items():
            # create a string for the page numbers
            page_numbers = ", ".join([str(x) for x in source["page_number"]])
            text_for_source = f"Page Number(s): {page_numbers}"
            source_elements.append(
                cl.Text(name=title, content=text_for_source, display="inline")
            )
            #source_elements.append(
            #    cl.Pdf(path="./"+title, name=title, display="side")
            #)
        
        source_names = [text_el.name for text_el in source_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
