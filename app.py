"""
RAG Toxic Interviewer Application
A system that uses RAG (Retrieval-Augmented Generation) to conduct interviews
"""

import os
from typing import List, Dict
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader, DirectoryLoader


class RAGToxicInterviewer:
    """RAG-based interviewer system"""
    
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the RAG Toxic Interviewer
        
        Args:
            knowledge_base_path: Path to knowledge base documents
        """
        self.knowledge_base_path = knowledge_base_path
        self.vectorstore = None
        self.conversation_chain = None
        self.chat_history = []
        
    def load_documents(self, file_path: str = None) -> List:
        """Load documents from file or directory"""
        if file_path:
            loader = TextLoader(file_path)
        elif self.knowledge_base_path:
            loader = DirectoryLoader(self.knowledge_base_path, glob="**/*.txt")
        else:
            # Return empty list if no documents
            return []
            
        documents = loader.load()
        return documents
    
    def setup_vectorstore(self, documents: List = None):
        """Setup vector store from documents"""
        if not documents:
            documents = self.load_documents()
        
        if not documents:
            # Create a dummy document if no documents available
            from langchain.schema import Document
            documents = [Document(page_content="Welcome to the toxic interviewer. Please provide documents to analyze.")]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        
    def setup_conversation_chain(self):
        """Setup conversational retrieval chain"""
        if not self.vectorstore:
            self.setup_vectorstore()
            
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        
    def chat(self, question: str) -> tuple:
        """
        Process a chat message
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, chat_history)
        """
        if not self.conversation_chain:
            self.setup_conversation_chain()
        
        result = self.conversation_chain({"question": question})
        answer = result["answer"]
        
        self.chat_history.append((question, answer))
        
        return answer, self.chat_history
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.chat_history = []
        if self.conversation_chain:
            self.conversation_chain.memory.clear()


def create_gradio_interface():
    """Create Gradio web interface"""
    interviewer = RAGToxicInterviewer()
    
    def chat_interface(message: str, history: List) -> tuple:
        """Chat interface handler"""
        try:
            answer, _ = interviewer.chat(message)
            return answer
        except Exception as e:
            return f"Error: {str(e)}. Please ensure OPENAI_API_KEY is set in environment variables."
    
    def upload_documents(file):
        """Handle document upload"""
        try:
            if file is not None:
                # Load the uploaded file
                from langchain.document_loaders import TextLoader
                from langchain.schema import Document
                
                loader = TextLoader(file.name)
                documents = loader.load()
                interviewer.setup_vectorstore(documents)
                return "Documents uploaded and processed successfully!"
            return "No file uploaded."
        except Exception as e:
            return f"Error uploading documents: {str(e)}"
    
    def reset():
        """Reset conversation"""
        interviewer.reset_conversation()
        return None
    
    # Create Gradio interface
    with gr.Blocks(title="RAG Toxic Interviewer") as demo:
        gr.Markdown("# RAG Toxic Interviewer")
        gr.Markdown("An AI-powered interviewer using Retrieval-Augmented Generation")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Interview Conversation")
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Reset Conversation")
            
            with gr.Column(scale=1):
                gr.Markdown("### Upload Knowledge Base")
                file_upload = gr.File(
                    label="Upload Documents (.txt)",
                    file_types=[".txt"]
                )
                upload_btn = gr.Button("Process Documents")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
        
        # Event handlers
        submit.click(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[msg]
        ).then(
            lambda: "", None, msg
        )
        
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[msg]
        ).then(
            lambda: "", None, msg
        )
        
        clear.click(reset, outputs=[chatbot])
        
        upload_btn.click(
            upload_documents,
            inputs=[file_upload],
            outputs=[upload_status]
        )
    
    return demo


def main():
    """Main application entry point"""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set it before running: export OPENAI_API_KEY='your-key-here'")
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
