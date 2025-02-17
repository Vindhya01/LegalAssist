# LegalAssist: AI-Powered Legal Query Assistant

**By Vindhya Namdeo**

## Introduction
LegalAssist is an AI-powered legal query assistant designed to simplify access to legal information from the Bharatiya Nyaya Sanhita (BNS) and Bharatiya Sakshya Adhiniyam (BSA). It allows users to ask legal questions and receive detailed responses based on specific sections of these legal documents, including relevant offenses, punishments, and definitions. The objective is to provide an efficient platform for legal research, enabling individuals to quickly retrieve precise legal information.

## Dataset Preparation and Annotation
### Initial Approach
Initially, a structured dataset was manually created by extracting key legal information from BNS and BSA, identifying offenses, punishments, definitions, and illustrative examples. Libraries like PyPDF2 and pdfminer were used for PDF parsing. However, manual annotation was error-prone and time-consuming.

### Revised Approach
To overcome these challenges, an automated solution using pre-trained models for document retrieval and embedding generation was implemented:
- **Document Retrieval:** HuggingFace Embeddings
- **Pre-trained Models:** Used to generate document embeddings for querying
- **Text Summarization:** Summarization pipeline for generating concise summaries
- **Punishment Extraction:** Custom logic to extract punishments from retrieved legal sections

This revised approach, utilizing FAISS (Facebook AI Similarity Search), enabled real-time access to legal sections and summaries, significantly improving the efficiency of information retrieval.

## Model Development and Evaluation
### Models Used
- **Text Summarization:** Facebook BART-large CNN model for summarization tasks
- **Embeddings Model:** Sentence-Transformers All-MiniLM-L6-v2 model for generating vector embeddings

### Model Workflow
1. **Preprocessing:** Split legal documents into manageable chunks using RecursiveCharacterTextSplitter.
2. **Embedding Generation:** Created embeddings for each text chunk using HuggingFace Embeddings and stored in a FAISS vector database.
3. **Summarization:** Summarized long legal texts using BART-large CNN.

### Evaluation
The model's performance was measured based on the relevance of retrieved legal sections and the accuracy of punishment extraction. The models were tested with a range of legal queries to assess their effectiveness.

## Streamlit App Features and Deployment
### App Development
- **Frontend & Backend:** Developed using Streamlit for real-time input and output.
- **Chat-like Interface:** Displayed user input and AI-generated responses in a conversational format.

### Model Integration
- **Text Summarization & Embeddings:** Handled using HuggingFace’s transformers library.
- **Punishment Extraction:** Custom logic to identify punishments within the retrieved legal sections.

### Deployment
- **Platform:** Streamlit Cloud for hosting and deployment.
- **Technologies Used:** Python Libraries (streamlit, transformers, HuggingFace Embeddings, FAISS, requests)

## Conclusion
The LegalAssist project successfully integrates AI-powered models to provide relevant legal information from BNS and BSA. Despite initial challenges, the use of pre-trained models for document retrieval, summarization, and punishment extraction proved effective, making legal information more accessible to the public.
