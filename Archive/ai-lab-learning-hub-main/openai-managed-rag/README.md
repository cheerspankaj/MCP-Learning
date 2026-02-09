OPENAI-Managed-RAG - Your RAG app in one hour

Welcome to **OpenAI-Managed-RAG**! This repository contains the code and resources for deploying the creating and running a RAG application use OPENAI's file search Managed RAG.

With file search OpenAI does the heavy lifting of creating and hosting and managing the backend of the RAG app including your file repository, your vector store and calls to the various models. This allows for efficient and accurate retrieval of information from your files.

## Getting Started

Follow the steps below to get started with setting up and running the sample OPENAI-Managed-RAG application.

### Prerequisites
open terminal in the root directory of the  AI Lab Learning hub
Navigate into the project directory:

```bash
cd  openai-managed-rag
```

### Setting Up the Environment
Create a virtual environment and install the required dependencies following the steps in [the setup guide](../setup.md#common-steps-for-all-setups). 

### Running the Application
Start the application using Streamlit:
```bash
streamlit run openai-managed-rag/ragapp.py 
``` 

On your web browser, open http://localhost:8501 to see the application in action.

On first start the app will prompt you for your openai api key, your assistant id and your vector store id.
If you are doing this as part of a training session, use the info provided by your training session.
Access the application in your web browser at http://localhost:8501.

### Contributing
Farhat Siddiqui FASI

