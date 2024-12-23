# 🗣️ **Law Talk ChatBot** 📚🎓

Welcome to the **Law Talk ChatBot** project, an AI-powered chatbot that allows users to interact with their legal documents in a conversational manner. It leverages powerful language models and document processing to provide answers based on uploaded documents such as PDFs, DOCX files, and text files.

The chatbot is built using **Streamlit** for the frontend interface, **Langchain** for conversational AI, and **Replicate** for utilizing LLaMA-2 models for natural language processing. It also incorporates **Text-to-Speech** functionality powered by **gTTS** for audio responses!

---

## 🚀 **Features**

- 📄 **Document Upload**: Upload PDFs, DOCX files, and text files.
- 💬 **Conversational AI**: Chat with the bot and get answers based on the documents you upload.
- 🎤 **Text-to-Speech**: Hear the responses from the chatbot.
- 🔄 **Live Search**: Retrieve answers based on the content of the documents you upload.
- ⚡ **Customizable Temperature**: Adjust the "creativity" of the responses using a slider.

---

## 🛠️ **Tech Stack**

This project uses the following libraries and frameworks:

- **[Streamlit](https://streamlit.io/)**: For building the interactive web application.
- **[Langchain](https://www.langchain.com/)**: To power the conversational chain and document retrieval.
- **[Replicate](https://replicate.com/)**: For using the LLaMA-2 language model to generate conversational responses.
- **[FAISS](https://github.com/facebookresearch/faiss)**: For efficient similarity search and document indexing.
- **[gTTS](https://pypi.org/project/gTTS/)**: For converting the chatbot responses into speech.
- **[Sentence Transformers](https://www.sbert.net/)**: For embedding text and documents to be searched for relevant information.
- **[Hugging Face](https://huggingface.co/)**: To access pre-trained models like LLaMA-2.
- **[Python-dotenv](https://pypi.org/project/python-dotenv/)**: For managing environment variables.

---

## 🔧 **Installation**

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ramasaimehar/Law-Talk-ChatBot.git
   cd law-talk-chatbot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the necessary environment variables:
   - Create a `.env` file in the root directory.
   - Add your credentials and configuration to the `.env` file.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 📂 **Project Structure**

```
law-talk-chatbot/
│
├── app.py                   # Main application file
├── requirements.txt         # List of dependencies
├── .env                     # Environment variables
└── README.md                # Project description
```



---

## 📦 **Requirements**

The following libraries are required to run this project:

```txt
langchain
torch
accelerate
sentence_transformers
streamlit_chat
streamlit
faiss-cpu
tiktoken
ctransformers
huggingface-hub
pypdf
python-dotenv
replicate
docx2txt
gtts
```
## 🖥️ **User Interface**

The **Law Talk ChatBot** features an interactive and user-friendly interface built with **Streamlit**. Once the application is launched, users are presented with a clean layout where they can:

![image](https://github.com/user-attachments/assets/3c3ebf60-6cbb-499e-b3b4-b10bbc5cf53e)


- **Upload Files**: The sidebar allows users to upload legal documents like PDFs, DOCX, and text files.
- **Ask Questions**: The main interface provides a text input box to ask questions based on the uploaded documents.
- **View Chat History**: The chatbot's responses are displayed in a conversational format, making it easy to follow along.
- **Text-to-Speech**: Hear the chatbot's responses in audio form.

### **Demo**

Here is a preview of the **Law Talk ChatBot** interface with prefered temperature parameter range difference:

Chatbot giving answers to related query asked by the user at temperature 0.5 – 1.0

![image](https://github.com/user-attachments/assets/f65401fa-5a5e-4a49-b3b3-fb4f177269a2)

Chatbot giving answers to related query asked by the user at temperature  0 - 0.15

![image](https://github.com/user-attachments/assets/083b222c-c5e0-471e-ad37-90155b3561d3)


---

## 🤝 **Acknowledgments**

- **Langchain** and **Replicate** for enabling advanced conversational AI.
- **Streamlit** for creating a seamless and interactive web experience.
- **Hugging Face** for providing access to powerful language models.

---
