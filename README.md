# **Interactive PDF AI - You Get What You Want**

Interactive PDF AI is an online PDF question-answering system that allows users to ask questions related to any PDF and receive output directly from it. This application supports inputting multiple PDFs simultaneously and generates responses based on the content of the PDFs.

## **Table of Contents**

- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## **Features**

- **Streamlit**: This application is built using Streamlit, making the interface user-friendly.
- **Langchain**: A crucial framework for developing generative AI applications and utilizing open-source large language models.
- **Huggingface**: An essential framework for leveraging open-source large language models.
- **Vector Database**: Allows storing data as vectors, which are very useful for checking similarity.

## **Technologies**

- Streamlit
- Langchain
- Huggingface Hub and Huggingface Embeddings
- Meta Llama2
- faiss-cpu

## **Installation**

1. **Set up Huggingface**:

   - Create a Huggingface account at [Huggingface](https://huggingface.co/).
   - Generate your own Huggingface API Key in the settings -> access tokens section of the Huggingface Hub.
   - Search for `llama-2-7b-chat.ggmlv3.q8_0` and request a license to use the model.
   - Download the `llama-2-7b-chat` model, create a new project, and save the model in a directory named `models`.

2. **Clone this repository**

3. Install the Requirements:
   ```
   pip install -r requirements.txt                                                                                                                                              
   ```
4. Run the app:

   ```
   streamlit run main.py
   ```


## Usage

1. After running main.py, an interface will be displayed where you can upload your PDFs.

 2.After selecting and uploading the PDFs, click on the "Process" button and wait until "Process Completed" is displayed.

 3.Now, ask your questions related to your PDFs and wait for the response.

 4.Your responses will be stored until your session state ends.


## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.
