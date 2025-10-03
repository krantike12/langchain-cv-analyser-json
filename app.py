import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import pandas as pd
from dotenv import load_dotenv
import json
import sys
# import pandas as pd

load_dotenv()
api_key = os.environ.get('API_KEY')


print(sys.executable)
#Text Splitter
splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap = 100)

#embedding
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

#promptTemplate
prompt_template = PromptTemplate(
    input_variables=["content"],
    template = """
You are a Resume Analyzer. Extract all relevant details from the following resume text 
and return the output strictly in JSON format matching this template:

[
  {{
    "name": "",
    "role_title": "",
    "contact_details": {{
      "mobile": "",
      "email": "",
      "location": ""
    }},
    "professional_summary": "",
    "skills": [],
    "technical_skills": [],
    "experience": [
      {{
        "designation": "",
        "company": "",
        "duration": "",
        "project": "",
        "key_responsibilities_achievements": []
      }}
    ],
    "certifications": [],
    "languages_known": [],
    "education": [
      {{
        "degree": "",
        "institution": "",
        "university": "",
        "year": "",
        "percentage": ""
      }}
    ],
    "hard_skills": [],
    "additional_info": ""
  }}
]

Resume Content:
{content}

Make sure the JSON is valid and all fields are filled as much as possible.
"""
)

#llm model
llm = GoogleGenerativeAI(model= "gemini-2.5-pro", google_api_key = api_key)
# llm = OllamaLLM(model = "mistral")

#chain
chain = prompt_template | llm 

#chain execution

cwd = os.getcwd()
folder = os.path.join(cwd, "pdfs")

file_paths = []
for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    file_paths.append(file_path)
#print(file_paths)

# document = []
all_texts = []
for path in file_paths:
    file_loader = PyPDFLoader(path)
    #print(path)
    #document Load
    document = file_loader.load()
    #response from documentLoad
    response = document[0].page_content
    #tokenization, text splitting
    extracted = splitter.split_text(response)
    all_texts.extend(extracted)

#vectorStore to Store embeddings
db = Chroma.from_texts(texts=all_texts, embedding=embeddings, collection_name="cv_collection")

# print(all_texts)
    #query
#query = db.similarity_search("What are the Skills?")
result = chain.invoke({
    "content" : "\n\n".join([doc for doc in all_texts])})



#cleaning data to handle some errors
cleaned_json_string = result.strip().removeprefix('```json').removesuffix('```').strip()
json_result = json.loads(cleaned_json_string)
print(json_result)


# # df = pd.DataFrame(json_result)

# # df.to_csv("summary.csv", index=False)
# # # print(document)
    
# # #print(files)


# # #file loader
  
  
# # data = pd.read_csv('summary.csv', encoding='utf-8')

# # df = pd.DataFrame(data)

# # json_file = df.to_json("summary.json", indent=2, orient="records")