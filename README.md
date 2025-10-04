# langchain-cv-analyser-json

# Resume Analyzer with LangChain + Google Gemini

This project is a **Resume Analyzer** built using **LangChain**, **Google Gemini API**, and **HuggingFace embeddings**. It extracts structured information from PDF resumes and outputs the results in clean **JSON format** for further use in applications like recruitment dashboards, HR systems, or candidate profiling.

### 🔹 Features

* **PDF Resume Parsing**: Loads and processes multiple resumes from the `pdfs/` folder.
* **Text Splitting**: Uses `CharacterTextSplitter` to chunk resume text into manageable sections for better context handling.
* **Embeddings & Vector Store**: Creates embeddings with `sentence-transformers/all-mpnet-base-v2` and stores them in a **ChromaDB** collection.
* **LLM-Powered Extraction**: Leverages **Google Gemini 2.5 Pro** (via LangChain) to analyze resumes and extract structured details.
* **JSON Output**: Produces standardized JSON following this schema:

  ```json
  [
    {
      "name": "",
      "role_title": "",
      "contact_details": {
        "mobile": "",
        "email": "",
        "location": ""
      },
      "professional_summary": "",
      "skills": [],
      "technical_skills": [],
      "experience": [
        {
          "designation": "",
          "company": "",
          "duration": "",
          "project": "",
          "key_responsibilities_achievements": []
        }
      ],
      "certifications": [],
      "languages_known": [],
      "education": [
        {
          "degree": "",
          "institution": "",
          "university": "",
          "year": "",
          "percentage": ""
        }
      ],
      "hard_skills": [],
      "additional_info": ""
    }
  ]
  ```
* **Data Export**: Option to convert the JSON into CSV or a new JSON file (`summary.csv` or `summary.json`) for downstream use.

### 🔹 Tech Stack

* **LangChain**: Framework for chaining LLM prompts & data pipelines
* **Google Generative AI (Gemini)**: For natural language understanding and structured extraction
* **HuggingFace Transformers**: For text embeddings
* **ChromaDB**: Vector database to store embeddings
* **PyPDFLoader**: To parse PDF resumes
* **Pandas**: For post-processing and exporting results

### 🔹 How It Works

1. Place resumes inside the `pdfs/` folder.
2. Run the script — it will parse, split, and embed the documents.
3. The LLM processes the content and outputs structured JSON.
4. (Optional) Save results as `summary.csv` or `summary.json`.

### 🔹 Future
