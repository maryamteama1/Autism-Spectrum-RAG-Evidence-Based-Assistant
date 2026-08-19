# 🧩 Autism Spectrum RAG: Evidence-Based Assistant

An AI-powered, evidence-based assistant designed to provide accurate, reliable, and verifiable information regarding **Autism Spectrum Disorder (ASD)**. By leveraging **Retrieval-Augmented Generation (RAG)**, this project eliminates AI hallucinations by grounding responses strictly in peer-reviewed medical and psychological literature.

---

## 📸 System Demo & Interface

<img width="1746" height="927" alt="Screenshot 2026-08-19 170906" src="https://github.com/user-attachments/assets/2d49fdfd-ada4-4e1e-aca6-71ec68a8c8f9" />



---

## 💡 Key Features

* **📌 Evidence-Based Responses:** Answers are retrieved strictly from scientific papers and trusted clinical guides.
* **🛡️ Hallucination Mitigation:** Utilizes RAG architecture to ensure domain-specific factual accuracy.
* **🔍 Source Citations:** Links generated content directly to retrieved literature for transparency.
* **💬 Intuitive UI:** User-friendly interface designed for parents, caregivers, educators, and healthcare professionals.

---

## 🏗️ System Architecture

The pipeline processes clinical literature into vector embeddings, stores them in a Vector Database, and uses contextual retrieval to prompt the LLM.

<img width="1902" height="876" alt="Screenshot 2026-08-19 171529" src="https://github.com/user-attachments/assets/dac2129e-a8b9-4dd4-aa38-5ef4fb10537a" />


### Tech Stack

* **LLM & Orchestration:** LangChain / LlamaIndex / OpenAI API *(Adjust as applicable)*
* **Vector Database:** ChromaDB / FAISS / Qdrant *(Adjust as applicable)*
* **Embeddings:** OpenAI Embeddings / HuggingFace Sentence-Transformers
* **Frontend:** Streamlit / Gradio
* **Language:** Python 3.9+

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.9 or higher
* API Key for LLM provider (e.g., OpenAI API Key)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/maryamteama1/Autism-Spectrum-RAG-Evidence-Based-Assistant.git
   cd Autism-Spectrum-RAG-Evidence-Based-Assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup:**
   Create a `.env` file in the root directory and add your credentials:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

```text
├── data/                  # Research papers and clinical documents (PDFs, Markdown)
├── src/                   # Core application logic
│   ├── ingestion.py       # Document loading, text splitting, and vector indexing
│   ├── rag_chain.py       # RAG pipeline setup and LLM prompt templates
│   └── utils.py           # Helper functions
├── app.py                 # Streamlit / Gradio Web UI
├── requirements.txt       # Project dependencies
├── .env.example           # Environment variables template
└── README.md              # Project documentation
```

---

## ⚠️ Disclaimer

*This application is an educational and informational tool intended to assist users in accessing peer-reviewed research on Autism Spectrum Disorder. It does **NOT** provide medical diagnosis or substitute for professional healthcare advice, diagnosis, or treatment. Always seek the advice of a qualified health provider.*

---

## 🤝 Contributing & Contact

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

* **Developer:** Maryam Teama
* **GitHub:** [@maryamteama1](https://github.com/maryamteama1)
