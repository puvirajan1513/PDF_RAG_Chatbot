# PDF RAG Chatbot

A Streamlit app for uploading PDFs and asking questions about their contents using Google Gemini embeddings and chat.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Keep `.env` out of Git; it is already ignored.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. In the app's **Settings ? Secrets**, add:

   ```toml
   GOOGLE_API_KEY = "your-google-ai-studio-api-key"
   ```

4. Deploy. The app reads the key from Streamlit secrets in the cloud and from `.env` for local development.

The deployment dependency file is `requirements.txt`. The existing `requirments.txt` is retained for compatibility with the local project setup.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file in the project root containing `GOOGLE_API_KEY=your-key` before running locally.
