# RAG Chatbot - React + FastAPI

A modern RAG (Retrieval-Augmented Generation) chatbot with React frontend and FastAPI backend. Upload PDF documents and chat with them using Google Gemini AI and Pinecone vector database.

## 🌐 Live Demo

- **Frontend**: https://rag-chatbot-01.vercel.app
- **Backend API**: https://ragchatbot-01.onrender.com
- **API Docs**: https://ragchatbot-01.onrender.com/docs

## ✨ Features

- 📄 **PDF Upload**: Upload and process PDF documents
- 💬 **AI Chat**: Ask questions about your documents using Google Gemini
- 🔍 **Source Citations**: See which parts of the document were used to answer
- 🗑️ **Knowledge Management**: Clear the knowledge base anytime
- 🎨 **Premium UI**: Notion-inspired design with cream color palette

## 🎨 Design

Premium Notion-inspired UI with:
- Warm cream color palette (#FAF9F6, #8B7355)
- Inter font family
- Smooth animations and micro-interactions
- Responsive design

## 🛠️ Technology Stack

**Frontend:**
- React 18 + Vite
- Axios for API calls
- React Icons
- Custom CSS (Notion-inspired)

**Backend:**
- FastAPI (Python 3.11)
- LangChain 0.2.x
- Google Gemini (LLM)
- Pinecone (Vector DB)
- LangSmith (Observability)

## ⚡ Quick Start (Local Development)

### Prerequisites
- Node.js 18+
- Python 3.11
- API keys: Google AI (Gemini), Pinecone, LangSmith (optional)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/hemanth090/RagChatbot-01.git
cd RagChatbot-01
```

2. **Backend Setup**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Create .env file with your API keys
cat > .env << EOF
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-chatbot-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-chatbot
EOF
```

3. **Frontend Setup**
```bash
cd frontend
npm install

# Create .env for frontend
echo "VITE_API_URL=http://localhost:8000" > .env
```

4. **Run Locally**

Terminal 1 - Backend:
```bash
uvicorn main:app --reload
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Visit: http://localhost:5173

## 🚀 Deployment

### Backend (Render)

**Already Deployed**: https://ragchatbot-01.onrender.com

To deploy your own:

1. Push to GitHub
2. Create web service on [Render](https://render.com)
3. Connect GitHub repository
4. Add environment variables in Render dashboard
5. Deploy!

**Important Files:**
- `.python-version` - Forces Python 3.11.9
- `render.yaml` - Deployment configuration
- `requirements.txt` - Pinned package versions

### Frontend (Vercel)

Coming soon! Deploy to Vercel with:

1. Import GitHub repository
2. Set Root Directory: `frontend`
3. Add environment variable: `VITE_API_URL=https://ragchatbot-01.onrender.com`
4. Deploy!

## 📁 Project Structure

```
RagChatbot-01/
├── .python-version       # Python 3.11.9
├── main.py              # FastAPI backend
├── rag_service.py       # RAG logic
├── requirements.txt     # Python dependencies
├── render.yaml          # Render config
├── .env                 # API keys (gitignored)
└── frontend/
    ├── src/
    │   ├── components/  # UI components
    │   ├── services/    # API client
    │   ├── App.jsx
    │   └── index.css    # Design system
    ├── vercel.json      # Vercel config
    └── package.json
```

## 🔑 Environment Variables

### Backend (.env)
```env
GOOGLE_API_KEY=          # Get from https://ai.google.dev/
PINECONE_API_KEY=        # Get from https://www.pinecone.io/
PINECONE_INDEX_NAME=rag-chatbot-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
LANGSMITH_API_KEY=       # Optional: https://smith.langchain.com/
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-chatbot
```

### Frontend (.env)
```env
VITE_API_URL=http://localhost:8000  # Local development
# VITE_API_URL=https://ragchatbot-01.onrender.com  # Production
```

## 📖 API Endpoints

- `POST /upload` - Upload PDF document
- `POST /chat` - Send message and get AI response
- `POST /clear` - Clear knowledge base
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger)

## 🎯 Usage

1. **Upload a PDF**: Drag and drop or click to upload
2. **Wait for Processing**: System chunks and indexes your document
3. **Ask Questions**: Type questions about the uploaded document
4. **View Sources**: Expand source citations to see relevant chunks
5. **Clear Knowledge**: Remove all documents when done

## 🐛 Troubleshooting

### Backend won't start
- Check Python version: `python --version` (should be 3.11.x)
- Verify API keys in `.env` file
- Check Pinecone index exists

### Frontend shows "Disconnected"
- Ensure backend is running on port 8000
- Check `VITE_API_URL` in frontend `.env`
- Verify CORS settings in `main.py`

## 📝 Deployment Notes

**Python Version:**
- Uses Python 3.11.9 (not 3.13) for package compatibility
- `.python-version` file ensures correct version on Render

**Package Versions:**
- All packages pinned to exact versions
- LangChain 0.2.x (stable) instead of 0.3.x (cutting-edge)
- See `deployment_issues.md` for full deployment story


