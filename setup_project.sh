# 1. Make sure you're in the virtual environment
cd ~/pdf-query-system
source venv/bin/activate

# 2. Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# 3. Clear any previous attempts (if you tried before)
pip uninstall -y chromadb chroma-hnswlib

# 4. Create the fixed requirements.txt
cat > requirements.txt << 'EOF'
# Core Dependencies
pymupdf==1.24.0
openai==1.51.0
streamlit==1.39.0
pandas==2.2.3
python-dotenv==1.0.1

# Computer Vision & Tables
opencv-python-headless==4.9.0.80
camelot-py[cv]==0.11.0

# Vector Database (manages its own dependencies)
chromadb==0.5.3

# Text Processing
langchain==0.3.0
langchain-text-splitters==0.3.0
langchain-community==0.3.0
langchain-core==0.3.0

# Python 3.13 Support
pydantic==2.9.0
typing-extensions==4.12.2
EOF

# 5. Install in stages (recommended approach)
echo "Installing core packages..."
pip install pymupdf==1.24.0 openai==1.51.0 pandas==2.2.3 python-dotenv==1.0.1

echo "Installing Streamlit..."
pip install streamlit==1.39.0

echo "Installing OpenCV..."
pip install opencv-python-headless==4.9.0.80

echo "Installing Camelot..."
pip install camelot-py[cv]==0.11.0

echo "Installing ChromaDB (this will install correct chroma-hnswlib)..."
pip install chromadb==0.5.3

echo "Installing LangChain..."
pip install langchain==0.3.0 langchain-text-splitters==0.3.0 langchain-community==0.3.0 langchain-core==0.3.0

echo "Installing Python 3.13 compatibility packages..."
pip install pydantic==2.9.0 typing-extensions==4.12.2

echo "✅ All packages installed!"