@echo off
REM Automatic setup script for Legal AI virtual environment on Windows
REM Run this file in Command Prompt or PowerShell

echo ================================================
echo 🚀 SETUP LEGAL AI ENVIRONMENT FOR WINDOWS
echo ================================================

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv legal_ai_env

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call legal_ai_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install libraries with strategic ordering to avoid conflicts
echo 📚 Installing PyTorch first...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo 📚 Installing core data science libraries...
pip install numpy pandas scikit-learn scipy

echo 📚 Installing HuggingFace ecosystem in order...
pip install huggingface_hub
pip install tokenizers
pip install safetensors
pip install transformers

echo 📚 Installing sentence-transformers...
pip install sentence-transformers

echo 📚 Installing remaining utilities...
pip install python-docx tqdm matplotlib seaborn
pip install ipykernel jupyter notebook

echo ✅ All libraries installed successfully!

REM Setup Jupyter kernel
echo 🔧 Configuring Jupyter kernel...
python -m ipykernel install --user --name=legal_ai_env --display-name "Legal AI Environment"

REM Test imports
echo 🧪 Testing installation...
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print('✅ PyTorch OK')
except:
    print('❌ PyTorch failed')

try:
    from transformers import AutoTokenizer
    print('✅ Transformers OK')
except:
    print('❌ Transformers failed')

try:
    from sentence_transformers import SentenceTransformer
    print('✅ Sentence-transformers OK')
except:
    print('❌ Sentence-transformers failed')

try:
    from docx import Document
    print('✅ python-docx OK')
except:
    print('❌ python-docx failed')

print('🎉 Setup completed!')
"

echo.
echo ================================================
echo ✅ SETUP COMPLETED!
echo ================================================
echo.
echo To use:
echo 1. Activate virtual environment: legal_ai_env\Scripts\activate
echo 2. Run Jupyter: jupyter notebook
echo 3. Select kernel "Legal AI Environment"
echo.
echo To deactivate virtual environment: deactivate
echo.
pause
