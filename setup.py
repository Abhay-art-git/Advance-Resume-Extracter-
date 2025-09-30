# setup.py
import subprocess
import sys

def setup_project():
    """Setup the resume extractor project."""
    
    print("Setting up Resume Extractor Project...")
    
    # Install requirements
    print("\n1. Installing Python packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Download spaCy model
    print("\n2. Downloading spaCy language model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    # Download NLTK data
    print("\n3. Downloading NLTK data...")
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    
    print("\n✅ Setup complete!")
    print("\nTo run the project:")
    print("1. CLI mode: python resume_extractor.py <path_to_pdf>")
    print("2. API mode: uvicorn resume_extractor:app --reload")

if __name__ == "__main__":
    setup_project()
