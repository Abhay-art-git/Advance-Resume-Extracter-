
import os
import json
import re
from typing import List, Dict, Optional, Union
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field, validator
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize OCR (will download models on first run)
# ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='cpu')

try:
    ocr = PaddleOCR(
        use_textline_orientation=True,  # Updated parameter name
        lang='en',
        device='cpu',  # Use 'gpu' if you have CUDA installed
        show_log=False  # Disable verbose logging
    )
except Exception as e:
    print(f"Warning: OCR initialization failed: {e}")
    print("OCR features will be limited.")
    ocr = None


# Pydantic models for structured output
class Experience(BaseModel):
    company: str
    position: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    responsibilities: List[str] = []
    
    @validator('start_date', 'end_date', pre=True)
    def parse_dates(cls, v):
        if v and v.lower() in ['present', 'current', 'now']:
            return 'Present'
        return v

class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    location: Optional[str] = None

class PersonalInfo(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None

class Skills(BaseModel):
    technical: List[str] = []
    soft: List[str] = []
    languages: List[str] = []
    tools: List[str] = []

class ResumeData(BaseModel):
    personal_info: PersonalInfo
    summary: Optional[str] = None
    experience: List[Experience] = []
    education: List[Education] = []
    skills: Skills = Skills()
    certifications: List[str] = []
    projects: List[Dict[str, str]] = []
    achievements: List[str] = []

class ResumeExtractor:
    def __init__(self, model_name: str = "mistral:7b-instruct"):
        """
        Initialize the resume extractor with specified model.
        
        Args:
            model_name: Ollama model name (mistral:7b-instruct, llama3.1:8b, etc.)
        """
        self.model_name = model_name
        self.ensure_model_available()
        
    def ensure_model_available(self):
        """Check if model is available, pull if not."""
        try:
            ollama.show(self.model_name)
        except:
            print(f"Pulling model {self.model_name}... This may take a few minutes.")
            ollama.pull(self.model_name)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF, with OCR fallback for scanned documents."""
        text = ""
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Try direct text extraction first
            page_text = page.get_text()
            
            # If no text found, use OCR
            if len(page_text.strip()) < 50:  # Likely a scanned page
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_data = pix.tobytes("png")
                
                # Save temporarily for OCR
                temp_img_path = f"temp_page_{page_num}.png"
                with open(temp_img_path, "wb") as f:
                    f.write(img_data)
                
                # Perform OCR
                result = ocr.ocr(temp_img_path, cls=True)
                
                # Extract text from OCR result
                for line in result:
                    if line:
                        for word_info in line:
                            page_text += word_info[1][0] + " "
                        page_text += "\n"
                
                # Clean up temp file
                os.remove(temp_img_path)
            
            text += page_text + "\n"
        
        pdf_document.close()
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camelCase
        
        return text.strip()
    
    def create_extraction_prompt(self, resume_text: str) -> str:
        """Create a detailed prompt for structured extraction."""
        prompt = f"""You are an expert resume parser. Extract all information from the following resume and return it in valid JSON format.

IMPORTANT RULES:
1. Extract ALL information present in the resume
2. Use "Present" for current positions
3. Extract dates in original format found in resume
4. Preserve exact wording for job responsibilities
5. Categorize skills appropriately
6. Return ONLY valid JSON, no explanations

Required JSON structure:
{{
    "personal_info": {{
        "name": "",
        "email": "",
        "phone": "",
        "location": "",
        "linkedin": "",
        "github": "",
        "website": ""
    }},
    "summary": "",
    "experience": [
        {{
            "company": "",
            "position": "",
            "start_date": "",
            "end_date": "",
            "location": "",
            "responsibilities": []
        }}
    ],
    "education": [
        {{
            "institution": "",
            "degree": "",
            "field_of_study": "",
            "start_date": "",
            "end_date": "",
            "gpa": "",
            "location": ""
        }}
    ],
    "skills": {{
        "technical": [],
        "soft": [],
        "languages": [],
        "tools": []
    }},
    "certifications": [],
    "projects": [
        {{
            "name": "",
            "description": "",
            "technologies": "",
            "link": ""
        }}
    ],
    "achievements": []
}}

Resume text:
{resume_text}

Return ONLY the JSON:"""
        
        return prompt
    
    def extract_with_llm(self, text: str, max_retries: int = 3) -> Dict:
        """Extract structured data using LLM with retry logic."""
        prompt = self.create_extraction_prompt(text)
        
        for attempt in range(max_retries):
            try:
                # Call Ollama
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are a precise resume parser. Always return valid JSON.'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    options={
                        'temperature': 0.1,  # Low temperature for consistency
                        'top_p': 0.9,
                        'num_predict': 4096  # Enough tokens for full resume
                    }
                )
                
                # Extract JSON from response
                json_text = response['message']['content']
                
                # Clean up JSON if needed
                json_text = self.clean_json_response(json_text)
                
                # Parse JSON
                data = json.loads(json_text)
                return data
                
            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1} failed: JSON parsing error - {e}")
                if attempt == max_retries - 1:
                    # Last attempt - try to fix JSON
                    return self.fix_json_errors(json_text)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
    
    def clean_json_response(self, text: str) -> str:
        """Clean LLM response to extract valid JSON."""
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON content
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx + 1]
        
        return text
    
    def fix_json_errors(self, json_text: str) -> Dict:
        """Attempt to fix common JSON errors."""
        try:
            # Fix common issues
            json_text = json_text.replace("'", '"')  # Single to double quotes
            json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
            json_text = re.sub(r',\s*]', ']', json_text)
            
            return json.loads(json_text)
        except:
            # Return minimal structure if all fails
            return {
                "personal_info": {"name": "Error parsing resume"},
                "experience": [],
                "education": [],
                "skills": {"technical": [], "soft": [], "languages": [], "tools": []}
            }
    
    def validate_and_enhance(self, data: Dict) -> ResumeData:
        """Validate extracted data and enhance with post-processing."""
        # Ensure all required fields exist
        if 'personal_info' not in data:
            data['personal_info'] = {}
        
        # Validate email format
        if 'email' in data['personal_info']:
            email = data['personal_info']['email']
            if email and '@' not in email:
                data['personal_info']['email'] = None
        
        # Sort experience by date (most recent first)
        if 'experience' in data and data['experience']:
            data['experience'] = sorted(
                data['experience'],
                key=lambda x: x.get('end_date', 'Present') == 'Present',
                reverse=True
            )
        
        # Create ResumeData object for validation
        return ResumeData(**data)
    
    def extract_resume(self, pdf_path: str) -> ResumeData:
        """Main method to extract structured data from resume PDF."""
        print(f"Processing: {pdf_path}")
        
        # Step 1: Extract text from PDF
        print("Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        
        if len(text.strip()) < 100:
            raise ValueError("Could not extract sufficient text from PDF")
        
        # Step 2: Extract structured data with LLM
        print("Analyzing with AI model...")
        raw_data = self.extract_with_llm(text)
        
        # Step 3: Validate and enhance
        print("Validating and structuring data...")
        resume_data = self.validate_and_enhance(raw_data)
        
        return resume_data

# FastAPI application for web interface
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import tempfile
import shutil
import os

app = FastAPI(title="Resume Extractor API")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Professional resume extractor interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Resume Data Extractor</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                --primary: #6366f1;
                --primary-dark: #4f46e5;
                --secondary: #8b5cf6;
                --success: #10b981;
                --danger: #ef4444;
                --dark: #1f2937;
                --light: #f3f4f6;
                --white: #ffffff;
                --shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                width: 100%;
                max-width: 1200px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 24px;
                box-shadow: var(--shadow);
                overflow: hidden;
                animation: fadeIn 0.5s ease-out;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .content {
                padding: 40px;
            }
            
            .upload-section {
                background: var(--light);
                border-radius: 16px;
                padding: 40px;
                text-align: center;
                transition: all 0.3s ease;
                border: 2px dashed transparent;
            }
            
            .upload-section.dragover {
                border-color: var(--primary);
                background: rgba(99, 102, 241, 0.05);
                transform: scale(1.02);
            }
            
            .upload-icon {
                font-size: 4rem;
                color: var(--primary);
                margin-bottom: 20px;
            }
            
            .file-input-wrapper {
                position: relative;
                overflow: hidden;
                display: inline-block;
            }
            
            .file-input {
                position: absolute;
                left: -9999px;
            }
            
            .file-label {
                display: inline-block;
                padding: 12px 30px;
                background: var(--primary);
                color: white;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            
            .file-label:hover {
                background: var(--primary-dark);
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
            }
            
            .selected-file {
                margin-top: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                display: none;
                align-items: center;
                justify-content: space-between;
            }
            
            .selected-file.show {
                display: flex;
            }
            
            .file-info {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .file-icon {
                color: var(--danger);
                font-size: 1.5rem;
            }
            
            .extract-btn {
                padding: 12px 30px;
                background: var(--success);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
                display: none;
            }
            
            .extract-btn.show {
                display: inline-block;
            }
            
            .extract-btn:hover {
                background: #059669;
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(16, 185, 129, 0.3);
            }
            
            .extract-btn:disabled {
                background: #9ca3af;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin-top: 40px;
            }
            
            .loading.show {
                display: block;
            }
            
            .spinner {
                display: inline-block;
                width: 50px;
                height: 50px;
                border: 3px solid rgba(99, 102, 241, 0.1);
                border-radius: 50%;
                border-top-color: var(--primary);
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .results {
                display: none;
                margin-top: 40px;
                animation: slideIn 0.5s ease-out;
            }
            
            .results.show {
                display: block;
            }
            
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            
            .result-header h2 {
                color: var(--dark);
                font-size: 1.8rem;
            }
            
            .action-buttons {
                display: flex;
                gap: 10px;
            }
            
            .action-btn {
                padding: 8px 16px;
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 0.9rem;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .action-btn:hover {
                background: var(--light);
                transform: translateY(-1px);
            }
            
            .resume-content {
                background: white;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            
            .section {
                margin-bottom: 30px;
                padding-bottom: 30px;
                border-bottom: 1px solid #e5e7eb;
            }
            
            .section:last-child {
                border-bottom: none;
                margin-bottom: 0;
                padding-bottom: 0;
            }
            
            .section-title {
                color: var(--primary);
                font-size: 1.3rem;
                font-weight: 600;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .section-icon {
                font-size: 1.1rem;
            }
            
            .personal-info {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }
            
            .info-item {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px;
                background: var(--light);
                border-radius: 8px;
            }
            
            .info-icon {
                color: var(--primary);
                font-size: 1.2rem;
                width: 30px;
            }
            
            .experience-item, .education-item {
                background: var(--light);
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 15px;
            }
            
            .experience-header, .education-header {
                display: flex;
                justify-content: space-between;
                align-items: start;
                margin-bottom: 10px;
            }
            
            .position-title, .degree-title {
                font-weight: 600;
                color: var(--dark);
                font-size: 1.1rem;
            }
            
            .company-name, .institution-name {
                color: var(--primary);
                font-weight: 500;
                margin-top: 5px;
            }
            
            .date-range {
                color: #6b7280;
                font-size: 0.9rem;
            }
            
            .responsibilities {
                margin-top: 10px;
                padding-left: 20px;
            }
            
            .responsibilities li {
                color: #4b5563;
                margin-bottom: 5px;
                line-height: 1.6;
            }
            
            .skills-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }
            
            .skill-category {
                background: var(--light);
                padding: 15px;
                border-radius: 8px;
            }
            
            .skill-category h4 {
                color: var(--primary);
                margin-bottom: 10px;
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .skill-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .skill-tag {
                background: white;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.85rem;
                color: var(--dark);
                border: 1px solid #e5e7eb;
            }
            
            .error-message {
                background: #fee;
                color: var(--danger);
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                display: none;
                align-items: center;
                gap: 10px;
            }
            
            .error-message.show {
                display: flex;
            }
            
            @media (max-width: 768px) {
                .header h1 {
                    font-size: 2rem;
                }
                
                .content {
                    padding: 20px;
                }
                
                .upload-section {
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-file-alt"></i> AI Resume Data Extractor</h1>
                <p>Transform your PDF resume into structured data using advanced AI</p>
            </div>
            
            <div class="content">
                <div class="upload-section" id="uploadSection">
                    <i class="fas fa-cloud-upload-alt upload-icon"></i>
                    <h2>Upload Your Resume</h2>
                    <p style="color: #6b7280; margin: 10px 0 20px;">Drag & drop your PDF file here or click to browse</p>
                    
                    <div class="file-input-wrapper">
                        <input type="file" id="fileInput" class="file-input" accept=".pdf">
                        <label for="fileInput" class="file-label">
                            <i class="fas fa-folder-open"></i> Choose PDF File
                        </label>
                    </div>
                    
                    <div class="selected-file" id="selectedFile">
                        <div class="file-info">
                            <i class="fas fa-file-pdf file-icon"></i>
                            <span id="fileName"></span>
                        </div>
                        <button class="extract-btn" id="extractBtn" onclick="extractResume()">
                            <i class="fas fa-magic"></i> Extract Data
                        </button>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 20px; color: #6b7280;">Analyzing your resume with AI...</p>
                </div>
                
                <div class="error-message" id="errorMessage">
                    <i class="fas fa-exclamation-circle"></i>
                    <span id="errorText"></span>
                </div>
                
                <div class="results" id="results">
                    <div class="result-header">
                        <h2>Extracted Resume Data</h2>
                        <div class="action-buttons">
                            <button class="action-btn" onclick="downloadJSON()">
                                <i class="fas fa-download"></i> Download JSON
                            </button>
                            <button class="action-btn" onclick="copyToClipboard()">
                                <i class="fas fa-copy"></i> Copy
                            </button>
                            <button class="action-btn" onclick="resetForm()">
                                <i class="fas fa-redo"></i> New Resume
                            </button>
                        </div>
                    </div>
                    
                    <div class="resume-content" id="resumeContent">
                        <!-- Resume data will be inserted here -->
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let extractedData = null;
            
            // File input handling
            const fileInput = document.getElementById('fileInput');
            const uploadSection = document.getElementById('uploadSection');
            const selectedFile = document.getElementById('selectedFile');
            const fileName = document.getElementById('fileName');
            const extractBtn = document.getElementById('extractBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const errorMessage = document.getElementById('errorMessage');
            const errorText = document.getElementById('errorText');
            
            // Drag and drop functionality
            uploadSection.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadSection.classList.add('dragover');
            });
            
            uploadSection.addEventListener('dragleave', () => {
                uploadSection.classList.remove('dragover');
            });
            
            uploadSection.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0 && files[0].type === 'application/pdf') {
                    fileInput.files = files;
                    handleFileSelect();
                } else {
                    showError('Please upload a PDF file');
                }
            });
            
            fileInput.addEventListener('change', handleFileSelect);
            
            function handleFileSelect() {
                const file = fileInput.files[0];
                if (file) {
                    fileName.textContent = file.name;
                    selectedFile.classList.add('show');
                    extractBtn.classList.add('show');
                    hideError();
                }
            }
            
            async function extractResume() {
                const file = fileInput.files[0];
                if (!file) return;
                
                // Hide previous results and errors
                results.classList.remove('show');
                hideError();
                
                // Show loading
                loading.classList.add('show');
                extractBtn.disabled = true;
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/extract', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        extractedData = data;
                        displayResults(data);
                        results.classList.add('show');
                    } else {
                        showError(data.detail || 'Failed to extract resume data');
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    loading.classList.remove('show');
                    extractBtn.disabled = false;
                }
            }
            
            function displayResults(data) {
                const resumeContent = document.getElementById('resumeContent');
                resumeContent.innerHTML = '';
                
                // Personal Information
                if (data.personal_info) {
                    const personalSection = createSection('Personal Information', 'fa-user');
                    const personalGrid = document.createElement('div');
                    personalGrid.className = 'personal-info';
                    
                    const info = data.personal_info;
                    if (info.name) personalGrid.appendChild(createInfoItem('fa-id-card', info.name));
                    if (info.email) personalGrid.appendChild(createInfoItem('fa-envelope', info.email));
                    if (info.phone) personalGrid.appendChild(createInfoItem('fa-phone', info.phone));
                    if (info.location) personalGrid.appendChild(createInfoItem('fa-map-marker-alt', info.location));
                    if (info.linkedin) personalGrid.appendChild(createInfoItem('fa-linkedin', info.linkedin));
                    if (info.github) personalGrid.appendChild(createInfoItem('fa-github', info.github));
                    if (info.website) personalGrid.appendChild(createInfoItem('fa-globe', info.website));
                    
                    personalSection.appendChild(personalGrid);
                    resumeContent.appendChild(personalSection);
                }
                
                // Summary
                if (data.summary) {
                    const summarySection = createSection('Professional Summary', 'fa-quote-left');
                    const summaryText = document.createElement('p');
                    summaryText.style.color = '#4b5563';
                    summaryText.style.lineHeight = '1.6';
                    summaryText.textContent = data.summary;
                    summarySection.appendChild(summaryText);
                    resumeContent.appendChild(summarySection);
                }
                
                // Experience
                if (data.experience && data.experience.length > 0) {
                    const expSection = createSection('Work Experience', 'fa-briefcase');
                    
                    data.experience.forEach(exp => {
                        const expItem = document.createElement('div');
                        expItem.className = 'experience-item';
                        
                        const expHeader = document.createElement('div');
                        expHeader.className = 'experience-header';
                        
                        const leftDiv = document.createElement('div');
                        const positionTitle = document.createElement('div');
                        positionTitle.className = 'position-title';
                        positionTitle.textContent = exp.position || 'Position';
                        
                        const companyName = document.createElement('div');
                        companyName.className = 'company-name';
                        companyName.textContent = exp.company || 'Company';
                        
                        leftDiv.appendChild(positionTitle);
                        leftDiv.appendChild(companyName);
                        
                        const dateRange = document.createElement('div');
                        dateRange.className = 'date-range';
                        dateRange.textContent = `${exp.start_date || ''} - ${exp.end_date || ''}`;
                        
                        expHeader.appendChild(leftDiv);
                        expHeader.appendChild(dateRange);
                        expItem.appendChild(expHeader);
                        
                        if (exp.responsibilities && exp.responsibilities.length > 0) {
                            const respList = document.createElement('ul');
                            respList.className = 'responsibilities';
                            exp.responsibilities.forEach(resp => {
                                const li = document.createElement('li');
                                li.textContent = resp;
                                respList.appendChild(li);
                            });
                            expItem.appendChild(respList);
                        }
                        
                        expSection.appendChild(expItem);
                    });
                    
                    resumeContent.appendChild(expSection);
                }
                
                // Education
                if (data.education && data.education.length > 0) {
                    const eduSection = createSection('Education', 'fa-graduation-cap');
                    
                    data.education.forEach(edu => {
                        const eduItem = document.createElement('div');
                        eduItem.className = 'education-item';
                        
                        const eduHeader = document.createElement('div');
                        eduHeader.className = 'education-header';
                        
                        const leftDiv = document.createElement('div');
                        const degreeTitle = document.createElement('div');
                        degreeTitle.className = 'degree-title';
                        degreeTitle.textContent = edu.degree || 'Degree';
                        
                        const institutionName = document.createElement('div');
                        institutionName.className = 'institution-name';
                        institutionName.textContent = edu.institution || 'Institution';
                        
                        leftDiv.appendChild(degreeTitle);
                        leftDiv.appendChild(institutionName);
                        
                        const dateRange = document.createElement('div');
                        dateRange.className = 'date-range';
                        dateRange.textContent = `${edu.start_date || ''} - ${edu.end_date || ''}`;
                        
                        eduHeader.appendChild(leftDiv);
                        eduHeader.appendChild(dateRange);
                        eduItem.appendChild(eduHeader);
                        
                        if (edu.field_of_study) {
                            const field = document.createElement('div');
                            field.style.color = '#6b7280';
                            field.style.marginTop = '5px';
                            field.textContent = `Field of Study: ${edu.field_of_study}`;
                            eduItem.appendChild(field);
                        }
                        
                        if (edu.gpa) {
                            const gpa = document.createElement('div');
                            gpa.style.color = '#6b7280';
                            gpa.style.marginTop = '5px';
                            gpa.textContent = `GPA: ${edu.gpa}`;
                            eduItem.appendChild(gpa);
                        }
                        
                        eduSection.appendChild(eduItem);
                    });
                    
                    resumeContent.appendChild(eduSection);
                }
                
                // Skills
                if (data.skills) {
                    const skillsSection = createSection('Skills', 'fa-tools');
                    const skillsGrid = document.createElement('div');
                    skillsGrid.className = 'skills-grid';
                    
                    const skillCategories = [
                        { key: 'technical', title: 'Technical Skills' },
                        { key: 'soft', title: 'Soft Skills' },
                        { key: 'languages', title: 'Languages' },
                        { key: 'tools', title: 'Tools & Technologies' }
                    ];
                    
                    skillCategories.forEach(category => {
                        if (data.skills[category.key] && data.skills[category.key].length > 0) {
                            const categoryDiv = document.createElement('div');
                            categoryDiv.className = 'skill-category';
                            
                            const categoryTitle = document.createElement('h4');
                            categoryTitle.textContent = category.title;
                            categoryDiv.appendChild(categoryTitle);
                            
                            const skillTags = document.createElement('div');
                            skillTags.className = 'skill-tags';
                            
                            data.skills[category.key].forEach(skill => {
                                const tag = document.createElement('span');
                                tag.className = 'skill-tag';
                                tag.textContent = skill;
                                skillTags.appendChild(tag);
                            });
                            
                            categoryDiv.appendChild(skillTags);
                            skillsGrid.appendChild(categoryDiv);
                        }
                    });
                    
                    skillsSection.appendChild(skillsGrid);
                    resumeContent.appendChild(skillsSection);
                }
                
                // Projects
                if (data.projects && data.projects.length > 0) {
                    const projectsSection = createSection('Projects', 'fa-project-diagram');
                    
                    data.projects.forEach(project => {
                        const projectItem = document.createElement('div');
                        projectItem.style.marginBottom = '15px';
                        projectItem.style.padding = '15px';
                        projectItem.style.background = '#f3f4f6';
                        projectItem.style.borderRadius = '8px';
                        
                        if (project.name) {
                            const projectName = document.createElement('h4');
                            projectName.style.color = '#1f2937';
                            projectName.style.marginBottom = '8px';
                            projectName.textContent = project.name;
                            projectItem.appendChild(projectName);
                        }
                        
                        if (project.description) {
                            const desc = document.createElement('p');
                            desc.style.color = '#4b5563';
                            desc.style.marginBottom = '8px';
                            desc.textContent = project.description;
                            projectItem.appendChild(desc);
                        }
                        
                        if (project.technologies) {
                            const tech = document.createElement('div');
                            tech.style.color = '#6b7280';
                            tech.style.fontSize = '0.9rem';
                            tech.innerHTML = `<strong>Technologies:</strong> ${project.technologies}`;
                            projectItem.appendChild(tech);
                        }
                        
                        projectsSection.appendChild(projectItem);
                    });
                    
                    resumeContent.appendChild(projectsSection);
                }
                
                // Certifications
                if (data.certifications && data.certifications.length > 0) {
                    const certSection = createSection('Certifications', 'fa-certificate');
                    const certList = document.createElement('ul');
                    certList.style.paddingLeft = '20px';
                    
                    data.certifications.forEach(cert => {
                        const li = document.createElement('li');
                        li.style.color = '#4b5563';
                        li.style.marginBottom = '8px';
                        li.textContent = cert;
                        certList.appendChild(li);
                    });
                    
                    certSection.appendChild(certList);
                    resumeContent.appendChild(certSection);
                }
            }
            
            function createSection(title, iconClass) {
                const section = document.createElement('div');
                section.className = 'section';
                
                const sectionTitle = document.createElement('h3');
                sectionTitle.className = 'section-title';
                sectionTitle.innerHTML = `<i class="fas ${iconClass} section-icon"></i> ${title}`;
                
                section.appendChild(sectionTitle);
                return section;
            }
            
            function createInfoItem(iconClass, text) {
                const item = document.createElement('div');
                item.className = 'info-item';
                item.innerHTML = `<i class="fas ${iconClass} info-icon"></i> <span>${text}</span>`;
                return item;
            }
            
            function showError(message) {
                errorText.textContent = message;
                errorMessage.classList.add('show');
            }
            
            function hideError() {
                errorMessage.classList.remove('show');
            }
            
            function downloadJSON() {
                if (!extractedData) return;
                
                const dataStr = JSON.stringify(extractedData, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'resume_data.json';
                link.click();
                URL.revokeObjectURL(url);
            }
            
            function copyToClipboard() {
                if (!extractedData) return;
                
                const dataStr = JSON.stringify(extractedData, null, 2);
                navigator.clipboard.writeText(dataStr).then(() => {
                    alert('Resume data copied to clipboard!');
                }).catch(err => {
                    console.error('Failed to copy:', err);
                });
            }
            
            function resetForm() {
                fileInput.value = '';
                selectedFile.classList.remove('show');
                extractBtn.classList.remove('show');
                results.classList.remove('show');
                hideError();
                extractedData = null;
            }
        </script>
    </body>
    </html>
    """

extractor = ResumeExtractor()

@app.post("/extract")
async def extract_resume(file: UploadFile = File(...)):
    """Extract structured data from uploaded resume."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Extract resume data
        resume_data = extractor.extract_resume(tmp_path)
        
        # Convert to dict for JSON response
        return JSONResponse(content=resume_data.dict())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)
