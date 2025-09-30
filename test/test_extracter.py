# test_extractor.py
import json
from resume_extractor import ResumeExtractor
from advanced_features import ResumeEnhancer

def test_resume_extraction():
    """Test the resume extraction pipeline."""
    
    # Initialize extractor
    print("Initializing Resume Extractor...")
    extractor = ResumeExtractor(model_name="mistral:7b-instruct")
    
    # Test with a sample PDF
    pdf_path = "test_resume.pdf"  # Make sure you have a test PDF
    
    try:
        print(f"\nExtracting data from: {pdf_path}")
        resume_data = extractor.extract_resume(pdf_path)
        
        print("\n" + "="*50)
        print("EXTRACTED DATA:")
        print("="*50)
        print(json.dumps(resume_data.dict(), indent=2))
        
        # Test advanced features
        print("\n" + "="*50)
        print("ADVANCED ANALYSIS:")
        print("="*50)
        
        enhancer = ResumeEnhancer()
        
        # Calculate ATS score
        ats_score = enhancer.calculate_ats_score(resume_data.dict())
        print(f"\nATS Score: {ats_score['percentage']}%")
        
        # Generate summary
        summary = enhancer.generate_summary(resume_data.dict())
        print(f"\nGenerated Summary: {summary}")
        
        # Get suggestions
        suggestions = enhancer.suggest_improvements(resume_data.dict())
        print("\nImprovement Suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_resume_extraction()
