# advanced_features.py

from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime
from dateutil import parser
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class ResumeEnhancer:
    """Advanced features for resume analysis and enhancement."""
    
    def __init__(self):
        # Load spaCy model for NER (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("SpaCy model not found. Some features will be limited.")
            self.nlp = None
        
        # Download NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            print("NLTK data not found. Some features will be limited.")
            self.stop_words = set()
    
    def calculate_experience_years(self, experience: List[Dict]) -> float:
        """Calculate total years of experience."""
        total_months = 0
        
        for exp in experience:
            start_date = exp.get('start_date')
            end_date = exp.get('end_date')
            
            if not start_date:
                continue
            
            try:
                # Parse start date
                start = self._parse_date(start_date)
                if not start:
                    continue
                
                # Parse end date
                if end_date and end_date.lower() in ['present', 'current', 'now']:
                    end = datetime.now()
                else:
                    end = self._parse_date(end_date)
                    if not end:
                        continue
                
                # Calculate months
                months = (end.year - start.year) * 12 + (end.month - start.month)
                total_months += max(0, months)
                
            except Exception as e:
                print(f"Error parsing dates: {e}")
                continue
        
        return round(total_months / 12, 1)
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_str:
            return None
        
        # Common date patterns
        patterns = [
            r'(\d{4})',  # Just year
            r'(\d{1,2})/(\d{4})',  # MM/YYYY
            r'(\w+)\s+(\d{4})',  # Month YYYY
            r'(\w+)\s*,?\s*(\d{4})',  # Month, YYYY
        ]
        
        try:
            # Try dateutil parser first
            return parser.parse(date_str, fuzzy=True)
        except:
            # Try patterns
            for pattern in patterns:
                match = re.search(pattern, date_str)
                if match:
                    if len(match.groups()) == 1:  # Just year
                        return datetime(int(match.group(1)), 1, 1)
                    elif len(match.groups()) == 2:
                        try:
                            if match.group(1).isdigit():  # MM/YYYY
                                return datetime(int(match.group(2)), int(match.group(1)), 1)
                            else:  # Month YYYY
                                return parser.parse(f"{match.group(1)} 1, {match.group(2)}")
                        except:
                            continue
        
        return None
    
    def extract_key_skills(self, resume_data: Dict, top_n: int = 10) -> List[str]:
        """Extract key skills using TF-IDF from experience and skills sections."""
        # Combine all text
        text_parts = []
        
        # Add experience descriptions
        for exp in resume_data.get('experience', []):
            text_parts.extend(exp.get('responsibilities', []))
        
        # Add skills
        skills = resume_data.get('skills', {})
        for skill_type in ['technical', 'soft', 'languages', 'tools']:
            text_parts.extend(skills.get(skill_type, []))
        
        # Add projects
        for project in resume_data.get('projects', []):
            text_parts.append(project.get('description', ''))
            text_parts.append(project.get('technologies', ''))
        
        # Combine and clean
        full_text = ' '.join(text_parts).lower()
        
        # Extract important terms using TF-IDF
        if full_text.strip():
            try:
                vectorizer = TfidfVectorizer(
                    max_features=top_n * 2,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                tfidf_matrix = vectorizer.fit_transform([full_text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                # Get top terms
                top_indices = scores.argsort()[-top_n:][::-1]
                key_skills = [feature_names[i] for i in top_indices]
                
                return key_skills
            except:
                pass
        
        return []
    
    def calculate_skill_match(self, resume_data: Dict, job_requirements: List[str]) -> float:
        """Calculate how well resume matches job requirements."""
        # Extract all skills from resume
        resume_skills = set()
        
        # From skills section
        skills = resume_data.get('skills', {})
        for skill_type in ['technical', 'soft', 'languages', 'tools']:
            resume_skills.update([s.lower() for s in skills.get(skill_type, [])])
        
        # From experience
        for exp in resume_data.get('experience', []):
            for resp in exp.get('responsibilities', []):
                # Extract potential skills from responsibilities
                words = word_tokenize(resp.lower())
                resume_skills.update([w for w in words if len(w) > 2 and w not in self.stop_words])
        
        # Calculate match
        job_requirements_lower = [req.lower() for req in job_requirements]
        matches = sum(1 for req in job_requirements_lower if any(req in skill for skill in resume_skills))
        
        return round(matches / len(job_requirements) * 100, 2) if job_requirements else 0
    
    def generate_summary(self, resume_data: Dict) -> str:
        """Generate a professional summary based on resume content."""
        # Extract key information
        name = resume_data.get('personal_info', {}).get('name', 'Professional')
        
        # Calculate experience
        experience_years = self.calculate_experience_years(resume_data.get('experience', []))
        
        # Get latest position
        latest_position = None
        if resume_data.get('experience'):
            latest_position = resume_data['experience'][0].get('position')
        
        # Get top skills
        key_skills = self.extract_key_skills(resume_data, top_n=5)
        
        # Get education
        education = resume_data.get('education', [])
        highest_degree = education[0].get('degree') if education else None
        
        # Build summary
        summary_parts = []
        
        if latest_position:
            summary_parts.append(f"Experienced {latest_position}")
        
        if experience_years > 0:
            summary_parts.append(f"with {experience_years} years of professional experience")
        
        if highest_degree:
            summary_parts.append(f"holding a {highest_degree}")
        
        if key_skills:
            skills_str = ", ".join(key_skills[:3])
            summary_parts.append(f"Skilled in {skills_str}")
        
        summary = ". ".join(summary_parts) + "."
        
        return summary
    
    def analyze_career_progression(self, experience: List[Dict]) -> Dict:
        """Analyze career progression and growth."""
        if not experience:
            return {"progression": "No experience data"}
        
        positions = [exp.get('position', '') for exp in experience]
        companies = [exp.get('company', '') for exp in experience]
        
        # Detect progression patterns
        progression_indicators = {
            'promotion': ['senior', 'lead', 'manager', 'director', 'vp', 'chief'],
            'lateral': ['specialist', 'analyst', 'consultant', 'engineer'],
            'career_change': []
        }
        
        # Analyze position titles
        progression_score = 0
        for i in range(len(positions) - 1):
            current = positions[i].lower()
            previous = positions[i + 1].lower()
            
            # Check for promotions
            for indicator in progression_indicators['promotion']:
                if indicator in current and indicator not in previous:
                    progression_score += 2
                elif indicator in current and indicator in previous:
                    # Check if same company (internal promotion)
                    if companies[i] == companies[i + 1]:
                        progression_score += 1
        
        # Determine progression type
        if progression_score >= 4:
            progression_type = "Strong upward progression"
        elif progression_score >= 2:
            progression_type = "Steady career growth"
        else:
            progression_type = "Lateral movement or early career"
        
        return {
            "progression_type": progression_type,
            "progression_score": progression_score,
            "total_positions": len(positions),
            "unique_companies": len(set(companies))
        }
    
    def extract_achievements(self, resume_data: Dict) -> List[str]:
        """Extract quantifiable achievements from experience."""
        achievements = []
        
        # Patterns for achievements
        achievement_patterns = [
            r'(?:increased|improved|enhanced|boosted).*?(\d+%?)',
            r'(?:reduced|decreased|cut|saved).*?(\d+%?)',
            r'(?:generated|drove|delivered).*?(\$[\d,]+|\d+%?)',
            r'(?:managed|led|oversaw).*?(\d+)',
            r'(?:achieved|accomplished|completed).*',
            r'(?:awarded|recognized|honored).*'
        ]
        
        # Search in experience
        for exp in resume_data.get('experience', []):
            for resp in exp.get('responsibilities', []):
                for pattern in achievement_patterns:
                    if re.search(pattern, resp, re.IGNORECASE):
                        achievements.append(resp)
                        break
        
        # Add explicit achievements
        achievements.extend(resume_data.get('achievements', []))
        
        return list(set(achievements))  # Remove duplicates
    
    def suggest_improvements(self, resume_data: Dict) -> List[str]:
        """Suggest improvements for the resume."""
        suggestions = []
        
        # Check personal info completeness
        personal_info = resume_data.get('personal_info', {})
        if not personal_info.get('email'):
            suggestions.append("Add email address for contact information")
        if not personal_info.get('phone'):
            suggestions.append("Add phone number for easier contact")
        if not personal_info.get('linkedin'):
            suggestions.append("Add LinkedIn profile to showcase professional network")
        
        # Check summary
        if not resume_data.get('summary'):
            suggestions.append("Add a professional summary to highlight your value proposition")
        
        # Check experience
        experience = resume_data.get('experience', [])
        if experience:
            for i, exp in enumerate(experience):
                if len(exp.get('responsibilities', [])) < 3:
                    suggestions.append(f"Add more details to position at {exp.get('company', 'Company ' + str(i+1))}")
                
                # Check for quantifiable achievements
                has_numbers = any(re.search(r'\d+', resp) for resp in exp.get('responsibilities', []))
                if not has_numbers:
                    suggestions.append(f"Add quantifiable achievements for position at {exp.get('company', 'Company ' + str(i+1))}")
        
        # Check skills
        skills = resume_data.get('skills', {})
        total_skills = sum(len(skills.get(k, [])) for k in ['technical', 'soft', 'languages', 'tools'])
        if total_skills < 10:
            suggestions.append("Add more relevant skills to improve keyword matching")
        
        # Check education dates
        for edu in resume_data.get('education', []):
            if not edu.get('end_date'):
                suggestions.append(f"Add graduation date for {edu.get('institution', 'education')}")
        
        return suggestions
    
    def calculate_ats_score(self, resume_data: Dict) -> Dict:
        """Calculate ATS (Applicant Tracking System) compatibility score."""
        score = 0
        max_score = 100
        breakdown = {}
        
        # Check personal information (20 points)
        personal_score = 0
        personal_info = resume_data.get('personal_info', {})
        if personal_info.get('name'): personal_score += 5
        if personal_info.get('email'): personal_score += 5
        if personal_info.get('phone'): personal_score += 5
        if personal_info.get('location'): personal_score += 5
        breakdown['personal_info'] = personal_score
        score += personal_score
        
        # Check experience section (30 points)
        experience_score = 0
        experience = resume_data.get('experience', [])
        if experience:
            experience_score += 10
            # Check for dates
            if all(exp.get('start_date') for exp in experience):
                experience_score += 10
            # Check for descriptions
            if all(len(exp.get('responsibilities', [])) >= 2 for exp in experience):
                experience_score += 10
        breakdown['experience'] = experience_score
        score += experience_score
        
        # Check education (15 points)
        education_score = 0
        education = resume_data.get('education', [])
        if education:
            education_score += 10
            if all(edu.get('degree') and edu.get('institution') for edu in education):
                education_score += 5
        breakdown['education'] = education_score
        score += education_score
        
        # Check skills (20 points)
        skills_score = 0
        skills = resume_data.get('skills', {})
        total_skills = sum(len(skills.get(k, [])) for k in ['technical', 'soft', 'languages', 'tools'])
        if total_skills >= 5: skills_score += 10
        if total_skills >= 10: skills_score += 10
        breakdown['skills'] = skills_score
        score += skills_score
        
        # Check for keywords and formatting (15 points)
        keyword_score = 0
        # Check for action verbs in experience
        action_verbs = ['managed', 'led', 'developed', 'created', 'implemented', 'achieved']
        exp_text = ' '.join([' '.join(exp.get('responsibilities', [])) for exp in experience]).lower()
        if any(verb in exp_text for verb in action_verbs):
            keyword_score += 10
        # Check for quantifiable results
        if re.search(r'\d+%|\$\d+|\d+\+', exp_text):
            keyword_score += 5
        breakdown['keywords'] = keyword_score
        score += keyword_score
        
        return {
            'total_score': score,
            'max_score': max_score,
            'percentage': round(score / max_score * 100, 2),
            'breakdown': breakdown 

                    }  # This closing brace was missing for the return statement
    
    def compare_resumes(self, resume1: Dict, resume2: Dict) -> Dict:
        """Compare two resumes and highlight differences."""
        comparison = {
            'experience_years': {
                'resume1': self.calculate_experience_years(resume1.get('experience', [])),
                'resume2': self.calculate_experience_years(resume2.get('experience', []))
            },
            'skills_count': {
                'resume1': sum(len(resume1.get('skills', {}).get(k, [])) for k in ['technical', 'soft', 'languages', 'tools']),
                'resume2': sum(len(resume2.get('skills', {}).get(k, [])) for k in ['technical', 'soft', 'languages', 'tools'])
            },
            'ats_scores': {
                'resume1': self.calculate_ats_score(resume1),
                'resume2': self.calculate_ats_score(resume2)
            },
            'unique_skills': {
                'resume1_only': [],
                'resume2_only': [],
                'common': []
            }
        }
        
        # Compare skills
        skills1 = set()
        skills2 = set()
        
        for skill_type in ['technical', 'soft', 'languages', 'tools']:
            skills1.update([s.lower() for s in resume1.get('skills', {}).get(skill_type, [])])
            skills2.update([s.lower() for s in resume2.get('skills', {}).get(skill_type, [])])
        
        comparison['unique_skills']['resume1_only'] = list(skills1 - skills2)
        comparison['unique_skills']['resume2_only'] = list(skills2 - skills1)
        comparison['unique_skills']['common'] = list(skills1 & skills2)
        
        return comparison
    
    def generate_cover_letter_points(self, resume_data: Dict, job_description: str) -> List[str]:
        """Generate key points for a cover letter based on resume and job description."""
        points = []
        
        # Extract key achievements
        achievements = self.extract_achievements(resume_data)
        if achievements:
            points.append(f"Key Achievement: {achievements[0]}")
        
        # Highlight relevant experience
        experience_years = self.calculate_experience_years(resume_data.get('experience', []))
        if experience_years > 0:
            latest_position = resume_data.get('experience', [{}])[0].get('position', '')
            points.append(f"{experience_years} years of experience as {latest_position}")
        
        # Match skills with job description
        job_keywords = word_tokenize(job_description.lower())
        job_keywords = [w for w in job_keywords if len(w) > 3 and w not in self.stop_words]
        
        matching_skills = []
        skills = resume_data.get('skills', {})
        for skill_type in ['technical', 'soft', 'languages', 'tools']:
            for skill in skills.get(skill_type, []):
                if any(keyword in skill.lower() for keyword in job_keywords):
                    matching_skills.append(skill)
        
        if matching_skills:
            points.append(f"Relevant skills: {', '.join(matching_skills[:3])}")
        
        # Add education if relevant
        education = resume_data.get('education', [])
        if education:
            degree = education[0].get('degree', '')
            field = education[0].get('field_of_study', '')
            if degree:
                points.append(f"Educational background: {degree} {f'in {field}' if field else ''}")
        
        return points
    
    def extract_industry_keywords(self, resume_data: Dict) -> Dict[str, List[str]]:
        """Extract industry-specific keywords from resume."""
        industry_keywords = {
            'tech': ['software', 'development', 'programming', 'api', 'database', 'cloud', 'agile', 'devops'],
            'finance': ['financial', 'analysis', 'investment', 'portfolio', 'risk', 'compliance', 'audit'],
            'marketing': ['marketing', 'campaign', 'seo', 'social media', 'brand', 'content', 'analytics'],
            'healthcare': ['patient', 'clinical', 'medical', 'healthcare', 'diagnosis', 'treatment', 'care'],
            'education': ['teaching', 'curriculum', 'student', 'learning', 'education', 'training', 'instruction']
        }
        
        # Combine all text from resume
        all_text = []
        for exp in resume_data.get('experience', []):
            all_text.extend(exp.get('responsibilities', []))
        
        skills = resume_data.get('skills', {})
        for skill_type in skills:
            all_text.extend(skills.get(skill_type, []))
        
        full_text = ' '.join(all_text).lower()
        
        # Identify industries
        detected_industries = {}
        for industry, keywords in industry_keywords.items():
            matches = [kw for kw in keywords if kw in full_text]
            if matches:
                detected_industries[industry] = matches
        
        return detected_industries


# Example usage script
def main():
    """Example usage of ResumeEnhancer with advanced features."""
    import json
    
    # Initialize enhancer
    enhancer = ResumeEnhancer()
    
    # Example resume data (you would get this from ResumeExtractor)
    sample_resume = {
        "personal_info": {
            "name": "John Doe",
            "email": "john.doe@email.com",
            "phone": "+1-234-567-8900",
            "location": "San Francisco, CA"
        },
        "experience": [
            {
                "company": "Tech Corp",
                "position": "Senior Software Engineer",
                "start_date": "2020",
                "end_date": "Present",
                "responsibilities": [
                    "Led development of microservices architecture reducing latency by 40%",
                    "Managed team of 5 developers",
                    "Implemented CI/CD pipeline using Jenkins and Docker"
                ]
            }
        ],
        "skills": {
            "technical": ["Python", "JavaScript", "Docker", "AWS"],
            "soft": ["Leadership", "Communication"],
            "tools": ["Git", "Jenkins", "Kubernetes"]
        }
    }
    
    # Calculate ATS score
    ats_score = enhancer.calculate_ats_score(sample_resume)
    print(f"ATS Score: {ats_score['percentage']}%")
    print(f"Breakdown: {json.dumps(ats_score['breakdown'], indent=2)}")
    
    # Generate summary
    summary = enhancer.generate_summary(sample_resume)
    print(f"\nGenerated Summary: {summary}")
    
    # Extract achievements
    achievements = enhancer.extract_achievements(sample_resume)
    print(f"\nKey Achievements: {achievements}")
    
    # Get improvement suggestions
    suggestions = enhancer.suggest_improvements(sample_resume)
    print(f"\nImprovement Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")


if __name__ == "__main__":
    main()
