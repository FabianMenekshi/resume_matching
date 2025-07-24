import re

# Create features for talent scoring model
class TalentScoringFeatureEngineer: 
    
    def __init__(self, parser):
        self.parser = parser
        self.skill_vectorizer = None
        self.edu_vectorizer = None
        self.scaler = StandardScaler()
        
        # Create skill matching features (partial credit on token overlap)
    def create_skill_match_features(self, resume_text, jd_skills):
        import re

        # 1) Clean and tokenize resume
        resume_clean = re.sub(r'[^a-z0-9\s]', ' ', resume_text.lower())
        resume_tokens = set(re.findall(r'\b\w+\b', resume_clean))

        total_score = 0.0
        cover_count = 0

        for skill in jd_skills:
            # clean & tokenize each JD skill phrase
            skill_clean = re.sub(r'[^a-z0-9\s]', ' ', skill.lower())
            tokens = re.findall(r'\b\w+\b', skill_clean)
            if not tokens:
                continue

            # fraction of tokens present
            match_cnt = sum(1 for t in tokens if t in resume_tokens)
            frac = match_cnt / len(tokens)
            total_score += frac

            # count as “covered” if at least half of the tokens matched
            if frac >= 0.5:
                cover_count += 1

        n = len(jd_skills)
        u = len(set(jd_skills))
        return {
            'skill_match_count': total_score, # now a float sum of partial scores
            'skill_match_ratio': total_score / n if n else 0.0,
            'skill_coverage': cover_count / u if u else 0.0
        }


    # Create education matching features
    def create_education_match_features(self, resume_text, jd_education): 
        if not jd_education:
            return {
                'edu_match_count': 0,
                'edu_match_present': 0
            }
        
        resume_lower = resume_text.lower()
        matches = 0
        
        # Common education keywords
        edu_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma', 
                       'university', 'college', 'graduate', 'undergraduate']
        
        for edu in jd_education:
            if edu and edu in resume_lower:
                matches += 1
        
        # Check for general education presence
        edu_present = any(keyword in resume_lower for keyword in edu_keywords)
        
        return {
            'edu_match_count': matches,
            'edu_match_present': 1 if edu_present else 0
        }
    
    # Create experience matching features
    def create_experience_match_features(self, resume_text, jd_experience): 
        if not jd_experience:
            return {
                'exp_match_count': 0,
                'exp_years_mentioned': 0
            }
        
        resume_lower = resume_text.lower()
        matches = 0
        
        for exp in jd_experience:
            if exp and exp in resume_lower:
                matches += 1
        
        # Extract years mentioned in resume
        year_pattern = r'\b(\d{1,2})\s*(?:years?|yrs?)\b'
        years_found = re.findall(year_pattern, resume_lower)
        max_years = max([int(y) for y in years_found], default=0)
        
        return {
            'exp_match_count': matches,
            'exp_years_mentioned': max_years
        }
    
    # Create text similarity features using pre-computed matrices
    def create_text_similarity_features(self, resume_text, jd_text, similarity_matrices):
        # This will be filled with similarity scores from Task 2
        return {
            'tfidf_similarity': 0.0,
            'word2vec_similarity': 0.0,
            'transformer_similarity': 0.0
        }

    # Create all features for a resume-JD pair
    def create_comprehensive_features(self, resume_idx, jd_idx, resume_df, jd_df, similarity_matrices):
        
        # Get resume and JD data
        resume_text = resume_df.iloc[resume_idx]['cleaned_resume']
        jd_text = jd_df.iloc[jd_idx]['cleaned_job_description']
        jd_model_response = jd_df.iloc[jd_idx]['model_response']
        
        # Parse JD model response
        jd_features = self.parser.extract_all_features(jd_model_response)
        
        #  NEW: coerce jd_features['skills'] into a proper list 
        raw_skills = jd_features.get('skills', [])
        if isinstance(raw_skills, str):
            # split comma-separated string into list
            skills_list = [s.strip() for s in raw_skills.split(',') if s.strip()]
        else:
            skills_list = raw_skills
        
        # Create feature dictionary
        features = {}
        
        # 1. Skill matching features (now getting a real list)
        skill_features = self.create_skill_match_features(resume_text, skills_list)
        features.update(skill_features)
        
        # 2. Education matching features
        edu_features = self.create_education_match_features(resume_text, jd_features['education'])
        features.update(edu_features)
        
        # 3. Experience matching features
        exp_features = self.create_experience_match_features(resume_text, jd_features['experience'])
        features.update(exp_features)
        
        # 4. Text similarity features (swap in Word2Vec, add best-of-all)
        tfidf_sim = similarity_matrices['TF-IDF'][resume_idx, jd_idx]
        w2v_sim = similarity_matrices['Word2Vec'][resume_idx, jd_idx]
        xformer_sim = similarity_matrices['all-MiniLM-L6-v2'][resume_idx, jd_idx]

        features['tfidf_similarity']    = w2v_sim
        features['word2vec_similarity'] = w2v_sim
        features['transformer_similarity'] = w2v_sim 

        # opt
        # features['best_similarity'] = max(tfidf_sim, w2v_sim, xformer_sim)

        
        # 5. Text length features
        features['resume_length'] = len(resume_text)
        features['jd_length']     = len(jd_text)
        features['length_ratio']  = len(resume_text) / len(jd_text) if len(jd_text) > 0 else 0
        
        # 6. Category matching (if available)
        resume_category = resume_df.iloc[resume_idx]['Category']
        jd_position     = jd_df.iloc[jd_idx]['position_title'].lower()
        
        category_match = 0
        if resume_category.lower() in jd_position or any(
            word in jd_position 
            for word in resume_category.lower().split()
        ):
            category_match = 1
        
        features['category_match'] = category_match
        
        return features