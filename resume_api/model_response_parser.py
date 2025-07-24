import json, ast, re, pandas as pd

class ModelResponseParser:
    def __init__(self):
        self.skill_keywords = set()
        self.education_keywords = set()
        self.experience_keywords = set()

    def safe_json_parse(self, raw):
        if isinstance(raw, dict):
            return raw
        if pd.isna(raw) or not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(raw)
            except Exception:
                return {}

    def _normalize_keys(self, d):
        return {
            k.lower().strip().replace(" ", "_"): v
            for k, v in d.items()
        }

    def _split_str(self, text):
        # split on periods, semicolons, or newlines into clean chunks
        parts = re.split(r'[.;\n]\s*', text)
        return [p.strip() for p in parts if p.strip()]

    def extract_skills(self, data):
        skills = []
        for field in ['required_skills', 'core_responsibilities', 'preferred_qualifications']:
            if field not in data:
                continue

            val = data[field]
            # 1) If top-level is a string: split into pieces
            if isinstance(val, str):
                skills.extend(self._split_str(val))

            # 2) If it’s a list, unpack each element
            elif isinstance(val, list):
                for item in val:
                    # a) bare string
                    if isinstance(item, str):
                        skills.extend(self._split_str(item))
                    # b) dict with a name/value field
                    elif isinstance(item, dict):
                        # try a few common keys
                        for key in ('name', 'skill', 'value', 'title'):
                            if key in item and isinstance(item[key], str):
                                skills.extend(self._split_str(item[key]))
                                break

        # Clean up: strip whitespace, lowercase, filter out non-strings, dedupe
        clean_skills = {
            s.strip().lower()
            for s in skills
            if isinstance(s, str) and s.strip()
        }
        return list(clean_skills)
  
    def extract_education(self, data):
        ed = []
        for field in ['educational_requirements', 'education']:
            if field in data:
                val = data[field]
                if isinstance(val, list):
                    ed.extend(val)
                elif isinstance(val, str):
                    ed.extend(self._split_str(val))
        return list(set(ed))

    def extract_experience(self, data):
        exp = []
        for field in ['experience_level', 'experience', 'years_experience']:
            if field in data:
                val = data[field]
                if isinstance(val, list):
                    exp.extend(val)
                elif isinstance(val, str):
                    exp.extend(self._split_str(val))
        return list(set(exp))

    def extract_all_features(self, raw):
        # 1) parse & normalize
        parsed = self.safe_json_parse(raw)
        parsed = self._normalize_keys(parsed)
        # 2) extract each feature
        skills    = self.extract_skills(parsed)
        education = self.extract_education(parsed)
        exp       = self.extract_experience(parsed)

        # debug
        # print(f"\nDEBUG JD RAW\n{raw}")
        # print(f"DEBUG PARSED JSON\n{parsed}")
        # print(f"DEBUG EXTRACTED SKILLS ({len(skills)}) -\n{skills}")

        # 3) update global sets as before
        self.skill_keywords.update(skills)
        self.education_keywords.update(education)
        self.experience_keywords.update(exp)

        # 4) return the feature dict
        return {
            'skills':    skills,
            'education': education,
            'experience':exp,
            'raw_json':  parsed
        }
