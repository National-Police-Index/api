from Levenshtein import distance as levenshtein_distance


def validate_agency_match(mention_agency: str, mentioned_agencies: str, post_agency: str, threshold: float = 0.85) -> tuple[bool, str]:
    """
    Validate if the POST agency matches either the source agency or one of the mentioned agencies.
    """
    
    def similarity_score(str1: str, str2: str) -> float:
        """Calculate normalized similarity score (1.0 = identical, 0.0 = completely different)"""
        if not str1 or not str2:
            return 0.0
        
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()
        
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        dist = levenshtein_distance(str1, str2)
        return 1.0 - (dist / max_len)
    
    if not post_agency:
        return False, "POST agency is empty"
    
    # Normalize post_agency once for all comparisons
    post_agency_normalized = post_agency.lower().strip()
    
    # Check source agency
    if mention_agency:
        mention_agency_normalized = mention_agency.lower().strip()
        score = similarity_score(mention_agency_normalized, post_agency_normalized)
        if score >= threshold:
            return True, ""
    
    # Parse and check mentioned agencies list
    if mentioned_agencies and mentioned_agencies.strip():
        try:
            # Handle string representation of list
            import ast
            if isinstance(mentioned_agencies, str):
                agencies_list = ast.literal_eval(mentioned_agencies)
            else:
                agencies_list = mentioned_agencies
            
            if isinstance(agencies_list, list):
                for agency in agencies_list:
                    if agency:  # Skip empty strings
                        # Normalize the agency string
                        agency_normalized = str(agency).lower().strip()
                        score = similarity_score(agency_normalized, post_agency_normalized)
                        if score >= threshold:
                            return True, ""
        except (ValueError, SyntaxError):
            # If parsing fails, treat as single string
            mentioned_normalized = mentioned_agencies.lower().strip()
            score = similarity_score(mentioned_normalized, post_agency_normalized)
            if score >= threshold:
                return True, ""
    
    return False, "Agency cannot be validated"