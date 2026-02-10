import pandas as pd
# Load the rules CSV (already generated in your notebook)
rules_path= r"D:\HealthCare System\Data\Processed\top_association_rules.csv"

rules_df = pd.read_csv(rules_path)

def get_top_rules(n=10):
    """
    Returns the top N association rules.
    """
    return rules_df.head(n)

def search_rules(keyword):
    """
    Returns rules where the keyword appears in antecedents or consequents.
    Handles both raw values and human-readable terms.
    """
    keyword = keyword.lower()
    
    # Map human-readable terms to actual column values
    keyword_map = {
        'high': ['_3', 'cat_High', 'glu_serum_3', 'A1Cresult_3'],
        'low': ['cat_Low', 'cat_low'],
        'medium': ['cat_Medium', 'cat_medium'],
        'short': ['cat_Short', 'cat_short'],
        'long': ['cat_Long', 'cat_long'],
        'glucose': ['glu_serum', 'max_glu'],
        'medication': ['medication_cat'],
        'readmit': ['readmitted'],
        'diagnos': ['diagnoses_cat'],
        'gender': ['gender_'],
        'race': ['race_'],
        'a1c': ['A1Cresult']
    }
    
    # Check if keyword has a mapping
    search_terms = [keyword]
    for key, values in keyword_map.items():
        if keyword in key:
            search_terms.extend(values)
    
    # Search in both columns
    mask = pd.Series([False] * len(rules_df))
    for term in search_terms:
        mask = mask | (
            rules_df['antecedents'].str.lower().str.contains(term, na=False) | 
            rules_df['consequents'].str.lower().str.contains(term, na=False)
        )
    
    return rules_df[mask]