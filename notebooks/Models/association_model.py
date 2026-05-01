import pandas as pd

# Load the rules CSV (already generated in your notebook)
rules_path = r"D:\HealthCare System\Data\Processed\top_association_rules.csv"

rules_df = pd.read_csv(rules_path)


# -------------------------------------------------------
# STEP 1: Plain-English dictionary
# Each encoded name → what it means in simple words
# -------------------------------------------------------
plain_english_words = {
    "gender_1"                   : "Male patient",
    "gender_2"                   : "Patient of unknown gender",
    "medication_cat_Low"         : "on few medications (0-10)",
    "medication_cat_Medium"      : "on moderate medications (11-20)",
    "medication_cat_High"        : "on many medications (20+)",
    "time_in_hospital_cat_Short" : "SHORT hospital stay (0-3 days)",
    "time_in_hospital_cat_Medium": "MODERATE hospital stay (4-6 days)",
    "time_in_hospital_cat_Long"  : "LONG hospital stay (7+ days)",
    "readmitted_1"               : "readmitted after 30+ days",
    "readmitted_2"               : "NOT previously readmitted",
    "max_glu_serum_1"            : "very high blood glucose",
    "max_glu_serum_2"            : "glucose test not done",
    "max_glu_serum_3"            : "normal blood glucose",
    "A1Cresult_1"                : "high A1C blood sugar (>8%)",
    "A1Cresult_2"                : "A1C test not done",
    "A1Cresult_3"                : "normal A1C blood sugar",
    "diagnoses_cat_Low"          : "few diagnoses (1-3)",
    "diagnoses_cat_Medium"       : "moderate diagnoses (4-7)",
    "diagnoses_cat_High"         : "many diagnoses (8+)",
    "race_1"                     : "Asian patient",
    "race_2"                     : "Caucasian patient",
    "race_3"                     : "Hispanic patient",
    "race_4"                     : "patient of other race",
    "race_5"                     : "patient of unknown race",
}


# -------------------------------------------------------
# STEP 2: Helper - pull out the words from a frozenset string
# Example: frozenset({'gender_1', 'medication_cat_Low'})
#          → ['gender_1', 'medication_cat_Low']
# -------------------------------------------------------
def get_words_from_frozenset(frozenset_text):
    words = []
    # Split by single quote and grab every other piece
    parts = frozenset_text.split("'")
    for i, part in enumerate(parts):
        # Odd positions hold the actual words
        if i % 2 == 1:
            words.append(part)
    return words


# -------------------------------------------------------
# STEP 3: Convert one encoded word to plain English
# If it is not in the dictionary, just clean it up a little
# -------------------------------------------------------
def decode_one_word(word):
    if word in plain_english_words:
        return plain_english_words[word]
    else:
        # Not in dictionary - just replace underscores with spaces
        return word.replace("_", " ")


# -------------------------------------------------------
# STEP 4: Build the full plain-English sentence for one rule
# -------------------------------------------------------
def make_plain_english(antecedents, consequents, confidence, support, lift):
    # Get the encoded words from antecedents and consequents
    if_words  = get_words_from_frozenset(antecedents)
    then_words = get_words_from_frozenset(consequents)

    # Decode each word to plain English
    if_plain   = [decode_one_word(w) for w in if_words]
    then_plain = [decode_one_word(w) for w in then_words]

    # Join them into readable strings
    if_text   = " + ".join(if_plain)
    then_text = " + ".join(then_plain)

    # Round the numbers for display
    conf_pct = round(confidence * 100, 1)
    sup_pct  = round(support * 100, 1)
    lift_r   = round(lift, 2)

    # Build the sentence
    sentence = (
        f"Patients who are {if_text} "
        f"tend to have {then_text} — "
        f"and this is true {conf_pct}% of the time "
        f"(found in {sup_pct}% of all patients, lift = {lift_r})."
    )
    return sentence


# -------------------------------------------------------
# STEP 5: Add plain_english column to any dataframe of rules
# -------------------------------------------------------
def add_plain_english_column(df):
    plain_english_list = []
    for i, row in df.iterrows():
        sentence = make_plain_english(
            row["antecedents"],
            row["consequents"],
            row["confidence"],
            row["support"],
            row["lift"]
        )
        plain_english_list.append(sentence)
    df = df.copy()
    df["plain_english"] = plain_english_list
    return df


def get_top_rules_with_english(n=10):
    df = rules_df.head(n)
    return add_plain_english_column(df)


def search_rules_with_english(keyword):
    results = search_rules(keyword)
    if results.empty:
        return results
    return add_plain_english_column(results)


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