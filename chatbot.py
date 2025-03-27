# Import necessary libraries
import pandas as pd
import numpy as np
import os
import time
from openai import OpenAI

# Set up the OpenAI client
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key="YOUR API KEY"  # Replace with your actual key during development
)

# Load the character descriptions dataset
def load_dataset(file_path='data/character_descriptions.csv'):
    try:
        characters_df = pd.read_csv(file_path)
        print(f"Dataset shape: {characters_df.shape}")
        print(f"Number of rows: {len(characters_df)}")
        print("\nFirst few rows:")
        print(characters_df.head())
        print("\nColumn names:", characters_df.columns.tolist())
        return characters_df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Helper function to safely extract column names
def get_character_columns(df):
    columns = df.columns.tolist()
    
    # Find the most likely column names for the fields we need
    character_col = None
    description_col = None
    medium_col = None
    setting_col = None
    
    # Check for character name column
    for col in columns:
        if col.lower() in ['character', 'name', 'character name']:
            character_col = col
            break
    
    # Check for description column
    for col in columns:
        if col.lower() in ['description', 'character description']:
            description_col = col
            break
    
    # Check for medium column
    for col in columns:
        if col.lower() in ['medium', 'media', 'type']:
            medium_col = col
            break
    
    # Check for setting column
    for col in columns:
        if col.lower() in ['setting', 'location', 'world']:
            setting_col = col
            break
    
    print(f"Character column: {character_col}")
    print(f"Description column: {description_col}")
    print(f"Medium column: {medium_col}")
    print(f"Setting column: {setting_col}")
    
    return character_col, description_col, medium_col, setting_col

# Create a function to generate the text field
def prepare_data(df):
    # Get column names
    character_col, description_col, medium_col, setting_col = get_character_columns(df)
    
    # Function to generate the text field
    def generate_text(row):
        text_parts = []
        
        if character_col and pd.notna(row[character_col]):
            text_parts.append(f"Character Name: {row[character_col]}")
        
        if description_col and pd.notna(row[description_col]):
            text_parts.append(f"Description: {row[description_col]}")
        
        if medium_col and pd.notna(row[medium_col]):
            text_parts.append(f"Medium: {row[medium_col]}")
        
        if setting_col and pd.notna(row[setting_col]):
            text_parts.append(f"Setting: {row[setting_col]}")
        
        return "\n".join(text_parts)

    # Apply the function to create the text column
    df['text'] = df.apply(generate_text, axis=1)

    # Display the result
    print("\nText column examples:")
    for i in range(min(3, len(df))):
        print(f"\nExample {i+1}:")
        print(df['text'].iloc[i])
        print("-" * 50)

    # Verify we have adequate rows with text data
    text_count = df['text'].notna().sum()
    print(f"\nNumber of rows with text data: {text_count}")
    print(f"Final columns: {df.columns.tolist()}")
    
    return df, character_col, description_col, medium_col, setting_col

# Function to make OpenAI API calls with retry logic
def call_openai_api(func, *args, **kwargs):
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"API error on attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise e

# Function for basic response (without custom data)
def get_basic_response(query):
    try:
        print("Sending basic query to OpenAI...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        
        # Using the OpenAI API to get a response
        response = call_openai_api(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Extract the response text
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error getting basic response: {str(e)}")
        # Return a descriptive error message
        return f"Error: Could not get a response from the API. {str(e)}"

# Function to build context from our dataset for custom response
def build_context_for_query(query, df, character_col, description_col, medium_col, setting_col, max_length=4000):
    """
    Build context for a query by finding the most relevant character descriptions
    from our dataset using simple matching logic without embeddings.
    """
    query = query.lower()
    
    # First try exact name matches (highest priority)
    matched_rows = []
    
    # Try to find character names in the query
    if character_col:
        character_matches = df[df[character_col].str.lower().str.contains(query, na=False)]
        if not character_matches.empty:
            matched_rows.append(character_matches)
    
    # If we don't have direct name matches, look for setting or medium matches
    if not matched_rows and setting_col:
        setting_matches = df[df[setting_col].str.lower().str.contains(query, na=False)]
        if not setting_matches.empty:
            matched_rows.append(setting_matches)
    
    if not matched_rows and medium_col:
        medium_matches = df[df[medium_col].str.lower().str.contains(query, na=False)]
        if not medium_matches.empty:
            matched_rows.append(medium_matches)
    
    # If still no matches, add some random entries to provide some context
    if not matched_rows:
        # Take a random sample of 5 entries
        matched_rows.append(df.sample(min(5, len(df))))
    
    # Combine all matched rows and remove duplicates
    relevant_df = pd.concat(matched_rows).drop_duplicates()
    
    # Build the context string
    context = ""
    for _, row in relevant_df.iterrows():
        text = row['text']
        if len(context) + len(text) + 4 > max_length:
            break
        context += text + "\n\n"
    
    print(f"Built context with {len(relevant_df)} character descriptions")
    return context

# Function for custom response (with custom data)
def get_custom_response(query, df, character_col, description_col, medium_col, setting_col):
    try:
        print("Building context for custom query...")
        # Get relevant context for the query
        context = build_context_for_query(query, df, character_col, description_col, medium_col, setting_col)
        
        # Create system prompt with our context
        system_prompt = f"""You are a helpful assistant with specialized knowledge about fictional characters.
        
Below is detailed information about various fictional characters that you should reference when answering questions:

{context}

When answering questions about these characters, provide detailed and accurate information based on the descriptions above.
If asked about a character not in your specialized knowledge, acknowledge this limitation."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        print("Sending custom query to OpenAI...")
        # Using the OpenAI API to get a response
        response = call_openai_api(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Extract the response text
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error getting custom response: {str(e)}")
        # Return a descriptive error message
        return f"Error: Could not get a response from the API. {str(e)}"

# Function to compare basic and custom responses
def compare_responses(question, df, character_col, description_col, medium_col, setting_col):
    print(f"QUESTION: {question}\n")
    print("-" * 80)
    
    print("BASIC RESPONSE (without character data):")
    basic_answer = get_basic_response(question)
    print(basic_answer)
    
    print("\n" + "-" * 80 + "\n")
    
    print("CUSTOM RESPONSE (with character data):")
    custom_answer = get_custom_response(question, df, character_col, description_col, medium_col, setting_col)
    print(custom_answer)
    
    print("\n" + "=" * 80 + "\n")
    
    return basic_answer, custom_answer

# Main execution code
def main():
    # Step 1 & 2: Load and prepare the dataset
    characters_df = load_dataset()
    if characters_df is None:
        print("Could not load dataset. Exiting.")
        return
    
    characters_df, character_col, description_col, medium_col, setting_col = prepare_data(characters_df)
    
    # Step 3 & 4: Test with questions that demonstrate different behaviors
    # Question 1: Specific character question
    print("DEMONSTRATING CUSTOM PERFORMANCE - QUESTION 1\n")
    question1 = "Tell me about Dr. Eliza Chen. What is her background and personality?"
    basic_answer1, custom_answer1 = compare_responses(question1, characters_df, character_col, description_col, medium_col, setting_col)
    
    # Question 2: Question about science fiction characters
    print("DEMONSTRATING CUSTOM PERFORMANCE - QUESTION 2\n")
    question2 = "Describe the characters who appear in science fiction settings. What are their traits?"
    basic_answer2, custom_answer2 = compare_responses(question2, characters_df, character_col, description_col, medium_col, setting_col)

if __name__ == "__main__":
    main()
