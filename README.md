# Character-Aware Chatbot

A specialized chatbot system that provides enhanced responses about fictional characters by leveraging custom data integration with OpenAI's API.

## 🌟 Overview

This project creates an intelligent chatbot that combines the power of OpenAI's GPT models with a custom dataset of fictional character descriptions. The system demonstrates how domain-specific knowledge can significantly improve the quality and accuracy of AI responses when discussing fictional characters from various media.

## ✨ Features

- **Dataset Integration**: Automatically loads and processes character description data
- **Smart Context Building**: Identifies relevant character information based on user queries
- **Comparison Framework**: Demonstrates the difference between basic and knowledge-enhanced responses
- **Resilient API Handling**: Implements retry logic for reliable API communication
- **Flexible Column Detection**: Automatically identifies relevant columns in different dataset formats

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- OpenAI API access
- Pandas and NumPy libraries

### Installation

1. Clone this repository:
```bash
git clone https://github.com/YourGitHubUsername/character-aware-chatbot.git
cd character-aware-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install pandas numpy openai
```

4. Configure your API keys:
   - Replace `"YOUR API KEY"` in the code with your actual OpenAI API key

### Data Format

The chatbot expects a CSV file with columns for:
- Character name
- Character description
- Medium (book, movie, TV show, etc.)
- Setting (world or location)

Place your CSV file in the `data/` directory as `character_descriptions.csv`.

## 🧠 How It Works

1. **Data Loading**: The system loads character descriptions from the CSV file
2. **Data Preparation**: Column identification and text field generation
3. **Context Building**: When a user asks a question, the system finds the most relevant character data
4. **Enhanced Responses**: The chatbot provides responses enhanced with specific character knowledge

## 📊 Performance Comparison

The system demonstrates improved performance by comparing:
- **Basic responses**: Standard GPT responses without additional context
- **Custom responses**: Enhanced responses using the character dataset

This comparison highlights how domain-specific knowledge integration can significantly improve chatbot performance for specialized topics.

## 🛠️ Customization

You can customize the chatbot by:
- Modifying the system prompt in the `get_custom_response` function
- Adding additional data fields to the character descriptions
- Adjusting the matching logic in `build_context_for_query`
- Changing the OpenAI model used for responses

## 📝 License

MIT License

Copyright (c) 2025 Alex Kuzmin

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Alex Kuzmin  
GitHub: [YourGitHubUsername](https://github.com/alexander-kuzmin-us)  
Email: [your.email@example.com](mailto:alex.kuzminn@gmail.com)
