# AI Agents for Product Review Analysis and Generation

## A Thesis Project for the Master's Degree in Data Analytics for Business

This repository contains the implementation of an AI recommendation system based on LLMs and Sentiment Analysis, developed as part of a Master's thesis in Data Analytics for Business.
Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023


## Project Overview

This project develops an intelligent agent capable of providing personalized product recommendations by analyzing and understanding user preferences, past purchase behaviors, and product reviews. The system leverages Large Language Models (LLMs) and Sentiment Analysis techniques to process and generate insights from textual data.

### Key Features

- Natural language interaction for product recommendations (e.g., "I want to buy a TV, which one do you recommend?")
- Personalized recommendations based on user purchase history
- Review analysis using sentiment analysis and vector embeddings
- Semantic search for finding relevant product reviews
- Integration with user database and purchase history

## System Architecture

The recommendation agent connects to:
1. A relational database storing:
   - User profiles
   - Purchase history
2. A vector database containing:
   - Embedded product reviews for semantic similarity search

## Installation

### Prerequisites
- Python 3.8+
- Poetry package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/biguz-commits/thesis

```

2. Install dependencies:
```bash
poetry install
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your database credentials and API keys
```

## Usage

### Running the recommendation agent:

```bash
poetry run dev
```

### Example queries:

```
"I want to buy a TV, which one do you recommend?"
"What's the best laptop for a computer science student?"
"Can you suggest a coffee machine under $100?"
```

## Project Structure

```
thesis-ai-recommendation/
├── data/                  # Data processing scripts and sample data
├── models/                # ML models and embeddings
├── src/                   # Source code
│   ├── agent/             # Recommendation agent logic
│   ├── database/          # Database connectors
│   ├── embeddings/        # Vector embedding utilities
│   ├── nlp/               # NLP processing modules
│   └── sentiment/         # Sentiment analysis components
├── tests/                 # Unit and integration tests
├── main.py                # Application entry point
├── pyproject.toml         # Poetry configuration
└── README.md              # This file
```

## License

This project is academic research and is not licensed for commercial use without permission.

## Author

[Your Name]  
Master's Student in Data Analytics for Business  
[Your University]
