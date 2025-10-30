# Interactive Chatbot for FRED Search Terms

## Overview

Chat with an AI assistant to derive relevant FRED search terms. Uses curl for API calls and filters results through FRED topic categories.

## Features

- **3 LLM Providers**: Ollama (local), Claude API, OpenAI API
- **Curl-based**: All API calls via subprocess + curl
- **FRED Filtering**: Extracts only valid FRED topics
- **Saves Everything**: Conversation log + filtered terms
- **Timestamped**: `search_terms_{name}_{date}.txt`

## Quick Start

### Ollama (Local, Free)

```bash
# Start Ollama
ollama serve

# Run assistant
python3 search_term_assistant.py --provider ollama
```

### Claude (Recommended)

```bash
export ANTHROPIC_API_KEY='your-key'
python3 search_term_assistant.py --provider claude
```

### OpenAI

```bash
export OPENAI_API_KEY='your-key'
python3 search_term_assistant.py --provider openai
```

## How It Works

1. **Chat**: Describe your research goals
2. **AI Response**: Assistant asks clarifying questions
3. **Extraction**: System identifies FRED-relevant terms
4. **Filtering**: Matches against 14 FRED topic categories
5. **Output**: Saves filtered terms to file

## FRED Topic Categories

The system filters for these categories:
- GDP and growth
- Inflation (CPI, PPI, PCE)
- Employment and unemployment
- Wages and compensation
- Interest rates
- Money supply (M1, M2, reserves)
- Housing market
- Consumer spending
- Business indicators
- Trade and exchange rates
- Government debt/deficit
- Financial markets
- Commodities and energy
- Credit and loans

## Example Workflow

```bash
# 1. Generate terms via chat
python3 search_term_assistant.py --provider claude
> Chat name: housing_analysis
> You: studying housing prices and mortgage rates
> Assistant: [asks questions]
> You: done

# Output files created:
# - conversation_housing_analysis_20251029_220000.txt
# - search_terms_housing_analysis_20251029_220000.txt

# 2. Fetch FRED data
python3 01_fetch_fred_data.py --from-file search_terms_housing_analysis_*.txt

# 3. Parse and use
python3 02_parse_fred_txt.py --input txt --output data --format parquet
python3 recommend_series.py --text "housing market"
```

## Custom Models

```bash
# Ollama with specific model
python3 search_term_assistant.py --provider ollama --model llama2:13b

# Claude with different model
python3 search_term_assistant.py --provider claude --model claude-3-opus-20240229
```

## Output Files

### Conversation Log
`conversation_{name}_{date}.txt`
```
# FRED Search Term Assistant - Conversation Log
# Chat: housing_analysis
# Date: 20251029_220000
# Provider: claude (claude-3-haiku-20240307)

User: studying housing prices and mortgage rates
Assistant: [response with questions]
...
```

### Search Terms
`search_terms_{name}_{date}.txt`
```
# FRED Search Terms
# Generated from: housing_analysis
# Date: 20251029_220000
# Total terms: 8

housing starts
home prices
Case-Shiller
mortgage rate
housing market
building permits
home sales
ZHVI
```

## Tips

- **Be Specific**: Mention exact indicators ("CPI", "unemployment rate")
- **Use Categories**: Reference broad categories ("housing market", "inflation")
- **Ask Questions**: The assistant will clarify your needs
- **Type 'done'**: When ready to extract terms
- **Multiple Chats**: Run multiple sessions for different research areas

## Troubleshooting

### Ollama not found
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Start server
ollama serve

# Pull model (optional)
ollama pull llama2
```

### API Key errors
```bash
# Check env vars
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Set in .env file
cp .env.example .env
# Edit .env with your keys

# Or export directly
export ANTHROPIC_API_KEY='sk-ant-...'
```

### No terms found
- Be more specific with indicator names
- Mention FRED data series explicitly
- Use terminology from the FRED topic list

## Integration with Pipeline

The chatbot integrates seamlessly with your existing FRED data pipeline:

```bash
# 1. Interactive term generation
python3 search_term_assistant.py --provider claude

# 2. Fetch data for those terms
python3 01_fetch_fred_data.py --from-file search_terms_*.txt

# 3. Parse and convert
python3 02_parse_fred_txt.py --input txt --output data --format parquet

# 4. Build Kumo graph
python3 03_build_kumo_graph.py --input data/fred_data.parquet

# 5. Train RFM model
python3 04_train_kumo_rfm.py --graph-id your_graph_id

# 6. Get recommendations
python3 recommend_series.py --text "housing market" --top 10
```

## Advanced Usage

### Batch Research Sessions

Create multiple focused chat sessions:

```bash
# Session 1: Inflation analysis
python3 search_term_assistant.py --provider claude
# > Chat name: inflation_2024

# Session 2: Labor market
python3 search_term_assistant.py --provider claude  
# > Chat name: labor_trends

# Combine terms from both
cat search_terms_inflation_*.txt search_terms_labor_*.txt > combined_terms.txt
python3 01_fetch_fred_data.py --from-file combined_terms.txt
```

### Using Different Models

```bash
# Larger Ollama model for complex queries
python3 search_term_assistant.py --provider ollama --model llama2:70b

# Claude Opus for nuanced research
python3 search_term_assistant.py --provider claude --model claude-3-opus-20240229

# GPT-4 for detailed analysis
python3 search_term_assistant.py --provider openai --model gpt-4
```

## Next Steps

1. **Try it**: Run your first chat session
2. **Review output**: Check generated terms for relevance
3. **Iterate**: Refine your descriptions if needed
4. **Fetch data**: Use terms with your FRED pipeline
5. **Analyze**: Feed into Kumo RFM for recommendations

## Support

For issues or questions:
- Check conversation logs for context
- Review FRED topic categories list
- Try different LLM providers
- Adjust specificity of your descriptions
