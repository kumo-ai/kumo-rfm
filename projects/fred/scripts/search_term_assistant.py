#!/usr/bin/env python3
"""
FRED Search Term Assistant

Interactive chat to derive relevant FRED search terms.
Supports: Ollama (local), Claude API, OpenAI API via curl.
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# FRED topic categories for filtering
FRED_TOPICS = {
    'gdp': ['GDP', 'Gross Domestic Product', 'Real GDP', 'Nominal GDP', 'GDP Growth'],
    'inflation': ['CPI', 'Consumer Price Index', 'PPI', 'Producer Price Index', 'PCE', 'inflation', 'deflation'],
    'employment': ['unemployment', 'unemployment rate', 'employment', 'nonfarm payrolls', 'PAYEMS', 'labor force', 'jobless claims'],
    'wages': ['wages', 'average hourly earnings', 'compensation', 'labor cost'],
    'interest_rates': ['federal funds rate', 'interest rate', 'treasury yield', '10-year treasury', 'prime rate', 'mortgage rate', 'LIBOR'],
    'money_supply': ['M1', 'M2', 'M3', 'money supply', 'monetary base', 'bank reserves'],
    'housing': ['housing starts', 'building permits', 'home sales', 'home prices', 'Case-Shiller', 'ZHVI', 'rent', 'mortgage'],
    'consumer': ['consumer spending', 'retail sales', 'consumer confidence', 'personal income', 'disposable income', 'savings rate'],
    'business': ['industrial production', 'capacity utilization', 'manufacturing', 'durable goods', 'ISM', 'business inventories', 'productivity'],
    'trade': ['exports', 'imports', 'trade balance', 'trade deficit', 'current account', 'exchange rate'],
    'government': ['government spending', 'federal debt', 'deficit', 'budget', 'government receipts'],
    'markets': ['S&P 500', 'stock market', 'VIX', 'volatility', 'corporate profits', 'dividend yield'],
    'commodities': ['oil prices', 'WTI', 'Brent', 'natural gas', 'gold', 'commodity prices', 'food prices'],
    'credit': ['credit', 'loans', 'consumer credit', 'commercial credit', 'credit conditions']
}


class SearchTermAssistant:
    """Chat-based assistant to derive FRED search terms."""
    
    def __init__(self, provider: str = 'ollama', model: Optional[str] = None):
        """
        Initialize assistant.
        
        Args:
            provider: 'ollama', 'claude', or 'openai'
            model: Model name (default varies by provider)
        """
        self.provider = provider
        self.model = model or self._get_default_model()
        self.conversation = []
        self.chat_name = None
        self.chat_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check availability
        if provider == 'ollama':
            if not self._check_ollama():
                raise RuntimeError("Ollama server not running. Start with: ollama serve")
        elif provider == 'claude':
            if not os.getenv('ANTHROPIC_API_KEY'):
                raise RuntimeError("ANTHROPIC_API_KEY not set")
        elif provider == 'openai':
            if not os.getenv('OPENAI_API_KEY'):
                raise RuntimeError("OPENAI_API_KEY not set")
    
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            'ollama': 'llama3.2',
            'claude': 'claude-3-haiku-20240307',
            'openai': 'gpt-3.5-turbo'
        }
        return defaults.get(self.provider, 'llama3.2')
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/tags'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API via curl."""
        # For Ollama, maintain conversation history in prompt
        if self.conversation:
            # Build full conversation context
            context = "You are helping a researcher identify economic data series from FRED.\n\n"
            for msg in self.conversation:
                role = msg['role'].capitalize()
                context += f"{role}: {msg['content']}\n\n"
            prompt = context + f"User: {prompt}\n\nAssistant:"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/generate',
                 '-d', json.dumps(payload)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                return response.get('response', '').strip()
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error calling Ollama: {e}"
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API via curl."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        messages = [{"role": "user", "content": prompt}]
        if self.conversation:
            messages = self.conversation + messages
        
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": messages
        }
        
        try:
            result = subprocess.run([
                'curl', '-s', 'https://api.anthropic.com/v1/messages',
                '-H', f'x-api-key: {api_key}',
                '-H', 'anthropic-version: 2023-06-01',
                '-H', 'content-type: application/json',
                '-d', json.dumps(payload)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                if 'content' in response:
                    return response['content'][0]['text'].strip()
                elif 'error' in response:
                    return f"API Error: {response['error'].get('message', 'Unknown error')}"
            return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error calling Claude: {e}"
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API via curl."""
        api_key = os.getenv('OPENAI_API_KEY')
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant for economic research."},
            {"role": "user", "content": prompt}
        ]
        if self.conversation:
            messages = self.conversation + [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 512
        }
        
        try:
            result = subprocess.run([
                'curl', '-s', 'https://api.openai.com/v1/chat/completions',
                '-H', f'Authorization: Bearer {api_key}',
                '-H', 'Content-Type: application/json',
                '-d', json.dumps(payload)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                if 'choices' in response:
                    return response['choices'][0]['message']['content'].strip()
                elif 'error' in response:
                    return f"API Error: {response['error'].get('message', 'Unknown error')}"
            return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error calling OpenAI: {e}"
    
    def call_llm(self, prompt: str) -> str:
        """Call the configured LLM."""
        if self.provider == 'ollama':
            return self._call_ollama(prompt)
        elif self.provider == 'claude':
            return self._call_claude(prompt)
        elif self.provider == 'openai':
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _summarize_text(self, text: str, max_words: int = 5) -> str:
        """Create a summarized slug from text."""
        # Remove special characters and split into words
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        
        # Take first max_words words
        summary_words = words[:max_words]
        
        # Join with underscores
        slug = '_'.join(summary_words)
        
        # Limit total length to 50 chars
        if len(slug) > 50:
            slug = slug[:50]
        
        return slug if slug else "chat"
    
    def start_conversation(self) -> List[str]:
        """Start interactive conversation."""
        print("\n" + "="*70)
        print("FRED Search Term Assistant")
        print("="*70)
        print(f"\nProvider: {self.provider}")
        print(f"Model: {self.model}")
        print("\nDescribe your economic research or analysis goals.")
        print("I'll help you identify relevant FRED data series.")
        print("\nStart chatting, and I'll auto-generate a filename.")
        print("Commands: 'done' (finish), 'quit' (exit)\n")
        
        # Don't ask for chat name upfront - will auto-generate from first message
        self.chat_name = None
        
        print("="*70 + "\n")
        
        # Initial system context
        system_context = """You are helping a researcher identify economic data series from FRED (Federal Reserve Economic Data).

Your goal is to:
1. Understand their research question or analysis goals
2. Identify relevant economic indicators they need
3. Suggest specific search terms for the FRED API

Ask clarifying questions and be concise. When you identify search terms, list them clearly."""
        
        full_conversation = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("\nExiting.")
                    sys.exit(0)
                
                if user_input.lower() == 'done':
                    break
                
                # Auto-generate chat name from first user message
                if self.chat_name is None:
                    self.chat_name = self._summarize_text(user_input, max_words=5)
                    print(f"Chat name: {self.chat_name}_{self.chat_date}\n")
                
                # Build prompt
                if len(full_conversation) == 0:
                    prompt = f"{system_context}\n\nUser: {user_input}\n\nAssistant:"
                else:
                    prompt = user_input
                
                # Get response
                print("Assistant: ", end='', flush=True)
                response = self.call_llm(prompt)
                print(response + "\n")
                
                # Store conversation
                full_conversation.append(f"User: {user_input}")
                full_conversation.append(f"Assistant: {response}")
                
                # Update conversation history for context
                self.conversation.append({"role": "user", "content": user_input})
                self.conversation.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'done' to finish or 'quit' to exit.")
                continue
            except Exception as e:
                print(f"\nError: {e}")
                continue
        
        # Save conversation
        conv_file = self.save_conversation(full_conversation)
        print(f"\nConversation saved to: {conv_file}")
        
        # Extract and filter terms
        terms = self.extract_and_filter_terms(full_conversation)
        
        return terms
    
    def save_conversation(self, conversation: List[str]) -> str:
        """Save conversation to file."""
        filename = f"conversation_{self.chat_name}_{self.chat_date}.txt"
        
        with open(filename, 'w') as f:
            f.write("# FRED Search Term Assistant - Conversation Log\n")
            f.write(f"# Chat: {self.chat_name}\n")
            f.write(f"# Date: {self.chat_date}\n")
            f.write(f"# Provider: {self.provider} ({self.model})\n\n")
            
            for line in conversation:
                f.write(line + "\n\n")
        
        return filename
    
    def extract_and_filter_terms(self, conversation: List[str]) -> List[str]:
        """Extract search terms from conversation and filter through FRED topics."""
        print("\n" + "="*70)
        print("Extracting and filtering search terms...")
        print("="*70 + "\n")
        
        # Combine all conversation text
        full_text = " ".join(conversation).lower()
        
        # Find matching FRED topics
        matched_terms = set()
        matched_categories = []
        
        for category, terms in FRED_TOPICS.items():
            for term in terms:
                if term.lower() in full_text:
                    matched_terms.add(term)
                    if category not in matched_categories:
                        matched_categories.append(category)
        
        if matched_terms:
            print(f"Found {len(matched_terms)} relevant FRED search terms")
            print(f"Categories: {', '.join(matched_categories)}\n")
            
            # Show terms by category
            for category in matched_categories:
                category_terms = [t for t in matched_terms if t in FRED_TOPICS[category]]
                if category_terms:
                    print(f"\n{category.upper().replace('_', ' ')}:")
                    for term in sorted(category_terms):
                        print(f"  - {term}")
        else:
            print("No FRED-relevant terms found in conversation.")
            print("Consider mentioning specific economic indicators.")
        
        return sorted(matched_terms)
    
    def save_terms(self, terms: List[str]) -> str:
        """Save search terms to file."""
        filename = f"search_terms_{self.chat_name}_{self.chat_date}.txt"
        
        with open(filename, 'w') as f:
            f.write("# FRED Search Terms\n")
            f.write(f"# Generated from: {self.chat_name}\n")
            f.write(f"# Date: {self.chat_date}\n")
            f.write(f"# Provider: {self.provider}\n")
            f.write(f"# Total terms: {len(terms)}\n\n")
            
            for term in terms:
                f.write(f"{term}\n")
        
        return filename


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive chat to derive FRED search terms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Ollama (local, free)
  python3 search_term_assistant.py --provider ollama
  
  # Use Claude API
  export ANTHROPIC_API_KEY='your-key'
  python3 search_term_assistant.py --provider claude
  
  # Use OpenAI
  export OPENAI_API_KEY='your-key'
  python3 search_term_assistant.py --provider openai
  
  # Custom model
  python3 search_term_assistant.py --provider ollama --model llama2:13b

After generating terms:
  python3 01_fetch_fred_data.py --from-file search_terms_*.txt
        """
    )
    
    parser.add_argument('--provider', choices=['ollama', 'claude', 'openai'],
                       default='ollama',
                       help='LLM provider (default: ollama)')
    parser.add_argument('--model', type=str,
                       help='Model name (default varies by provider)')
    
    args = parser.parse_args()
    
    try:
        # Initialize assistant
        assistant = SearchTermAssistant(provider=args.provider, model=args.model)
        
        # Start conversation
        terms = assistant.start_conversation()
        
        if terms:
            # Save terms
            terms_file = assistant.save_terms(terms)
            
            print("\n" + "="*70)
            print("Next Steps")
            print("="*70)
            print(f"\n1. Review terms: cat {terms_file}")
            print(f"\n2. Fetch data:")
            print(f"   python3 01_fetch_fred_data.py --from-file {terms_file}")
            print(f"\n3. Parse data:")
            print(f"   python3 02_parse_fred_txt.py --input txt --output data --format parquet")
            print()
        else:
            print("\nNo terms extracted. Try mentioning specific economic indicators.")
    
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
