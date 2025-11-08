import google.generativeai as genai
import yfinance as yf
import re
from config import GEMINI_API_KEY, EXCHANGES
import os

class TickerFinder:
    """
    Uses Google Gemini AI to find stock ticker symbols for Indian companies
    listed on NSE (National Stock Exchange) or BSE (Bombay Stock Exchange).
    """
    
    def __init__(self):
        """Initialize Gemini API and configure the model"""
        try:
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.0-flash-001')
            print("✅ Gemini API configured successfully")
        except Exception as e:
            print(f"❌ Error configuring Gemini API: {e}")
            print("Please check your API key in config.py")
            raise
    
    def find_ticker(self, company_name):
        """
        Find the Yahoo Finance ticker symbol for a company using Gemini AI
        Now follows: User Input → Registered Company Name → Ticker
        
        Parameters:
        company_name (str): Company name provided by user
        
        Returns:
        str: Ticker symbol with exchange suffix (.NS or .BO) or None if not found
        """
        if not company_name or company_name.strip() == "":
            print("❌ Error: Company name cannot be empty")
            return None
        
        print(f"\n🔍 Step 1: Finding registered company name for '{company_name}'...")
        
        # Step 1: Use Gemini to find registered company name, then ticker
        ticker_candidates = self._query_gemini_for_ticker(company_name)
        
        if not ticker_candidates:
            print(f"❌ Could not find any ticker candidates for '{company_name}'")
            print(f"   This could mean:")
            print(f"   • The company is not listed on NSE or BSE")
            print(f"   • The company name is incorrect")
            print(f"   • The company has no listed parent company")
            return None
        
        # Step 2: Validate each candidate ticker
        print(f"\n🔎 Step 2: Validating ticker candidates: {ticker_candidates}")
        
        for ticker in ticker_candidates:
            if self._validate_ticker(ticker):
                # Get company info to show what was found
                info = self.get_company_info(ticker)
                if info:
                    print(f"\n✅ Successfully Matched:")
                    print(f"   User Input: '{company_name}'")
                    print(f"   Registered Name: {info['name']}")
                    print(f"   Ticker: {info['ticker']}")
                    print(f"   Exchange: {info['exchange']}")
                return ticker
        
        # If no candidates validated
        print(f"\n❌ None of the ticker candidates could be validated.")
        print(f"   Please ensure the company is listed on NSE or BSE.")
        return None
    
    def _query_gemini_for_ticker(self, company_name):
        """
        Query Gemini AI to find the registered company name first, then derive ticker
        
        Parameters:
        company_name (str): Company name to search for
        
        Returns:
        list: List of potential ticker symbols with exchange suffixes
        """
        # STEP 1: First ask Gemini to find the registered company name
        name_prompt = f"""You are a financial expert specializing in Indian stock markets.

    Task: Find the EXACT REGISTERED COMPANY NAME for a company listed on Indian stock exchanges (NSE or BSE).

    User Input: "{company_name}"

    CRITICAL INSTRUCTIONS:

    1. FIND THE REGISTERED COMPANY NAME:
    - Identify the official registered name of the company in Indian stock exchanges
    - Handle partial names (e.g., "Jio" → "Reliance Industries Limited")
    - Handle brand names (e.g., "Zomato" → "Zomato Limited")
    - Handle abbreviations (e.g., "TCS" → "Tata Consultancy Services Limited")
    - Handle misspellings and fuzzy matches

    2. PARENT COMPANY LOGIC (VERY IMPORTANT):
    - If the input is a subsidiary/brand NOT directly listed, find the PARENT company's registered name
    - Use the same fuzzy matching and validation techniques
    - ALWAYS check for and return the parent company's registered name if applicable
    - Examples:
        * "Jio" or "Reliance Jio" → "Reliance Industries Limited" (parent company)
        * "Big Bazaar" → Look for parent company's registered name
        * "Tanishq" → "Titan Company Limited" (parent company)
    
    3. RESPONSE FORMAT (STRICT):
    - If found: Return ONLY the exact registered company name 
    - Example: "Reliance Industries Limited"
    - If not found: Return ONLY "NOT_FOUND"
    - NO explanations, NO extra text, NO ticker symbols

    Examples:
    Input: "reliance" → Reliance Industries Limited
    Input: "jio" → Reliance Industries Limited
    Input: "tcs" → Tata Consultancy Services Limited
    Input: "hdfc bank" → HDFC Bank Limited
    Input: "nonexistent company" → NOT_FOUND

    Now find the registered company name for: "{company_name}"
    """
        
        try:
            # Get the registered company name from Gemini
            response = self.model.generate_content(name_prompt)
            company_registered_name = response.text.strip()
            
            print(f"\n📡 Gemini Found Company Name: {company_registered_name}")
            
            # Check if company not found
            if "NOT_FOUND" in company_registered_name.upper():
                return []
            
            # STEP 2: Now find the ticker based on the registered company name
            ticker_prompt = f"""You are a financial data expert for Indian stock markets.

    Task: Find the Yahoo Finance ticker symbol for this EXACT company.

    Registered Company Name: "{company_registered_name}"

    INSTRUCTIONS:
    1. Find the ticker symbol for this exact company on NSE or BSE
    2. Ticker format:
    - NSE (preferred): Add .NS suffix (e.g., RELIANCE.NS)
    - BSE (alternative): Add .BO suffix (e.g., RELIANCE.BO)
    3. If listed on both exchanges, provide BOTH (NSE first, then BSE)
    4. If ticker not found, respond: "NOT_FOUND"

    RESPONSE FORMAT (STRICT):
    - Return ONLY ticker symbol(s), one per line
    - Example for both exchanges:
    RELIANCE.NS
    RELIANCE.BO
    - Example for NSE only:
    TCS.NS
    - NO explanations, NO extra text

    Find ticker for: "{company_registered_name}"
    """
            
            ticker_response = self.model.generate_content(ticker_prompt)
            ticker_text = ticker_response.text.strip()
            
            print(f"📡 Gemini Found Ticker(s): {ticker_text}")
            
            # Check if ticker not found
            if "NOT_FOUND" in ticker_text.upper():
                return []
            
            # Parse the ticker response
            tickers = self._parse_ticker_response(ticker_text)
            return tickers
            
        except Exception as e:
            print(f"❌ Error querying Gemini API: {e}")
            return []
    
    def _parse_ticker_response(self, response_text):
        """
        Parse Gemini's response to extract valid ticker symbols
        
        Parameters:
        response_text (str): Raw response from Gemini
        
        Returns:
        list: List of extracted ticker symbols
        """
        tickers = []
        
        # Split response by lines and clean up
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for patterns like "TICKER.NS" or "TICKER.BO"
            # Allow for potential extra text in the response
            matches = re.findall(r'([A-Z0-9&\-]+\.(?:NS|BO))', line, re.IGNORECASE)
            
            for match in matches:
                ticker = match.upper()
                if ticker not in tickers:
                    tickers.append(ticker)
        
        # If no regex matches, try to extract from plain text
        if not tickers:
            # Check if response contains .NS or .BO
            words = response_text.upper().split()
            for word in words:
                if '.NS' in word or '.BO' in word:
                    # Clean up the word
                    ticker = re.sub(r'[^A-Z0-9.\-&]', '', word)
                    if ticker and (ticker.endswith('.NS') or ticker.endswith('.BO')):
                        if ticker not in tickers:
                            tickers.append(ticker)
        
        return tickers
    
    def _validate_ticker(self, ticker):
        """
        Validate that a ticker symbol exists and has data on Yahoo Finance
        
        Parameters:
        ticker (str): Ticker symbol to validate (e.g., "RELIANCE.NS")
        
        Returns:
        bool: True if ticker is valid and has data, False otherwise
        """
        try:
            print(f"   Checking {ticker}...", end=" ")
            
            # Try to fetch basic info from Yahoo Finance
            stock = yf.Ticker(ticker)
            
            # Try to get recent data (last 5 days)
            hist = stock.history(period="5d")
            
            # Check if we got any data
            if hist.empty:
                print("❌ No data available")
                return False
            
            # Check if we have closing prices
            if 'Close' not in hist.columns or hist['Close'].isna().all():
                print("❌ No price data")
                return False
            
            # Get company info to verify it's a real stock
            info = stock.info
            
            # Check if we got meaningful info
            if not info or len(info) < 5:
                print("❌ Insufficient information")
                return False
            
            # Successful validation
            company_name = info.get('longName', info.get('shortName', 'Unknown'))
            current_price = hist['Close'].iloc[-1]
            print(f"✅ Valid - {company_name} (₹{current_price:.2f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {str(e)[:50]}")
            return False
    
    def get_company_info(self, ticker):
        """
        Get detailed company information for a validated ticker
        
        Parameters:
        ticker (str): Validated ticker symbol
        
        Returns:
        dict: Company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            company_info = {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 'N/A'),
                'exchange': 'NSE' if '.NS' in ticker else 'BSE'
            }
            
            return company_info
            
        except Exception as e:
            print(f"❌ Error fetching company info: {e}")
            return None
    
    def interactive_search(self):
        """
        Interactive mode for testing ticker search functionality
        """
        print(" " * 15 + "TICKER FINDER - INTERACTIVE MODE")
        print("\nExamples of what you can enter:")
        print("  - Reliance")
        print("  - TCS")
        print("  - Tata Motors")
        print("  - HDFC Bank")
        print("  - Infosys")
        print("\nType 'quit' to exit\n")
        
        while True:
            company_name = input("Enter company name: ").strip()
            
            if company_name.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not company_name:
                print("❌ Please enter a company name\n")
                continue
            
            # Find ticker
            ticker = self.find_ticker(company_name)
            
            if ticker:
                # Get additional info
                info = self.get_company_info(ticker)
                if info:
                    print(f"\n📊 Company Details:")
                    print(f"   Name: {info['name']}")
                    print(f"   Ticker: {info['ticker']}")
                    print(f"   Exchange: {info['exchange']}")
                    print(f"   Sector: {info['sector']}")
                    print(f"   Industry: {info['industry']}")
            
            


# Test the ticker finder independently
if __name__ == "__main__":
    finder = TickerFinder()
    finder.interactive_search()