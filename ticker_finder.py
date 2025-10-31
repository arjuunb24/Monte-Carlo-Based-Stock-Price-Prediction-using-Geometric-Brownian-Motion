import google.generativeai as genai
import yfinance as yf
import re
from config import GEMINI_API_KEY, EXCHANGES

class TickerFinder:
    """
    Uses Google Gemini AI to find stock ticker symbols for Indian companies
    listed on NSE (National Stock Exchange) or BSE (Bombay Stock Exchange).
    """
    
    def __init__(self):
        """Initialize Gemini API and configure the model"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini API configured successfully")
        except Exception as e:
            print(f"❌ Error configuring Gemini API: {e}")
            print("Please check your API key in config.py")
            raise
    
    def find_ticker(self, company_name):
        """
        Find the Yahoo Finance ticker symbol for a company using Gemini AI
        
        Parameters:
        company_name (str): Company name (can be informal, partial, or formal name)
        
        Returns:
        str: Ticker symbol with exchange suffix (.NS or .BO) or None if not found
        """
        if not company_name or company_name.strip() == "":
            print("❌ Error: Company name cannot be empty")
            return None
        
        print(f"\n🔍 Searching for ticker symbol for: '{company_name}'")
        
        # Step 1: Use Gemini to find the ticker
        ticker_candidates = self._query_gemini_for_ticker(company_name)
        
        if not ticker_candidates:
            print(f"❌ Could not find any ticker candidates for '{company_name}'")
            return None
        
        # Step 2: Validate each candidate ticker
        print(f"\n🔎 Validating ticker candidates: {ticker_candidates}")
        
        for ticker in ticker_candidates:
            if self._validate_ticker(ticker):
                print(f"✅ Successfully validated ticker: {ticker}")
                return ticker
        
        # If no candidates validated, inform user
        print(f"\n❌ None of the ticker candidates could be validated.")
        print(f"   The company '{company_name}' may not be listed on NSE or BSE,")
        print(f"   or the company name might be incorrect.")
        return None
    
    def _query_gemini_for_ticker(self, company_name):
        """
        Query Gemini AI to find ticker symbol(s) for the given company
        
        Parameters:
        company_name (str): Company name to search for
        
        Returns:
        list: List of potential ticker symbols with exchange suffixes
        """
        # Craft a detailed prompt for Gemini
        prompt = f"""You are a financial expert specializing in Indian stock markets.

Task: Find the Yahoo Finance ticker symbol for a company listed on Indian stock exchanges (NSE or BSE).

Company Name: "{company_name}"

Instructions:
1. Identify the company even if the name provided is:
   - Informal (e.g., "Reliance" for "Reliance Industries Limited")
   - Abbreviated (e.g., "TCS" for "Tata Consultancy Services")
   - Misspelled or partially correct
   - A brand name rather than official company name

2. Provide the Yahoo Finance ticker format:
   - For NSE (National Stock Exchange): Add .NS suffix (e.g., RELIANCE.NS)
   - For BSE (Bombay Stock Exchange): Add .BO suffix (e.g., RELIANCE.BO)

3. If the company is listed on both exchanges, provide BOTH tickers (NSE first, then BSE)

4. If the company doesn't exist or isn't listed on Indian exchanges, respond with: "NOT_FOUND"

5. Response format (very important):
   - If found: Return ONLY the ticker symbol(s), one per line
   - Example response for a company on both exchanges:
     RELIANCE.NS
     RELIANCE.BO
   - Example response for NSE only:
     TCS.NS
   - If not found:
     NOT_FOUND

Do not include any explanation, just the ticker symbol(s) or "NOT_FOUND".

Company to search: "{company_name}"
"""
        
        try:
            # Query Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            print(f"\n📡 Gemini Response: {response_text}")
            
            # Check if company not found
            if "NOT_FOUND" in response_text.upper():
                return []
            
            # Parse the response to extract tickers
            tickers = self._parse_ticker_response(response_text)
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
        print("\n" + "="*60)
        print(" " * 15 + "TICKER FINDER - INTERACTIVE MODE")
        print("="*60)
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
            
            print("\n" + "-"*60 + "\n")


# Test the ticker finder independently
if __name__ == "__main__":
    finder = TickerFinder()
    finder.interactive_search()