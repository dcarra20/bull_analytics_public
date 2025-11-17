"""
Data Loader Module
Loads S&P 500 company data from CSV files and enriches with stock statistics.
"""

import csv
from typing import List, Dict, Optional
from collections import defaultdict
import statistics


def load_companies(csv_path: str) -> List[Dict[str, str]]:
    """
    Load company data from CSV file.
    
    Args:
        csv_path: Path to the sp500_companies.csv file
        
    Returns:
        List of dictionaries containing company information
    """
    companies = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract relevant fields
            company = {
                'symbol': row['Symbol'],
                'name': row['Shortname'],
                'longname': row['Longname'],
                'sector': row['Sector'],
                'industry': row['Industry'],
                'description': row['Longbusinesssummary'],
                'city': row['City'],
                'state': row['State'],
                'country': row['Country'],
                'employees': row['Fulltimeemployees'],
                'current_price': row['Currentprice'],
                'market_cap': row['Marketcap']
            }
            companies.append(company)
    
    print(f"Loaded {len(companies)} companies from {csv_path}")
    return companies


def load_stock_data(csv_path: str, symbol: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Load historical stock price data.
    
    Args:
        csv_path: Path to sp500_stocks.csv
        symbol: Optional - load only specific symbol, None for all
        
    Returns:
        Dictionary mapping symbol to list of price records
    """
    stock_data = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows without price data
            if not row['Close'] or not row['Close'].strip():
                continue
            
            # Filter by symbol if specified
            if symbol and row['Symbol'] != symbol:
                continue
            
            stock_data[row['Symbol']].append({
                'date': row['Date'],
                'open': float(row['Open']) if row['Open'] else None,
                'high': float(row['High']) if row['High'] else None,
                'low': float(row['Low']) if row['Low'] else None,
                'close': float(row['Close']) if row['Close'] else None,
                'volume': int(float(row['Volume'])) if row['Volume'] else None
            })
    
    return dict(stock_data)


def calculate_stock_stats(stock_records: List[Dict]) -> Dict[str, any]:
    """
    Calculate summary statistics from stock price history.
    
    Args:
        stock_records: List of stock price records for a symbol
        
    Returns:
        Dictionary of calculated statistics
    """
    if not stock_records:
        return {}
    
    closes = [r['close'] for r in stock_records if r['close']]
    volumes = [r['volume'] for r in stock_records if r['volume']]
    
    if not closes:
        return {}
    
    # Calculate returns
    returns = []
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i-1]) / closes[i-1] * 100
        returns.append(ret)
    
    stats = {
        'first_date': stock_records[0]['date'],
        'last_date': stock_records[-1]['date'],
        'first_price': closes[0],
        'last_price': closes[-1],
        'min_price': min(closes),
        'max_price': max(closes),
        'avg_price': statistics.mean(closes),
        'total_return_pct': ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] > 0 else 0,
        'avg_volume': statistics.mean(volumes) if volumes else 0,
        'volatility': statistics.stdev(returns) if len(returns) > 1 else 0,
        'num_trading_days': len(closes)
    }
    
    return stats


def enriched_stock_data(companies: List[Dict], stock_csv_path: str) -> List[Dict]:
    """
    Enrich company data with stock statistics.
    
    Args:
        companies: List of company dictionaries
        stock_csv_path: Path to stock data CSV
        
    Returns:
        Enriched company list
    """
    print("Loading stock data for enrichment...")
    
    enriched_companies = []
    
    for i, company in enumerate(companies):
        symbol = company['symbol']
        
        # Load stock data for this symbol
        stock_data = load_stock_data(stock_csv_path, symbol=symbol)
        
        if symbol in stock_data:
            stats = calculate_stock_stats(stock_data[symbol])
            company['stock_stats'] = stats
        else:
            company['stock_stats'] = {}
        
        enriched_companies.append(company)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(companies)} companies...")
    
    print(f"Enrichment complete!")
    return enriched_companies


if __name__ == "__main__":
    # Test the loader
    companies = load_companies('sp500_companies.csv')