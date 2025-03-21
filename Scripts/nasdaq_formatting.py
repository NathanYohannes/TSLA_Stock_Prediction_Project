import pandas as pd
import argparse
import os
from datetime import datetime

def format_date(date_str):
    """Convert date from MM/DD/YYYY to YYYY-MM-DD format."""
    try:
        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format in {date_str}")
        return None

def clean_price(price_str):
    """Remove '$' and ',' from price strings and convert to float."""
    try:
        return float(price_str.replace('$', '').replace(',', ''))
    except ValueError:
        print(f"Error: Invalid price format in {price_str}")
        return None

def format_nasdaq_data(input_file, output_file):
    """Convert NASDAQ format to TSLA format."""
    try:
        # Read the input file
        df = pd.read_csv(input_file)
        
        # Rename columns to match TSLA format
        df = df.rename(columns={'Close/Last': 'Close'})
        
        # Convert date format
        df['Date'] = df['Date'].apply(format_date)
        
        # Clean price columns
        price_columns = ['Close', 'Open', 'High', 'Low']
        for col in price_columns:
            df[col] = df[col].apply(clean_price)
        
        # Clean volume column
        df['Volume'] = df['Volume'].apply(lambda x: int(str(x).replace(',', '')))
        
        # Sort by date in ascending order
        df = df.sort_values('Date')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to output file
        df.to_csv(output_file, index=False)
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert NASDAQ format to TSLA format')
    parser.add_argument('input_file', help='Path to input NASDAQ format CSV file')
    parser.add_argument('output_file', help='Path to output TSLA format CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the file
    format_nasdaq_data(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
