import requests
import pandas as pd
import time
import logging
import zipfile
import io
from pathlib import Path
import argparse
from typing import Optional, List
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data/spot/daily/klines/ETHUSDC/1s/"

# Based on: https://github.com/binance/binance-public-data/#klines
COLUMNS: List[str] = [
                    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close time', 'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
                ]

def setup_logging():
    """Configure logging to both file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('binance_processor.log'),
            logging.StreamHandler()
        ]
    )


def download_and_process_file(date: str) -> Optional[pd.DataFrame]:
    """
    Download and process a single date's data file
    """
    url = f"{BASE_URL}ETHUSDC-1s-{date}.zip"

    try:
        # Download zip file
        response = requests.get(url)
        response.raise_for_status()

        # Extract CSV from zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            csv_filename = zip_file.namelist()[0]  # Should only be one file
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file, header=None, names=COLUMNS)

        # Process timestamps
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'Quote asset volume', 'Number of trades',
                        'Taker buy base asset volume', 'Taker buy quote asset volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

        return df

    except Exception as e:
        logging.error(f"Error processing file for {date}: {e}")
        return None


def save_to_parquet(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to parquet format with compression
    """
    try:

        df.to_parquet(
            output_path,
            engine='fastparquet',
            compression='snappy',
            index=False
        )
        logging.info(f"Saved {output_path}")

    except Exception as e:
        logging.error(f"Error saving parquet file on {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default="2024-01-01")
    parser.add_argument('--end_date', type=str, default="2024-01-31")
    parser.add_argument('--output_dir', type=str, default="./../simulation_results/test2/binance")

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    date_range = pd.date_range(start=args.start_date, end=args.end_date)

    for date_val in tqdm(date_range, total=len(date_range)):

        logging.info(f"Processing date: {date_val}")

        # Create output directory if it doesn't exist
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Save to parquet
        output_path = Path(args.output_dir) / f"ETHUSDC-1s-{date_val.strftime('%Y-%m-%d')}.parquet"

        if output_path.exists():
            logging.warning(f"File already exists, skipping: {output_path}")
            continue

        df = download_and_process_file(date_val.strftime('%Y-%m-%d'))

        # Save to parquet
        save_to_parquet(df, output_path)

        # Rate limiting
        time.sleep(1)


if __name__ == "__main__":
    main()