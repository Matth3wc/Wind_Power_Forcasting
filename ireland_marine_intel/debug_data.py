#!/usr/bin/env python3
"""Debug script to test data fetchers and identify issues."""
import asyncio
from ingestion.lighthouse_fetcher import LighthouseFetcher

async def test():
    print("=== Testing Lighthouse Data After Fix ===")
    async with LighthouseFetcher() as lf:
        df = await lf.fetch_met_eireann_data("MalinHead", days_back=1)
    print(f"MalinHead rows: {len(df)}")
    if len(df) > 0:
        print(f"columns: {list(df.columns)}")
        latest = df.iloc[-1].to_dict()
        print(f"Latest data: {latest}")

if __name__ == "__main__":
    asyncio.run(test())
