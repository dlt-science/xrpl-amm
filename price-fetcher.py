import requests

'''
Get the aset1/asset2 pair minute price over a day (1440 data points).

The initial timestamp 1677888000 represents the 4th of March 2023 00:00:00 GMT,
so that the first set of price data will be that of the 3rd of March 2023 since 
CryptoCompare returns the historical data before that timestamp.
We then compare the price change per day starting March 3rd until March 8th since
CryptoCompare only provides the past 7 days of minute data.
We then store the price data of the day with the highest price change.
'''

asset1 = 'ETH'
asset2 = 'USDC'

initial_timestamp = 1677888000
timestamps = [initial_timestamp + i*86400 for i in range(6)]

pcs = []
for timestamp in timestamps:
    response = requests.get(
        f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={asset1}&tsym={asset2}&limit=1&toTs={timestamp}')
    prices = response.json()
    # pc = price change over the day
    pc = ((prices['Data']['Data'][0]['close'] - prices['Data']
          ['Data'][0]['open'])/prices['Data']['Data'][0]['close'])
    pcs += [abs(pc)]

# get index of the timestamp when the price change was the highest
highest_pc_index = pcs.index(max(pcs))

# get the price data for the day with the highest price change
response = requests.get(
    f'https://min-api.cryptocompare.com/data/v2/histominute?fsym={asset1}&tsym={asset2}&limit=1440&toTs={timestamps[highest_pc_index]}')
prices = response.json()

# add the price data to a file
with open('data/prices_cryptocompare.txt', 'w') as file:
    for price in prices['Data']['Data']:
        file.write(str(price['open'])+'\n')
