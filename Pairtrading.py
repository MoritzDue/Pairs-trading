import yfinance as yf
import pandas as pd


# DAX 40 tickers on Yahoo Finance
dax_tickers = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "1COV.DE", "DHER.DE", "DBK.DE", "DB1.DE", "DTE.DE", "DTG.DE", "EOAN.DE", "FME.DE", "FRE.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "P911.DE", "QIA.DE", "RWE.DE", "SAP.DE",
    "SIE.DE", "SHL.DE", "SY1.DE", "SRT3.DE", "VOW3.DE", "VNA.DE", "ZAL.DE", "RHM.DE", "MUV2.DE", "ENR.DE"
]

# Download data
data = yf.download(tickers, start='2020-07-31', end='2025-07-31', interval='1d', group_by='ticker', auto_adjust=True)

# Example: Access BMW.DE's data
bmw_data = data['BMW.DE']
print(bmw_data.head())
