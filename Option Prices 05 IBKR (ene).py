import numpy as np
import matplotlib.pyplot as plt


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C

def black_scholes_put(S, K, T, r, sigma):
    
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return P

from scipy.stats import norm

S = 100
K = 110
T = 1.5
r = 0.03
sigma = 0.2

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

print("Call Price: ", call_price)
print("Put Price: ", put_price)


import yfinance as yf
import arch

company_stock = yf.download('AAPL',start='2023-05-01',end='2023-06-01')

daily_returns = company_stock['Close'].pct_change().dropna()

garch_model = arch.arch_model(daily_returns, vol='GARCH', p=1, o=0, q=1)
res = garch_model.fit(update_freq=5)
print(res.summary())

fig = res.plot()
plt.show()

import pandas as pd
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from datetime import datetime, timedelta
import random
import threading
import time

# class IBapi(EWrapper, EClient):
#     def __init__(self):
#         EClient.__init__(self, self)

#     def stop(self):
#         # Disconnecting the client
#         self.disconnect()

# def run_loop():
#     app.run()

# app = IBapi()
# app.connect('127.0.0.1', 4002, clientId=client_id)

# # Start the loop in a new thread
# api_thread = threading.Thread(target=run_loop, daemon=True)
# api_thread.start()

# print("The main thread remains interactive.")

# Example: Disconnecting after some time
# import time
# time.sleep(10)  # or however long you want the event loop to run
# app.stop()  # This will stop the event loop

'''
#Uncomment this section if unable to connect
#and to prevent errors on a reconnect
import time
time.sleep(2)
app.disconnect()
'''

def generate_client_id():
    return random.randint(10000000, 99999999)

client_id = generate_client_id()

class OptionDataWrapper(EWrapper, EClient):

    def connectionClosed(self):
        print("Connection closed.")

    def __init__(self):
        EClient.__init__(self, self)  # Initialize the EClient part
        self.df = pd.DataFrame(columns=["Expiry", "Strike"])
        self.received_contract_details = None
        self.option_data_processed = False
        self.lastPrice = None

    def contractDetails(self, reqId, contractDetails):
        print(f"Received contract details for ReqID: {reqId}")
        print(contractDetails.contract.symbol, contractDetails.contract.secType, contractDetails.contract.currency)
        print(contractDetails.longName)
        print(contractDetails.underConId)
        # Save received contract details
        self.contract_details = contractDetails.contract
        # Make subsequent request using the saved details
        self.reqSecDefOptParams(2, underlyingSymbol=self.contract_details.symbol, 
                                futFopExchange="", 
                                underlyingSecType=self.contract_details.secType, 
                                underlyingConId=self.contract_details.conId)

    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass, multiplier,
                                          expirations, strikes):
        
        if self.option_data_processed:
            # If data has already been processed before, just return
            return
        
        print("Received Option Data:")

        # Current date
        current_date = datetime.now()

        # Get the date six months from now
        six_months_later = current_date + timedelta(days=180)

        data = []
        for expiry in expirations:
            expiry_date = datetime.strptime(expiry, "%Y%m%d")

            # Only consider expiries less than six months away
            if expiry_date <= six_months_later:
                for strike in strikes:
                   data.append({"Expiry": expiry, "Strike": strike})

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)

        # Set the flag to True after processing
        self.option_data_processed = True

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 4:  # Last trade price
            self.lastPrice = price
            print(f"Last trade price: {price}")

    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId}, Code: {errorCode}, Msg: {errorString}")

    def stop_app(app):
        """Stops the IB event loop"""
        app.done = True
        app.disconnect()

def main():
    global app
    global global_df  # Declare the global variable to update it from within the thread
    global global_last_price

    # Setup the connection
    app = OptionDataWrapper()
    app.connect("127.0.0.1", 4002, clientId=client_id)

    # Sleep for 10 seconds after making the connection
    time.sleep(10)

    # Create a contract object for the underlying you're interested in
    contract = Contract()
    contract.symbol = "GLTO"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    # Check that contracts exist
    app.reqContractDetails(1, contract)

    #Get price quote of underlying
    app.reqMktData(3, contract, "", False, False, [])

    # Run the client event loop
    app.run()   

    # Update the global variable with the populated dataframe
    global_df = app.df
    global_last_price = app.lastPrice
    
# Wrap the main function in a thread
t = threading.Thread(target=main)

if __name__ == "__main__":
    t.start()

    # Keep the script interactive
    while True:
        cmd = input("Enter 'stop' to stop the connection or 'exit' to quit: ").strip().lower()
        
        if cmd == 'stop':
            app.disconnect()  # Directly calling disconnect on the app object
            t.join()  # Wait for the thread to finish
            break

        elif cmd == 'exit':
            app.disconnect()  # Ensure the connection is stopped before exiting
            break


# Now, global_df holds the populated dataframe and can be used in the rest of your script
print(global_df)
print(global_last_price)
global_df.to_csv('dataframe.csv', index=False)
global_df = pd.read_csv('dataframe.csv')


### Section for visualizing the data
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Assuming global_df is your dataframe
# df = global_df.copy()

# # Convert 'Expiry' column to datetime for sorting and better plotting
# df['Expiry'] = pd.to_datetime(df['Expiry'], format='%Y%m%d')

# # Order the expiry dates and count unique strike prices for each expiry
# ordered_dates = df['Expiry'].sort_values().unique()
# strike_counts = df.groupby('Expiry')['Strike'].nunique()

# plt.figure(figsize=(12,6))
# sns.boxplot(data=df, x='Expiry', y='Strike', order=ordered_dates)

# # Annotate with number of unique strike prices
# for i, date in enumerate(ordered_dates):
#     y_position = df[df['Expiry'] == date]['Strike'].min()  # you can adjust this for better position
#     plt.text(i, y_position, f'n={strike_counts[date]}', ha='center', va='bottom', fontweight='bold', color='red')

# plt.xticks(rotation=45)
# plt.title('Option Chain: Distribution of Strike Prices by Expiry Date')
# plt.tight_layout()  # Ensures all labels fit in the figure
# plt.show()
