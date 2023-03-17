# computations for use in testing

import numpy as np

W = 0.5  # normalized weight of tokens in the AMM instance pool
TFee = 0.005  # trading fee


def C(bal_A, bal_B):  # also equal to LPTokens!
    # bal_A = Current balance of token A in the AMM instance pool
    # bal_B = Current balance of token B in the AMM instance pool
    W = 0.5
    C = bal_A**W * bal_B**W
    return C


def spotPrice(bal_A, bal_B, TFee):  # spot-price of asset A relative to asset B
    sp = (bal_B/W)/(bal_A/W) * 1/(1 - TFee)
    return sp


# ratio of the tokens the trader sold or swapped in (Token B) and the token they got in return or swapped out (Token A)
def effectivePrice(delta_A, delta_B):
    ep = delta_B/delta_A
    return ep


# --------------------------- DOUBLE ASSET DEPOSIT ---------------------------


def delta_token_Double(LPTokens_In_or_Out: float, bal_token: float, bal_LPTokens: float) -> float:
    # bal_token = balance of asset in pool (bal_A would be balance of asset A)
    delta_token = (LPTokens_In_or_Out / bal_LPTokens) * bal_token
    # delta_token = (LPTokensOut / b) * bal_token
    return delta_token


# --------------------------- SINGLE ASSET DEPOSIT ---------------------------


# how many LP Tokens you receive for a single-asset deposit
def delta_LPTokens_Single(LPTokens: float, bal_asset: float, amount: float) -> float:
    # LPTokens = total outstanding LP Tokens before the deposit
    # bal_asset = balance of asset in pool (bal_A would be balance of asset A)
    # L = amount of LP Tokens returned
    L = LPTokens * \
        ((1 + (amount-TFee*(1-W)*amount)/bal_asset)**W - 1)
    return L


def delta_token_Single(LPTokens: float, LPTokenOut: float, bal_asset: float) -> float:
    return (((LPTokenOut/LPTokens + 1)**(1/W) - 1)/(1-TFee*(1-W))) * bal_asset

# --------------------------- WITHDRAWAL ---------------------------


# WS = withdraw single


def delta_LPTokens_WS(LPTokens: float, amount: float, bal_asset: float) -> float:
    # bal_asset = balance of asset in pool (bal_A would be balance of asset A)
    # L = amount of LPTokensIn
    L = LPTokens * (1 - (1 - amount / (bal_asset * (1-(1-W)*TFee)))**W)
    return np.real(L)


def delta_token_WS(bal_asset: float, LPTokenIn: float, LPTokens: float) -> float:
    return bal_asset * (1 - (1 - LPTokenIn/LPTokens)**(1/W) * (1 - (1-W) * TFee))


# --------------------------- SWAP ---------------------------

def delta_tokenOut_Swap(bal_assetIn: float, bal_assetOut: float, delta_tokenIn: float) -> float:
    # delta_tokenIn = amount of asset to swap in
    return bal_assetOut * (1 - (bal_assetIn/(bal_assetIn + delta_tokenIn*(1-TFee)))**(W/W))


def delta_tokenIn_Swap(bal_tokenIn: float, bal_tokenOut: float, delta_tokenOut: float) -> float:
    # delta_tokenOut = self.delta_tokenOut_Swap(self, bal_tokenIn, bal_tokenOut)
    return bal_tokenIn * ((bal_tokenOut/(bal_tokenOut-delta_tokenOut))**(W/W) - 1) * 1/(1-TFee)


# spot-price of asset/token Out relative to asset/token In
def spot_price(bal_assetIn: float, bal_assetOut: float) -> float:
    sp = (bal_assetIn/W) / (bal_assetOut/W) * 1/(1 - TFee)
    return sp


def delta_tokenIn_given_spotprices(bal_assetIn: float, bal_assetOut: float, post_sp: float) -> float:
    # pre_sp = spot price before trade
    # post_sp = spot price after trade,  to be provided by user
    pre_sp = spot_price(bal_assetIn, bal_assetOut)
    delta_tokenIn = bal_assetIn * ((post_sp/pre_sp)**(W/(W+W)) - 1)
    return delta_tokenIn
