import numpy as np

# TODO: update i.e. LPs with Users object or keep as user_name str?


# AMM instance
class AMMi():
    # W = weight of the deposit asset in the pool (W_A = W_B)
    W = 0.5

    def __init__(self, ammi):
        self.ammi = ammi        # AMM instance (id)
        self.assets = {}
        self.VoteSlots = []
        self.voters = []
        self.TFee = 0.005       # TFee = trading fee
        self.curr_codes = []    # currency codes
        self.LPs = {}           # liquidity providers (users)
        self.B = 0              # price at which the last slot was bought
        self.AuctionSlot = {'user': '', 't': 0,
                            'discountedFee': 0, 'price': 0}
        self.MinSlotPrice = 2

    def add_asset(self, asset: str, amount: float):
        if asset in self.assets:
            self.assets[asset] += amount
        else:
            self.assets[asset] = amount

    def remove_asset(self, asset: str, amount: float):
        self.assets[asset] -= amount

    def add_LP(self, user: str, amount: float):
        if user in list(self.LPs):
            self.LPs[user] += amount
        else:
            self.LPs[user] = amount

    def remove_LP(self, user: str, amount: float):
        self.LPs[user] -= amount
        if self.LPs[user] == 0:
            del self.LPs[user]

# --------------------------- DOUBLE/ALL ASSET DEPOSIT ---------------------------

    def delta_token_Double(self, LPTokens_In_or_Out: float, bal_token: float) -> float:
        # bal_token = balance of asset in pool (bal_A would be balance of asset A)
        delta_token = (LPTokens_In_or_Out /
                       self.ammi.assets['LPTokens']) * bal_token
        # delta_token = (LPTokensOut / b) * bal_token
        return delta_token

# --------------------------- SINGLE ASSET DEPOSIT ---------------------------

    # how many LP Tokens you receive for a single-asset deposit
    def delta_LPTokens_Single(self, amount: float, bal_token: float, TFee: float) -> float:
        # bal_token = balance of asset in pool (bal_A would be balance of asset A)
        # L = amount of LP Tokens returned
        L = self.ammi.assets['LPTokens'] * \
            ((1 + (amount-TFee*(1-self.ammi.W)*amount)/bal_token)**self.ammi.W - 1)
        return L

    def delta_token_Single(self, bal_token: float, LPTokenOut: float, TFee: float) -> float:
        return (((LPTokenOut/self.ammi.assets['LPTokens'] + 1)**(1/self.W) - 1)/(1-TFee*(1-self.ammi.W))) * bal_token

    # ratio of the tokens the trader sold or swapped in (Token B) and the token they got in return or swapped out (Token A)
    def effectivePrice(self, delta_A: float, delta_B: float) -> float:
        ep = delta_B/delta_A
        return ep

    # --------------------------- SINGLE ASSET WITHDRAWAL ---------------------------
   # WS = withdraw single
    def delta_LPTokens_WS(self, amount: float, bal_token: float, TFee: float) -> float:
        # bal_token = balance of asset in pool (bal_A would be balance of asset A)
        # L = amount of LPTokensIn
        L = self.ammi.assets['LPTokens'] * \
            (1 - (1 - amount / (bal_token * (1-(1-self.W)*TFee)))**self.W)
        return np.real(L)

    def delta_token_WS(self, bal_token: float, LPTokenIn: float, TFee: float) -> float:
        return bal_token * (1 - (1 - LPTokenIn/self.ammi.assets['LPTokens'])**(1/self.W) * (1 - (1-self.W) * TFee))

    # --------------------------- SWAPS ---------------------------

    def delta_tokenOut_Swap(self, bal_tokenIn: float, bal_tokenOut: float, delta_tokenIn: float, TFee: float) -> float:
        # delta_tokenIn = amount of asset to swap in
        return bal_tokenOut * (1 - (bal_tokenIn/(bal_tokenIn + delta_tokenIn*(1-TFee)))**(self.W/self.W))

    def delta_tokenIn_Swap(self, bal_tokenIn: float, bal_tokenOut: float, delta_tokenOut: float, TFee: float) -> float:
        # delta_tokenOut = amount of asset to swap out
        return bal_tokenIn * ((bal_tokenOut/(bal_tokenOut-delta_tokenOut))**(self.W/self.W) - 1) * 1/(1-TFee)

    def delta_tokenIn_given_spotprices(self, assetIn: str, assetOut: str, post_sp: float) -> float:
        # pre_sp = spot price before trade
        # post_sp = spot price after trade,  to be provided by user
        pre_sp = self.spot_price(assetIn, assetOut)
        delta_tokenIn = self.ammi.assets[assetIn] * \
            ((post_sp/pre_sp)**(self.W/(self.W+self.W)) - 1)
        return delta_tokenIn

    # the slippage slope is the derivative of the slippage when the traded amount tends to zero
    def slippage_slope_tokenIn(self, bal_tokenIn: float, TFee: float) -> float:
        # bal_tokenIn = balance of token in the pool to trade in
        SS_In = (1-TFee) * (self.W + self.W) / (2 * bal_tokenIn * self.W)
        return SS_In

    def avg_slippage_tokenIn(self, bal_tokenIn: float, bal_tokenOut: float) -> float:
        S_delta_tokenIn = self.slippage_slope_tokenIn(
            self, bal_tokenIn) * self.delta_tokenIn_Swap(bal_tokenIn, bal_tokenOut)
        return S_delta_tokenIn

    def slippage_slope_tokenOut(self, bal_tokenIn: float) -> float:
        # bal_tokenIn = balance of token in the pool to trade in
        SS_Out = (self.W + self.W) / (2 * bal_tokenIn * self.W)
        return SS_Out

    def avg_slippage_tokenOut(self, bal_tokenIn: float, bal_tokenOut: float) -> float:
        S_delta_tokenOut = self.slippage_slope_tokenOut(
            self, bal_tokenIn) * self.delta_tokenOut_Swap(bal_tokenIn, bal_tokenOut)
        return S_delta_tokenOut

    # spot-price of asset/token Out relative to asset/token In
    def spot_price(self, assetIn: str, assetOut: str) -> float:
        sp = (self.ammi.assets[assetIn]/self.ammi.W) / \
            (self.ammi.assets[assetOut]/self.ammi.W) * 1/(1 - self.ammi.TFee)
        return sp

    # use this to get spot price through an AMM object (i.e. amm.spot_price1(...))
    def spot_price1(self, assetIn: str, assetOut: str) -> float:
        sp = (self.assets[assetIn]/self.W) / \
            (self.assets[assetOut]/self.W) * 1/(1 - self.TFee)
        return sp

    # --------------------------- TRADING FEE ---------------------------

    def trading_fee(self):
        if len(self.ammi.VoteSlots) > 0:
            numerator = 0
            denominator = 0
            for i in range(len(self.ammi.VoteSlots)):
                vote_weight = self.ammi.VoteSlots[i]['vote_weight']
                fee_val = self.ammi.VoteSlots[i]['tfee']
                numerator += (vote_weight * fee_val)
                denominator += vote_weight
            self.ammi.TFee = numerator/denominator


# ---------------------------     AMMVote CLASS     ---------------------------


class AMMVote(AMMi):
    def __init__(self, user, ammi):
        super().__init__(ammi)
        self.user = user

    def vote_entry(self, fee_val: float):
        # fee_val = the trading fee the user (liquidity provider) wants to vote for (between 0% and 1%)
        assert 0 <= fee_val <= 0.01
        assert self.user.assets['LPTokens'] > 0

        vote_weight = self.user.assets['LPTokens'] / \
            self.ammi.assets['LPTokens']
        vote_slot = {'user': self.user.user_name,
                     'tfee': fee_val, 'vote_weight': vote_weight}

        if len(self.ammi.VoteSlots) == 0:
            self.ammi.VoteSlots += [vote_slot]
            self.ammi.voters += [self.user.user_name]
            self.trading_fee()
        else:
            for i in range(len(self.ammi.VoteSlots)):
                # if user already has a vote slot, update vote slot
                if self.user.user_name == self.ammi.VoteSlots[i]['user']:
                    updated_vote_slot = vote_slot
                    self.ammi.VoteSlots[i] = updated_vote_slot
                    self.trading_fee()
                    break
            else:
                if len(self.ammi.VoteSlots) < 8:
                    self.ammi.VoteSlots += [vote_slot]
                    self.ammi.voters += [self.user.user_name]
                    self.trading_fee()
                else:
                    vweights = []
                    for voteEntry in self.ammi.VoteSlots:
                        vweights += [voteEntry['vote_weight']]
                    min_vweight_index = vweights.index(min(vweights))
                    if self.ammi.VoteSlots[min_vweight_index]['vote_weight'] < vote_weight:
                        self.ammi.voters.remove(
                            self.ammi.VoteSlots[min_vweight_index]['user'])
                        self.ammi.VoteSlots[min_vweight_index] = vote_slot
                        self.ammi.voters += [self.user.user_name]
                        self.trading_fee()
                    else:
                        # FAIL TX
                        pass
                        # print('Your vote weight is not enough to vote')

    def monitor_VoteSlots(self):
        for i in range(len(self.ammi.VoteSlots)):
            if self.user.user_name == self.ammi.VoteSlots[i]['user']:
                if self.user.assets['LPTokens'] == 0:
                    del self.ammi.VoteSlots[i]
                # update vote entries
                for i in range(len(self.ammi.VoteSlots)):
                    self.ammi.VoteSlots[i]['vote_weight'] = self.ammi.LPs[self.ammi.VoteSlots[i]
                                                                          ['user']] / self.ammi.assets['LPTokens']
                self.trading_fee()
        else:
            # update vote entries
            for i in range(len(self.ammi.VoteSlots)):
                self.ammi.VoteSlots[i]['vote_weight'] = self.ammi.LPs[self.ammi.VoteSlots[i]
                                                                      ['user']] / self.ammi.assets['LPTokens']
            self.trading_fee()


# ---------------------------     USER CLASS     ---------------------------

class User():
    def __init__(self, user_name: str, assets={}):
        self.user_name = user_name
        self.assets = assets

    def add_asset(self, asset: str, amount: float):
        if asset in self.assets:
            self.assets[asset] += amount
        else:
            self.assets[asset] = amount

    def remove_asset(self, asset: str, amount: float):
        self.assets[asset] -= amount

    def createAMM(self, ammID: int, asset1: str, asset2: str, amount1: float, amount2: float) -> AMMi:
        if (self.assets[asset1] >= amount1) and (self.assets[asset2] >= amount2):
            ammi = AMMi(ammID)
            ammi.add_asset(asset1, amount1)
            ammi.add_asset(asset2, amount2)
            ammi.curr_codes += [asset1, asset2]
            self.remove_asset(asset1, amount1)
            self.remove_asset(asset2, amount2)
            LPTokens = (amount1**ammi.W) * (amount2**ammi.W)
            self.add_asset('LPTokens', LPTokens)
            ammi.add_asset('LPTokens', LPTokens)
            ammi.LPs[self.user_name] = LPTokens
            return ammi
        else:
            # TX FAIL
            raise Exception('Not enough tokens')
