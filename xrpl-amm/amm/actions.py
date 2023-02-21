from amm.env import AMMi, AMMVote, User
from sympy import Symbol
from sympy.solvers import solve


class Deposit(AMMVote):
    def __init__(self, user: User, ammi: AMMi):
        super().__init__(self, ammi)
        self.user = user
        if self.user == self.ammi.AuctionSlot['user']:
            self.TFee = self.ammi.AuctionSlot['discountedFee']
        else:
            self.TFee = self.ammi.TFee

# ---------------------------                           ---------------------------
# --------------------------- DOUBLE/ALL ASSET DEPOSITS ---------------------------

    # if "LPTokenOut" is specified
    def deposit_LPTokenOut(self, LPTokenOut: float):
        delta_A = self.delta_token_Double(
            LPTokenOut, self.ammi.assets[self.ammi.curr_codes[0]])
        delta_B = self.delta_token_Double(
            LPTokenOut, self.ammi.assets[self.ammi.curr_codes[1]])

        if self.user.assets[self.ammi.curr_codes[0]] >= delta_A and self.user.assets[self.ammi.curr_codes[1]] >= delta_B:
            self.user.add_asset('LPTokens', LPTokenOut)
            self.user.remove_asset(self.ammi.curr_codes[0], delta_A)
            self.user.remove_asset(self.ammi.curr_codes[1], delta_B)
            self.ammi.add_asset('LPTokens', LPTokenOut)
            self.ammi.add_asset(self.ammi.curr_codes[0], delta_A)
            self.ammi.add_asset(self.ammi.curr_codes[1], delta_B)
            self.ammi.add_LP(self.user.user_name, LPTokenOut)
            self.monitor_VoteSlots()
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # Amount1(2) = amount_A(B)
    # if "Amount1" and "Amount2" are specified
    def deposit_Amount1_Amount2(self, asset_A: str, asset_B: str, amount_A: float, amount_B: float):
        if self.user.assets[asset_A] < amount_A or self.user.assets[asset_B] < amount_B:
            raise Exception('Not enough tokens!')
        # Z = LPTokens to be returned/issued
        Z = amount_A * self.ammi.assets['LPTokens'] / self.ammi.assets[asset_A]
        # X = amount of asset B (delta_B)
        X = Z / self.ammi.assets['LPTokens'] * self.ammi.assets[asset_B]
        if X <= amount_B:
            self.ammi.add_asset(asset_A, amount_A)
            self.ammi.add_asset(asset_B, X)
            self.ammi.add_asset('LPTokens', Z)
            self.user.add_asset('LPTokens', Z)
            self.user.remove_asset(asset_A, amount_A)
            self.user.remove_asset(asset_B, X)
            self.ammi.add_LP(self.user.user_name, Z)
            self.monitor_VoteSlots()
        elif X > amount_B:
            # W = LPTokens to be returned/issued
            W = amount_B * \
                self.ammi.assets['LPTokens'] / self.ammi.assets[asset_B]
            # Y = amount of asset A (delta_A)
            Y = W / self.ammi.assets['LPTokens'] * self.ammi.assets[asset_A]
            if Y <= amount_A:
                self.ammi.add_asset(asset_A, Y)
                self.ammi.add_asset(asset_B, amount_B)
                self.ammi.add_asset('LPTokens', W)
                self.user.add_asset('LPTokens', W)
                self.user.remove_asset(asset_A, Y)
                self.user.remove_asset(asset_B, amount_B)
                self.ammi.add_LP(self.user.user_name, W)
                self.monitor_VoteSlots()
            else:
                # FAIL TX
                raise Exception("Not enough tokens")

    # --------------------------- SINGLE ASSET DEPOSITS ---------------------------

    # if "Amount" is specified
    def deposit_Amount(self, asset: str, amount: float):
        if self.user.assets[asset] > amount:
            L = self.delta_LPTokens_Single(
                amount, self.ammi.assets[asset], self.TFee)
            self.ammi.add_asset(asset, amount)
            self.ammi.add_asset('LPTokens', L)
            self.user.add_asset('LPTokens', L)
            self.user.remove_asset(asset, amount)
            self.ammi.add_LP(self.user.user_name, L)
            self.monitor_VoteSlots()
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # if "Amount" and "LPTokenOut" are specified
    def deposit_Amount_LPTokenOut(self, asset: str, amount: float, LPTokenOut: float):
        if self.user.assets[asset] > amount:
            delta_token = self.delta_token_Single(
                self.ammi.assets[asset], LPTokenOut, self.TFee)
            self.ammi.add_asset(asset, delta_token)
            self.ammi.add_asset('LPTokens', LPTokenOut)
            self.user.add_asset('LPTokens', LPTokenOut)
            self.user.remove_asset(asset, delta_token)
            self.ammi.add_LP(self.user.user_name, LPTokenOut)
            self.monitor_VoteSlots()
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # if "Amount" and "EPrice" are specified
    # ep = Effective Price
    # TODO: simplify the function
    def deposit_Amount_EPrice(self, asset: str, ep: float, amount=''):
        if amount:
            # X = amount of LPTokenOut to be issued
            X = self.delta_LPTokens_Single(
                float(amount), self.ammi.assets[asset], self.TFee)
            # Y = effective-price of the trade
            Y = self.effectivePrice(X, float(amount))
            if Y <= ep:
                self.ammi.add_asset(asset, float(amount))
                self.ammi.add_asset('LPTokens', X)
                self.user.add_asset('LPTokens', X)
                self.user.remove_asset(asset, float(amount))
                self.ammi.add_LP(self.user.user_name, X)
            else:
                amount = Symbol('x')
                L = self.ammi.assets['LPTokens'] * \
                    ((1 + (amount-self.TFee*(1-self.ammi.W)*amount) /
                     self.ammi.assets[asset])**1-self.ammi.W - 1)
                # Q = amount of asset in
                Q = solve(amount / L - ep, amount)[0]
                # W = amount of LPToken out
                W = self.ammi.assets['LPTokens'] * \
                    ((1 + (Q-self.TFee*(1-self.ammi.W)*Q) /
                     self.ammi.assets[asset])**self.ammi.W - 1)
                self.ammi.add_asset(asset, Q)
                self.ammi.add_asset('LPTokens', W)
                self.user.add_asset('LPTokens', W)
                self.user.remove_asset(asset, Q)
                self.ammi.add_LP(self.user.user_name, W)
        else:
            amount = Symbol('x')
            L = self.ammi.assets['LPTokens'] * \
                ((1 + (amount-self.TFee*(0.5)*amount) /
                 self.ammi.assets[asset])**0.5 - 1)
            # Q = amount of asset in
            Q = solve(amount / L - ep, amount)[0]
            # W = amount of LPToken out
            W = self.ammi.assets['LPTokens'] * \
                ((1 + (Q-self.TFee*(1-self.ammi.W)*Q) /
                 self.ammi.assets[asset])**self.ammi.W - 1)
            self.ammi.add_asset(asset, Q)
            self.ammi.add_asset('LPTokens', W)
            self.user.add_asset('LPTokens', W)
            self.user.remove_asset(asset, Q)
            self.ammi.add_LP(self.user.user_name, W)
        self.monitor_VoteSlots()


# --------------------------- WITHDRAW CLASS ---------------------------

class Withdraw(AMMVote):
    def __init__(self, user, ammi):
        super().__init__(self, ammi)
        self.user = user
        if self.user == self.ammi.AuctionSlot['user']:
            self.TFee = self.ammi.AuctionSlot['discountedFee']
        else:
            self.TFee = self.ammi.TFee

    # --------------------------- DOUBLE/ALL ASSET WITHDRAWALS ---------------------------

    # if "LPTokenIn" is specified
    def withdraw_LPTokenIn(self, LPTokenIn: float):
        if self.user.assets['LPTokens'] >= LPTokenIn:
            delta_A = self.delta_token_Double(
                LPTokenIn, self.ammi.assets[self.ammi.curr_codes[0]])
            delta_B = self.delta_token_Double(
                LPTokenIn, self.ammi.assets[self.ammi.curr_codes[1]])
            if self.ammi.assets[self.ammi.curr_codes[0]] > delta_A and self.ammi.assets[self.ammi.curr_codes[1]] > delta_B:
                self.user.remove_asset('LPTokens', LPTokenIn)
                self.ammi.remove_asset('LPTokens', LPTokenIn)
                self.ammi.remove_asset(
                    self.ammi.curr_codes[0], delta_A)
                self.ammi.remove_asset(
                    self.ammi.curr_codes[1], delta_B)
                self.user.add_asset(self.ammi.curr_codes[0], delta_A)
                self.user.add_asset(self.ammi.curr_codes[1], delta_B)
                self.ammi.remove_LP(self.user.user_name, LPTokenIn)
                self.monitor_VoteSlots()
            else:
                # FAIL TX
                raise Exception("Not enough tokens")
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # if "Amount1" and "Amount2" are specified
    def withdraw_Amount1_Amount2(self, asset_A: str, asset_B: str, amount_A: float, amount_B: float):
        # Z = LPTokensIn
        Z = amount_A * self.ammi.assets['LPTokens'] / self.ammi.assets[asset_A]
        # X = amount of asset B (delta_B)
        X = Z / self.ammi.assets['LPTokens'] * self.ammi.assets[asset_B]
        if X <= amount_B and self.user.assets['LPTokens'] >= Z:
            self.ammi.remove_asset(asset_A, amount_A)
            self.ammi.remove_asset(asset_B, X)
            self.ammi.remove_asset('LPTokens', Z)
            self.user.remove_asset('LPTokens', Z)
            self.user.add_asset(asset_A, amount_A)
            self.user.add_asset(asset_B, X)
            self.ammi.remove_LP(self.user.user_name, Z)
            self.monitor_VoteSlots()
        elif X > amount_B and self.user.assets['LPTokens'] >= Z:
            # Q = LPTokensIn
            Q = amount_B * \
                self.ammi.assets['LPTokens'] / self.ammi.assets[asset_B]
            # W = amount of asset A (delta_A)
            W = Q / self.ammi.assets['LPTokens'] * self.ammi.assets[asset_A]
            self.ammi.remove_asset(asset_A, W)
            self.ammi.remove_asset(asset_B, amount_B)
            self.ammi.remove_asset('LPTokens', Q)
            self.user.remove_asset('LPTokens', Q)
            self.user.add_asset(asset_A, W)
            self.user.add_asset(asset_B, amount_B)
            self.ammi.remove_LP(self.user.user_name, Q)
            self.monitor_VoteSlots()
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # --------------------------- SINGLE ASSET WITHDRAWALS ---------------------------

    # if "Amount" is specified
    def withdraw_Amount(self, asset: str, amount: float):
        L = self.delta_LPTokens_WS(amount, self.ammi.assets[asset], self.TFee)
        if self.user.assets['LPTokens'] >= L and self.ammi.assets[asset] > amount:
            self.ammi.remove_asset(asset, amount)
            self.ammi.remove_asset('LPTokens', L)
            self.user.remove_asset('LPTokens', L)
            self.user.add_asset(asset, amount)
            self.ammi.remove_LP(self.user.user_name, L)
            self.monitor_VoteSlots()
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # if "Amount" and "LPTokenIn" are specified
    def withdraw_Amount_LPTokenIn(self, asset: str, LPTokenIn: float, amount=''):
        if self.user.assets['LPTokens'] >= LPTokenIn:
            # Y = amount of asset A
            Y = self.delta_token_WS(
                self.ammi.assets[asset], LPTokenIn, self.TFee)
            if (amount and Y >= float(amount)) or (not amount):
                self.ammi.remove_asset(asset, Y)
                self.ammi.remove_asset('LPTokens', LPTokenIn)
                self.user.remove_asset('LPTokens', LPTokenIn)
                self.user.add_asset(asset, Y)
                self.ammi.remove_LP(self.user.user_name, LPTokenIn)
                self.monitor_VoteSlots()
            else:
                # FAIL TX
                raise Exception("Not enough tokens")

    # if "Amount" and "EPrice" are specified
    # ep = Effective Price
    def withdraw_Amount_EPrice(self, asset: str, ep: float, amount=''):
        # computed amount
        comp_amount = Symbol('x')
        # L = amount of LPTokensIn
        L = self.ammi.assets['LPTokens'] * \
            (1 - (1 - comp_amount /
             (self.ammi.assets[asset] * (1-(1-self.ammi.W)*self.TFee)))**self.ammi.W)
        # Y = amount of asset/token out
        Y = solve(ep * comp_amount - L, comp_amount)[0]
        # X = asset in as LPTokenIn
        X = self.delta_LPTokens_WS(Y, self.ammi.assets[asset], self.TFee)

        if (amount and Y >= float(amount)) or (not amount):
            self.ammi.remove_asset(asset, Y)
            self.ammi.remove_asset('LPTokens', X)
            self.user.remove_asset('LPTokens', X)
            self.user.add_asset(asset, Y)
            self.ammi.remove_LP(self.user.user_name, X)
            self.monitor_VoteSlots()
        else:
            pass  # FAIL TX
            # print("Not enough tokens")


# ---------------------------     SWAP CLASS     ---------------------------

class Swap(AMMi):
    def __init__(self, user, ammi):
        super().__init__(ammi)
        self.user = user
        if self.user == self.ammi.AuctionSlot['user']:
            self.TFee = self.ammi.AuctionSlot['discountedFee']
        else:
            self.TFee = self.ammi.TFee

    # given amount to swap in
    def swap_given_amount_In(self, assetIn: str, assetOut: str, amountIn: float):
        # delta_tokenOut = amount of asset to swap out, given amount of the other asset to swap in
        delta_tokenOut = self.delta_tokenOut_Swap(
            self.ammi.assets[assetIn], self.ammi.assets[assetOut], amountIn, self.TFee)

        if self.user.assets[assetIn] > amountIn and self.ammi.assets[assetOut] > delta_tokenOut:
            self.ammi.remove_asset(assetOut, delta_tokenOut)
            self.ammi.add_asset(assetIn, amountIn)
            self.user.remove_asset(assetIn, amountIn)
            self.user.add_asset(assetOut, delta_tokenOut)
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # given amount to swap out
    def swap_given_amount_Out(self, assetIn: str, assetOut: str, amountOut: float):
        # delta_tokenIn = amount of asset to swap in, given amount of the other asset to swap out
        if self.ammi.assets[assetOut] > amountOut:
            delta_tokenIn = self.delta_tokenIn_Swap(
                self.ammi.assets[assetIn], self.ammi.assets[assetOut], amountOut, self.TFee)
            if self.user.assets[assetIn] > delta_tokenIn:
                self.ammi.remove_asset(assetOut, amountOut)
                self.ammi.add_asset(assetIn, delta_tokenIn)
                self.user.remove_asset(assetIn, delta_tokenIn)
                self.user.add_asset(assetOut, amountOut)
            else:
                # FAIL TX
                raise Exception("Not enough tokens")
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # swap given the desired spot price the user wants token Out to reach
    def swap_given_postSP(self, assetIn: str, assetOut: str, post_sp: float):
        # pre_sp = spot-price pre trade
        delta_tokenIn = self.delta_tokenIn_given_spotprices(
            assetIn, assetOut, post_sp)
        delta_tokenOut = self.delta_tokenOut_Swap(
            self.ammi.assets[assetIn], self.ammi.assets[assetOut], delta_tokenIn, self.TFee)
        if self.ammi.assets[assetOut] > delta_tokenOut and self.user.assets[assetIn] > delta_tokenIn:
            self.ammi.remove_asset(assetOut, delta_tokenOut)
            self.ammi.add_asset(assetIn, delta_tokenIn)
            self.user.remove_asset(assetIn, delta_tokenIn)
            self.user.add_asset(assetOut, delta_tokenOut)
        else:
            # FAIL TX
            raise Exception("Not enough tokens")


# ---------------------------   AMMBid Class       ---------------------------

class AMMBid(AMMVote):
    def __init__(self, user, ammi):
        super().__init__(self, ammi)
        self.user = user

    # possible values for t: 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, ..., 0.9, 0.95, 1
    def compute_price(self, t: float) -> float:
        minBidPrice = 0
        if (not self.ammi.AuctionSlot['user']) or (t == 1):
            minBidPrice = self.ammi.MinSlotPrice
        elif t == 0.05:
            minBidPrice = self.ammi.B * 1.05 + self.ammi.MinSlotPrice
        elif 0.1 <= t < 1:
            minBidPrice = self.ammi.B * 1.05 * \
                (1-t**60) + self.ammi.MinSlotPrice
        return minBidPrice

    def bid(self, t: float, min_price='', max_price=''):
        minBidPrice = self.compute_price(t)
        if minBidPrice > self.user.assets['LPTokens']:
            raise Exception('Not enough tokens')

        if min_price:
            if float(min_price) > self.user.assets['LPTokens']:
                raise Exception('Not enough tokens')
        if max_price:
            if float(max_price) > self.user.assets['LPTokens']:
                raise Exception('Not enough tokens')

        bidPrice = minBidPrice
        if min_price and max_price:
            bidPrice = float(max_price)
        elif min_price and not max_price:
            if min_price <= minBidPrice:
                bidPrice = minBidPrice
            else:
                bidPrice = float(min_price)
        elif max_price and not min_price:
            if max_price >= minBidPrice:
                bidPrice = minBidPrice
            else:
                bidPrice = float(max_price)

        if self.ammi.AuctionSlot['user'] and self.ammi.AuctionSlot['t'] < t:
            refund = (1-t) * self.ammi.B
            self.ammi.AuctionSlot['user'].add_asset('LPTokens', refund)
            self.ammi.add_LP(self.ammi.AuctionSlot['user'].user_name, refund)
            # burn the remaining LPTokens
            self.ammi.remove_asset('LPTokens', bidPrice-refund)
            self.user.remove_asset('LPTokens', bidPrice)
            self.ammi.B = bidPrice
            self.ammi.AuctionSlot = {'user': self.user,
                                     't': t, 'discountedFee': 0, 'price': bidPrice}
            self.ammi.B = bidPrice
        else:
            self.ammi.AuctionSlot = {'user': self.user,
                                     't': t, 'discountedFee': 0, 'price': bidPrice}
            self.ammi.B = bidPrice
            # burn the LPTokens paid for the slot
            self.ammi.remove_asset('LPTokens', bidPrice)
            self.ammi.remove_LP(self.user.user_name, bidPrice)
            self.user.remove_asset('LPTokens', bidPrice)
        self.monitor_VoteSlots()
