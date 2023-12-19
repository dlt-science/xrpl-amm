from amms.xrpl.amm.env import AMMi, AMMVote, User
from sympy import Symbol
from sympy.solvers import solve


class Deposit(AMMVote):
    def __init__(self, user: User, ammi: AMMi):
        super().__init__(self, ammi)
        self.user = user
        if self.user == self.ammi.AuctionSlot["slot_owner"]:
            self.TFee = self.ammi.AuctionSlot["discountedFee"]
        else:
            self.TFee = self.ammi.TFee

    # --------------------------- DOUBLE/ALL ASSET DEPOSITS ---------------------------

    # if "LPTokenOut" is specified
    def deposit_LPTokenOut(self, LPTokenOut: float):
        delta_A = self.delta_token_Double(
            LPTokenOut, self.ammi.assets[self.ammi.curr_codes[0]]
        )
        delta_B = self.delta_token_Double(
            LPTokenOut, self.ammi.assets[self.ammi.curr_codes[1]]
        )

        if (
            self.user.assets[self.ammi.curr_codes[0]] >= delta_A
            and self.user.assets[self.ammi.curr_codes[1]] >= delta_B
        ):
            self.user.add_asset("LPTokens", LPTokenOut)
            self.user.remove_asset(self.ammi.curr_codes[0], delta_A)
            self.user.remove_asset(self.ammi.curr_codes[1], delta_B)
            self.ammi.add_asset("LPTokens", LPTokenOut)
            self.ammi.add_asset(self.ammi.curr_codes[0], delta_A)
            self.ammi.add_asset(self.ammi.curr_codes[1], delta_B)
            self.ammi.add_LP(self.user.user_name, LPTokenOut)
            self.monitor_VoteSlots()
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # Amount1(2) = amount_A(B)
    # if "Amount1" and "Amount2" are specified
    def deposit_Amount1_Amount2(
        self, asset_A: str, asset_B: str, amount_A: float, amount_B: float
    ):
        if self.user.assets[asset_A] < amount_A or self.user.assets[asset_B] < amount_B:
            raise Exception("Not enough tokens!")
        # Z = LPTokens to be returned/issued
        Z = amount_A * self.ammi.assets["LPTokens"] / self.ammi.assets[asset_A]
        # X = amount of asset B (delta_B)
        X = Z / self.ammi.assets["LPTokens"] * self.ammi.assets[asset_B]
        if X <= amount_B:
            self.ammi.add_asset(asset_A, amount_A)
            self.ammi.add_asset(asset_B, X)
            self.ammi.add_asset("LPTokens", Z)
            self.user.add_asset("LPTokens", Z)
            self.user.remove_asset(asset_A, amount_A)
            self.user.remove_asset(asset_B, X)
            self.ammi.add_LP(self.user.user_name, Z)
            self.monitor_VoteSlots()
        elif X > amount_B:
            # W = LPTokens to be returned/issued
            W = amount_B * self.ammi.assets["LPTokens"] / self.ammi.assets[asset_B]
            # Y = amount of asset A (delta_A)
            Y = W / self.ammi.assets["LPTokens"] * self.ammi.assets[asset_A]
            if Y <= amount_A:
                self.ammi.add_asset(asset_A, Y)
                self.ammi.add_asset(asset_B, amount_B)
                self.ammi.add_asset("LPTokens", W)
                self.user.add_asset("LPTokens", W)
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
            L = self.delta_LPTokens_Single(amount, self.ammi.assets[asset], self.TFee)
            self.ammi.add_asset(asset, amount)
            self.ammi.add_asset("LPTokens", L)
            self.user.add_asset("LPTokens", L)
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
                self.ammi.assets[asset], LPTokenOut, self.TFee
            )
            self.ammi.add_asset(asset, delta_token)
            self.ammi.add_asset("LPTokens", LPTokenOut)
            self.user.add_asset("LPTokens", LPTokenOut)
            self.user.remove_asset(asset, delta_token)
            self.ammi.add_LP(self.user.user_name, LPTokenOut)
            self.monitor_VoteSlots()
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # if "Amount" and "EPrice" are specified
    # ep = Effective Price
    # TODO: simplify the function
    def deposit_Amount_EPrice(self, asset: str, ep: float, amount=""):
        if amount:
            # X = amount of LPTokenOut to be issued
            X = self.delta_LPTokens_Single(
                float(amount), self.ammi.assets[asset], self.TFee
            )
            # Y = effective-price of the trade
            Y = self.effectivePrice(X, float(amount))
            if Y <= ep:
                self.ammi.add_asset(asset, float(amount))
                self.ammi.add_asset("LPTokens", X)
                self.user.add_asset("LPTokens", X)
                self.user.remove_asset(asset, float(amount))
                self.ammi.add_LP(self.user.user_name, X)
            else:
                amount = Symbol("x")
                L = self.ammi.assets["LPTokens"] * (
                    (
                        1
                        + (amount - self.TFee * (1 - self.ammi.W) * amount)
                        / self.ammi.assets[asset]
                    )
                    ** 1
                    - self.ammi.W
                    - 1
                )
                # Q = amount of asset in
                Q = solve(amount / L - ep, amount)[0]
                # W = amount of LPToken out
                W = self.ammi.assets["LPTokens"] * (
                    (
                        1
                        + (Q - self.TFee * (1 - self.ammi.W) * Q)
                        / self.ammi.assets[asset]
                    )
                    ** self.ammi.W
                    - 1
                )
                self.ammi.add_asset(asset, Q)
                self.ammi.add_asset("LPTokens", W)
                self.user.add_asset("LPTokens", W)
                self.user.remove_asset(asset, Q)
                self.ammi.add_LP(self.user.user_name, W)
        else:
            amount = Symbol("x")
            L = self.ammi.assets["LPTokens"] * (
                (1 + (amount - self.TFee * (0.5) * amount) / self.ammi.assets[asset])
                ** 0.5
                - 1
            )
            # Q = amount of asset in
            Q = solve(amount / L - ep, amount)[0]
            # W = amount of LPToken out
            W = self.ammi.assets["LPTokens"] * (
                (1 + (Q - self.TFee * (1 - self.ammi.W) * Q) / self.ammi.assets[asset])
                ** self.ammi.W
                - 1
            )
            self.ammi.add_asset(asset, Q)
            self.ammi.add_asset("LPTokens", W)
            self.user.add_asset("LPTokens", W)
            self.user.remove_asset(asset, Q)
            self.ammi.add_LP(self.user.user_name, W)
        self.monitor_VoteSlots()


# --------------------------- WITHDRAW CLASS ---------------------------


class Withdraw(AMMVote):
    def __init__(self, user, ammi):
        super().__init__(self, ammi)
        self.user = user
        if self.user == self.ammi.AuctionSlot["slot_owner"]:
            self.TFee = self.ammi.AuctionSlot["discountedFee"]
        else:
            self.TFee = self.ammi.TFee

    # --------------------------- DOUBLE/ALL ASSET WITHDRAWALS ---------------------------

    # if "LPTokenIn" is specified
    def withdraw_LPTokenIn(self, LPTokenIn: float):
        if self.user.assets["LPTokens"] >= LPTokenIn:
            delta_A = self.delta_token_Double(
                LPTokenIn, self.ammi.assets[self.ammi.curr_codes[0]]
            )
            delta_B = self.delta_token_Double(
                LPTokenIn, self.ammi.assets[self.ammi.curr_codes[1]]
            )
            if (
                self.ammi.assets[self.ammi.curr_codes[0]] > delta_A
                and self.ammi.assets[self.ammi.curr_codes[1]] > delta_B
            ):
                self.user.remove_asset("LPTokens", LPTokenIn)
                self.ammi.remove_asset("LPTokens", LPTokenIn)
                self.ammi.remove_asset(self.ammi.curr_codes[0], delta_A)
                self.ammi.remove_asset(self.ammi.curr_codes[1], delta_B)
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
    def withdraw_Amount1_Amount2(
        self, asset_A: str, asset_B: str, amount_A: float, amount_B: float
    ):
        # Z = LPTokensIn
        Z = amount_A * self.ammi.assets["LPTokens"] / self.ammi.assets[asset_A]
        # X = amount of asset B (delta_B)
        X = Z / self.ammi.assets["LPTokens"] * self.ammi.assets[asset_B]
        if X <= amount_B and self.user.assets["LPTokens"] >= Z:
            self.ammi.remove_asset(asset_A, amount_A)
            self.ammi.remove_asset(asset_B, X)
            self.ammi.remove_asset("LPTokens", Z)
            self.user.remove_asset("LPTokens", Z)
            self.user.add_asset(asset_A, amount_A)
            self.user.add_asset(asset_B, X)
            self.ammi.remove_LP(self.user.user_name, Z)
            self.monitor_VoteSlots()
        elif X > amount_B and self.user.assets["LPTokens"] >= Z:
            # Q = LPTokensIn
            Q = amount_B * self.ammi.assets["LPTokens"] / self.ammi.assets[asset_B]
            # W = amount of asset A (delta_A)
            W = Q / self.ammi.assets["LPTokens"] * self.ammi.assets[asset_A]
            self.ammi.remove_asset(asset_A, W)
            self.ammi.remove_asset(asset_B, amount_B)
            self.ammi.remove_asset("LPTokens", Q)
            self.user.remove_asset("LPTokens", Q)
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
        if self.user.assets["LPTokens"] >= L and self.ammi.assets[asset] > amount:
            self.ammi.remove_asset(asset, amount)
            self.ammi.remove_asset("LPTokens", L)
            self.user.remove_asset("LPTokens", L)
            self.user.add_asset(asset, amount)
            self.ammi.remove_LP(self.user.user_name, L)
            self.monitor_VoteSlots()
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # if "Amount" and "LPTokenIn" are specified
    def withdraw_Amount_LPTokenIn(self, asset: str, LPTokenIn: float, amount=""):
        if self.user.assets["LPTokens"] >= LPTokenIn:
            # Y = amount of asset A
            Y = self.delta_token_WS(self.ammi.assets[asset], LPTokenIn, self.TFee)
            if (amount and Y >= float(amount)) or (not amount):
                self.ammi.remove_asset(asset, Y)
                self.ammi.remove_asset("LPTokens", LPTokenIn)
                self.user.remove_asset("LPTokens", LPTokenIn)
                self.user.add_asset(asset, Y)
                self.ammi.remove_LP(self.user.user_name, LPTokenIn)
                self.monitor_VoteSlots()
            else:
                # FAIL TX
                raise Exception("Not enough tokens")

    # if "Amount" and "EPrice" are specified
    # ep = Effective Price
    def withdraw_Amount_EPrice(self, asset: str, ep: float, amount=""):
        # computed amount
        comp_amount = Symbol("x")
        # L = amount of LPTokensIn
        L = self.ammi.assets["LPTokens"] * (
            1
            - (
                1
                - comp_amount
                / (self.ammi.assets[asset] * (1 - (1 - self.ammi.W) * self.TFee))
            )
            ** self.ammi.W
        )
        # Y = amount of asset/token out
        Y = solve(ep * comp_amount - L, comp_amount)[0]
        # X = asset in as LPTokenIn
        X = self.delta_LPTokens_WS(Y, self.ammi.assets[asset], self.TFee)

        if (amount and Y >= float(amount)) or (not amount):
            self.ammi.remove_asset(asset, Y)
            self.ammi.remove_asset("LPTokens", X)
            self.user.remove_asset("LPTokens", X)
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
        if user.user_name == ammi.AuctionSlot["slot_owner"]:
            self.TFee = self.ammi.AuctionSlot["discountedFee"]
        else:
            self.TFee = self.ammi.TFee
            # print(ammi.AuctionSlot['slot_owner'])
        # print(user.user_name)

    # given amount to swap in
    def swap_given_amount_In(self, assetIn: str, assetOut: str, amount_in: float):
        # delta_tokenOut = amount of asset to swap out, given amount of the other asset to swap in
        delta_tokenOut = self.delta_tokenOut_Swap(
            self.ammi.assets[assetIn], self.ammi.assets[assetOut], amount_in, self.TFee
        )

        if (
            self.user.assets[assetIn] > amount_in
            and self.ammi.assets[assetOut] > delta_tokenOut
        ):
            self.ammi.remove_asset(assetOut, delta_tokenOut)
            self.ammi.add_asset(assetIn, amount_in)
            self.user.remove_asset(assetIn, amount_in)
            self.user.add_asset(assetOut, delta_tokenOut)
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # given amount to swap out
    def swap_given_amount_Out(self, assetIn: str, assetOut: str, amount_out: float):
        # delta_tokenIn = amount of asset to swap in, given amount of the other asset to swap out
        if self.ammi.assets[assetOut] > amount_out:
            delta_tokenIn = self.delta_tokenIn_Swap(
                self.ammi.assets[assetIn],
                self.ammi.assets[assetOut],
                amount_out,
                self.TFee,
            )
            if self.user.assets[assetIn] > delta_tokenIn:
                self.ammi.remove_asset(assetOut, amount_out)
                self.ammi.add_asset(assetIn, delta_tokenIn)
                self.user.remove_asset(assetIn, delta_tokenIn)
                self.user.add_asset(assetOut, amount_out)
                return delta_tokenIn, amount_out, delta_tokenIn * self.TFee
            else:
                # FAIL TX
                raise Exception("Not enough tokens")
        else:
            # FAIL TX
            raise Exception("Not enough tokens")

    # swap given the desired spot price the user wants token Out to reach
    def swap_given_postSP(
        self,
        assetIn: str,
        assetOut: str,
        balAssetIn: float,
        balAssetOut: float,
        pre_sp: float,
        post_sp: float,
        amount_in=None,
        skip_pool_update=False,
    ):
        if self.user == self.ammi.AuctionSlot["slot_owner"]:
            TFee = 0
        else:
            TFee = self.ammi.TFee

        delta_tokenIn = amount_in or self.delta_tokenIn_given_spotprices(
            balAssetIn, pre_sp, post_sp
        )
        delta_tokenOut = self.delta_tokenOut_Swap(
            balAssetIn, balAssetOut, delta_tokenIn, TFee
        )
        if delta_tokenIn > 0 and delta_tokenOut > 0:
            if (
                self.ammi.assets[assetOut] > delta_tokenOut
                and self.user.assets[assetIn] > delta_tokenIn
            ):
                if skip_pool_update:
                    return delta_tokenIn, delta_tokenOut, delta_tokenIn * TFee
                else:
                    self.ammi.remove_asset(assetOut, delta_tokenOut)
                    self.ammi.add_asset(assetIn, delta_tokenIn)
                    self.user.remove_asset(assetIn, delta_tokenIn)
                    self.user.add_asset(assetOut, delta_tokenOut)
                    return TFee == 0
            else:
                # FAIL TX
                raise Exception("Not enough tokens")
        elif delta_tokenIn <= 0 and delta_tokenOut <= 0:
            if (
                self.ammi.assets[assetIn] > delta_tokenIn
                and self.user.assets[assetOut] > delta_tokenOut
            ):
                if skip_pool_update:
                    return delta_tokenIn, delta_tokenOut, delta_tokenIn * TFee
                else:
                    self.user.remove_asset(assetOut, delta_tokenOut)
                    self.user.add_asset(assetIn, delta_tokenIn)
                    self.ammi.remove_asset(assetIn, delta_tokenIn)
                    self.ammi.add_asset(assetOut, delta_tokenOut)
                    return TFee == 0
            else:
                # FAIL TX
                raise Exception("Not enough tokens")
        else:
            # delta_tokenIn and delta_tokenOut can't have different signs
            raise Exception("Error?")


# ---------------------------   AMMBid Class       ---------------------------


class AMMBid(AMMVote):
    # in this version, assume max only 1 user per slot instead of 4
    def __init__(self, user, ammi):
        super().__init__(self, ammi)
        self.user = user

    # possible values for slot_time_interval (t): 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, ..., 0.9, 0.95, 1
    def compute_price(self, slot_time_interval: float) -> float:
        if (not self.ammi.AuctionSlot["slot_owner"]) or (slot_time_interval == 1):
            minBidPrice = self.ammi.MinSlotPrice
        elif slot_time_interval == 0.05:
            minBidPrice = self.ammi.auction_slot_price * 1.05 + self.ammi.MinSlotPrice
        elif 0.1 <= slot_time_interval <= 1:
            minBidPrice = (
                self.ammi.auction_slot_price * 1.05 * (1 - slot_time_interval**60)
                + self.ammi.MinSlotPrice
            )
        return minBidPrice

    def bid(
        self,
        slot_time_interval: float,
        min_price=None,
        max_price=None,
        skip_pool_update=False,
    ):
        minBidPrice = self.compute_price(slot_time_interval)

        if minBidPrice > self.user.assets["LPTokens"]:
            # raise Exception('Not enough tokens')
            return None, None

        if min_price:
            if float(min_price) > self.user.assets["LPTokens"]:
                # raise Exception('Not enough tokens')
                bidPrice = None
                return None, None
        if max_price:
            if float(max_price) > self.user.assets["LPTokens"]:
                # raise Exception('Not enough tokens')
                bidPrice = None
                return None, None

        bidPrice = minBidPrice
        if min_price and max_price:
            bidPrice = (
                max(minBidPrice, min_price) if minBidPrice < max_price else None
            )  # else fail
        elif min_price and not max_price:
            bidPrice = max(minBidPrice, min_price)
        elif max_price and not min_price:
            bidPrice = minBidPrice if max_price >= minBidPrice else None  # else fail
            # FAIL

        refund = (
            0
            if not self.ammi.AuctionSlot["slot_owner"] or slot_time_interval == 1
            else (1 - slot_time_interval) * self.ammi.auction_slot_price
        )

        if (
            bidPrice is not None
            and bidPrice > refund
            and skip_pool_update == True
            and bidPrice
            < min(self.ammi.assets["LPTokens"], self.user.assets["LPTokens"])
        ):
            return bidPrice, refund
        elif not bidPrice and skip_pool_update == True:
            return None, None

        if (
            bidPrice
            and bidPrice > refund
            and skip_pool_update == False
            and bidPrice
            < min(self.ammi.assets["LPTokens"], self.user.assets["LPTokens"])
        ):
            if bidPrice and self.ammi.AuctionSlot["slot_owner"]:
                self.ammi.AuctionSlot["slot_owner"].add_asset("LPTokens", refund)
                self.ammi.add_LP(self.ammi.AuctionSlot["slot_owner"].user_name, refund)
                # burn the remaining LPTokens
                self.ammi.remove_asset("LPTokens", bidPrice - refund)
                self.user.remove_asset("LPTokens", bidPrice)
                self.ammi.auction_slot_price = bidPrice
                self.ammi.AuctionSlot = {
                    "slot_owner": self.user,
                    "t": slot_time_interval,
                    "discountedFee": 0,
                    "price": bidPrice,
                }
                self.ammi.remove_LP(
                    self.ammi.AuctionSlot["slot_owner"].user_name, bidPrice
                )
            elif bidPrice and not self.ammi.AuctionSlot["slot_owner"]:
                self.ammi.AuctionSlot = {
                    "slot_owner": self.user,
                    "t": slot_time_interval,
                    "discountedFee": 0,
                    "price": bidPrice,
                }
                self.ammi.auction_slot_price = bidPrice
                # burn the LPTokens paid for the slot
                self.ammi.remove_asset("LPTokens", bidPrice)
                self.ammi.remove_LP(self.user.user_name, bidPrice)
                self.user.remove_asset("LPTokens", bidPrice)
            self.monitor_VoteSlots()
            return bidPrice, refund
