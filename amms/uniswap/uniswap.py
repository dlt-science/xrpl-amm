import math


class Amm:
    def __init__(self, fee_rate, asset_A_amount, asset_B_amount):
        # initialize the AMM

        self.fee_rate = fee_rate
        self.asset_A_amount = asset_A_amount
        self.asset_B_amount = asset_B_amount
        self.total_LP_token = 0

    def print_detailed_info(self):
        # print detailed info of the Amm

        print("Total number of outstanding tokens: ", self.total_LP_token)
        print("A reserves: ", self.asset_A_amount)
        print("B reserves: ", self.asset_B_amount)
        print("Transaction fee:", self.fee_rate)


class Uniswap_amm(Amm):
    def __init__(
        self, fee_rate, asset_A_amount, asset_B_amount, initial_LP_token_number
    ):
        # initialize the Uniswap_amm

        super(Uniswap_amm, self).__init__(fee_rate, asset_A_amount, asset_B_amount)
        self.total_LP_token = initial_LP_token_number
        self.constant = self.asset_A_amount * self.asset_B_amount

    def total_liquidity(self):
        # Here, use sqrt(XY) to calculate the total liquidity
        # Output: float

        return math.sqrt(self.asset_A_amount * self.asset_B_amount)

    def deposit(self, type_of_added_asset, amount_of_added_asset):
        # input the type of asset to deposit (str: 'A' or 'B') and the amount of the asset (float)
        # return the amount of another type of asset (float) to deposit and the number of returned LP tokens (float)

        total_liquidity_before = self.total_liquidity()

        # db = Bda/A
        # da = Adb/B
        if type_of_added_asset == "A":
            amount_of_added_B_asset = (
                self.asset_B_amount * amount_of_added_asset / self.asset_A_amount
            )
            self.asset_A_amount += amount_of_added_asset
            self.asset_B_amount += amount_of_added_B_asset

            # S = (L1-L0)/L0 * T
            total_liquidity_after = self.total_liquidity()
            number_of_new_tokens = (
                (total_liquidity_after - total_liquidity_before)
                / total_liquidity_before
                * self.total_LP_token
            )
            self.total_LP_token += number_of_new_tokens
            return amount_of_added_B_asset, number_of_new_tokens

        elif type_of_added_asset == "B":
            amount_of_added_A_asset = (
                self.asset_A_amount * amount_of_added_asset / self.asset_B_amount
            )
            self.asset_B_amount += amount_of_added_asset
            self.asset_A_amount += amount_of_added_A_asset

            # S = (L1-L0)/L0 * T
            total_liquidity_after = self.total_liquidity()
            number_of_new_tokens = (
                (total_liquidity_after - total_liquidity_before)
                / total_liquidity_before
                * self.total_LP_token
            )
            self.total_LP_token += number_of_new_tokens
            return amount_of_added_A_asset, number_of_new_tokens

        else:
            raise Exception("Wrong input! Enter eithor A or B for asset type!")

    def withdraw(self, LP_tokens_to_burn):
        # input the number of LP tokens (float) to burn
        # return the number of token A (float) and B (float) to withdraw

        # dx = X * S/T
        # dy = Y * S/T
        A_to_withdraw = self.asset_A_amount * LP_tokens_to_burn / self.total_LP_token
        B_to_withdraw = self.asset_B_amount * LP_tokens_to_burn / self.total_LP_token

        # update the AMM
        self.asset_A_amount -= A_to_withdraw
        self.asset_B_amount -= B_to_withdraw
        self.total_LP_token -= LP_tokens_to_burn

        return A_to_withdraw, B_to_withdraw

    def spot_price(self, asset_type, other_asset, discounted_fee=False):
        # input the asset type (str: 'A' or 'B')
        # return the reference/spot price (float) for this type of asset

        trans_fee_multiplier = 1 / (1 - self.fee_rate)

        if asset_type == "A":
            return self.asset_B_amount / self.asset_A_amount * trans_fee_multiplier
        elif asset_type == "B":
            return self.asset_A_amount / self.asset_B_amount * trans_fee_multiplier
        else:
            raise Exception("Wrong input! Enter eithor A or B!")

    def swap(self, target_asset_type, amount, SP_price):
        # input the type of asset you want to buy (str: 'A' or 'B'),
        # the amount of that asset (float)
        # and the reference/spot price (float) for that

        # return the final price and slippage

        # dx = Xdy/(Y-dy)

        if target_asset_type == "A":
            # you need to pay B in this case
            amount_without_fee = (
                self.asset_B_amount * amount / (self.asset_A_amount - amount)
            )
            # update the pool
            self.asset_A_amount -= amount
            self.asset_B_amount += amount_without_fee
            # swap fee
            fee = amount_without_fee * self.fee_rate
            final_amount = amount_without_fee + fee
            # deposit the swap fee to the pool
            self.asset_B_amount += fee
            self.constant = self.asset_A_amount * self.asset_B_amount

        elif target_asset_type == "B":
            # you need to pay A in this case
            amount_without_fee = (
                self.asset_A_amount * amount / (self.asset_B_amount - amount)
            )
            # update the pool
            self.asset_A_amount += amount_without_fee
            self.asset_B_amount -= amount
            # swap fee
            fee = amount_without_fee * self.fee_rate
            final_amount = amount_without_fee + fee
            # deposit the swap fee to the pool
            self.asset_A_amount += fee
            self.constant = self.asset_A_amount * self.asset_B_amount

        else:
            raise Exception("Wrong input! Enter eithor A or B!")

        effective_price = final_amount / amount
        slippage = (float(effective_price) - float(SP_price)) / float(SP_price)

        # return final_amount, slippage
        return final_amount, amount, fee

    def delta_tokenIn_given_spotprices(
        self, balAssetIn: float, pre_sp: float, post_sp: float
    ) -> float:
        # pre_sp = spot price before trade
        # post_sp = spot price after trade
        W = 0.5
        delta_tokenIn = balAssetIn * ((post_sp / pre_sp) ** (W / (W + W)) - 1)
        return delta_tokenIn

    def delta_tokenOut_Swap(
        self, balAssetIn: float, balAssetOut: float, delta_tokenIn: float
    ) -> float:
        # delta_tokenIn = amount of asset to swap in
        W = 0.5
        delta_tokenOut = balAssetOut * (
            1
            - (balAssetIn / (balAssetIn + delta_tokenIn * (1 - self.fee_rate)))
            ** (W / W)
        )
        return delta_tokenOut

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
        delta_tokenIn = amount_in or self.delta_tokenIn_given_spotprices(
            balAssetIn, pre_sp, post_sp
        )

        delta_tokenOut = self.delta_tokenOut_Swap(
            balAssetIn, balAssetOut, delta_tokenIn
        )

        if delta_tokenIn > 0 and delta_tokenOut > 0:
            if assetIn == "A":
                if self.asset_B_amount > delta_tokenOut:
                    if skip_pool_update:
                        return (
                            delta_tokenIn,
                            delta_tokenOut,
                            delta_tokenIn * self.fee_rate,
                        )
                    else:
                        self.asset_B_amount -= delta_tokenOut
                        self.asset_A_amount += delta_tokenIn
                        return (
                            delta_tokenIn,
                            delta_tokenOut,
                            delta_tokenIn * self.fee_rate,
                        )
                else:
                    # FAIL TX
                    raise Exception("Not enough tokens")
            elif assetIn == "B":
                if self.asset_A_amount > delta_tokenOut:
                    if skip_pool_update:
                        return (
                            delta_tokenIn,
                            delta_tokenOut,
                            delta_tokenIn * self.fee_rate,
                        )
                    else:
                        self.asset_A_amount -= delta_tokenOut
                        self.asset_B_amount += delta_tokenIn
                        return (
                            delta_tokenIn,
                            delta_tokenOut,
                            delta_tokenIn * self.fee_rate,
                        )
                else:
                    # FAIL TX
                    raise Exception("Not enough tokens")

        elif delta_tokenIn <= 0 and delta_tokenOut <= 0:
            if assetIn == "A":
                if self.asset_A_amount > delta_tokenOut:
                    if skip_pool_update:
                        return (
                            delta_tokenIn,
                            delta_tokenOut,
                            delta_tokenIn * self.fee_rate,
                        )
                    else:
                        self.asset_A_amount -= delta_tokenOut
                        self.asset_B_amount += delta_tokenIn
                        return (
                            delta_tokenIn,
                            delta_tokenOut,
                            delta_tokenIn * self.fee_rate,
                        )
                else:
                    # FAIL TX
                    raise Exception("Not enough tokens")
            elif assetIn == "B":
                if self.asset_B_amount > delta_tokenOut:
                    if skip_pool_update:
                        return (
                            delta_tokenIn,
                            delta_tokenOut,
                            delta_tokenIn * self.fee_rate,
                        )
                    else:
                        self.asset_B_amount -= delta_tokenOut
                        self.asset_A_amount += delta_tokenIn
                        return (
                            delta_tokenIn,
                            delta_tokenOut,
                            delta_tokenIn * self.fee_rate,
                        )
                else:
                    # FAIL TX
                    raise Exception("Not enough tokens")
        else:
            # delta_tokenIn and delta_tokenOut can't have different signs
            raise Exception("Error?")
