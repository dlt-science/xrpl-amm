from typing import Any, Union, List

# from collections.abc import Iterable
from amms.xrpl.amm.actions import Swap
from amms.xrpl.amm.env import User
from amms.uniswap.uniswap import Uniswap_amm
import numpy as np
import random


class SimulatorComputations:
    def normal_user_action(self, prev_action: str = "buy") -> str:
        """Determine if the user should buy, sell or pass.

        There's an 80% chance that a normal user performs an action (buy or sell).
        Out of the 80%, if the previous user bought, the current user has a 60% probability
        of buying and 40% probability of selling (and vice versa if the previous user sold).

        Args:
            prev_action (str): The previous user's action.

        Returns:
            str: The action the current user will perform (buy, sell, pass).
        """

        # probability that the user performs an action (either buy or sell)
        buy_sell_prob = 0.8
        # probability that the user doesn't perform any action
        pass_prob = 1 - buy_sell_prob
        action = random.choices(
            ["buy", "sell", "pass"], [buy_sell_prob, buy_sell_prob, pass_prob]
        )[0]
        if action in ["buy", "sell"]:
            if prev_action == "buy":
                buy_prob, sell_prob = 0.6, 0.4
                action = random.choices(
                    [action], [buy_prob if action == "buy" else sell_prob]
                )[0]
            if prev_action == "sell":
                buy_prob, sell_prob = 0.4, 0.6
                action = random.choices(
                    [action], [buy_prob if action == "buy" else sell_prob]
                )[0]
        return action

    def compute_slippage(
        self, pre_sp: float, amount_in: float, final_amount_out: float
    ) -> float:
        """Calculate the slippage of a transaction.

        Args:
            pre_sp (float): The asset price when the user placed the transaction.
            amount_in (float): The amount paid by the user.
            final_amount_out (float): The amount the user received in return of amount_in.

        Returns:
            tuple:
                - (bool): True if slippage is lower than the maximum slippage tolerance.
                - (float): Slippage value of the transaction.
        """

        effective_price = amount_in / final_amount_out
        slippage = (effective_price / pre_sp) - 1
        return abs(slippage)

    def price_impact(self, pre_sp, post_sp):
        return (post_sp - pre_sp) / pre_sp

    def get_balAsset(self, amm, asset):
        """Get the reserve of the specified asset in the specified AMM.

        Args:
            amm (obj): The AMM instance to check the balance in.
                Class object of either AMMi (for XRPL AMM) or Uniswap_amm classes.
            asset (str): The asset we want to know the balance of
        Returns:
            float: The asset reserve in the AMM instance.
        """

        if isinstance(amm, Uniswap_amm):
            balAsset = amm.asset_A_amount if asset == "A" else amm.asset_B_amount
        else:
            balAsset = amm.assets[asset]
        return balAsset

    def LPT_price(self, reserve_LPT, reserve_A, reserve_B, price_A, price_B):
        """Compute the price of an LPToken"""
        return (price_A * reserve_A + price_B * reserve_B) / reserve_LPT

    def compute_impermanent_loss1(
        self,
        initial_A_reserve: float,
        initial_B_reserve: float,
        current_reserve_A: float,
        current_reserve_B: float,
        spot_price: float,
        current_external_price: float,
        amm_initial_A_price,
        pct_change,
        bids_profit: float = 0,
    ) -> float:
        """Compute the impermanent loss experienced by LPs"""
        value_held = (
            initial_A_reserve * current_external_price + initial_B_reserve
        ) / (initial_A_reserve * current_external_price + initial_B_reserve)
        return (
            current_reserve_A * spot_price + current_reserve_B + bids_profit
        ) - value_held

    def compute_impermanent_loss(
        self,
        initial_A_reserve: float,
        initial_B_reserve: float,
        current_reserve_A: float,
        current_reserve_B: float,
        spot_price: float,
        current_external_price: float,
        amm_initial_A_price,
        pct_change,
        bids_profit: float = 0,
    ) -> float:
        """Compute the impermanent loss experienced by LPs"""
        original_pool_value = (
            initial_A_reserve * amm_initial_A_price + initial_B_reserve
        )
        value_held = (
            original_pool_value + initial_A_reserve * amm_initial_A_price * pct_change
        )

        new_pool_value = (
            current_reserve_A * spot_price + current_reserve_B + bids_profit
        )

        return new_pool_value / value_held - 1

    def compute_impermanent_loss(
        self,
        a,
        initial_A_reserve: float,
        initial_B_reserve: float,
        current_reserve_A: float,
        current_reserve_B: float,
        spot_price: float,
        current_external_price: float,
        amm_initial_A_price,
        pct_change,
        bids_profit: float = 0,
    ) -> float:
        """Compute the impermanent loss experienced by LPs"""
        original_pool_value = (
            initial_A_reserve * amm_initial_A_price + initial_B_reserve
        )
        value_held = (
            original_pool_value + initial_A_reserve * amm_initial_A_price * pct_change
        )

        new_pool_value = (
            current_reserve_A * spot_price + current_reserve_B + bids_profit
        )

        return new_pool_value / value_held - 1

    def compute_ArbitrageursProfits_advantage(
        self, amm1_profits: list, amm2_profits: list, profits_advantage: int
    ) -> int:
        """Compare arbitrageurs' profits in both AMMs.

        Adds 1 to profits_advantage if profits in amm1 => profits in amm2.

        Args:
            amm1_profits (list): Arbitrageurs' profits in amm1.
            amm2_profits (list): Arbitrageurs' profits in amm2.
            profits_advantage (int): Number of times profits in amm1 => profits in amm2.

        Returns:
            int: Updated profits_advantage value.
        """

        profits_advantage += sum(amm1_profits) >= sum(amm2_profits)
        return profits_advantage

    def mean_absolute_error(self, external_prices: list, amm_prices: list) -> float:
        """Compute the Mean Absolute Error (MAE) of prices between the reference market and the AMM"""
        return np.mean(np.abs(np.array(external_prices) - np.array(amm_prices)))

    def compute_PriceGap_advantage(
        self,
        external_prices: list,
        amm1_prices: list,
        amm2_prices: list,
        PG_advantage: int,
    ) -> int:
        """Compare price differences in both AMMs to the external market using MAE.

        Adds 1 to PG_advantage if amm1's MAE <= amm2's MAE
        (if the price gap in amm1 is smaller than in amm2 compared to ext. market).

        Args:
            external_prices (list): Asset A prices in external market.
            amm1_prices (list): Asset A prices in amm1.
            amm2_prices (list): Asset A prices in amm2.
            PG_advantage (int): Number of times amm1's prices were closer
                to ext. market compared to amm2.

        Returns:
            int: Updated PG_advantage value.
        """

        mae_values = [
            self.mean_absolute_error(external_prices, amm_prices)
            for amm_prices in [amm1_prices, amm2_prices]
        ]
        PG_advantage += mae_values[0] <= mae_values[1]
        return PG_advantage

    def compute_LP_returns_advantage(
        self, amm1_LP_returns: float, amm2_LP_returns: float, LP_returns_advantage: int
    ) -> int:
        """Compare LPs' returns in both AMMs.

        Adds 1 to LP_returns_advantage if returns in amm1 => returns in amm2.

        Args:
            amm1_LP_returns (float): LPs' returns in amm1.
            amm2_LP_returns (float): LPs' returns in amm2.
            LP_returns_advantage (int): Number of times returns in amm1 => profits in amm2.

        Returns:
            int: Updated LP_returns_advantage value.
        """

        LP_returns_advantage += amm1_LP_returns >= amm2_LP_returns
        return LP_returns_advantage

    def compute_initial_B_reserve(
        self,
        initial_target_A_price: float,
        initial_reserve_A: float,
        trading_fee: float,
    ) -> int:
        """Get asset B initial reserve to create an AMM.

        Args:
            initial_target_A_price (float): AMM Asset A price we wish the simulation to begin with.
                Should be equal to the initial external market price.
            initial_reserve_A (float): AMM asset A initial reserve we wish the simulation to begin with.
            trading_fee (float): AMM pool trading fee.

        Returns:
            float: Asset B reserve to initilize the AMMs.
        """
        tolerance = 1e-8
        lower_bound = 0
        upper_bound = 1e100

        while upper_bound - lower_bound > tolerance:
            mid = (upper_bound + lower_bound) / 2
            current_value = (mid / initial_reserve_A) / (1 - trading_fee)

            if abs(current_value - initial_target_A_price) < tolerance:
                return int(mid)
            elif current_value < initial_target_A_price:
                lower_bound = mid
            else:
                upper_bound = mid

    def compute_minimum_bid_price(self, amm, slot_time_interval: float) -> float:
        if (not amm.AuctionSlot["slot_owner"]) or (slot_time_interval == 1):
            minBidPrice = amm.MinSlotPrice
        elif slot_time_interval == 0.05:
            minBidPrice = amm.auction_slot_price * 1.05 + amm.MinSlotPrice
        elif 0.1 <= slot_time_interval <= 1:
            minBidPrice = (
                amm.auction_slot_price * 1.05 * (1 - slot_time_interval**60)
                + amm.MinSlotPrice
            )
        return minBidPrice

    def estimate_profits(
        self,
        initial_A_reserve,
        tfee_rate,
        external_prices,
        xrpl_block_conf,
        xrpl_fees,
        max_slippage,
        safe_profit_margin,
        one_day,
        normal_users,
    ):
        initial_B_reserve = self.compute_initial_B_reserve(
            initial_target_A_price=external_prices[0],
            initial_reserve_A=initial_A_reserve,
            trading_fee=tfee_rate,
        )
        # ESTIMATE PROFITS
        bobCAM_0fee = User(
            user_name="bobCAM_0fee", assets={"XRP": 1000, "A": 1e450, "B": 1e450}
        )
        xrplCAM_0fee = bobCAM_0fee.createAMM(
            ammID=3,
            asset1="A",
            asset2="B",
            amount1=initial_A_reserve,
            amount2=initial_B_reserve,
            TFee=tfee_rate,
        )
        arbit_xrplCAM_0fee = User(
            user_name="arbit_xrplCAM_0fee", assets={"XRP": 1000, "A": 1e450, "B": 1e450}
        )
        xrplCAM_0fee.AuctionSlot["slot_owner"] = arbit_xrplCAM_0fee
        bobCAM_0fee_Swaps, arbit_xrplCAM_0fee_Swaps = Swap(
            bobCAM_0fee, xrplCAM_0fee
        ), Swap(arbit_xrplCAM_0fee, xrplCAM_0fee)
        current_xrplCAM_0fee_block = []
        xrplCAM_0fee_profits = []
        xrplCAM_0fee_profits_total = []
        action = "buy"
        for time_step in range(len(external_prices)):
            if time_step and time_step % one_day == 0:
                xrplCAM_0fee_profits_total.append(sum(xrplCAM_0fee_profits))
                xrplCAM_0fee_profits = []
            if time_step % xrpl_block_conf == 0:
                (
                    xrplCAM_0fee_profits,
                    current_xrplCAM_0fee_block,
                ) = self.process_all_txs(
                    time_step,
                    xrplCAM_0fee,
                    current_xrplCAM_0fee_block,
                    max_slippage,
                    external_prices,
                    xrpl_fees,
                    xrplCAM_0fee_profits,
                    0,
                    [],
                    0,
                    0,
                    0,
                    [],
                    estimate=True,
                )
            for _ in range(normal_users):
                action = self.normal_user_action(action)
                amount = random.uniform(0.01, 1)
                current_xrplCAM_0fee_block = self.simulate_normal_users_txs(
                    xrplCAM_0fee,
                    current_xrplCAM_0fee_block,
                    time_step,
                    action,
                    amount,
                    xrplCAM_0fee.spot_price("A", "B"),
                    swapper=bobCAM_0fee_Swaps,
                )

            current_xrplCAM_0fee_block = self.check_arbit_opportunity(
                time_step,
                xrplCAM_0fee,
                current_xrplCAM_0fee_block,
                external_prices[time_step],
                xrpl_fees,
                safe_profit_margin,
                swapper=arbit_xrplCAM_0fee_Swaps,
            )

        return xrplCAM_0fee_profits_total
