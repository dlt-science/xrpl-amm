from typing import Any, Union, List
# from collections.abc import Iterable
# from amms.xrpl.amm.actions import Swap, Deposit, AMMBid
# from amms.xrpl.amm.env import User, AMMi
from amms.uniswap.uniswap import Uniswap_amm
import numpy as np


class SimulatorComputations:
    def compute_slippage(self, pre_sp: float, amount_in: float, final_amount_out: float) -> float:
        """Calculate the slippage of a transaction.

        Args:
            max_slippage (float): The maximum slippage allowed.
            pre_sp (float): The asset price when the user placed the transaction.
            amount_in (float): The amount paid by the user.
            final_amount_out (float): The amount the user received in return of amount_in.

        Returns:
            tuple:
                - (bool): True if slippage is lower than the maximum slippage tolerance.
                - (float): Slippage value of the transaction.
        """

        effective_price = amount_in/final_amount_out
        slippage = (effective_price / pre_sp) - 1
        return abs(slippage)

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
            balAsset = amm.asset_A_amount if asset == 'A' else amm.asset_B_amount
        else:
            balAsset = amm.assets[asset]
        return balAsset

    def LPT_price(self, reserve_LPT, reserve_A, reserve_B, price_A, price_B):
        """Compute the price of an LPToken"""
        return (price_A*reserve_A + price_B*reserve_B)/reserve_LPT

    def compute_impermanent_loss1(self, initial_A_reserve: float, initial_B_reserve: float, current_reserve_A: float, current_reserve_B: float, spot_price: float,
                                  current_external_price: float, amm_initial_A_price, pct_change, bids_profit: float = 0) -> float:
        """Compute the impermanent loss experienced by LPs"""
        value_held = (initial_A_reserve * current_external_price + initial_B_reserve) / \
            (initial_A_reserve*current_external_price + initial_B_reserve)
        return (current_reserve_A*spot_price + current_reserve_B + bids_profit) - value_held

    def compute_impermanent_loss(self, initial_A_reserve: float, initial_B_reserve: float, current_reserve_A: float, current_reserve_B: float,
                                 spot_price: float, current_external_price: float, amm_initial_A_price, pct_change, bids_profit: float = 0) -> float:
        """Compute the impermanent loss experienced by LPs"""
        original_pool_value = initial_A_reserve * \
            amm_initial_A_price + initial_B_reserve
        value_held = original_pool_value + initial_A_reserve*amm_initial_A_price*pct_change

        new_pool_value = current_reserve_A * spot_price + current_reserve_B + bids_profit

        return new_pool_value/value_held - 1

    def compute_ArbitrageursProfits_advantage(self, amm1_profits: list, amm2_profits: list, profits_advantage: int) -> int:
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


    def compute_PriceGap_advantage(self, external_prices: list, amm1_prices: list, amm2_prices: list, PG_advantage: int) -> int:
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

        mae_values = [self.mean_absolute_error(
            external_prices, amm_prices) for amm_prices in [amm1_prices, amm2_prices]]
        PG_advantage += mae_values[0] <= mae_values[1]
        return PG_advantage

    def compute_LP_returns_advantage(self, amm1_LP_returns: float, amm2_LP_returns: float, LP_returns_advantage: int) -> int:
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

    def compute_initial_B_reserve(self, initial_target_A_price: float, trading_fee: float) -> float:
        """Get asset B initial reserve to create an AMM.

        Args:
            initial_target_A_price (float): AMM Asset A price we wish the simulation to begin with.
                Should be equal to the initial external market price.
            trading_fee (float): AMM pool trading fee.

        Returns:
            float: Asset B reserve to initilize the AMMs. 
        """

        initial_reserve_A = 1000
        initial_reserve_B = 1000
        step = 10
        while abs(((initial_reserve_B/initial_reserve_A) / (1 - trading_fee) - initial_target_A_price)) > 1e-6:
            initial_reserve_B += step
            new_value = abs(
                ((initial_reserve_B/initial_reserve_A) / (1 - trading_fee) - initial_target_A_price))
            if new_value < 1e-6:
                break
            if new_value < 10:
                step = 0.01
            elif new_value < 100:
                step = 0.1
            elif new_value < 500:
                step = 1
            else:
                step = 10

        return initial_reserve_B

    def compute_minimum_bid_price(self, amm, slot_time_interval: float) -> float:
        if (not amm.AuctionSlot['slot_owner']) or (slot_time_interval == 1):
            minBidPrice = amm.MinSlotPrice
        elif slot_time_interval == 0.05:
            minBidPrice = amm.auction_slot_price * 1.05 + amm.MinSlotPrice
        elif 0.1 <= slot_time_interval <= 1:
            minBidPrice = amm.auction_slot_price * 1.05 * \
                (1-slot_time_interval**60) + amm.MinSlotPrice
        return minBidPrice
