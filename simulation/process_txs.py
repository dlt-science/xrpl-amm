import random
from typing import Any, Union, List
from collections.abc import Iterable

# from amms.xrpl.amm.actions import Swap, Deposit, AMMBid
from amms.xrpl.amm.env import User, AMMi
from amms.uniswap.uniswap import Uniswap_amm


class ProcessTransactions:
    def process_arbitrageurs_txs(
        self,
        amm: Union[AMMi, Uniswap_amm],
        tx: dict,
        max_slippage: float,
        current_external_price: float,
        fees: float,
        profits: list,
        arbit_txs: int,
        slippages: List[float],
        unrealized_tx: int,
        tfees: float,
        trading_volume: float,
        slot_holders_txs: int = None,
    ) -> tuple[int, float, list, list, float, int, int]:
        """Execute an arbitrageur's buy or sell order.

        Args:
            amm (obj): The AMM instance to check the balance in.
                Class object of either AMMi (for XRPL AMM) or Uniswap_amm classes.
            tx (dict): Transaction details of the following form:
                {
                    'time-step' (int): Time-step when transaction was placed.
                    'external_market_price' (float): External market price when transaction was placed.
                        If action is 'buy', it is the price of asset A. If action is 'sell', it is the price of asset B.
                    'amm_price' (float): Asset price in the AMM instance when transaction was placed.
                        If action is 'buy', it is the price of asset A. If action is 'sell', it is the price of asset B.
                    'tx_type' (str): Transaction type; 'buy' or 'sell' asset A.
                    'amount_in' (float): Amount of asset paid by the arbitrageur to swap in the pool.
                    'process_tx' (func): Transaction function to be executed.
                }.
            max_slippage (float): Maximum slippage tolerated.
            current_external_price (float): Current external market price of asset A.
            fees (float): AMM network fees.
            profits (list): Arbitrageurs' profits.
            arbit_txs (int): Number of arbitrage transactions.
            slippages (list): Slippage values of successful transactions.
            unrealized_tx (int): Number of unprocessed transactions because of the slippage condition.
            tfees (float): Trading fees earned by LPs.
            trading_volume (float): Current pool trading volume.
            slot_holders_txs (int; Optional: None): Number of arbitrage transactions made by auction slot owners.
                For XRPL AMM with Continuous Auction Mechanism.
        Returns:
            tuple: Updated values for (arbit_txs, trading_volume, profits, slippages, tfees, unrealized_tx, slot_holders_txs)
        """

        amount_in, final_amount_out, tfee = tx["process_tx"](
            post_sp=tx["external_market_price"],
            amount_in=tx["amount_in"],
            skip_pool_update=True,
        )

        slippage = self.compute_slippage(
            tx["amm_price"], tx["amount_in"], final_amount_out
        )

        if abs(slippage) <= max_slippage:
            if isinstance(amm, AMMi) and slot_holders_txs is not None:  # (= if xrplCAM)
                slot_holders_txs += tx["process_tx"](
                    post_sp=tx["external_market_price"], amount_in=tx["amount_in"]
                )
            else:
                tx["process_tx"](
                    post_sp=tx["external_market_price"], amount_in=tx["amount_in"]
                )
            arbit_txs += 1
            trading_volume += amount_in if tx["tx_type"] == "buy" else final_amount_out
            profits.append(
                final_amount_out * current_external_price - amount_in - fees
                if tx["tx_type"] == "buy"
                else (final_amount_out / current_external_price - amount_in)
                * current_external_price
                - fees
            )
            slippages.append(slippage)
            tfees += tfee if tx["tx_type"] == "buy" else tfee * amm.spot_price("A", "B")
        else:
            unrealized_tx += 1

        return (
            arbit_txs,
            trading_volume,
            profits,
            slippages,
            tfees,
            unrealized_tx,
            slot_holders_txs,
        )

    def process_normal_users_txs(
        self, amm: Union[AMMi, Uniswap_amm], tx: dict, tfees: float
    ) -> float:
        """Execute an arbitrageur's transaction.

        Args:
            amm (obj): The AMM instance to check the balance in.
                Class object of either AMMi (for XRPL AMM) or Uniswap_amm classes.
            tx (dict): Transaction details of the following form:
                {
                    'time-step' (int): Time-step when transaction was placed.
                    'tx_type' (str): Transaction type; 'normal_user_buy' or 'normal_user_sell' asset A.
                    'process_tx' (func): Transaction function to be executed.
                }.
            tfees (float): Trading fees earned by LPs.

        Returns:
            float: Updated tfees value.
        """

        tfee = tx["process_tx"]()
        tfees += (
            tfee
            if tx["tx_type"] == "normal_user_buy"
            else tfee * amm.spot_price("A", "B")
        )
        return tfees

    def process_bids_txs(self, tx: dict, bids: list) -> list:
        """Execute an arbitrageur's bid transaction.

        For XRPL AMM with Continous Auction Mechanism (CAM) only.

        Args:
            tx (dict): Transaction details of the following form:
                {
                    'time-step' (int): Time-step when transaction was placed.
                    'tx_type' (str): Transaction type; 'bid'.
                    'process_tx' (func): Transaction function to be executed.
                }.
            bids (list): Nested list of bid transactions in the following form:
                [[
                    bidPrice in LPTokens (float),
                    tx (dict)
                ]]

        Returns:
            list: Updated list of bids placed.
        """
        bidPrice, refund = tx["process_tx"](skip_pool_update=True)
        bids.append([bidPrice, refund, tx]) if bidPrice else None
        return bids

    def process_all_txs(
        self,
        time_step: int,
        amm: Union[AMMi, Uniswap_amm],
        current_block: Iterable,
        max_slippage: float,
        external_prices: list[float],
        fees: float,
        profits: list,
        arbit_txs,
        slippages: list[float],
        unrealized_tx: int,
        tfees: float,
        trading_volume: int,
        auction_slot_price: list = None,
        slot_holders_txs: int = None,
        bids_profit: float = None,
        bids_refunds: float = None,
    ):
        """Execute all transactions after shuffling their order in the block.

        Args:
            time-step (int): Time-step when transaction was placed.
            amm (obj): The AMM instance to check the balance in.
                Class object of either AMMi (for XRPL AMM) or Uniswap_amm classes.
            current_block (list): Block containing all transactions to be processed.
            max_slippage (float): Maximum slippage tolerated.

            external_prices (list): Current external market price of asset A.
            fees (float): AMM network fees.
            profits (list): Arbitrageurs' profits.
            arbit_txs (int): Number of arbitrage transactions.
            slippages (list): Slippage values of successful transactions.
            unrealized_tx (int): Number of unprocessed transactions because of the slippage condition.
            tfees (float): Trading fees earned by LPs.
            trading_volume (int): Current pool trading volume.
            auction_slot_price (list; Optional: None): Previous winning bids value.
                For XRPL AMM with Continuous Auction Mechanism (CAM).
            slot_holders_txs (int; Optional: None): Number of arbitrage transactions made by auction slot owners.
                For XRPL AMM with Continuous Auction Mechanism (CAM).
            bids_profit (float; Optional: None): Profits made by LPs through arbitrageurs' bids.
                For XRPL AMM with Continuous Auction Mechanism (CAM).

        Returns:
            tuple: Updated list of bids placed.
        """

        bid_won = False
        if current_block:
            # shuffle the transactions in the current block to randomize their order of execution
            random.shuffle(current_block)
            bids_orders = []
            for tx in current_block:
                if tx["tx_type"] in ["buy", "sell"]:
                    (
                        arbit_txs,
                        trading_volume,
                        profits,
                        slippages,
                        tfees,
                        unrealized_tx,
                        slot_holders_txs,
                    ) = self.process_arbitrageurs_txs(
                        amm,
                        tx,
                        max_slippage,
                        external_prices[time_step],
                        fees,
                        profits,
                        arbit_txs,
                        slippages,
                        unrealized_tx,
                        tfees,
                        trading_volume,
                        slot_holders_txs,
                    )

                # arbitrageur bid tx
                elif tx["tx_type"] == "bid":
                    bids_orders = self.process_bids_txs(tx, bids_orders)

                # normal user tx
                else:
                    tfees = self.process_normal_users_txs(amm, tx, tfees)

            if bids_orders:
                max_bid = max(bids_orders, key=lambda x: x[0])
                max_bid_index = bids_orders.index(max_bid)
                bid_price, refund = bids_orders[max_bid_index][2]["process_tx"](
                    skip_pool_update=False
                )  # bid_price == max_bid[0]
                auction_slot_price.append([time_step, bid_price])
                bid_won = True if bid_price is not None else False

                bids_profit["LPTokens"] += bid_price - refund
                bids_profit["B"] += (bid_price - refund) * self.LPT_price(
                    amm.assets["LPTokens"],
                    amm.assets["A"],
                    amm.assets["B"],
                    amm.spot_price("A", "B"),
                    1,
                )

                bf = (bid_price - refund) * self.LPT_price(
                    amm.assets["LPTokens"],
                    amm.assets["A"],
                    amm.assets["B"],
                    amm.spot_price("A", "B"),
                    1,
                )
                bids_refunds["LPTokens"] += refund
                bids_refunds["A"] += refund * self.LPT_price(
                    amm.assets["LPTokens"],
                    amm.assets["A"],
                    amm.assets["B"],
                    1,
                    amm.spot_price("B", "A"),
                )
                bids_refunds["B"] += refund * self.LPT_price(
                    amm.assets["LPTokens"],
                    amm.assets["A"],
                    amm.assets["B"],
                    amm.spot_price("A", "B"),
                    1,
                )
                profits.append(-bf)
                profits.append(
                    refund
                    * self.LPT_price(
                        amm.assets["LPTokens"],
                        amm.assets["A"],
                        amm.assets["B"],
                        amm.spot_price("A", "B"),
                        1,
                    )
                )

        current_block = []
        sp = amm.spot_price("A", "B")
        sp_B = amm.spot_price("B", "A")

        return (
            sp,
            sp_B,
            profits,
            arbit_txs,
            current_block,
            slippages,
            unrealized_tx,
            tfees,
            trading_volume,
            bid_won,
            auction_slot_price,
            slot_holders_txs,
            bids_profit,
            bids_refunds,
        )
