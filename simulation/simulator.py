# TODO: FIX style/code pres.

import random
from typing import Any, Union, List
# from collections.abc import Iterable
from amms.xrpl.amm.actions import Swap, Deposit, AMMBid
from amms.xrpl.amm.env import User, AMMi
from amms.uniswap.uniswap import Uniswap_amm
from simulation.process_txs import ProcessTransactions
from simulation.simulator_computations import SimulatorComputations
import numpy as np
import itertools


class Simulator(SimulatorComputations, ProcessTransactions):

    def normal_user_action(self, prev_action: str) -> str:
        """Determine if the user should buy, sell or pass.

        There's an 80% chance that a normal user performs an action (buy or sell).
        Out of the 80%, if the previous user bought, the current user has a 60% probability
        of buying and 40% probability of selling (and vice versa if the previous user sold).

        Args:
            prev_action (str): The previous user's action.

        Returns:
            str: The action the user will perform (buy, sell, pass).
        """

        # probability that the user performs an action (either buy or sell)
        buy_sell_prob = 0.8
        # probability that the user doesn't perform any action
        pass_prob = 1 - buy_sell_prob
        action = random.choices(['buy', 'sell', 'pass'], [
                                buy_sell_prob, buy_sell_prob, pass_prob])[0]
        if action in ['buy', 'sell']:
            if prev_action == 'buy':
                buy_prob, sell_prob = 0.6, 0.4
                action = random.choices(
                    [action], [buy_prob if action == 'buy' else sell_prob])[0]
            if prev_action == 'sell':
                buy_prob, sell_prob = 0.4, 0.6
                action = random.choices(
                    [action], [buy_prob if action == 'buy' else sell_prob])[0]
        return action

    def simulate_normal_users_txs(self, time_step: int, n_users: int, prev_action: str, current_xrplCAM_block: List[dict], current_xrpl_block: List[dict], current_uniswap_block: List[dict],
                                  xrplCAM_swapper: Swap, xrpl_swapper: Swap, uniswap_amm: Uniswap_amm, xrpl_sp: float, xrplCAM_sp: float, uniswap_sp: float, uniswap_sp_B: float) -> tuple[List[dict], List[dict], List[dict]]:
        """Simulate normal users placing trades.

        The exact same trades are added to the 3 AMMs.

        Args:
            time-step (int): Time-step when transaction was placed.
            n_users (int): Number of normal user transactions to simulate.
            prev_action (str): The previous user's action.
            current_xrplCAM_block (list): 
            current_xrpl_block (list): 
            current_uniswap_block (list): 
            xrpl_swapper (obj): The arbitrageur's swap object to perform swaps.
                Class object of the Swap class (for XRPL AMM).
            uniswap_amm (obj): Uniswap AMM object of the Uniswap_amm class.
            xrpl_sp (float): Asset A price in XRPL-AMM.
            xrplCAM_sp (float): Asset A price in XRPL-AMM-CAM.
            uniswap_sp (float): Asset A price in Uniswap.
            uniswap_sp_B (float): Asset B price in Uniswap.

        Returns:
            tuple: Updated lists for current_xrplCAM_block, current_xrpl_block and current_uniswap_block.
        """

        for _ in range(n_users):
            action = self.normal_user_action(prev_action)
            if action == 'buy':
                # user will buy between 0.01 and 1 ETH
                amount = random.uniform(0.01, 1)
                current_xrplCAM_block.append({
                    'time_step': time_step,
                    'tx_type': 'normal_user_buy',
                    'process_tx': lambda: xrplCAM_swapper.swap_given_amount_Out(
                        assetIn='B', assetOut='A', amount_out=amount)})

                current_xrpl_block.append({
                    'time_step': time_step,
                    'tx_type': 'normal_user_buy',
                    'process_tx': lambda: xrpl_swapper.swap_given_amount_Out(
                        assetIn='B', assetOut='A', amount_out=amount)})

                current_uniswap_block.append({
                    'time_step': time_step,
                    'tx_type': 'normal_user_buy',
                    'process_tx': lambda: uniswap_amm.swap('A', amount, uniswap_sp)})
            elif action == 'sell':
                # user will sell between 0.01 and 1 ETH
                amount = random.uniform(0.01, 1)
                current_xrplCAM_block.append({
                    'time_step': time_step,
                    'tx_type': 'normal_user_sell',
                    'process_tx': lambda: xrplCAM_swapper.swap_given_amount_Out(
                        assetIn='A', assetOut='B', amount_out=amount*xrplCAM_sp)})

                current_xrpl_block.append({
                    'time_step': time_step,
                    'tx_type': 'normal_user_sell',
                    'process_tx': lambda: xrpl_swapper.swap_given_amount_Out(
                        assetIn='A', assetOut='B', amount_out=amount*xrpl_sp)})

                current_uniswap_block.append({
                    'time_step': time_step,
                    'tx_type': 'normal_user_sell',
                    'process_tx': lambda: uniswap_amm.swap('B', amount*uniswap_sp, uniswap_sp_B)})
            prev_action = action
        return current_xrplCAM_block, current_xrpl_block, current_uniswap_block

    def populate_xrpl_arbs(self, num_arbs: int, amm: AMMi, is_xrplCAM: bool):
        """Create arbitrageurs in the specified AMM.

        Args:
            num_arbs (int): Number of arbitrageurs to create.
            amm (obj): The XRPL AMM instance to add arbitrageurs.
                Class object of AMMi class.
            is_xrplCAM (bool): True if the XRPL AMM is with CAM

        Returns:
            tuple: List of objects to populate the AMM with arbitrageurs. 
        """

        users_obj = []
        for i in range(1, num_arbs+1):
            username = f"xrplCAM_arbit{i}" if is_xrplCAM else f"xrpl_arbit{i}"
            user_assets = {'XRP': 1000, 'A': 1e450, 'B': 1e450}
            users_obj.append(User(username, user_assets))

        bids_obj = []
        deposits_obj = []
        swaps_obj = []
        for user in users_obj:
            bids_obj.append(AMMBid(user, amm)) if is_xrplCAM else None
            deposits_obj.append(Deposit(user, amm))
            swaps_obj.append(Swap(user, amm))

        if is_xrplCAM:
            return deposits_obj, swaps_obj, bids_obj
        return swaps_obj


    def simulate_gbm(self, s0, mu, sigma, t, n_steps):
        dt = t / n_steps
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        W = np.concatenate(([0], np.cumsum(dW)))
        time_points = np.linspace(0, t, n_steps + 1)
        S = s0 * np.exp((mu - 0.5 * sigma**2) * time_points + sigma * W)
        return time_points.tolist(), S.tolist()

    def check_arbit_opportunity(self, time_step: int, amm: type[AMMi | Uniswap_amm], current_block: List[dict], external_price: float, fees: float, safe_profit_margin: float, swapper: Swap = None) -> List[dict]:
        """Check if there is a profit-making opportunity through arbitrage.

        Check for any arbitrage opportunity by either buying or selling A. 
        If potential profits > safe profit margin, then place transaction.

        Example: 
        If A is cheaper on AMM than external market, the arbitrageur will first
        check if the potential profits generated by buying A on AMM and selling
        external market is greater than the safe profit margin. If it is,
        on the trade is placed and added to the block.

        Args:
            time-step (int): Time-step when transaction was placed.
            amm (obj): The AMM instance to check the balance in.
                Class object of either AMMi (for XRPL AMM) or Uniswap_amm classes.
            current_block (list): Block containing all transactions to be processed.
            external_price (float): Current external market price of asset A.
            fees (float): AMM network fees.
            safe_profit_margin (float): Arbitrageur's minimum profit margin accepted.
            swapper (obj; Optional: None): The arbitrageur's swap object to perform swaps.
                Class object of the Swap class (for XRPL AMM).
                This is necessary only because of the distinct coding approaches used for the XRPL AMM and Uniswap classes.

        Returns:
            list: Updated current_block list.
        """

        discounted_fee = True if isinstance(amm, AMMi) and amm.AuctionSlot[
            'slot_owner'] == swapper.user else False
        swapper = swapper or amm

        # sell A
        if amm.spot_price('B', 'A', discounted_fee) < 1/external_price:
            amount_in, amount_out = swapper.swap_given_postSP(assetIn='A', assetOut='B', balAssetIn=self.get_balAsset(amm, 'A'), balAssetOut=self.get_balAsset(
                amm, 'B'), pre_sp=amm.spot_price('B', 'A', discounted_fee), post_sp=1/external_price, skip_pool_update=True)[0:2]
            potential_profits = (
                amount_out / external_price - amount_in) - fees*amm.spot_price('B', 'A')  # in units of A
            if potential_profits/amount_in > safe_profit_margin:
                current_block.append({'time_step': time_step,
                                      'external_market_price': 1/external_price,
                                      'amm_price': amm.spot_price('B', 'A'),
                                      'tx_type': 'sell',
                                      'amount_in': amount_in,
                                      'process_tx': lambda post_sp, amount_in, skip_pool_update=False:
                                      swapper.swap_given_postSP(assetIn='A', assetOut='B', balAssetIn=self.get_balAsset(amm, 'A'), balAssetOut=self.get_balAsset(amm, 'B'),
                                                                pre_sp=amm.spot_price('B', 'A'), post_sp=post_sp, amount_in=amount_in, skip_pool_update=skip_pool_update)})
        # buy A
        if amm.spot_price('A', 'B', discounted_fee) < external_price:
            amount_in, amount_out = swapper.swap_given_postSP(assetIn='B', assetOut='A', balAssetIn=self.get_balAsset(amm, 'B'), balAssetOut=self.get_balAsset(
                amm, 'A'), pre_sp=amm.spot_price('A', 'B', discounted_fee), post_sp=external_price, skip_pool_update=True)[0:2]
            potential_profits = (
                amount_out * external_price - amount_in) - fees
            if amount_in and potential_profits/amount_in > safe_profit_margin:
                current_block.append({'time_step': time_step,
                                      'external_market_price': external_price,
                                      'amm_price': amm.spot_price('A', 'B'),
                                      'tx_type': 'buy',
                                      'amount_in': amount_in,
                                      'process_tx': lambda post_sp, amount_in, skip_pool_update=False:
                                      swapper.swap_given_postSP(assetIn='B', assetOut='A', balAssetIn=self.get_balAsset(amm, 'B'), balAssetOut=self.get_balAsset(amm, 'A'),
                                                                pre_sp=amm.spot_price('A', 'B', discounted_fee), post_sp=post_sp, amount_in=amount_in, skip_pool_update=skip_pool_update)})

        return current_block


    def check_bid(self, time_step: int, slot_time_interval: float, amm: AMMi, current_block: List[dict], bidder: AMMBid, worst_case=False) -> List[dict]:
        """Check if bidding would be profitable and if yes, place a bid.

        Args:
            time-step (int): Time-step when transaction was placed.
            slot_time_interval (float): One of the 20 intervals (from 0.05 to 1).
                Interval when bid is placed.
            amm (obj): The XRPL AMM instance with CAM.
                Class object of AMMi.
            current_block (list): Block containing all transactions to be processed.
            bidder (obj): The arbitrageur's bid object to perform bids.
                Class object of the AMMBid class.
            worst_case (float): True if worst case for LPs.
                (Arbitrageurs bidding once a day and holding the slot for the full 24h).

        Returns:
            list: Updated current_block list.
        """
        minBidPrice = self.compute_minimum_bid_price(amm, slot_time_interval)

        # if worst_case and slot_time_interval == 0.05 and not amm.AuctionSlot['slot_owner']:
        #     current_block.append({
        #         'time_step': time_step,
        #         'tx_type': 'bid',
        #         'process_tx': lambda skip_pool_update: bidder.bid(
        #             slot_time_interval, min_price=minBidPrice, max_price=None, skip_pool_update=skip_pool_update)})
        #     return current_block

        #                                       -------------------- KEEP FOR WORSE 2ND CASE --------------------
        if worst_case and (slot_time_interval == 0.5 or not amm.AuctionSlot['slot_owner']):
            current_block.append({
                'time_step': time_step,
                'tx_type': 'bid',
                'process_tx': lambda skip_pool_update: bidder.bid(
                    slot_time_interval, min_price=minBidPrice, max_price=None, skip_pool_update=skip_pool_update)})
            return current_block

        elif not worst_case:
            pass
            #TODO: implement twma strategy

        return current_block

    def bidder_utility_function(self):
        pass

    def run_simulation(self, initial_A_price: float, external_prices: list, xrpl_block_conf: int, eth_block_conf: int, xrpl_fees: float,
                       eth_fees: float, normal_users: int, arbitrageurs: int, safe_profit_margin: float, max_slippage: float, iterations: int):
        """Simulate XRPL & Uniswap AMMs with a reference market.

        Args:
            initial_A_price (float): AMM Asset A price we wish the simulation to begin with.
                Should be equal to the initial external market price.
            external_prices (list): Number of normal user transactions to simulate.
            xrpl_block_conf (int): XRPL block confirmation time.
            eth_block_conf (int): Ethereum block confirmation time.
            xrpl_fees (float): XRPL network fees.
            eth_fees (float): Ethereum network fees.
            normal_users (int): Number of normal users to simulate.
            arbitrageurs (int): Number of arbitrageurs to simulate.
            safe_profit_margin (float): Arbitrageur's minimum profit margin accepted.
            max_slippage (float): The maximum slippage tolerated.
            iterations (int): Number of times to run the simulation with the same parameters.
                Each iteration is independent of the previous one.

        Returns:
            dict: Check return statement.
            TODO: update docs
        """

        initial_A_reserve = 1000
        initial_B_reserve = self.compute_initial_B_reserve(
            initial_A_price, 0.005)

        xrplCAM_profits_total, xrpl_profits_total, uniswap_profits_total = [], [], []
        xrplCAM_arbit_txs_total, xrpl_arbit_txs_total, uniswap_arbit_txs_total = [], [], []

        xrpl_ArbProfits_advantage, xrplCAM_ArbProfits_advantage, xrpls_CAM_ArbProfits_advantage = 0, 0, 0
        xrplCAM_PriceGap_advantage, xrpl_PriceGap_advantage, xrpls_CAM_PriceGap_advantage = 0, 0, 0
        xrplCAM_LP_returns_advantage, xrpl_LP_returns_advantage, xrpls_CAM_LP_returns_advantage = 0, 0, 0

        xrplCAM_slippages_total, xrpl_slippages_total, uniswap_slippages_total = [], [], []

        xrplCAM_unrealized_tx_total, xrpl_unrealized_tx_total, uniswap_unrealized_tx_total = [], [], []

        auction_slot_price_total = []

        xrpl_sps_total, xrplCAM_sps_total, uniswap_sps_total = [], [], []

        xrplCAM_tfees_total, xrpl_tfees_total, uniswap_tfees_total = [], [], []

        slot_holders_txs = []
        bids_profit_total = []

        xrplCAM_trading_volumes, xrpl_trading_volumes, uniswap_trading_volumes = [], [], []

        impermanent_losses = {'xrplCAM': [], 'xrpl': [], 'uniswap': []}

        bids_refunds_total = []

        for iteration in range(iterations):
            '''
            At each iteration, we set the users and AMMs statuses to their initial one and re-run the simulation.
            Each iteration is independent of the previous one. 
            '''
            one_day = 1000
            one_interval_duration = one_day/20
            # one_day = one_interval_duration*20
            slot_time_intervals = [round(n, 2) for n in np.linspace(
                0.05, 1, 20) for _ in range(int(one_interval_duration))]

            # bob = normal user on xrpl
            bob = User(user_name='bob', assets={
                       'XRP': 1000, 'A': 1e450, 'B': 1e450})
            bobCAM = User(user_name='bobCAM', assets={
                          'XRP': 1000, 'A': 1e450, 'B': 1e450})
            # arbit = arbitrageur on xrpl
            xrpl_amm = bob.createAMM(
                ammID=1, asset1='A', asset2='B', amount1=initial_A_reserve, amount2=initial_B_reserve)

            xrpl_arbits_swaps_obj = self.populate_xrpl_arbs(
                arbitrageurs, xrpl_amm, is_xrplCAM=False)

            # arbit = arbitrageur with no trading fee on xrpl
            xrplCAM = bobCAM.createAMM(ammID=2, asset1='A', asset2='B', amount1=initial_A_reserve/(
                arbitrageurs+1), amount2=initial_B_reserve/(arbitrageurs+1))

            xrplCAM_arbits_deposits_obj, xrplCAM_arbits_swaps_obj, xrplCAM_arbits_bids_obj = self.populate_xrpl_arbs(
                arbitrageurs, xrplCAM, is_xrplCAM=True)

            for arbit in xrplCAM_arbits_deposits_obj:
                Deposit(arbit.user, xrplCAM).deposit_Amount1_Amount2(
                    'A', 'B', initial_A_reserve/(arbitrageurs+1), initial_B_reserve/(arbitrageurs+1))

            # initiate both bob and arbit swap objects on xrpl in order to let them make swaps
            bob_swaps = Swap(bob, xrpl_amm)
            bobCAM_swaps = Swap(bobCAM, xrplCAM)

            # uniswap AMM
            uniswap_amm = Uniswap_amm(fee_rate=0.005, asset_A_amount=initial_A_reserve,
                                      asset_B_amount=initial_B_reserve, initial_LP_token_number=1000)

            xrplCAM_profits, xrpl_profits, uniswap_profits = [], [], []
            current_xrplCAM_block, current_xrpl_block, current_uniswap_block = [], [], []

            # number of transactions made by the arbitrageur on the AMMs (placed and successfully executed)
            xrpl_arbit_txs, xrplCAM_arbit_txs, uniswap_arbit_txs = 0, 0, 0

            xrplCAM_sps, xrpl_sps, uniswap_sps = [], [], []
            xrplCAM_sps_B, xrpl_sps_B, uniswap_sps_B = [], [], []

            prev_action = random.choice(['buy', 'sell'])

            xrplCAM_slippages, xrpl_slippages, uniswap_slippages = [], [], []
            xrplCAM_unrealized_tx, xrpl_unrealized_tx, uniswap_unrealized_tx = 0, 0, 0

            auction_slot_price = []

            xrplCAM_tfees, xrpl_tfees, uniswap_tfees = 0, 0, 0

            bids_profit = {'LPTokens': 0, 'B': 0}
            current_slot_holders_txs = 0

            xrplCAM_current_trading_volume, xrpl_current_trading_volume, uniswap_current_trading_volume = 0, 0, 0

            bids_refunds = {'LPTokens': 0, 'A': 0, 'B': 0}

            count_interval_1, new_24h_start = 0, 0
            iter = itertools.cycle(slot_time_intervals)
            for time_step in range(len(external_prices)):
                if not xrplCAM.AuctionSlot['slot_owner']:
                    iter = itertools.cycle(slot_time_intervals)
                    current_slot_time_interval = next(iter)
                else:
                    current_slot_time_interval = next(iter)

                # UPDATE XRPL POOLs
                if time_step % xrpl_block_conf == 0:
                    xrplCAM_sp, xrplCAM_sp_B, xrplCAM_profits, xrplCAM_arbit_txs, current_xrplCAM_block, xrplCAM_slippages, xrplCAM_unrealized_tx, xrplCAM_tfees, xrplCAM_current_trading_volume, bid_won, auction_slot_price, current_slot_holders_txs, bids_profit, bids_refunds = self.process_all_txs(
                        time_step, xrplCAM, current_xrplCAM_block, max_slippage, external_prices, xrpl_fees, xrplCAM_profits, xrplCAM_arbit_txs, xrplCAM_slippages, xrplCAM_unrealized_tx, xrplCAM_tfees, xrplCAM_current_trading_volume, auction_slot_price, current_slot_holders_txs, bids_profit, bids_refunds)
                    if bid_won:
                        iter = itertools.cycle(slot_time_intervals)
                        current_slot_time_interval = next(iter)
                        new_24h_start = time_step
                    elif not bid_won and current_slot_time_interval == 1 and time_step == new_24h_start + one_day:
                        iter = itertools.cycle(slot_time_intervals)
                        current_slot_time_interval = next(iter)
                        xrplCAM.AuctionSlot['slot_owner'] = None

                    xrpl_sp, xrpl_sp_B, xrpl_profits, xrpl_arbit_txs, current_xrpl_block, xrpl_slippages, xrpl_unrealized_tx, xrpl_tfees, xrpl_current_trading_volume = self.process_all_txs(
                        time_step, xrpl_amm, current_xrpl_block, max_slippage, external_prices, xrpl_fees, xrpl_profits, xrpl_arbit_txs, xrpl_slippages, xrpl_unrealized_tx, xrpl_tfees, xrpl_current_trading_volume)[0:9]
                # UPDATE UNISWAP POOL
                if time_step % eth_block_conf == 0:
                    uniswap_sp, uniswap_sp_B, uniswap_profits, uniswap_arbit_txs, current_uniswap_block, uniswap_slippages, uniswap_unrealized_tx, uniswap_tfees, uniswap_current_trading_volume = self.process_all_txs(
                        time_step, uniswap_amm, current_uniswap_block, max_slippage, external_prices, eth_fees, uniswap_profits, uniswap_arbit_txs, uniswap_slippages, uniswap_unrealized_tx, uniswap_tfees, uniswap_current_trading_volume)[0:9]
                # keep track of the price evolutions on xrpl-amm and uniswap
                xrpl_sps.append(xrpl_sp)
                xrpl_sps_B.append(xrpl_sp_B)

                xrplCAM_sps.append(xrplCAM_sp)
                xrplCAM_sps_B.append(xrplCAM_sp_B)

                uniswap_sps.append(uniswap_sp)
                uniswap_sps_B.append(uniswap_sp_B)

                # NORMAL USERS TRANSACTIONS
                # simulate normal users placing transactions
                current_xrplCAM_block, current_xrpl_block, current_uniswap_block = self.simulate_normal_users_txs(time_step, normal_users, prev_action, current_xrplCAM_block, current_xrpl_block, current_uniswap_block,
                                                                                                                  bobCAM_swaps, bob_swaps, uniswap_amm, xrpl_sp, xrplCAM_sp, uniswap_sp, uniswap_sp_B)

                # ARBITRAGE TRANSACTIONS
                for arb in range(arbitrageurs):
                    current_xrplCAM_block = self.check_bid(time_step, current_slot_time_interval, xrplCAM, current_xrplCAM_block, xrplCAM_arbits_bids_obj[arb], worst_case=True)
                    current_xrplCAM_block = self.check_arbit_opportunity(
                        time_step, xrplCAM, current_xrplCAM_block, external_prices[time_step], xrpl_fees, safe_profit_margin, swapper=xrplCAM_arbits_swaps_obj[arb])
                    current_xrpl_block = self.check_arbit_opportunity(
                        time_step, xrpl_amm, current_xrpl_block, external_prices[time_step], xrpl_fees, safe_profit_margin, swapper=xrpl_arbits_swaps_obj[arb])
                    current_uniswap_block = self.check_arbit_opportunity(
                        time_step, uniswap_amm, current_uniswap_block, external_prices[time_step], eth_fees, safe_profit_margin)

            bids_profit['B_adjusted'] = bids_profit['LPTokens'] * self.LPT_price(
                xrplCAM.assets['LPTokens'], xrplCAM.assets['A'], xrplCAM.assets['B'], xrplCAM.spot_price('A', 'B'), 1)


            p = external_prices[-1]/external_prices[0] - 1
            impermanent_losses['xrplCAM'].append(self.compute_impermanent_loss(
                initial_A_reserve, initial_B_reserve, xrplCAM.assets['A'], xrplCAM.assets['B'], xrplCAM_sp, external_prices[-1], xrplCAM_sps[0], p, bids_profit['B_adjusted']))
            impermanent_losses['xrpl'].append(self.compute_impermanent_loss(
                initial_A_reserve, initial_B_reserve, xrpl_amm.assets['A'], xrpl_amm.assets['B'], xrpl_sp, external_prices[-1], xrpl_sps[0], p))
            impermanent_losses['uniswap'].append(self.compute_impermanent_loss(
                initial_A_reserve, initial_B_reserve, uniswap_amm.asset_A_amount, uniswap_amm.asset_B_amount, uniswap_sp, external_prices[-1], uniswap_sps[0], p))



            xrpl_profits_total.append(xrpl_profits)
            xrplCAM_profits_total.append(xrplCAM_profits)
            uniswap_profits_total.append(uniswap_profits)
            xrpl_arbit_txs_total.append(xrpl_arbit_txs)
            xrplCAM_arbit_txs_total.append(xrplCAM_arbit_txs)
            uniswap_arbit_txs_total.append(uniswap_arbit_txs)

            xrplCAM_slippages_total.append(xrplCAM_slippages)
            xrpl_slippages_total.append(xrpl_slippages)
            uniswap_slippages_total.append(uniswap_slippages)

            xrplCAM_unrealized_tx_total.append(xrplCAM_unrealized_tx)
            xrpl_unrealized_tx_total.append(xrpl_unrealized_tx)
            uniswap_unrealized_tx_total.append(uniswap_unrealized_tx)

            auction_slot_price_total.append(auction_slot_price)

            xrpl_sps_total.append([iteration, xrpl_sps])
            xrplCAM_sps_total.append([iteration, xrplCAM_sps])
            uniswap_sps_total.append([iteration, uniswap_sps])

            bids_profit_total.append(bids_profit)

            slot_holders_txs.append(current_slot_holders_txs)

            xrplCAM_trading_volumes.append(xrplCAM_current_trading_volume)
            xrpl_trading_volumes.append(xrpl_current_trading_volume)
            uniswap_trading_volumes.append(uniswap_current_trading_volume)

            xrplCAM_tfees_total.append(xrplCAM_tfees)
            xrpl_tfees_total.append(xrpl_tfees)
            uniswap_tfees_total.append(uniswap_tfees)

            bids_refunds_total.append(bids_refunds)

            xrplCAM_ArbProfits_advantage = self.compute_ArbitrageursProfits_advantage(
                xrplCAM_profits_total[-1], uniswap_profits_total[-1], xrplCAM_ArbProfits_advantage)
            xrpl_ArbProfits_advantage = self.compute_ArbitrageursProfits_advantage(
                xrpl_profits_total[-1], uniswap_profits_total[-1], xrpl_ArbProfits_advantage)
            xrpls_CAM_ArbProfits_advantage = self.compute_ArbitrageursProfits_advantage(
                xrplCAM_profits_total[-1], xrpl_profits_total[-1], xrpls_CAM_ArbProfits_advantage)
            xrplCAM_PriceGap_advantage = self.compute_PriceGap_advantage(
                external_prices, xrplCAM_sps, uniswap_sps, xrplCAM_PriceGap_advantage)
            xrpl_PriceGap_advantage = self.compute_PriceGap_advantage(
                external_prices, xrpl_sps, uniswap_sps, xrpl_PriceGap_advantage)
            xrpls_CAM_PriceGap_advantage = self.compute_PriceGap_advantage(
                external_prices, xrplCAM_sps, xrpl_sps, xrpls_CAM_PriceGap_advantage)

            xrplCAM_LP_returns_advantage = self.compute_LP_returns_advantage(
                xrplCAM_tfees+bids_profit['B_adjusted'], uniswap_tfees, xrplCAM_LP_returns_advantage)
            xrpl_LP_returns_advantage = self.compute_LP_returns_advantage(
                xrpl_tfees, uniswap_tfees, xrpl_LP_returns_advantage)
            xrpls_CAM_LP_returns_advantage = self.compute_LP_returns_advantage(
                xrplCAM_tfees+bids_profit['B_adjusted'], xrpl_tfees, xrpls_CAM_LP_returns_advantage)

  
        return {'xrplCAM_profits_total': xrplCAM_profits_total, 'xrpl_profits_total': xrpl_profits_total, 'uniswap_profits_total': uniswap_profits_total,
                'xrplCAM_arbit_txs_total': xrplCAM_arbit_txs_total, 'xrpl_arbit_txs_total': xrpl_arbit_txs_total, 'uniswap_arbit_txs_total': uniswap_arbit_txs_total,
                'xrplCAM_sps_total': xrplCAM_sps_total, 'xrpl_sps_total': xrpl_sps_total, 'uniswap_sps_total': uniswap_sps_total,
                'xrplCAM_slippages_total': xrplCAM_slippages_total, 'xrpl_slippages_total': xrpl_slippages_total, 'uniswap_slippages': uniswap_slippages_total,
                'xrplCAM_unrealized_tx_total': xrplCAM_unrealized_tx_total, 'xrpl_unrealized_tx_total': xrpl_unrealized_tx_total, 'uniswap_unrealized_tx_total': uniswap_unrealized_tx_total,
                'xrplCAM_tfees_total': xrplCAM_tfees_total, 'xrpl_tfees_total': xrpl_tfees_total, 'uniswap_tfees_total': uniswap_tfees_total,
                'xrplCAM_trading_volumes': xrplCAM_trading_volumes, 'xrpl_trading_volumes': xrpl_trading_volumes, 'uniswap_trading_volumes': uniswap_trading_volumes,
                'auction_slot_price_total': auction_slot_price_total, 'bids_profit_total': bids_profit_total,
                'impermanent_losses': impermanent_losses,
                'xrplCAM_ArbProfits_advantage': round(xrplCAM_ArbProfits_advantage/iterations*100), 'xrpl_ArbProfits_advantage': round(xrpl_ArbProfits_advantage/iterations*100), 'xrpls_CAM_ArbProfits_advantage': round(xrpls_CAM_ArbProfits_advantage/iterations*100),
                'xrplCAM_PriceGap_advantage': round(xrplCAM_PriceGap_advantage/iterations*100), 'xrpl_PriceGap_advantage': round(xrpl_PriceGap_advantage/iterations*100), 'xrpls_CAM_PriceGap_advantage': round(xrpls_CAM_PriceGap_advantage/iterations*100),
                'xrplCAM_LP_returns_advantage': round(xrplCAM_LP_returns_advantage/iterations*100), 'xrpl_LP_returns_advantage': round(xrpl_LP_returns_advantage/iterations*100), 'xrpls_CAM_LP_returns_advantage': round(xrpls_CAM_LP_returns_advantage/iterations*100),
                'bids_refunds_total': bids_refunds_total,
                'slot_holders_txs': slot_holders_txs}
