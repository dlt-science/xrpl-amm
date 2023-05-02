import random
from amms.xrpl.amm.actions import Swap
from amms.xrpl.amm.env import User
from amms.uniswap.uniswap import Uniswap_amm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Simulator(Swap, User, Uniswap_amm):
    def __init__(self):
        pass

    '''
    user_action(...):
        There's an 80% chance that a normal user performs an action (buy or sell).
        Out of the 80%, if the previous user bought, the current user has a 60% probability 
        of buying and 40% probability of selling (and vice versa if the previous user sold)
    '''

    def user_action(self, prev_action):
        # probability that the user performs an action (either buy or sell)
        buy_sell_prob = 0.8
        # probability that the user doesn't perform any action
        pass_prob = 1 - buy_sell_prob
        action = random.choices(['buy', 'sell', 'pass'], [
                                buy_sell_prob, buy_sell_prob, pass_prob])[0]
        if action in ['buy', 'sell']:
            if prev_action == 'buy':
                buy_prob = 0.6
                sell_prob = 0.4
                action = random.choices(
                    [action], [buy_prob if action == 'buy' else sell_prob])[0]
            if prev_action == 'sell':
                buy_prob = 0.4
                sell_prob = 0.6
                action = random.choices(
                    [action], [buy_prob if action == 'buy' else sell_prob])[0]
        return action

    '''
    generate_price_data(...):
        Generate price data with an initial price of 1000.
        min_vol and max_vol are the min and max volatilities
        (price change) between each consecutive value.
        n = number of data points (each price is a second)
    '''

    def generate_price_data(self, n, min_vol, max_vol):
        prices = [1000]  # set starting price to 1000
        for i in range(n-1):
            vol = random.uniform(min_vol, max_vol) * random.choice([-1, 1])
            prices += [prices[i]*(1+vol)]
        return prices

    '''
    check_slippage(...):
        returns True if the slippage from the trade is smaller than the maximum accepted slippage
    '''

    def check_slippage(self, max_slippage, pre_sp, amountIn, final_amountOut):
        effective_price = amountIn/final_amountOut
        slippage = (effective_price / pre_sp) - 1
        return abs(slippage) <= max_slippage

    '''
    process_txs(...):
        Execute all transactions after shuffling their order in the block;
        If arbitrage transaction, first check for the slippage condition.
        If normal user transaction, no check necessary, execute the trade directly.
    '''

    def process_txs(self, i, amm, current_block, spot_prices, spot_prices_B, max_slippage, external_prices, fees, profits, arbit_txs, no_fee=False):
        if len(current_block) > 0:
            # shuffle the transactions in the current block to randomize their order of execution
            random.shuffle(current_block)
            for tx in current_block:
                # arbitrageur tx
                if tx[1] == 'buy':
                    amountIn, final_amountOut = tx[3](
                        post_sp=external_prices[tx[0]], amountIn=tx[2], skip_pool_update=True)
                    if self.check_slippage(max_slippage, spot_prices[tx[0]], tx[2], final_amountOut):
                        tx[3](post_sp=external_prices[tx[0]], amountIn=tx[2])
                        profit = final_amountOut * \
                            external_prices[i] - amountIn - fees
                        profits += [profit]
                        arbit_txs += 1

                # arbitrageur tx
                elif tx[1] == 'sell':
                    amountIn, final_amountOut = tx[3](
                        post_sp=1/external_prices[tx[0]], amountIn=tx[2], skip_pool_update=True)
                    if self.check_slippage(max_slippage, spot_prices_B[tx[0]], tx[2], final_amountOut):
                        tx[3](post_sp=1/external_prices[tx[0]], amountIn=tx[2])
                        profit = (
                            final_amountOut / external_prices[i] - amountIn)*external_prices[i] - fees
                        profits += [profit]
                        arbit_txs += 1

                # normal user tx
                else:
                    tx[1]()

        current_block = []
        sp = amm.check_SP_price('A', 'B', no_fee)
        sp_B = amm.check_SP_price('B', 'A', no_fee)
        return sp, sp_B, profits, arbit_txs, current_block

    ''' 
    get_balAsset(...):
        get the balance of asset A or B in AMM 
    '''

    def get_balAsset(self, amm, asset):
        if isinstance(amm, Uniswap_amm):
            balAsset = amm.asset_A_amount if asset == 'A' else amm.asset_B_amount
        else:
            balAsset = amm.assets[asset]
        return balAsset

    ''' 
    check_arbit_opportunity(...):
        checks for any arbitrage opportunity by either buying or selling A. 
        if potential profits > safe profit margin, then place transaction.
        Example: if A is cheaper on AMM than external market, the arbitrageur will first
                check if the potential profits generated by buying A on AMM and selling on
                external market is greater than the safe profit margin. If it is,
                the trade is placed and added to the block.
    '''

    def check_arbit_opportunity(self, i, amm, current_block, spot_price, spot_price_B, external_prices, fees, safe_profit_margin, swapper=None, no_fee=False):
        swapper = swapper or amm
        # SELL A
        if spot_price_B < 1/external_prices[i]:
            amountIn, amountOut = swapper.swap_given_postSP(assetIn='A', assetOut='B', balAssetIn=self.get_balAsset(
                amm, 'A'), balAssetOut=self.get_balAsset(amm, 'B'), pre_sp=spot_price_B, post_sp=1/external_prices[i], skip_pool_update=True)

            # potential_profits = ((amountOut / external_prices[i]) - amountIn)*external_prices[i] - fees # in B
            potential_profits = (
                amountOut / external_prices[i] - amountIn) - fees*spot_price_B  # in A
            if potential_profits/amountIn > safe_profit_margin:
                current_block += [[i, 'sell', amountIn, lambda post_sp, amountIn, skip_pool_update=False: swapper.swap_given_postSP(assetIn='A', assetOut='B', balAssetIn=self.get_balAsset(amm, 'A'), balAssetOut=self.get_balAsset(amm, 'B'),
                                                                                                                                    pre_sp=amm.check_SP_price('B', 'A', no_fee), post_sp=post_sp, amountIn=amountIn, skip_pool_update=skip_pool_update)]]
        # BUY A
        if spot_price < external_prices[i]:
            amountIn, amountOut = swapper.swap_given_postSP(assetIn='B', assetOut='A', balAssetIn=self.get_balAsset(
                amm, 'B'), balAssetOut=self.get_balAsset(amm, 'A'), pre_sp=spot_price, post_sp=external_prices[i], skip_pool_update=True)

            potential_profits = (
                amountOut * external_prices[i] - fees) - amountIn
            if amountIn and potential_profits/amountIn > safe_profit_margin:
                current_block += [[i, 'buy', amountIn, lambda post_sp, amountIn, skip_pool_update=False: swapper.swap_given_postSP(assetIn='B', assetOut='A', balAssetIn=self.get_balAsset(amm, 'B'), balAssetOut=self.get_balAsset(amm, 'A'),
                                                                                                                                   pre_sp=amm.check_SP_price('A', 'B', no_fee), post_sp=post_sp, amountIn=amountIn, skip_pool_update=skip_pool_update)]]
        return current_block

    '''
    add_normal_users_txs(...):
        add trades for normal users
    '''

    def add_normal_users_txs(self, i, n_users, prev_action, current_xrpl_block_0fee, current_xrpl_block, current_uniswap_block, xrpl0fee_swapper, xrpl_swapper, uniswap_amm, xrpl_sp, xrpl_sp_0fee, uniswap_sp, uniswap_sp_B):
        for user in range(n_users):
            action = self.user_action(prev_action)
            if action == 'buy':
                # user will buy between 0.1 and 5 ETH
                amount = random.uniform(0.1, 5)
                current_xrpl_block += [[i, lambda: xrpl_swapper.swap_given_amount_Out(
                    assetIn='B', assetOut='A', amountOut=amount)]]
                current_xrpl_block_0fee += [[i, lambda: xrpl0fee_swapper.swap_given_amount_Out(
                    assetIn='B', assetOut='A', amountOut=amount)]]
                current_uniswap_block += [[i,
                                           lambda: uniswap_amm.swap('A', amount, uniswap_sp)]]
                prev_action = 'buy'
            elif action == 'sell':
                # user will sell between 0.1 and 5 ETH
                amount = random.uniform(0.1, 5)
                current_xrpl_block += [[i, lambda: xrpl_swapper.swap_given_amount_Out(
                    assetIn='A', assetOut='B', amountOut=amount*xrpl_sp)]]
                current_xrpl_block_0fee += [[i, lambda: xrpl0fee_swapper.swap_given_amount_Out(
                    assetIn='A', assetOut='B', amountOut=amount*xrpl_sp_0fee)]]
                current_uniswap_block += [
                    [i, lambda: uniswap_amm.swap('B', amount*uniswap_sp, uniswap_sp_B)]]
                prev_action = 'sell'
        return current_xrpl_block_0fee, current_xrpl_block, current_uniswap_block

    '''
    run_simulaton(...):
        run the simulation to compare arbitrageur profits on XRPL (both with 0% trading fee and positive fee of 0.5%) and Uniswap (0.5% trading fee)
        at each second, the arbitrageur will look for trade opportunities and at each block confirmation, transactions get executed randomly
    '''

    def run_simulaton(self, external_prices, xrpl_block_conf, eth_block_conf, xrpl_fees, eth_fees, safe_profit_margin, max_slippage, iterations):
        xrpl_profits_total, xrpl_arbit_txs_total = [], []
        xrpl0fee_profits_total, xrpl_arbit0fee_txs_total = [], []
        uniswap_profits_total, uniswap_arbit_txs_total = [], []
        xrpl_advantage = 0

        for iteration in range(iterations):
            '''
            At each iteration, we set the users and AMMs statuses to their initial one and re-run the simulation.
            Each iteration is independent of the previous one.
            '''
            # bob = normal user on xrpl
            bob = User(user_name='bob', assets={
                       'XRP': 1000, 'A': 1e450, 'B': 1e450})
            bob0fee = User(user_name='bob0fee', assets={
                           'XRP': 1000, 'A': 1e450, 'B': 1e450})
            # arbit = arbitrageur on xrpl
            arbit = User(user_name='arbit', assets={
                         'XRP': 1000, 'A': 1e450, 'B': 1e450})
            # arbit = arbitrageur with no trading fee on xrpl
            arbit0fee = User(user_name='arbit0fee', assets={
                             'XRP': 1000, 'A': 1e450, 'B': 1e450})
            # xrpl AMM
            xrpl_amm = bob.createAMM(
                ammID=1, asset1='A', asset2='B', amount1=1e4, amount2=995e4)
            xrpl_amm_0fee = bob0fee.createAMM(
                ammID=2, asset1='A', asset2='B', amount1=1e4, amount2=995e4)
            # initiate both bob and arbit swap objects on xrpl in order to let them make swaps
            bob_swaps = Swap(bob, xrpl_amm)
            bob_swaps_0fee = Swap(bob0fee, xrpl_amm_0fee)
            arbit_swaps = Swap(arbit, xrpl_amm)
            arbit0fee_swaps = Swap(arbit0fee, xrpl_amm_0fee)
            # set arbitrageur trading fee to 0
            arbit0fee_swaps.TFee = 0
            # uniswap AMM
            uniswap_amm = Uniswap_amm(
                fee_rate=0.005, asset_A_amount=1e4, asset_B_amount=995e4, initial_LP_token_number=1e4)

            xrpl_profits, current_xrpl_block = [], []
            xrpl_profits_0fee, current_xrpl_block_0fee = [], []
            uniswap_profits, current_uniswap_block = [], []
            # number of transactions made by the arbitrageur on the AMMs (placed and successfully executed)
            xrpl_arbit_txs, xrpl_arbit0fee_txs, uniswap_arbit_txs = 0, 0, 0

            xrpl_sps, xrpl_sps_0fee = [], []
            xrpl_sps_B, xrpl_sps_B_0fee = [], []
            uniswap_sps, uniswap_sps_B = [], []
            prev_action = random.choice(['buy', 'sell'])
            for i in range(len(external_prices)):
                # UPDATE XRPL AND XRPL0FEE POOLS
                if i % xrpl_block_conf == 0:
                    # XRPL0FEE
                    xrpl_sp_0fee, xrpl_sp_B_0fee, xrpl_profits_0fee, xrpl_arbit0fee_txs, current_xrpl_block_0fee = self.process_txs(
                        i, xrpl_amm_0fee, current_xrpl_block_0fee, xrpl_sps_0fee, xrpl_sps_B_0fee, max_slippage, external_prices, xrpl_fees, xrpl_profits_0fee, xrpl_arbit0fee_txs, no_fee=True)
                    # XRPL
                    xrpl_sp, xrpl_sp_B, xrpl_profits, xrpl_arbit_txs, current_xrpl_block = self.process_txs(
                        i, xrpl_amm, current_xrpl_block, xrpl_sps, xrpl_sps_B, max_slippage, external_prices, xrpl_fees, xrpl_profits, xrpl_arbit_txs)
                # UPDATE UNISWAP POOL
                if i % eth_block_conf == 0:
                    uniswap_sp, uniswap_sp_B, uniswap_profits, uniswap_arbit_txs, current_uniswap_block = self.process_txs(
                        i, uniswap_amm, current_uniswap_block, uniswap_sps, uniswap_sps_B, max_slippage, external_prices, eth_fees, uniswap_profits, uniswap_arbit_txs)

                # keep track of the price evolutions on xrpl-amm and uniswap (last iteration only)
                # XRPL0FEE
                xrpl_sps_0fee += [xrpl_sp_0fee]
                xrpl_sps_B_0fee += [xrpl_sp_B_0fee]
                # XRPL
                xrpl_sps += [xrpl_sp]
                xrpl_sps_B += [xrpl_sp_B]
                # UNISWAP
                uniswap_sps += [uniswap_sp]
                uniswap_sps_B += [uniswap_sp_B]

                # NORMAL USERS TRANSACTIONS
                # simulate 50 normal users placing transactions
                current_xrpl_block_0fee, current_xrpl_block, current_uniswap_block = self.add_normal_users_txs(i, 50, prev_action, current_xrpl_block_0fee, current_xrpl_block, current_uniswap_block,
                                                                                                               bob_swaps_0fee, bob_swaps, uniswap_amm, xrpl_sp, xrpl_sp_0fee, uniswap_sp, uniswap_sp_B)

                # ARBITRAGE TRANSACTIONS
                # XRPL0FEE
                current_xrpl_block_0fee = self.check_arbit_opportunity(
                    i, xrpl_amm_0fee, current_xrpl_block_0fee, xrpl_sp_0fee, xrpl_sp_B_0fee, external_prices, xrpl_fees, safe_profit_margin, swapper=arbit0fee_swaps, no_fee=True)
                # XRPL
                current_xrpl_block = self.check_arbit_opportunity(
                    i, xrpl_amm, current_xrpl_block, xrpl_sp, xrpl_sp_B, external_prices, xrpl_fees, safe_profit_margin, swapper=arbit_swaps)
                # UNISWAP
                current_uniswap_block = self.check_arbit_opportunity(
                    i, uniswap_amm, current_uniswap_block, uniswap_sp, uniswap_sp_B, external_prices, eth_fees, safe_profit_margin)

            xrpl_profits_total += [xrpl_profits]
            xrpl0fee_profits_total += [xrpl_profits_0fee]
            uniswap_profits_total += [uniswap_profits]
            xrpl_arbit_txs_total += [xrpl_arbit_txs]
            xrpl_arbit0fee_txs_total += [xrpl_arbit0fee_txs]
            uniswap_arbit_txs_total += [uniswap_arbit_txs]

            xrpl_advantage += sum(xrpl_profits_total[iteration]) > sum(
                uniswap_profits_total[iteration])

        xrpl_advantage = round(xrpl_advantage/iterations*100)
        return [xrpl_profits_total, uniswap_profits_total, xrpl_arbit_txs_total, uniswap_arbit_txs_total, xrpl_sps, uniswap_sps, xrpl0fee_profits_total, xrpl_arbit0fee_txs_total, xrpl_sps_0fee, xrpl_advantage]

    '''
    display_results(...):
        display the simulation's results
    '''

    def display_results(self, sim, external_prices, iterations, xrpl_fees, eth_fees, safe_profit_margin, max_slippage):
        print('Arbitrageurs are more profitable on XRPL',
              f'{sim[-1]}%', 'of the time')

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)
              ) = plt.subplots(3, 3, figsize=(18, 14))
        fig.text(0.4, 0.92, s='XRPL fees: '+f'{xrpl_fees}'+', ETH fees: '+f'{eth_fees}'+', Safe profit margin: ' +
                 f'{safe_profit_margin}%'+',  Max. slippage: '+f'{max_slippage}%', ha='right', va='top')

        x_axis = [i+1 for i in range(iterations)]
        # --- plot 1 ---
        xrpl_profits_sum = [sum(sim[0][i]) for i in range(iterations)]
        xrpl0fee_profits_sum = [sum(sim[6][i]) for i in range(iterations)]
        uniswap_profits_sum = [sum(sim[1][i]) for i in range(iterations)]

        ax1.plot(x_axis, xrpl0fee_profits_sum, label='XRPL0fee-AMM')
        ax1.plot(x_axis, xrpl_profits_sum, label='XRPL-AMM')
        ax1.plot(x_axis, uniswap_profits_sum, label='Uniswap')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Profits (in asset B)')
        ax1.set_title('Arbitrageur profits')
        ax1.legend()
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # --- plot 2 ---
        avg_xrpl_profits = round(np.average(xrpl_profits_sum))
        avg_xrpl0fee_profits = round(np.average(xrpl0fee_profits_sum))
        avg_eth_profits = round(np.average(uniswap_profits_sum))

        x = ['XRPL0FEE', 'XRPL', 'Ethereum']
        y = [avg_xrpl0fee_profits, avg_xrpl_profits, avg_eth_profits]

        ax2.bar(x, y)
        ax2.text(x[0], y[0], avg_xrpl0fee_profits, ha='center', va='bottom')
        ax2.text(x[1], y[1], avg_xrpl_profits, ha='center', va='bottom')
        ax2.text(x[2], y[2], avg_eth_profits, ha='center', va='bottom')

        ax2.set_ylim(min(y)-max(y)/50, max(y)+max(y)/50)
        # diff = max(y) - min(y)
        # ax2.text(0.5, (max(y)+min(y))/2, f'Diff: {diff}', ha='center', va='center')
        ax2.set_ylabel('Profits (in asset B)')
        ax2.set_title('Average arbitrageur profits')
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # --- plot 3 ---
        ax3.plot(x_axis, sim[7], label='XRPL0fee-AMM')
        ax3.plot(x_axis, sim[2], label='XRPL-AMM')
        ax3.plot(x_axis, sim[3], label='Uniswap')

        ax3.set_xlabel('Iteration #')
        ax3.set_ylabel('Number of Transactions')
        ax3.set_title(
            'Number of Txs made by the arbitrageur at each iteration')
        ax3.legend()
        ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # --- plot 4 ---
        def compute_losses(revs):
            losses = []
            for rev in revs:
                loss = len([x for x in rev if x < 0])
                losses.append(loss)
            avg_losses = round(np.average(losses))
            return avg_losses

        avg_xrpl_txs = round(np.average(sim[2]))
        avg_xrpl0fee_txs = round(np.average(sim[7]))
        avg_eth_txs = round(np.average(sim[3]))

        x = ['XRPL0fee', 'XRPL', 'Ethereum']
        y = [avg_xrpl0fee_txs, avg_xrpl_txs, avg_eth_txs]

        ax4.bar(x, y)
        ax4.text(x[0], y[0], avg_xrpl0fee_txs, ha='center', va='bottom')
        ax4.text(
            x[0], y[0]-2, '(~'+f'{compute_losses(sim[6])} losses)', ha='center', va='top', fontsize=8)
        ax4.text(x[1], y[1], avg_xrpl_txs, ha='center', va='bottom')
        ax4.text(
            x[1], y[1]-2, '(~'+f'{compute_losses(sim[0])} losses)', ha='center', va='top', fontsize=8)
        ax4.text(x[2], y[2], avg_eth_txs, ha='center', va='bottom')
        ax4.text(
            x[2], y[2]-2, '(~'+f'{compute_losses(sim[1])} losses)', ha='center', va='top', fontsize=8)
        ax4.set_ylabel('# of Txs')
        ax4.set_title('Average total arbitrage Txs')
        ax4.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # --- plot 5 ---
        xrpl_fees_total = [i*xrpl_fees for i in sim[2]]
        xrpl0fee_fees_total = [i*xrpl_fees for i in sim[7]]
        uniswap_fees_total = [i*eth_fees for i in sim[3]]

        ax5.plot(x_axis, xrpl0fee_fees_total, label='XRPL0fee-AMM')
        ax5.plot(x_axis, xrpl_fees_total, label='XRPL-AMM')
        ax5.plot(x_axis, uniswap_fees_total, label='Uniswap')

        ax5.set_xlabel('Iteration #')
        ax5.set_ylabel('Cumulative Tx fees (USD)')
        ax5.set_title(
            'Total Tx fees paid by the arbitrageur at each iteration')
        ax5.legend()
        ax5.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # --- plot 6 ---
        avg_xrpl_fees = round(np.average(xrpl_fees_total), 4)
        avg_xrpl0fee_fees = round(np.average(xrpl0fee_fees_total), 4)
        avg_eth_fees = round(np.average(uniswap_fees_total), 4)

        avg_xrpl_txs = round(np.average(sim[2]))
        avg_xrpl0fee_txs = round(np.average(sim[7]))
        avg_eth_txs = round(np.average(sim[3]))

        x = ['XRPL fees for \n ~'+f'{avg_xrpl0fee_txs} txs', 'XRPL fees for \n ~' +
             f'{avg_xrpl_txs} txs', 'ETH fees for \n ~'+f'{avg_eth_txs} txs']
        y = [avg_xrpl0fee_fees, avg_xrpl_fees, avg_eth_fees]

        ax6.bar(x, y)
        ax6.text(x[0], y[0], avg_xrpl0fee_fees, ha='center', va='bottom')
        ax6.text(x[1], y[1], avg_xrpl_fees, ha='center', va='bottom')
        ax6.text(x[2], y[2], avg_eth_fees, ha='center', va='bottom')
        ax6.set_ylabel('Tx fees (USD)')
        ax6.set_title('Average total Tx fees')
        ax6.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # --- plot 7 ---
        ax7.plot(external_prices, label='External Market')
        ax7.plot(sim[8], label='XRPL0fee-AMM')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Price (asset A)')
        ax7.set_title('Price Evolution (last iteration only)')
        ax7.legend()

        # --- plot 7 ---
        ax8.plot(external_prices, label='External Market')
        ax8.plot(sim[4], label='XRPL-AMM')
        ax8.set_xlabel('Time')
        ax8.set_ylabel('Price (asset A)')
        ax8.set_title('Price Evolution (last iteration only)')
        ax8.legend()

        # --- plot 7 ---
        ax9.plot(external_prices, label='External Market')
        ax9.plot(sim[5], label='Uniswap')
        ax9.set_xlabel('Time')
        ax9.set_ylabel('Price (asset A)')
        ax9.set_title('Price Evolution (last iteration only)')
        ax9.legend()

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        plt.show()
