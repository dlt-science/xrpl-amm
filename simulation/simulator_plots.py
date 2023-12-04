import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import textwrap
import pickle


class SimulatorResults:
    def __init__(
        self,
        sim,
        assetA,
        assetB,
        external_prices,
        gbm_mu,
        gbm_sigma,
        xrpl_block_conf,
        eth_block_conf,
        xrpl_fees,
        eth_fees,
        safe_profit_margin,
        max_slippage,
        iterations,
        with_cases,
        equality_condition=None,
    ):
        self.sim = sim
        self.assetA, self.assetB = assetA, assetB
        self.external_prices = external_prices
        self.gbm_mu, self.gbm_sigma = gbm_mu, gbm_sigma
        self.xrpl_block_conf, self.eth_block_conf = xrpl_block_conf, eth_block_conf
        self.xrpl_fees, self.eth_fees = xrpl_fees, eth_fees
        self.safe_profit_margin = safe_profit_margin
        self.max_slippage = max_slippage
        self.iterations = iterations
        self.x_axis = [i + 1 for i in range(self.iterations)]
        self.with_cases = with_cases
        self.labels = (
            ["XRPL-CAM-A", "XRPL-CAM-B", "Uniswap"]
            if with_cases
            else ["XRPL-AMM-CAM", "XRPL-AMM", "Uniswap"]
        )
        self.equality_condition = equality_condition

    def create_txt_info(self):
        return textwrap.dedent(
            f"""\
                - External Price Data:
                    - Time Steps: {len(self.external_prices)}
                    - Drift/Expected Return (mu): {self.gbm_mu}
                    - Volatility (sigma): {self.gbm_sigma}

                - XRPL Block Time: {self.xrpl_block_conf}
                - Ethereum Block Time: {self.eth_block_conf}
                
                - XRPL Fees: {self.xrpl_fees} {self.assetB}
                - Ethereum Fees: {self.eth_fees} {self.assetB}
                
                - Safe Profit Margin: {self.safe_profit_margin}%
                - Max. Slippage: {self.max_slippage}%
          
                - Iterations: {self.iterations}

                Summary of Results:
                    • Arbitrageurs' Profitability:
                        - Arbitrageurs are more profitable on XRPL {self.sim["xrpl_ArbProfits_advantage"]}% of the time.
                        - Arbitrageurs are more profitable on XRPL-CAM {self.sim["xrplCAM_ArbProfits_advantage"]}% of the time.
                        - Arbitrageurs are more profitable on XRPL-CAM than XRPL {self.sim["xrpls_CAM_ArbProfits_advantage"]}% of the time.

                    • LPs Returns:
                        - LPs earn more on XRPL {self.sim["xrpl_LP_returns_advantage"]}% of the time.
                        - LPs earn more on XRPL-CAM {self.sim["xrplCAM_LP_returns_advantage"]}% of the time.
                        - LPs earn more on XRPL-CAM than XRPL {self.sim["xrpls_CAM_LP_returns_advantage"]}% of the time.
                    
                    • Price Sync:
                        - Price Sync. is better on XRPL {self.sim["xrpl_PriceGap_advantage"]}% of the time.
                        - Price Sync. is better on XRPL-CAM {self.sim["xrplCAM_PriceGap_advantage"]}% of the time.
                        - Price Sync. is better on XRPL-CAM than XRPL {self.sim["xrpls_CAM_PriceGap_advantage"]}% of the time."""
        )

    def save_simulation_data(self, foldername):
        folder_path = os.path.join("simulation_results", foldername, "data")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(
            os.path.join(os.path.join("simulation_results", foldername), "info.txt"),
            "w",
        ) as f:
            f.write(self.create_txt_info())

        filepaths = {
            "external_prices.pickle": self.external_prices,
            "simulation_results.pickle": self.sim,
        }

        for filename, data in filepaths.items():
            with open(os.path.join(folder_path, filename), "wb") as f:
                pickle.dump(data, f)

    def save_plot(self, ax, foldername, filename):
        folder_path = os.path.join("simulation_results", foldername, "plots")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(
            os.path.join(os.path.join("simulation_results", foldername), "info.txt"),
            "w",
        ) as f:
            f.write(self.create_txt_info())

        filepath = os.path.join(folder_path, filename)

        fig = ax.figure

        for ax_tmp in fig.axes:
            if ax_tmp != ax:
                ax_tmp.set_visible(False)

        text_elements = [
            artist for artist in fig.get_children() if isinstance(artist, plt.Text)
        ]
        for text in text_elements:
            text.set_visible(False)

        fig.savefig(filepath, bbox_inches="tight")

        for ax_tmp in fig.axes:
            ax_tmp.set_visible(True)
        for text in text_elements:
            text.set_visible(True)

    def plot_arbitrageurs_profits(self, ax, set_title=True):
        xrpl_profits_sum = [
            sum(self.sim["xrpl_profits_total"][i]) for i in range(self.iterations)
        ]
        xrplCAM_profits_sum = [
            sum(self.sim["xrplCAM_profits_total"][i]) for i in range(self.iterations)
        ]
        uniswap_profits_sum = [
            sum(self.sim["uniswap_profits_total"][i]) for i in range(self.iterations)
        ]

        ax.plot(self.x_axis, xrplCAM_profits_sum, label=self.labels[0])
        ax.plot(self.x_axis, xrpl_profits_sum, label=self.labels[1])
        ax.plot(self.x_axis, uniswap_profits_sum, label=self.labels[2])

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Profits ({})".format(self.assetB))
        ax.set_title("Arbitrageur profits") if set_title else None
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        return ax

    def plot_average_arbitrageurs_profits(self, ax, set_title=True):
        xrpl_profits_sum = [
            sum(self.sim["xrpl_profits_total"][i]) for i in range(self.iterations)
        ]
        xrplCAM_profits_sum = [
            sum(self.sim["xrplCAM_profits_total"][i]) for i in range(self.iterations)
        ]
        uniswap_profits_sum = [
            sum(self.sim["uniswap_profits_total"][i]) for i in range(self.iterations)
        ]

        avg_xrpl_profits = round(np.average(xrpl_profits_sum))
        avg_xrplCAM_profits = round(np.average(xrplCAM_profits_sum))
        avg_eth_profits = round(np.average(uniswap_profits_sum))

        x = self.labels
        y = [avg_xrplCAM_profits, avg_xrpl_profits, avg_eth_profits]

        ax.bar(x, y)

        ax.bar_label(ax.containers[0], fmt="{:,.0f}")

        ax.set_ylim(min(y) - max(y) / 50, max(y) + max(y) / 50)
        ax.set_ylabel("Profits ({})".format(self.assetB))
        ax.set_title("Average arbitrageur profits") if set_title else None
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        return ax

    def plot_number_of_txs(self, ax, set_title=True):
        ax.plot(self.x_axis, self.sim["xrplCAM_arbit_txs_total"], label=self.labels[0])
        ax.plot(self.x_axis, self.sim["xrpl_arbit_txs_total"], label=self.labels[1])
        ax.plot(self.x_axis, self.sim["uniswap_arbit_txs_total"], label=self.labels[2])

        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Number of Transactions")
        ax.set_title(
            "Number of Txs made by the arbitrageur at each iteration"
        ) if set_title else None
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        return ax

    def compute_losses(self, revs):
        losses = []
        for rev in revs:
            loss = len([x for x in rev if x < 0])
            losses.append(loss)
        avg_losses = round(np.average(losses))
        return avg_losses

    def plot_average_total_txs(self, ax, set_title=True):
        avg_xrpl_txs = round(np.average(self.sim["xrpl_arbit_txs_total"]))
        avg_xrplCAM_txs = round(np.average(self.sim["xrplCAM_arbit_txs_total"]))
        avg_eth_txs = round(np.average(self.sim["uniswap_arbit_txs_total"]))

        x = self.labels
        y = [avg_xrplCAM_txs, avg_xrpl_txs, avg_eth_txs]

        ax.bar(x, y)

        ax.text(
            x[0],
            y[0] - 2,
            "(~" + f'{self.compute_losses(self.sim["xrplCAM_profits_total"])} losses)',
            ha="center",
            va="top",
            fontsize=8,
        )

        ax.text(
            x[1],
            y[1] - 2,
            "(~" + f'{self.compute_losses(self.sim["xrpl_profits_total"])} losses)',
            ha="center",
            va="top",
            fontsize=8,
        )

        ax.text(
            x[2],
            y[2] - 2,
            "(~" + f'{self.compute_losses(self.sim["uniswap_profits_total"])} losses)',
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.bar_label(ax.containers[0], fmt="{:,.0f}")

        ax.set_ylabel("# of Txs")
        ax.set_title("Average total arbitrage Txs") if set_title else None
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        return ax

    def plot_total_fees_paid(self, ax, set_title=True):
        xrpl_fees_total = [i * self.xrpl_fees for i in self.sim["xrpl_arbit_txs_total"]]
        xrplCAM_fees_total = [
            i * self.xrpl_fees for i in self.sim["xrplCAM_arbit_txs_total"]
        ]
        uniswap_fees_total = [
            i * self.eth_fees for i in self.sim["uniswap_arbit_txs_total"]
        ]

        ax.plot(self.x_axis, xrplCAM_fees_total, label=self.labels[0])
        ax.plot(self.x_axis, xrpl_fees_total, label=self.labels[1])
        ax.plot(self.x_axis, uniswap_fees_total, label=self.labels[2])

        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Cumulative Tx fees ({})".format(self.assetB))
        ax.set_title(
            "Total Tx fees paid by the arbitrageur at each iteration"
        ) if set_title else None
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        return ax

    def plot_average_total_fees(self, ax, set_title=True):
        avg_xrpl_txs = round(np.average(self.sim["xrpl_arbit_txs_total"]))
        avg_xrplCAM_txs = round(np.average(self.sim["xrplCAM_arbit_txs_total"]))
        avg_eth_txs = round(np.average(self.sim["uniswap_arbit_txs_total"]))

        xrpl_fees_total = [i * self.xrpl_fees for i in self.sim["xrpl_arbit_txs_total"]]
        xrplCAM_fees_total = [
            i * self.xrpl_fees for i in self.sim["xrplCAM_arbit_txs_total"]
        ]
        uniswap_fees_total = [
            i * self.eth_fees for i in self.sim["uniswap_arbit_txs_total"]
        ]

        avg_xrpl_fees = round(np.average(xrpl_fees_total), 4)
        avg_xrplCAM_fees = round(np.average(xrplCAM_fees_total), 4)
        avg_eth_fees = round(np.average(uniswap_fees_total), 4)

        x = [
            "XRPL fees for \n ~" + f"{avg_xrplCAM_txs} txs",
            "XRPL fees for \n ~" + f"{avg_xrpl_txs} txs",
            "ETH fees for \n ~" + f"{avg_eth_txs} txs",
        ]
        y = [avg_xrplCAM_fees, avg_xrpl_fees, avg_eth_fees]

        ax.bar(x, y)

        ax.bar_label(ax.containers[0], fmt="{0:,.3f}")

        ax.set_ylabel("Tx fees ({})".format(self.assetB))
        ax.set_title("Average total Tx/network fees") if set_title else None
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        return ax

    def plot_amm_prices(self, ax, key, label, set_title=True):
        ax.plot(self.external_prices, label="External Market")
        ax.plot(self.sim[key][0][1], linestyle="dashed", label=label)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Price")
        ax.set_title("Price Sync. (last iteration only)") if set_title else None
        ax.legend()
        return ax

    def plot_cdf(
        self, ax, times1, times2, times3, bin, xlabel, multiplier=1, set_title=True
    ):
        sns.set()

        if not self.with_cases or xlabel == "Impermanent Loss (%)":
            curve1 = [t * multiplier for t in times1]
            curve1 = np.asarray(curve1)
            count, bins_count1 = np.histogram(curve1, bins=bin)
            curve1 = count / sum(count)
            cdf1 = np.cumsum(curve1)

        times2 = (
            times2
            if (not self.with_cases or xlabel == "Impermanent Loss (%)")
            else times1 + times2
        )

        curve2 = [t * multiplier for t in times2]
        curve2 = np.asarray(curve2)
        count, bins_count2 = np.histogram(curve2, bins=bin)
        curve2 = count / sum(count)
        cdf2 = np.cumsum(curve2)

        curve3 = [t * multiplier for t in times3]
        curve3 = np.asarray(curve3)
        count, bins_count3 = np.histogram(curve3, bins=bin)
        curve3 = count / sum(count)
        cdf3 = np.cumsum(curve3)

        size = 13

        if not self.with_cases or xlabel == "Impermanent Loss (%)":
            ax.plot(bins_count1[1:], cdf1, color="r", linewidth=3, label=self.labels[0])
        ax.plot(
            bins_count2[1:],
            cdf2,
            color="g",
            linewidth=3,
            label=self.labels[1]
            if (not self.with_cases or xlabel == "Impermanent Loss (%)")
            else "XRPL-CAM",
        )

        ax.plot(bins_count3[1:], cdf3, color="b", linewidth=3, label=self.labels[2])
        ax.set_ylabel("CDF", fontsize=size)
        ax.set_xlabel(xlabel, fontsize=size)
        ax.tick_params(axis="both", which="major", labelsize=size)
        ax.set_title(
            "Test-1 (equal network fees)"
            if self.equality_condition == "equal_fees"
            else (
                "Test-2 (equal block time)"
                if self.equality_condition == "equal_block_time"
                else "\u03C3 = {}%".format(self.gbm_sigma * 100)
            )
        )

        ax.legend(loc="lower right", fontsize=11)

        return ax

    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size), "valid") / window_size

    def plot_prices_moving_average(self, ax, set_title=True):
        (
            percent_differences_uni_total,
            percent_differences_xrpl_total,
            percent_differences_xrplCAM_total,
        ) = ([], [], [])
        for i in range(self.iterations):
            percent_differences_xrplCAM_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["xrplCAM_sps_total"][i][1]
                    )
                ]
            )
            percent_differences_xrpl_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["xrpl_sps_total"][i][1]
                    )
                ]
            )
            percent_differences_uni_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["uniswap_sps_total"][i][1]
                    )
                ]
            )

        averages_uni = [
            sum(values) / len(values) for values in zip(*percent_differences_uni_total)
        ]
        averages_xrpl = [
            sum(values) / len(values) for values in zip(*percent_differences_xrpl_total)
        ]
        averages_xrplCAM = [
            sum(values) / len(values)
            for values in zip(*percent_differences_xrplCAM_total)
        ]
        averages_xrpls = [(x + y) / 2 for x, y in zip(averages_xrpl, averages_xrplCAM)]

        indices = list(range(len(self.external_prices)))

        # window_size = int(len(self.external_prices) * 0.0125)
        window_size = 75

        smoothed_percent_differences_uni = self.moving_average(
            averages_uni, window_size
        )
        smoothed_percent_differences_xrplCAM = self.moving_average(
            averages_xrplCAM, window_size
        )
        smoothed_percent_differences_xrpl = self.moving_average(
            averages_xrpl, window_size
        )
        smoothed_percent_differences_xrpls = self.moving_average(
            averages_xrpls, window_size
        )

        ax.axhline(y=0, color="black", alpha=0.2)

        if self.with_cases:
            ax.plot(
                indices[: -window_size + 1],
                smoothed_percent_differences_xrpls,
                marker="",
                linestyle="-",
                linewidth=1,
                label="XRPL-CAM ({}MA)".format(window_size),
                color="g",
            )
        else:
            ax.plot(
                indices[: -window_size + 1],
                smoothed_percent_differences_xrplCAM,
                marker="",
                linestyle="-",
                linewidth=1,
                label="{} ({}MA)".format(self.labels[0], window_size),
                color="r",
            )
            ax.plot(
                indices[: -window_size + 1],
                smoothed_percent_differences_xrpl,
                marker="",
                linestyle="-",
                linewidth=1,
                label="{} ({}MA)".format(self.labels[1], window_size),
                color="g",
            )

        ax.plot(
            indices[: -window_size + 1],
            smoothed_percent_differences_uni,
            marker="",
            linestyle="-",
            linewidth=1,
            label="{} ({}MA)".format(self.labels[2], window_size),
            color="b",
        )

        ax.set_xlabel("Time step")
        ax.set_ylabel("Difference with Ext. \nMarket 75MA (%)")
        ax.set_xlim(0, len(smoothed_percent_differences_uni))
        ax.legend()
        ax.grid(axis="x")
        ax.set_title("\u03C3 = {}%".format(self.gbm_sigma * 100))
        return ax

    def plot_LP_returns(self, ax, set_title=True):
        avg_B_xrplCAM = sum(self.sim["xrplCAM_tfees_total"]) / len(
            self.sim["xrplCAM_tfees_total"]
        )
        avg_B_xrpl = sum(self.sim["xrpl_tfees_total"]) / len(
            self.sim["xrpl_tfees_total"]
        )
        avg_B_uniswap = sum(self.sim["uniswap_tfees_total"]) / len(
            self.sim["uniswap_tfees_total"]
        )

        df = pd.DataFrame(
            {
                "Categories": self.labels,
                "Values": [avg_B_xrplCAM, avg_B_xrpl, avg_B_uniswap],
            }
        )

        cmap = plt.cm.get_cmap("Blues")
        bars_bottom = ax.bar(
            df["Categories"],
            [avg_B_xrplCAM, avg_B_xrpl, avg_B_uniswap],
            color=cmap(0.7),
            label="Trading fee returns",
        )
        bars_top = ax.bar(
            df["Categories"],
            [
                (
                    sum(
                        self.sim["bids_profit_total"][i]["B_adjusted"]
                        for i in range(len(self.sim["bids_profit_total"]))
                    )
                    / self.iterations
                ),
                0,
                0,
            ],
            bottom=[avg_B_xrplCAM, avg_B_xrpl, avg_B_uniswap],
            color=cmap(0.9),
            label="CAM bids returns",
        )

        ax.set_ylabel("Returns ({})".format(self.assetB))
        ax.get_yaxis().set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )
        ax.bar_label(bars_bottom, fmt="{:,.0f}")
        ax.bar_label(bars_top, fmt="{:,.0f}")
        ylim = ax.get_ylim()
        ax.set_ylim(0, 1.1 * ylim[1])
        ax.set_title("LPs Returns") if set_title else None
        ax.legend()
        return ax

    def plot_trading_volume(self, ax, set_title=True):
        average_xrplCAM = np.mean(self.sim["xrplCAM_trading_volumes"])
        average_xrp = np.mean(self.sim["xrpl_trading_volumes"])
        average_uniswap = np.mean(self.sim["uniswap_trading_volumes"])

        x = self.labels
        y = [average_xrplCAM, average_xrp, average_uniswap]

        ax.bar(x, y)
        ax.set_ylabel("Average Trading Volume ({})".format(self.assetB))
        ax.bar_label(ax.containers[0], fmt="{:,.0f}")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

        return ax

    def display_results(self, to_save=False, foldername=None):
        sns.set()
        if not to_save:
            print(
                f"Arbitrageurs are more profitable on XRPL {self.sim['xrpl_ArbProfits_advantage']}% of the time\n"
                f"Arbitrageurs are more profitable on XRPL_CAM {self.sim['xrplCAM_ArbProfits_advantage']}% of the time\n"
                f"Arbitrageurs are more profitable on XRPL_CAM than XRPL {self.sim['xrpls_CAM_ArbProfits_advantage']}% of the time\n"
                "---\n"
                f"Price Sync. is better on XRPL {self.sim['xrpl_PriceGap_advantage']}% of the time\n"
                f"Price Sync. is better on XRPL_CAM {self.sim['xrplCAM_PriceGap_advantage']}% of the time\n"
                f"Price Sync. is better on XRPL_CAM than XRPL {self.sim['xrpls_CAM_PriceGap_advantage']}% of the time\n"
                "---\n"
                f"LPs earn more on XRPL {self.sim['xrpl_LP_returns_advantage']}% of the time\n"
                f"LPs earn more on XRPL_CAM {self.sim['xrplCAM_LP_returns_advantage']}% of the time\n"
                f"LPs earn more on XRPL_CAM than XRPL {self.sim['xrpls_CAM_LP_returns_advantage']}% of the time"
            )

        fig, (
            (ax1, ax2, ax3),
            (ax4, ax5, ax6),
            (ax7, ax8, ax9),
            (ax10, ax11, _),
            (ax13, ax14, ax15),
            (ax16, ax17, ax18),
        ) = plt.subplots(6, 3, figsize=(20, 30))
        fig.text(
            0.37,
            0.9,
            s="XRPL fees: "
            + f"{self.xrpl_fees}"
            + ", ETH fees: "
            + f"{self.eth_fees}"
            + ", Safe profit margin: "
            + f"{self.safe_profit_margin}%"
            + ",  Max. slippage: "
            + f"{self.max_slippage}%",
            ha="right",
            va="top",
        )

        self.plot_arbitrageurs_profits(ax1)
        self.plot_average_arbitrageurs_profits(ax2)
        self.plot_number_of_txs(ax3)
        self.plot_average_total_txs(ax4)
        self.plot_total_fees_paid(ax5)
        self.plot_average_total_fees(ax6)
        self.plot_amm_prices(ax7, "xrplCAM_sps_total", self.labels[0])
        self.plot_amm_prices(ax8, "xrpl_sps_total", self.labels[1])
        self.plot_amm_prices(ax9, "uniswap_sps_total", self.labels[2])

        percent_differences_uniswap_total = []
        percent_differences_xrpl_total = []
        percent_differences_xrplCAM_total = []
        for i in range(self.iterations):
            percent_differences_xrplCAM_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["xrplCAM_sps_total"][i][1]
                    )
                ]
            )
            percent_differences_xrpl_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["xrpl_sps_total"][i][1]
                    )
                ]
            )
            percent_differences_uniswap_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["uniswap_sps_total"][i][1]
                    )
                ]
            )

        self.plot_cdf(
            ax10,
            [
                abs(item)
                for sublist in percent_differences_xrplCAM_total
                for item in sublist
            ],
            [
                abs(item)
                for sublist in percent_differences_xrpl_total
                for item in sublist
            ],
            [
                abs(item)
                for sublist in percent_differences_uniswap_total
                for item in sublist
            ],
            500,
            "Price Difference with Reference Market (%)",
        )

        ax11 = plt.subplot2grid((6, 3), (3, 1), colspan=2)
        self.plot_prices_moving_average(ax11)

        slips_xrpl_cam = [
            item for sublist in self.sim["xrplCAM_slippages_total"] for item in sublist
        ]
        slips_xrpl = [
            item for sublist in self.sim["xrpl_slippages_total"] for item in sublist
        ]
        slips_uniswap = [
            item for sublist in self.sim["uniswap_slippages"] for item in sublist
        ]
        self.plot_cdf(
            ax13, slips_xrpl_cam, slips_xrpl, slips_uniswap, 500, "Slippage (%)", 100
        )

        self.plot_cdf(
            ax14,
            self.sim["impermanent_losses"]["xrplCAM"],
            self.sim["impermanent_losses"]["xrpl"],
            self.sim["impermanent_losses"]["uniswap"],
            500,
            "Impermanent Loss (%)",
            100,
        )
        self.plot_LP_returns(ax15)
        self.plot_trading_volume(ax16)
        self.plot_cdf(
            ax17,
            self.sim["price_impacts"]["xrplCAM"],
            self.sim["price_impacts"]["xrpl"],
            self.sim["price_impacts"]["uniswap"],
            500,
            "Price Impact (%)",
            100,
        )

        plt.subplots_adjust(hspace=0.3, wspace=0.4)

        if to_save:
            folder_path = os.path.join("simulation_results", foldername, "plots")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            fig.savefig(
                os.path.join(folder_path, "figure_plots_combined.pdf"), format="pdf"
            )
            plt.close()
        else:
            plt.show()

    def save_plots_results(self, foldername, set_title):
        sns.set()

        (
            percent_differences_uniswap_total,
            percent_differences_xrpl_total,
            percent_differences_xrplCAM_total,
        ) = ([], [], [])
        for i in range(self.iterations):
            percent_differences_xrplCAM_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["xrplCAM_sps_total"][i][1]
                    )
                ]
            )
            percent_differences_xrpl_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["xrpl_sps_total"][i][1]
                    )
                ]
            )
            percent_differences_uniswap_total.append(
                [
                    (b - a) / a * 100
                    for a, b in zip(
                        self.external_prices, self.sim["uniswap_sps_total"][i][1]
                    )
                ]
            )

        # slippages:
        slips_xrpl_cam = [
            item for sublist in self.sim["xrplCAM_slippages_total"] for item in sublist
        ]
        slips_xrpl = [
            item for sublist in self.sim["xrpl_slippages_total"] for item in sublist
        ]
        slips_uniswap = [
            item for sublist in self.sim["uniswap_slippages"] for item in sublist
        ]

        plots_dict = {
            "arbit_profits.pdf": lambda ax: self.plot_arbitrageurs_profits(
                ax, set_title
            ),
            "avg_arbit_profits.pdf": lambda ax: self.plot_average_arbitrageurs_profits(
                ax, set_title
            ),
            "numb_of_txs.pdf": lambda ax: self.plot_number_of_txs(ax, set_title),
            "avg_numb_of_txs.pdf": lambda ax: self.plot_average_total_txs(
                ax, set_title
            ),
            "fees_paid.pdf": lambda ax: self.plot_total_fees_paid(ax, set_title),
            "avg_fees_paid.pdf": lambda ax: self.plot_average_total_fees(ax, set_title),
            "xrplCAM_price_sync_last_iter.pdf": lambda ax: self.plot_amm_prices(
                ax, "xrplCAM_sps_total", self.labels[0], set_title
            ),
            "xrpl_price_sync_last_iter.pdf": lambda ax: self.plot_amm_prices(
                ax, "xrpl_sps_total", self.labels[1], set_title
            ),
            "uniswap_price_sync_last_iter.pdf": lambda ax: self.plot_amm_prices(
                ax, "uniswap_sps_total", self.labels[2], set_title
            ),
            "cdf_price_diff.pdf": lambda ax: self.plot_cdf(
                ax,
                np.abs(percent_differences_xrplCAM_total),
                np.abs(percent_differences_xrpl_total),
                np.abs(percent_differences_uniswap_total),
                500,
                "Price Difference with Reference Market (%)",
                set_title=set_title,
            ),
            "prices_MA.pdf": lambda ax: self.plot_prices_moving_average(ax, set_title),
            "cdf_slippages.pdf": lambda ax: self.plot_cdf(
                ax,
                slips_xrpl_cam,
                slips_xrpl,
                slips_uniswap,
                500,
                "Slippage (%)",
                100,
                set_title,
            ),
            "LP_returns.pdf": lambda ax: self.plot_LP_returns(ax, set_title),
            "cdf_divergence_loss": lambda ax: self.plot_cdf(
                ax,
                self.sim["impermanent_losses"]["xrplCAM"],
                self.sim["impermanent_losses"]["xrpl"],
                self.sim["impermanent_losses"]["uniswap"],
                500,
                "Impermanent Loss (%)",
                100,
            ),
            "trading_volume.pdf": lambda ax: self.plot_trading_volume(ax, set_title),
            "cdf_price_impact.pdf": lambda ax: self.plot_cdf(
                ax,
                self.sim["price_impacts"]["xrplCAM"],
                self.sim["price_impacts"]["xrpl"],
                self.sim["price_impacts"]["uniswap"],
                500,
                "Price Impact (%)",
                100,
                set_title,
            ),
            "cdf_slippages_fair.pdf": lambda ax: self.plot_cdf(
                ax,
                slips_xrpl_cam,
                slips_xrpl,
                slips_uniswap,
                500,
                "Slippage (%)",
                100,
                set_title,
            ),
            "cdf_price_diff_fair.pdf": lambda ax: self.plot_cdf(
                ax,
                np.abs(percent_differences_xrplCAM_total),
                np.abs(percent_differences_xrpl_total),
                np.abs(percent_differences_uniswap_total),
                500,
                "Price Difference with Reference Market (%)",
                set_title=set_title,
            ),
        }

        for filename in plots_dict:
            fig, ax = (
                plt.subplots(figsize=(15, 6))
                if filename == "prices_MA.pdf"
                else plt.subplots()
            )
            self.save_plot(plots_dict[filename](ax), foldername, filename)
            plt.close(fig)

        # save all plots in a combined figure:
        self.display_results(to_save=True, foldername=foldername)
