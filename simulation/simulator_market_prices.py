#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import argparse

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Adding the path to be able to import the analytics module
import sys

sys.path.append('./')
from simulator import Simulator
from simulator_plots import SimulatorResults

# Keep only the columns we need for the open price
COLS_TO_USE: List[str] = ['Open time', 'Open']

# Variables for simulation
ASSET_A = 'ETH'
ASSET_B = 'USDC'

NORMAL_USERS = 60
ARBITRAGEURS = 5
SAFE_PROFIT_MARGIN = 1.5 # (%)
TRADING_FEE_RATE = 0.3 # (%)
MAX_SLIPPAGE = 4 # (%)
ITERATIONS = 2
START_SIMULATION_AT_DAY = 3
# ONE_DAY = 1000 # Scale of what a day is for the timestep
SLOT_INTERVAL_CAM = 20 # The 24-hour slot is segmented into 20 intervals

WORST_CASE = False
WITH_CASES = False

# Define
EQUALITY_CONDITION = 'equal_fees'


def setup_logging():
    """Configure logging to both file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulator.log'),
            logging.StreamHandler()
        ]
    )


def binance_price_distribution(plot_path, external_prices, x):

    # visualize the external prices distribution
    sns.set_style(style="whitegrid")
    sns.set_context("paper")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=external_prices['date'], y=external_prices['price'])

    # Mark the start of the simulation, which is the third day in the dataset
    plt.axvline(x=x, color='red', linestyle='--')

    # Rotate the x-axis labels
    plt.xticks(rotation=90)

    plt.ylabel("Price")
    plt.tight_layout()

    plt.savefig(plot_path, format='pdf', dpi=300)


def calculate_percentage_difference(actual_prices, simulated_prices):
    """
    Calculate percentage difference between actual CEX price and (simulated) prices from AMM from users interaction.

    Args:
        actual_prices: List of reference/actual prices
        simulated_prices: List of simulated prices

    Returns:
        List of percentage differences
    """
    return [((sim - actual) / actual) * 100
            for actual, sim in zip(actual_prices, simulated_prices)]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default="2024-01-01")
    parser.add_argument('--end_date', type=str, default="2024-01-05")
    parser.add_argument('--output_dir', type=str, default="./simulation_results/test2")

    args = parser.parse_args()
    simulator = Simulator()


    external_prices_path = os.path.join(args.output_dir, "external_prices.parquet")
    if os.path.exists(external_prices_path):
        external_prices = pd.read_parquet(external_prices_path)

    else:

        prices_dir = os.path.join(args.output_dir, "binance")
        files = [os.path.join(prices_dir, f) for f in os.listdir(prices_dir)]

        dfs = []

        logging.info(f"Processing {len(files)} files")
        for file in tqdm(files):
            df = pd.read_parquet(file, engine='fastparquet')

            # Keep only the columns we need for the open price
            df = df[COLS_TO_USE]

            dfs.append(df)


        # Concatenate all the dataframes
        external_prices = pd.concat(dfs)

        # Arrange the date values
        external_prices = external_prices.rename(columns={'Open time': 'timestamp', "Open": "price"})
        external_prices['timestamp'] = pd.to_datetime(external_prices['timestamp'], unit='ms')
        external_prices['date'] = external_prices['timestamp'].dt.date

        # Convert date column to datetime
        external_prices['date'] = pd.to_datetime(external_prices['date'])

        # Sort the values by date
        external_prices.sort_values(by='timestamp', inplace=True)

        # Generate the timestep column from timestamp
        external_prices['timestep'] = external_prices['timestamp'].diff().dt.total_seconds().fillna(0).cumsum()

        # Save the combined dataframe
        external_prices.to_parquet(external_prices_path, engine='fastparquet')

    # Use only 5 days of data
    external_prices = external_prices[(external_prices['date'] >= args.start_date)
                                      & (external_prices['date'] <= args.end_date)]

    # Get the third date
    start_simulation_at_date = external_prices['date'].unique()[START_SIMULATION_AT_DAY - 1]

    # Get the equivalent of one day in timestep, which should be equivalent to 60 seconds * 60 minutes * 24 hours.
    # It should be equivalent to 86400 seconds = 1 day
    ONE_DAY = int(external_prices[external_prices['date'] == "2024-01-01"]['timestep'].tolist()[-1]) + 1

    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "binance_prices_simulation.pdf")

    if not os.path.exists(plot_path):
        binance_price_distribution(plot_path, external_prices, start_simulation_at_date)

    # Get the initial price
    initial_price = external_prices['price'].iloc[0]

    # Get all the prices
    prices = external_prices['price'].tolist()

    # Calculate volatility of prices = standard deviation * square root of the number of 252 trading days
    volatility = external_prices['price'].std() * np.sqrt(252)

    scenarios = {"test-1" : {
        "XRPL_BLOCK_INTERARRIVAL": 4,
        "ETH_BLOCK_INTERARRIVAL": 12,
        "XRPL_FEES": 1,
        "ETH_FEES": 1,
        "WITH_CASES": False  # XRPL AMM with CAM, which is the worst-case for LPs (XRPL-CAM-B)
    },
    "test-2" : {
        "XRPL_BLOCK_INTERARRIVAL": 8,
        "ETH_BLOCK_INTERARRIVAL": 8,
        "XRPL_FEES": 0.00001,
        "ETH_FEES": 4,
        "WITH_CASES": False  # XRPL AMM with CAM, which is the worst-case for LPs (XRPL-CAM-B)

    },
    "xrpl_amm_dex-cam" : {
        "XRPL_BLOCK_INTERARRIVAL": 4,
        "ETH_BLOCK_INTERARRIVAL": 12,
        "XRPL_FEES": 0.00001,
        "ETH_FEES": 4,
        "WITH_CASES": True # XRPL AMM without CAM
    }
    }

    # Save the simulation results
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save the aggregated simulation results
    simulation_results_path = os.path.join(results_dir, f"aggregated_results.json")

    sim_results = {}

    if not os.path.exists(simulation_results_path):

        for scenario, values in scenarios.items():

            scenario_simulation_results_path = os.path.join(results_dir, f"{scenario}.json")

            if not os.path.exists(scenario_simulation_results_path):

                print(f"Running simulation for {scenario}...")

                sim = simulator.run_simulation(initial_price, prices, values["XRPL_BLOCK_INTERARRIVAL"],
                                               values["ETH_BLOCK_INTERARRIVAL"], values["XRPL_FEES"], values["ETH_FEES"],
                                               TRADING_FEE_RATE/100,
                                               NORMAL_USERS, ARBITRAGEURS, SAFE_PROFIT_MARGIN/100, MAX_SLIPPAGE/100,
                                               ITERATIONS, ONE_DAY, start_simulation_at_date, START_SIMULATION_AT_DAY,
                                               WORST_CASE, values['WITH_CASES'],
                                               SLOT_INTERVAL_CAM,
                                               # use_historical_prices=False
                                               )

                print(f"Saving simulation results to {scenario_simulation_results_path}")
                with open(simulation_results_path, 'w') as f:
                    json.dump(sim, f, indent=4)

            else:

                print(f"Loading simulation results for {scenario}...")

                with open(scenario_simulation_results_path, 'r') as f:
                    sim = json.load(f)


            # (
            #     percent_differences_uniswap_total,
            #     percent_differences_xrpl_total,
            #     percent_differences_xrplCAM_total,
            # ) = ([], [], [])
            # for i in range(ITERATIONS):
            #     percent_differences_xrplCAM_total.append(
            #         [
            #             (b - a) / a * 100
            #             for a, b in zip(
            #                 prices, sim["xrplCAM_sps_total"][i][1]
            #             )
            #         ]
            #     )
            #     percent_differences_xrpl_total.append(
            #         [
            #             (b - a) / a * 100
            #             for a, b in zip(
            #                 prices, sim["xrpl_sps_total"][i][1]
            #             )
            #         ]
            #     )
            #     percent_differences_uniswap_total.append(
            #         [
            #             (b - a) / a * 100
            #             for a, b in zip(
            #                 prices, sim["uniswap_sps_total"][i][1]
            #             )
            #         ]
            #     )

            # Initialize empty lists to store percentage differences for each protocol
            protocol_differences = {}

            # # Calculate percentage differences only using the last iteration
            # # for iteration in range(ITERATIONS):
            #
            # # Map each protocol to its simulated prices for current iteration
            # simulated_prices = {
            #     'xrplCAM': sim["xrplCAM_sps_total"][iteration][1],
            #     'xrpl': sim["xrpl_sps_total"][iteration][1],
            #     'uniswap': sim["uniswap_sps_total"][iteration][1]
            # }

            # Map each protocol to its simulated prices for current iteration
            simulated_prices = {
                'xrplCAM': sim["xrplCAM_sps_total"][-1][1],
                'xrpl': sim["xrpl_sps_total"][-1][1],
                'uniswap': sim["uniswap_sps_total"][-1][1]
            }

            # Calculate and store percentage differences for each protocol
            for protocol in simulated_prices.keys():
                percent_diff = calculate_percentage_difference(prices, simulated_prices[protocol])
                protocol_differences[protocol] = list(np.abs(percent_diff))


            slippages = {"xrplCAM":sim["xrplCAM_slippages_total"][0] + sim["xrplCAM_slippages_total"][1],
                         "xrpl": sim["xrpl_slippages_total"][0] + sim["xrpl_slippages_total"][1],
                         "uniswap": sim["uniswap_slippages"][0] + sim["uniswap_slippages"][1]}

            sim_results[scenario] = {
                "price_impacts": sim["price_impacts"],
                "slippages": slippages,
                "impermanent_losses": sim["impermanent_losses"],
                "price_variations": protocol_differences,
            }

        print(f"Saving simulation results to {simulation_results_path}")
        with open(simulation_results_path, 'w') as f:
            json.dump(sim_results, f, indent=4)

        # results = SimulatorResults(sim, ASSET_A, ASSET_B, prices[ONE_DAY * START_SIMULATION_AT_DAY:], 0,
        #                            volatility, values["XRPL_BLOCK_INTERARRIVAL"], values['ETH_BLOCK_INTERARRIVAL'],
        #                            values['XRPL_FEES'], values['ETH_FEES'], SAFE_PROFIT_MARGIN,
        #                            MAX_SLIPPAGE, ITERATIONS, values['WITH_CASES'], EQUALITY_CONDITION)
        #
        # results.display_results()
        #
        # output_figures_dir = os.path.join(figures_dir, scenario)
        # results.save_plots_results(output_figures_dir, scenario)

if __name__ == "__main__":
    main()