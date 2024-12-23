#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define a color palette so that each protocol always uses the same color.
# Example: XRPL-AMM-CAM = red, XRPL-AMM = green, Uniswap = blue
PROTOCOL_COLORS = {
    "XRPL-AMM": "#2ca02c",       # green
    "XRPL-AMM-CAM": "#d62728",   # red
    "Uniswap": "#1f77b4",        # blue
    "G-AMM": "#1f77b4",          # also blue
}

# Shared kwargs for ecdfplot so that we keep consistent colors + no legend title
ecdf_kwargs = dict(
    hue="protocol",
    palette=PROTOCOL_COLORS,
    legend=True  # We'll set legend title=None right afterward
)

# To keep the similar style for all plots

# 3) Generate plots
sns.set_style("darkgrid")

# Decrease font size a bit or keep default
sns.set_context("paper")

def flatten_cdf_data(
    scenario_data: dict,
    scenario_name: str,
    metric_key: str,
    protocols=("XRPL-AMM", "G-AMM"),
    protocol_map=None,
):
    """
    Flatten data under scenario_data[metric_key] into a DataFrame
    for easy plotting with seaborn. Returns a dataframe with columns:
        [ scenario, protocol, value ]

    Example:
        metric_key: "slippages"
        scenario_data["slippages"] -> {
            "XRPL-AMM": [0.015, 0.017, 0.014, ...],
            "G-AMM": [0.022, 0.020, 0.021, ...]
        }
    """
    if protocol_map is None:
        protocol_map = {
            "xrplCAM": "XRPL-AMM",
            "xrpl": "XRPL-AMM",
            "uniswap": "G-AMM",
            "XRPL-AMM": "XRPL-AMM",
            "G-AMM": "G-AMM",
        }

    dfs = []
    for proto in scenario_data[metric_key]:
        # Because your JSON might store keys like "xrplCAM", or "XRPL-AMM", etc.
        # We'll map them to a friendly name or default back to the same string.
        new_proto_name = protocol_map.get(proto, proto)

        # scenario_data[metric_key][proto] is presumably a list of numeric values
        values = scenario_data[metric_key][proto]

        tmp_df = pd.DataFrame(
            {
                "scenario": [scenario_name] * len(values),
                "protocol": [new_proto_name] * len(values),
                "value": values,
            }
        )
        dfs.append(tmp_df)

    return pd.concat(dfs, ignore_index=True)


def flatten_divergence_data(
    scenario_data: dict,
    scenario_name: str,
    protocols=("XRPL-AMM", "G-AMM"),
    protocol_map=None,
):
    """
    If you have something akin to "impermanent_losses" or "divergence gains"
    under each scenario, you can flatten it in a similar manner.
    Returns a dataframe with columns: [ scenario, protocol, divergence_gain ]
    """
    if protocol_map is None:
        protocol_map = {
            "xrplCAM": "XRPL-AMM",
            "xrpl": "XRPL-AMM",
            "uniswap": "G-AMM",
            "XRPL-AMM": "XRPL-AMM",
            "G-AMM": "G-AMM",
        }

    # Suppose scenario_data["impermanent_losses"] has a dict
    #   { 'xrplCAM': [...], 'xrpl': [...], 'uniswap': [...] }
    # but you want to plot it as “divergence gains” or “losses”.
    # This is just an example. Adjust as needed:
    metric_key = "impermanent_losses"
    if metric_key not in scenario_data:
        return pd.DataFrame()

    dfs = []
    for proto in scenario_data[metric_key]:
        new_proto_name = protocol_map.get(proto, proto)
        values = scenario_data[metric_key][proto]
        tmp_df = pd.DataFrame(
            {
                "scenario": [scenario_name] * len(values),
                "protocol": [new_proto_name] * len(values),
                "divergence_gain": values,
            }
        )
        dfs.append(tmp_df)
    return pd.concat(dfs, ignore_index=True)


def rename_scenario(label: str) -> str:
    """
    Helper to rename scenario:
    - "test-1" -> "Test-1"
    - "test-2" -> "Test-2"
    - "xrpl_amm_dex-cam" -> "XRPL AMM with CAM"
    """
    if label == "xrpl_amm_dex-cam":
        return "XRPL-CAM"
    # For "test-1", "test-2", etc., just uppercase the first letter:
    # e.g. "test-1" -> "Test-1"
    return label[:1].upper() + label[1:]  # e.g. "test-1" -> "Test-1"


def gen_cdf_subplots(args, df, scenario_order, x_axis_label="Price Variation (%)",
                     y_axis_label="CDF", filename="price_variation_cdf_subplots.pdf"):

    # 3) Generate plots
    sns.set_style("darkgrid")

    # Decrease font size a bit or keep default
    sns.set_context("paper")

    os.makedirs(args.output_dir, exist_ok=True)

    # Create subplots: one row per scenario
    fig, axes = plt.subplots(
        nrows=len(scenario_order),
        ncols=1,
        sharex=True,
        figsize=(5, 1.5 * len(scenario_order))
    )

    if len(scenario_order) == 1:
        axes = [axes]

    # We will ask Seaborn to draw a legend only on the last subplot
    # Then we can customize that legend’s location/title
    for i, scenario in enumerate(scenario_order):
        ax = axes[i]

        # Subset data
        scenario_data = df.loc[df["scenario"] == scenario]

        # If no data, hide Axes
        if scenario_data.empty:
            ax.set_visible(False)
            continue

        # Should we show the legend on this subplot?
        # We'll only show it on the LAST subplot
        show_legend = (i == len(scenario_order) - 1)

        # Draw the ECDF
        sns.ecdfplot(
            data=scenario_data,
            x="value",
            hue="protocol",
            palette=PROTOCOL_COLORS,
            legend=show_legend,  # only show legend on last Axes
            ax=ax
        )

        # Make sure each subplot has a dark outline by enabling edges
        # (And optionally adjusting line widths & colors)
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)  # ensure spines are visible
            spine.set_edgecolor("black")  # or a dark gray
            spine.set_linewidth(0.7)

        # Scenario name on y-axis and fontsize
        ax.set_ylabel(rename_scenario(scenario), fontsize=14)

        ax.tick_params(axis="both", which="major", labelsize=14)

        # Set the y-axis limit to avoid 0 overlap with other subplots
        ax.set_ylim(-0.10, 1.10)

        # Only label X on the last subplot
        if i < len(scenario_order) - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(x_axis_label, fontsize=14)

        if show_legend:

            # After ecdfplot, we need to grab the Axes’ legend
            legend_obj = ax.get_legend()

            if legend_obj:
                # Increase legend font size
                plt.setp(legend_obj.get_texts(), fontsize=10)

                legend_obj.set_title(None)  # remove “protocol” label
                legend_obj.set_frame_on(True)  # box around it
                legend_obj._loc = 4  # code for 'lower right'
                # or: legend_obj.set_bbox_to_anchor((1.0, 0.0)) for more precise placement

    # Remove vertical spacing between each subplot
    plt.subplots_adjust(hspace=0.0)
    outpath = os.path.join(args.output_dir, filename)

    print(f"Saving subplots to {outpath}")
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot AMM simulation results.")
    parser.add_argument(
        "--input_json",
        type=str,
        default="./../simulation_results/test2/results/aggregated_results.json",
        help="Path to the aggregated_results.json file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./../simulation_results/figures",
        help="Directory to save the plots.",
    )

    parser.add_argument(
        "--output_csv_dir",
        type=str,
        default="./../simulation_results/test2/results",
        help="Directory to save the dataframes.",
    )

    args = parser.parse_args()

    # 1) Load the aggregated results
    with open(args.input_json, "r") as f:
        all_data = json.load(f)

    # 2) Flatten data for each metric you want to plot
    #    We'll create separate dataframes for each metric.
    df_slippage = []
    df_price_impact = []
    df_price_variation = []
    df_divergence = []

    # To update the naming of the plots
    protocol_map = {"xrpl": "XRPL-AMM", "xrplCAM": "XRPL-AMM", "uniswap": "G-AMM"}

    for scenario_name, scenario_data in all_data.items():
        if not isinstance(scenario_data, dict):
            # skip non-scenario items
            continue

        if "slippages" in scenario_data:
            df_slippage.append(
                flatten_cdf_data(
                    scenario_data,
                    scenario_name,
                    metric_key="slippages",
                    protocol_map=protocol_map,
                )
            )
        if "price_impacts" in scenario_data:
            df_price_impact.append(
                flatten_cdf_data(
                    scenario_data,
                    scenario_name,
                    metric_key="price_impacts",
                    protocol_map=protocol_map,
                )
            )
        if "price_variations" in scenario_data:
            df_price_variation.append(
                flatten_cdf_data(
                    scenario_data,
                    scenario_name,
                    metric_key="price_variations",
                    protocol_map=protocol_map,
                )
            )
        # Impermanent losses or "divergence gains"
        df_divergence.append(
            flatten_divergence_data(scenario_data, scenario_name, protocol_map=protocol_map)
        )

    df_slippage = pd.concat(df_slippage, ignore_index=True) if df_slippage else None
    df_price_impact = (
        pd.concat(df_price_impact, ignore_index=True) if df_price_impact else None
    )
    df_price_variation = (
        pd.concat(df_price_variation, ignore_index=True) if df_price_variation else None
    )
    df_divergence = (
        pd.concat(df_divergence, ignore_index=True) if df_divergence else None
    )

    # Save the dataframes
    df_slippage.to_csv(os.path.join(args.output_csv_dir, "slippage.csv"), index=False)
    df_price_impact.to_csv(os.path.join(args.output_csv_dir, "price_impact.csv"), index=False)
    df_price_variation.to_csv(os.path.join(args.output_csv_dir, "price_variation.csv"), index=False)
    df_divergence.to_csv(os.path.join(args.output_csv_dir, "divergence.csv"), index=False)

    # Identify all scenarios that appear in df_price_variation.
    # You can also define a manual order, e.g. scenario_order = ["test-2","test-1","xrpl_amm_dex-cam"]
    scenario_order = df_price_variation["scenario"].unique().tolist()

    # -- Figure 1: Price Variation CDF --
    gen_cdf_subplots(args, df_price_variation, scenario_order,
                     filename="price_variation_cdf_subplots.pdf",
                     x_axis_label="Price Variation (%)")

    # plt.figure(figsize=(6, 5))
        #
        # # We can use seaborn’s ECDF plot to replicate a CDF:
        # #  “Price Variation (%)”
        # ax = sns.ecdfplot(
        #     data=df_price_variation,
        #     x="value",
        #     # hue="protocol",
        #     linestyle="-",
        #     ** ecdf_kwargs
        # )
        # ax.set_xlabel("Price Variation (%)")
        # ax.set_ylabel("CDF")
        # ax.set_title("")
        #
        # # plt.legend(title="Protocol", loc="lower right")
        # # Remove legend title
        # ax.legend(title=None, loc="lower right")
        #
        # outpath = os.path.join(args.output_dir, "price_variation_cdf.pdf")
        # plt.savefig(outpath, bbox_inches="tight")
        # plt.close()

    # -- Figure 2: Slippage CDF --
    if df_slippage is not None and not df_slippage.empty:

        gen_cdf_subplots(args, df_slippage, scenario_order, x_axis_label="Slippage (%)",
                         filename="slippage_cdf_subplots.pdf")

    #     plt.figure(figsize=(6, 5))
    #
    #     # If your slippage is stored as fraction, multiply by 100 for %
    #     # or if it's already in %, just rename label:
    #     ax = sns.ecdfplot(
    #         data=df_slippage,
    #         x="value",
    #         # hue="protocol", ### Already set in ecdf_kwargs
    #         ** ecdf_kwargs
    #     )
    #     ax.set_xlabel("Slippage (%)")
    #     ax.set_ylabel("CDF")
    #     ax.set_title("")
    #     # plt.legend(title="Protocol", loc="lower right")
    #     # Remove legend title
    #     ax.legend(title=None, loc="lower right")
    #
    #     outpath = os.path.join(args.output_dir, "slippage_cdf.pdf")
    #     plt.savefig(outpath, bbox_inches="tight")
    #     plt.close()
    #
    # -- Figure 3: Price Impact CDF --
    if df_price_impact is not None and not df_price_impact.empty:

        gen_cdf_subplots(args, df_price_impact, scenario_order, x_axis_label="Price Impact (%)",
                         filename="price_impact_cdf_subplots.pdf")

        # plt.figure(figsize=(6, 5))
        # ax = sns.ecdfplot(
        #     data=df_price_impact,
        #     x="value",
        #     # hue="protocol",  ### Already set in ecdf_kwargs
        #     **ecdf_kwargs
        # )
        # ax.set_xlabel("Price Impact (%)")
        # ax.set_ylabel("CDF")
        # ax.set_title("")
        # # plt.legend(title="Protocol", loc="lower right")
        #
        # # Remove legend title
        # ax.legend(title=None, loc="lower right")
        #
        # outpath = os.path.join(args.output_dir, "price_impact_cdf.pdf")
        # plt.savefig(outpath, bbox_inches="tight")
        # plt.close()

    # -- Figure 4: Divergence Gains / Impermanent Losses Dot Plot --
    # Suppose you want a grouped swarmplot or stripplot for each scenario & protocol
    # (like your “Fig. 8: LPs’ divergence gains for Test-1 and Test-2”).
    if df_divergence is not None and not df_divergence.empty:

        # plt.figure(figsize=(7, 4.5))
        #
        # # For small datasets, swarmplot works. For larger ones, stripplot or jittered scatter:
        # # ax = sns.stripplot(
        # #     data=df_divergence,
        # #     x="scenario",
        # #     y="divergence_gain",  # or "impermanent_loss"
        # #     hue="protocol",
        # #     dodge=True,
        # #     alpha=0.6,
        # #     size=6,
        # # )
        #
        # # Use stripplot (or swarmplot):
        # ax = sns.stripplot(
        #     data=df_divergence,
        #     y="scenario",  # so scenarios are on y-axis
        #     x="divergence_gain",  # numeric measure on x-axis
        #     hue="protocol",
        #     palette=PROTOCOL_COLORS,
        #     dodge=True,
        #     alpha=0.6,
        #     size=6,
        # )
        #
        # # If your divergence gains are already in %, rename the axis label
        # ax.set_ylabel("Divergence Impact (%)")
        # ax.set_xlabel("")
        # # ax.set_title("LPs’ Divergence Gains Across Scenarios")
        # ax.set_title("")
        # plt.legend(title="", loc="best")
        #
        # outpath = os.path.join(args.output_dir, "divergence_dotplot.pdf")
        # plt.savefig(outpath, bbox_inches="tight")
        # plt.close()

        df_divergence["scenario"] = pd.Categorical(
            df_divergence["scenario"], categories=scenario_order, ordered=True
        )

        # Correct the scenario name
        df_divergence["scenario"] = df_divergence["scenario"].apply(rename_scenario)

        # Rename the scenario names
        scenario_order = [rename_scenario(scenario) for scenario in scenario_order]

        plt.figure(figsize=(8, 2))

        ax = sns.stripplot(
            data=df_divergence,
            y="scenario",  # discrete categories on y-axis
            x="divergence_gain",  # numeric measure on x-axis
            order=scenario_order,  # ensure consistent ordering top-to-bottom
            hue="protocol",
            palette=PROTOCOL_COLORS,
            dodge=False,  # separate different protocols horizontally (true) or stack them (false)
            jitter=False,  # no vertical scatter; each scenario is one line
            alpha=0.6,
            size=12,
            marker="o"
        )

        # Axis labels
        ax.set_xlabel("Divergence Impact (%)", fontsize=14)
        ax.set_ylabel("")

        plt.draw()  # Force figure to render and define tick locations
        xticks = ax.get_xticks()
        ax.set_xticklabels([f"+{tick:.2f}" for tick in xticks], fontsize=14)

        ax.set_axisbelow(True)  # Ensure grid is behind the points
        ax.grid(axis="y", alpha=0.7)

        ax.tick_params(axis="both", which="major", labelsize=14)

        # Legend
        ax.legend(title="", loc="best")

        plt.tight_layout()
        outpath = os.path.join(args.output_dir, "divergence_dotplot.pdf")
        plt.savefig(outpath, bbox_inches="tight", dpi=300)
        plt.close()

    print(f"All plots saved to {args.output_dir}")



if __name__ == "__main__":
    main()
