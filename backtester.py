import pandas as pd
import numpy as np
import dill as pickle
import lzma
import random


def save_pickle(filepath, obj):
    """
    Saves an object to a compressed file using dill.

    Parameters:
    filepath (str): Path of the file to write the object to.
    obj: The object to be saved.
    """
    with lzma.open(filepath, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(filepath):
    """
    Loads an object from a compressed file using dill.

    Parameters:
    filepath (str): Path of the file to read the object from.

    Returns:
    The loaded object.
    """
    with lzma.open(filepath, 'rb') as file:
        return pickle.load(file)


class Alpha():
    """
    A class to represent an Alpha model for the trading simulation.
    """

    def __init__(self, insts, dfs, start, end):
        self.insts = insts
        self.dfs = dfs
        self.period_start = start
        self.period_end = end
        self.portfolio_df = self.init_portfolio_settings()

    def init_portfolio_settings(self):
        """Initializes the portfolio settings."""
        portfolio_df = pd.DataFrame(index=pd.date_range(start=self.period_start, end=self.period_end, freq="D"))
        portfolio_df = portfolio_df.reset_index().rename(columns={"index": "datetime"})
        portfolio_df.loc[0, "capital"] = 10000

        for inst in self.insts:
            portfolio_df[f"{inst} units"] = 0
            portfolio_df[f"{inst} w"] = 0

        return portfolio_df

    def compute_meta_info(self):
        """Prepares the data for the simulation."""
        df = pd.DataFrame(index=self.portfolio_df["datetime"])

        for inst in self.insts:
            self.dfs[inst].index = self.dfs[inst].index - pd.Timedelta(hours=5)
            self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
            self.dfs[inst]["ret"] = self.dfs[inst]["close"] / self.dfs[inst]["close"].shift(1) - 1
            sampled = self.dfs[inst]["close"].diff().ne(0)
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
            self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)

    def run_simulation(self):
        """Runs the trading simulation."""
        print("Running simulation")
        self.compute_meta_info()

        for i in self.portfolio_df.index:
            date = self.portfolio_df.loc[i, "datetime"]
            eligibles = [inst for inst in self.insts if self.dfs[inst].loc[date, "eligible"]]

            if i != 0:
                self.calculate_portfolio_values(i, date, eligibles)

            self.allocate_to_eligibles(i, date, eligibles)
            yield self.portfolio_df.loc[i]

        return self.portfolio_df

    def calculate_portfolio_values(self, i, date, eligibles):
        """Calculates the portfolio values for a given date."""
        prev_date = self.portfolio_df.loc[i - 1, "datetime"]
        day_pnl = nominal_ret = 0
        for inst in eligibles:
            units = self.portfolio_df.loc[i - 1, f"{inst} units"]
            if units:
                delta = self.dfs[inst].loc[date, "close"] - self.dfs[inst].loc[prev_date, "close"]
                inst_pnl = delta * units
                day_pnl += inst_pnl
                nominal_ret += self.portfolio_df.loc[i - 1, f"{inst} w"] * self.dfs[inst].loc[date, "ret"]

        capital_ret = nominal_ret * self.portfolio_df.loc[i - 1, "leverage"]
        self.portfolio_df.loc[i, "nominal_ret"] = nominal_ret
        self.portfolio_df.loc[i, "day_pnl"] = day_pnl
        self.portfolio_df.loc[i, "capital"] = self.portfolio_df.loc[i - 1, "capital"] + day_pnl
        self.portfolio_df.loc[i, "capital_ret"] = capital_ret

    def allocate_to_eligibles(self, i, date, eligibles):
        """Allocates the capital to eligible instruments."""
        alpha_scores = {inst: random.random() for inst in eligibles}
        alpha_scores = dict(sorted(alpha_scores.items(), key=lambda item: item[1]))
        alpha_long, alpha_short = self.split_into_long_and_short(alpha_scores)

        non_eligibles = set(self.insts) - set(eligibles)
        for inst in non_eligibles:
            self.portfolio_df.loc[i, f"{inst} units"] = 0
            self.portfolio_df.loc[i, f"{inst} w"] = 0

        nominal_tot = self.calculate_nominal_tot(i, date, alpha_long, alpha_short, eligibles)
        self.portfolio_df.loc[i, "nominal"] = nominal_tot
        self.portfolio_df.loc[i, "leverage"] = nominal_tot / self.portfolio_df.loc[i, "capital"]

    def split_into_long_and_short(self, alpha_scores):
        """Splits the eligible instruments into long and short based on their alpha scores."""
        num_eligibles = len(alpha_scores)
        alpha_long = list(alpha_scores.keys())[-int(num_eligibles / 4):]
        alpha_short = list(alpha_scores.keys())[:int(num_eligibles / 4)]
        return alpha_long, alpha_short

    def calculate_nominal_tot(self, i, date, alpha_long, alpha_short, eligibles):
        """Calculates the total nominal value for the portfolio."""
        nominal_tot = 0
        cap = self.portfolio_df.loc[i, "capital"]
        for inst in eligibles:
            forecast = 1 if inst in alpha_long else (-1 if inst in alpha_short else 0)
            dollar_allocation = cap / (len(alpha_long) + len(alpha_short))
            inst_units = forecast * dollar_allocation / self.dfs[inst].loc[date, "close"]
            self.portfolio_df.loc[i, f"{inst} units"] = inst_units
            nominal_tot += abs(inst_units * self.dfs[inst].loc[date, "close"])

            nominal_inst = inst_units * self.dfs[inst].loc[date, "close"]
            inst_w = nominal_inst / nominal_tot
            self.portfolio_df.loc[i, f"{inst} w"] = inst_w

        return nominal_tot