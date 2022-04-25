import numpy as np
import pandas as pd
import numpy_financial as npf
import matplotlib.pyplot as plt

import modules.IR as IR


class MBS_pricing:
    def __init__(self, portfolio, flows, flow_tranchwise):
        self.portfolio = portfolio
        self.flows = flows
        self.flow_tranchwise = flow_tranchwise

    def import_MBS_Cashflows(self):
        loan_cashflows = self.flows

        loan_total_payments_futurevalue = loan_cashflows['total_payments']

        discounting_rate = IR.CIR()[0]
        size = len(loan_total_payments_futurevalue)

        self.discounting_rate1 = discounting_rate[:size]

        self.tranch_cashflow = (self.flow_tranchwise)[:-1]
        size4 = len(self.tranch_cashflow)
        self.discounting_rate4 = discounting_rate[:size4]

    def caculate_tranch_value(self):
        discounted_tranch_senior = pd.DataFrame()
        discounted_tranch_junior = pd.DataFrame()
        discounted_tranch_total = pd.DataFrame()

        for i, r in enumerate(self.discounting_rate4):
            discounted_tranch_senior[i] = (npf.pv(self.discounting_rate4[i], self.tranch_cashflow['period'], 0,
                                                  -self.tranch_cashflow['cashflow_senior']))
            discounted_tranch_junior[i] = (npf.pv(self.discounting_rate4[i], self.tranch_cashflow['period'], 0,
                                                  -self.tranch_cashflow['cashflow_junior']))
            discounted_tranch_total[i] = (npf.pv(self.discounting_rate4[i], self.tranch_cashflow['period'], 0,
                                                 -self.tranch_cashflow['total_cashflow']))

        self.npv_mbs_senior = np.sum(discounted_tranch_senior)
        self.npv_mbs_junior = np.sum(discounted_tranch_junior)
        self.npv_mbs_total = np.sum(discounted_tranch_senior)

    def plot_mbs_value(self):
        x4 = self.discounting_rate4

        plt.scatter(x4, self.npv_mbs_total)
        plt.title("Price-Yield Relationship - Total MBS")
        plt.xlabel("Interest Rate")
        plt.ylabel("Value")
