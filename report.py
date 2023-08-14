import pandas as pd

import config as cfg
from metric_builder import CalculateMetric, Metric
from stattests import TTestFromStats, calculate_linearization, calculate_statistics


class Report:
    def __init__(self, report):
        self.report = report


def get_critetia_res(estimator: str, stats):
    if estimator == "t_test":
        ttest = TTestFromStats()
        criteria_res = ttest(stats)
    else:
        raise ValueError("estimator: Unknown estimator")

    return criteria_res


class BuildMetricReport:
    def __call__(
        self, calculated_metric: CalculateMetric, metric_items: Metric
    ) -> Report:
        cfg.logger.info(f"{metric_items.name}")

        df_ = calculate_linearization(calculated_metric)
        stats = calculate_statistics(df_, metric_items.type)

        criteria_res = get_critetia_res(metric_items.estimator, stats)

        report_items = pd.DataFrame(
            {
                "metric_name": metric_items.name,
                "mean_0": stats.mean_0,
                "mean_1": stats.mean_1,
                "var_0": stats.var_0,
                "var_1": stats.var_1,
                "delta": stats.mean_1 - stats.mean_0,
                "lift": (stats.mean_1 - stats.mean_0) / stats.mean_0,
                "pvalue": criteria_res.pvalue,
                "statistic": criteria_res.statistic,
            },
            index=[0],
        )

        return Report(report_items)


def build_experiment_report(df, metric_config):
    reports = []

    for metric_params in metric_config:
        metric_parsed = Metric(metric_params)
        calculated_metric = CalculateMetric(metric_parsed)(df)
        metric_report = BuildMetricReport()(calculated_metric, metric_parsed)
        reports.append(metric_report.report)

    return pd.concat(reports)
