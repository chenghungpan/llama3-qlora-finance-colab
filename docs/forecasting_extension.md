# Optional next project: does sentiment help forecast returns?

The delivered project is complete for sentiment classification. A credible forecasting experiment needs additional data; this dataset alone cannot establish one.

## Required data contract

For every example, collect a licensed source post/news item, a verified publication timestamp with timezone, the security or securities mentioned, the time the data actually became available, and adjusted market prices on a documented exchange calendar. Ambiguous ticker mappings should be excluded or explicitly handled.

Choose the prediction target before training: for example, next-session close-to-close return or direction. Specify exactly when a prediction is made and which information was available then. Same-day close cannot be an input if the prediction is supposedly made before the close.

## Evaluation design

1. Split chronologically and use walk-forward validation. Fit preprocessing, sentiment adaptation, and downstream models using past data only.
2. Aggregate sentiment over a fixed window that ends before the decision time. Check market holidays, after-hours releases, timezones, and overlapping return windows.
3. Compare price-only/majority baselines with the same model plus sentiment features. Include a no-change price baseline when predicting prices.
4. Evaluate out-of-sample error and uncertainty. A direction classifier needs class metrics; a numeric return forecast needs error metrics.
5. If simulating trades, define executable entry/exit prices, fees, slippage, exposure limits, and delisted securities. Report turnover and drawdowns, not just accuracy.
6. Check pretrained-model contamination: an old historical split can still leak through the language model's original training. Prefer an evaluation period after the model's training-data cutoff, with documented limitations.

This is a research plan, not runnable forecasting code or a claim that language-model sentiment improves trading returns. The repository intentionally does not fabricate timestamps or returns for the sentiment dataset.
