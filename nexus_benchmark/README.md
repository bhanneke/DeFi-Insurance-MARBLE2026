# Nexus Mutual premium benchmark

`nexus_premium_analysis.py` computes the realized annualized premium rates on
Nexus Mutual quoted in Section 6 ("Empirical benchmark") of the paper:

> Across 10,893 covers written between July 2019 and June 2025 (total cover
> volume $5.9B), the median realized premium is 2.6% per year of cover
> (mean 4.2%; 5–95%: [1.30%, 13.40%]).

The underlying input, `nexus_combined_data.csv` (~4MB), is an export of
on-chain Nexus Mutual V1/V2 cover data (compiled via Dune) and is **not
committed** to this repository; it is available from the author on request.
The script documents the aggregation methodology: V2 rows are per
staking-pool allocation, so premiums are summed per `cover_id` before rates
are computed, and rates above 100%/yr are dropped as data errors.
