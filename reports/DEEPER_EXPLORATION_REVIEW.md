# Deeper exploration: revised initial-training period

Updated 19 October 2026 from the executed [supplement](../notebooks/02a_deeper_training_exploration.ipynb). Current exploration covers January–September 2025: 698,670 eligible departures from 712,534 schedules. No validation or 2026 outcome relationships were explored. The supplement reads existing handoffs and verifies their checksums remain unchanged during its execution; it trains no classifier.

## Representation corrections now applied

| Finding | Current action and audit |
|---|---|
| Later months would become all-zero one-hot categories | A/B now use fixed month sine/cosine; raw month remains for reporting. This corrects representation, not evidence for unseen seasons. |
| Five identical condition-absence coordinates | One explicit condition flag remains; repeated automatic copies and the redundant ceiling indicator are removed. Wind/gust indicators remain. Numeric preview: 35 rather than 40 columns. |
| Four cloud slots omit extra body layers | All METAR body layers are parsed before RMK. Recovered ceilings affect 719 training flights; zero known body ceilings remain omitted. |

These fixes need no additional source data. [The preparation report](PREPARATION_TIMELINE_UPDATE.md) documents the executed core changes.

## Airline composition can reverse rankings

Crude F9 delay risk is 34.58% versus AA at 33.64%. Standardizing both to their shared origin × month × departure-band composition gives 29.44% versus 33.37%. The comparison retains 99.4% of their combined population; the reversal persists at minimum group counts of 10 and 50. Route and other differences remain; this is not a causal ranking of airline quality.

## Weather associations need careful interpretation

The positive-versus-zero accumulation contrast is 13.26 percentage points in the morning and 27.91 in the evening. Relative risks are similar (1.86 versus 1.83), so the apparent interaction depends on the scale.

Latest-zero flights with earlier positive accumulation have about 39.5% delay versus 23.9% without it. Broad shared-mix adjustment gives a 14.17-point gap. Restricting to the same airport/day/time gives only 0.35 points, with 30,974 retained flights (4.9% of the latest-zero comparison population). The latter changes coverage and cannot establish a universal effect, but weakens a simple recovery-time explanation. Shared disrupted days and other weather may drive the broad gap. A compact prior-weather feature remains a validation candidate.

Accumulation totals must not be summed across overlapping reports. [IEM explains reporting caveats](https://mesonet.agron.iastate.edu/onsite/news.phtml?id=1434). At exactly 10 miles, visibility is stored for 87.7% of training flights; [NOAA documents 10SM as 10 miles or more](https://www.weather.gov/media/asos/aum-toc.pdf#page=36), rather than identical true visibility.

Absent condition codes alone do not establish station failure. [NOAA conventions](https://www.weather.gov/media/asos/aum-toc.pdf#page=49) and [IEM's M code](https://mesonet.agron.iastate.edu/request/download.phtml) require interpreting source fields carefully. The project retains explicit unknown flags.

## Traffic and sudden bad days

High-versus-low traffic differs by 5.46 percentage points crudely and 4.27 after shared origin/month/hour adjustment. The adjusted comparison retains 184,278 flights across 223 groups (46.3% of the high/low population). Absolute volume is not airport capacity; relative-to-typical traffic remains a candidate.

At ATL on 10 January, 643 of 764 schedules were cancelled. Of the remaining 121 eligible departures, 112 were delayed (92.6%); lagged history averaged 22.7%. Across airport-days, observed operated-delay rate and lagged history correlate about 0.25. The worst 10% of airport-days account for 20.1% of delays and 9.4% of operated flights. Slow history captures background risk but can miss shocks.

The target is delay among noncancelled departures with known labels. There are 647 cancelled training records with delay values; explicit cancellation status correctly excludes them. A cancellation is not an on-time query.

## Other useful limits

- Four destinations, covering 67 training flights, are absent from the 20,000-flight initial sample. Keep the rows and use the declared unknown-category rule.
- Distance and duration correlate about 0.986; compare their incremental value later rather than assuming both are essential.
- 698,670 training flights share 18,231 weather reports: median 41 flights per report, maximum 120. Flights are not independent weather experiments.
- Dew-point spread is associated with low visibility without a monotonic delay pattern. Existing raw dew point can support a later candidate; extra data is not required for it.
- Valid signed delays range from −60 to 2,364 minutes. Keep legitimate extremes and exclude actual delay from predictors.

The formal objectives still require fitted model comparisons, prediction errors and ranking results. Select using October–December 2025 validation, freeze choices, refit on the full-2025 sample and evaluate once on Q1 2026. The exploratory associations above do not demonstrate predictive gains.
