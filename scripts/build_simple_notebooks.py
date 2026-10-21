"""Author the short teaching sequence. Rebuilding clears outputs; rerun all four."""
from pathlib import Path
from textwrap import dedent
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'notebooks/simple'
cells = []


def md(source):
    cells.append(nb.v4.new_markdown_cell(dedent(source).strip()))


def code(source):
    cells.append(nb.v4.new_code_cell(dedent(source).strip()))


def setup():
    code('''
        from pathlib import Path
        import sys
        here = Path.cwd()
        candidates = [here, *here.parents, here/'departure-delay-classification']
        root = next(p for p in candidates if (p/'config/project.json').is_file())
        sys.path.insert(0, str(root/'scripts'))
        import simple_workflow as project
    ''')


def plotting(numpy=False):
    code(('import numpy as np\n' if numpy else '')+'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')''')


def save(name):
    notebook = nb.v4.new_notebook(cells=list(cells), metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13'},
    })
    nb.write(notebook, DEST/name)
    cells.clear()


md('''
    # 01 — Cleaning and EDA

    We ask which flights are eligible and what the original training data show.
    The main cells are short; reusable source decoding and checks live in
    [simple_workflow.py](../../scripts/simple_workflow.py). The label and treatment
    rules remain visible. This notebook has **three figures**; other checks use tables.

    All four notebooks explain and reproduce the completed project. July 2026 is
    already exposed, and will not be used to choose features or models.
    [Run guide](README.md) · [Objectives](../../OBJECTIVES.md)
''')
setup()
plotting()
md('''
    ## Load and inspect the raw flights

    Read the same 17 archived BTS files, keeping ATL, ORD and DFW. The reader preserves
    stable file/CSV-line IDs. `decode_flights` converts dates and finite numbers and
    checks cancellation status and duplicate natural identities. It does not create labels.
    The tables show source counts, sample rows, types and conversion failures.
''')
code('''
    flights, sources = project.read_development_flights()
    flights, conversions = project.decode_flights(flights)
    display(sources, flights.head())
''')
code('''
    flights.info()
    display(conversions)
    display(flights.isna().sum().rename('missing_count').to_frame())
''')
md('''
    ## Create eligibility and the target

    A flight must be noncancelled with a known departure delay. A later diversion does
    not exclude a known departure. The target is 1 for departure delay ≥15 minutes;
    ineligible rows remain unlabeled schedule/history context. These outcome columns
    are excluded from predictors.
''')
code('''
    flights['label_eligible'] = flights['cancelled'].eq(0) & flights['departure_delay_minutes'].notna()
    flights['delayed_departure'] = flights['departure_delay_minutes'].ge(15)
    flights['delayed_departure'] = flights['delayed_departure'].where(flights['label_eligible']).astype('Int8')
''')
md('''
    ## Define time and chronological roles

    The timestamp step converts HHMM to minutes (1330 → 810), localizes with archived
    airport time zones and subtracts two hours for the prediction cutoff. Invalid or
    unresolved daylight-saving times stop execution. Actual departure time is audit/history
    context only. The table separates schedules from eligible queries by original period.
''')
code('''
    flights = project.add_timestamps(flights)
    flights = project.assign_periods(flights)
    display(flights.groupby('split').agg(scheduled=('flight_id', 'size'), eligible=('query_eligible', 'sum')))
''')
md('''
    ## Review distributions and outliers

    Outcome EDA uses eligible January–September 2025 flights. Numerical summaries show
    spread and skewness. The three outlier rules flag unusual values without deleting
    them. Only nonpositive distance/duration becomes missing; valid extremes and the
    label-defining delay remain unchanged. Missing predictors will be imputed during training.
''')
code('''
    train = flights.loc[flights['query_eligible'] & flights['split'].eq('train')].copy()
    numeric = ['distance_miles', 'scheduled_duration_minutes', 'departure_delay_minutes']
    display(train[numeric].describe().T, train[numeric].skew().rename('skewness'))
    display(project.outlier_summary(train, numeric[:2]))
''')
code('''
    for column in numeric[:2]:
        invalid = flights[column].notna() & flights[column].le(0)
        flights[column+'_invalid'] = invalid.astype('int8')
        flights[column] = flights[column].mask(invalid)
    project.check_cleaning(flights)
''')
code('''
    train = flights.loc[train.index].copy()
    plot_rows = train.sample(10000, random_state=42)
    display(train['delayed_departure'].value_counts().rename('count').to_frame())
''')
md('''
    The first figure retains the full sampled delay range: negative values are early
    departures, while the long right tail contains real late departures. The boxplot
    compares distance by outcome and shows substantial class overlap.
''')
code('''
    plot_rows['departure_delay_minutes'].plot.hist(bins=60, title='Training departure delay · full sampled range')
    plt.xlabel('Departure delay (minutes)')
    plt.show()
''')
code('''
    sns.boxplot(data=plot_rows, x='delayed_departure', y='distance_miles')
    plt.title('Training distance by outcome: overlapping distributions')
    plt.show()
''')
md('''
    ## Compare groups and numerical relationships

    Group sizes make the delay rates interpretable. The hour plot reveals schedule-related
    variation; the correlation table checks related numeric predictors without another
    heatmap. These are training associations, not causal explanations or model rankings.
''')
code('''
    for group in ['origin', 'airline']:
        display(train.groupby(group)['delayed_departure'].agg(n='size', rate='mean'))
    display(train[numeric[:2]].corr())
''')
code('''
    hours = train['scheduled_departure_minutes']//60
    train.groupby(hours)['delayed_departure'].mean().plot.bar(title='Training delay rate by local departure hour')
    plt.ylabel('Delayed / eligible')
    plt.show()
''')
md('''
    Cleaning matches the existing handoff, including missingness. Cancelled/unknown-label
    rows stay as context, and no statistical outlier deletion was applied.
    Continue to [02 — Feature Engineering](02_feature_engineering.ipynb).
''')
save('01_cleaning_and_eda.ipynb')

md('''
    # 02 — Feature Engineering

    We build the final inputs in small, named steps. Repeated joins and historical-window
    calculations are reusable functions; six nonlinear formulas remain visible below.
    There are **two figures**. Missingness, join examples and feature summaries use tables.

    A has 26 schedule/time inputs. Model-specific B packages contain 66 (SVM), 72
    (Random Forest) or 78 (LR/KNN) inputs. These are the frozen 2025 choices, not new
    selection decisions. [Function definitions](../../scripts/simple_workflow.py)
''')
setup()
plotting(numpy=True)
md('''
    ## Reuse the same fitting and evaluation identities

    Extend the original clean context through July with the rules demonstrated in 01.
    Keep all schedules for traffic, including cancellations. Select the frozen 40,000
    all-2025 fitting IDs and reproduce the original 10,000-flight July sample.
    Sampling is seed-42 proportional month × outcome, preserving natural prevalence.
''')
code('''
    context = project.load_context()
    features = project.shared_queries(context)
    display(features.groupby('role')['flight_date'].agg(['size', 'min', 'max']))
''')
md('''
    ## Schedule and time features

    Daily coordinates are sin(2π × minutes/1440) and cos(2π × minutes/1440), keeping
    midnight continuous. Month is cyclic too. Second harmonics permit multiple daily
    peaks, and airport interactions allow different time patterns. The example table
    shows minutes and their resulting coordinates; weekday is categorical.
''')
code('''
    features = project.calendar_features(features)
    example = ['scheduled_departure_minutes', 'departure_sin', 'departure_cos', 'month_sin']
    display(features[example].head())
''')
md('''
    ## Match weather to the two-hour cutoff

    Decode archived reports, remove identical payloads and exclude both versions at
    approved conflicting times. Unexpected conflicts stop. Select the latest observation
    ≤cutoff−20 minutes, with age ≤180 minutes at cutoff. For a 12:00 cutoff, 11:40 is
    eligible while 11:45 is too recent. Keep unmatched flights with missing weather.

    Unknown gust/conditions/ceiling remain unknown. Trace precipitation becomes zero plus
    a trace flag. The decoder also calculates dewpoint spread, wind vectors, freezing/wet
    status and inclusive prior-three-hour report summaries. Pressure parsing uses the
    METAR body before RMK. See the [formula guide](README.md#feature-formulas).
''')
code('''
    weather, weather_audit = project.read_weather()
    features = project.match_weather(features, weather)
    example = ['prediction_cutoff_utc', 'weather_observed_utc', 'weather_age_minutes']
    display(features[example].head())
''')
code('''
    fitting = features.loc[features['role'].eq('refit')]
    weather_columns = ['temperature_f', 'wind_knots', 'gust_knots', 'ceiling_ft', 'rain_reported']
    display(fitting[weather_columns].isna().mean().rename('missing_fraction'))
    display(fitting[weather_columns[:3]].describe().T)
''')
md('''
    Missing fields do not imply an unmatched report. The wind boxplot shows class overlap
    on fitting rows; missingness remains explicit until training-only preprocessing.
''')
code('''
    sns.boxplot(data=fitting.sample(5000, random_state=42), x='delayed_departure', y='wind_knots')
    plt.title('Cutoff-matched wind by fitting outcome')
    plt.show()
''')
md('''
    ## Earlier history and scheduled traffic

    Count eligible outcome events in **[cutoff−35 days, cutoff−7 days)** for origin,
    airline/origin, route and flight number. Origin risk is delayed/n; subgroup risk is
    **(delayed + 20 × origin risk)/(n + 20)**. For 2 delays in 5 flights and origin risk
    .30, this gives .32. Counts and no-history flags retain evidence strength.

    Traffic counts all origin schedules in **[departure−60 minutes, departure)**.
    This is scheduled demand assumed known at cutoff, not actual departures. Timely
    outcomes and archived schedule availability remain historical-simulation assumptions.
''')
code('''
    events = context.loc[context['label_eligible']]
    features = project.history_features(features, events)
    features = project.traffic_features(features, context)
    display(features[['history_origin_rate', 'history_flight_n', 'airport_scheduled_departures_prior_60m']].head())
''')
md('''
    The next figure bins earlier airline/origin risk, then measures actual delay rate on
    fitting rows. The association supports examining history as an input; it does not
    establish causality or guarantee separation between classes.
''')
code('''
    fitting = features.loc[features['role'].eq('refit')]
    bins = pd.qcut(fitting['history_airline_origin_rate'], 8, duplicates='drop')
    fitting.groupby(bins, observed=True)['delayed_departure'].mean().plot.bar(title='Earlier risk versus fitting delay rate')
    plt.ylabel('Delayed / eligible')
    plt.tight_layout()
    plt.show()
''')
md('''
    ## Pressure and nonlinear representations

    A2992 means altimeter setting 29.92 inHg. Join pressure, its inclusive three-hour
    change/range, variable-wind flag and gust excess to the already matched report.
    Missing/ambiguous pressure stays missing. Change subtracts the first valid pressure.
    The six formulas below preserve unknown measurements while exposing curvature and
    diminishing scale to the basic classifiers. No new feature selection occurs.
''')
code('''
    features = project.pressure_features(features, weather)
    display(features[['altimeter_inhg', 'altimeter_change3h_inhg', 'variable_wind_flag']].head())
''')
code('''
    features['temperature_centered_square'] = ((features['temperature_f']-32)/10)**2
    features['wind_square'] = features['wind_knots']**2
    features['dewpoint_spread_square'] = features['dewpoint_spread_f']**2
    features['precip_log1p'] = np.log1p(features['precip_inches'])
    features['ceiling_log1p'] = np.log1p(features['ceiling_ft'])
    features['visibility_inverse'] = 1/(1+features['visibility_miles'])
''')
md('''
    ## Save one unscaled feature handoff

    The table contains labels and audit fields too, so 03 selects X through explicit
    frozen allowlists. Save only the existing reproduction Parquet path; the two original
    processed tables stay unchanged. Independent verification compares all 78 selected
    features and missing values on all 50,000 rows with the previous calculations.
''')
code('''
    project.save_features(features, weather_audit)
''')
md('''
    Continue to [03 — Model Training](03_model_training.ipynb).
''')
save('02_feature_engineering.ipynb')

md('''
    # 03 — Model Training

    We visibly instantiate and fit Logistic Regression, KNN, SVM and Random Forest.
    Short A/B loops compare identical fitting flights using different feature allowlists.
    Scikit-learn handles preprocessing inside each pipeline. **This notebook uses tables
    and has no figures.** All settings and thresholds remain the frozen 2025 choices.
''')
setup()
code('''
    from time import perf_counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
''')
md('''
    ## Define training rows, y and chronological roles

    The final refit uses the same 40,000 all-2025 flights. It overlaps development; it is
    not another independent validation set. In each development fold, training precedes
    April/July/October calibration, followed by two scoring months. The tables show those
    boundaries and the earlier development evidence selecting Logistic Regression / B.
''')
code('''
    train = project.training_data()
    y = train['delayed_departure'].astype(int)
    fitted, seconds = {}, {}
    display(project.cohort_table())
''')
code('''
    display(project.development_table())
    display(project.threshold_table())
''')
md('''
    ## Let scikit-learn handle preprocessing

    `model_input` returns X in the frozen column order and an **unfitted sklearn Pipeline**.
    Its `ColumnTransformer` uses `SimpleImputer`, `StandardScaler`, `OneHotEncoder` and
    `MissingIndicator`. No fitting happens in the helper; every `.fit(X, y)` is below.

    LR/KNN retain the separate continuous/binary/indicator recipe. SVM/RF retain the
    original numeric/explicit-flag recipe; RF omits scaling. Each pipeline learns its
    medians, scaling and categories from these 40,000 fitting rows only.
    [Preprocessing implementation](../../scripts/final_modelling.py)

    A has 26 inputs; B uses combined/78 for LR and KNN, current_66/66 for SVM, and
    pressure_wind/72 for RF. The same classifier settings apply to A and B.
''')
md('''
    ## Logistic Regression

    A regularized linear boundary in the encoded features. C=.01 controls regularization.
    The two fits learn separate preprocessing on identical rows. Elapsed time is recorded.
''')
code('''
    lr = LogisticRegression(C=.01, max_iter=3000, random_state=42)
    for fs in ['A', 'B']:
        X, model = project.model_input(train, lr, 'logistic_regression', fs)
        started = perf_counter()
        fitted['logistic_regression__'+fs] = model.fit(X, y)
        seconds['logistic_regression__'+fs] = perf_counter()-started
''')
md('''
    ## KNN

    Predict from 101 nearby training flights using Euclidean distance and distance
    weighting. Scaling matters because the original measurements use different units.
    KNN stores training examples, so scoring can cost more than fitting.
''')
code('''
    knn = KNeighborsClassifier(n_neighbors=101, weights='distance', metric='euclidean',
                              algorithm='brute', n_jobs=2)
    for fs in ['A', 'B']:
        X, model = project.model_input(train, knn, 'knn', fs)
        started = perf_counter()
        fitted['knn__'+fs] = model.fit(X, y)
        seconds['knn__'+fs] = perf_counter()-started
''')
md('''
    ## SVM

    RBF similarity permits a nonlinear boundary. The fixed C=3 and gamma='scale' came
    from development. SVM outputs margins, not probabilities, and uses a frozen margin
    threshold. Its fitting step takes several minutes on this dataset.
''')
code('''
    svm = SVC(C=3, gamma='scale', kernel='rbf', cache_size=512, random_state=42)
    for fs in ['A', 'B']:
        X, model = project.model_input(train, svm, 'svm', fs)
        started = perf_counter()
        fitted['svm__'+fs] = model.fit(X, y)
        seconds['svm__'+fs] = perf_counter()-started
''')
md('''
    ## Random Forest

    Average 200 trees with depth 16, minimum leaf size 5 and balanced class weights.
    Numerical scaling is omitted. All four visible classifier definitions match the
    published settings; the helper stops if a definition differs.
''')
code('''
    rf = RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=2)
    for fs in ['A', 'B']:
        X, model = project.model_input(train, rf, 'random_forest', fs)
        started = perf_counter()
        fitted['random_forest__'+fs] = model.fit(X, y)
        seconds['random_forest__'+fs] = perf_counter()-started
''')
md('''
    ## Save the eight fits with their unchanged thresholds

    October scores from 40k Jan–Sep fits chose maximum-F1 thresholds (largest threshold
    breaks ties). These transfer to all-2025 refits without using July to adjust them.
    The independent verifier checks their derivation. Saving below writes the already
    fitted models separately from the published bundle and displays actual fitting times.
''')
code('''
    display(project.save_models(fitted, seconds))
''')
md('''
    Continue to [04 — Evaluation and Findings](04_evaluation_and_findings.ipynb).
''')
save('03_model_training.ipynb')

md('''
    # 04 — Evaluation and Findings

    This is a **reproduction of the completed July 2026 evaluation**, using the same
    10,000 flights and frozen thresholds. July is already exposed; no new selection is
    made. We use sklearn metrics and display tools, with **four figures**. Other evidence
    uses small tables. Logistic Regression / B remains the validation-selected classifier.
''')
setup()
plotting()
code('''
    import joblib
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay
    bundle = joblib.load(project.OUT/'models.joblib')
    test = pd.read_parquet(project.OUT/'features.parquet').query("role == 'july_reproduction'")
    test = test.sort_values('flight_id').reset_index(drop=True)
    y = test['delayed_departure'].to_numpy(dtype=int)
''')
md('''
    ## O1: compare all four classifiers and simple baselines

    Score the eight fitted pipelines with their frozen thresholds. Add the original
    always-on-time, always-delayed, seed-42 fair-coin and historical-risk baselines.
    The history baseline uses earlier airline/origin risk, origin fallback, then the
    original training constant. `metric_table` uses sklearn's precision, recall, F1,
    F0.5, accuracy, AP, ROC AUC and confusion-matrix functions for each comparison.
''')
code('''
    scores, thresholds, timings = project.score_models(bundle, test)
    project.add_baselines(scores, thresholds, test)
    metrics = project.metric_table(y, scores, thresholds)
    display(metrics[['model_id', 'feature_set', 'f1', 'average_precision', 'precision', 'recall', 'fp', 'fn']].round(4))
''')
code('''
    sns.barplot(data=metrics.query("feature_set != 'baseline'"), x='model_id', y='f1', hue='feature_set')
    plt.title('Same July flights: schedule A versus richer B')
    plt.xticks(rotation=15)
    plt.show()
''')
md('''
    F1 balances delayed-class precision/recall; AP assesses ranking. No-skill AP equals
    prevalence (.3215). Always-on-time accuracy is .6785 but its delayed F1 is zero;
    always-delayed F1 is about .487. A fair coin's expected F1 here is about .391, not
    universally .50. Read all metrics together. KNN's larger July F1 does not reselect it.
''')
code('''
    chosen = bundle['selected_model_id']+'__B'
    prediction = scores[chosen] >= thresholds[chosen]
    print(classification_report(y, prediction, target_names=['On time', 'Delayed'], digits=3))
    ConfusionMatrixDisplay.from_predictions(y, prediction, display_labels=['On time', 'Delayed'], colorbar=False)
    plt.show()
''')
md('''
    The selected LR/B detects 2,971 delays and misses 244, with 4,827 false warnings.
    Its recall .924 comes with precision .381 and F1 .540. The curves compare B rankings
    without changing the warning policy; scores remain uncalibrated (SVM uses margins).
    Scoring times complement the fitting times in 03 when assessing practicality.
''')
code('''
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in [k for k in scores if k.endswith('__B')]:
        PrecisionRecallDisplay.from_predictions(y, scores[key], name=key, ax=ax)
    ax.axhline(y.mean(), color='grey', linestyle='--')
    plt.show()
    display(timings.rename_axis('model_inputs').to_frame())
''')
md('''
    ## O2: does richer information help?

    A/B share classifier settings, fitting IDs and time harmonics. Their thresholds use
    the same October procedure separately. The saved 95% paired day-block intervals
    come from the completed 300-repeat bootstrap over 31 dates, conditional on fixed fits.
    This table reuses those intervals, not independent evidence. LR gains about .038 F1
    and .062 AP over A; RF's F1 interval crosses zero. Individual causal contributions
    of weather, history and traffic are not identified by this joint comparison.
''')
code('''
    gains = project.paired_gains()
    display(gains[['model_id', 'f1_gain', 'f1_low', 'f1_high', 'ap_gain']].round(4))
''')
md('''
    ## O3: where does the selected model fail?

    Compute counts and supported rates by airport, airline and departure band. Precision
    is TP/(TP+FP), recall TP/(TP+FN), and false-positive rate FP/(FP+TN). Small-denominator
    rates stay blank under the published support rules. ORD has lower warning precision;
    evening recall is high alongside nearly universal false-positive rates. These are
    descriptive July patterns, not causal effects.
''')
code('''
    subgroups = project.error_summary(test, scores[chosen], thresholds[chosen])
    columns = ['dimension', 'group_value', 'n', 'precision', 'recall', 'false_positive_rate']
    display(subgroups[columns].round(3))
''')
md('''
    ## O4: is a smaller review list useful?

    Rank the full July cohort by descending score, using the original seeded hash to
    break ties. Review exactly 5%, 10% or 20%. Filtering is not used to recompute quotas.
    At 10%, LR finds 688 delays among 1,000 reviewed (.688 precision, 2.14× random lift);
    historical ranking finds 442. Random expected recall equals the review budget.
''')
code('''
    priority = project.priority_table(test, scores, chosen)
    display(priority.round(3))
    ax = priority.pivot(index='budget', columns='ranking', values='recall').plot(marker='o')
    ax.plot([0, .2], [0, .2], '--', color='grey', label='Random expectation')
    ax.set(xlabel='Review budget', ylabel='Actual delays captured / all actual delays')
    ax.legend()
    plt.show()
''')
md('''
    ## Save the reproduction and interpret its limits

    Ranking concentrates risk but does not establish money saved or delays prevented.
    The broad warning policy still creates many false alerts. One summer month at three
    origins cannot establish year-round operational reliability. Delivery/schedule
    availability is assumed, and threshold transfer after refitting can alter warnings.
    No July threshold was repaired.

    The earlier retuned-66 comparison found LR F1 gain +.015 but negligible AP gain;
    KNN improved modestly, RF's pressure package reduced AP, and SVM retained 66 inputs.
    All original findings remain in the historical notebooks and published report.
    The independent verification script checks every feature, principal prediction,
    metric, quota, subgroup rule, dashboard summary and protected checksum.
''')
code('''
    project.save_evaluation(metrics, subgroups, priority, scores)
''')
md('''
    [Run and verification guide](README.md) · [Published findings](../../reports/final_models/RESULTS.md)
    · [Offline dashboard](../../dashboard/dist/departure_delay_dashboard.html)
    · [Project report](../../reports/final_report/README.md)
''')
save('04_evaluation_and_findings.ipynb')
