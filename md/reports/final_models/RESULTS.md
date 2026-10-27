# Current model findings

Train on 40,000 flights from all of 2025; evaluate the same 10,000 July 2026 flights. Of those test flights, 3,215 departed at least 15 minutes late.

| Richer B model | Accuracy | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 49.29% | 38.10% | 92.41% | 0.540 | 0.566 |
| KNN | 54.25% | 40.07% | 85.35% | 0.545 | 0.532 |
| SVM | 61.18% | 43.49% | 69.24% | 0.534 | 0.513 |
| Random Forest | 52.37% | 39.26% | 87.96% | 0.543 | 0.540 |

Logistic Regression B remains the classifier selected using chronological development within 2025. KNN's slightly higher July F1 does not change that earlier choice.

The selected model detects 2,971 delays and misses 244, while raising 4,827 false warnings. Its high recall therefore comes with low warning precision. Always predicting on time achieves 67.85% accuracy but detects zero delays; always predicting delayed achieves F1 0.487.

Adding weather, earlier-delay history and traffic improves Logistic Regression from A's F1 0.501/AP 0.504 to B's F1 0.540/AP 0.566. This combined comparison does not isolate the contribution of each feature family.

The completed study's top-10% Logistic Regression review list contains 688 delays among 1,000 flights, versus 442 for historical risk. The final report retains the fuller findings and limitations. One summer month at three hubs does not establish year-round or live operational reliability.

[Current metrics](../../../reports/simple_reproduction/metrics.json) · [Evaluation notebook](../../../notebooks/simple/04_evaluation_and_findings.ipynb) · [Final report](../../../reports/final_report/output/pdf/predeparture_flight_delay_report.pdf)
