---
layout: post
title: "Zero trust data pipeline framework"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---




We have all experienced the pain of data loss or data incorrectness in our batch data pipelines. To address this, I designed a framework for ensuring data correctness at every stage of the data pipeline. I call it the **Zero Trust Data Pipeline (ZTDP)**, inspired by Zero Trust Networks (ZTN). However, unlike ZTN, which focuses on data security, ZTDP is solely focused on data correctness — specifically, the confidence that data flowing through the pipeline remains correct, along with a provable method to demonstrate that correctness. When data is corrupted at the producer end, we must be able to catch those corruption cases. If we can prove the data is corrupted, we can identify the issue easily by applying a validation layer on top of it.

In this blog post, I want to demonstrate this concept using practical code and a sample data pipeline. For the example, I will use click-rate data.

The framework is divided into three layers:

-   **Data Layer:** The layer in which data exists in its raw format.
-   **Traceability Layer:** In this layer, the data is aggregated, and each subset of the aggregation is assigned a unique UUID. For each subset, correctness metrics — such as null value counts, 99th percentile values, and other domain-specific measures — are computed.
-   **Correctness Layer:** This layer ensures that the subset metrics can be tracked both within and across layers to measure data correctness throughout the pipeline.





