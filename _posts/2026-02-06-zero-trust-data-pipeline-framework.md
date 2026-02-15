---
layout: post
title: "Zero trust data pipeline framework"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---




We have all experienced the pain of data loss or data incorrectness in our batch data pipelines. To address this, I designed a framework for ensuring data correctness at every stage of the data pipeline. I call it the **Zero Trust Data Pipeline (ZTDP)**, inspired by Zero Trust Networks (ZTN). However, unlike ZTN, which focuses on data security, ZTDP is solely focused on data correctness — specifically, the confidence that data flowing through the pipeline remains correct, along with a provable method to demonstrate that correctness. 

In this blog post, I want to demonstrate this concept using practical code and a sample data pipeline. For the example, I will use click-rate data.

<br>
<br>

The framework is divided into three layers:
---

-   **Data Layer:** <br> The layer in which data exists in its raw format.
<br>
---
-   **Traceability Layer:** <br> In this layer, the data is aggregated, and each subset of the aggregation is assigned a unique hash. The number of rows constributing for the aggregation is constantly tracked. 
<br>

##### One application is for anomaly detection:

| Layer | Rows In | Rows Out | % Retained | Status | Action |
|-------|---------|----------|------------|--------|--------|
| **Layer 1** | **100** | 100 | **100%** | ✅ PASS | `hash=abc123` |
| **Layer 2** | **100** | **10** | **10%** | ❌ **ANOMALY** | **🚨 INVESTIGATE** |
| **Layer 3** | 10 | 10 | 100% | ✅ PASS | `hash=xyz789` |

<br>

##### Few other use cases: 

| | **Use Case** | **Layer 1** | **Layer 2** | **% Retained** | **🚨 Alert** |
|---|--------------|-------------|-------------|----------------|--------------|
| 1 | **Aggregation Drop** | 100 rows | **10 rows** | **10%** | `90% data lost!` |
| 2 | **Join Explosion** | 1K rows | **10K rows** | **1,000%** | `Unexpected duplication!` |
| 3 | **Filter Drift** | 500 rows | **450 rows** | **90%** | `10% silently dropped daily` |
| 4 | **ML Feature Drop** | 10K rows | **8K rows** | **80%** | `2K features discarded!` |
| 5 | **Deduplication Fail** | 2K rows | **1.9K rows** | **95%** | `100 duplicates slipped through` |


<br>

---

-   **Diagnosis Layer:**  <br> This layer ensures that the subset metrics can be tracked both within and across layers to measure data correctness throughout the pipeline.
<br>

**90% data drop → Click for histograms + business rules**

| **Layer** | **Rows In** | **Rows Out** | **% Retained** | **Diagnosis** |
|-----------|-------------|--------------|----------------|---------------|
| Layer 1   | **100**     | 100          | **100%**       | `Normal`      |
| **Layer 2** | **100**   | **10**       | **10%**        | **🚨 90% DROP** |

<br>

### **Click Layer 2 → Auto-Analysis**

| **Amount Range** | **Rows** | **Status**      |
|------------------|----------|-----------------|
| $0-$50           | ██████████ 45 | **Dropped** (<$75) |
| $50-$100         | ████ 25     | **Dropped**     |
| $100+            | ███ 20      | **PASSED ✓**    |

<br>


| **Business Rule**    | **Rows Failed** |
|----------------------|-----------------|
| `amount < $75`       | **70**          |
| `invalid category`   | **20**          |
| **Clean data**       | **10 ✓**        |


---
<br>
<br>

This framework provides a deep view of how data moves through the entire pipeline. It encourages us to stop assuming that our data pipeline always works perfectly and instead maintain layered traceability to detect and understand errors. That’s the main idea behind this blog — introducing the concept of a “zero-trust” data pipeline.

<br>

One challenge with this approach is the additional investment in parallel compute and storage needed to capture and store metadata. However, when balancing cost against data accuracy, correctness and traceability should always take priority.

<br>




