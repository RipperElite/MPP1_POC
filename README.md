# MPP-1: Adversarial Benchmarking POC

This repository contains the Proof of Concept (POC) codebase for my Master's Project Phase 1 (MPP-1) at Mahatma Gandhi University. It evaluates the vulnerability of rewrite-based AI text detectors against gradient-based discrete optimization.

## Project Overview
A foundational advancement in the rewrite-based ai text detection landscape is the Learning2Rewrite (L2R) framework. L2R shifts from traditional classification by relying on a behavioral quirk of LLMs ; when an LLM is prompted to rewrite a text, it significantly alters human written texts but makes minimal changes to the text it originally generated.
L2R fine-tunes a model to exaggerate this behavior, establishing a threshold based on the "edit distance" between the original input and the rewritten output. By measuring this structural variation, L2R establishes a massive edit distance for human text and a minor edit distance for AI text. Because of this unique methodology, L2R claims exceptional robustness, reportedly outperforming current state-of-the-art detectors by up to 48.66% under standard adversarial attacks.

While L2R claims high resilience against basic "Decoherence" or standard paraphrasing attacks, its reliance on a static, mathematical edit distance metric introduces a vulnerability. Standard attacks fail because they apply blind, and generalized obfuscation.

However against an adaptive White-Box attack this defense can be mathematically broken.

This POC demonstrates that by employing the Greedy Coordinate Gradient (GCG) optimization framework, we can project 4,096-dimensional model gradients into a discrete vocabulary space. This generates precise, universal adversarial suffixes that can artificially manipulate the resulting edit distance,forcing the target LLM to bypass the L2R detection thresholds.
## Experimental Findings
Initial mini-test evaluations were executed on an A100 GPU cluster targeting the L2R framework which yielded the following empirical findings:
* **Attack Success Rate (ASR):** 90.0%
* **Average Transfer L2R Score:** 0.6615
* **Baseline Detection Threshold:** 0.5764
* **Vulnerability Analysis:** The findings demonstrate that standard rewrite-based detectors rely heavily on structural edit-distance metrics, which are highly susceptible to discrete token optimization. 


##  Challenges & Methodological Adaptations
During the development of the PoC, several technical hurdles were addressed to stabilize the adversarial optimization pipeline:
* **Short Sequence Instability:** Initial iterations revealed that running GCG optimization on very short text samples caused severe tensor shape mismatches during batching and disproportionately skewed the target L2R edit-distance metrics.
* **Dynamic Dataset Filtering:** To overcome this, I created a strict filtering mechanism within the data processing loop. By setting a `min_token_count` threshold, I systematically discarded short sentences and re-calculated the new average token length for the dataset. This stabilization ensured consistent gradient extraction and prevented Out-Of-Memory (OOM) spikes during backpropagation.
* **Discrete Token Alignment:** Generating adversarial suffixes required rigorous checks to ensure only valid, printable tokens were passed into the final string, preventing tokenization collapse before the L2R evaluation step.

##  Repository Structure
* `eval_mpp1_benchmark.py`: The core benchmarking script. It iterates over the dataset, filters samples by sequence length, computes the cross-entropy loss over target token sequences, and evaluates the L2R evaluation shift.


##  Execution Instructions
This codebase is designed to run headlessly on a DGX cluster (or similar HPC environment) using PyTorch.

1.  **Environment Setup**
    Ensure you have an active conda/venv environment with PyTorch and standard NLP libraries installed.
2.  **Dataset Location**
    Place your baseline dataset (`ag_news_filtered_baseline.csv`) inside the `src/data/` folder.
3.  **Run the Benchmark**
    Execute the Python script via terminal:
    ```bash
    python eval_mpp1_benchmark.py
    ```

##  References
* Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.*
* Hao et al. (ACL 2025)*Learning to Rewrite (L2R) Framework for AI-Generated Text Detection.*
---
