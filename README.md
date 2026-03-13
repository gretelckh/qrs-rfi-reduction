# qrs-rfi-reduction
# Transaction Context Enhancement for Reducing RFI in Financial Payment Systems

This repository contains the implementation and dataset for the QRS 2025 submission.

## Experiment Results
- Baseline (Amount Only): Accuracy = 0.50
- Proposed Method (Amount + Transaction Reason): Accuracy = 0.78 (+28%)

## How to Run
python run.py

## Abstract
Modern financial payment systems often generate unnecessary RFI (Request for Information) inquiries due to the lack of transaction context.
We propose a lightweight, software-engineering-friendly approach that uses transaction reason text to improve risk prediction and reduce RFIs.
Experiments on a high-fidelity synthetic dataset show significant performance improvement.
