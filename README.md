# FairBayes-DDP+
Codebase and Experiments for "Minimax Optimal Fair Classification with Bounded Demographic Disparity"


# Codebase Overview
This repository provide simulation codes of our paper "Minimax Optimal Fair Classification with Bounded Demographic Disparity". Simulation codes for synthetic data is contained in document ``Synthetic_data'' and  Simulation codes for real data is contained in document ``Real_data''.


# Algorithms Considered in Read data analysis
This repository provide python realization of 5 benchmark methods of fair classification.

--FairBayes-DDP+: X. Zeng, G. Cheng, and E. Dobriban. Minimax Optimal Fair Classification with Bounded Demographic Disparity.

--KDE based constrained optimization (KDE)：
  J. Cho, G. Hwang, and C. Suh. A fair classifier using kernel density estimation.
  
--Adversarial Debiasing (ADV):
  B. H. Zhang, B. Lemoine, and M. Mitchell. Mitigating unwanted biases with adversarial learning.
  
--Post-processing through Flipping (FFP)
  W. Chen, Y. Klochkov, and Y. Liu. Post-hoc bias scoring is optimal for fair classification.
  
--Post-processing through Optimal Transport (PPOT)
  R. Xian, L. Yin, and H. Zhao. Fair and optimal classification via post-processing.

# Related Works and Repositories
This repository has draw lessons from other open resourses. 

--Codes for ADV take inspiration from the AI Fairness 360 platform:  https://github.com/Trusted-AI/AIF360;

--Codes for KDE follows the original code provided by: J. Cho, G. Hwang, and C. Suh. A fair classifier using kernel density estimation;

--Codes for PPOT take inspiration from: https://github.com/rxian/fair-classification;

--Codes for PPF take inspiration from the paper:   W. Chen, Y. Klochkov, and Y. Liu. Post-hoc bias scoring is optimal for fair classification.


# Data
This repository uses the AdultCensus. It can be found in the Datasets folder and are loaded using dataloader.py.
