# Will you buy an EV?

This notebook tackles [Playground Series - Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9), part of Kaggle's ongoing Playground Series: a set of recurring, lightweight tabular competitions built on synthetic datasets, designed to give the community a fast, approachable way to practice modeling techniques without the complexity of a full research-grade dataset. Each episode centers on a different prediction task; this one asks competitors to predict whether a customer will purchase an electric vehicle (Will_Buy_EV) based on a set of demographic, behavioral, and infrastructure-related features — things like income, environmental concern level, subsidy availability, range anxiety, and charging access.

The task is framed as a binary classification problem, with submissions typically scored using ROC AUC. Because the data is synthetically generated (rather than pulled from a real-world source), it can contain some interesting quirks and artificial patterns that reward careful EDA as much as strong modeling — making it a good sandbox for practicing the full pipeline: exploration, preprocessing, feature engineering, model tuning, and interpretability.

## Upcoming Features
1. **Streamlit** for a hosted UI to test the model
2. **Containerization** with Docker for reproducability

## Out-of-Scope
Productionizing the LGBM model into an API with FastAPI is out-of-scope for this project because if I want to host it on Streamlit Community Cloud, I would have to put the API into a hosted service. Streamlit Community Cloud is free, an API/container service is not.