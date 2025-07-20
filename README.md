# Feature Prioritization using RICE & ICE Scoring

## Overview  
This project helps prioritize product features using RICE and ICE scoring models, enhanced with sensitivity analysis, customizable weights, clustering, and red-flag detection.

## Objective  
- Score features using RICE and ICE metrics  
- Analyze confidence sensitivity  
- Customize weights for decision-making  
- Cluster features based on strategic profiles  
- Detect high-risk or low-ROI features

## Dataset & Inputs  
- Synthetic dataset (`feature_prioritization.csv`) with columns:
  - `Feature`, `Reach`, `Impact`, `Confidence`, `Effort`, `Ease`

## Technologies Used  
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## How to Run  
```bash
python feature_prioritization.py
```

## Workflow  
- Load dataset  
- Compute RICE and ICE scores  
- Run sensitivity analysis  
- Apply user-defined weighted scoring  
- Perform KMeans clustering  
- Detect red flags  
- Visualize outputs

## Results  
- Top 20 features by RICE and ICE  
- Sensitivity plots for confidence  
- Clustering scatterplots  
- Correlation heatmap

## Feature Importance  
- Uses RICE and ICE as core importance metrics  
- Sensitivity and cluster profiles for deeper insight

## Key Takeaways  
- Helps teams prioritize effectively under constraints  
- Surfaces high-risk or low-confidence features  
- Enables data-driven product decisioning

## Future Enhancements  
- Interactive Streamlit UI  
- Roadmap generation from priority queue  
- Integration with project management tools
