# Simple Recommendation System

This project implements a basic recommendation system using two different techniques:

1.  **Collaborative Filtering (User-Based):** Suggests movies to a user by finding similar users and recommending movies they liked but the target user hasn't seen yet. It uses cosine similarity to calculate user similarities and weighted averages for rating predictions.
2.  **Content-Based Filtering:** Suggests movies similar to a specific movie based on shared genres. It uses `CountVectorizer` and cosine similarity on movie genres.

## Project Structure

- `movies.csv`: Contains movie IDs, titles, and genres.
- `ratings.csv`: Contains user ratings for various movies.
- `recommender.py`: The main Python script containing the `RecommendationSystem` class and example usage.

## How to Run

1.  Ensure you have the required libraries installed:
    ```bash
    pip install pandas numpy scikit-learn
    ```
2.  Run the script:
    ```bash
    python recommender.py
    ```

## Example Usage

```python
recommender = RecommendationSystem('movies.csv', 'ratings.csv')

# Get collaborative recommendations for User 1
user_recs = recommender.get_user_recommendations(user_id=1)

# Get content-based recommendations for 'Inception'
content_recs = recommender.get_content_recommendations('Inception')
```
