# Movie Recommendation System

A simple Python recommendation system that suggests movies using collaborative
filtering and content-based filtering.

## Features

- **User-based collaborative filtering:** Finds users with similar rating
  patterns and recommends movies the target user has not rated. Similarity is
  measured with cosine similarity, and ratings from similar users are combined
  into predicted scores.
- **Content-based filtering:** Recommends movies with genres similar to a
  selected movie. Genre overlap is measured with cosine similarity.
- Handles missing or invalid ratings and matches movie titles without
  case-sensitivity.

## Project files

- `recommender.py` — recommendation logic and runnable example.
- `movies.csv` — movie IDs, titles, and pipe-separated genres.
- `ratings.csv` — user IDs, movie IDs, and ratings.

The CSV files must include these columns:

| File | Required columns |
| --- | --- |
| `movies.csv` | `movie_id`, `title`, `genres` |
| `ratings.csv` | `user_id`, `movie_id`, `rating` |

## Requirements

- Python 3
- pandas
- scikit-learn

Install the dependencies from a terminal:

```powershell
python -m pip install pandas scikit-learn
```

## Run

From the `recommendation_system` folder:

```powershell
python .\recommender.py
```

Or, from the parent `CODSOFT_TASKS3` folder:

```powershell
python .\recommendation_system\recommender.py
```

The script loads the CSV files relative to its own location and prints
collaborative recommendations for User 1 and content-based recommendations
for *Inception*.

If `python` is not the interpreter with the dependencies installed, run the
script with that interpreter's full path instead.

## Use the recommendation methods

```python
from recommender import RecommendationSystem

recommender = RecommendationSystem("movies.csv", "ratings.csv")

user_recommendations = recommender.get_user_recommendations(
    user_id=1,
    num_recommendations=5,
)

movie_recommendations = recommender.get_content_recommendations(
    movie_title="Inception",
    num_recommendations=5,
)

print(user_recommendations)
print(movie_recommendations)
```

Each method returns a list of dictionaries. Collaborative recommendations
include `movie_id`, `title`, and `predicted_rating`; content-based
recommendations include `movie_id`, `title`, `genres`, and `similarity`.
An empty list is returned if no recommendations are available or the requested
user/movie is not found.
