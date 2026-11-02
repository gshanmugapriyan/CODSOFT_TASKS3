
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_DIR = Path(__file__).resolve().parent


class RecommendationSystem:

    def __init__(self, movies_path, ratings_path):

        # Load datasets
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

        # Validate required columns
        required_movies = {'movie_id', 'title', 'genres'}
        required_ratings = {'user_id', 'movie_id', 'rating'}

        if not required_movies.issubset(self.movies.columns):
            raise ValueError("movies.csv must contain movie_id, title, genres")

        if not required_ratings.issubset(self.ratings.columns):
            raise ValueError("ratings.csv must contain user_id, movie_id, rating")

        # Clean datasets
        self.movies['title'] = self.movies['title'].fillna('').astype(str)
        self.movies['genres'] = self.movies['genres'].fillna('').astype(str)

        self.ratings['rating'] = pd.to_numeric(
            self.ratings['rating'], errors='coerce'
        )

        self.ratings = self.ratings.dropna(
            subset=['user_id', 'movie_id', 'rating']
        )

        self.user_item_matrix = None
        self.user_similarity_df = None
        self.genre_matrix = None
        self.content_similarity = None

        self._prepare_data()

    def _prepare_data(self):

        # Create user-item matrix
        self.user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            aggfunc='mean',
            fill_value=0
        )

        # Calculate user similarity
        similarity = cosine_similarity(self.user_item_matrix)

        self.user_similarity_df = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

        # Prepare genre features
        cv = CountVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False
        )

        self.genre_matrix = cv.fit_transform(
            self.movies['genres'].str.replace('|', ' ', regex=False)
        )

        # Calculate movie similarity
        self.content_similarity = cosine_similarity(
            self.genre_matrix
        )

    def get_user_recommendations(self, user_id, num_recommendations=5):

        if user_id not in self.user_item_matrix.index:
            return []

        # Get user similarity scores
        sim_scores = self.user_similarity_df.loc[user_id]

        # Find movies not watched by the user
        user_ratings = self.user_item_matrix.loc[user_id]

        unrated_movies = user_ratings[
            user_ratings == 0
        ].index

        recommendations = []

        for movie_id in unrated_movies:

            other_ratings = self.user_item_matrix[movie_id]

            # Select users who rated this movie
            relevant_users = other_ratings[
                other_ratings > 0
            ].index

            # Exclude the current user
            relevant_users = relevant_users[
                relevant_users != user_id
            ]

            if len(relevant_users) == 0:
                continue

            # Use only positive similarity scores
            weights = sim_scores.loc[relevant_users]
            weights = weights[weights > 0]

            if weights.empty:
                continue

            # Calculate weighted predicted rating
            ratings = other_ratings.loc[weights.index]

            weighted_sum = (ratings * weights).sum()
            sum_of_weights = weights.sum()

            if sum_of_weights <= 0:
                continue

            predicted_rating = weighted_sum / sum_of_weights

            recommendations.append(
                (movie_id, predicted_rating)
            )

        # Sort by predicted rating
        recommendations.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Get movie details
        result = []

        for movie_id, score in recommendations[:num_recommendations]:

            movie_info = self.movies[
                self.movies['movie_id'] == movie_id
            ]

            if movie_info.empty:
                continue

            result.append({
                'movie_id': movie_id,
                'title': movie_info.iloc[0]['title'],
                'predicted_rating': round(float(score), 2)
            })

        return result

    def get_content_recommendations(
        self, movie_title, num_recommendations=5
    ):

        # Find movie using case-insensitive title matching
        matches = self.movies[
            self.movies['title'].str.casefold()
            == str(movie_title).casefold()
        ]

        if matches.empty:
            return []

        idx = matches.index[0]

        # Calculate similarity scores
        sim_scores = list(
            enumerate(self.content_similarity[idx])
        )

        # Sort by similarity
        sim_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Exclude selected movie
        sim_scores = [
            item for item in sim_scores
            if item[0] != idx
        ]

        sim_scores = sim_scores[:num_recommendations]

        result = []

        for movie_index, score in sim_scores:

            movie = self.movies.iloc[movie_index]

            result.append({
                'movie_id': movie['movie_id'],
                'title': movie['title'],
                'genres': movie['genres'],
                'similarity': round(float(score), 2)
            })

        return result


def main():

    try:

        recommender = RecommendationSystem(
            PROJECT_DIR / 'movies.csv',
            PROJECT_DIR / 'ratings.csv'
        )

        print("\n===== MOVIE RECOMMENDATION SYSTEM =====")

        # Collaborative Filtering
        print("\n--- Collaborative Filtering Recommendations ---")

        user_recs = recommender.get_user_recommendations(
            user_id=1,
            num_recommendations=5
        )

        if user_recs:

            print("Recommendations for User 1:\n")

            for i, rec in enumerate(user_recs, 1):

                print(
                    f"{i}. {rec['title']} "
                    f"(Predicted Rating: {rec['predicted_rating']})"
                )

        else:
            print("No collaborative recommendations found.")

        # Content-Based Filtering
        print("\n--- Content-Based Recommendations for Inception ---")

        content_recs = recommender.get_content_recommendations(
            movie_title='Inception',
            num_recommendations=5
        )

        if content_recs:

            for i, rec in enumerate(content_recs, 1):

                print(
                    f"{i}. {rec['title']} "
                    f"[{rec['genres']}] "
                    f"(Similarity: {rec['similarity']})"
                )

        else:
            print("No similar movies found.")

    except FileNotFoundError as e:
        print(f"Error: CSV file not found - {e}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
