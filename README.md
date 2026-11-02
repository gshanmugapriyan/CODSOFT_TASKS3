# Movie Recommendation System

## CODSOFT Internship – Task 4

### Project Description
This project is a Movie Recommendation System developed using Python. It suggests movies to users based on their preferences using Collaborative Filtering and Content-Based Filtering techniques.

### Technologies Used
- Python
- Pandas
- Scikit-learn
- Cosine Similarity

### Recommendation Techniques

**1. Collaborative Filtering**
- Uses user movie ratings.
- Identifies users with similar interests.
- Recommends movies based on ratings from similar users.

**2. Content-Based Filtering**
- Uses movie genres.
- Compares movies using Cosine Similarity.
- Recommends movies with similar genres.

### Project Structure

```text
recommendation_system/
│
├── recommender.py
├── movies.csv
├── ratings.csv
└── README.md
```

### Installation

Install the required Python libraries:

```bash
pip install pandas scikit-learn
```

### How to Run

Execute the following command:

```bash
python recommender.py
```

### Output
The system provides:
- Top 5 movie recommendations for a selected user.
- Predicted movie ratings using Collaborative Filtering.
- Similar movie recommendations based on genres.
- Similarity scores using Content-Based Filtering.

### Learning Outcomes
- Understanding recommendation systems.
- Implementing Collaborative Filtering.
- Implementing Content-Based Filtering.
- Working with Pandas and Scikit-learn.
- Understanding Cosine Similarity.

### Author
**Shanmugapriyan G**

CODSOFT Internship – Task 4
