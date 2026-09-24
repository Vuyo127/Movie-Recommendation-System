Movie Recommendation System
A movie recommender built in a Jupyter notebook. It cleans the raw movie and ratings data, then builds and compares two approaches:
- Content-based filtering: recommends movies whose genres and plot overviews are similar to a movie you pick.
- Collaborative filtering: recommends movies using only patterns in user ratings, with two models (item-based k-NN and SVD).

Project Structure
Movie-Recommendation-System/
├── Movie_Recommendation_Data_Cleaning.ipynb   # cleaning + both recommender models
├── data/                                      # CSV files 
└── README.md

Data
The notebook expects three CSV files (from The Movies Dataset on Kaggle):

File	Used for
movies_metadata.csv	Titles, genres, overviews, vote counts, release dates
ratings_small.csv	User ratings (about 100k ratings, 0.5 to 5 stars)
links.csv	Maps MovieLens movieId to TMDB tmdbId

The data files are not stored in this repo. Download them and place them where the notebook loads them.

What the notebook does
1. Data cleaning
Removes duplicate rows and duplicate movie IDs.
Converts IDs, dates and numeric columns to proper types, coercing bad values to missing.
Removes invalid vote values and empty titles.
Drops budget and revenue (too many zeros or missing values) and fills missing runtime with the median.
Cleans the ratings data (types, range check, duplicates, timestamps).
Links the two datasets: ratings use MovieLens IDs while movies use TMDB IDs, so links.csv is used to map between them.
2. Content-based model
Parses the genres field (stored as a string of dicts) into genre lists.
Builds TF-IDF vectors for genres and for plot overviews, weights them, and combines them.
Ranks movies by cosine similarity to the chosen movie. Similarity is computed one movie at a time to avoid building a huge 45k x 45k matrix.
3. Collaborative model
Keeps only ratings that link to a known movie and movies with at least 5 ratings.
Splits the ratings 80/20 into train and test sets.
Centers each user's ratings on their own average.
Item-based k-NN: finds the 30 most similar movies by rating pattern and predicts from the user's ratings of those neighbors.
SVD: factorizes the user-movie matrix into hidden taste factors and uses them to predict missing ratings.

Results

Error is measured as RMSE on held-out ratings (lower is better):

Model	RMSE
Global mean baseline	1.040
User mean baseline	0.951
Item-based k-NN	0.956
SVD (10 factors)	0.919
SVD is the best model, about 3.4% better than the user-mean baseline.
Item-based k-NN does not beat the user-mean baseline on this small dataset because most movie pairs share very few raters.
SVD error rises as factors increase past 10, a sign of overfitting.

Limitations
SVD is applied to a matrix where missing ratings are treated as zeros after centering. For users with few ratings the predictions collapse toward that user's average, so recommendations are barely personalized for them.
Dense matrices work for ratings_small.csv but will not scale to the full ratings file.
RMSE measures rating accuracy, not the quality of a top-10 list. Ranking metrics such as precision@k have not been added yet.
Content-based results rely on genres and overview text only, so they can miss things like cast, director or keywords.
