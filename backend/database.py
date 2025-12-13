"""
MongoDB database connection and utilities
"""
from pymongo import MongoClient
from config import MONGODB_URI, MONGODB_DB
import pandas as pd

class Database:
    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB]
        
        # Collections
        self.movies = self.db['movies']
        self.ratings = self.db['ratings']
        self.users = self.db['users']
        self.metrics = self.db['metrics']
        
    def seed_data(self, movies_df, ratings_df):
        """Seed database with movies and ratings data"""
        print("Seeding database...")
        
        # Clear existing data
        self.movies.delete_many({})
        self.ratings.delete_many({})
        self.users.delete_many({})
        
        # Insert movies
        movies_records = movies_df.to_dict('records')
        if movies_records:
            self.movies.insert_many(movies_records)
            print(f"Inserted {len(movies_records)} movies")
        
        # Insert ratings
        ratings_records = ratings_df.to_dict('records')
        if ratings_records:
            # Insert in batches to avoid memory issues
            batch_size = 10000
            for i in range(0, len(ratings_records), batch_size):
                batch = ratings_records[i:i+batch_size]
                self.ratings.insert_many(batch)
            print(f"Inserted {len(ratings_records)} ratings")
        
        # Create unique users collection
        unique_users = ratings_df['userId'].unique()
        users_records = [{'userId': int(uid)} for uid in unique_users]
        if users_records:
            self.users.insert_many(users_records)
            print(f"Inserted {len(users_records)} users")
        
        # Create indexes for better performance
        self.movies.create_index([('movieId', 1)])
        self.ratings.create_index([('userId', 1)])
        self.ratings.create_index([('movieId', 1)])
        self.users.create_index([('userId', 1)])
        
        print("Database seeded successfully")
    
    def save_metrics(self, metrics):
        """Save evaluation metrics"""
        self.metrics.delete_many({})  # Clear old metrics
        self.metrics.insert_many(metrics)
        print("Metrics saved to database")
    
    def get_movie(self, movie_id):
        """Get movie by ID"""
        return self.movies.find_one({'movieId': movie_id})
    
    def get_user(self, user_id):
        """Get user by ID"""
        return self.users.find_one({'userId': user_id})
    
    def search_movies(self, query, limit=50):
        """Search movies by title or genre"""
        regex = {'$regex': query, '$options': 'i'}
        results = self.movies.find({
            '$or': [
                {'title': regex},
                {'genres': regex}
            ]
        }).limit(limit)
        return list(results)
    
    def get_user_ratings(self, user_id):
        """Get all ratings for a user"""
        return list(self.ratings.find({'userId': user_id}))
    
    def get_statistics(self):
        """Get dataset statistics"""
        stats = {
            'total_movies': self.movies.count_documents({}),
            'total_ratings': self.ratings.count_documents({}),
            'total_users': self.users.count_documents({}),
            'avg_rating': 0,
            'top_genres': []
        }
        
        # Calculate average rating
        pipeline = [
            {'$group': {'_id': None, 'avg': {'$avg': '$rating'}}}
        ]
        result = list(self.ratings.aggregate(pipeline))
        if result:
            stats['avg_rating'] = result[0]['avg']
        
        # Get top genres
        all_movies = list(self.movies.find({}, {'genres': 1}))
        genre_counts = {}
        for movie in all_movies:
            genres = movie.get('genres', '').split('|')
            for genre in genres:
                genre = genre.strip()
                if genre and genre != '(no genres listed)':
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        stats['top_genres'] = sorted(
            genre_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.client.close()

# Global database instance
db = Database()
