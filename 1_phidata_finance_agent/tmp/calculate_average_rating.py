import pandas as pd

def calculate_average_rating(file_path):
    data = pd.read_csv(file_path)
    average_rating = data['Rating'].mean()
    return average_rating

if __name__ == "__main__":
    file_path = "https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv"
    average_rating = calculate_average_rating(file_path)
    print(average_rating)