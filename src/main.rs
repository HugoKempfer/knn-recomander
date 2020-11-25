use rustml::matrix::*;
use rustml::Distance;
use rustml::Euclid;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::collections::HashMap;
use std::collections::HashSet;

use std::convert::TryInto;
use std::vec::*;

#[derive(Debug, Deserialize)]
struct Movie {
    movie_id: i32,
    title: String,
    genres: HashSet<String>,
}

#[derive(Debug, Deserialize)]
struct Rating {
    user_id: i32,
    movie_id: i32,
    rating: f32,
    timestamp: String,
}

fn get_csv_values<T>(path: &str) -> Vec<T>
where
    T: DeserializeOwned,
{
    let mut reader = csv::Reader::from_path(path).expect("Can't open csv file");
    let mut values: Vec<T> = vec![];

    for entry in reader.deserialize() {
        values.push(entry.unwrap());
    }
    values
}

// Key = movie id | vec offset = userId | vec value = rate
fn get_rating_index(ratings: &Vec<Rating>, movies: &Vec<Movie>) -> HashMap<i32, Vec<f32>> {
    let mut index = HashMap::new();
    let mut user_rates: HashMap<i32, HashMap<i32, f32>> = HashMap::new(); // userid | movie_id | rate
    let nb_users: usize = ratings.last().expect("No rating provided").user_id as usize;

    for rate in ratings {
        let user_rates = match user_rates.contains_key(&rate.user_id) {
            true => user_rates.get_mut(&rate.user_id).unwrap(),
            false => {
                user_rates.insert(rate.user_id, HashMap::new());
                user_rates.get_mut(&rate.user_id).unwrap()
            }
        };
        user_rates.insert(rate.movie_id, rate.rating);
    }
    for movie in movies {
        index.insert(movie.movie_id, Vec::with_capacity(nb_users));
        let movie_rates = index.get_mut(&(movie.movie_id)).unwrap();
        for user_id in 0..nb_users {
            movie_rates.push(
                match user_rates
                    .get(&((user_id + 1) as i32)) // +1 due to Vec off by 1
                    .unwrap()
                    .get(&(movie.movie_id))
                {
                    Some(movie_rate) => *movie_rate,
                    None => 0f32,
                },
            );
        }
    }
    index
}

fn find_movie_id_from_name(name: &str, movies: &Vec<Movie>) -> Option<i32> {
    match movies.iter().find(|&mov| mov.title == name) {
        Some(mov) => Some(mov.movie_id),
        None => None,
    }
}

fn find_movie_offset_from_id(id: i32, movies: &Vec<Movie>) -> usize {
    movies
        .iter()
        .position(|mov| mov.movie_id == id)
        .expect(format!("No movies found with provided ID {}", id).as_str())
}

fn main() {
    let mut matrix_rows: Vec<Vec<f32>> = vec![];
    let movies = get_csv_values::<Movie>("./dataset/ml-latest-small/movies.csv");
    let rating = get_csv_values::<Rating>("./dataset/ml-latest-small/ratings.csv");

    println!("NB movies {}", movies.len());
    let target_movie_id =
        find_movie_id_from_name(std::env::args().nth(1).unwrap().as_str(), &movies)
            .expect("No movie with that name");

    let rate_index = get_rating_index(&rating, &movies);
    for movie in &movies {
        let mvid: i32 = (movie.movie_id).try_into().unwrap();
        matrix_rows.push(rate_index.get(&mvid).unwrap().clone());
    }
    let matrix = Matrix::from_row_vectors(&matrix_rows);
    match rustml::knn::scan(
        &matrix,
        &matrix_rows[find_movie_offset_from_id(target_movie_id, &movies)][..],
        5,
        |x, y| Euclid::compute(x, y).unwrap(),
    ) {
        Some(res) => {
            for movie_id in res {
                println!("{}", movies[movie_id].title);
            }
        }
        None => println!("No result"),
    }
}
