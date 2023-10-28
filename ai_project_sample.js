import React, { useState } from 'react';
import './App.css';

function App() {
  const [movies, setMovies] = useState([
    { title: '', rating: 0 },
    { title: '', rating: 0 },
    { title: '', rating: 0 }
  ]);
  const [recommendations, setRecommendations] = useState('');
  const [hasClickedRecommend, setHasClickedRecommend] = useState(false);

  

  return (
    <div className="App">
      <header className="App-header">
        <p>Hello AISC!</p>
        <form onSubmit={e => e.preventDefault()}>
          {movies.map((movie, index) => (
            <div key={index}>
              <input 
                placeholder="Movie title" 
                value={movie.title}
                onChange={e => handleInputChange(index, 'title', e.target.value)}
              />
              <div>
                {[1, 2, 3, 4, 5].map(starNumber => (
                  <span
                    key={starNumber}
                    onClick={() => handleStarClick(index, starNumber)}
                    style={{ cursor: 'pointer' }}
                  >
                    {starNumber <= movie.rating ? '★' : '☆'}
                  </span>
                ))}
              </div>
            </div>
          ))}
          <button onClick={handleSubmit}>Get Recommendations</button>
        </form>

        {hasClickedRecommend && (
          <div>
            <h2>Based on your preferences, here are some more movie recommendations:</h2>
            <p>{recommendations}</p>
          </div>
        )}
      </header>
    </div>
  );

  function handleInputChange(index, field, value) {
    const newMovies = [...movies];
    newMovies[index][field] = value;
    setMovies(newMovies);
  }

  
}

export default App;



