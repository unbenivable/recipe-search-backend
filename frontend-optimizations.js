/**
 * FRONTEND OPTIMIZATIONS FOR REDUCING API CALLS
 * 
 * This file contains code examples for reducing excessive API calls 
 * to the recipe search backend
 */

// ========================
// 1. IMPROVED DEBOUNCING
// ========================
import { useState, useEffect, useCallback } from 'react';

// BEFORE: Original search code with some debouncing but still issues
function OriginalSearch() {
  const [searchTerm, setSearchTerm] = useState('');
  
  useEffect(() => {
    // This still sends a request for every keystroke after the timeout
    const handler = setTimeout(() => {
      searchTerm.trim().length > 0 && fetchRecipes();
    }, 500);
    
    return () => clearTimeout(handler);
  }, [searchTerm]);
  
  // Rest of the component...
}

// AFTER: Improved debouncing with minimum length and reference tracking
function ImprovedSearch() {
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedTerm, setDebouncedTerm] = useState('');
  const [lastSearched, setLastSearched] = useState('');
  
  // Separate the search term change from the API call
  useEffect(() => {
    const handler = setTimeout(() => {
      // Only update debounced term if it's different and meets minimum length
      if (searchTerm.trim().length >= 3) {
        setDebouncedTerm(searchTerm);
      }
    }, 800); // Increased to 800ms (from 500ms)
    
    return () => clearTimeout(handler);
  }, [searchTerm]);
  
  // Only fetch when debounced term changes
  useEffect(() => {
    if (debouncedTerm && debouncedTerm !== lastSearched) {
      fetchRecipes();
      setLastSearched(debouncedTerm);
    }
  }, [debouncedTerm]);
  
  // Rest of component...
}

// ========================
// 2. REQUEST BATCHING
// ========================

// BEFORE: Making separate requests for each search
function NoBatchingExample() {
  const searchMultipleTerms = (terms) => {
    terms.forEach(term => {
      axios.post('/search', { ingredients: [term] });
    });
  };
}

// AFTER: Batch multiple search terms into one request
function WithBatchingExample() {
  const searchMultipleTerms = (terms) => {
    // Combine searches into a single request
    axios.post('/batch-search', { ingredientSets: terms });
  };
}

// ========================
// 3. CLIENT-SIDE CACHING
// ========================

// Simple cache implementation
const searchCache = {
  cache: {},
  
  // Get cached result
  get(cacheKey) {
    const cachedItem = this.cache[cacheKey];
    if (!cachedItem) return null;
    
    // Check if the cache is still valid (30 min expiry)
    if (Date.now() - cachedItem.timestamp > 30 * 60 * 1000) {
      delete this.cache[cacheKey];
      return null;
    }
    
    return cachedItem.data;
  },
  
  // Store result in cache
  set(cacheKey, data) {
    this.cache[cacheKey] = {
      data,
      timestamp: Date.now()
    };
  }
};

// Using the cache in search function
const searchWithCache = async (ingredients) => {
  // Create cache key
  const sortedIngredients = [...ingredients].sort();
  const cacheKey = sortedIngredients.join(',');
  
  // Check cache
  const cachedResult = searchCache.get(cacheKey);
  if (cachedResult) {
    console.log('Using cached result');
    return cachedResult;
  }
  
  // If not in cache, make the API call
  try {
    const response = await axios.post('/search', { ingredients });
    const result = response.data;
    
    // Store in cache
    searchCache.set(cacheKey, result);
    
    return result;
  } catch (error) {
    console.error('Error fetching recipes:', error);
    throw error;
  }
};

// ========================
// 4. REACT MEMOIZATION / LIFECYCLE OPTIMIZATION
// ========================

// Use React.memo to prevent unnecessary re-renders
import React, { memo, useMemo } from 'react';

// Memoize component
const RecipeCard = memo(function RecipeCard({ recipe }) {
  return (
    <div className="recipe-card">
      <h3>{recipe.title}</h3>
      {/* Rest of card content */}
    </div>
  );
});

// Memoize expensive calculations/filtering
function RecipeList({ recipes, filter }) {
  // Only recalculate when recipes or filter changes
  const filteredRecipes = useMemo(() => {
    console.log('Filtering recipes');
    return recipes.filter(recipe => 
      recipe.title.toLowerCase().includes(filter.toLowerCase())
    );
  }, [recipes, filter]);
  
  return (
    <div className="recipe-list">
      {filteredRecipes.map(recipe => (
        <RecipeCard key={recipe.id} recipe={recipe} />
      ))}
    </div>
  );
}

// ========================
// 5. PREVENT UNINTENTIONAL RERENDERS
// ========================

// PROBLEM: New function created on every render causing child components to rerender
function ProblemComponent() {
  // This function is recreated on every render
  const handleClick = () => {
    console.log('Button clicked');
  };
  
  return <ChildComponent onClick={handleClick} />;
}

// SOLUTION: Use useCallback to memoize functions
function SolutionComponent() {
  // This function is only recreated if dependencies change
  const handleClick = useCallback(() => {
    console.log('Button clicked');
  }, []);
  
  return <ChildComponent onClick={handleClick} />;
}

// PROBLEM: Rerenders from prop changes
class IneffectiveParent extends React.Component {
  state = { count: 0 };
  
  increment = () => {
    this.setState({ count: this.state.count + 1 });
  };
  
  render() {
    // Child rerenders even though searchData didn't change
    return (
      <div>
        <button onClick={this.increment}>Count: {this.state.count}</button>
        <SearchResults searchData={this.props.searchData} />
      </div>
    );
  }
}

// SOLUTION: Use PureComponent or React.memo
const MemoizedSearchResults = memo(SearchResults);
// Now it only rerenders when searchData actually changes 

// ========================
// 6. FIX CIRCULAR REFERENCE ERRORS
// ========================

// PROBLEM: The current error you're seeing with SVGSVGElement
// This happens when React DOM elements get included in API requests

// BROKEN EXAMPLE - This causes "Converting circular structure to JSON" errors
function BrokenSearchComponent() {
  const [searchTerm, setSearchTerm] = useState('');
  const svgRef = useRef(null);
  
  const handleSearch = async () => {
    try {
      // WRONG: Accidentally including the SVG element or event object in the request
      const response = await axios.post('https://web-production-9df5.up.railway.app/search', {
        ingredients: [searchTerm],
        // The problem is here - including DOM references or event objects
        svgElement: svgRef.current, // This creates circular references!
        // Or sometimes event handlers accidentally pass the entire event object:
        // event: e, // This also creates circular references!
      });
      
      // Process response...
    } catch (error) {
      console.error('Error fetching recipes:', error);
    }
  };
  
  return (
    <div>
      <input 
        type="text" 
        value={searchTerm} 
        onChange={(e) => setSearchTerm(e.target.value)} 
      />
      <svg ref={svgRef}>...</svg>
      <button onClick={handleSearch}>Search</button>
    </div>
  );
}

// FIXED EXAMPLE - Always sanitize data before sending to API
function FixedSearchComponent() {
  const [searchTerm, setSearchTerm] = useState('');
  const svgRef = useRef(null);
  
  const handleSearch = async () => {
    try {
      // CORRECT: Only send the data needed by the API
      const response = await axios.post('https://web-production-9df5.up.railway.app/search', {
        ingredients: [searchTerm.trim()], // Just the plain string data
        // No DOM elements or circular references
      });
      
      // Process response...
    } catch (error) {
      console.error('Error fetching recipes:', error);
    }
  };
  
  return (
    <div>
      <input 
        type="text" 
        value={searchTerm} 
        onChange={(e) => setSearchTerm(e.target.value)} 
      />
      <svg ref={svgRef}>...</svg>
      <button onClick={handleSearch}>Search</button>
    </div>
  );
}

// Helper function to sanitize data before sending to API
// Use this to ensure you're only sending serializable data
function sanitizeDataForAPI(data) {
  // Method 1: Manual approach - specify exactly what to include
  return {
    ingredients: Array.isArray(data.ingredients) 
      ? data.ingredients.map(ing => typeof ing === 'string' ? ing.trim() : '')
      : [],
    page: typeof data.page === 'number' ? data.page : 1,
    page_size: typeof data.page_size === 'number' ? data.page_size : 20,
    // Add other needed fields...
  };
  
  // Method 2: Alternative using JSON serialization to strip non-serializable data
  // return JSON.parse(JSON.stringify(data));
  // Note: This can be inefficient for large objects but works for simple cases
}

// Example usage:
async function safeApiCall(ingredients) {
  try {
    const dataToSend = sanitizeDataForAPI({ 
      ingredients: ingredients,
      page: 1
    });
    
    const response = await axios.post('https://web-production-9df5.up.railway.app/search', dataToSend);
    return response.data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
} 