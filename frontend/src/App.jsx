import { useState, useCallback, useRef } from "react";
import SearchBar from "./components/SearchBar";
import UserSwitcher from "./components/UserSwitcher";
import MovieCard from "./components/MovieCard";
import Toast from "./components/Toast";

async function fetchResults(query, userId) {
  const params = new URLSearchParams({ q: query, k: 10 });
  if (userId != null) params.append("user_id", userId);
  const res = await fetch(`/search?${params}`);
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}

export default function App() {
  const [query, setQuery]       = useState("");
  const [userId, setUserId]     = useState(null);
  const [results, setResults]   = useState([]);
  const [variant, setVariant]   = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [searched, setSearched] = useState(false);
  const [toast, setToast]       = useState(false);
  const refreshTimer            = useRef(null);

  const runSearch = useCallback(async (q, uid) => {
    setLoading(true);
    setError(null);
    try {
      const { variant: v, results: data } = await fetchResults(q, uid);
      setResults(data);
      setVariant(v);
      setSearched(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  function handleSearch(q) {
    setQuery(q);
    runSearch(q, userId);
  }

  function handleUserChange(uid) {
    setUserId(uid);
    if (query) runSearch(query, uid);
  }

  function handleFeedback() {
    setToast(true);
    clearTimeout(refreshTimer.current);
    refreshTimer.current = setTimeout(() => {
      setToast(false);
      runSearch(query, userId);
    }, 800);
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#0a0a0a] text-gray-100">
      <Toast show={toast} message="Profile updated — refreshing results…" />
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-3">
        <span className="text-brand font-black text-2xl tracking-tight">MOVIRANK</span>
        <span className="text-gray-600 text-sm hidden sm:block">
          Semantic search · Personalised ranking
        </span>
      </header>

      {/* Hero / Search */}
      <div className="px-4 py-10 flex flex-col items-center gap-6 border-b border-gray-800">
        <h1 className="text-3xl sm:text-4xl font-bold text-white text-center leading-tight">
          Find your next favourite film
        </h1>
        <p className="text-gray-400 text-center max-w-lg text-sm sm:text-base">
          Describe what you want to watch. Switch users to see how personalisation changes results.
        </p>
        <div className="w-full max-w-2xl px-2">
          <SearchBar onSearch={handleSearch} loading={loading} />
        </div>

        {/* User switcher */}
        <div className="w-full max-w-3xl px-2">
          <p className="text-gray-600 text-xs text-center mb-3 uppercase tracking-widest">
            Viewing as
          </p>
          <UserSwitcher activeUserId={userId} onUserChange={handleUserChange} />
        </div>
      </div>

      {/* Results */}
      <main className="flex-1 px-4 sm:px-6 py-8">
        {error && (
          <div className="text-center text-red-400 py-10 text-sm">
            {error} — is the backend running on port 8000?
          </div>
        )}

        {loading && (
          <div className="text-center py-20">
            <div className="text-5xl animate-bounce mb-4">🎬</div>
            <p className="text-gray-500">Searching…</p>
          </div>
        )}

        {!loading && searched && results.length === 0 && !error && (
          <p className="text-center text-gray-500 py-16">No results found.</p>
        )}

        {!loading && results.length > 0 && (
          <>
            <div className="flex items-center gap-3 mb-6">
              <p className="text-gray-500 text-sm">
                Top {results.length} results for{" "}
                <span className="text-white font-medium">"{query}"</span>
              </p>
              {variant && (
                <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
                  variant === "ranker"
                    ? "bg-brand/20 text-brand border border-brand/30"
                    : "bg-gray-800 text-gray-400 border border-gray-700"
                }`}>
                  {variant === "ranker" ? "B · LightGBM ranker" : "A · Baseline (FAISS)"}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 sm:gap-4">
              {results.map((movie, i) => (
                <MovieCard
                  key={movie.movieId}
                  movie={movie}
                  rank={i + 1}
                  userId={userId}
                  onFeedback={handleFeedback}
                />
              ))}
            </div>
          </>
        )}

        {!loading && !searched && (
          <div className="text-center py-20 text-6xl opacity-20">🎥</div>
        )}
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-gray-700 text-xs">
        FAISS retrieval · LightGBM re-ranking · Redis feature store
      </footer>
    </div>
  );
}
