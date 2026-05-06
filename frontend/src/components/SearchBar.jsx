import { useState } from "react";

export default function SearchBar({ onSearch, loading }) {
  const [query, setQuery] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (query.trim()) onSearch(query.trim());
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 w-full max-w-2xl mx-auto">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="e.g. dark psychological thriller with a twist…"
        className="
          flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-3
          text-white placeholder-gray-500 outline-none
          focus:border-brand focus:ring-1 focus:ring-brand transition-colors
        "
      />
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="
          bg-brand hover:bg-red-700 disabled:bg-gray-700 disabled:text-gray-500
          text-white font-semibold px-6 py-3 rounded-xl transition-colors
          flex items-center gap-2
        "
      >
        {loading ? (
          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
          </svg>
        ) : (
          <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z"/>
          </svg>
        )}
        Search
      </button>
    </form>
  );
}
