import { useState } from "react";

const GENRE_COLORS = {
  Action:      "bg-orange-900/60 text-orange-300",
  Adventure:   "bg-yellow-900/60 text-yellow-300",
  Animation:   "bg-purple-900/60 text-purple-300",
  Comedy:      "bg-green-900/60 text-green-300",
  Crime:       "bg-red-900/60 text-red-300",
  Documentary: "bg-blue-900/60 text-blue-300",
  Drama:       "bg-pink-900/60 text-pink-300",
  Fantasy:     "bg-violet-900/60 text-violet-300",
  Horror:      "bg-red-950/80 text-red-400",
  Mystery:     "bg-indigo-900/60 text-indigo-300",
  Romance:     "bg-rose-900/60 text-rose-300",
  "Sci-Fi":    "bg-cyan-900/60 text-cyan-300",
  Thriller:    "bg-slate-800 text-slate-300",
};

const FALLBACK = "https://placehold.co/300x450/1f2937/6b7280?text=No+Poster";

function Stars({ rating }) {
  const pct = (rating / 5) * 100;
  return (
    <div className="flex items-center gap-1.5">
      <div className="relative text-gray-700 text-sm leading-none select-none">
        {"★★★★★".split("").map((s, i) => <span key={i}>{s}</span>)}
        <div className="absolute inset-0 overflow-hidden text-yellow-400" style={{ width: `${pct}%` }}>
          {"★★★★★".split("").map((s, i) => <span key={i}>{s}</span>)}
        </div>
      </div>
      <span className="text-gray-400 text-xs">{rating.toFixed(1)}</span>
    </div>
  );
}

export default function MovieCard({ movie, rank, userId, onFeedback }) {
  const [action, setAction]   = useState(null);   // "like" | "dislike" | null
  const [pending, setPending] = useState(false);

  const genres = movie.genres ? movie.genres.split("|") : [];
  const year   = movie.year ? Math.round(movie.year) : null;
  const scorePct = Math.max(0, Math.min(100, ((movie.score + 0.5) / 2) * 100));

  async function handleFeedback(act) {
    if (pending || action) return;
    setPending(true);
    try {
      await fetch(`/user/${userId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ movie_id: movie.movieId, action: act }),
      });
      setAction(act);
      onFeedback?.();          // triggers results refresh in parent
    } catch (_) {
      // silent — feedback is best-effort
    } finally {
      setPending(false);
    }
  }

  const liked    = action === "like";
  const disliked = action === "dislike";

  return (
    <div className={`
      group flex flex-col bg-gray-900 rounded-2xl overflow-hidden border transition-all
      hover:shadow-xl hover:shadow-black/50 hover:-translate-y-0.5
      ${liked    ? "border-green-600/70 shadow-green-900/30 shadow-lg" : ""}
      ${disliked ? "border-red-800/70"  : ""}
      ${!action  ? "border-gray-800 hover:border-gray-600" : ""}
    `}>
      {/* Poster */}
      <div className="relative aspect-[2/3] overflow-hidden bg-gray-800 flex-shrink-0">
        <img
          src={movie.poster_url || FALLBACK}
          alt={movie.clean_title}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          onError={(e) => { e.target.src = FALLBACK; }}
        />
        <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-sm text-white text-xs font-bold w-7 h-7 rounded-full flex items-center justify-center">
          #{rank}
        </div>
        {/* Score bar */}
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-700/60">
          <div className="h-full bg-brand transition-all" style={{ width: `${scorePct}%` }} />
        </div>
      </div>

      {/* Info */}
      <div className="p-3 space-y-2 flex flex-col flex-1">
        <div>
          <h3 className="font-semibold text-white text-sm leading-snug line-clamp-2">
            {movie.clean_title}
          </h3>
          {year && <p className="text-gray-500 text-xs mt-0.5">{year}</p>}
        </div>

        <div className="flex flex-wrap gap-1">
          {genres.slice(0, 3).map((g) => (
            <span key={g} className={`text-xs px-1.5 py-0.5 rounded-md font-medium ${GENRE_COLORS[g] ?? "bg-gray-800 text-gray-400"}`}>
              {g}
            </span>
          ))}
        </div>

        {movie.avg_rating != null && <Stars rating={movie.avg_rating} />}

        {/* Like / Dislike — only when a named user is active */}
        {userId != null && (
          <div className="flex gap-2 pt-1 mt-auto">
            <button
              onClick={() => handleFeedback("like")}
              disabled={pending || !!action}
              title="Like"
              className={`
                flex-1 flex items-center justify-center gap-1 py-1.5 rounded-lg text-sm
                transition-all disabled:cursor-default
                ${liked
                  ? "bg-green-700 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-green-900/50 hover:text-green-400 disabled:hover:bg-gray-800 disabled:hover:text-gray-400"}
              `}
            >
              {pending ? "…" : liked ? "👍 Liked" : "👍"}
            </button>
            <button
              onClick={() => handleFeedback("dislike")}
              disabled={pending || !!action}
              title="Dislike"
              className={`
                flex-1 flex items-center justify-center gap-1 py-1.5 rounded-lg text-sm
                transition-all disabled:cursor-default
                ${disliked
                  ? "bg-red-800 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-red-900/50 hover:text-red-400 disabled:hover:bg-gray-800 disabled:hover:text-gray-400"}
              `}
            >
              {pending ? "…" : disliked ? "👎 Nope" : "👎"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
