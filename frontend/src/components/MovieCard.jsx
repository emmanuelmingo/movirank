const GENRE_COLORS = {
  Action: "bg-orange-900/60 text-orange-300",
  Adventure: "bg-yellow-900/60 text-yellow-300",
  Animation: "bg-purple-900/60 text-purple-300",
  Comedy: "bg-green-900/60 text-green-300",
  Crime: "bg-red-900/60 text-red-300",
  Documentary: "bg-blue-900/60 text-blue-300",
  Drama: "bg-pink-900/60 text-pink-300",
  Fantasy: "bg-violet-900/60 text-violet-300",
  Horror: "bg-red-950/80 text-red-400",
  Mystery: "bg-indigo-900/60 text-indigo-300",
  Romance: "bg-rose-900/60 text-rose-300",
  "Sci-Fi": "bg-cyan-900/60 text-cyan-300",
  Thriller: "bg-slate-800 text-slate-300",
};

const FALLBACK_POSTER = "https://via.placeholder.com/300x450/1f2937/6b7280?text=No+Poster";

function StarRating({ rating, max = 5 }) {
  const pct = (rating / max) * 100;
  return (
    <div className="flex items-center gap-1.5">
      <div className="relative text-gray-700 text-sm leading-none">
        {"★★★★★".split("").map((s, i) => (
          <span key={i}>{s}</span>
        ))}
        <div className="absolute inset-0 overflow-hidden text-yellow-400" style={{ width: `${pct}%` }}>
          {"★★★★★".split("").map((s, i) => <span key={i}>{s}</span>)}
        </div>
      </div>
      <span className="text-gray-400 text-xs">{rating.toFixed(1)}</span>
    </div>
  );
}

export default function MovieCard({ movie, rank }) {
  const genres = movie.genres ? movie.genres.split("|") : [];
  const year   = movie.year ? Math.round(movie.year) : null;
  const score  = Math.max(0, Math.min(1, (movie.score + 0.5) / 2)); // normalise to [0,1] for bar

  return (
    <div className="group bg-gray-900 rounded-2xl overflow-hidden border border-gray-800 hover:border-gray-600 transition-all hover:shadow-xl hover:shadow-black/50 hover:-translate-y-0.5">
      {/* Poster */}
      <div className="relative aspect-[2/3] overflow-hidden bg-gray-800">
        <img
          src={movie.poster_url || FALLBACK_POSTER}
          alt={movie.clean_title}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          onError={(e) => { e.target.src = FALLBACK_POSTER; }}
        />
        {/* Rank badge */}
        <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-sm text-white text-xs font-bold w-7 h-7 rounded-full flex items-center justify-center">
          #{rank}
        </div>
        {/* Score bar */}
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-700/60">
          <div
            className="h-full bg-brand transition-all"
            style={{ width: `${score * 100}%` }}
          />
        </div>
      </div>

      {/* Info */}
      <div className="p-3 space-y-2">
        <div>
          <h3 className="font-semibold text-white text-sm leading-snug line-clamp-2">
            {movie.clean_title}
          </h3>
          {year && <p className="text-gray-500 text-xs mt-0.5">{year}</p>}
        </div>

        {/* Genres */}
        <div className="flex flex-wrap gap-1">
          {genres.slice(0, 3).map((g) => (
            <span
              key={g}
              className={`text-xs px-1.5 py-0.5 rounded-md font-medium ${GENRE_COLORS[g] ?? "bg-gray-800 text-gray-400"}`}
            >
              {g}
            </span>
          ))}
        </div>

        {/* Rating */}
        {movie.avg_rating != null && (
          <StarRating rating={movie.avg_rating} />
        )}
      </div>
    </div>
  );
}
