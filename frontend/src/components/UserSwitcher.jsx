const USERS = [
  { id: null,  label: "Anonymous",      emoji: "👤", desc: "No personalization" },
  { id: 1,     label: "Drama Fan",      emoji: "🎭", desc: "Dramas · 2000s" },
  { id: 591,   label: "Action Fan",     emoji: "💥", desc: "Action · 1980s" },
  { id: 1324,  label: "Horror Fan",     emoji: "👻", desc: "Horror · 1990s" },
  { id: 76,    label: "Comedy Fan",     emoji: "😂", desc: "Comedy · 2000s" },
];

export default function UserSwitcher({ activeUserId, onUserChange }) {
  return (
    <div className="flex flex-wrap gap-2 justify-center">
      {USERS.map((u) => {
        const active = activeUserId === u.id;
        return (
          <button
            key={u.id ?? "anon"}
            onClick={() => onUserChange(u.id)}
            className={`
              flex flex-col items-center px-4 py-2 rounded-xl border transition-all text-sm
              ${active
                ? "bg-brand border-brand text-white shadow-lg shadow-brand/30"
                : "bg-gray-900 border-gray-700 text-gray-300 hover:border-gray-500 hover:text-white"
              }
            `}
          >
            <span className="text-xl">{u.emoji}</span>
            <span className="font-semibold mt-0.5">{u.label}</span>
            <span className={`text-xs mt-0.5 ${active ? "text-red-200" : "text-gray-500"}`}>
              {u.desc}
            </span>
          </button>
        );
      })}
    </div>
  );
}
