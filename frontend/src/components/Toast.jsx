export default function Toast({ show, message }) {
  return (
    <div
      className={`
        fixed bottom-6 left-1/2 -translate-x-1/2 z-50
        bg-brand text-white text-sm font-medium px-5 py-2.5 rounded-full shadow-lg
        transition-all duration-300
        ${show ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4 pointer-events-none"}
      `}
    >
      {message}
    </div>
  );
}
