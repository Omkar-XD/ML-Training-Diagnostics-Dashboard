function Diagnostics({ diagnostics }) {
  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-semibold mb-2">Diagnostics</h2>

      <ul className="list-disc ml-5 text-sm">
        {diagnostics.map((d, idx) => (
          <li key={idx}>
            <b>{d.type}:</b> {d.message}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default Diagnostics
