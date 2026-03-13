function LossChart({ history }) {
  return (
    <div className="bg-white p-4 rounded shadow mb-6">
      <h2 className="font-semibold mb-2">Training History</h2>

      <ul className="text-sm">
        {history.map((h, idx) => (
          <li key={idx}>
            Epoch {idx + 1} — Train: {h.train_loss}, Val: {h.val_loss}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default LossChart
