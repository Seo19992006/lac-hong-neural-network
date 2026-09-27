"""Buổi 5 — Adaline & Widrow–Hoff nhận biết mức độ giống chữ C từ feature C/I/T.

Quy trình học: dự đoán bằng tay -> hoàn thiện các TODO -> chạy kiểm tra -> giải thích.
Khác với Perceptron (Buổi 4), Adaline dùng chính `net` làm output (số thực),
sai số e = t - net là liên tục nên biết "sai bao nhiêu", không chỉ "đúng/sai".

File này là starter dành cho sinh viên, không kèm lời giải.
"""

from __future__ import annotations

import numpy as np


LABELS = ("C", "I", "T")
X = np.array(
    [
        [1, 0, 1, 1, 0],  # C
        [0, 0, 1, 1, 1],  # I
        [0, 0, 1, 0, 1],  # T
    ],
    dtype=float,
)
T = np.array([1.0, 0.0, 0.0])  # target liên tục: mức độ giống C

ETA = 0.1
W0 = np.zeros(X.shape[1], dtype=float)
B0 = 0.0


def net_output(x: np.ndarray, w: np.ndarray, b: float) -> float:
  """Adaline: output chính là net = w·x + b (KHÔNG qua hàm bước)."""
  # TODO 1: tính và trả về net = w dot x + b (kiểu float).
  return float(np.dot(w, x) + b)


def update(
    x: np.ndarray,
    t: float,
    w: np.ndarray,
    b: float,
    eta: float,
) -> tuple[float, float, np.ndarray, float]:
  """Một lượt Widrow–Hoff: tính net, sai số liên tục, rồi cập nhật w, b."""
  net = net_output(x, w, b)

  # TODO 2: tính error = t - net (sai số liên tục của Adaline).
  error = t - net

  # TODO 3: tạo w_new = w + eta * error * x và b_new = b + eta * error.
  #         Gợi ý: không sửa trực tiếp mảng w được truyền vào hàm.
  w_new = w + eta * error * x
  b_new = b + eta * error

  # Trả về (net, error, w_new, b_new).
  return net, error, w_new, b_new


def mse(w: np.ndarray, b: float) -> float:
  """Sai số bình phương trung bình E = mean(0.5 * e^2) trên cả 3 mẫu."""
  # TODO 4: với mỗi (x, t) tính e = t - net rồi lấy trung bình của 0.5 * e^2.
  errors = np.array([t - net_output(x, w, b) for x, t in zip(X, T)])
  return float(np.mean(0.5 * errors**2))


def train_one_epoch(
    eta: float = ETA,
) -> tuple[np.ndarray, float, list[dict[str, object]]]:
  """Chạy đúng một epoch theo thứ tự C, I, T và lưu lịch sử từng lượt."""
  w = W0.copy()
  b = B0
  history: list[dict[str, object]] = []

  for label, x_i, t_i in zip(LABELS, X, T):
    net, error, w, b = update(x_i, float(t_i), w, b, eta)
    history.append(
        {
            "mau": label,
            "net": net,
            "error": error,
            "w_sau_luot": w.copy(),
            "b_sau_luot": b,
        }
    )

  return w, b, history


def run_checks() -> None:
  """Các kiểm tra chỉ chạy được sau khi hoàn thiện đủ các TODO.

  Đối chiếu với bảng tính tay trong bài giảng Buổi 5 (eta=0.1, w0=0, b0=0).
  """
  net_C = net_output(X[0], W0, B0)
  assert np.isclose(net_C, 0.0), "net ban đầu của mẫu C phải bằng 0."

  w_final, b_final, history = train_one_epoch()
  assert len(history) == 3, "Một epoch phải đi qua đủ C, I, T."

  nets = [row["net"] for row in history]
  assert np.isclose(nets[0], 0.0), "net của C phải bằng 0."
  assert np.isclose(nets[1], 0.3), "net của I phải bằng 0.3."
  assert np.isclose(nets[2], 0.11), "net của T phải bằng 0.11."

  assert np.allclose(w_final, [0.1, 0.0, 0.059, 0.07, -0.041]), (
      "Trọng số sau một epoch chưa đúng."
  )
  assert np.isclose(b_final, 0.059), "Bias sau một epoch chưa đúng."

  # Nhiều epoch: MSE phải giảm.
  w, b = W0.copy(), B0
  mse_dau = None
  for epoch in range(30):
    for x_i, t_i in zip(X, T):
      _, _, w, b = update(x_i, float(t_i), w, b, ETA)
    if epoch == 0:
      mse_dau = mse(w, b)
  mse_cuoi = mse(w, b)
  assert mse_cuoi < mse_dau, "MSE phải giảm sau nhiều epoch."


if __name__ == "__main__":
  run_checks()
  _, _, rows = train_one_epoch()
  for row in rows:
    print(row)
  print("Đã vượt qua toàn bộ kiểm tra Buổi 5.")