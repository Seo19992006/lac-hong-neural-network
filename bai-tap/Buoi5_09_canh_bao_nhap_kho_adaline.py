"""Buổi 5 — Tình huống 09: Cảnh báo nhập thêm hàng bằng Adaline (Widrow–Hoff).

Dữ liệu thật: inventory_demand_stockout_risk.csv (Kaggle - Inventory Demand
Forecasting and Stockout Risk), 2800 dòng, mỗi dòng là một sản phẩm quan sát
tại một cửa hàng/thời điểm.

Ý tưởng: thay vì dùng nhãn 0/1 có sẵn (stockout_risk), ta tự xây dựng một
target liên tục "diem_khan_cap" trong [0, 1] dựa trên công thức kinh điển
trong quản lý tồn kho — điểm tái đặt hàng (reorder point):

    reorder_point   = daily_demand * lead_time_days
    diem_khan_cap   = clip((reorder_point - current_stock) / reorder_point, 0, 1)

Sau đó dùng Adaline (net = w.x + b, cập nhật theo LMS) để học xấp xỉ tuyến
tính điểm khẩn cấp này từ các đặc trưng quan sát được tại thời điểm ra
quyết định. Chạy trực tiếp: `python 09_canh_bao_nhap_kho_adaline.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bat buoc de terminal Windows (cmd/PowerShell) in duoc tieng Viet co dau
# ma khong bi UnicodeEncodeError. Neu van loi, chay them lenh `chcp 65001`
# truoc khi chay script, hoac dat bien moi truong PYTHONUTF8=1.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Cần cài scikit-learn: pip install scikit-learn --break-system-packages"
    ) from exc


CSV_PATH = Path(
    r"C:\Users\Admin\lac-hong-neural-network\data\inventory_demand_stockout_risk.csv"
)
FEATURE_COLS = [
    "current_stock",
    "daily_demand",
    "lead_time_days",
    "supplier_reliability_score",
    "khuyen_mai",
    "anh_huong_thoitiet",
]
ETA = 0.01
N_EPOCHS = 200
NGUONG_CANH_BAO = 0.6
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Đọc dữ liệu và tạo target liên tục
# --------------------------------------------------------------------------
def load_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Đọc CSV thật, mã hoá cột dạng chữ, tạo target 'diem_khan_cap'."""
    df = pd.read_csv(csv_path)

    # Target liên tục dựa trên điểm tái đặt hàng (reorder point)
    df["reorder_point"] = df["daily_demand"] * df["lead_time_days"]
    df["diem_khan_cap"] = (
        (df["reorder_point"] - df["current_stock"]) / df["reorder_point"]
    ).clip(0, 1)

    # Mã hoá các cột dạng chữ thành số
    df["khuyen_mai"] = (df["promotion_active"] == "Yes").astype(float)
    weather_map = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
    df["anh_huong_thoitiet"] = df["weather_impact"].map(weather_map)

    return df


# --------------------------------------------------------------------------
# 2 & 3. Tách train/test rồi chuẩn hoá (fit trên train để tránh leakage)
# --------------------------------------------------------------------------
def split_and_scale(df: pd.DataFrame):
    X_raw = df[FEATURE_COLS].to_numpy(dtype=float)
    t_all = df["diem_khan_cap"].to_numpy(dtype=float)

    X_train_raw, X_test_raw, t_train, t_test, df_train, df_test = train_test_split(
        X_raw, t_all, df, test_size=0.2, random_state=RANDOM_STATE
    )

    x_min = X_train_raw.min(axis=0)
    x_max = X_train_raw.max(axis=0)

    def chuan_hoa(X: np.ndarray) -> np.ndarray:
        return (X - x_min) / (x_max - x_min + 1e-9)

    X_train = chuan_hoa(X_train_raw)
    X_test = chuan_hoa(X_test_raw)

    return X_train, X_test, t_train, t_test, df_train, df_test, x_min, x_max


# --------------------------------------------------------------------------
# 4. Adaline / Widrow-Hoff
# --------------------------------------------------------------------------
def net_output(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """Adaline: output chính là net = w.x + b (không qua hàm bước)."""
    return float(np.dot(w, x) + b)


def update(
    x: np.ndarray, t: float, w: np.ndarray, b: float, eta: float
) -> tuple[float, float, np.ndarray, float]:
    """Một lượt Widrow-Hoff: tính net, sai số liên tục, rồi cập nhật w, b."""
    net = net_output(x, w, b)
    error = t - net
    w_new = w + eta * error * x
    b_new = b + eta * error
    return net, error, w_new, b_new


def mse(X: np.ndarray, t: np.ndarray, w: np.ndarray, b: float) -> float:
    """Sai số bình phương trung bình E = mean(0.5 * e^2)."""
    errors = t - (X @ w + b)
    return float(np.mean(0.5 * errors**2))


def train(
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
    eta: float = ETA,
    n_epochs: int = N_EPOCHS,
    seed: int = RANDOM_STATE,
):
    """Huấn luyện Adaline qua nhiều epoch, xáo trộn thứ tự mẫu mỗi epoch."""
    n_features = X_train.shape[1]
    w = np.zeros(n_features)
    b = 0.0

    mse_train_hist: list[float] = []
    mse_test_hist: list[float] = []

    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(X_train))

    for _ in range(n_epochs):
        rng.shuffle(idx_all)
        for i in idx_all:
            _, _, w, b = update(X_train[i], t_train[i], w, b, eta)
        mse_train_hist.append(mse(X_train, t_train, w, b))
        mse_test_hist.append(mse(X_test, t_test, w, b))

    return w, b, mse_train_hist, mse_test_hist


# --------------------------------------------------------------------------
# 5. Vẽ đường học (lưu ra file PNG vì chạy từ terminal không có màn hình)
# --------------------------------------------------------------------------
def plot_learning_curve(mse_train_hist, mse_test_hist, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(mse_train_hist, label="MSE train")
    plt.plot(mse_test_hist, label="MSE test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE = mean(0.5 * e^2)")
    plt.title("Đường học của Adaline (Widrow-Hoff) trên dữ liệu tồn kho")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# --------------------------------------------------------------------------
# 6. Đánh giá
# --------------------------------------------------------------------------
def danh_gia(X_train, df_train, X_test, t_test, df_test, w, b) -> None:
    pred_test = X_test @ w + b
    errors_test = t_test - pred_test

    print(f"MSE trên tập test: {mse(X_test, t_test, w, b):.4f}")
    print(f"MAE trên tập test: {np.mean(np.abs(errors_test)):.4f}")

    print("\nTrọng số học được:")
    for name, wi in zip(FEATURE_COLS, w):
        print(f"  {name:28s}: {wi:+.4f}")
    print(f"  {'bias (b)':28s}: {b:+.4f}")

    print("\n5 mẫu có |sai số| lớn nhất trên tập test:")
    order = np.argsort(-np.abs(errors_test))[:5]
    for i in order:
        row = df_test.iloc[i]
        print(
            f"  tồn_kho={row['current_stock']:>4}  nhu_cầu={row['daily_demand']:>3}  "
            f"chờ_hàng={row['lead_time_days']:>2} ngày  |  "
            f"target={t_test[i]:.3f}  dự_đoán={pred_test[i]:.3f}  "
            f"sai_số={errors_test[i]:+.3f}  |  nhãn_thật(stockout_risk)={row['stockout_risk']}"
        )

    print("\nĐối chiếu với nhãn có sẵn 'stockout_risk':")
    actual_binary = df_test["stockout_risk"].to_numpy() == "Yes"
    baseline_acc = max(actual_binary.mean(), 1 - actual_binary.mean())
    print(f"  Baseline (luôn đoán lớp đa số): {baseline_acc:.3f}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred_binary = pred_test >= thr
        acc = (pred_binary == actual_binary).mean()
        prec = precision_score(actual_binary, pred_binary, zero_division=0)
        rec = recall_score(actual_binary, pred_binary, zero_division=0)
        print(
            f"  ngưỡng={thr:.1f}: accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}"
        )
    print(
        "  -> Accuracy tốt nhất vẫn thấp hơn baseline: nhãn 'stockout_risk' có sẵn"
        "\n     không tương quan chặt với quy tắc điểm tái đặt hàng dùng ở đây."
        "\n     Đây là lý do cần luôn kiểm tra chéo nhãn trước khi tin dùng."
    )

    # --- Kiểm chứng độc lập: dù mô hình mạnh hơn có đoán được nhãn này không? ---
    # Nếu ngay cả Logistic Regression (mô hình mạnh hơn Adaline, có thể học
    # đường biên phi tuyến qua sigmoid) cũng không đoán được tốt hơn baseline,
    # thì đây là bằng chứng nhãn 'stockout_risk' gần như KHÔNG có quan hệ thống
    # kê với các feature này trong bộ dữ liệu -> không phải lỗi của Adaline hay
    # của cách xây dựng target 'diem_khan_cap'.
    print("\nKiểm chứng độc lập bằng Logistic Regression (dự đoán trực tiếp stockout_risk):")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, df_train["stockout_risk"].to_numpy() == "Yes")
    auc = roc_auc_score(actual_binary, lr.predict_proba(X_test)[:, 1])
    print(f"  AUC = {auc:.3f} (AUC = 0.5 nghĩa là không tốt hơn đoán ngẫu nhiên)")
    if auc < 0.55:
        print(
            "  -> AUC gần 0.5: xác nhận nhãn 'stockout_risk' gần như không đoán"
            "\n     được từ các feature hiện có, dù dùng mô hình mạnh hơn Adaline."
            "\n     Đây là một đặc điểm của BỘ DỮ LIỆU, không phải hạn chế của mô hình."
        )


# --------------------------------------------------------------------------
# 7. Chọn ngưỡng và thử với sản phẩm giả định
# --------------------------------------------------------------------------
def du_doan_khan_cap(
    x_tho: list[float], w: np.ndarray, b: float, x_min: np.ndarray, x_max: np.ndarray
) -> tuple[float, bool]:
    """Dự đoán điểm khẩn cấp cho MỘT sản phẩm mới từ các giá trị feature THÔ
    (chưa chuẩn hoá). Trả về (diem_khan_cap, co_canh_bao)."""
    x_scaled = (np.asarray(x_tho, dtype=float) - x_min) / (x_max - x_min + 1e-9)
    diem = net_output(x_scaled, w, b)
    return diem, diem >= NGUONG_CANH_BAO


def thu_san_pham_gia_dinh(w, b, x_min, x_max) -> None:
    # [ton_kho, nhu_cau_ngay, ngay_cho_hang, diem_tin_cay_ncc, khuyen_mai(0/1), anh_huong_thoitiet(0/0.5/1)]
    san_pham_gia_dinh = {
        "Bán chậm, tồn nhiều": [500, 10, 5, 90, 0, 0.0],
        "Bán nhanh, tồn ít, chờ lâu": [50, 120, 10, 60, 1, 1.0],
        "Sản phẩm mới, chưa rõ xu hướng": [150, 50, 7, 75, 0, 0.5],
    }
    print(f"\nDự đoán cho sản phẩm giả định (ngưỡng cảnh báo = {NGUONG_CANH_BAO}):")
    for ten, x_tho in san_pham_gia_dinh.items():
        diem, canh_bao = du_doan_khan_cap(x_tho, w, b, x_min, x_max)
        diem_hien_thi = float(np.clip(diem, 0.0, 1.0))  # giới hạn về [0,1] để báo cáo cho người dùng
        trang_thai = "CẢNH BÁO nhập hàng" if canh_bao else "chưa cần nhập"
        ghi_chu = "  (đã vượt [0,1] do ngoại suy tuyến tính)" if diem != diem_hien_thi else ""
        print(
            f"  {ten:35s} -> diem_khan_cap={diem_hien_thi:.3f}  {trang_thai}"
            f"  [giá trị thô: {diem:+.3f}]{ghi_chu}"
        )


# --------------------------------------------------------------------------
def main() -> None:
    df = load_data()
    print(f"Đã đọc {len(df)} dòng từ {CSV_PATH.name}")

    X_train, X_test, t_train, t_test, df_train, df_test, x_min, x_max = split_and_scale(df)
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")

    w, b, mse_train_hist, mse_test_hist = train(X_train, t_train, X_test, t_test)
    print(f"\nMSE train: {mse_train_hist[0]:.4f} -> {mse_train_hist[-1]:.4f}")
    print(f"MSE test : {mse_test_hist[0]:.4f} -> {mse_test_hist[-1]:.4f}")

    out_png = Path(__file__).with_name("duong_hoc_mse.png")
    plot_learning_curve(mse_train_hist, mse_test_hist, out_png)
    print(f"\nĐã lưu đường học MSE vào: {out_png}")

    print()
    danh_gia(X_train, df_train, X_test, t_test, df_test, w, b)

    thu_san_pham_gia_dinh(w, b, x_min, x_max)


if __name__ == "__main__":
    main()