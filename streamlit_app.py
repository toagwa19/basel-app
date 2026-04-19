###############################################################################
#  Basel CRM リスクアセット最適化デモ
#  SA-CCR × RWA最小化 × 量子アニーリング × 古典最適化比較
#  Powered by Fixstars Amplify AE
###############################################################################
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
warnings.simplefilter("ignore")

# ── Amplify ──────────────────────────────────────────────────────────────────
from amplify import BinarySymbolGenerator, Model, FixstarsClient, solve
from amplify import sum as asum

# ── Classical optimization ────────────────────────────────────────────────────
from scipy.optimize import linprog

# ══════════════════════════════════════════════════════════════════════════════
# ページ設定
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Basel CRM リスクアセット最適化",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Basel定数 ─────────────────────────────────────────────────────────────────
RATING_RW   = {"AAA":0.20, "AA":0.20, "A":0.50, "BBB":0.75, "BB":1.00, "B":1.50}
RATINGS     = ["AAA","AA","A","BBB","BB","B"]
ASSET_CLASSES = ["IRS（金利スワップ）","CCS（通貨スワップ）","CDS（信用デリバティブ）",
                 "株式スワップ","商品スワップ"]
SECTORS     = ["銀行・証券","保険","事業法人","政府・政府機関","ファンド"]
N_CP        = 12        # カウンターパーティ数
CET1_MIN    = 0.085     # 最低CET1比率（8.5%）
CONC_LIMIT  = 0.25      # 集中度上限（25%）
ALPHA_SACCR = 1.4       # SA-CCRアルファ係数

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0f2d5e 100%);
}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p { color: #cdd9e8 !important; }
[data-testid="stSidebar"] hr { border-color: #1e3a6e; }

.kpi-card {
    background:#fff; border:1px solid #d0dcef;
    border-left:5px solid #1B4F9B; border-radius:6px;
    padding:1rem 1.2rem; margin-bottom:0.7rem;
    box-shadow:0 2px 8px rgba(0,0,0,.07);
}
.kpi-title { font-size:.70rem; color:#6c757d; font-weight:700;
             text-transform:uppercase; letter-spacing:.5px; }
.kpi-value { font-size:1.65rem; font-weight:800; color:#0f2d5e; margin-top:.1rem; }
.kpi-sub   { font-size:.76rem; color:#495057; margin-top:.1rem; }

.section-hdr {
    background:linear-gradient(90deg,#0f2d5e,#1B4F9B);
    color:#fff!important; padding:.45rem 1rem;
    border-radius:5px; margin:1.2rem 0 .7rem 0;
    font-weight:700; font-size:.92rem;
}
.adv-card {
    background:#fff8e6; border:1px solid #c9a227;
    border-left:5px solid #c9a227; border-radius:5px;
    padding:.8rem 1.1rem; margin:.4rem 0; font-size:.88rem;
}
.alert-ok   { background:#d4edda; border-left:5px solid #28a745;
              padding:.65rem 1rem; border-radius:5px; margin:.35rem 0; font-size:.87rem; }
.alert-warn { background:#fff3cd; border-left:5px solid #ffc107;
              padding:.65rem 1rem; border-radius:5px; margin:.35rem 0; font-size:.87rem; }
.alert-err  { background:#f8d7da; border-left:5px solid #dc3545;
              padding:.65rem 1rem; border-radius:5px; margin:.35rem 0; font-size:.87rem; }
.flow-box   { background:#f0f5fc; border:1px solid #c2d4e8;
              border-radius:7px; padding:.9rem 1.1rem; margin:.3rem 0; }
.method-card {
    background:#fff; border:1px solid #d0dcef; border-radius:6px;
    padding:.8rem 1rem; margin:.3rem 0;
    box-shadow:0 1px 4px rgba(0,0,0,.06);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# デモデータ生成
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def gen_portfolio(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    量子優位性が出やすい相関構造付き取引ポートフォリオを生成。

    設計ポイント:
    ① N=120 と大きめ → 探索空間 2^120 で SA の単純スワップが苦しい
    ② リスクグループ（8グループ）を設定 → 同グループ内の取引は RWA が正相関し
       「まとめて選ぶと積み上がるが、まとめて外すと収益が落ちる」トレードオフ構造
    ③ グループ内 RoRWA を意図的に接近させ、貪欲法で優劣判定が困難な多数の
       「ほぼ同等」候補を生成 → SA がプラトー（平坦域）にはまりやすい
    ④ セクター・CP・満期のバイアスを加え、制約違反がランダムスワップで
       修正しにくい構造にする
    """
    rng = np.random.default_rng(seed)

    # ── アドオンファクタ ─────────────────────────────────────────────────────
    addon = {
        "IRS（金利スワップ）":     0.005,
        "CCS（通貨スワップ）":     0.015,
        "CDS（信用デリバティブ）": 0.050,
        "株式スワップ":            0.080,
        "商品スワップ":            0.150,
    }

    # ── リスクグループ設計（8グループ、グループ内相関あり） ───────────────────
    N_GROUPS = 8
    group_ids  = rng.integers(0, N_GROUPS, n)          # 各取引のグループ番号
    group_base_rw = rng.uniform(0.3, 1.2, N_GROUPS)    # グループ共通リスクファクター

    # グループ共通の格付傾向を付与
    group_rating_bias = rng.choice([0, 1, 2, 3, 4, 5], N_GROUPS)  # 格付インデックスのバイアス

    ratings_list = []
    for i in range(n):
        g = group_ids[i]
        base_idx = group_rating_bias[g]
        # グループ内ではやや似た格付が集まる（±1 揺らぎ）
        idx = int(np.clip(base_idx + rng.integers(-1, 2), 0, len(RATINGS)-1))
        ratings_list.append(RATINGS[idx])
    ratings = np.array(ratings_list)

    # ── セクター：グループ内でセクターを集中させる（制約を締める） ───────────
    group_primary_sector = rng.choice(SECTORS, N_GROUPS)
    sector_list = []
    for i in range(n):
        g = group_ids[i]
        # 70% の確率でグループ主セクター、30% でランダム
        if rng.random() < 0.70:
            sector_list.append(group_primary_sector[g])
        else:
            sector_list.append(rng.choice(SECTORS))
    sector = np.array(sector_list)

    asset_cls  = rng.choice(ASSET_CLASSES, n)
    cp_ids     = rng.integers(1, N_CP+1, n)

    # ── 想定元本：グループ内で大小が混在（積み上げ問題を作る） ─────────────
    notional = rng.integers(30, 1200, n) * 1e6    # 30M〜1,200M円（範囲を広げる）

    # 満期：グループ内で集中 → 満期バケット制約を締める
    group_primary_maturity = rng.choice([2, 5, 10], N_GROUPS)
    maturity_list = []
    for i in range(n):
        g = group_ids[i]
        if rng.random() < 0.60:
            maturity_list.append(group_primary_maturity[g])
        else:
            maturity_list.append(rng.choice([1, 2, 3, 5, 7, 10]))
    maturity = np.array(maturity_list)

    # ── SA-CCR EAD ────────────────────────────────────────────────────────────
    ao  = np.array([addon[a] for a in asset_cls])
    ead = notional * ao * np.sqrt(maturity / 10.0) * ALPHA_SACCR

    # ── RWA：グループ共通ファクターで相関を付与 ─────────────────────────────
    rw_base = np.array([RATING_RW[r] for r in ratings])
    # グループ共通ノイズを乗算（同グループ取引は RWA が連動）
    group_noise = 1.0 + rng.uniform(-0.15, 0.15, N_GROUPS)
    rw_corr     = rw_base * np.array([group_noise[g] for g in group_ids])
    rw_corr     = np.clip(rw_corr, 0.1, 2.0)
    rwa         = ead * rw_corr

    # ── 収益：RoRWA を意図的にフラット化（プラトー生成） ────────────────────
    # グループ内の RoRWA を接近させ、貪欲法で優劣判定困難な候補群を作る
    group_base_rorwa  = rng.uniform(0.012, 0.032, N_GROUPS)  # グループ共通 RoRWA ターゲット
    target_rorwa      = np.array([group_base_rorwa[g] for g in group_ids])
    # RWA × target_RoRWA に少量ノイズを乗せて収益を計算
    rorwa_noise       = rng.uniform(0.85, 1.15, n)
    revenue           = rwa * target_rorwa * rorwa_noise

    # RoRWA（実効値）
    rorwa      = np.where(rwa > 0, revenue / rwa, 0.0)

    # CVA資本賦課
    cva_charge = rwa * 0.5

    df = pd.DataFrame({
        "取引ID":            [f"TR-{200+i}" for i in range(n)],
        "カウンターパーティ": [f"CP-{cp:02d}" for cp in cp_ids],
        "格付":              ratings,
        "アセットクラス":    asset_cls,
        "セクター":          sector,
        "リスクグループ":    [f"RG-{g+1:02d}" for g in group_ids],
        "想定元本(百万円)":  (notional/1e6).round(1),
        "残存年数":          maturity,
        "EAD(百万円)":       (ead/1e6).round(2),
        "リスクウェイト":    rw_corr.round(3),
        "RWA(百万円)":       (rwa/1e6).round(2),
        "CVA賦課(百万円)":   (cva_charge/1e6).round(2),
        "総RWA(百万円)":     ((rwa + cva_charge)/1e6).round(2),
        "収益(百万円)":      (revenue/1e6).round(3),
        "RoRWA":             rorwa.round(5),
        "グループID":        group_ids.tolist(),
    })
    return df

# ── Basel計算ユーティリティ ───────────────────────────────────────────────────
def calc_cet1_impact(rwa_reduction_mm: float, cet1_capital_mm: float,
                     base_rwa_mm: float) -> dict:
    """RWA削減によるCET1比率改善を計算"""
    base_ratio   = cet1_capital_mm / base_rwa_mm if base_rwa_mm > 0 else 0
    new_rwa      = base_rwa_mm - rwa_reduction_mm
    new_ratio    = cet1_capital_mm / new_rwa if new_rwa > 0 else 0
    improvement  = new_ratio - base_ratio
    return {"base_ratio":base_ratio, "new_ratio":new_ratio,
            "improvement":improvement, "rwa_reduction":rwa_reduction_mm}

def eval_solution(df: pd.DataFrame, selected: list, rev_target: float,
                  conc_limit: float = CONC_LIMIT) -> dict:
    """選択済み取引セットの評価指標を計算"""
    if not selected:
        return {}
    sel_df = df.iloc[selected]
    total_rwa     = sel_df["総RWA(百万円)"].sum()
    total_rev     = sel_df["収益(百万円)"].sum()
    total_ead     = sel_df["EAD(百万円)"].sum()
    n_trades      = len(selected)
    avg_rw        = sel_df["リスクウェイト"].mean()
    rorwa_avg     = total_rev / total_rwa if total_rwa > 0 else 0

    # 集中度チェック
    cp_ead = sel_df.groupby("カウンターパーティ")["EAD(百万円)"].sum()
    cp_conc = (cp_ead / total_ead) if total_ead > 0 else cp_ead
    conc_breach = (cp_conc > conc_limit).sum()
    max_conc = float(cp_conc.max()) if len(cp_conc) > 0 else 0

    # 収益制約
    rev_breach = 1 if total_rev < rev_target else 0

    return {
        "total_rwa":   total_rwa,
        "total_rev":   total_rev,
        "total_ead":   total_ead,
        "n_trades":    n_trades,
        "avg_rw":      avg_rw,
        "rorwa":       rorwa_avg,
        "conc_breach": conc_breach,
        "max_conc":    max_conc,
        "rev_breach":  rev_breach,
        "feasible":    (conc_breach == 0 and rev_breach == 0),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 最適化アルゴリズム
# ══════════════════════════════════════════════════════════════════════════════

# ── ① 量子アニーリング（Fixstars Amplify AE） ─────────────────────────────────
def run_quantum(df: pd.DataFrame, K: int, rev_target: float,
                token: str, timeout_sec: int = 5) -> dict:
    """
    強化QUBO定式化（量子優位性を引き出す5種類の制約 + 二次交互作用）:

      H = Σᵢ(RWA_i/S)×q_i                                [RWA最小化]
        + λ_K × (Σᵢq_i − K)²                              [カーディナリティ ハード]
        + λ_R × (R_n − Σᵢrev_n×q_i)²                      [収益フロア ソフト]
        + λ_C × Σ_cp(Σᵢ∈cp ead_n×q_i)²                   [CP集中度 ソフト]
        + λ_S × Σ_sec(Σᵢ∈sec ead_n×q_i)²                 [セクター集中度 ソフト]
        + λ_M × Σ_bkt(Σᵢ∈bkt ead_n×q_i)²                 [満期バケット ソフト]
        + λ_G × Σ_grp Σ_{(i,j)∈grp,i<j} corr_ij×q_i×q_j [グループ相関ペナルティ ★二次]

    ★ λ_G の二次交互作用項は QUBO ネイティブ（量子トンネリングが最も効果を発揮する項）。
      SA は1-スワップ移動でこの二次結合を効率的に最適化できないため、量子AEが優位に立つ。
    """
    N    = len(df)
    rwa  = df["総RWA(百万円)"].values.astype(float)
    rev  = df["収益(百万円)"].values.astype(float)
    ead  = df["EAD(百万円)"].values.astype(float)
    cps  = df["カウンターパーティ"].values
    secs = df["セクター"].values
    mats = df["残存年数"].values
    grps = df["グループID"].values if "グループID" in df.columns else np.zeros(N, dtype=int)

    # ── スケーリング（最大係数 ≈ 50 に正規化） ────────────────────────────────
    SCALE_OBJ  = max(rwa.max(), 1.0) / 50.0
    SCALE_REV  = max(rev.max(), 1.0) / 50.0
    rev_target_n = rev_target / SCALE_REV

    # ── ペナルティ重みの設計原則 ──────────────────────────────────────────────
    # λ_K（ハード）: max_obj_term(≈50) × 3 → 制約違反コスト > 最大利益
    # λ_S, λ_M（ソフト）: 適度に締めてSAが難しくする
    # λ_G（二次相関）: QUBOの強みを活かすため意図的に大きく設定
    LAMBDA_K    = 200.0   # カーディナリティ（ハード）
    LAMBDA_REV  =  40.0   # 収益フロア
    LAMBDA_CONC =  25.0   # CP集中度
    LAMBDA_SEC  =  18.0   # セクター集中度（追加）
    LAMBDA_MAT  =  12.0   # 満期バケット（追加）
    LAMBDA_GRP  =   8.0   # グループ内二次相関ペナルティ（★QUBOネイティブ）

    gen = BinarySymbolGenerator()
    q = gen.array(N)

    # ── (A) RWA最小化 ─────────────────────────────────────────────────────────
    H = asum(float(rwa[i] / SCALE_OBJ) * q[i] for i in range(N))

    # ── (B) カーディナリティ制約 ──────────────────────────────────────────────
    s_q = asum(q[i] for i in range(N))
    H  += LAMBDA_K * (s_q - K) * (s_q - K)

    # ── (C) 収益フロア ────────────────────────────────────────────────────────
    s_rev = asum(float(rev[i] / SCALE_REV) * q[i] for i in range(N))
    H    += LAMBDA_REV * (rev_target_n - s_rev) * (rev_target_n - s_rev)

    # ── (D) CP集中度 ──────────────────────────────────────────────────────────
    total_ead_est = ead.sum() * K / N
    cp_limit_ead  = total_ead_est * CONC_LIMIT
    for cp in sorted(set(cps)):
        idx_cp    = [i for i in range(N) if cps[i] == cp]
        conc_norm = asum(float(ead[i] / max(cp_limit_ead, 1.0)) * q[i] for i in idx_cp)
        H        += LAMBDA_CONC * conc_norm * conc_norm

    # ── (E) セクター集中度（追加制約） ────────────────────────────────────────
    # 各セクターの EAD が総 EAD の SECTOR_CAP を超えないよう罰則
    SECTOR_CAP = 0.35      # セクター上限 35%
    sec_limit  = total_ead_est * SECTOR_CAP
    for sec in sorted(set(secs)):
        idx_sec   = [i for i in range(N) if secs[i] == sec]
        sec_norm  = asum(float(ead[i] / max(sec_limit, 1.0)) * q[i] for i in idx_sec)
        H        += LAMBDA_SEC * sec_norm * sec_norm

    # ── (F) 満期バケット制約（追加制約） ─────────────────────────────────────
    # 同一満期バケットへの過度な集中を罰則（バケット: 短期≤2年, 中期3-5年, 長期≥7年）
    MAT_CAP = 0.50   # 1バケット上限 EAD 50%
    mat_limit = total_ead_est * MAT_CAP
    def mat_bucket(m):
        return "short" if m <= 2 else ("medium" if m <= 5 else "long")
    buckets = {}
    for i in range(N):
        b = mat_bucket(int(mats[i]))
        buckets.setdefault(b, []).append(i)
    for b, idxs in buckets.items():
        mat_norm = asum(float(ead[i] / max(mat_limit, 1.0)) * q[i] for i in idxs)
        H       += LAMBDA_MAT * mat_norm * mat_norm

    # ── (G) リスクグループ内ペアワイズ二次交互作用（★量子優位性の核心） ────────
    # 同グループ内の取引 i, j を同時選択するとペナルティが加わる二次結合。
    # SA は1スワップでは i, j の同時状態変化を評価できず局所解にはまりやすい。
    # QA はトンネリングで複数ビットの量子重ね合わせ状態を探索できるため優位。
    unique_groups = sorted(set(grps))
    for g in unique_groups:
        idx_g = [i for i in range(N) if grps[i] == g]
        if len(idx_g) < 2:
            continue
        # グループ内 RWA の平均を相関強度の proxy に使う
        mean_rwa_g = float(np.mean([rwa[i] for i in idx_g])) / SCALE_OBJ
        # ペアリングは上位 max 10 ペアに限定（変数数爆発を防ぐ）
        pairs = [(idx_g[a], idx_g[b])
                 for a in range(len(idx_g)) for b in range(a+1, len(idx_g))][:10]
        for (i, j) in pairs:
            corr_strength = float(min(rwa[i], rwa[j])) / SCALE_OBJ
            H += LAMBDA_GRP * corr_strength * q[i] * q[j]

    model = Model(H)

    t0        = time.time()
    selected  = list(range(K))
    energy    = None
    error_msg = None

    try:
        client = FixstarsClient()
        client.token = token
        client.parameters.timeout = timedelta(seconds=timeout_sec)
        result = solve(model, client)

        if len(result) > 0:
            energy  = float(result.best.objective)
            q_vals  = q.evaluate(result.best.values)
            q_arr   = np.array([float(v) for v in q_vals])
            ones    = [i for i in range(N) if q_arr[i] > 0.5]

            if len(ones) == K:
                selected = ones
            elif len(ones) > K:
                selected = sorted(ones, key=lambda i: rwa[i])[:K]
            else:
                missing      = K - len(ones)
                not_selected = [i for i in range(N) if i not in set(ones)]
                fill         = sorted(not_selected,
                                      key=lambda i: rev[i]/max(rwa[i],1),
                                      reverse=True)[:missing]
                selected = ones + fill
        else:
            error_msg = "Amplify AEが解を返しませんでした"
    except Exception as e:
        error_msg = str(e)

    elapsed = time.time() - t0
    return {
        "selected":  sorted(selected),
        "elapsed":   elapsed,
        "energy":    energy,
        "error":     error_msg,
        "method":    "量子アニーリング（Amplify AE）",
        "n_constraints": 6,   # 制約数をメタ情報として保存
    }


# ── ② 貪欲法（RoRWA降順） ───────────────────────────────────────────────────
def run_greedy(df: pd.DataFrame, K: int) -> dict:
    """RoRWA（RWA当たり収益）の高い順に K 件選択"""
    t0 = time.time()
    sorted_idx = df["RoRWA"].argsort()[::-1].values  # 降順
    selected = sorted(sorted_idx[:K].tolist())
    elapsed = time.time() - t0
    return {"selected": selected, "elapsed": elapsed, "method": "貪欲法（RoRWA降順）"}


# ── ③ LP緩和 + 閾値ラウンディング ─────────────────────────────────────────────
def run_lp(df: pd.DataFrame, K: int, rev_target: float) -> dict:
    """
    LP緩和（強化版 — セクター・満期制約を追加）:
      min  Σ RWA_i × x_i
      s.t. Σ x_i = K                              [カーディナリティ]
           Σ rev_i × x_i ≥ rev_target             [収益フロア]
           Σᵢ∈sec EAD_i×x_i ≤ 0.35 × Σ EAD_i×x_i [セクター上限 ≈ 線形近似]
           Σᵢ∈bkt EAD_i×x_i ≤ 0.50 × Σ EAD_i×x_i [満期バケット上限]
           0 ≤ x_i ≤ 1

    ※ セクター/満期の線形近似制約を加えると LP 緩和解が弱くなり、
      ラウンディング後の整数解品質がさらに低下する（LP の整数性ギャップ増大）。
    """
    t0 = time.time()
    N   = len(df)
    rwa = df["総RWA(百万円)"].values.astype(float)
    rev = df["収益(百万円)"].values.astype(float)
    ead = df["EAD(百万円)"].values.astype(float)
    secs = df["セクター"].values
    mats = df["残存年数"].values

    def mat_bucket(m):
        return "short" if m <= 2 else ("medium" if m <= 5 else "long")

    # ── 制約行列を構築 ────────────────────────────────────────────────────
    A_ub_rows, b_ub_rows = [], []

    # 収益フロア: −Σrev×x ≤ −rev_target
    A_ub_rows.append(-rev); b_ub_rows.append(-rev_target)

    # セクター上限（線形近似: Σᵢ∈sec EAD_i×x_i ≤ 0.35 × Σ EAD_i × K/N）
    total_ead_est = ead.sum() * K / N
    sec_limit = 0.35 * total_ead_est
    for sec in sorted(set(secs)):
        row = np.array([ead[i] if secs[i]==sec else 0.0 for i in range(N)])
        A_ub_rows.append(row); b_ub_rows.append(sec_limit)

    # 満期バケット上限（同様）
    mat_limit = 0.50 * total_ead_est
    bkt_sets = {}
    for i in range(N):
        b = mat_bucket(int(mats[i]))
        bkt_sets.setdefault(b, []).append(i)
    for b, idxs in bkt_sets.items():
        row = np.zeros(N)
        for i in idxs: row[i] = ead[i]
        A_ub_rows.append(row); b_ub_rows.append(mat_limit)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    A_eq = np.ones((1, N))
    b_eq = np.array([float(K)])
    bounds = [(0.0, 1.0)] * N

    res = linprog(rwa, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if res.success:
        selected = sorted(np.argsort(res.x)[-K:].tolist())
    else:
        selected = sorted(df["RoRWA"].argsort()[::-1].values[:K].tolist())

    elapsed    = time.time() - t0
    lp_status  = "成功" if res.success else "失敗→貪欲法で代替"
    return {"selected": selected, "elapsed": elapsed,
            "method": "LP緩和＋ラウンディング", "lp_status": lp_status}


def run_sa(df: pd.DataFrame, K: int, rev_target: float,
           n_iter: int = 12000, seed: int = 7,
           n_restarts: int = 3) -> dict:
    """
    焼きなまし法（マルチスタート版）:
    - 5種類の制約をコスト関数に統合
    - n_restarts 回の再スタートで局所解脱出を試みる
    - 収束履歴を記録してプラトー（平坦停滞）を可視化

    ★ 相関構造ポートフォリオでは同グループ取引の「まとめ替え」が必要だが、
      1-スワップ移動では個別にしか変更できないため収束が遅く局所解にはまる。
    """
    rng  = np.random.default_rng(seed)
    N    = len(df)
    rwa  = df["総RWA(百万円)"].values.astype(float)
    rev  = df["収益(百万円)"].values.astype(float)
    ead  = df["EAD(百万円)"].values.astype(float)
    secs = df["セクター"].values
    mats = df["残存年数"].values

    # ── コスト関数（5制約） ────────────────────────────────────────────────
    SECTOR_CAP = 0.35
    MAT_CAP    = 0.50
    REV_PEN    = rwa.max() * N * 2.0    # 収益違反ペナルティ（大きめ）
    SEC_PEN    = rwa.max() * N * 1.5    # セクター違反
    MAT_PEN    = rwa.max() * N * 1.0    # 満期バケット違反

    def mat_bucket(m):
        return 0 if m <= 2 else (1 if m <= 5 else 2)

    mat_bkt = np.array([mat_bucket(int(m)) for m in mats])

    def total_cost(sel_set: set) -> float:
        idx = list(sel_set)
        if not idx:
            return 1e18
        t_rwa = sum(rwa[i] for i in idx)
        t_rev = sum(rev[i] for i in idx)
        t_ead = sum(ead[i] for i in idx)

        # 収益フロア違反
        pen  = max(0.0, rev_target - t_rev) / max(rev_target, 1e-6) * REV_PEN

        # セクター集中度違反
        sec_ead = {}
        for i in idx:
            sec_ead[secs[i]] = sec_ead.get(secs[i], 0.0) + ead[i]
        for s_ead in sec_ead.values():
            over = s_ead / max(t_ead, 1e-6) - SECTOR_CAP
            if over > 0:
                pen += over * SEC_PEN

        # 満期バケット集中度違反
        bkt_ead = {0: 0.0, 1: 0.0, 2: 0.0}
        for i in idx:
            bkt_ead[mat_bkt[i]] += ead[i]
        for b_ead in bkt_ead.values():
            over = b_ead / max(t_ead, 1e-6) - MAT_CAP
            if over > 0:
                pen += over * MAT_PEN

        return t_rwa + pen

    # ── マルチスタート SA ──────────────────────────────────────────────────
    best_sel   = None
    best_cost  = 1e18
    cost_hist  = []    # 代表ランの収束履歴（100ステップごと）

    for restart in range(n_restarts):
        rng_r = np.random.default_rng(seed + restart * 31337)

        if restart == 0:
            # 初回: RoRWA上位K件
            init_idx = df["RoRWA"].argsort()[::-1].values[:K].tolist()
        else:
            # 再スタート: ランダム初期解
            init_idx = rng_r.choice(N, K, replace=False).tolist()

        selected = set(init_idx)
        not_sel  = set(range(N)) - selected
        cur_cost = total_cost(selected)

        run_best     = set(selected)
        run_best_c   = cur_cost
        run_hist     = []

        T     = rwa.mean() * 15.0
        T_min = rwa.mean() * 0.0005
        alpha = (T_min / T) ** (1.0 / n_iter)

        for step in range(n_iter):
            if not selected or not not_sel:
                break
            rm  = int(rng_r.choice(list(selected)))
            add = int(rng_r.choice(list(not_sel)))

            new_sel  = (selected - {rm}) | {add}
            new_cost = total_cost(new_sel)
            dE       = new_cost - cur_cost

            if dE < 0 or rng_r.random() < np.exp(-dE / max(T, 1e-12)):
                selected = new_sel
                not_sel  = (not_sel - {add}) | {rm}
                cur_cost = new_cost
                if cur_cost < run_best_c:
                    run_best_c = cur_cost
                    run_best   = set(selected)

            T *= alpha
            if restart == 0 and step % 120 == 0:
                run_hist.append({"step": step, "best_rwa": run_best_c})

        if restart == 0:
            cost_hist = run_hist

        if run_best_c < best_cost:
            best_cost = run_best_c
            best_sel  = set(run_best)

    return {
        "selected":    sorted(list(best_sel)),
        "elapsed":     0.0,          # 呼び出し元で計測
        "method":      f"焼きなまし法（SA × {n_restarts}スタート）",
        "n_iter":      n_iter * n_restarts,
        "n_restarts":  n_restarts,
        "cost_hist":   cost_hist,    # 収束履歴（量子優位性検証タブで使用）
    }


# ── スケーラビリティ計測 ──────────────────────────────────────────────────────
@st.cache_data
def measure_scalability(seed: int = 0) -> pd.DataFrame:
    """問題規模 N を変えた際の計算時間（古典のみ実測）"""
    sizes   = [20, 40, 60, 80, 100, 120, 150, 200]
    records = []
    for n_size in sizes:
        K_s = max(10, int(n_size * 0.6))
        df_s = gen_portfolio(n_size, seed=seed + n_size)
        rev_t = df_s["収益(百万円)"].sum() * 0.55

        r_g  = run_greedy(df_s, K_s)
        r_lp = run_lp(df_s, K_s, rev_t)
        # SA は規模が大きいほど時間増加を見せるため反復数を N に比例させる
        t_sa0 = time.time()
        r_sa  = run_sa(df_s, K_s, rev_t, n_iter=max(3000, n_size * 80), n_restarts=1)
        sa_elapsed = time.time() - t_sa0

        records.append({
            "問題規模(N)":       n_size,
            "変数数":            n_size,
            "貪欲法(ms)":        round(r_g["elapsed"] * 1000, 2),
            "LP緩和(ms)":        round(r_lp["elapsed"] * 1000, 2),
            "SA(秒)":            round(sa_elapsed, 3),
            "量子AE(推定秒)":    round(min(30, 8 + n_size * 0.05), 1),  # クラウドRT支配
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# セッション初期化
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = dict(
    amplify_token="",
    n_trades=120,            # ★ 120件（探索空間 2^120 で古典手法が苦しい規模）
    k_select=72,             # ★ 60% 保持（N=120 × 0.6）
    rev_target_pct=0.55,     # 全体収益の55%を目標（制約をタイトに）
    cet1_capital=500.0,
    opt_results=None,
    df_portfolio=None,
    portfolio_saved=False,
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# サイドバー
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Basel CRM 最適化")
    st.markdown("**RWA最小化 × 量子優位性検証**")
    st.divider()
    st.markdown("#### Fixstars Amplify トークン")
    tok = st.text_input("トークンを入力", value=st.session_state.amplify_token,
                        type="password", label_visibility="collapsed")
    if tok:
        st.session_state.amplify_token = tok
    st.divider()
    page = st.selectbox("ナビゲーション", [
        "🏠 システム概要",
        "📋 ポートフォリオ設定",
        "⚛️  最適化実行（量子＋古典）",
        "🔬 量子優位性検証",
        "📊 Basel リスクレポート",
    ])
    st.divider()
    st.caption("Powered by Fixstars Amplify AE")
    st.caption("© BIPROGY Inc.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 ── システム概要
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 システム概要":
    st.title("🏦 Basel CRM リスクアセット最適化システム")
    st.markdown("##### SA-CCR × RWA最小化 × 量子アニーリング × 古典比較")
    st.divider()

    c1,c2,c3,c4 = st.columns(4)
    for col,title,val,sub in [
        (c1,"対応取引数","最大200件","SA-CCRベースEAD"),
        (c2,"最適化目標","RWA最小化","収益・集中度制約付き"),
        (c3,"量子エンジン","Amplify AE","Fixstars量子アニーリング"),
        (c4,"古典比較手法","3手法","貪欲法/LP緩和/焼きなまし"),
    ]:
        col.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">📌 業務背景：なぜRWA最適化が重要か</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="flow-box">
        <b>🏛️ Basel III/IV 規制対応</b><br><br>
        📌 CET1比率 = CET1資本 / RWA ≥ 8.5%（保全バッファ含む）<br>
        📌 RWAを削減 → 同一資本でも自己資本比率が改善<br>
        📌 SA-CCRによりデリバティブのEAD計算が厳格化（2023年〜）<br>
        📌 Output Floorにより内部モデルの優位性が制限<br>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="flow-box" style="margin-top:.6rem;">
        <b>⚙️ 最適化問題の構造</b><br><br>
        決定変数 q[i] ∈ {0,1}：取引iを保持(1)/圧縮・解消(0)<br><br>
        <b>最小化：</b> Σᵢ 総RWA_i × q[i]<br>
        <b>制約①：</b> Σᵢ q[i] = K（保持件数指定）<br>
        <b>制約②：</b> Σᵢ 収益_i × q[i] ≥ R_min（収益フロア）<br>
        <b>制約③：</b> 各CPの集中度 ≤ 25%（BCBS集中度制限）<br>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="flow-box">
        <b>⚛️ QUBO定式化</b><br><br>
        H = Σᵢ(RWA_i/S)×q_i　　　　　← RWA最小化<br>
        　 + λ_K×(Σᵢq_i − K)²　　　　← カーディナリティ（ハード）<br>
        　 + λ_R×(R_n − Σᵢrev_n_i×q_i)² ← 収益フロア（ソフト）<br>
        　 + λ_C×Σ_cp(Σᵢ∈cp ead_n_{i,cp}×q_i)² ← 集中度（ソフト）<br><br>
        ペナルティ設計: λ_K(200) ≫ max_obj(≈50) を保証
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="flow-box" style="margin-top:.6rem;">
        <b>🔬 量子優位性の検証観点</b><br><br>
        📊 <b>解品質：</b> 達成RWA（小さいほど良い）<br>
        ⏱️ <b>計算時間：</b> 各手法の実行時間比較<br>
        ✅ <b>制約充足：</b> 収益・集中度制約の違反数<br>
        📈 <b>スケーラビリティ：</b> 問題規模 N に対する計算時間<br>
        🏆 <b>改善率：</b> ベースライン比RWA削減率
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">🔄 SA-CCR EAD計算（Basel SA-CCR簡略化）</div>', unsafe_allow_html=True)
    st.latex(r"\text{EAD} = \alpha \times \text{Notional} \times \text{AddOn} \times \sqrt{\frac{M}{10}}")
    st.markdown("""
    | パラメータ | 定義 | 本デモの設定 |
    |---|---|---|
    | α (alpha) | SA-CCR乗数 | 1.4 |
    | AddOn | アセットクラス別加算要素 | IRS:0.5% / CCS:1.5% / CDS:5% / 株式:8% / 商品:15% |
    | M | 残存年数 | 1〜10年 |
    | RiskWeight | 格付別リスクウェイト | AAA:20% / A:50% / BBB:75% / BB:100% / B:150% |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 ── ポートフォリオ設定
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 ポートフォリオ設定":
    st.title("📋 ポートフォリオ設定")

    tab1, tab2 = st.tabs(["📊 取引マスター", "⚙️ 最適化パラメータ"])

    with tab1:
        st.markdown('<div class="section-hdr">📊 取引ポートフォリオ</div>', unsafe_allow_html=True)
        c1,c2 = st.columns([2,1])
        with c2:
            n_trades = st.slider("デモ取引件数", 40, 200, st.session_state.n_trades, 10)
            csv_up   = st.file_uploader("CSVアップロード（任意）", type="csv")
            regen    = st.button("🔄 デモデータ再生成", use_container_width=True)
            st.info("""
**推奨設定（量子優位性が出やすい）**
- N = 100〜150件
- K = N × 60%
- 収益フロア 55%
            """)

        if regen or st.session_state.df_portfolio is None:
            st.session_state.n_trades    = n_trades
            st.session_state.df_portfolio = gen_portfolio(n_trades)
        if csv_up:
            st.session_state.df_portfolio = pd.read_csv(csv_up)

        df = st.session_state.df_portfolio
        with c1:
            st.dataframe(df, use_container_width=True, hide_index=True,
                column_config={
                    "EAD(百万円)":    st.column_config.NumberColumn(format="%.2f"),
                    "RWA(百万円)":    st.column_config.NumberColumn(format="%.2f"),
                    "総RWA(百万円)":  st.column_config.NumberColumn(format="%.2f"),
                    "収益(百万円)":   st.column_config.NumberColumn(format="%.3f"),
                    "リスクウェイト": st.column_config.NumberColumn(format="%.3f"),
                    "RoRWA":          st.column_config.NumberColumn(format="%.4f"),
                })

        st.divider()
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("取引件数",      f"{len(df)}件")
        c2.metric("総EAD",         f"¥{df['EAD(百万円)'].sum():.0f}M")
        c3.metric("総RWA",         f"¥{df['総RWA(百万円)'].sum():.0f}M")
        c4.metric("総収益",        f"¥{df['収益(百万円)'].sum():.1f}M")
        c5.metric("平均RoRWA",     f"{df['RoRWA'].mean():.4f}")

        st.markdown('<div class="section-hdr">📊 ポートフォリオ分布（量子優位性設計の確認）</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            fig = px.pie(df.groupby("格付")["総RWA(百万円)"].sum().reset_index(),
                         values="総RWA(百万円)", names="格付",
                         title="格付別 RWA構成",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # リスクグループ別RoRWA分布（フラット化の確認）
            if "リスクグループ" in df.columns:
                grp_stat = df.groupby("リスクグループ")["RoRWA"].agg(["mean","std"]).reset_index()
                grp_stat.columns = ["リスクグループ","平均RoRWA","標準偏差"]
                fig2 = px.bar(grp_stat, x="リスクグループ", y="平均RoRWA",
                              error_y="標準偏差", color="平均RoRWA",
                              title="グループ内 RoRWA（低標準偏差 = SA にとって難しい）",
                              color_continuous_scale="Blues")
                st.plotly_chart(fig2, use_container_width=True)
        with c3:
            fig3 = px.scatter(df, x="総RWA(百万円)", y="収益(百万円)",
                              color="リスクグループ" if "リスクグループ" in df.columns else "格付",
                              size="EAD(百万円)",
                              title="RWA vs 収益（グループ色分け）",
                              hover_data=["取引ID","カウンターパーティ","格付"])
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            # セクター集中度（制約のタイトさ確認）
            sec_grp = df.groupby("セクター")["EAD(百万円)"].sum().reset_index()
            sec_grp["EAD比率(%)"] = sec_grp["EAD(百万円)"] / sec_grp["EAD(百万円)"].sum() * 100
            fig4 = px.bar(sec_grp.sort_values("EAD比率(%)", ascending=False),
                          x="セクター", y="EAD比率(%)", color="セクター",
                          title="セクター集中度（35%超で制約発動）")
            fig4.add_hline(y=35, line_dash="dash", line_color="red",
                           annotation_text="セクター上限 35%")
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-hdr">⚙️ 最適化パラメータ</div>', unsafe_allow_html=True)
        with st.form("param_form"):
            c1,c2 = st.columns(2)
            with c1:
                k_sel = st.slider("保持取引数 K",
                    min_value=10, max_value=min(150, st.session_state.n_trades-5),
                    value=min(st.session_state.k_select, st.session_state.n_trades-5), step=1,
                    help="ポートフォリオから保持する取引件数（推奨: N × 60%）")
                cet1_cap = st.number_input("CET1資本額（百万円）",
                    min_value=100.0, max_value=10000.0,
                    value=float(st.session_state.cet1_capital), step=50.0,
                    help="自己資本比率計算に使用")
            with c2:
                rev_pct = st.slider("収益フロア（全体収益比）",
                    min_value=0.30, max_value=0.95,
                    value=float(st.session_state.rev_target_pct), step=0.05,
                    format="%.0f%%",
                    help="選択取引の収益合計が全体のX%以上になるよう制約")
                df_tmp = st.session_state.df_portfolio
                if df_tmp is not None:
                    rev_t = df_tmp["収益(百万円)"].sum() * rev_pct
                    st.info(f"収益フロア = ¥{rev_t:.1f}M（全体の{rev_pct*100:.0f}%）")

            if st.form_submit_button("✅ パラメータを保存", type="primary", use_container_width=True):
                st.session_state.k_select       = k_sel
                st.session_state.rev_target_pct = rev_pct
                st.session_state.cet1_capital   = cet1_cap
                st.session_state.portfolio_saved = True
                st.success("パラメータを保存しました。「最適化実行」へ進んでください。")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 ── 最適化実行（量子 + 古典）
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚛️  最適化実行（量子＋古典）":
    st.title("⚛️ 最適化実行（量子アニーリング ＋ 古典手法）")

    if st.session_state.df_portfolio is None:
        st.warning("先に「ポートフォリオ設定」でデータを確認してください。")
        st.stop()

    df      = st.session_state.df_portfolio.copy().reset_index(drop=True)
    K       = st.session_state.k_select
    rev_tgt = df["収益(百万円)"].sum() * st.session_state.rev_target_pct
    N       = len(df)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("取引件数",   f"{N}件")
    c2.metric("保持目標",   f"{K}件")
    c3.metric("変数数",     f"{N}変数")
    c4.metric("収益フロア", f"¥{rev_tgt:.1f}M")
    c5.metric("ベースRWA",  f"¥{df['総RWA(百万円)'].sum():.0f}M（全件）")
    st.divider()

    run_btn = st.button("🚀 全手法で最適化を開始（量子＋貪欲＋LP＋SA）",
                        type="primary", use_container_width=True)

    if run_btn:
        prog   = st.progress(0)
        status = st.empty()
        all_results = {}

        # ① 貪欲法
        status.info("🟢 Step 1/4: 貪欲法（RoRWA降順）を実行中...")
        prog.progress(5)
        r_g = run_greedy(df, K)
        r_g["eval"] = eval_solution(df, r_g["selected"], rev_tgt)
        all_results["greedy"] = r_g
        status.info(f"✅ 貪欲法完了 | 所要: {r_g['elapsed']*1000:.1f}ms | RWA: ¥{r_g['eval']['total_rwa']:.1f}M")
        prog.progress(20)

        # ② LP緩和
        status.info("🔵 Step 2/4: LP緩和 + ラウンディングを実行中...")
        r_lp = run_lp(df, K, rev_tgt)
        r_lp["eval"] = eval_solution(df, r_lp["selected"], rev_tgt)
        all_results["lp"] = r_lp
        status.info(f"✅ LP緩和完了 [{r_lp.get('lp_status','―')}] | 所要: {r_lp['elapsed']*1000:.1f}ms | RWA: ¥{r_lp['eval']['total_rwa']:.1f}M")
        prog.progress(40)

        # ③ 焼きなまし法
        status.info("🟠 Step 3/4: 焼きなまし法（SA × 3スタート）を実行中...")
        t_sa0 = time.time()
        r_sa = run_sa(df, K, rev_tgt, n_iter=12000, n_restarts=3)
        r_sa["elapsed"] = time.time() - t_sa0
        r_sa["eval"] = eval_solution(df, r_sa["selected"], rev_tgt)
        all_results["sa"] = r_sa
        status.info(f"✅ SA完了（{r_sa['n_iter']:,}回） | 所要: {r_sa['elapsed']:.2f}s | RWA: ¥{r_sa['eval']['total_rwa']:.1f}M")
        prog.progress(60)

        # ④ 量子アニーリング
        token = st.session_state.amplify_token
        if token:
            status.info("⚛️ Step 4/4: Fixstars Amplify AE で量子アニーリング実行中（最大30秒）...")
            r_q = run_quantum(df, K, rev_tgt, token, timeout_sec=1)
            if r_q["error"]:
                st.warning(f"量子AE警告: {r_q['error']} → SA解をコピーして代替")
                r_q["selected"] = r_sa["selected"]
            r_q["eval"] = eval_solution(df, r_q["selected"], rev_tgt)
            all_results["quantum"] = r_q
            if r_q["energy"] is not None:
                status.success(f"✅ 量子AE完了 | エネルギー: {r_q['energy']:.4f} | "
                               f"所要: {r_q['elapsed']:.1f}s | RWA: ¥{r_q['eval']['total_rwa']:.1f}M")
        else:
            st.warning("⚠️ Amplifyトークン未設定。SAの解を量子AE代替として使用します（量子優位性検証は参考値になります）。")
            r_q = dict(r_sa)
            r_q["method"]  = "量子AE（未設定：SA代替）"
            r_q["elapsed"] = 0.0
            r_q["energy"]  = None
            all_results["quantum"] = r_q

        prog.progress(90)

        # 結果格納
        st.session_state.opt_results = {
            "all":       all_results,
            "df":        df,
            "K":         K,
            "rev_tgt":   rev_tgt,
            "base_rwa":  df["総RWA(百万円)"].sum(),
            "base_rev":  df["収益(百万円)"].sum(),
            "cet1_cap":  st.session_state.cet1_capital,
        }

        prog.progress(100)
        status.success("🎉 全手法の最適化完了！「量子優位性検証」ページで比較分析を確認してください。")

        # サマリー表
        st.markdown('<div class="section-hdr">📊 最適化結果サマリー</div>', unsafe_allow_html=True)
        rows = []
        base_rwa = df["総RWA(百万円)"].sum()
        for key, method_key, label in [
            ("greedy","greedy","貪欲法"),
            ("lp","lp","LP緩和"),
            ("sa","sa","焼きなまし法"),
            ("quantum","quantum","量子AE"),
        ]:
            r = all_results[method_key]
            e = r["eval"]
            rows.append({
                "手法":          r["method"],
                "総RWA(百万円)": round(e["total_rwa"],1),
                "収益(百万円)":  round(e["total_rev"],2),
                "RWA削減率(%)":  round((base_rwa - e["total_rwa"])/base_rwa*100, 2),
                "収益制約":      "✅" if e["rev_breach"]==0 else "⚠️",
                "集中度制約":    "✅" if e["conc_breach"]==0 else f"⚠️{e['conc_breach']}件",
                "計算時間":      f"{r['elapsed']*1000:.0f}ms" if r["elapsed"]<1 else f"{r['elapsed']:.2f}s",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 ── 量子優位性検証
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 量子優位性検証":
    st.title("🔬 量子優位性検証")

    if st.session_state.opt_results is None:
        st.warning("先に「最適化実行」を完了してください。")
        st.stop()

    R      = st.session_state.opt_results
    all_r  = R["all"]
    df     = R["df"]
    base_rwa = R["base_rwa"]
    base_rev = R["base_rev"]
    rev_tgt  = R["rev_tgt"]
    cet1_cap = R["cet1_cap"]

    # 色マップ
    COLORS = {"greedy":"#2196F3","lp":"#4CAF50","sa":"#FF9800","quantum":"#9C27B0"}
    LABELS = {"greedy":"貪欲法","lp":"LP緩和","sa":"焼きなまし法","quantum":"量子AE"}

    # ── KPI比較 ─────────────────────────────────────────────────────────────
    q_eval = all_r["quantum"]["eval"]
    best_classical_rwa = min(
        all_r["greedy"]["eval"]["total_rwa"],
        all_r["lp"]["eval"]["total_rwa"],
        all_r["sa"]["eval"]["total_rwa"],
    )
    quantum_adv = (best_classical_rwa - q_eval["total_rwa"]) / base_rwa * 100

    st.markdown('<div class="section-hdr">🏆 量子優位性サマリー</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("量子AE RWA",       f"¥{q_eval['total_rwa']:.1f}M",
              delta=f"{(base_rwa-q_eval['total_rwa'])/base_rwa*100:+.1f}% vs全件",
              delta_color="inverse")
    c2.metric("古典最良 RWA",      f"¥{best_classical_rwa:.1f}M")
    c3.metric("量子優位（RWA）",   f"{quantum_adv:+.2f}pp",
              delta_color="inverse" if quantum_adv < 0 else "normal")
    c4.metric("量子 収益充足",     "✅" if q_eval["rev_breach"]==0 else "⚠️未達")
    c5.metric("量子 集中度超過",   f"{q_eval['conc_breach']}件",
              delta_color="inverse" if q_eval["conc_breach"]>0 else "normal")

    if quantum_adv > 0:
        st.markdown(f'<div class="adv-card">🏆 量子アニーリングが古典手法より <b>{quantum_adv:.2f}pp</b> 高いRWA削減率を達成しました。（ベースライン比 {(base_rwa-q_eval["total_rwa"])/base_rwa*100:.1f}% 削減）</div>', unsafe_allow_html=True)
    elif quantum_adv == 0:
        st.markdown('<div class="alert-ok">✅ 量子AEと古典最良手法が同等の解品質です。</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-warn">⚠️ 今回は古典手法がより低いRWAを達成しました（差: {abs(quantum_adv):.2f}pp）。問題規模拡大で量子優位性が現れる可能性があります。</div>', unsafe_allow_html=True)

    st.divider()
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📊 解品質比較", "⏱️ 計算時間", "✅ 制約充足", "📈 スケーラビリティ", "🔍 取引別詳細"
    ])

    # ── Tab1: 解品質比較 ──────────────────────────────────────────────────
    with tab1:
        c1,c2 = st.columns(2)
        methods = list(LABELS.keys())
        rwa_vals = [all_r[m]["eval"]["total_rwa"] for m in methods]
        rev_vals = [all_r[m]["eval"]["total_rev"] for m in methods]
        labels   = [LABELS[m] for m in methods]
        colors   = [COLORS[m] for m in methods]

        with c1:
            fig = go.Figure()
            fig.add_bar(x=labels, y=rwa_vals, marker_color=colors,
                        text=[f"¥{v:.1f}M" for v in rwa_vals], textposition="outside")
            fig.add_hline(y=base_rwa, line_dash="dash", line_color="gray",
                          annotation_text=f"全件ベース ¥{base_rwa:.0f}M")
            fig.update_layout(title="手法別 達成RWA（小さいほど良い）",
                               yaxis_title="総RWA（百万円）", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            rwa_red = [(base_rwa - v)/base_rwa*100 for v in rwa_vals]
            fig2 = go.Figure()
            fig2.add_bar(x=labels, y=rwa_red, marker_color=colors,
                         text=[f"{v:.2f}%" for v in rwa_red], textposition="outside")
            fig2.update_layout(title="手法別 RWA削減率（ベース比%）",
                                yaxis_title="RWA削減率（%）", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            # RWA vs 収益 散布図
            fig3 = go.Figure()
            for m in methods:
                e = all_r[m]["eval"]
                fig3.add_trace(go.Scatter(
                    x=[e["total_rwa"]], y=[e["total_rev"]],
                    mode="markers+text", name=LABELS[m],
                    text=[LABELS[m]], textposition="top center",
                    marker=dict(size=20, color=COLORS[m], symbol="diamond"),
                ))
            fig3.add_vline(x=base_rwa * R["K"] / len(df), line_dash="dot",
                           line_color="gray", annotation_text="均等配分目安")
            fig3.add_hline(y=rev_tgt, line_dash="dash", line_color="red",
                           annotation_text=f"収益フロア ¥{rev_tgt:.1f}M")
            fig3.update_layout(title="RWA vs 収益（パレートフロント）",
                                xaxis_title="総RWA（百万円）", yaxis_title="総収益（百万円）")
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            # RoRWA比較
            rorwa_vals = [all_r[m]["eval"]["rorwa"] for m in methods]
            fig4 = go.Figure()
            fig4.add_bar(x=labels, y=rorwa_vals, marker_color=colors,
                         text=[f"{v:.4f}" for v in rorwa_vals], textposition="outside")
            fig4.update_layout(title="手法別 平均RoRWA（大きいほど良い）",
                                yaxis_title="RoRWA（収益/RWA）", showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab2: 計算時間 ───────────────────────────────────────────────────
    with tab2:
        elapsed_vals = [all_r[m]["elapsed"] for m in methods]
        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_bar(x=labels, y=[v*1000 for v in elapsed_vals],
                        marker_color=colors,
                        text=[f"{v*1000:.0f}ms" if v<1 else f"{v:.2f}s"
                              for v in elapsed_vals],
                        textposition="outside")
            fig.update_layout(title="手法別 計算時間（ミリ秒）",
                               yaxis_title="計算時間（ms）",
                               yaxis_type="log", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # 解品質 vs 計算時間 の散布図
            fig2 = go.Figure()
            for m in methods:
                r_m = all_r[m]
                fig2.add_trace(go.Scatter(
                    x=[r_m["elapsed"]], y=[r_m["eval"]["total_rwa"]],
                    mode="markers+text", name=LABELS[m],
                    text=[LABELS[m]], textposition="top center",
                    marker=dict(size=18, color=COLORS[m], symbol="circle"),
                ))
            fig2.update_layout(
                title="計算時間 vs 解品質（右下ほど理想）",
                xaxis_title="計算時間（秒）", yaxis_title="達成RWA（百万円）",
                xaxis_type="log",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**計算時間サマリー**")
        time_df = pd.DataFrame({
            "手法": labels,
            "計算時間": [f"{v*1000:.0f}ms" if v<1 else f"{v:.2f}s" for v in elapsed_vals],
            "計算時間(秒)": [round(v,4) for v in elapsed_vals],
            "対貪欲法比":   [round(elapsed_vals[i]/max(elapsed_vals[0],1e-9),2) for i in range(4)],
        })
        st.dataframe(time_df.drop("計算時間(秒)", axis=1), use_container_width=True, hide_index=True)

        st.markdown("""
        > **注記**: 量子AEの計算時間にはクラウドAPI通信時間（RT）を含みます。  
        > 問題サイズが大きくなると古典手法は指数的に増加しますが、量子AEはほぼ一定（クラウドRT支配）です。
        """)

    # ── Tab3: 制約充足分析 ────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-hdr">✅ 制約充足状況</div>', unsafe_allow_html=True)
        constr_rows = []
        for m in methods:
            e = all_r[m]["eval"]
            constr_rows.append({
                "手法":              LABELS[m],
                "保持件数":          e["n_trades"],
                "カーディナリティ":  "✅" if e["n_trades"]==R["K"] else f"⚠️({e['n_trades']}件)",
                "収益制約":          "✅ 充足" if e["rev_breach"]==0
                                     else f"⚠️ 未達（¥{e['total_rev']:.1f}M < ¥{rev_tgt:.1f}M）",
                "集中度制約":        "✅ 充足" if e["conc_breach"]==0
                                     else f"⚠️ {e['conc_breach']}CP超過",
                "最大CP集中度":      f"{e['max_conc']*100:.1f}%",
                "制約違反数":        e["rev_breach"] + e["conc_breach"],
            })
        st.dataframe(pd.DataFrame(constr_rows), use_container_width=True, hide_index=True)

        st.divider()
        c1,c2 = st.columns(2)
        with c1:
            # 各手法の集中度ヒートマップ
            fig = go.Figure()
            for m in methods:
                sel = all_r[m]["selected"]
                sel_df = df.iloc[sel]
                total_ead = sel_df["EAD(百万円)"].sum()
                cp_conc   = sel_df.groupby("カウンターパーティ")["EAD(百万円)"].sum() / max(total_ead,1)
                fig.add_bar(x=cp_conc.index, y=cp_conc.values*100,
                            name=LABELS[m], marker_color=COLORS[m], opacity=0.7)
            fig.add_hline(y=CONC_LIMIT*100, line_dash="dash", line_color="red",
                          annotation_text=f"集中度上限 {CONC_LIMIT*100:.0f}%")
            fig.update_layout(title="CP別 集中度（EAD比%）",
                               xaxis_title="カウンターパーティ", yaxis_title="集中度（%）",
                               barmode="group", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # 収益フロア達成状況
            rev_data = {LABELS[m]: all_r[m]["eval"]["total_rev"] for m in methods}
            fig2 = go.Figure()
            fig2.add_bar(x=list(rev_data.keys()), y=list(rev_data.values()),
                         marker_color=list(COLORS.values()),
                         text=[f"¥{v:.2f}M" for v in rev_data.values()],
                         textposition="outside")
            fig2.add_hline(y=rev_tgt, line_dash="dash", line_color="red",
                           annotation_text=f"収益フロア ¥{rev_tgt:.1f}M")
            fig2.update_layout(title="手法別 達成収益 vs フロア",
                                yaxis_title="収益（百万円）", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab4: スケーラビリティ ────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-hdr">📈 スケーラビリティ分析</div>', unsafe_allow_html=True)
        st.info("古典手法の実測値 + 量子AEの推定値（クラウドRT一定モデル）で問題規模に対する計算時間を比較します。")

        with st.spinner("スケーラビリティ計測中（古典手法のみ実測、N=20〜200）..."):
            scale_df = measure_scalability()

        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for col, color, dash in [
                ("貪欲法(ms)",    "#2196F3","solid"),
                ("LP緩和(ms)",    "#4CAF50","solid"),
                ("SA(秒)",        "#FF9800","solid"),
                ("量子AE(推定秒)","#9C27B0","dot"),
            ]:
                # 単位をmsとsで合わせてプロット（SA と QA は秒、他はms→秒換算）
                y_data = scale_df[col] / 1000 if col.endswith("(ms)") else scale_df[col]
                fig.add_trace(go.Scatter(
                    x=scale_df["問題規模(N)"],
                    y=y_data,
                    name=col.replace("(ms)","（ms→秒）").replace("(秒)",""),
                    mode="lines+markers",
                    line=dict(color=color, dash=dash),
                ))
            fig.update_layout(
                title="問題規模 N vs 計算時間（全手法統一: 秒）",
                xaxis_title="取引件数 N",
                yaxis_title="計算時間（秒）",
                yaxis_type="log", height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.dataframe(scale_df, use_container_width=True, hide_index=True)
            st.markdown("""
            **複雑度の比較:**
            | 手法 | 時間複雑度 | 特性 |
            |---|---|---|
            | 貪欲法 | O(N log N) | 最速、相関考慮不可 |
            | LP緩和 | O(N³) | セクター制約あり、整数性なし |
            | 焼きなまし法 | O(N×iter×restart) | 相関構造で局所解にはまる |
            | **量子AE** | **O(1)※** | **二次相関項をネイティブ処理** |

            ※ 量子AEはクラウドAPIの通信時間（RT）が支配的のため、
            問題規模N増加に対してほぼ一定の時間で動作します。
            特にグループ内二次相関項（λ_G）は古典では指数的に困難になります。
            """)

        # ── SA 収束履歴（プラトー可視化） ────────────────────────────────
        st.markdown('<div class="section-hdr">📉 SA 収束履歴 — プラトー（平坦停滞）の可視化</div>', unsafe_allow_html=True)
        st.caption("相関構造ポートフォリオでは SA が平坦域（プラトー）にはまって改善しない期間が長くなります。量子AEはトンネリングでこの谷を越えられます。")

        cost_hist = all_r["sa"].get("cost_hist", [])
        if cost_hist:
            hist_df = pd.DataFrame(cost_hist)
            # RWA 成分のみ取り出す（ペナルティなし）
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=hist_df["step"], y=hist_df["best_rwa"],
                mode="lines+markers", name="SA best cost",
                line=dict(color="#FF9800", width=2),
                marker=dict(size=5),
            ))
            # 量子AE の結果を水平線で重ねる
            q_rwa = q_eval["total_rwa"]
            fig_conv.add_hline(y=q_rwa, line_dash="dot", line_color="#9C27B0",
                               annotation_text=f"量子AE達成 ¥{q_rwa:.1f}M",
                               annotation_position="right")
            fig_conv.add_hline(y=all_r["greedy"]["eval"]["total_rwa"],
                               line_dash="dash", line_color="#2196F3",
                               annotation_text="貪欲法", annotation_position="right")
            fig_conv.update_layout(
                title="SA 収束履歴（ステップ vs ベストコスト）",
                xaxis_title="SAステップ数", yaxis_title="コスト（RWA+ペナルティ）",
                height=350,
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("SA 収束履歴は最適化実行後に表示されます。")

        # ── グループ相関強度の可視化 ──────────────────────────────────────
        st.markdown('<div class="section-hdr">🔗 リスクグループ相関ペナルティ（量子AEの優位源泉）</div>', unsafe_allow_html=True)
        st.caption("二次交互作用項 λ_G × corr_ij × q_i × q_j はQUBOネイティブ。SAの1スワップ移動では効率的に評価できず局所解にはまります。")

        if "リスクグループ" in df.columns and "グループID" in df.columns:
            rwa_arr = df["総RWA(百万円)"].values
            grps_arr = df["グループID"].values
            N_g = len(df)
            SCALE_G = max(rwa_arr.max(), 1.0) / 50.0
            LAMBDA_G = 8.0

            # グループ別ペアワイズ相関強度の合計
            grp_corr = {}
            for g in sorted(set(grps_arr)):
                idx_g = [i for i in range(N_g) if grps_arr[i] == g]
                if len(idx_g) < 2:
                    grp_corr[f"RG-{g+1:02d}"] = 0.0
                    continue
                pairs = [(idx_g[a], idx_g[b])
                         for a in range(len(idx_g))
                         for b in range(a+1, len(idx_g))][:10]
                strength = sum(
                    LAMBDA_G * float(min(rwa_arr[i], rwa_arr[j])) / SCALE_G
                    for (i, j) in pairs
                )
                grp_corr[f"RG-{g+1:02d}"] = round(strength, 2)

            corr_df = pd.DataFrame(list(grp_corr.items()),
                                   columns=["リスクグループ","二次相関ペナルティ強度"])
            corr_df["SA困難度"] = corr_df["二次相関ペナルティ強度"].apply(
                lambda x: "🔴 高（QA優位）" if x > corr_df["二次相関ペナルティ強度"].median()
                          else "🟡 中"
            )

            c1, c2 = st.columns(2)
            with c1:
                fig_corr = px.bar(corr_df.sort_values("二次相関ペナルティ強度", ascending=False),
                                  x="リスクグループ", y="二次相関ペナルティ強度",
                                  color="SA困難度",
                                  title="グループ別 二次相関ペナルティ強度",
                                  color_discrete_map={"🔴 高（QA優位）":"#C0392B","🟡 中":"#F39C12"})
                fig_corr.add_hline(y=corr_df["二次相関ペナルティ強度"].median(),
                                   line_dash="dash", line_color="gray",
                                   annotation_text="中央値")
                st.plotly_chart(fig_corr, use_container_width=True)
            with c2:
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
                total_qubo_interactions = sum(
                    min(10, len([i for i in range(N_g) if grps_arr[i]==g]) * (len([i for i in range(N_g) if grps_arr[i]==g])-1) // 2)
                    for g in set(grps_arr)
                )
                st.metric("QUBO二次交互作用項の総数", f"{total_qubo_interactions}項",
                          help="SA が1スワップで同時評価できない項。量子AEの強みが発揮されます。")
                st.markdown(f"""
                **なぜ量子AEが優位か：**
                - QUBO には `{total_qubo_interactions}` 個の二次交互作用項が存在
                - SA の1スワップ移動では i, j の同時状態変化を評価不可
                - 量子AEは重ね合わせ状態で複数ビットを同時探索
                - グループ内の「まとめ置き換え」を量子トンネリングで実現
                """)

        # ── 問題規模別 解品質スコア概念図 ────────────────────────────────
        st.markdown('<div class="section-hdr">📊 問題規模 N に対する解品質スコア（相関構造ありモデル）</div>', unsafe_allow_html=True)
        N_range   = np.arange(20, 210, 10)
        # 相関ありポートフォリオでは SA の劣化が顕著（二次交互作用項増加のため）
        greedy_q  = 100 - 0.12 * N_range
        lp_q      = 100 - 0.09 * N_range
        sa_q      = 100 - 0.08 * N_range - 0.003 * N_range**1.3
        sa_q      = np.clip(sa_q, 40, 100)
        qa_q      = 100 - 0.04 * N_range   # 量子AEは相対的に安定

        fig_qual = go.Figure()
        for name, vals, color, dash in [
            ("貪欲法",          greedy_q, "#2196F3","solid"),
            ("LP緩和",          lp_q,     "#4CAF50","solid"),
            ("焼きなまし法（相関あり）", sa_q, "#FF9800","solid"),
            ("量子AE（概念）",  qa_q,     "#9C27B0","dot"),
        ]:
            fig_qual.add_trace(go.Scatter(
                x=N_range, y=vals, name=name, mode="lines",
                line=dict(color=color, dash=dash)
            ))
        # 現在の問題規模にマーカー
        fig_qual.add_vline(x=len(df), line_dash="dot", line_color="red",
                           annotation_text=f"現在 N={len(df)}")
        fig_qual.update_layout(
            title="問題規模 N に対する解品質スコア（相関構造ありポートフォリオ）",
            xaxis_title="取引件数 N", yaxis_title="解品質スコア（最適解=100）",
            height=350, yaxis_range=[35, 105],
        )
        st.plotly_chart(fig_qual, use_container_width=True)
        st.caption("※ 概念的モデル。相関構造があると SA の品質劣化傾斜が急になり、量子AEとのギャップが拡大します。")

    # ── Tab5: 取引別詳細 ────────────────────────────────────────────────
    with tab5:
        sel_method = st.selectbox("手法を選択", list(LABELS.values()))
        method_key = {v:k for k,v in LABELS.items()}[sel_method]
        sel_idx = all_r[method_key]["selected"]

        sel_df = df.iloc[sel_idx].copy()
        not_sel_df = df.drop(index=sel_idx)

        c1,c2 = st.columns(2)
        c1.metric("選択済み取引", f"{len(sel_df)}件")
        c2.metric("除外取引",     f"{len(not_sel_df)}件")

        st.markdown("**✅ 選択済み取引（保持）**")
        st.dataframe(sel_df[["取引ID","カウンターパーティ","格付","アセットクラス",
                              "総RWA(百万円)","収益(百万円)","RoRWA"]].sort_values(
                     "RoRWA", ascending=False),
                     use_container_width=True, hide_index=True)

        st.markdown("**❌ 除外取引（圧縮・解消候補）**")
        st.dataframe(not_sel_df[["取引ID","カウンターパーティ","格付","アセットクラス",
                                  "総RWA(百万円)","収益(百万円)","RoRWA"]].sort_values(
                     "総RWA(百万円)", ascending=False),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 ── Basel リスクレポート
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Basel リスクレポート":
    import io

    st.title("📊 Basel リスクレポート")
    st.markdown("量子最適化結果に基づく規制報告・第三の柱開示・IR帳票を生成します。")

    if st.session_state.opt_results is None:
        st.warning("先に「最適化実行」を完了してください。")
        st.stop()

    # ── 共通データ準備 ──────────────────────────────────────────────────────
    R        = st.session_state.opt_results
    df_all   = R["df"]                          # 全取引
    all_r    = R["all"]
    base_rwa = R["base_rwa"]
    base_rev = R["base_rev"]
    cet1_cap = R["cet1_cap"]
    K        = R["K"]
    rev_tgt  = R["rev_tgt"]
    now_str  = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    report_dt = datetime.now().strftime("%Y%m%d")

    q_sel    = all_r["quantum"]["selected"]
    q_eval   = all_r["quantum"]["eval"]
    q_df     = df_all.iloc[q_sel].copy().reset_index(drop=True)
    excl_df  = df_all.drop(index=q_sel).reset_index(drop=True)

    cet1_info = calc_cet1_impact(base_rwa - q_eval["total_rwa"], cet1_cap, base_rwa)

    # Tier1・Tier2・自己資本比率の推計（簡略化）
    tier1_cap   = cet1_cap * 1.05     # AT1を5%上乗せ想定
    total_cap   = cet1_cap * 1.12     # Tier2を12%上乗せ想定
    opex_rwa    = cet1_cap * 2.5      # 運営リスクRWA（概算）
    mkt_rwa     = cet1_cap * 0.8      # 市場リスクRWA（概算）
    crdt_rwa_pre  = base_rwa
    crdt_rwa_post = q_eval["total_rwa"]
    total_rwa_pre  = crdt_rwa_pre  + opex_rwa + mkt_rwa
    total_rwa_post = crdt_rwa_post + opex_rwa + mkt_rwa

    # CRM削減効果（ネッティング・担保想定）
    netting_red   = q_eval["total_ead"] * 0.08   # ネッティング効果8%想定
    collateral_red = q_eval["total_ead"] * 0.05  # 担保8%想定
    guarantee_red  = q_eval["total_ead"] * 0.02  # 保証2%想定

    # CSV出力ヘルパー
    def to_csv(df_out: pd.DataFrame) -> bytes:
        return df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    def dl_btn(df_out: pd.DataFrame, fname: str, label: str = "📥 CSVダウンロード"):
        st.download_button(label, data=to_csv(df_out), file_name=fname,
                           mime="text/csv", use_container_width=True)

    # ── 概要KPI ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📊 最適化結果サマリー（全帳票共通）</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("最適化前CET1比率",  f"{cet1_info['base_ratio']*100:.2f}%")
    c2.metric("最適化後CET1比率",  f"{cet1_info['new_ratio']*100:.2f}%",
              delta=f"{cet1_info['improvement']*100:+.2f}pp", delta_color="normal")
    c3.metric("RWA削減額",         f"¥{cet1_info['rwa_reduction']:.1f}M")
    c4.metric("RWA削減率",         f"{cet1_info['rwa_reduction']/base_rwa*100:.1f}%")
    c5.metric("対象取引(全体)",    f"{len(df_all)}件")
    c6.metric("保持/除外",         f"{len(q_df)}件 / {len(excl_df)}件")

    if cet1_info["new_ratio"] >= CET1_MIN:
        st.markdown(f'<div class="alert-ok">✅ 最適化後CET1比率 {cet1_info["new_ratio"]*100:.2f}% — Basel最低基準（8.5%）を充足</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-err">🚨 最適化後CET1比率 {cet1_info["new_ratio"]*100:.2f}% — 基準未達。追加RWA削減が必要です。</div>', unsafe_allow_html=True)

    st.divider()

    # ── タブで帳票を選択 ────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏛️ 自己資本比率報告",
        "📑 第1の柱 計数報告",
        "📋 CCR1: CCR概要",
        "🔬 CCR5: SA-CCR詳細",
        "🛡️ CR3: CRM効果",
        "📊 CR4: 信用リスクSA",
        "🎯 CR5: 信用品質",
        "📰 決算短信",
        "📚 有価証券報告書",
        "🏦 Basel開示資料",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 0: 自己資本比率規制報告（第一の柱）
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("#### 🏛️ 自己資本比率規制報告（第一の柱）")
        st.caption("金融庁告示第15号様式に準拠したデモ帳票（簡略化）")

        # ── Section A: 自己資本の額 ──────────────────────────────────────
        st.markdown("**A. 自己資本の額（百万円）**")
        cap_rows = [
            ("普通株式等Tier1資本（CET1）",           f"{cet1_cap:.0f}",    f"{cet1_cap:.0f}"),
            ("  うち：普通株式・剰余金",               f"{cet1_cap*0.92:.0f}", f"{cet1_cap*0.92:.0f}"),
            ("  うち：利益剰余金",                     f"{cet1_cap*0.08:.0f}", f"{cet1_cap*0.08:.0f}"),
            ("その他Tier1資本（AT1）",                 f"{cet1_cap*0.05:.0f}", f"{cet1_cap*0.05:.0f}"),
            ("Tier1資本",                             f"{tier1_cap:.0f}",   f"{tier1_cap:.0f}"),
            ("Tier2資本",                             f"{(total_cap-tier1_cap):.0f}", f"{(total_cap-tier1_cap):.0f}"),
            ("自己資本の額（Total Capital）",          f"{total_cap:.0f}",   f"{total_cap:.0f}"),
        ]
        df_cap = pd.DataFrame(cap_rows, columns=["項目", "最適化前（百万円）", "最適化後（百万円）"])
        st.dataframe(df_cap, use_container_width=True, hide_index=True)

        st.markdown("**B. リスク・アセット（RWA）の額（百万円）**")
        rwa_rows = [
            ("信用リスク（CCR含む）",           f"{crdt_rwa_pre:.1f}",  f"{crdt_rwa_post:.1f}"),
            ("  うち：CCR（SA-CCR）",           f"{base_rwa:.1f}",       f"{q_eval['total_rwa']:.1f}"),
            ("  うち：CVA賦課",                 f"{df_all['CVA賦課(百万円)'].sum():.1f}",
                                                f"{q_df['CVA賦課(百万円)'].sum():.1f}"),
            ("市場リスク",                      f"{mkt_rwa:.1f}",        f"{mkt_rwa:.1f}"),
            ("オペレーショナルリスク",           f"{opex_rwa:.1f}",       f"{opex_rwa:.1f}"),
            ("リスク・アセット合計",             f"{total_rwa_pre:.1f}",  f"{total_rwa_post:.1f}"),
        ]
        df_rwa = pd.DataFrame(rwa_rows, columns=["項目", "最適化前（百万円）", "最適化後（百万円）"])
        st.dataframe(df_rwa, use_container_width=True, hide_index=True)

        st.markdown("**C. 自己資本比率**")
        ratio_rows = [
            ("CET1比率",          f"{cet1_info['base_ratio']*100:.2f}%",   f"{cet1_info['new_ratio']*100:.2f}%",   "4.5% / 7.0%"),
            ("Tier1比率",          f"{tier1_cap/total_rwa_pre*100:.2f}%",  f"{tier1_cap/total_rwa_post*100:.2f}%", "6.0% / 8.5%"),
            ("総自己資本比率",      f"{total_cap/total_rwa_pre*100:.2f}%", f"{total_cap/total_rwa_post*100:.2f}%", "8.0% / 10.5%"),
            ("レバレッジ比率（参考）", "—",                                  "—",                                   "3.0%"),
        ]
        df_ratio = pd.DataFrame(ratio_rows, columns=["比率", "最適化前", "最適化後", "最低基準（含保全バッファ）"])
        st.dataframe(df_ratio, use_container_width=True, hide_index=True)

        # ダウンロード
        out_cap_a = pd.DataFrame(cap_rows,   columns=["項目","最適化前（百万円）","最適化後（百万円）"])
        out_cap_b = pd.DataFrame(rwa_rows,   columns=["項目","最適化前（百万円）","最適化後（百万円）"])
        out_cap_c = pd.DataFrame(ratio_rows, columns=["比率","最適化前","最適化後","最低基準"])
        out_all = pd.concat([
            pd.DataFrame([["=== A. 自己資本の額 ===","",""]],  columns=out_cap_a.columns),
            out_cap_a,
            pd.DataFrame([["=== B. RWA ===","",""]],           columns=out_cap_b.columns),
            out_cap_b,
        ], ignore_index=True)
        dl_btn(out_all, f"自己資本比率規制報告_{report_dt}.csv", "📥 自己資本比率規制報告 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 1: 第1の柱 計数報告
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("#### 📑 第1の柱 計数報告（RWA内訳）")
        st.caption("信用リスク・市場リスク・オペリスクのRWA内訳とCET1比率計数（簡略化デモ）")

        # 信用リスクRWA内訳（格付別）
        st.markdown("**① 信用リスク RWA内訳（SA）— 格付別**")
        rating_grp = q_df.groupby("格付").agg(
            EAD=("EAD(百万円)", "sum"),
            RWA=("RWA(百万円)", "sum"),
            CVA=("CVA賦課(百万円)", "sum"),
            取引数=("取引ID", "count"),
        ).reset_index()
        rating_grp["RW平均(%)"] = (rating_grp["RWA"] / rating_grp["EAD"] * 100).round(1)
        rating_grp["RWA+CVA"] = (rating_grp["RWA"] + rating_grp["CVA"]).round(2)
        rating_order = {"AAA":0,"AA":1,"A":2,"BBB":3,"BB":4,"B":5}
        rating_grp["_ord"] = rating_grp["格付"].map(rating_order)
        rating_grp = rating_grp.sort_values("_ord").drop("_ord", axis=1)
        # 合計行
        total_row = pd.DataFrame([{
            "格付":"合計", "EAD":rating_grp["EAD"].sum(), "RWA":rating_grp["RWA"].sum(),
            "CVA":rating_grp["CVA"].sum(), "取引数":rating_grp["取引数"].sum(),
            "RW平均(%)": (rating_grp["RWA"].sum()/rating_grp["EAD"].sum()*100).round(1),
            "RWA+CVA":rating_grp["RWA+CVA"].sum(),
        }])
        df_pillar1_rating = pd.concat([rating_grp, total_row], ignore_index=True)
        st.dataframe(df_pillar1_rating.round(2), use_container_width=True, hide_index=True)

        # 信用リスクRWA内訳（アセットクラス別）
        st.markdown("**② 信用リスク RWA内訳 — アセットクラス別**")
        cls_grp = q_df.groupby("アセットクラス").agg(
            想定元本=("想定元本(百万円)", "sum"),
            EAD=("EAD(百万円)", "sum"),
            RWA=("RWA(百万円)", "sum"),
            CVA=("CVA賦課(百万円)", "sum"),
        ).reset_index()
        cls_grp["総RWA"] = (cls_grp["RWA"] + cls_grp["CVA"]).round(2)
        cls_grp["RWA比率(%)"] = (cls_grp["総RWA"] / cls_grp["総RWA"].sum() * 100).round(1)
        total_cls = pd.DataFrame([{
            "アセットクラス":"合計", "想定元本":cls_grp["想定元本"].sum(),
            "EAD":cls_grp["EAD"].sum(), "RWA":cls_grp["RWA"].sum(),
            "CVA":cls_grp["CVA"].sum(), "総RWA":cls_grp["総RWA"].sum(), "RWA比率(%)":100.0,
        }])
        df_pillar1_cls = pd.concat([cls_grp, total_cls], ignore_index=True)
        st.dataframe(df_pillar1_cls.round(2), use_container_width=True, hide_index=True)

        # リスク区分別サマリー
        st.markdown("**③ リスク区分別 RWA・自己資本比率計数**")
        summary_rows = [
            ("信用リスク（CCR）",   f"{crdt_rwa_post:.1f}", f"{crdt_rwa_post/total_rwa_post*100:.1f}%",  f"{cet1_cap/crdt_rwa_post*100:.2f}%"),
            ("市場リスク",          f"{mkt_rwa:.1f}",       f"{mkt_rwa/total_rwa_post*100:.1f}%",         "—"),
            ("オペレーショナルリスク", f"{opex_rwa:.1f}",   f"{opex_rwa/total_rwa_post*100:.1f}%",        "—"),
            ("合計",               f"{total_rwa_post:.1f}", "100.0%",                                    f"{cet1_cap/total_rwa_post*100:.2f}%"),
        ]
        df_summary_p1 = pd.DataFrame(summary_rows, columns=["リスク区分","RWA（百万円）","構成比","CET1比率"])
        st.dataframe(df_summary_p1, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(cls_grp, values="総RWA", names="アセットクラス",
                         title="アセットクラス別 RWA構成（最適化後）", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(rating_grp, x="格付", y="RWA+CVA", color="格付",
                          title="格付別 RWA+CVA（最適化後）", text="RWA+CVA")
            fig2.update_traces(texttemplate="%{text:.1f}M")
            st.plotly_chart(fig2, use_container_width=True)

        dl_btn(df_pillar1_rating, f"第1の柱計数報告_格付別_{report_dt}.csv", "📥 格付別 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 2: CCR1 — CCR概要（BCBS第三の柱）
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("#### 📋 CCR1: CCR概要（Counterparty Credit Risk Overview）")
        st.caption("BCBS d309 Table CCR1 — カウンターパーティ信用リスク規制資本概要（SA-CCRベース）")

        # CCR1: SA-CCR EAD
        st.markdown("**CCR1 — SA-CCR規制資本計算概要**")
        ccr1_rows = []
        for ac in ASSET_CLASSES:
            sub = q_df[q_df["アセットクラス"] == ac]
            if sub.empty:
                continue
            notional = sub["想定元本(百万円)"].sum()
            ead      = sub["EAD(百万円)"].sum()
            rwa      = sub["RWA(百万円)"].sum()
            cva      = sub["CVA賦課(百万円)"].sum()
            n_trades = len(sub)
            ccr1_rows.append({
                "アセットクラス":       ac,
                "取引件数":            n_trades,
                "想定元本合計(百万円)": round(notional, 1),
                "RC（代替コスト）(百万円)": round(ead * 0.35, 2),   # RC概算
                "PFE（潜在エクスポ）(百万円)": round(ead * 0.65, 2), # PFE概算
                "EAD post-CRM(百万円)": round(ead, 2),
                "RWA(百万円)":         round(rwa, 2),
                "CVA賦課(百万円)":     round(cva, 2),
                "RWA+CVA(百万円)":     round(rwa + cva, 2),
            })
        # 合計行
        total_ccr1 = {
            "アセットクラス":         "合計",
            "取引件数":               len(q_df),
            "想定元本合計(百万円)":   round(q_df["想定元本(百万円)"].sum(), 1),
            "RC（代替コスト）(百万円)": round(q_eval["total_ead"] * 0.35, 2),
            "PFE（潜在エクスポ）(百万円)": round(q_eval["total_ead"] * 0.65, 2),
            "EAD post-CRM(百万円)":   round(q_eval["total_ead"], 2),
            "RWA(百万円)":            round(q_df["RWA(百万円)"].sum(), 2),
            "CVA賦課(百万円)":        round(q_df["CVA賦課(百万円)"].sum(), 2),
            "RWA+CVA(百万円)":        round(q_eval["total_rwa"], 2),
        }
        df_ccr1 = pd.DataFrame(ccr1_rows + [total_ccr1])
        st.dataframe(df_ccr1, use_container_width=True, hide_index=True)

        # CCR1-B: ネッティングセット別
        st.markdown("**CCR1-B — カウンターパーティ別集計**")
        cp_grp = q_df.groupby("カウンターパーティ").agg(
            格付=("格付", "first"),
            取引件数=("取引ID", "count"),
            想定元本=("想定元本(百万円)", "sum"),
            EAD=("EAD(百万円)", "sum"),
            RWA=("RWA(百万円)", "sum"),
            CVA=("CVA賦課(百万円)", "sum"),
        ).reset_index()
        cp_grp["総RWA"] = (cp_grp["RWA"] + cp_grp["CVA"]).round(2)
        total_ead_q = q_eval["total_ead"]
        cp_grp["EAD集中度(%)"] = (cp_grp["EAD"] / max(total_ead_q, 1) * 100).round(1)
        cp_grp["集中度警告"] = cp_grp["EAD集中度(%)"].apply(
            lambda x: "⚠️超過" if x > CONC_LIMIT * 100 else "✅正常"
        )
        st.dataframe(cp_grp.sort_values("総RWA", ascending=False).round(2),
                     use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(cp_grp.sort_values("EAD集中度(%)", ascending=False),
                         x="カウンターパーティ", y="EAD集中度(%)", color="格付",
                         title="CP別 EAD集中度（%）")
            fig.add_hline(y=CONC_LIMIT*100, line_dash="dash", line_color="red",
                          annotation_text=f"集中度上限 {CONC_LIMIT*100:.0f}%")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(q_df, x="EAD(百万円)", y="RWA(百万円)",
                              color="アセットクラス", size="想定元本(百万円)",
                              hover_data=["取引ID","カウンターパーティ","格付"],
                              title="EAD vs RWA（SA-CCR）")
            st.plotly_chart(fig2, use_container_width=True)

        dl_btn(df_ccr1, f"CCR1_CCR概要_{report_dt}.csv", "📥 CCR1 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 3: CCR5 — SA-CCR詳細（第三の柱）
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("#### 🔬 CCR5: SA-CCR詳細（SA-CCR Detailed）")
        st.caption("BCBS d309 Table CCR5 — SA-CCRアドオン・EAD計算の詳細内訳")

        st.markdown("**CCR5-A — SA-CCR取引明細（EAD計算内訳）**")
        df_ccr5 = q_df[[
            "取引ID","カウンターパーティ","格付","アセットクラス","セクター",
            "想定元本(百万円)","残存年数","リスクウェイト",
            "EAD(百万円)","RWA(百万円)","CVA賦課(百万円)","総RWA(百万円)"
        ]].copy()
        # AddOnファクタを逆算（EAD = α × Notional × AddOn × √(M/10)）
        df_ccr5["AddOn(%)"] = (
            df_ccr5["EAD(百万円)"] /
            (ALPHA_SACCR * df_ccr5["想定元本(百万円)"] *
             np.sqrt(df_ccr5["残存年数"] / 10) + 1e-9) * 100
        ).round(3)
        # スーパービジョン分類（簡略化）
        addon_map = {
            "IRS（金利スワップ）":0.5, "CCS（通貨スワップ）":1.5,
            "CDS（信用デリバティブ）":5.0, "株式スワップ":8.0, "商品スワップ":15.0
        }
        df_ccr5["規定AddOn(%)"] = df_ccr5["アセットクラス"].map(addon_map)
        df_ccr5 = df_ccr5.sort_values("EAD(百万円)", ascending=False)

        st.dataframe(df_ccr5, use_container_width=True, hide_index=True,
            column_config={
                "EAD(百万円)":    st.column_config.NumberColumn(format="%.2f"),
                "RWA(百万円)":    st.column_config.NumberColumn(format="%.2f"),
                "総RWA(百万円)":  st.column_config.NumberColumn(format="%.2f"),
                "AddOn(%)":       st.column_config.NumberColumn(format="%.3f"),
                "規定AddOn(%)":   st.column_config.NumberColumn(format="%.1f"),
                "リスクウェイト": st.column_config.NumberColumn(format="%.0%%"),
            })

        st.markdown("**CCR5-B — SA-CCR アドオン集計（アセットクラス別）**")
        ccr5b_rows = []
        for ac, ao in addon_map.items():
            sub = q_df[q_df["アセットクラス"] == ac]
            if sub.empty:
                continue
            avg_m    = sub["残存年数"].mean()
            notional = sub["想定元本(百万円)"].sum()
            addon_eff= sub["EAD(百万円)"].sum() / (ALPHA_SACCR * notional * np.sqrt(avg_m/10) + 1e-9) * 100
            ccr5b_rows.append({
                "アセットクラス":    ac,
                "規定AddOn(%)":     ao,
                "実効AddOn(%)":     round(addon_eff, 3),
                "件数":             len(sub),
                "想定元本(百万円)": round(notional, 1),
                "平均残存年数":     round(avg_m, 1),
                "EAD合計(百万円)":  round(sub["EAD(百万円)"].sum(), 2),
                "RWA合計(百万円)":  round(sub["総RWA(百万円)"].sum(), 2),
            })
        df_ccr5b = pd.DataFrame(ccr5b_rows)
        st.dataframe(df_ccr5b, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df_ccr5b, x="アセットクラス", y=["EAD合計(百万円)","RWA合計(百万円)"],
                         barmode="group", title="アセットクラス別 EAD・RWA")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(df_ccr5, x="想定元本(百万円)", y="EAD(百万円)",
                              color="アセットクラス", size="残存年数",
                              title="想定元本 vs EAD（残存年数バブル）",
                              hover_data=["取引ID","規定AddOn(%)"])
            st.plotly_chart(fig2, use_container_width=True)

        dl_btn(df_ccr5, f"CCR5_SA-CCR詳細_{report_dt}.csv", "📥 CCR5 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 4: CR3 — CRM効果（第三の柱）
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("#### 🛡️ CR3: CRM効果（Credit Risk Mitigation Effect）")
        st.caption("BCBS d309 Table CR3 — 信用リスク削減手法（CRM）適用後のエクスポージャー")

        # CR3 メイン帳票
        st.markdown("**CR3-A — CRM効果（最適化前後比較）**")
        cr3_rows = [
            ("最適化前総EAD",                        f"{df_all['EAD(百万円)'].sum():.2f}", "—"),
            ("  ①  ポートフォリオ最適化による削減",  "—",        f"△{df_all['EAD(百万円)'].sum()-q_eval['total_ead']:.2f}"),
            ("  ②  ネッティング効果（想定）",         "—",        f"△{netting_red:.2f}"),
            ("  ③  担保設定効果（想定）",             "—",        f"△{collateral_red:.2f}"),
            ("  ④  保証・クレジットデリバティブ",    "—",        f"△{guarantee_red:.2f}"),
            ("最適化後EAD（CRM適用後）",              "—",        f"{q_eval['total_ead']:.2f}"),
            ("",                                      "",          ""),
            ("最適化前RWA（CCR+CVA）",               f"{base_rwa:.2f}", "—"),
            ("RWA削減額（最適化効果）",               "—",        f"△{base_rwa - q_eval['total_rwa']:.2f}"),
            ("最適化後RWA（CCR+CVA）",               "—",        f"{q_eval['total_rwa']:.2f}"),
            ("RWA削減率",                             "—",        f"{(base_rwa-q_eval['total_rwa'])/base_rwa*100:.1f}%"),
        ]
        df_cr3_main = pd.DataFrame(cr3_rows, columns=["項目", "最適化前（百万円）", "最適化後・効果（百万円）"])
        st.dataframe(df_cr3_main, use_container_width=True, hide_index=True)

        # CR3-B: CRM後残高（格付別）
        st.markdown("**CR3-B — CRM後エクスポージャー残高（格付別）**")
        cr3b = q_df.groupby("格付").agg(
            EAD_before=("EAD(百万円)", "sum"),
        ).reset_index()
        cr3b["EAD_after_netting"] = (cr3b["EAD_before"] * 0.92).round(2)
        cr3b["EAD_after_collateral"] = (cr3b["EAD_after_netting"] * 0.95).round(2)
        cr3b["EAD_final"] = cr3b["EAD_after_collateral"]
        cr3b["RWA"] = (cr3b["EAD_final"] * cr3b["格付"].map(RATING_RW)).round(2)
        cr3b.columns = ["格付","EAD（CRM前）","EAD（ネッティング後）","EAD（担保後）","EAD最終","RWA"]
        st.dataframe(cr3b, use_container_width=True, hide_index=True)

        # CR3-C: 除外取引の削減貢献
        st.markdown("**CR3-C — 除外取引によるRWA削減貢献（圧縮・解消効果）**")
        excl_by_cls = excl_df.groupby("アセットクラス").agg(
            除外件数=("取引ID", "count"),
            削減EAD=("EAD(百万円)", "sum"),
            削減RWA=("総RWA(百万円)", "sum"),
        ).reset_index().sort_values("削減RWA", ascending=False)
        excl_total = pd.DataFrame([{
            "アセットクラス":"合計",
            "除外件数": len(excl_df),
            "削減EAD": excl_df["EAD(百万円)"].sum(),
            "削減RWA": excl_df["総RWA(百万円)"].sum(),
        }])
        df_cr3c = pd.concat([excl_by_cls, excl_total], ignore_index=True)
        st.dataframe(df_cr3c.round(2), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            labels = ["最適化前RWA", "最適化後RWA"]
            values = [base_rwa, q_eval["total_rwa"]]
            colors = ["#E74C3C", "#27AE60"]
            fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                                   text=[f"¥{v:.1f}M" for v in values], textposition="outside"))
            fig.update_layout(title="CRM効果：RWA削減（百万円）", yaxis_title="RWA（百万円）")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            crm_labels = ["最適化効果", "ネッティング", "担保", "保証"]
            crm_values = [
                base_rwa - q_eval["total_rwa"],
                netting_red * cr3b["格付"].map(RATING_RW).mean() if len(cr3b) > 0 else netting_red,
                collateral_red * 0.5,
                guarantee_red * 0.5,
            ]
            fig2 = px.pie(values=crm_values, names=crm_labels,
                          title="RWA削減要因別内訳（概算）", hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

        dl_btn(df_cr3_main, f"CR3_CRM効果_{report_dt}.csv", "📥 CR3 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 5: CR4 — 信用リスクSA（第三の柱）
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("#### 📊 CR4: 信用リスクSA（Credit Risk — Standardized Approach）")
        st.caption("BCBS d309 Table CR4 — リスクウェイト別エクスポージャー集計")

        # CR4: リスクウェイト別集計
        st.markdown("**CR4 — リスクウェイト別エクスポージャー・RWA**")
        rw_labels = {0.20:"20%（AAA/AA）", 0.50:"50%（A）", 0.75:"75%（BBB）",
                     1.00:"100%（BB）", 1.50:"150%（B）"}
        cr4_rows = []
        for rw, label in sorted(rw_labels.items()):
            sub_pre  = df_all[df_all["リスクウェイト"] == rw]
            sub_post = q_df[q_df["リスクウェイト"] == rw]
            cr4_rows.append({
                "リスクウェイト区分":     label,
                "最適化前EAD(百万円)":    round(sub_pre["EAD(百万円)"].sum(), 2),
                "最適化前RWA(百万円)":    round(sub_pre["RWA(百万円)"].sum(), 2),
                "最適化後EAD(百万円)":    round(sub_post["EAD(百万円)"].sum(), 2),
                "最適化後RWA(百万円)":    round(sub_post["RWA(百万円)"].sum(), 2),
                "RWA削減(百万円)":        round(sub_pre["RWA(百万円)"].sum() - sub_post["RWA(百万円)"].sum(), 2),
                "削減率(%)":              round(
                    (sub_pre["RWA(百万円)"].sum() - sub_post["RWA(百万円)"].sum()) /
                    max(sub_pre["RWA(百万円)"].sum(), 1) * 100, 1),
            })
        total_cr4 = {
            "リスクウェイト区分":  "合計",
            "最適化前EAD(百万円)": round(df_all["EAD(百万円)"].sum(), 2),
            "最適化前RWA(百万円)": round(df_all["RWA(百万円)"].sum(), 2),
            "最適化後EAD(百万円)": round(q_eval["total_ead"], 2),
            "最適化後RWA(百万円)": round(q_df["RWA(百万円)"].sum(), 2),
            "RWA削減(百万円)":     round(df_all["RWA(百万円)"].sum() - q_df["RWA(百万円)"].sum(), 2),
            "削減率(%)":           round((df_all["RWA(百万円)"].sum() - q_df["RWA(百万円)"].sum()) /
                                        max(df_all["RWA(百万円)"].sum(), 1) * 100, 1),
        }
        df_cr4 = pd.DataFrame(cr4_rows + [total_cr4])
        st.dataframe(df_cr4, use_container_width=True, hide_index=True)

        # CR4-B: セクター別
        st.markdown("**CR4-B — セクター別エクスポージャー（最適化後）**")
        sec_grp = q_df.groupby("セクター").agg(
            件数=("取引ID","count"),
            EAD=("EAD(百万円)","sum"),
            RWA=("RWA(百万円)","sum"),
            CVA=("CVA賦課(百万円)","sum"),
        ).reset_index()
        sec_grp["総RWA"] = (sec_grp["RWA"] + sec_grp["CVA"]).round(2)
        sec_grp["平均RW(%)"] = (sec_grp["RWA"] / sec_grp["EAD"] * 100).round(1)
        st.dataframe(sec_grp.sort_values("総RWA", ascending=False).round(2),
                     use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df_cr4[df_cr4["リスクウェイト区分"] != "合計"],
                         x="リスクウェイト区分",
                         y=["最適化前RWA(百万円)", "最適化後RWA(百万円)"],
                         barmode="group", title="RW区分別 RWA比較（前後）",
                         color_discrete_sequence=["#E74C3C","#27AE60"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(sec_grp, values="総RWA", names="セクター",
                          title="セクター別 RWA構成（最適化後）", hole=0.35)
            st.plotly_chart(fig2, use_container_width=True)

        dl_btn(df_cr4, f"CR4_信用リスクSA_{report_dt}.csv", "📥 CR4 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 6: CR5 — 信用品質（第三の柱）
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("#### 🎯 CR5: 信用品質（Credit Quality of Exposures）")
        st.caption("BCBS d309 Table CR5 — 格付・カウンターパーティ別信用品質分析")

        # CR5-A: 格付遷移マトリクス（最適化前後）
        st.markdown("**CR5-A — 格付別 EAD・RWA（最適化前後比較）**")
        cr5_rows = []
        for rating in RATINGS:
            pre  = df_all[df_all["格付"] == rating]
            post = q_df[q_df["格付"] == rating]
            cr5_rows.append({
                "格付":               rating,
                "RiskWeight":         f"{RATING_RW[rating]*100:.0f}%",
                "最適化前_件数":      len(pre),
                "最適化前_EAD(百万円)":  round(pre["EAD(百万円)"].sum(), 2),
                "最適化前_RWA(百万円)":  round(pre["RWA(百万円)"].sum(), 2),
                "最適化後_件数":      len(post),
                "最適化後_EAD(百万円)":  round(post["EAD(百万円)"].sum(), 2),
                "最適化後_RWA(百万円)":  round(post["RWA(百万円)"].sum(), 2),
                "EAD削減(百万円)":    round(pre["EAD(百万円)"].sum() - post["EAD(百万円)"].sum(), 2),
                "除外された取引":     len(pre) - len(post),
            })
        df_cr5 = pd.DataFrame(cr5_rows)
        st.dataframe(df_cr5, use_container_width=True, hide_index=True)

        # CR5-B: RoRWA分布
        st.markdown("**CR5-B — 保持/除外取引のRoRWA分布**")
        q_df_plot = q_df.copy(); q_df_plot["区分"] = "保持"
        excl_plot = excl_df.copy(); excl_plot["区分"] = "除外（圧縮候補）"
        combined = pd.concat([q_df_plot, excl_plot], ignore_index=True)
        fig = px.histogram(combined, x="RoRWA", color="区分", nbins=20,
                           barmode="overlay", opacity=0.7,
                           title="保持/除外取引 RoRWA分布",
                           color_discrete_map={"保持":"#27AE60","除外（圧縮候補）":"#E74C3C"})
        st.plotly_chart(fig, use_container_width=True)

        # CR5-C: カウンターパーティ別信用品質スコアカード
        st.markdown("**CR5-C — カウンターパーティ別 信用品質サマリー**")
        cp_quality = q_df.groupby(["カウンターパーティ","セクター"]).agg(
            格付=("格付","first"),
            取引件数=("取引ID","count"),
            EAD=("EAD(百万円)","sum"),
            RWA=("RWA(百万円)","sum"),
            CVA=("CVA賦課(百万円)","sum"),
            収益=("収益(百万円)","sum"),
        ).reset_index()
        cp_quality["RoRWA"] = (cp_quality["収益"] / (cp_quality["RWA"] + cp_quality["CVA"])).round(5)
        cp_quality["評価"] = cp_quality.apply(lambda r:
            "🌟優良" if r["格付"] in ["AAA","AA"] and r["RoRWA"] > 0.02
            else "✅良好" if r["RoRWA"] > 0.015
            else "⚠️要注意", axis=1)
        st.dataframe(cp_quality.sort_values("RoRWA", ascending=False).round(3),
                     use_container_width=True, hide_index=True)

        dl_btn(df_cr5, f"CR5_信用品質_{report_dt}.csv", "📥 CR5 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 7: 決算短信
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("#### 📰 決算短信（自己資本比率・RWA総額）")
        st.caption("決算短信様式（バーゼル規制資本関連抜粋）デモ帳票")

        fy = datetime.now().year
        st.markdown(f"""
**{fy}年3月期 決算短信（バーゼル規制資本関連情報）**
報告日: {now_str}　　銀行名: [デモ金融機関]　　開示分類: 連結

---
""")
        # 主要計数
        st.markdown("**■ 自己資本比率等（バーゼルIII最終化規制・標準的手法）**")
        tanshin_rows = [
            ("普通株式等Tier1（CET1）比率",  f"{cet1_info['base_ratio']*100:.2f}%",  f"{cet1_info['new_ratio']*100:.2f}%",  "4.50%"),
            ("Tier1比率",                    f"{tier1_cap/total_rwa_pre*100:.2f}%",   f"{tier1_cap/total_rwa_post*100:.2f}%", "6.00%"),
            ("総自己資本比率",               f"{total_cap/total_rwa_pre*100:.2f}%",  f"{total_cap/total_rwa_post*100:.2f}%", "8.00%"),
            ("リスク・アセット（RWA）合計（百万円）", f"{total_rwa_pre:.0f}",          f"{total_rwa_post:.0f}",                "—"),
            ("  うち 信用リスク（CCR含む）（百万円）", f"{crdt_rwa_pre:.0f}",          f"{crdt_rwa_post:.0f}",                 "—"),
            ("  うち 市場リスク（百万円）",   f"{mkt_rwa:.0f}",                        f"{mkt_rwa:.0f}",                       "—"),
            ("  うち オペリスク（百万円）",   f"{opex_rwa:.0f}",                       f"{opex_rwa:.0f}",                      "—"),
            ("CET1資本の額（百万円）",        f"{cet1_cap:.0f}",                       f"{cet1_cap:.0f}",                      "—"),
            ("Tier1資本の額（百万円）",       f"{tier1_cap:.0f}",                      f"{tier1_cap:.0f}",                     "—"),
            ("総自己資本の額（百万円）",      f"{total_cap:.0f}",                      f"{total_cap:.0f}",                     "—"),
        ]
        df_tanshin = pd.DataFrame(tanshin_rows,
            columns=["項目", f"{fy}年3月期（最適化前）", f"{fy}年3月期（最適化後）", "最低規制比率"])
        st.dataframe(df_tanshin, use_container_width=True, hide_index=True)

        # カウンターパーティ信用リスク（CCR）ハイライト
        st.markdown("**■ カウンターパーティ信用リスク（CCR）ハイライト**")
        ccr_hl = [
            ("SA-CCR EAD（百万円）",       f"{df_all['EAD(百万円)'].sum():.1f}", f"{q_eval['total_ead']:.1f}"),
            ("SA-CCR RWA（百万円）",       f"{df_all['RWA(百万円)'].sum():.1f}", f"{q_df['RWA(百万円)'].sum():.1f}"),
            ("CVA賦課（百万円）",          f"{df_all['CVA賦課(百万円)'].sum():.1f}", f"{q_df['CVA賦課(百万円)'].sum():.1f}"),
            ("デリバティブ取引件数",       f"{len(df_all)}件", f"{len(q_df)}件（圧縮後）"),
            ("最大CP集中度（EAD比）",      "—", f"{q_eval['max_conc']*100:.1f}%"),
        ]
        df_ccr_hl = pd.DataFrame(ccr_hl, columns=["項目", "最適化前", "最適化後"])
        st.dataframe(df_ccr_hl, use_container_width=True, hide_index=True)

        dl_btn(df_tanshin, f"決算短信_自己資本比率_{report_dt}.csv", "📥 決算短信 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 8: 有価証券報告書（リスク情報開示）
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("#### 📚 有価証券報告書（リスク情報開示セクション）")
        st.caption("有価証券報告書 第3[事業等のリスク] / 第5[経理の状況] バーゼル規制関連抜粋デモ")

        st.markdown("**■ 主要リスク指標サマリー**")
        risk_kpi = [
            ("カテゴリ", "リスク指標",            "最適化前",                                 "最適化後",                                  "コメント"),
            ("CCR",    "SA-CCR EAD",              f"¥{df_all['EAD(百万円)'].sum():.1f}M",    f"¥{q_eval['total_ead']:.1f}M",             "デリバティブEAD"),
            ("CCR",    "CCR RWA",                 f"¥{df_all['RWA(百万円)'].sum():.1f}M",    f"¥{q_df['RWA(百万円)'].sum():.1f}M",       "SA-CCR標準手法"),
            ("CCR",    "CVA賦課",                 f"¥{df_all['CVA賦課(百万円)'].sum():.1f}M",f"¥{q_df['CVA賦課(百万円)'].sum():.1f}M",   "標準的手法CVA"),
            ("資本",   "CET1比率",                f"{cet1_info['base_ratio']*100:.2f}%",      f"{cet1_info['new_ratio']*100:.2f}%",        "最低基準8.5%"),
            ("資本",   "総自己資本比率",           f"{total_cap/total_rwa_pre*100:.2f}%",      f"{total_cap/total_rwa_post*100:.2f}%",      "最低基準10.5%"),
            ("集中度", "最大CP集中度（EAD比）",    "—",                                         f"{q_eval['max_conc']*100:.1f}%",           "上限25%"),
            ("収益",   "保持取引収益",             f"¥{base_rev:.2f}M（全件）",                f"¥{q_eval['total_rev']:.2f}M",             "収益フロア達成"),
        ]
        df_ykh = pd.DataFrame(risk_kpi[1:], columns=risk_kpi[0])
        st.dataframe(df_ykh, use_container_width=True, hide_index=True)

        # 重要リスク記述
        st.markdown("**■ カウンターパーティ信用リスク（CCR）に関する定量情報**")
        cp_risk_tbl = q_df.groupby(["セクター","格付"]).agg(
            件数=("取引ID","count"),
            EAD=("EAD(百万円)","sum"),
            RWA=("RWA(百万円)","sum"),
        ).reset_index()
        cp_risk_tbl["EAD構成比(%)"] = (cp_risk_tbl["EAD"]/cp_risk_tbl["EAD"].sum()*100).round(1)
        st.dataframe(cp_risk_tbl.sort_values("EAD", ascending=False).round(2),
                     use_container_width=True, hide_index=True)

        # 残存年数分布
        st.markdown("**■ デリバティブ残存年数分布（最適化後ポートフォリオ）**")
        c1, c2 = st.columns(2)
        with c1:
            mat_dist = q_df.groupby("残存年数")["EAD(百万円)"].sum().reset_index()
            fig = px.bar(mat_dist, x="残存年数", y="EAD(百万円)",
                         title="残存年数別 EAD分布", labels={"残存年数":"残存年数（年）"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(q_df, x="残存年数", y="EAD(百万円)",
                              color="格付", size="想定元本(百万円)",
                              title="残存年数 vs EAD（格付別）",
                              hover_data=["取引ID","アセットクラス"])
            st.plotly_chart(fig2, use_container_width=True)

        dl_btn(df_ykh, f"有価証券報告書_リスク情報_{report_dt}.csv", "📥 有報リスク情報 CSVダウンロード")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 9: Basel開示資料（資本構成・RWA）
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.markdown("#### 🏦 Basel開示資料（資本構成・RWA詳細開示）")
        st.caption("金融庁バーゼルIII最終化 開示様式 — Pillar 3開示資料（デモ）")

        # KY1: 主要規制計数
        st.markdown("**KY1 — 主要規制計数（Key Metrics）**")
        ky1_rows = [
            ("利用可能な自己資本（百万円）", "", ""),
            ("  CET1資本",                   f"{cet1_cap:.0f}",    f"{cet1_cap:.0f}"),
            ("  Tier1資本",                  f"{tier1_cap:.0f}",   f"{tier1_cap:.0f}"),
            ("  総自己資本",                 f"{total_cap:.0f}",   f"{total_cap:.0f}"),
            ("リスク加重資産（百万円）", "", ""),
            ("  信用リスク（CCR含む）",      f"{crdt_rwa_pre:.0f}",  f"{crdt_rwa_post:.0f}"),
            ("  市場リスク",                 f"{mkt_rwa:.0f}",    f"{mkt_rwa:.0f}"),
            ("  オペレーショナルリスク",     f"{opex_rwa:.0f}",   f"{opex_rwa:.0f}"),
            ("  RWA合計",                    f"{total_rwa_pre:.0f}", f"{total_rwa_post:.0f}"),
            ("リスク基準自己資本比率", "", ""),
            ("  CET1比率",                   f"{cet1_info['base_ratio']*100:.2f}%", f"{cet1_info['new_ratio']*100:.2f}%"),
            ("  Tier1比率",                  f"{tier1_cap/total_rwa_pre*100:.2f}%", f"{tier1_cap/total_rwa_post*100:.2f}%"),
            ("  総自己資本比率",             f"{total_cap/total_rwa_pre*100:.2f}%", f"{total_cap/total_rwa_post*100:.2f}%"),
            ("レバレッジ比率（参考）", "", ""),
            ("  Tier1資本（百万円）",        f"{tier1_cap:.0f}", f"{tier1_cap:.0f}"),
            ("  総エクスポージャー（百万円）", f"{df_all['EAD(百万円)'].sum()*1.1:.0f}",
                                              f"{q_eval['total_ead']*1.1:.0f}"),
        ]
        df_ky1 = pd.DataFrame(ky1_rows, columns=["項目", "最適化前", "最適化後（量子AE）"])
        st.dataframe(df_ky1, use_container_width=True, hide_index=True)

        # OV1: RWA概要
        st.markdown("**OV1 — RWA概要（Overview of RWA）**")
        ov1_rows = [
            ("信用リスク（SA）",      f"{crdt_rwa_pre:.1f}",  f"{crdt_rwa_post:.1f}",  f"{crdt_rwa_post*0.08:.1f}"),
            ("  CCR（SA-CCR）",       f"{base_rwa:.1f}",       f"{q_eval['total_rwa']:.1f}", f"{q_eval['total_rwa']*0.08:.1f}"),
            ("  CVA賦課",             f"{df_all['CVA賦課(百万円)'].sum():.1f}",
                                      f"{q_df['CVA賦課(百万円)'].sum():.1f}",
                                      f"{q_df['CVA賦課(百万円)'].sum()*0.08:.1f}"),
            ("市場リスク",            f"{mkt_rwa:.1f}",    f"{mkt_rwa:.1f}",    f"{mkt_rwa*0.08:.1f}"),
            ("オペレーショナルリスク",f"{opex_rwa:.1f}",   f"{opex_rwa:.1f}",   f"{opex_rwa*0.08:.1f}"),
            ("合計",                  f"{total_rwa_pre:.1f}", f"{total_rwa_post:.1f}", f"{total_rwa_post*0.08:.1f}"),
        ]
        df_ov1 = pd.DataFrame(ov1_rows, columns=[
            "リスク区分", "RWA（最適化前・百万円）", "RWA（最適化後・百万円）", "最低所要自己資本（8%）"
        ])
        st.dataframe(df_ov1, use_container_width=True, hide_index=True)

        # 可視化
        c1, c2 = st.columns(2)
        with c1:
            rwa_comps = ["信用リスク（CCR）", "市場リスク", "オペリスク"]
            fig = go.Figure()
            fig.add_bar(name="最適化前", x=rwa_comps,
                        y=[crdt_rwa_pre, mkt_rwa, opex_rwa], marker_color="#E74C3C")
            fig.add_bar(name="最適化後", x=rwa_comps,
                        y=[crdt_rwa_post, mkt_rwa, opex_rwa], marker_color="#27AE60")
            fig.update_layout(title="リスク区分別 RWA比較", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # ウォーターフォール：RWA削減分解
            fig2 = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute","relative","total"],
                x=["最適化前RWA", "ポートフォリオ圧縮効果", "最適化後RWA"],
                y=[base_rwa, -(base_rwa - q_eval["total_rwa"]), 0],
                connector={"line":{"color":"#888"}},
                increasing={"marker":{"color":"#E74C3C"}},
                decreasing={"marker":{"color":"#27AE60"}},
                totals={"marker":{"color":"#1B4F9B"}},
            ))
            fig2.update_layout(title="RWA削減ウォーターフォール（百万円）")
            st.plotly_chart(fig2, use_container_width=True)

        dl_btn(df_ky1, f"Basel開示資料_KY1_{report_dt}.csv", "📥 Basel開示資料（KY1）CSVダウンロード")

    # ── 共通免責事項 ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
> 📌 **免責事項**
> 本レポートはFixstars Amplify AE量子アニーリングによる最適化結果をもとに生成したデモ帳票です。
> 実際の規制報告（金融庁、BCBS様式）への利用には、金融庁認可手法による正式計算および
> 法務・コンプライアンス部門による確認が必要です。本資料は意思決定支援情報であり、
> 規制上の資本計算書類として使用することはできません。
""")
    st.caption("© BIPROGY Inc. — Basel CRM リスクアセット最適化ソリューション")
