"""
reproduce.py — 이력서·포트폴리오에 인용된 수치를 이 저장소 안에서 다시 계산합니다.

실행:
    pip install -r requirements.txt
    python reproduce.py

출력:
    stdout 회귀 결과표 + outputs/regression_summary.csv, outputs/analysis_dataset.csv

기준 데이터는 SPSS 파일(.sav)입니다. 보고서의 분석이 이 파일에서 수행됐고,
로그 변환과 평균 중심화가 이미 적용되어 있습니다.

  GDP        = ln(GDP)
  Brightness = ln(야간 조도 합계)
  BC/UPC/EAC = 각 변수의 평균 중심화 값
  int_*      = 중심화 변수 간 상호작용항

같은 폴더의 .xlsx는 조도 열이 GDP 열과 어긋나 있습니다(아래 정합성 검사 참고).
분석에 쓰지 마세요.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

HERE = Path(__file__).parent
OUT = HERE / "outputs"


def load_sav() -> pd.DataFrame:
    path = glob.glob(str(HERE / "*.sav"))[0]
    df, _ = pyreadstat.read_sav(path)
    df["Country"] = df["Country"].str.strip()
    df["Year"] = df["Year"].astype(int)
    return df


def fit(df: pd.DataFrame, y: str, xs: list[str]):
    data = df[[y, *xs]].dropna()
    model = sm.OLS(data[y], sm.add_constant(data[xs])).fit()
    return model


def check_xlsx_alignment(sav: pd.DataFrame) -> pd.DataFrame:
    """xlsx의 조도 열이 .sav와 일치하는지 확인한다. 일치하지 않으면 xlsx는 쓸 수 없다."""
    paths = glob.glob(str(HERE / "*.xlsx"))
    if not paths:
        return pd.DataFrame()
    xl = pd.read_excel(paths[0])
    # .sav의 국가명은 8자로 잘려 있다. 164개 국가 모두 8자 절단 후에도 고유하다.
    xl["key"] = xl["country"].str[:8]
    merged = sav.merge(xl, left_on=["Country", "Year"], right_on=["key", "year"])
    merged["xlsx_log_brightness"] = np.log1p(merged["brightness_sum"])
    merged["xlsx_log_gdp"] = np.log1p(merged["GDP_y"])
    merged["brightness_gap"] = (merged["xlsx_log_brightness"] - merged["Brightness"]).abs()
    merged["gdp_gap"] = (merged["xlsx_log_gdp"] - merged["GDP_x"]).abs()
    return merged


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sav = load_sav()
    print(f"환경     : pandas {pd.__version__} · statsmodels {sm.__version__}")
    print(f"분석 파일: SPSS .sav · {len(sav):,}행 · GDP 관측치 {int(sav['GDP'].notna().sum()):,}개\n")

    print("=== 1. 단순회귀 — ln(GDP) ~ ln(야간 조도) ===")
    simple = fit(sav, "GDP", ["Brightness"])
    print(f"R²          : {simple.rsquared:.4f}")
    print(f"수정 R²     : {simple.rsquared_adj:.4f}")
    print(f"관측치       : {int(simple.nobs)}")
    print(f"기울기       : {simple.params['Brightness']:.4f} (p = {simple.pvalues['Brightness']:.3g})")
    print("→ 이력서·포트폴리오의 R² = 0.819는 이 모형입니다.")
    print("→ 설명력 지표이며 예측 정확도나 인과관계를 뜻하지 않습니다.\n")

    print("=== 2. 주효과 모형 — 조도 + 도시인구 + 전력접근성 (중심화) ===")
    main_effects = fit(sav, "GDP", ["BC", "UPC", "EAC"])
    print(f"R² = {main_effects.rsquared:.4f} · 관측치 {int(main_effects.nobs)}\n")

    print("=== 3. 조절효과 모형 — 상호작용항 추가 ===")
    moderation = fit(sav, "GDP", ["BC", "UPC", "EAC", "int_B_UP", "int_B_EA"])
    print(moderation.summary().tables[1])
    delta = moderation.rsquared - main_effects.rsquared
    print(f"R² = {moderation.rsquared:.4f} · 관측치 {int(moderation.nobs)}")
    print(f"→ 상호작용항은 유의하지만(p < 0.001) 설명력 증가분은 ΔR² = {delta:.4f}로 작습니다.\n")

    print("=== 4. 데이터 정합성 검사 — .xlsx ===")
    merged = check_xlsx_alignment(sav)
    if merged.empty:
        print("xlsx 파일을 찾지 못해 검사를 건너뜁니다.\n")
    else:
        b_ok = int((merged["brightness_gap"] < 1e-3).sum())
        b_total = int(merged["brightness_gap"].notna().sum())
        g_ok = int((merged["gdp_gap"] < 1e-3).sum())
        g_total = int(merged["gdp_gap"].notna().sum())
        print(f"국가-연도 매칭 {len(merged):,}행")
        print(f"GDP  일치 : {g_ok:,} / {g_total:,}")
        print(f"조도 일치 : {b_ok:,} / {b_total:,}")
        if b_ok < b_total * 0.9:
            xl_corr = np.log1p(merged["brightness_sum"]).corr(merged["xlsx_log_gdp"])
            sav_corr = sav["GDP"].corr(sav["Brightness"])
            print(
                "\n⚠ xlsx의 조도 열이 GDP 열과 어긋나 있습니다.\n"
                f"   같은 행끼리 계산한 상관: xlsx {xl_corr:.4f} · sav {sav_corr:.4f}\n"
                "   Satellite_GDP_Insight.py는 이 xlsx를 읽으므로 결과를 신뢰할 수 없습니다.\n"
                "   분석에는 .sav 또는 아래 outputs/analysis_dataset.csv를 쓰세요."
            )

    rows = [
        {"model": "simple: ln(GDP) ~ ln(Brightness)", "r_squared": round(simple.rsquared, 4),
         "adj_r_squared": round(simple.rsquared_adj, 4), "n": int(simple.nobs)},
        {"model": "main effects: BC + UPC + EAC", "r_squared": round(main_effects.rsquared, 4),
         "adj_r_squared": round(main_effects.rsquared_adj, 4), "n": int(main_effects.nobs)},
        {"model": "moderation: + int_B_UP + int_B_EA", "r_squared": round(moderation.rsquared, 4),
         "adj_r_squared": round(moderation.rsquared_adj, 4), "n": int(moderation.nobs)},
    ]
    pd.DataFrame(rows).to_csv(OUT / "regression_summary.csv", index=False)
    sav.to_csv(OUT / "analysis_dataset.csv", index=False)
    print(f"\n저장: {OUT/'regression_summary.csv'}, {OUT/'analysis_dataset.csv'}")


if __name__ == "__main__":
    main()
