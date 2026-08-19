"""
⚠ 이 스크립트는 .xlsx를 읽습니다. 그 파일의 조도 열은 GDP 열과 어긋나 있어
   여기서 나오는 R^2(약 0.877)는 조도의 설명력이 아닙니다.
   조도를 모형에서 빼도 R^2가 같습니다. 값을 만드는 것은 인구·도시인구입니다.

   검증된 결과는 reproduce.py를 쓰세요 (.sav 기준, ln-ln 단순회귀 R^2 = 0.819, N = 791).
   자세한 내용은 README의 「7. 데이터 파일 주의사항」 참고.
   이 파일은 당시 작업 기록으로 남겨둡니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import platform

# 1. 한글 폰트 설정 (Mac/Windows 자동 대응)
if platform.system() == 'Darwin': # Mac 사용자라면
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows 사용자라면
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 2. 데이터 로드
file_name = '야간 위성 조도를 활용한 국가별 GDP 예측_2021510010 이상민.xlsx'
df = pd.read_excel(file_name)

# 3. 데이터 정제 (Warning 방지를 위해 .copy() 사용)
# GDP, 조도, 인구, 도시인구, 전력접근성 데이터가 모두 있는 행만 추출
cols = ['GDP', 'brightness_sum', 'Population', 'Urban_Population', 'Electricity_Access']
df_clean = df.dropna(subset=cols).copy()

# 로그 변환 (로그 변환 시 .loc를 사용하여 Warning 방지)
df_clean.loc[:, 'log_GDP'] = np.log1p(df_clean['GDP'])
df_clean.loc[:, 'log_brightness'] = np.log1p(df_clean['brightness_sum'])

# 4. 다중 회귀 분석 (초기 기록용)
# 이 파일은 어긋난 xlsx를 읽으므로 아래 결과는 검증 결과로 사용하지 않습니다.
X_multi = sm.add_constant(df_clean[['brightness_sum', 'Population', 'Urban_Population', 'Electricity_Access']])
y_multi = df_clean['GDP']
model_multi = sm.OLS(y_multi, X_multi).fit()

print("--- [다중 회귀 분석 결과 요약] ---")
print(model_multi.summary())

# 5. 시각화 및 저장
plt.figure(figsize=(12, 5))

# (1) 단순 상관관계 시각화 (로그 변환)
plt.subplot(1, 2, 1)
sns.regplot(x='log_brightness', y='log_GDP', data=df_clean,
            scatter_kws={'alpha':0.3, 'color':'gray'}, line_kws={'color':'orange'})
plt.title('야간 조도 vs GDP (로그 변환)')

# (2) 실제값 vs 적합값 시각화 (초기 기록용)
plt.subplot(1, 2, 2)
y_pred = model_multi.predict(X_multi)
plt.scatter(y_multi, y_pred, alpha=0.3, color='blue')
plt.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--', lw=2)
plt.title(f'다중 회귀 결과 - 초기 xlsx 기준 (R-squared: {model_multi.rsquared:.2f})')
plt.xlabel('실제 GDP')
plt.ylabel('적합 GDP')

plt.tight_layout()
plt.savefig('gdp_analysis_final.png', dpi=300)
print("\n✅ 분석 그래프가 'gdp_analysis_final.png'로 저장되었습니다.")
plt.show()
