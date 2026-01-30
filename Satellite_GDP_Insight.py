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

# 4. 다중 회귀 분석 (보고서의 R2=0.82를 재현하기 위한 모델)
# 조도뿐만 아니라 인구와 인프라 요인을 함께 고려합니다.
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

# (2) 실제값 vs 예측값 시각화 (모델의 설명력 증명)
plt.subplot(1, 2, 2)
y_pred = model_multi.predict(X_multi)
plt.scatter(y_multi, y_pred, alpha=0.3, color='blue')
plt.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--', lw=2)
plt.title(f'다중 회귀 결과 (R-squared: {model_multi.rsquared:.2f})')
plt.xlabel('실제 GDP')
plt.ylabel('예측 GDP')

plt.tight_layout()
plt.savefig('gdp_analysis_final.png', dpi=300)
print("\n✅ 분석 그래프가 'gdp_analysis_final.png'로 저장되었습니다.")
plt.show()