import pandas as pd

data = """
F01  &  0.00000000e+00 &  7.69000000e-08 &  5.57288235e-16 &  1.00000000e-09 &  0.00000000e+00 &  0.00000000e+00 \\
F02  &  3.43137255e+00 &  0.00000000e+00 &  0.00000000e+00 &  0.00000000e+00 &  0.00000000e+00 &  0.00000000e+00 \\
F03  &  0.00000000e+00 &  8.47000000e-08 &  0.00000000e+00 &  1.00000000e-09 &  0.00000000e+00 &  0.00000000e+00 \\
F04  &  1.98461340e+00 &  1.98461340e+00 &  1.98461340e+00 &  1.98461340e+00 &  1.98461340e+00 &  1.98461340e+00 \\
F05  &  6.69786245e+00 &  3.21099624e+01 &  1.49696152e+01 &  2.76501433e+01 &  2.54708578e+01 &  1.98779140e+01 \\
F06  &  0.00000000e+00 &  8.38900000e-07 &  0.00000000e+00 &  5.19000000e-08 &  2.60000000e-09 &  1.00000000e-10 \\
F07  &  1.98435296e+01 &  4.49162696e+01 &  3.07642035e+01 &  3.88538541e+01 &  3.66145157e+01 &  3.41415598e+01 \\
F08  &  7.96154586e+00 &  3.38166865e+01 &  1.50234458e+01 &  2.86540094e+01 &  2.47481670e+01 &  2.30961205e+01 \\
F09  &  0.00000000e+00 &  4.06000000e-08 &  0.00000000e+00 &  3.00000000e-10 &  0.00000000e+00 &  0.00000000e+00 \\
F10  &  3.02740359e+02 &  7.03077998e+02 &  2.50192800e+02 &  3.74217679e+02 &  3.46279563e+02 &  2.57245221e+02 \\
F11  &  8.23807461e-01 &  8.07570245e+00 &  7.45771487e-01 &  7.35924017e-01 &  7.35055510e-01 &  7.65281665e-01 \\
F12  &  4.30013868e+01 &  8.33472622e+01 &  3.05426229e+01 &  3.35054056e+01 &  3.18494868e+01 &  2.99707771e+01 \\
F13  &  4.07421483e+00 &  1.27334608e+01 &  4.05470586e+00 &  5.88066910e+00 &  4.16360587e+00 &  3.96153215e+00 \\
F14  &  5.65761041e-01 &  2.21272247e+01 &  3.31653275e-01 &  1.87508703e+00 &  3.90729832e-01 &  3.90180274e-01 \\
F15  &  3.14903935e-01 &  3.43341572e+00 &  9.02603488e-02 &  9.11211867e-02 &  1.28894814e-01 &  1.56419074e-01 \\
F16  &  4.06398207e+00 &  5.44159209e+00 &  1.96505602e+00 &  1.58768195e+00 &  1.77747568e+00 &  1.73347031e+00 \\
F17  &  4.41622788e+00 &  2.76308075e+01 &  2.12141023e+00 &  6.85319034e+00 &  2.26520002e+00 &  2.90012747e+00 \\
F18  &  7.16837184e-01 &  5.33991174e+00 &  1.52308691e-01 &  1.28150996e-01 &  1.37357832e-01 &  1.10889659e-01 \\
F19  &  1.32016195e-01 &  4.12010855e-01 &  3.52132054e-02 &  5.72835700e-03 &  4.88014912e-02 &  3.33796864e-02 \\
F20  &  3.22877812e-01 &  5.04686684e+00 &  1.89752386e-01 &  6.06476672e-01 &  2.26986856e-01 &  2.45987651e-01 \\
F21  &  1.81228788e+02 &  1.01095024e+02 &  1.47131657e+02 &  1.02844435e+02 &  1.17555249e+02 &  1.49565845e+02 \\
F22  &  8.58076104e+01 &  6.68067973e+01 &  7.41908084e+01 &  7.48977463e+01 &  8.05485781e+01 &  7.49104242e+01 \\
F23  &  3.06953665e+02 &  3.28704520e+02 &  3.06060396e+02 &  3.17919141e+02 &  3.07283067e+02 &  3.06698637e+02 \\
F24  &  3.28772451e+02 &  1.92708053e+02 &  3.10590314e+02 &  2.57861312e+02 &  3.03492459e+02 &  3.05929176e+02 \\
F25  &  4.12755104e+02 &  3.98946027e+02 &  4.03495300e+02 &  4.03473780e+02 &  4.05278255e+02 &  4.06201682e+02 \\
F26  &  3.05507971e+02 &  3.00000000e+02 &  3.00000000e+02 &  3.00000000e+02 &  3.00000000e+02 &  3.00000000e+02 \\
F27  &  3.90463614e+02 &  3.89299947e+02 &  3.89371702e+02 &  3.89313833e+02 &  3.89346141e+02 &  3.89361539e+02 \\
F28  &  3.70405545e+02 &  3.02039288e+02 &  3.08408680e+02 &  3.03932794e+02 &  3.13559104e+02 &  3.19334514e+02 \\
F29  &  2.30270749e+02 &  2.66208253e+02 &  2.30140163e+02 &  2.36180727e+02 &  2.30033525e+02 &  2.30086135e+02 \\
F30  &  1.51360673e+05 &  5.38081994e+02 &  6.41562278e+02 &  5.92897463e+02 &  5.90109959e+02 &  6.46526218e+02 \\
"""

columns = ['func', 'base', 'reinicio-1', 'reinicio-10', 'reinicio-3', 'reinicio-5', 'reinicio-7']
value_cols = columns[1:]

precision = 4

pd.set_option('display.float_format', f'{{:.{precision}f}}'.format)

rows = []
for line in data.strip().split('\n'):
    line_clean = line.replace('\\\\', '').replace('\\', '').strip()
    parts = [p.strip() for p in line_clean.split('&')]
    rows.append(parts)

df = pd.DataFrame(rows, columns=columns)

for col in value_cols:
    df[col] = df[col].astype(float)

# Copia usada para comparar con precisión 4
# Esto imita que TACOLAB compare valores con precisión 4
df_comp = df.copy()
df_comp[value_cols] = df_comp[value_cols].round(precision)

# =========================
# WINS
# =========================

wins = {col: 0 for col in value_cols}
details = []

for idx, row in df_comp.iterrows():
    vals = row[value_cols].to_dict()
    min_val = min(vals.values())
    min_cols = [k for k, v in vals.items() if v == min_val]

    if len(min_cols) == 1:
        wins[min_cols[0]] += 1
        details.append((row['func'], min_cols[0], min_val))
    else:
        details.append((row['func'], 'TIE', min_cols))

wins_df = (
    pd.Series(wins, name="wins")
    .reset_index()
    .rename(columns={"index": "columna"})
    .sort_values("wins", ascending=False)
)

print("Wins summary:", wins)

print("\nWins por columna:")
print(wins_df)

print("\nRow by row details:")
for d in details:
    print(d)

# =========================
# RANKING TABLE
# =========================

ranking_df = df_comp[['func']].copy()

ranking_df[value_cols] = df_comp[value_cols].rank(
    axis=1,
    method='average',
    ascending=True
)

ranking_summary = (
    ranking_df[value_cols]
    .mean()
    .sort_values()
    .reset_index()
    .rename(columns={
        'index': 'algoritmo',
        0: 'average_ranking'
    })
)

ranking_summary['average_ranking'] = ranking_summary['average_ranking'].round(precision)

print("\nValores usados para comparar con precision =", precision)
print(df_comp)

print("\nRanking por función:")
print(ranking_df)

print("\nAverage ranking por algoritmo:")
print(ranking_summary)

# =========================
# TABLA FINAL RESUMEN
# =========================

final_summary = wins_df.merge(
    ranking_summary,
    left_on='columna',
    right_on='algoritmo'
).drop(columns='algoritmo')

final_summary = final_summary.sort_values(
    by=['average_ranking', 'wins'],
    ascending=[True, False]
)

final_summary['average_ranking'] = final_summary['average_ranking'].round(precision)

print("\nResumen final: wins + average ranking")
print(final_summary)