# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# ── Data (percentages, from LDA_Topic_Proportions.csv; updated preprocessing) ──
periods = [
    '1851–1900', '1901–1930', '1931–1950',
    '1951–1960', '1961–1980',
    '1981–2001\n(pre-9/11)', '2001–2019\n(post-9/11)',
]

# Each list = proportions (%) across the 7 periods
topics = {
    'T0: Polit. Violence & Colonial Conflict': [ 24.29,  45.72,  60.61,  63.39,  48.86,   9.15,   1.28],
    'T11: War & Policy Commentary': [ 49.08,  32.13,  15.75,  10.37,  10.93,   2.85,   1.80],
    'T1: Domestic & Social Life': [ 23.45,  10.24,   6.32,   4.32,   9.56,  12.19,  12.21],
    'T7: Intelligence & Security': [  0.00,   0.00,   0.32,   0.27,   4.55,  12.62,  16.11],
    'T9: Afghanistan–Pakistan & Al-Qaeda': [  0.00,   0.00,   0.11,   0.27,   0.26,   4.54,  14.95],
    'T2: Middle East Diplomacy': [  0.00,   0.00,   2.68,   1.33,   6.18,  14.05,   5.58],
    'T4: Bombings & Armed Attacks': [  0.50,   2.38,   2.83,   2.99,   7.32,  13.11,   7.67],
    'T5: European & Soviet Politics': [  0.34,   5.47,   7.24,  13.02,   3.84,   6.23,   3.48],
    'T10: Bush-Era Politics': [  0.84,   0.26,   0.32,   0.13,   0.47,   4.18,  11.86],
    'Other (T3, T6, T8)': [  1.51,   3.80,   3.81,   3.92,   8.03,  21.07,  25.05],
}

# Colorblind-friendly palette (Tableau-10 inspired)
colors = [
    '#4e79a7',  # blue       – T5 European War
    '#59a14f',  # green      – T9 Anti-Colonial
    '#e15759',  # red        – T6 Criminal Violence
    '#f28e2b',  # orange     – T4 Electoral
    '#76b7b2',  # teal       – T2 Cold War
    '#9467bd',  # purple     – T7 Israeli-Palestinian
    '#d62728',  # dark red   – T8 S. Asia
    '#8c564b',  # brown      – T10 Iraq
    '#bcbd22',  # yellow-grn – T0 Governance
    '#c7c7c7',  # light grey – Other
]

x = np.arange(len(periods))
width = 0.62

fig, ax = plt.subplots(figsize=(13, 7))

bottom = np.zeros(len(periods))
for (label, vals), color in zip(topics.items(), colors):
    vals_arr = np.array(vals)
    ax.bar(x, vals_arr, width, bottom=bottom,
           label=label, color=color, edgecolor='white', linewidth=0.5)
    bottom += vals_arr

# Vertical dashed line marking 9/11 divide
ax.axvline(x=5.5, color='#333333', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(5.53, 97, '9/11', fontsize=9, color='#333333', va='top', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=9.5)
ax.set_ylabel('Proportion of Documents (%)', fontsize=11, labelpad=6)
ax.set_xlabel('Historical Period', fontsize=11, labelpad=8)
ax.set_ylim(0, 100)
ax.set_xlim(-0.45, len(periods) - 0.55)
ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.legend(
    loc='upper left', bbox_to_anchor=(1.01, 1.01),
    fontsize=9, framealpha=0.95, edgecolor='#cccccc',
    title='LDA Topics (K = 12)', title_fontsize=9.5,
    borderpad=0.8, labelspacing=0.5,
)

ax.set_title(
    'Figure 2. LDA Topic Proportions by Historical Period',
    fontsize=11, fontweight='bold', pad=10, loc='left',
)

plt.tight_layout()
# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
out = _os.path.join(REPO_ROOT, 'figures', 'Figure2_LDA_Topics.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out)

# ── Heatmap (supplementary / alternative view) ──────────────────────────────
topic_labels_short = [
    'T1: Eur. War & Rev.', 'T5: Cold War', 'T0: Pol. Viol. & Brit.',
    'T9: Armed Viol.', 'T10: Iraq & Afghan.', 'T8: Dom. & Soc.',
    'T7: Isr.-Pal.', 'T6: Crim. Justice', 'T4: Electoral', 'Other',
]
heatmap_data = np.array(list(topics.values()))   # shape (10 topics, 7 periods)

period_labels_short = [
    '1851–\n1900', '1901–\n1930', '1931–\n1950', '1951–\n1960',
    '1961–\n1980', '1981–\n2001', '2001–\n2019',
]

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
im = ax2.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=80)
ax2.set_xticks(np.arange(len(periods)))
ax2.set_xticklabels(period_labels_short, fontsize=9.5)
ax2.set_yticks(np.arange(len(topic_labels_short)))
ax2.set_yticklabels(topic_labels_short, fontsize=9.5)

# Annotate cells
for i in range(len(topic_labels_short)):
    for j in range(len(periods)):
        val = heatmap_data[i, j]
        color = 'white' if val > 40 else 'black'
        ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                 fontsize=8, color=color)

cb = fig2.colorbar(im, ax=ax2, shrink=0.85, pad=0.02)
cb.set_label('% of Documents', fontsize=10)
ax2.set_title('Figure 2 (alt). LDA Topic Proportions Heatmap by Historical Period',
              fontsize=10, fontweight='bold', pad=10, loc='left')
ax2.set_xlabel('Historical Period', fontsize=10, labelpad=6)

plt.tight_layout()
out2 = _os.path.join(REPO_ROOT, 'figures', 'Figure2_LDA_Heatmap.png')
plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out2)
