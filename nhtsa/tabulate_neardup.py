# Tabulates Supplementary Table S18 from results/nhtsa/neardup/ and checks the registered criterion.
import json,os
import os
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(os.path.dirname(HERE),'results','nhtsa','neardup')+'/'
g=json.load(open(D+'nhtsa_neardup_grouping.json')); res={r['key']:r['f1'] for r in json.load(open(D+'nhtsa_neardup_results.json'))}
con=json.load(open(D+'nhtsa_neardup_contrasts.json'))
leg2={r['key']:r['f1'] for r in json.load(open(os.path.join(os.path.dirname(HERE),'results','nhtsa','nhtsa_task_results.json')))}
log=open(D+'nhtsa_neardup.log').read()
rt=[l for l in log.splitlines() if l.startswith('runtime_s')]
F={'summary':'Defect summary','conseq':'Consequence','remedy':'Remedy'}
def c(arch,s,a,b):
    for x in con:
        if x['contrast']==f"{arch} s{s}: {a} vs {b}": return x
# criterion
bil=[c('bilstm',s,'summary',b) for s in (0,1,2) for b in ('conseq','remedy')]
tf=[c('tfidf',0,'summary',b) for b in ('conseq','remedy')]
met=all(x['delta']>0 and x['p_holm']<0.05 for x in bil) and all(x['delta']>0 for x in tf)
lo=min(x['delta'] for x in bil+tf); hi=max(x['delta'] for x in bil+tf)
sc=[x['delta'] for x in bil if 'conseq' in x['contrast']]+[tf[0]['delta']]
sr=[x['delta'] for x in bil if 'remedy' in x['contrast']]+[tf[1]['delta']]
rc=[c('bilstm',s,'remedy','conseq') for s in (0,1,2)]
out=[]
out.append('# NHTSA near-duplicate grouped split — result (registered v5.6, run 2026-09-13)\n')
out.append(f"Runtime: {rt[0].split()[1] if rt else '?'} s. Script: views_wip/nhtsa_neardup.py. Raw outputs: views_wip/neardup/nhtsa_neardup_*.\n")
out.append('## Grouping statistics\n')
out.append(f"- Leg-2 exact-key groups: {g['exact_groups_leg2']}; near-duplicate groups (Jaccard ≥ 0.80, any field, plus exact keys): {g['groups_neardup']}")
out.append(f"- New unions by field: Summary {g['Summary']['new_unions']}, Consequence {g['Consequence']['new_unions']}, Remedy {g['Remedy']['new_unions']}")
out.append(f"- Multi-member groups: {g['multi_groups']}; campaigns in multi-member groups: {g['campaigns_in_multi_groups']} of {g['n_task']} ({100*g['campaigns_in_multi_groups']/g['n_task']:.1f}%); largest group {g['largest_group']}")
out.append(f"- Train/test: {g['n_task']-g['n_test']}/{g['n_test']} (leg 2: {g['n_task']-g['n_test_leg2']}/{g['n_test_leg2']}); the test side is a different set of campaigns, so absolute scores are not comparable with leg 2.\n")
out.append('## Held-out macro-F1 (near-duplicate split; leg-2 value in brackets)\n')
out.append('| Field | TF-IDF | BiLSTM 1 | BiLSTM 2 | BiLSTM 3 |\n|---|---|---|---|---|')
for k,n in F.items():
    row=[f"{res[f'nhtsa_{k}_tfidf_s0']:.3f} ({leg2[f'nhtsa_{k}_tfidf_s0']:.3f})"]+[f"{res[f'nhtsa_{k}_bilstm_s{s}']:.3f} ({leg2[f'nhtsa_{k}_bilstm_s{s}']:.3f})" for s in (0,1,2)]
    out.append(f"| {n} | "+' | '.join(row)+' |')
out.append('\n## Contrasts (Δ macro-F1, Holm p within the twelve-contrast family)\n')
out.append('| Contrast | TF-IDF | BiLSTM 1 | BiLSTM 2 | BiLSTM 3 |\n|---|---|---|---|---|')
for a,b,n in (('summary','conseq','Summary − consequence'),('summary','remedy','Summary − remedy'),('remedy','conseq','Remedy − consequence')):
    t=c('tfidf',0,a,b); row=[f"{t['delta']:+.3f} ({t['p_holm']:.4f})"]+[f"{c('bilstm',s,a,b)['delta']:+.3f} ({c('bilstm',s,a,b)['p_holm']:.4f})" for s in (0,1,2)]
    out.append(f"| {n} | "+' | '.join(row)+' |')
out.append(f"\n## Pre-committed criterion: {'MET' if met else 'NOT MET'}\n")
out.append(f"All six BiLSTM summary contrasts positive and Holm-significant: {all(x['delta']>0 and x['p_holm']<0.05 for x in bil)}; both TF-IDF summary contrasts positive: {all(x['delta']>0 for x in tf)}. Summary − consequence range {min(sc):+.3f} to {max(sc):+.3f}; summary − remedy {min(sr):+.3f} to {max(sr):+.3f}. Consequence − remedy (descriptive): "+', '.join(f"{-x['delta']:+.3f} (p_Holm {x['p_holm']:.4f})" for x in rc)+'.\n')
# LaTeX table body in the S17 format
out.append('## Supplement Table S18 (S4.3, after Table S17) — ready to paste\n```latex')
out.append('\\begin{table}[h!]\\centering')
out.append('\\caption{Near-duplicate robustness of the NHTSA field hierarchy (prospectively specified before the analysis was run): campaigns whose token sets in any field have Jaccard similarity $\\geq$0.80 are confined to one side of the split, in addition to exact duplicates (%d groups against %d under exact keys; %d campaigns in multi-member groups). Held-out macro-F1 per field and run, and the summary contrasts per training with $p_{\\mathrm{Holm}}$ within the twelve-contrast family; TF-IDF contrasts are single deterministic evaluations. The test side differs from the random split, so absolute scores are not comparable with it.}' % (g['groups_neardup'],g['exact_groups_leg2'],g['campaigns_in_multi_groups']))
out.append('\\small\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{lcccc}\n\\toprule\n & TF-IDF & BiLSTM 1 & BiLSTM 2 & BiLSTM 3 \\\\\n\\midrule')
for k,n in F.items():
    out.append(f"{n} & {res[f'nhtsa_{k}_tfidf_s0']:.3f} & "+' & '.join(f"{res[f'nhtsa_{k}_bilstm_s{s}']:.3f}" for s in (0,1,2))+' \\\\')
out.append('\\midrule')
for a,b,n in (('summary','conseq','Summary $-$ consequence'),('summary','remedy','Summary $-$ remedy'),('remedy','conseq','Remedy $-$ consequence')):
    t=c('tfidf',0,a,b)
    out.append(f"{n} & ${t['delta']:+.3f}$ ({t['p_holm']:.4f}) & "+' & '.join(f"${c('bilstm',s,a,b)['delta']:+.3f}$ ({c('bilstm',s,a,b)['p_holm']:.4f})" for s in (0,1,2))+' \\\\')
out.append('\\bottomrule\n\\end{tabular}\n\\end{table}\n```\n')
out.append('## Main text, §6.2, after the temporal sentence (M:320)\n')
if met:
    out.append(f"Grouping near-duplicate filings to one side of the split (token-set Jaccard $\\geq$0.80 in any field, {g['campaigns_in_multi_groups']} campaigns in shared groups) leaves the ordering unchanged: the summary remains strongest under both models in every training, by ${min(sc):+.3f}$ to ${max(sc):+.3f}$ over the consequence field and ${min(sr):+.3f}$ to ${max(sr):+.3f}$ over remedy, all six BiLSTM contrasts Holm-significant (prospectively specified; Supplementary Table~S18).")
else:
    weak=[x['contrast'] for x in bil if not (x['delta']>0 and x['p_holm']<0.05)]
    out.append(f"Grouping near-duplicate filings to one side of the split (token-set Jaccard $\\geq$0.80 in any field) weakens the ordering: {'; '.join(weak)} no longer reach Holm significance (prospectively specified; Supplementary Table~S18). Template overlap therefore contributes to the hierarchy measured under the random split. [Discussion limitations gain one sentence per the register.]")
out.append('\n## Supplement S7 row (after the temporal-split row)\n')
out.append('Near-duplicate grouped split (Jaccard $\\geq$0.80, any field) & NHTSA & prospective extension \\\\')
out.append('\n## Pointer shift\n\nInserting Table S18 after S17 shifts the existing S18 and S19 to S19 and S20; main-text pointers to S18/S19 must be renumbered (QA register, Phase 4).')
open(D+'../NHTSA_NEARDUP_RESULT.md','w').write('\n'.join(out)+'\n')
print('\n'.join(out))
