# Paper TODO

This file is the active manuscript-cleanup queue. It was audited on 2026-04-22
against the current arXiv source. Items are kept only when the source text is
still live and the rewrite materially improves readability, precision, or
reviewer trust.

## Working Rule

- Work one item at a time.
- After each manuscript edit, rebuild from `paper/arxiv` with:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Reread the edited paragraph in context.
- Scan for the same pattern elsewhere before deleting the completed TODO item.
- Do not chase taste-only rewrites once the sentence is clear, accurate, and
  non-robotic.

## Audit Verdict

We agree with the core flags, but the previous TODO had too much duplication.
The real live issues are concentrated in:

- defensive or legalistic caveats around configuration selection and theory scope
- artifact style noun stacks in the Results captions and paragraph transitions
- `endpoint` wording where the paper should say `full feature set` or `$k=p$`
- stacked feature subsampling phrasing
- appendix headings/captions that still read like internal notes

We do not need to keep every six-agent suggestion. Several flagged sentences are
acceptable as written, especially where the proposed rewrite is only a style
preference.

## Hyphenation Audit Status

Completed on 2026-04-24. The source-facing true positives from the hyphenation
audit were fixed or deliberately left when they are technical notation, method
abbreviations, internal labels, or bibliography titles.

Allowed residual source patterns:
- technical notation: `p-value`, `top-$k$`, `top-1`, `high-$p$`, `fixed-$B$`,
  `Phipson--Smyth`, `Westfall--Young`, and table ranges such as `4.0--8.4`
- internal LaTeX labels such as `sec:feature-rankings`, `assump:fixed-node`,
  and `tab:real-dataset-inventory-clf`
- bibliography titles that contain `post-selection`
- the explicit method abbreviation definition `(CIF-all)`

## Active Queue

No active items remain after the 2026-04-24 abstract, introduction, and Methods
cleanup pass.

## Do Not Chase

These were raised by agents but should not become standalone work unless they
block a concrete edit:
- A new node-flow figure. Use prose first.
- A large main-text controls table. Consider only if the methods section still
  feels underdefined after prose cleanup.
- Broad hyphenation sweeps. Reopen only if a concrete source phrase regresses.

## Deferred or Demoted

Do not chase these unless they recur after the active queue is done.

- Abstract aim sentence: acceptable if it uses the current `ranking quality`
  wording. `Runtime` is the right word here; do not reopen unless the abstract
  is being rewritten anyway.
- Abstract 14-dataset sentence: dense but accurate; not a priority unless the
  abstract is being rewritten anyway.
- Introduction CART-bias sentence: acceptable. A light edit from `via` to
  `with` is fine, but not required.
- Introduction contribution list: acceptable after recent edits.
- Experiments averaging sentence: clear enough; no need to replace just because
  it is plain.
- Complete-results / complete-case wording: only fix visible `complete-case` or
  `complete support`. Current `complete results` phrasing is fine.
- CIT/CIF variant sentence in Section 4: acceptable.
- Appendix order: no current action. Keep the experiment-first appendix order
  unless a reviewer complains.
- Internal LaTeX labels containing `fixed-panel`: do not rename solely for style.
- Broad old roadmap items about adding result layers: most have already been
  addressed by the current manuscript. They are no longer active TODO items.

## Completed Verification

Completed on 2026-04-24:

- rebuilt the PDF with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- checked `main.log` for `LaTeX Warning`, `Overfull`, and `Underfull`
- ran the closing source scans for prose-regression and hyphenation terms
- verified the Table 2 regression note appears in the built PDF text
- checked the SVR methods sentence against the local `SVR()` defaults
