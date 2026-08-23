# Citable release procedure

The repository is **private** at `github.com/AKSHAJ-SHELL/pumpwatch`, tagged `v1.0.0`.
This file is the sequence for turning it into a DOI the paper can cite.

## The constraint that shapes everything

**Zenodo's GitHub integration only archives public repositories.** The webhook cannot
see a private repo, so no DOI can be minted through that route while the repo stays
private. This is not a setting that can be worked around.

That matters because making the repo public before submission puts your name on
discoverable work, which many double-blind venues treat as an anonymity breach. The
two goals — a citable DOI at submission, and anonymity during review — are in tension,
and which one wins depends on the venue.

## Route A — venue is single-blind or preprint-friendly

Most engineering journals are single-blind. If yours is, this is the short path.

1. `gh repo edit AKSHAJ-SHELL/pumpwatch --visibility public --accept-visibility-change-consequences`
2. Log into [zenodo.org](https://zenodo.org) with GitHub.
3. Go to **Account → GitHub**, find `pumpwatch`, toggle it **On**.
4. Create a GitHub *Release* from the existing tag — a tag alone does not fire the
   webhook:
   `gh release create v1.0.0 --title "v1.0.0" --notes-file RELEASE_NOTES.md`
5. Zenodo archives it within a minute or two and mints a DOI. Add the badge to
   `README.md` and cite the DOI in the paper's data-availability statement.

## Route B — venue is double-blind, and you want a DOI anyway

Zenodo lets you **reserve** a DOI on a record you have not published yet, so the paper
can cite a real, permanent identifier at submission while the artifact stays closed.

1. On Zenodo, **New upload**. Do not connect GitHub.
2. Upload a source snapshot: `git archive --format=zip -o pumpwatch-v1.0.0.zip v1.0.0`
3. Fill the metadata from [.zenodo.json](.zenodo.json) — it is already written.
4. Click **Reserve DOI**. Cite that DOI in the paper.
5. Set **Access** to *Restricted*, or open with an **Embargo** date past your expected
   decision. Publish the record; the DOI resolves, the files stay closed.
6. On acceptance: make the GitHub repo public, lift the embargo, and optionally
   connect the GitHub integration for future versions.

⚠️ A published Zenodo record cannot be deleted, and a minted DOI is permanent. Reserve
and embargo rather than publishing openly if there is any doubt about the venue.

## Route C — double-blind, no DOI needed at submission

Simplest, and adequate. Cite the artifact as *"Source code and results will be made
publicly available on acceptance"* — reviewers accept this routinely. Then run Route A
after the decision. Add an anonymised browsable mirror
([anonymous.4open.science](https://anonymous.4open.science)) only if the venue asks
for one.

## What is in the artifact

- 91 tracked files, 232 tests, MIT licensed
- `results/*.json` **is** tracked, so `make tables` and `make figures-all` reproduce
  every table and figure in the paper in seconds, without the 20.8 GB Twente download
- **No dataset files.** Every loader raises with download instructions for its public
  CC BY 4.0 source, so the artifact redistributes nothing
- Attribution obligations discharged in [ATTRIBUTION.md](ATTRIBUTION.md): TabPFN's
  Prior Labs License §10, and CC BY 4.0 for both datasets
- No hostname or personal identifier appears in any tracked filename or result file
