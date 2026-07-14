# AAER issuer-link adjudication

The initial fuzzy-name procedure produced candidate links between SEC filer names and AAER release text. Generic tokens, personal surnames, auditors, law firms, and substring collisions can create false matches, so fuzzy scores were never treated as final labels.

## Final adjudication set

- Candidate rows retained for the 2009–2026 analysis window: **495**
- Accepted genuine issuer/company links: **104**
- Rejected mismatches or non-issuer references: **391**
- Blank or invalid decisions: **0**

A candidate received `keep_label=1` only when the named filer was the issuer/reporting entity involved in the release, including verified successor or former-name relationships. Individual-only, auditor-only, law-firm-only, incidental, and false-substring matches received `keep_label=0`.

The committed CSV preserves the candidate metadata, decision, and audit columns. Downstream filing-period labels are generated with the configured forward AAER window rather than by treating every fuzzy candidate as positive.
