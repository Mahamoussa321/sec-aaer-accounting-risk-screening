# Data provenance

## SEC Financial Statement Data Sets

Quarterly archives are obtained from the U.S. Securities and Exchange Commission Financial Statement Data Sets. The analysis uses 2009Q1 through 2026Q1. The code requires `sub.txt` and `num.txt` from each archive and derives filing-period accounting variables from XBRL facts.

Official landing page:

`https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets`

Archive pattern:

`https://www.sec.gov/files/dera/data/financial-statement-data-sets/YYYYqQ.zip`

## Accounting and Auditing Enforcement Releases

AAER metadata are collected from the SEC Accounting and Auditing Enforcement Releases index:

`https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases`

## Versioning policy

Raw SEC archives and scraped AAER index snapshots are not committed because they are large or refreshable. The adjudicated issuer-link decisions are committed, and the latest clean-run manifest records hashes for source code, labels, compact tables, and figures.
