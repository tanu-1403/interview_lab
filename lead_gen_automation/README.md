# Lead Generation Automation

Automated Python pipeline that collects, enriches, cleans, and exports
nonprofit lead data to a formatted Excel workbook.

---

## Quickstart

```bash
# 1 — install dependencies
pip install -r requirements.txt

# 2 — run once (output saved to output/ folder)
python lead_scraper.py

# 3 — run on a schedule (repeats every 24 hours)
python lead_scraper.py --schedule 24
```

---

## How it works

The script runs a clean 4-stage pipeline:

| Stage | What happens |
|-------|-------------|
| 1. Collect | Hits the Open Data Soft US-Nonprofits public API (no key needed). Falls back to a curated 30-entry seed dataset if the API is unavailable. Always guarantees >= 30 records. |
| 2. Enrich  | Generates `info@<domain>` contact emails by parsing each org's website URL. Constructs probable LinkedIn `/company/` URLs from the org name. |
| 3. Clean   | Strips whitespace, deduplicates on Name+Location, fills missing values with "N/A", validates email format, sorts by sector. |
| 4. Export  | Writes a timestamped, colour-formatted `.xlsx` workbook to `output/`. |

---

## Output

Each run produces a file like `output/leads_20250509_143201.xlsx` with two sheets:

| Sheet | Contents |
|-------|----------|
| **Leads** | Full dataset — 30+ records with all fields |
| **Summary** | Key metrics: total leads, email coverage, sector count, timestamp |

---

## Fields collected

| Field | Description |
|-------|-------------|
| Name | Organisation name |
| Location | City, State |
| Website | Official website URL |
| Email | Generated `info@domain` contact email |
| LinkedIn | Inferred LinkedIn `/company/` URL |
| Sector | Human-readable NTEE sector |
| Assets_USD | Reported assets in USD |
| Email_Valid | Format validation (Yes / No) |
| Source | API or Seed |

---

## Project structure

```
lead_gen_automation/
├── lead_scraper.py       # main automation script
├── requirements.txt      # Python dependencies
├── README.md
├── submission_note.txt   # 5-line approach summary
├── output/               # Excel output files saved here
│   └── leads_YYYYMMDD_HHMMSS.xlsx
└── logs/
    └── run.log           # timestamped execution log
```

---

## Bonus features

- **Email generation** — derives `info@domain` from website URL
- **LinkedIn URL generation** — creates `/company/` slug from org name
- **Scheduled automation** — `--schedule N` flag for recurring runs
- **Styled Excel output** — blue header, alternating rows, pink N/A cells, frozen pane
- **Summary sheet** — key metrics auto-calculated per run
- **Run log** — all events timestamped in `logs/run.log`

---

## Tools used

| Library | Purpose |
|---------|---------|
| `requests` | HTTP data fetching from public API |
| `pandas` | Data manipulation, deduplication, cleaning |
| `openpyxl` | Styled Excel workbook generation |
| `schedule` | Optional recurring scheduler |
| `logging` | Timestamped run log |
