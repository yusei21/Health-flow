# Data and clinical safety gates (prototype)

This repository is an academic navigation prototype, **not a validated triage service**.

## Official facility data (CNES)

Primary registry: [Ministry of Health open CNES dataset](https://dadosabertos.saude.gov.br/dataset/cnes-cadastro-nacional-de-estabelecimentos-de-saude).
The [DATASUS CNES extraction guide](https://wiki.datasus.gov.br/cnes/index.php/Portal_CNES)
documents municipality/UF exports including CNES identifier, facility name, street,
postal code, latitude and longitude.

Implementation requirements **before** replacing the OSM provider:
1. Inspect the selected official resource schema and update date; do not guess API
   query parameters or assume different export formats have identical columns.
2. Record CNES identifier, source URL, reference month/date and import timestamp.
3. Validate latitude/longitude bounds, duplicate identifiers and missing addresses.
4. Resolve UBS, UPA and hospital type from official establishment/activity codes,
   not merely from a name substring; reject ambiguous/unmapped categories.
5. Verify service-to-SUS linkage using an appropriate official field or source.
   **Registered does not imply SUS eligibility, live opening hours or available beds.**
6. Reconcile geolocation discrepancies and stale coordinates without inventing
   facility locations.
7. Explicitly report data source, update date, unavailable searches and unmatched
   records. Keep OSM data clearly identified if used as a separate fallback.

The current live provider remains OSM; no CNES runtime integration is claimed.

## Clinical safety

The rules under `app/safety/rules.py` are labelled academic/simulated and **must
not be represented as Ministry of Health protocols**. Before deployment:
- Obtain an authorized clinician's review of every red flag and escalation.
- Link each rule to an identified version of a Brazilian clinical/public-health
  reference, with date and reviewer.
- Test affirmative, negative, uncertain, misspelled and contradictory Portuguese
  narratives, and mixed symptoms.
- Track false-negative and false-positive risk; emergency guidance must not rely
  solely on synthetic model confidence.
- Do not infer that an absent mention of a red flag is a denial.
- Treat missing or contradictory details as uncertainty, not a definitive
  diagnosis or facility recommendation.

## Test and release gates

`make lint && make typecheck && make test` and `npm run build` in frontend
are automated by GitHub Actions. The Ollama test is separately opt-in:
`make test-ollama` with the model server running and its model installed.
CI does not run that integration against an external/local Ollama service.
After CI is green, configure the **main branch ruleset** in GitHub Settings to
require the `backend` and `frontend` status checks and pull requests.

Do not use real identifiable patient narratives as test fixtures.
