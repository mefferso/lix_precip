[README.md](https://github.com/user-attachments/files/26858980/README.md)
# WFO LIX 24-Hour Precipitation Map Starter Repo

This repo builds a static PNG precipitation map for the WFO LIX CWA using:

- Synoptic Weather API `/stations/precip` for derived 24-hour totals
- NWS GIS shapefiles for county/parish boundaries and WFO CWA boundaries
- Python + Matplotlib + GeoPandas for the actual map drawing
- GitHub Actions to rebuild daily
- GitHub Pages to host the finished image and a tiny web page

## Files

- `scripts/fetch_lix_precip.py` — downloads data and builds the PNG
- `docs/index.html` — simple page for GitHub Pages
- `.github/workflows/daily_precip_map.yml` — scheduled workflow
- `requirements.txt` — Python dependencies

## Important

Do **not** hardcode your Synoptic API token in the repo. Add it as a GitHub Actions secret named `SYNOPTIC_TOKEN`.

## Quick setup

1. Create a new GitHub repo.
2. Upload everything in this starter repo.
3. In GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
4. Create a secret named `SYNOPTIC_TOKEN` and paste your token there.
5. In GitHub: **Settings → Pages**.
6. Under **Build and deployment**, choose **Deploy from a branch**.
7. Select the `main` branch and `/docs` folder.
8. Save.
9. Go to **Actions**, open **Build WFO LIX 24-hour precipitation map**, and click **Run workflow**.
10. When the run finishes, open the GitHub Pages URL shown in Settings → Pages.

## Optional edits you will probably want

- Change the bbox constants in `scripts/fetch_lix_precip.py`
- Change colors/levels to match your office style exactly
- Add CoCoRaHS direct CSV ingest if you want another citizen gauge layer
- Change the title banner text
- Tweak label density if the map gets too cluttered

## Manual run with a custom end time

You can also run the script locally with a custom UTC end time:

```bash
export SYNOPTIC_TOKEN=your_token_here
python scripts/fetch_lix_precip.py 202604161200
```

That example builds a map for the 24 hours ending 12Z on 2026-04-16.
