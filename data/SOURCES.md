# Knowledge-base sources

This RAG demo indexes **openly available, attributed excerpts** for educational / portfolio use.
It is **not medical advice** and is **not affiliated with** WHO, FDA, Health Canada, MHRA, NCI, or NLM.

| File | Organization | Licence / reuse notes | Original URL |
|------|--------------|-----------------------|--------------|
| `knowledge/medlineplus_mammography.md` | MedlinePlus / U.S. National Library of Medicine | Public domain U.S. government work; attribute MedlinePlus/NLM | https://medlineplus.gov/mammography.html |
| `knowledge/nci_mammograms.md` | National Cancer Institute (NIH) | Public domain U.S. government work | https://www.cancer.gov/types/breast/mammograms-fact-sheet |
| `knowledge/fda_hc_mhra_gmlp.md` | FDA, Health Canada, UK MHRA | Joint GMLP principles; FDA PDF is a U.S. government work; Canada.ca page reproduced with attribution for educational non-commercial use | https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/good-machine-learning-practice-medical-device-development.html · https://www.fda.gov/media/153486/download |
| `knowledge/fda_mlmd_transparency.md` | FDA, Health Canada, UK MHRA | FDA web publication / joint guiding principles with attribution | https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/transparency-machine-learning-enabled-medical-devices-guiding-principles |
| `knowledge/who_ai_ethics_principles.md` | World Health Organization | **CC BY-NC-SA 3.0 IGO** | https://www.who.int/publications/i/item/9789240029200 |
| `knowledge/who_lmm_guidance.md` | World Health Organization | **CC BY-NC-SA 3.0 IGO** | https://www.who.int/publications/i/item/9789240084759 |
| `knowledge/who_regulatory_considerations.md` | World Health Organization | **CC BY-NC-SA 3.0 IGO** | https://www.who.int/publications/i/item/9789240078871 |

## Licence split

- **Application code** in this repository: MIT (see `LICENSE`).
- **WHO excerpts** in `data/knowledge/who_*.md`: remain under **CC BY-NC-SA 3.0 IGO**. They are adapted for non-commercial educational use with attribution; share-alike applies to those adapted texts. The WHO logo is not used and no WHO endorsement is implied.
- **U.S. government works** (MedlinePlus topic summary text used here, NCI fact sheet text, FDA materials): generally not copyrighted under U.S. law; please keep source attribution.

## How attribution appears in the API

Each markdown file starts with YAML frontmatter (`title`, `source`, `license`, `url`).  
Those fields are stored with every FAISS chunk and returned in `POST /v1/query` under `sources[]`.
