# GovBA-GAR

## Governance-Aware Agentic Retrieval for Public-Sector Knowledge Work

**GovBA-GAR** is the third-generation research prototype of **GovBA Assistant**, a bilingual AI-assisted platform designed to support evidence-aware, governance-sensitive public-sector knowledge work.

The project evolved from rule-based business-analysis utilities into a modular architecture combining governed retrieval, authoritative-source handling, temporal intelligence, security controls, policy-change analysis, correspondence intelligence, requirements engineering, and controlled AI assistance.

> **Research status:** GovBA-GAR forms part of an academic research paper currently in preparation. The final paper citation and DOI will be added following publication.

---

## Live Prototype

### https://govba-gar.streamlit.app/

The public application is provided for research, demonstration, and evaluation purposes.

GovBA-GAR is a decision-support prototype. Human review remains required before generated, retrieved, or structured outputs are used in operational public-sector settings.

---

## System Evolution

| Version | Main Focus |
|---|---|
| **V1** | Rule-based public-sector business-analysis tools |
| **V2** | Controlled AI integration and AI-assisted correspondence processing |
| **V3 – GovBA-GAR** | Governance-aware retrieval, evidence, temporal reasoning, security, and multi-capability intelligence |

The repository preserves this evolution to support software-engineering traceability and research comparison.

---

## V3 Architecture

```text
User Request
      │
      ▼
Language / Security Controls
      │
      ▼
Capability Router
      │
      ├── Direct Assistance
      ├── Government RAG
      ├── Official-Web Intelligence
      ├── Policy Change Intelligence
      ├── Correspondence Intelligence
      └── Requirements / BRD Intelligence
      │
      ▼
Evidence + Provenance
      │
      ▼
Temporal / Governance Controls
      │
      ▼
READY / REVIEW / ABSTAIN
      │
      ▼
Human-Reviewed Output