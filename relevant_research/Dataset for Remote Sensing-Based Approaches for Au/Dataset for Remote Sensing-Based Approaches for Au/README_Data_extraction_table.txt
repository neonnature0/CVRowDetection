README — Data extraction table (systematic review)
Associated manuscript: “Remote Sensing-Based Approaches for Automatic Vineyard Area Identification: A Systematic Review”

Files
- LitRev_RS_GroupsALL_vF.xlsx  (main extraction table; 80 rows × 44 columns)

Unit of analysis
- One row = one empirical primary study included in the qualitative synthesis.

What this dataset contains
- Study-level extracted variables covering: bibliographic/venue metadata; applications & task flags; remote-sensing system/sensor/modality/resolution; input information types; method category and specific methods; best-reported metrics; risk-of-bias/applicability appraisal (D1–D5 + overall); and a reproducibility tag.

Missing-value conventions (as used in the spreadsheet)
- “—” (dash) = not reported / insufficiently specified in the source study (used in some RS-characteristic fields).
- “-” (hyphen) = not applicable or not available (used in SCImago/SJR fields).

Delimiters and formatting
- Multiple items within a cell are separated by “;”.
- Method Category may use “+” for integrated hybrids and “;” for multiple families reported side-by-side.

COLUMN-BY-COLUMN DATA DICTIONARY (44 columns, in sheet order)

1) ID_GROUP
   - Internal numeric identifier: sequential within each Group, ordered by publication year.
   - Auxiliary key only (no semantic meaning). (Adapted from the extraction template.)

2) ID_ANO
   - Internal numeric identifier: sequential over the whole corpus ordered by publication year.
   - Auxiliary key only (no semantic meaning). (Adapted from the extraction template.)

3) Group
   - Application-scale group:
     A = vineyard parcel/field
     B = vine rows
     C = individual vines (plant-level)
     R = regional assessments

4) Citation
   - Short in-text style reference string (author–year).

5) Applications & Outputs
   - Free-text description of the study’s mapping application(s)/outputs (may list multiple items).

Task flags (binary: 1 present, 0 absent; multi-task studies can have multiple 1s)
6) Detection
7) Delineation
8) Segmentation
9) Classification
10) Estimation

Remote sensing characteristics
11) System
   - Acquisition system type(s): Satellite / Aerial / Proximal (or combinations separated by “;”).

12) Platform+Sensor
   - Platform and/or sensor name(s) as reported.
   - “—” = not reported.

13) Spectral Modality
   - Modality/modalities as reported (e.g., RGB, Multispectral, Hyperspectral, Panchromatic, Thermal, etc.).

14) Resolution (m)
   - Spatial resolution(s) in meters as reported (single value, list, or range; “;” for multiple).
   - “—” = not reported.

15) Spectral Modality (m)
   - Harmonised modality+resolution notation (e.g., “RGB+MS [0.05]”, “MS [10; 0.05]”).
   - “—” = not reported / insufficient information.

Input information type flags (binary: 1 used, 0 not used; multiple can be 1)
16) Spectral (Bands)
17) Spectral (Vegetation Indices)
18) Spectral (Others)
19) Textural
20) Geometric
21) Phenological
22) Thermal

Methods and performance reporting
23) Category
   - Methodological family label(s); may include hybrids (use of “+” and “;” as noted above).

24) Method(s)
   - Specific algorithms/models as reported by the study (free text).

25) Best Metrics
   - Best-reported evaluation metrics and values (as reported by the study; free text).

Risk of bias / applicability (domain ratings)
- Domain coding (letters in the sheet):
  L = Low risk
  S = Some concern
  H = High risk
(Definitions follow the manuscript’s RoB framework and the extraction template.)

26) D1 (coverage)
27) D2 (sensor & pre-processing)
28) D3 (Ground-Truth)
29) D4 (validation)
30) D5 (portability)

31) RoB (Global)
   - Overall RoB/applicability label (letters in the sheet):
     L = Low, M = Moderate, H = High
   - Decision rule (as stated in the manuscript):
     High if any domain is High; Low only if all domains are Low; otherwise Moderate.

Reproducibility
32) Reproducibility Tag
   - Ordinal tag summarising study reproducibility/artefact availability (letters in the sheet):
     L = Low concern (code+data openly provided and directly accessible)
     S = Some concern (partial/unclear availability; e.g., data public but code limited/on request)
     H = High concern (no code/data availability stated or findable)
   - Assigned from what the paper/supplements provide via working links (per extraction template).

Publication venue metadata
33) Publisher
34) Journal
35) Type
   - Publication format label (e.g., Article; Conference paper; Book chapter).

SCImago / SJR fields
36) Rank Scimago
   - Journal SJR quartile for journal articles where available (e.g., Q1, Q2, Q3).
   - “-” = not applicable / not available.

37) SJR_proceedings_numeric_by_year
   - Numeric SJR value for proceedings venues (conference papers/book chapters) where available.
   - “-” = not applicable / not available.

38) SJR_proc_Rank
   - Proceedings rank category derived from SJR_proceedings_numeric_by_year (as used in the manuscript’s analysis):
     Top    = SJR > 0.300
     Medium = 0.250–0.300
     Low    = SJR < 0.250
   - “-” = not applicable / not available.
   - Note: SJR is a snapshot collected during the review and may change as SCImago updates rankings.

Study context and full bibliographic strings
39) Country
   - Country/countries of the study area (not author affiliations); multiple entries separated by “;”.

40) Year
   - Publication year (4 digits).

41) Reference
   - Full reference string.

42) Title
   - Full title.

43) DOI/Link
   - DOI URL and/or stable link.

44) Abstract
   - Abstract text as stored for traceability of extraction.

Notes
- This dataset contains extracted/derived information and does not include copyrighted full texts.
- Please cite the associated manuscript and this dataset DOI when reusing the extraction table.
