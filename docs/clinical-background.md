# Clinical and Scientific Background

This document explains the physiology behind the features extracted by the pipeline: what the ECG records, how a myocardial infarction (MI) changes it, how the autonomic nervous system shapes heart rate variability (HRV), and why the signal processing choices in the code were made. Where useful, it links each concept to the feature names in `data/processed/ptb_features.csv` and to what the PTB dataset actually shows.

> This is educational material for a portfolio project, not clinical guidance.

---

## Contents
1. [The heart as an electrical system](#1-the-heart-as-an-electrical-system)
2. [Anatomy of a heartbeat on the ECG](#2-anatomy-of-a-heartbeat-on-the-ecg)
3. [The 12-lead ECG](#3-the-12-lead-ecg)
4. [Myocardial infarction](#4-myocardial-infarction)
5. [Heart rate variability](#5-heart-rate-variability)
6. [Why the signal processing is designed this way](#6-why-the-signal-processing-is-designed-this-way)
7. [Confounders and caveats](#7-confounders-and-caveats)
8. [Feature reference](#8-feature-reference)
9. [References](#9-references)

---

## 1. The heart as an electrical system

### The cardiac action potential
Every heartbeat is triggered by an electrical wave that spreads through the heart muscle (myocardium). At rest, a cardiac muscle cell holds a voltage of about −90 mV across its membrane, maintained by potassium (K⁺) channels. When excited, it goes through a stereotyped **action potential**:

| Phase | Ion flow | Event |
|---|---|---|
| 0 — upstroke | Na⁺ rushes in | Rapid **depolarisation** |
| 1 — notch | Transient K⁺ out | Early partial repolarisation |
| 2 — plateau | Ca⁺⁺ in ≈ K⁺ out | Sustained depolarisation; Ca⁺⁺ triggers contraction |
| 3 — repolarisation | K⁺ out | Return to resting voltage |
| 4 — rest | K⁺ leak | Stable resting potential (in working myocardium) |

The ECG electrodes on the skin do not see single cells. They record the **sum of millions of these potentials**. The body surface voltage changes whenever the depolarisation or repolarisation front moves through the heart. A wave moving *toward* an electrode makes a positive deflection; a wave moving *away* makes a negative one.

### The conduction system
1. **Sinoatrial (SA) node:** pacemaker cells in the right atrium depolarise spontaneously, setting the heart rate (normally 60–100 bpm). Their firing rate is continuously adjusted by the autonomic nervous system, which is the physiological origin of HRV (section 5).
2. **Atria:** the wave spreads across both atria → **P wave**.
3. **Atrioventricular (AV) node:** conduction slows deliberately (≈ 100 ms) so the atria can finish filling the ventricles → flat **PR segment**.
4. **His bundle → bundle branches → Purkinje fibres:** fast conduction spreads the impulse to the ventricles almost at once → narrow, sharp **QRS complex**.
5. **Ventricular repolarisation** → **T wave**.

---

## 2. Anatomy of a heartbeat on the ECG

A single beat reads left to right as **P wave → PR segment → QRS complex (Q, R, S) → J-point → ST segment → T wave**. Real examples from this dataset, with the measured QRS shaded, are shown in [`reports/median_beats.png`](../reports/median_beats.png).

| Component | Represents | Typical adult values |
|---|---|---|
| P wave | Atrial depolarisation | < 120 ms |
| PR interval | Atrial depolarisation + AV-node delay | 120–200 ms |
| **PR segment** | Electrically quiet period → the **isoelectric baseline** | — |
| **QRS complex** | Ventricular depolarisation | 80–110 ms (≥ 120 ms = conduction block) |
| **Q wave** | First negative deflection of the QRS (septal depolarisation) | Small or absent in most leads |
| **R wave** | First positive deflection | Lead-dependent |
| **S wave** | Negative deflection after R | Lead-dependent |
| **J-point** | Junction where QRS ends and ST segment begins | — |
| **ST segment** | Ventricles fully depolarised (plateau, phase 2) → should be at baseline | Within ±0.1 mV |
| **T wave** | Ventricular repolarisation | Usually same direction as the QRS |
| R-R interval | Time between consecutive beats | 600–1000 ms at 60–100 bpm |

![ECG wave visualization](images/SinusRhythmLabels.svg 'By Created by Agateller (Anthony Atkielski), converted to svg by atom. - SinusRhythmLabels.png, Public Domain, https://commons.wikimedia.org/w/index.php?curid=1560893')

Two things matter for the morphology features:
- **The ST segment should be flat.** During phase 2 all ventricular cells are at a similar voltage, so no current flows and the trace returns to baseline. A shifted ST segment means something is creating a voltage difference between regions of the heart, classically injury from ischaemia.
- **The T wave is normally upright where the QRS is upright.** Repolarisation travels in the opposite direction to depolarisation (epicardium → endocardium) but with opposite polarity, so the two deflections end up pointing the same way. An inverted T wave in a lead where it should be upright is a sign of abnormal repolarisation.

---

## 3. The 12-lead ECG

Each lead is a different "camera angle" on the same electrical activity:

- **Limb leads (frontal plane):** I, II, III (bipolar) and aVR, aVL, aVF (augmented unipolar).
- **Precordial leads (horizontal plane):** V1–V6, placed across the chest from the right sternal border to the left mid-axillary line.

![Limb lead locations diagram](images/LimbLeads.png 'By Npatchett - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=39235260')
![Pericordial lads diagram](images/PericordialLeads.png 'By Mikael Häggström - Own work, CC0, https://commons.wikimedia.org/w/index.php?curid=20064293')

Because each lead faces a specific region of the left ventricle, and each region is supplied by a specific coronary artery, **the leads that show abnormalities locate the infarct**:

| Territory | Leads | Usual culprit artery |
|---|---|---|
| Septal | V1–V2 | Left anterior descending (LAD) |
| Anterior | V3–V4 | LAD |
| Lateral | I, aVL, V5–V6 | Left circumflex (LCx) or LAD diagonal branch |
| Inferior | II, III, aVF | Right coronary artery (RCA), sometimes LCx |
| Posterior | V1–V3 *reciprocal* changes (tall R, ST depression) | RCA or LCx |

This is why `morphology.py` measures every feature separately on all 12 leads rather than on lead II alone: an anterior MI may be invisible in the inferior leads, and the reverse.

The PTB database also records 3 Frank leads (vx, vy, vz) for vectorcardiography. The pipeline does not use them.

---

## 4. Myocardial infarction

### Pathophysiology
An MI happens when a coronary artery is blocked (usually by a ruptured atherosclerotic plaque with a blood clot on top). The heart muscle downstream goes through three stages, each with its own ECG signature:

1. **Ischaemia** (reduced oxygen, reversible): cells repolarise abnormally → **T-wave changes** (hyperacute tall T waves early on, later inversion).
2. **Injury** (severe ischaemia, still potentially reversible): injured cells cannot keep their resting potential, so they sit partially depolarised. The resulting voltage difference between healthy and injured tissue drives an **"injury current"** that shifts the ST segment:
   - **Transmural (full-thickness) injury** → **ST elevation** in leads facing the injured area (STEMI), with *reciprocal* ST depression in opposite leads.
   - **Subendocardial injury** → **ST depression**.
3. **Necrosis** (dead tissue, irreversible): dead myocardium produces no electrical activity. An electrode over it "looks through a window" at the opposite wall depolarising *away* from it → **pathological Q waves**, and the loss of muscle produces **reduced R-wave amplitude** (in severe cases a QS complex with no R at all).


This matters for interpreting the model: the PTB database contains both **acute and old (former) infarctions**. ST elevation is transient, but Q waves, R-wave loss and T-wave inversion are long-lasting. So we would expect the durable markers to be more consistently informative across the whole cohort, and they are (see the table below).

### Diagnostic thresholds (for context)
The Fourth Universal Definition of MI (Thygesen et al., 2018) uses, among others:
- **ST elevation** at the J-point in two contiguous leads: ≥ 0.1 mV in most leads; in V2–V3 ≥ 0.2 mV (men ≥ 40 y), ≥ 0.25 mV (men < 40 y), ≥ 0.15 mV (women).
- **Pathological Q waves** (prior MI): ≥ 30 ms wide and ≥ 0.1 mV deep in two contiguous leads, or any Q wave ≥ 20 ms in V2–V3, or QS complexes.

The pipeline measures the continuous quantities behind these criteria (`st_j`, `st_60`, `q_amp`, `r_amp`, `t_amp`) and lets the classifier learn the decision boundary, rather than hard-coding the thresholds.

### What the PTB data show

Medians after the quality filter (MI n = 341, Normal n = 78 recordings):

| Feature | MI | Normal | Interpretation |
|---|---|---|---|
| `v5_t_amp` (mV) | 0.11 | 0.51 | Flattened/inverted lateral T waves |
| `v6_t_amp` (mV) | 0.08 | 0.35 | ″ |
| `ii_t_amp` (mV) | 0.12 | 0.34 | Flattened/inverted inferior T waves |
| T inverted in V5 | 26 % of records | 0 % | ″ |
| `v5_r_amp` (mV) | 0.83 | 1.43 | R-wave loss from necrosis |
| `v2_q_amp` (mV) | 0.04 | 0.12 | Loss of the initial r wave in septal leads (see note) |
| `v2_st_60` (mV) | 0.11 | 0.12 | ST level — no consistent shift (many infarcts are not acute) |
| `qrs_duration` (ms) | 76 | 71 | Slight QRS widening |

**Note on `q_amp` in V1–V3.** In healthy hearts V1–V3 show an *rS* pattern with no Q wave, so the measured "Q" is simply the level at QRS onset, which is slightly positive as the small septal r wave begins. In anterior/septal MI that initial r wave is lost (a QS complex), pulling the value down. So in these leads `q_amp` behaves as a marker of **initial-r loss** rather than of Q-wave depth.

These are exactly the features the Random Forest ranks highest (`reports/feature_importance.png`): T-wave amplitudes in V5, II, V6, III and aVF, then Q/initial-QRS level in V1–V3, then R amplitude in V5.

---

## 5. Heart rate variability

### Why the heart rate varies
The SA node's intrinsic rate (≈ 100 bpm when isolated from nerves) is constantly adjusted by two opposing branches of the autonomic nervous system:

| Branch | Neurotransmitter | Effect on SA node | Speed |
|---|---|---|---|
| **Parasympathetic (vagal)** | Acetylcholine → muscarinic M2 receptors → opens K⁺ channels (I<sub>K,ACh</sub>) | Slows the rate | **Fast**: acts within one beat, fades within ~1 s |
| **Sympathetic** | Noradrenaline → β1 receptors → cAMP → speeds up the pacemaker current | Speeds up the rate | **Slow**: rises over seconds, fades over 10 s or more |

At rest the vagus dominates (which is why resting heart rate is below 100 bpm). The difference in speed is the foundation of HRV analysis. **Only the vagus can produce fast, beat-to-beat changes**, while sympathetic effects show up only in slower oscillations.

Three physiological rhythms modulate the R-R interval:
- **Respiratory sinus arrhythmia (RSA):** heart rate rises during inspiration and falls during expiration, mediated by vagal withdrawal and return. It oscillates at the breathing frequency (≈ 0.15–0.4 Hz, i.e. 9–24 breaths/min).
- **Baroreflex oscillations (Mayer waves):** the feedback loop between blood pressure and heart rate oscillates at ≈ 0.1 Hz, with both vagal and sympathetic contributions.
- **Very slow rhythms** (thermoregulation, hormones), visible only in long recordings.

### Why HRV falls after MI
Reduced HRV after MI is one of the best-established findings in the field. In a landmark study, post-MI patients with a 24-h SDNN < 50 ms had about **5× the mortality** of those with SDNN > 100 ms (Kleiger et al., 1987). Proposed mechanisms:
- **Reduced vagal and increased sympathetic activity**, partly driven by abnormal sensory signals from damaged, stretched myocardium.
- **Impaired baroreflex sensitivity.**
- Loss of vagal tone removes a protective effect against ventricular arrhythmias, which is why low HRV predicts sudden cardiac death.

HRV usually recovers partially over the weeks and months after an MI.

### Time-domain measures

| Feature | Definition | Physiological meaning |
|---|---|---|
| `mean_rr`, `mean_hr` | Average interval / rate | Overall autonomic set point |
| `sdnn` | Standard deviation of NN intervals | **Total variability**, both branches; in short recordings mainly RSA and baroreflex |
| `rmssd` | √(mean of squared successive differences) | **Beat-to-beat variability → vagal tone** (robust to slow trends) |
| `pnn50` | % of successive differences > 50 ms | Vagal tone; saturates at 0 when HRV is low |
| `sd_hr`, `cv_rr` | Variability of HR; SDNN normalised by mean RR | Variability corrected for heart rate, since slower hearts vary more in absolute ms |

"NN" (normal-to-normal) means intervals between normal sinus beats, after removing ectopic beats and detection errors (section 6).

### Frequency-domain measures
The NN series is treated as a signal and decomposed into frequencies with a power spectral density (PSD):

| Band | Range | Interpretation |
|---|---|---|
| **HF** (`hf_power`) | 0.15–0.40 Hz | Respiratory sinus arrhythmia → **vagal** |
| **LF** (`lf_power`) | 0.04–0.15 Hz | Baroreflex activity — **mixed** sympathetic + vagal |
| `lf_nu`, `hf_nu` | LF or HF / (LF + HF) × 100 | Relative distribution, independent of total power |
| `lf_hf_ratio` | LF / HF | Historically called "sympathovagal balance" |

**The LF/HF ratio should be interpreted with caution.** The idea that LF reflects sympathetic activity has been strongly challenged (Billman, 2013): LF power is largely driven by baroreflex-mediated *vagal* modulation and is not reduced by blocking sympathetic nerves. The PTB data illustrate the problem: MI and Normal groups have very different absolute LF and HF power (medians 74 vs 409 and 45 vs 325 ms²), yet **an identical median LF/HF ratio (1.47 in both)**. Both branches of the spectrum shrink together, and the ratio hides it.

### Non-linear measures

**Poincaré plot.** Plot each RR interval against the next (RRₙ vs RRₙ₊₁). Healthy hearts produce an elongated, comet-shaped cloud along the identity line.
- `sd1`: spread perpendicular to the identity line → **short-term, beat-to-beat** variability. Mathematically SD1 ≈ RMSSD/√2 (in this dataset the two agree to within 0.04 ms), so it is a vagal index.
- `sd2`: spread along the identity line → **longer-term** variability, related to SDNN.
- `sd1_sd2_ratio`: shape of the cloud — the balance between short- and long-term variability.

**Sample entropy** (`sample_entropy`) asks: if two short stretches of the rhythm (m = 2 beats) look similar, how likely are they to still look similar one beat later? Highly regular, predictable series have **low** entropy; complex, adaptive series have higher entropy. Healthy physiology tends to show "organised complexity" that is lost with ageing and disease. In PTB: median 1.38 (MI) vs 1.54 (Normal).

### Recording-length requirements
Each band needs the recording to contain several cycles of its slowest oscillation. The Task Force (1996) recommends ≥ 1 min for HF and ≥ 2 min for LF, and 5-minute recordings as the short-term standard. PTB recordings are mostly ~115 s, so:
- time-domain measures and HF power are reliable;
- LF is computed from 90 s upwards and should be treated as a short-term estimate;
- the strong 24-h prognostic findings (e.g. SDNN < 50 ms) **do not transfer directly** to 2-minute resting recordings.

---

## 6. Why the signal processing is designed this way

### Two filter bands
ECG noise has distinct sources at different frequencies:

| Noise | Frequency | Removed by |
|---|---|---|
| Baseline wander (breathing, electrode movement) | < 0.5 Hz | High-pass |
| Power-line interference | 50 Hz (Europe — PTB was recorded in Germany) | 45 Hz low-pass |
| Muscle activity (EMG) | 20 Hz up to several hundred Hz | Low-pass (partially) |

- **Detection band (0.5–45 Hz):** aggressive enough to give the QRS detector a clean, stable signal.
- **Diagnostic band (0.05–45 Hz):** the American Heart Association recommends a 0.05 Hz low-frequency cut-off for diagnostic ECGs (Kligfield et al., 2007). The ST segment and T wave contain very low-frequency content; a 0.5 Hz high-pass filter can **create artificial ST-segment shifts**, the very thing being measured.
- **Zero-phase filtering** (filtering forwards and backwards with `sosfiltfilt`) avoids shifting waves in time, which would distort interval and amplitude measurements.
- **Second-order sections:** at 1000 Hz sampling, 0.05 Hz is 0.0001 of the Nyquist frequency. A filter in the classic (b, a) polynomial form is numerically unstable at such extreme cut-offs, while the cascaded second-order form is stable.

A trade-off to know about: the AHA recommends a **150 Hz** upper cut-off for diagnostic ECGs in adults. The 45 Hz low-pass used here slightly blunts sharp QRS peaks, reducing measured Q/R/S amplitudes. Because the same filter is applied to every recording, the features remain comparable within the dataset.

### R-peak detection
Detection uses the Hamilton algorithm, a descendant of Pan-Tompkins (1985). The QRS complex is the steepest, highest-energy part of the ECG, so these algorithms differentiate the signal, square it to emphasise steep slopes, average it over a QRS-width window, and apply adaptive thresholds with a refractory period (no second beat within ~200 ms).

### From R-R to NN intervals
HRV assumes every interval comes from a normal sinus beat. Two things break this:
- **Ectopic beats** (e.g. premature ventricular contractions) produce a short interval followed by a long compensatory pause. A single one inflates RMSSD dramatically.
- **Detection errors**: a missed beat doubles an interval; a false detection splits one.

`clean_rr_intervals` rejects intervals outside 300–2000 ms (200–30 bpm) and those more than 20 % from the median. `nn_fraction` records how many intervals survived and is used as a quality filter during training.

### Median beat templates
Averaging aligned beats improves the signal-to-noise ratio (random noise falls with √N for N beats). The **median** is used rather than the mean because it ignores outliers such as an occasional ectopic beat or a burst of muscle noise. `template_corr` (how well each beat matches the template) flags recordings where beats are inconsistent, from noise or arrhythmia.

### Global QRS delineation
QRS onset and offset are found from the **summed slope of all 12 leads**, not from a single lead. Some leads show a wave starting earlier or ending later than others because they view the heart from different angles. Clinicians measure QRS duration from the earliest onset to the latest offset across leads, and this approach does the same. All amplitudes are measured relative to the **PR segment**, the closest thing the ECG has to an electrical zero.

---

## 7. Confounders and caveats

| Factor | Effect | Relevance to PTB |
|---|---|---|
| **Age** | HRV (SDNN, RMSSD, HF) declines steadily with age (Umetani et al., 1998); R and T amplitudes also change | Controls are ~23 years younger (median 37 vs 59). An **age-only model reaches AUC 0.82**, better than HRV alone (0.79) |
| **Sex** | Women have slightly higher HF power and different ST thresholds | Cohorts are not sex-matched |
| **Medication** | β-blockers *increase* HRV and lower HR; many drugs alter repolarisation | Many MI patients were on medication at recording time (recorded in the header comments) |
| **Respiration** | HF power depends on breathing rate and depth | Not controlled or recorded |
| **Posture / setting** | HRV differs between supine, standing and stressed states | Resting hospital recordings |
| **Infarct age** | Acute vs old MI produce different ECG patterns | Mixed in the MI class |

The practical consequence: **an HRV difference between MI patients and healthy controls in this dataset cannot be attributed to the infarction alone**. Morphology features are less exposed to this problem because Q waves, R-wave loss and T-wave inversion are direct electrical consequences of damaged myocardium. They still show some age-related change, which is why the README reports an age ≥ 40 robustness check.

---

## 8. Feature reference

| Feature | Group | Physiology | Expected change in MI |
|---|---|---|---|
| `mean_rr` / `mean_hr` | HRV – time | Autonomic set point | ↓ RR / ↑ HR |
| `sdnn` | HRV – time | Total variability | ↓ |
| `rmssd`, `pnn50` | HRV – time | Vagal tone | ↓ |
| `sd_hr`, `cv_rr` | HRV – time | Rate-normalised variability | ↓ |
| `hf_power`, `hf_nu` | HRV – frequency | Respiratory vagal modulation | ↓ (absolute) |
| `lf_power`, `lf_nu` | HRV – frequency | Baroreflex (mixed) | ↓ (absolute) |
| `lf_hf_ratio` | HRV – frequency | Contested "sympathovagal balance" | Unreliable |
| `sd1`, `sd2`, `sd1_sd2_ratio` | HRV – non-linear | Short- vs long-term variability | ↓ SD1, ↓ SD2 |
| `sample_entropy` | HRV – non-linear | Rhythm complexity | ↓ |
| `qrs_duration` | Morphology | Ventricular conduction time | ↑ (scar slows conduction) |
| `<lead>_q_amp` | Morphology | Initial QRS / Q-wave depth | More negative (Q waves), loss of initial r |
| `<lead>_r_amp` | Morphology | Viable depolarising myocardium | ↓ in infarct territory |
| `<lead>_s_amp` | Morphology | Terminal depolarisation | Lead-dependent |
| `<lead>_st_j`, `<lead>_st_60` | Morphology | Injury current | ↑ facing acute transmural injury; ↓ reciprocal / subendocardial |
| `<lead>_t_amp` | Morphology | Repolarisation | ↓ / inverted (ischaemia, evolving or old MI) |
| `n_beats`, `nn_fraction`, `template_corr` | Quality | Signal and rhythm quality | Used for filtering, not prediction |

---

## 9. References

- Billman GE. The LF/HF ratio does not accurately measure cardiac sympatho-vagal balance. *Frontiers in Physiology* 4:26, 2013.
- Bousseljot R, Kreiseler D, Schnabel A. Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über das Internet. *Biomedizinische Technik* 40(S1):317, 1995.
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220, 2000.
- Kleiger RE, Miller JP, Bigger JT, Moss AJ. Decreased heart rate variability and its association with increased mortality after acute myocardial infarction. *American Journal of Cardiology* 59(4):256–262, 1987.
- Kligfield P, et al. Recommendations for the standardization and interpretation of the electrocardiogram, Part I. *Circulation* 115(10):1306–1324, 2007.
- Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering* 32(3):230–236, 1985.
- Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology – Heart and Circulatory Physiology* 278(6):H2039–H2049, 2000.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation* 93(5):1043–1065, 1996.
- Thygesen K, et al. Fourth Universal Definition of Myocardial Infarction (2018). *Circulation* 138(20):e618–e651, 2018.
- Umetani K, Singer DH, McCraty R, Atkinson M. Twenty-four hour time domain heart rate variability and heart rate: relations to age and gender over nine decades. *Journal of the American College of Cardiology* 31(3):593–601, 1998.
