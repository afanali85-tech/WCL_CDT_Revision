# Cognitive Digital Twins for Self-Aware Channel Estimation

**A Model-Agnostic Framework for Distribution Drift Monitoring and Autonomous AI/ML Channel Estimator Lifecycle Management**


## 1. Overview

Artificial intelligence and machine learning (AI/ML) methods have demonstrated promising performance for wireless channel estimation. However, conventional neural channel estimators are generally trained under predefined propagation conditions and may experience significant performance degradation when deployed in environments with different channel statistics.

A channel estimator trained under Extended Pedestrian A (EPA) conditions, for example, may later encounter Extended Vehicular A (EVA) or Extended Typical Urban (ETU) propagation characteristics. Since ground-truth channel state information (CSI) is generally unavailable during deployment, the estimator cannot directly evaluate its instantaneous normalized mean-square error (NMSE).

This problem motivates the development of **self-aware channel estimation**, where a deployed estimator can monitor changes in its operating environment and initiate appropriate maintenance actions.

This repository implements a **model-agnostic cognitive digital twin (CDT) framework** that integrates receiver-observable distribution monitoring, autonomous estimator lifecycle decisions, and generative model replacement.

The framework supports three lifecycle actions:

- **CONTINUE:** Maintain the deployed estimator when no persistent distribution mismatch is confirmed.
- **UPDATE:** Fine-tune the existing estimator when recoverable distribution drift is detected.
- **RETIRE:** Initiate generative model replacement when severe mismatch is confirmed.

The framework does not require measured ground-truth full-channel CSI during deployment. Instead, receiver-side pseudo-targets obtained from correctly decoded data symbols support adaptation and validation.

## 2. Main Research Contributions

### 2.1 Dual-Domain Distribution Monitoring

The CDT integrates two monitoring indicators.

**VAE-based pilot monitoring** evaluates how closely newly received pilot observations resemble the reference distribution learned during initial training.

**Activation-statistics monitoring** tracks changes in the internal representations of the deployed channel estimator.

Together, these indicators provide receiver-observable information about potential distribution mismatch without requiring environment labels or ground-truth CSI.

### 2.2 Validation-Guided Lifecycle Management

A hierarchical lifecycle controller interprets the monitoring signals and determines whether the deployed estimator should be maintained, updated, or replaced.

The controller uses exponentially weighted moving average (EWMA) smoothing, calibrated thresholds, persistence checks, and additional confirmation before severe lifecycle actions.

Model updates are accepted only after held-out receiver-side validation.

### 2.3 Generative Estimator Replacement

For severe distribution mismatch, the CDT uses an exponential moving average (EMA) vector-quantized variational autoencoder (VQ-VAE) to learn post-drift channel representations.

The generative pipeline produces synthetic channel realizations that support supervised training of a replacement estimator without measured ground-truth full-channel CSI during deployment.

The replacement is deployed only when its validation loss improves over that of the existing estimator.

---

## 3. Overall CDT Framework

The complete framework connects the physical wireless environment to a digital estimator-monitoring and lifecycle-management system.

```text
                     PHYSICAL WIRELESS ENVIRONMENT
                                  |
                                  v
                         Received OFDM Signal
                                  |
                                  v
                      Cyclic Prefix Removal
                           and FFT Processing
                                  |
                                  v
                       Pilot Signal Extraction
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             Raw Pilot Vector              LS Estimation
                 y_p                            |
                    |                           v
                    |                  Real-Valued Conversion
                    |                           |
                    |                           v
                    |                   Baseline FCN Estimator
                    |                           |
                    |                 +---------+---------+
                    |                 |                   |
                    v                 v                   v
              VAE Scoring       Layer Activations    Channel Estimate
                    |                 |                   |
                    |                 v                   v
                    |          Activation Drift      Equalization
                    |              Monitoring        and Decoding
                    |                 |                   |
                    +--------+--------+                   |
                             |                            v
                             v                    CRC-Verified Symbols
                     EWMA Score Smoothing                 |
                             |                            v
                             v                     Pseudo-Targets
                    Calibrated Thresholds                 |
                             |                            |
                             v                            |
                      LIFECYCLE CONTROLLER <--------------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
           CONTINUE        UPDATE          RETIRE
              |              |              |
        Keep Existing    Fine-Tune       LS Fallback
           Model         Output Layer         |
                             |                v
                             |          Collect Post-Drift
                             |           Pseudo-Channels
                             |                |
                             |                v
                             |             VQ-VAE
                             |                |
                             |                v
                             |          Synthetic Channels
                             |                |
                             |                v
                             |          Replacement FCN
                             |                |
                             v                v
                       Held-Out Validation
                             |
                             v
                    Accept or Reject Model
                             |
                             v
                   Update Digital Twin State
                             |
                             v
                      Continue Monitoring
```

The monitoring and controller components operate during deployment, while estimator fine-tuning and generative replacement are event-triggered operations.

---

## 4. Wireless Communication System Model

The simulations consider a single-input single-output (SISO) orthogonal frequency-division multiplexing (OFDM) system.

After cyclic prefix removal and frequency-domain processing, the received signal is modeled as

\[
\mathbf Y=\mathbf H\circ\mathbf X+\mathbf W,
\]

where:

- \(\mathbf Y\) is the received OFDM resource grid.
- \(\mathbf H\) is the channel frequency-response matrix.
- \(\mathbf X\) is the transmitted symbol matrix.
- \(\mathbf W\) represents additive Gaussian noise.
- \(\circ\) denotes elementwise multiplication.

Known pilot symbols are extracted from the received signal to construct least squares (LS) channel estimates.

The pilot-domain LS estimate is

\[
\widehat{\mathbf H}^{\mathrm{LS}}_p
=
\mathbf Y_p\oslash\mathbf X_p.
\]

The real and imaginary components of the LS estimates are concatenated to form the neural estimator input.

## 5. Baseline FCN Channel Estimator

The default channel estimator is a lightweight four-layer fully connected network (FCN).

### Architecture

| Layer | Input features | Output features | Activation |
|---|---:|---:|---|
| FC1 | 48 | 512 | ReLU |
| FC2 | 512 | 512 | ReLU |
| FC3 | 512 | 512 | ReLU |
| FC4 | 512 | 2016 | Linear |

**Total trainable parameters:** 1,584,608

The input represents the real-valued LS pilot estimates, while the output represents the real and imaginary components of the full channel grid.

The FCN is trained offline using EPA channel realizations.

The **Baseline FCN** refers to the unchanged offline-trained estimator. The **Proposed CDT** refers to the framework that incorporates this estimator with monitoring and lifecycle adaptation.

The CDT architecture is model-agnostic and can also support other neural estimators when their intermediate activations are accessible.

---

## 6. Self-Monitoring Module

The CDT uses two complementary reliability indicators.

### 6.1 VAE-Based Pilot Distribution Monitoring

A variational autoencoder (VAE) is trained using received pilot observations under matched reference conditions.

For each deployment slot, the VAE reconstructs the received pilot vector and calculates

\[
\epsilon_{\mathrm{VAE}}(t)
=
\left\|
\mathbf y_p^{(t)}
-
\widehat{\mathbf y}_p^{(t)}
\right\|_2^2.
\]

Here, \(\mathbf y_p^{(t)}\) is the raw received pilot vector and \(\widehat{\mathbf y}_p^{(t)}\) is its VAE reconstruction.

An elevated reconstruction error indicates reduced familiarity with the training observation distribution.

The VAE operates on received pilot observations rather than channel estimates reconstructed by the deployed FCN, avoiding circular dependence on the estimator output.

**Important:** A reconstruction-error increase indicates potential distribution mismatch. It is not a direct measurement of channel-estimation NMSE.

### 6.2 Activation Statistics Drift Detection

The second indicator monitors changes in the estimator's intermediate neural representations.

For each monitored layer, reference activation means and standard deviations are recorded under matched conditions.

The layer-wise drift score compares the current activation statistics with these references.

The overall activation score is defined as

\[
\delta(t)=\max_l\delta_l(t).
\]

The layer-wise maximum retains the strongest monitored representation deviation.

In the evaluated EPA-to-EVA and EPA-to-ETU transitions, the VAE score exhibits stronger detection sensitivity, while the activation score remains below its calibrated threshold. This limitation is explicitly acknowledged in the accompanying manuscript.

### 6.3 EWMA Smoothing

To reduce sensitivity to transient variations, both monitoring scores are smoothed using EWMA.

\[
\bar{\epsilon}_{\mathrm{VAE}}(t)
=
\alpha\epsilon_{\mathrm{VAE}}(t)
+
(1-\alpha)\bar{\epsilon}_{\mathrm{VAE}}(t-1)
\]

\[
\bar{\delta}(t)
=
\alpha\delta(t)
+
(1-\alpha)\bar{\delta}(t-1).
\]

The experiments use \(\alpha=0.05\) and a monitoring window of 100 observations.

### 6.4 SNR-Conditioned Calibration

VAE reconstruction errors depend on the observation noise as well as the channel distribution.

Therefore, monitoring thresholds are calibrated using independent matched-channel observations at the operating signal-to-noise ratio (SNR).

The calibration procedure targets a trajectory-level false-alarm probability of 1%.

The monitoring experiments presented in the manuscript use matched EPA calibration at 12 dB SNR. The available results do not establish SNR-invariant false-alarm performance across the entire evaluation range.

---

## 7. Autonomous Lifecycle Controller

The lifecycle controller uses the smoothed monitoring indicators to select an estimator-maintenance action.

| Action | Trigger | Operation |
|---|---|---|
| CONTINUE | No confirmed drift | Retain the current estimator |
| UPDATE | Persistent recoverable drift | Fine-tune the FCN output layer |
| RETIRE | Confirmed severe mismatch | Activate LS fallback and initiate replacement |

The controller uses a persistence duration of \(T_{\mathrm{hold}}=10\) slots and a severe-drift confirmation window of \(T_{\mathrm{conf}}=250\) slots.

### 7.1 CONTINUE

The deployed estimator remains unchanged while the monitoring module continues collecting observations.

### 7.2 UPDATE

When recoverable drift is confirmed, the CDT collects receiver-side pseudo-targets and performs lightweight fine-tuning.

Only the FCN output layer is updated. The remaining layers are frozen to reduce computational overhead and limit unnecessary parameter changes.

The candidate update is deployed only if its held-out pseudo-target validation loss improves.

### 7.3 RETIRE

When severe mismatch is confirmed, the CDT activates an LS fallback while collecting post-drift pseudo-channel observations.

A previously validated estimator may be retrieved from the model repository. Otherwise, the VQ-VAE generative pipeline is used to train a replacement estimator.

The candidate must pass held-out validation before deployment.

---

## 8. Receiver-Side Pseudo-Target Generation

One of the central objectives of the framework is to support estimator adaptation without measured ground-truth full-channel CSI during deployment.

After successful cyclic redundancy check (CRC) verification, decoded data symbols are re-modulated to reconstruct the transmitted symbols.

The receiver constructs a noisy channel observation as

\[
\widetilde H[k,\ell]
=
\frac{Y[k,\ell]}{\widehat X[k,\ell]}
=
H[k,\ell]+\widetilde W[k,\ell].
\]

Under correct symbol reconstruction and unit-modulus QPSK modulation, the effective noise remains zero-mean Gaussian.

The resulting pseudo-channel observations are paired with the corresponding LS pilot inputs.

A mask selects non-pilot resource elements for the adaptation loss, ensuring that the pseudo-target noise does not overlap with the pilot-domain input noise.

Under the stated correct-decoding and noise-independence assumptions, the expected masked pseudo-target loss and corresponding masked true-channel loss share the same population minimizer.

**Simulation assumption:** The reported experiments assume ideal correct decoding, corresponding to error-free CRC-verified symbol reconstruction.

---

## 9. UPDATE Pipeline

The UPDATE operation performs event-triggered online adaptation.

```text
Persistent Recoverable Drift
            |
            v
Collect CRC-Verified Observations
            |
            v
Construct LS Input / Pseudo-Target Pairs
            |
            v
Split Training and Validation Data
            |
            v
Freeze Earlier FCN Layers
            |
            v
Fine-Tune Output Layer
            |
            v
Evaluate Held-Out Pseudo-Target Loss
            |
       +----+----+
       |         |
    Improved   Not Improved
       |         |
       v         v
  Deploy      Retain Original
  Update      Estimator
       |
       v
Recalibrate Monitoring References
```

### UPDATE configuration

| Parameter | Value |
|---|---|
| Adaptation samples | 100 pseudo-target pairs |
| Training split | 80% |
| Validation split | 20% |
| Fine-tuning epochs | 30 |
| Learning rate | \(10^{-4}\) |
| Updated layers | FCN output layer |
| Acceptance criterion | Reduced held-out pseudo-target loss |

The adaptation objective combines masked pseudo-target reconstruction error with proximal regularization to limit parameter deviation from the existing estimator.

---

## 10. RETIRE and Generative Replacement Pipeline

The RETIRE procedure addresses severe channel mismatch through generative model replacement.

```text
Confirmed Severe Distribution Mismatch
                    |
                    v
              LS Fallback
                    |
                    v
       Collect CRC-Verified Observations
                    |
                    v
         Post-Drift Pseudo-Channel Set
                    |
                    v
          Check Model Repository
                    |
            +-------+-------+
            |               |
      Model Available    New Environment
            |               |
            v               v
      Load Candidate    Train EMA VQ-VAE
            |               |
            |               v
            |         Generate Synthetic
            |           Channel Samples
            |               |
            |               v
            |         Construct Synthetic
            |          Pilot/Target Pairs
            |               |
            |               v
            |         Train Replacement FCN
            |               |
            +-------+-------+
                    |
                    v
          Held-Out Validation
                    |
              +-----+-----+
              |           |
            Pass         Fail
              |           |
              v           v
       Deploy Candidate  Retain Fallback
              |
              v
        Update Repository
              |
              v
     Reinitialize Monitoring References
```

### 10.1 Post-Drift Observations

The receiver collects pseudo-channel realizations after severe drift is confirmed.

These observations form the dataset

\[
\mathcal D_{\mathrm{obs}}
=
\left\{
\widetilde{\mathbf H}^{(n)}
\right\}_{n=1}^{N_{\mathrm{obs}}}.
\]

### 10.2 Generative Channel Learning

An EMA VQ-VAE is trained on the collected pseudo-channel observations.

The model learns a discrete latent representation of the post-drift channel distribution and generates synthetic channel realizations.

### 10.3 Synthetic Training Dataset

The generated channel realizations are combined with the known pilot configuration and channel observation model to construct supervised input-target pairs.

These pairs are used to train a replacement FCN.

### 10.4 Validation-Guided Replacement

The replacement estimator is evaluated using held-out CRC-verified pseudo-target observations.

It is deployed only when its validation loss improves over that of the existing estimator.

### RETIRE configuration

| Parameter | Value |
|---|---|
| Post-drift observations | 2,000 |
| Synthetic channel realizations | 4,000 |
| Generator | EMA VQ-VAE |
| Replacement estimator | FCN |
| Temporary fallback | LS estimation |
| Acceptance criterion | Held-out pseudo-target validation |

---

## 11. Simulation Configuration

| Parameter | Value |
|---|---|
| Channel profiles | EPA, EVA, ETU |
| Doppler frequencies | 5, 70, 300 Hz |
| Carrier frequency | 2.1 GHz |
| Subcarrier spacing | 15 kHz |
| OFDM resource grid | 72 subcarriers × 14 symbols |
| Cyclic prefix | 16 samples |
| Modulation | QPSK |
| Pilot resource elements | 24 |
| Training SNR | 12 dB |
| Evaluation SNR | −5 to 30 dB |
| Offline training samples | 32,000 EPA |
| Test samples | 4,000 per SNR |
| VAE training observations | 2,000 EPA |
| Calibration trajectories | 500 EPA |
| Target false-alarm rate | 1% |

The experiments use independent simulated channel realizations for training and evaluation.

---

## 12. Benchmark Methods

The proposed CDT is evaluated against conventional and neural channel-estimation methods.

| Method | Description |
|---|---|
| LS | Least squares channel estimation |
| LMMSE | Linear minimum mean-square error estimation |
| AttRNet | Attention-based residual network |
| SCCENet | Symmetric CNN-based channel-estimation network |
| Baseline FCN | Unchanged offline-trained four-layer estimator |
| Adapted CLCE | Continual-learning-based channel estimation with ensemble expansion |
| Proposed CDT | Baseline estimator with monitoring and lifecycle management |

### Adapted CLCE Benchmark

The repository includes an adapted implementation of the continual learning-based channel estimation (CLCE) approach of Kong *et al.*

The implementation uses VAE latent representations, contrastive learning, out-of-distribution detection, ensemble expansion, and mixing-network fine-tuning.

**This is not an exact reproduction of the original CLCE method.** The implementation differs in model architecture, adaptation supervision, sample budget, and initialization strategy. In particular, it does not reproduce the original model-agnostic meta-learning initialization or additional label-pilot procedure.

These differences should be considered when interpreting the comparative results.

---

## 13. Main Numerical Results

### 13.1 Channel Estimation Under Matched EPA Conditions

Under matched EPA conditions, the Baseline FCN achieves an NMSE of −27.89 dB at 20 dB SNR, compared with −27.07 dB for AttRNet and −27.65 dB for SCCENet.

At 30 dB SNR, the Baseline FCN achieves −29.81 dB.

### 13.2 Distribution-Shift Performance

At 15 dB SNR, the proposed CDT achieves the following improvements.

| Distribution shift | Gain over Baseline FCN | Gain over Adapted CLCE |
|---|---:|---:|
| EPA to EVA | 5.97 dB | Approximately 2.7 dB |
| EPA to ETU | 4.67 dB | Approximately 1.9 dB |

These results demonstrate improved estimation performance under the considered channel-distribution shifts.

### 13.3 Monitoring Behavior

The VAE monitoring score increases after the drift onset at slot 300 under both EPA-to-EVA and EPA-to-ETU transitions at 12 dB SNR.

The activation-monitoring score remains below its calibrated threshold in these experiments, indicating limited confirmation sensitivity under the evaluated conditions.

The presented trajectories do not constitute a complete statistical evaluation of missed-detection rates, detection delays, or false-alarm rates across all SNRs.

---

## 14. Computational Performance

Computational measurements were obtained using an NVIDIA GeForce RTX 5090 Laptop GPU.

The latency benchmark uses batch size one, 100 warm-up iterations, and 500 CUDA-timed measurements.

| Method | Mean GPU latency (ms) | P95 latency (ms) |
|---|---:|---:|
| AttRNet | 1.231 | 2.376 |
| SCCENet | 4.083 | 9.480 |
| Adapted CLCE | 2.323 | 4.313 |
| Baseline FCN | 0.193 | 0.398 |
| **Proposed CDT** | **0.830** | **1.623** |

P95 denotes the 95th percentile of measured latency.

The complete CDT timing includes estimator inference, VAE scoring, activation monitoring, and lifecycle-controller processing.

The Adapted CLCE measurement includes its three-basis ensemble and mixing network but excludes separate out-of-distribution scoring. Therefore, the two measured workloads are not functionally identical.

UPDATE and RETIRE are event-triggered operations, and generative replacement training is excluded from the online inference latency measurements.

---

## 15. Software and Hardware

The implementation uses:

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- CUDA-enabled GPU acceleration
- Jupyter Notebook

**Tested GPU:** NVIDIA GeForce RTX 5090 Laptop GPU

Exact Python, PyTorch, CUDA, and dependency versions should be obtained from the execution environment and recorded in `requirements.txt` before releasing a reproducible version.

## 16. Running the Notebook

Clone the repository:

```bash
git clone https://github.com/afanali85-tech/WCL_CDT_Revision.git
cd WCL_CDT_Revision
```

For a private repository, GitHub authentication and repository access are required.

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment using:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies after the verified `requirements.txt` has been added:

```bash
pip install -r requirements.txt
```

Open the project notebook in VS Code or Jupyter and select the Python environment with the required CUDA-enabled PyTorch installation.

### Recommended execution order

1. Initialize the simulation environment and OFDM parameters.
2. Generate EPA, EVA, and ETU channel datasets.
3. Train or restore the Baseline FCN.
4. Evaluate conventional and neural estimation benchmarks.
5. Train and calibrate the VAE monitoring module.
6. Compute reference activation statistics.
7. Evaluate distribution-drift monitoring.
8. Execute the UPDATE adaptation procedure.
9. Execute RETIRE and VQ-VAE-assisted replacement.
10. Evaluate channel-estimation NMSE under distribution shifts.
11. Evaluate the Adapted CLCE benchmark.
12. Generate the final figures, CSV results, and computational latency measurements.

**Note:** Some notebook cells depend on models, variables, or checkpoints initialized in earlier cells. The notebook should be executed in the intended order, and absolute local file paths may need to be updated for a different environment.

---

## 17. Reproducibility and Limitations

The implementation is intended to support reproducible investigation of autonomous channel-estimator lifecycle management.

Several assumptions should be considered when interpreting the results.

**Receiver-side supervision:** The pseudo-target experiments assume ideal correct decoding. Practical decoding errors may affect adaptation reliability.

**Pilot observability:** Distribution changes that do not alter the pilot-position channel marginal may remain undetectable from received pilots alone.

**SNR sensitivity:** VAE reconstruction errors depend on the operating noise conditions. The monitoring experiments directly illustrated in the manuscript use 12 dB SNR.

**Monitoring performance:** The VAE and activation indicators do not exhibit identical detection sensitivity. Additional ablation and detection-delay experiments are needed for a comprehensive assessment.

**Adaptive benchmark:** Adapted CLCE is a modified implementation, and its results should not be interpreted as an exact reproduction of the published method.

**Hardware measurements:** The reported execution times characterize the specific implementations and GPU configuration used in the experiments.

---

## 18. Research Status and Citation

This repository accompanies the following manuscript:

**A. Ali, A. A. Nasir, and D. B. da Costa, “Cognitive Digital Twins for Self-Aware Channel Estimation,” submitted to IEEE Wireless Communications Letters, Manuscript ID WCL2026-2358.**


## 19. License

A software license has not yet been specified.

Before making the repository public, the authors should select an appropriate license and verify that any adapted third-party benchmark implementations and dependencies can be redistributed under its terms.

## 20. Contact

For research questions, implementation details, or collaboration inquiries:

**Afan Ali**  
Interdisciplinary Research Center for Communication Systems and Sensing (IRC-CSS)  
King Fahd University of Petroleum and Minerals (KFUPM)  
Saudi Arabia

**Email:** afan.ali@kfupm.edu.sa

---

*This repository is maintained for research, reproducibility, and academic collaboration in AI-native wireless communications, cognitive digital twins, and autonomous channel-estimator lifecycle management.*
