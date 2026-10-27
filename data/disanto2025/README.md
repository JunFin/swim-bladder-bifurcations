# Dataset for: Inherent instability leads to high costs of hovering in near-neutrally buoyant fishes

Dataset DOI: [10.5061/dryad.9cnp5hqw7](10.5061/dryad.9cnp5hqw7)

## Description of the data and file structure

This dataset supports the manuscript "Inherent instability leads to high costs of hovering in near-neutrally buoyant fishes," by Di Santo et al., which investigates the energetic and biomechanical costs of hovering across 13 species of near-neutrally buoyant fishes.

Experimental Overview: Data were collected using a combination of respirometry, high-speed video, and morphometric analysis. Fish were placed in flow tanks where their oxygen consumption was measured during rest and while hovering. Simultaneously, body posture and fin movements were recorded and analyzed with high-speed cameras. Morphological measurements were used to estimate the spatial relationship between the center of mass (COM) and the center of buoyancy (COB), as well as to calculate fin positions and body shape parameters.

### Files and variables

\- MO2 hovering.txt: Individual-level oxygen consumption measurements (resting, hovering, net metabolic rate), and MO₂ hover/rest ratios.

\- hovering morphology.txt: Morphological data including COM–COB separation, body shape ratios, mass, and size metrics.

\- hovering body posture.txt: Body angle, roll, pitch, and curvature metrics during hovering trials.

\- hovering fin position.txt: Normalized pectoral and caudal fin positions along the body axis.

\- hovering kinematics.txt: distance traveled by fins during hovering.

**Column headers explanation:** 

### **MO2_hovering.txt:**

* species: Species' scientific name.
* common name: Common name of the species.
* Fish ID: Identifier for each individual fish tested.
* Mass (Kg): Mass of the fish in kilograms.
* MO2 value: Oxygen consumption (MO₂) rate measured in mg O₂/kg/h.
* MO2 type: Type of oxygen consumption measurement (hovering, resting, net [=hovering-rest], or hovering/resting ratio).

### **hovering_morphology.txt:**

* Species: Species' scientific name.
* COM-COB ant/post: Anterior/posterior distance (negative values indicate COM anterior to COB; positive values indicate COM posterior to COB). Values as proportion of body length.
* COM-COB dorsal/ventral: Dorsal/ventral distance (negative values indicate COM dorsal to COB; positive values indicate COM ventral to COB). Values as proportion of body depth.
* TL-Width Ratio: Ratio of total length to body width.
* fineness ratio (tl-width): Hydrodynamic shape ratio calculated as total length divided by body width.
* TL: Total length of the fish (cm).
* Width: Maximum width of the fish (cm).
* depth: Maximum depth (height) of the fish (cm).
* fineness ratio depth: Hydrodynamic shape ratio calculated as total length divided by body depth.
* mass: Body mass of the fish (g).
* Kj per 10 min: Energy expenditure in kilojoules per 10 minutes during hovering.

### **hovering_body_posture.txt:**

* Species: Species' scientific name.
* Fish: Individual fish identifier.
* BODY angle: Body angle relative to horizontal (degrees).
* Roll angle: Degree of lateral rotation about the body axis (degrees).
* Pitch angle: Degree of vertical rotation about the lateral axis (degrees).
* curvature: Degree of curvature of the fish’s body during hovering.

### **hovering_fin_position.txt:**

* common name: Common name of the species.
* Species: Species' scientific name.
* pectoral_start /BL: Starting position of the pectoral fin normalized by body length (BL).
* caudal_start /BL: Starting position of the caudal fin normalized by body length (BL).

### **hovering_kinematics.txt:**

* Species: Species' scientific name.
* Filename: Original data file for high-speed video tracking.
* Number of frames: Total number of video frames analyzed.
* TL: Total length of fish (cm).
* Pectoral fin base: Distance traveled by the base of pectoral fin (cm).
* Pectoral fin tip: Distance traveled by the tip of pectoral fin (cm).
* Pectoral fin tip offset: Lateral offset movement of pectoral fin tip (cm).
* Snout, Eye, Dorsal, Anal, Pelvic L, Pelvic R, Peduncle, Caudal: Distance traveled by respective anatomical landmarks (cm).
* Caudal offset: Lateral offset movement of caudal fin (cm).
* Lat: Side (L or R) predominantly active or tracked.
* Bin: Binary indicator used for data categorization (e.g., 0 or 1).
* Blank cell: N/A

## Code/software

n/a
