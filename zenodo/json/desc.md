<h1>ASTRA-DESI Data Release 1 (DR1)</h1>
<p>This record bundles the ASTRA-DESI DR1 products for the two DESI survey regions (NGC and SGC). Every catalogue is delivered as a compressed FITS binary table (<code>.fits.gz</code>) whose first extension (HDU 1) contains the data. File headers store the keywords <code>RELEASE=DR1</code> and the corresponding <code>ZONE</code> label for quick checks.</p>
<table>
<tbody>
<tr>
<th>Tracer</th>
<th>Number of objects</th>
</tr>
<tr>
<td>BGS ANY</td>
<td>5,522,353</td>
</tr>
<tr>
<td>BGS BRIGHT</td>
<td>3,957,865</td>
</tr>
<tr>
<td>LRG</td>
<td>2,138,627</td>
</tr>
<tr>
<td>ELG</td>
<td>2,432,072</td>
</tr>
<tr>
<td>QSO</td>
<td>1,223,401</td>
</tr>
</tbody>
</table>
<h2>Directory layout</h2>
<p>The archive preserves the NERSC production tree. Top-level folders contain:</p>
<ul>
<li><strong>raw/</strong> &ndash; merged real and random catalogues per region.</li>
<li><strong>classification/</strong> &ndash; neighbour count summaries per tracer and random iteration.</li>
<li><strong>probabilities/</strong> &ndash; web-type membership probabilities derived from the classification counts.</li>
</ul>
<h2>Naming conventions</h2>
<ul>
<li><code>raw/zone_REGION_TRACER.fits.gz</code> &ndash; Region NGC or SGC.</li>
<li><code>classification/TRACER/REGION/zone_REGION_TRACER_classified_iterXXX.fits.gz</code> &ndash; combined tracer summary for the region.</li>
<li><code>probabilities/TRACER/REGION/TRACER_REGION_TRACER_probability_iterdata.fits.gz</code> &ndash; per-object web-type probabilities.</li>
<li><code>properties/zone_REGION_properties.fits.gz</code> &ndash; per-object properties.</li>
</ul>
<h2>Column reference</h2>
<h3>Raw catalogues (<code>raw/*.fits.gz</code>)</h3>
<table>
<tbody>
<tr>
<th>Column</th>
<th>Type</th>
<th>Description</th>
</tr>
<tr>
<td>TARGETID</td>
<td>int64</td>
<td>DESI target identifier.</td>
</tr>
<tr>
<td>RA</td>
<td>float64</td>
<td>Right ascension in degrees.</td>
</tr>
<tr>
<td>DEC</td>
<td>float64</td>
<td>Declination in degrees.</td>
</tr>
<tr>
<td>Z</td>
<td>float64</td>
<td>Spectroscopic redshift.</td>
</tr>
<tr>
<td>XCART, YCART, ZCART</td>
<td>float64</td>
<td>Comoving Cartesian coordinates (Mpc) computed with the Planck18 cosmology.</td>
</tr>
<tr>
<td>TRACERTYPE</td>
<td>string</td>
<td>Tracer label with suffix <code>_DATA</code> or <code>_RAND</code> (BGS_BRIGHT, ELG, LRG, QSO).</td>
</tr>
<tr>
<td>RANDITER</td>
<td>int32</td>
<td>-1 for real objects, otherwise the random iteration index (0&ndash;99).</td>
</tr>
</tbody>
</table>
<h3>Properties catalogues (<code>properties/*.fits.gz</code>)</h3>
<table>
<tbody>
<tr>
<th>Column</th>
<th>Type</th>
<th>Description</th>
</tr>
<tr>
<td>TARGETID</td>
<td>int64</td>
<td>DESI target identifier.</td>
</tr>
<tr>
<td>SED_SFR</td>
<td>float64</td>
<td>Star formation rate derived from SED fitting.</td>
</tr>
<tr>
<td>SED_MASS</td>
<td>float64</td>
<td>Stellar mass derived from SED fitting.</td>
</tr>
<tr>
<td>FLUX_G</td>
<td>float64</td>
<td>Observed flux in the g photometric band.</td>
</tr>
<tr>
<td>FLUX_R</td>
<td>float64</td>
<td>Observed flux in the r photometric band.</td>
</tr>
</tbody>
</table>
<p>Each raw file merges all tracers for its zone. Random catalogues mirror the data counts for every tracer and random iteration.&nbsp;</p>
<blockquote>
<p><em>Additional columns SED_SFR, SED_MASS, FLUX_G, and FLUX_R were incorporated from the DESI DR1 value-added catalogues associated with the emission-line sample: <a href="https://data.desi.lbl.gov/doc/releases/dr1/vac/stellar-mass-emline/">https://data.desi.lbl.gov/doc/releases/dr1/vac/stellar-mass-emline/</a>. These quantities are derived via SED fitting using CIGALE (<a href="https://iopscience.iop.org/article/10.3847/1538-4357/ad1409">Zou et al. 2024</a>).</em></p>
</blockquote>
<h3>Classification catalogues (<code>classification/*.fits.gz</code>)</h3>
<table>
<tbody>
<tr>
<th>Column</th>
<th>Type</th>
<th>Description</th>
</tr>
</tbody>
<tbody>
<tr>
<td>TARGETID</td>
<td>int64</td>
<td>Identifier of the central object.</td>
</tr>
<tr>
<td>RANDITER</td>
<td>int32</td>
<td>-1 for data rows, otherwise the random catalogue index.</td>
</tr>
<tr>
<td>ISDATA</td>
<td>bool</td>
<td><code>True</code> for data rows; <code>False</code> for random iterations.</td>
</tr>
<tr>
<td>NDATA</td>
<td>int32</td>
<td>Number of neighbour pairs built from data tracers for this target.</td>
</tr>
<tr>
<td>NRAND</td>
<td>int32</td>
<td>Number of neighbour pairs contributed by random tracers.</td>
</tr>
<tr>
<td>TRACERTYPE</td>
<td>string</td>
<td>Tracer (BGS_ANY, BGS_BRIGHT, ELG, LRG, QSO).</td>
</tr>
</tbody>
</table>
<h3>Probability catalogues (<code>probabilities/*.fits.gz</code>)</h3>
<table>
<tbody>
<tr>
<th>Column</th>
<th>Type</th>
<th>Description</th>
</tr>
</tbody>
<tbody>
<tr>
<td>TARGETID</td>
<td>int64</td>
<td>Identifier of the object.</td>
</tr>
<tr>
<td>TRACERTYPE</td>
<td>string</td>
<td>Tracer family.</td>
</tr>
<tr>
<td>PVOID</td>
<td>float32</td>
<td>Probability of the object being a void class.</td>
</tr>
<tr>
<td>PSHEET</td>
<td>float32</td>
<td>Probability of the sheet class.</td>
</tr>
<tr>
<td>PFILAMENT</td>
<td>float32</td>
<td>Probability of the filament class.</td>
</tr>
<tr>
<td>PKNOT</td>
<td>float32</td>
<td>Probability of the knot class.</td>
</tr>
</tbody>
</table>
<h2>Using the catalogues</h2>
<ul>
<li>All FITS tables are compressed with <code>gzip</code>.</li>
<li>Random catalogues use <code>RANDITER</code> values 0&ndash;99; filtering on <code>RANDITER &gt;= 0</code> isolates the random iterations.</li>
<li>The Cartesian coordinates assume the Planck18 cosmology provided by <code>astropy.cosmology.Planck18</code>.</li>
<li>
<p>The masks are distributed as a compressed archive (<code>masks.tar.gz</code>) containing FITS maps with the following naming convention:</p>
<ul>
<li><code>dr1_mask_{program}_nside{nside}_final.fits</code> &ndash; binary mask defining the valid survey footprint for each program (<strong>bright</strong> or <strong>dark</strong>). Pixels with value <code>1</code> belong to the footprint, while <code>0</code> indicates excluded regions.</li>
<li><code>dr1_mask_{program}_nside{nside}_ngc.fits</code> &ndash; mask restricted to the North Galactic Cap (NGC).</li>
<li><code>dr1_mask_{program}_nside{nside}_sgc.fits</code> &ndash; mask restricted to the South Galactic Cap (SGC).</li>
</ul>
<p>Here, <code>{program}</code> corresponds to the DESI observing program (<code>bright</code> for BGS tracers and <code>dark</code> for LRG, ELG, and QSO), and <code>{nside}</code> indicates the HEALPix resolution parameter.</p>
<p>For reproducibility, additional files are provided:</p>
<ul>
<li><code>dr1_mask_{program}_nside{nside}_maps.npz</code> &ndash; compressed NumPy archive containing intermediate maps used in the mask construction, including tile coverage, LSS object counts, and smoothed mask representations.</li>
</ul>
</li>
</ul>