"""Checks of the star rows added to catalogues, their rendering and their saturated cores."""

import numpy as np
import pandas as pd

from mock_lsst_image_generation import IMAGE_CONFIG, STAR_CONFIG
from mock_lsst_image_generation.coadd_synthesis import MASK_PLANES, single_coadd, synthesise_coadd
from mock_lsst_image_generation.stars import add_stars, draw_stars, render_stars, spike_angles

MANY_BRIGHT = {**STAR_CONFIG, "bright_per_deg2": 5000}


def test_add_stars_appends_rows_and_keeps_galaxies():
    galaxies = pd.DataFrame(dict(id=[0, 1, 2], type="passive", x_pix=[10.0, 20.0, 30.0], y_pix=[5.0, 6.0, 7.0]))
    table = add_stars(galaxies, dict(npix=2000, pixscale=0.2), np.random.default_rng(0))
    stars = table.loc[table["type"] == "star"]
    assert table.iloc[:3][galaxies.columns].equals(galaxies)
    assert len(stars) > 50 and stars["id"].tolist() == list(range(3, 3 + len(stars)))
    assert stars["x_pix"].between(0, 2000).all() and stars["mag_r_total"].between(9, 23).all()
    for band in "ugrizy":
        assert np.isfinite(stars[f"mag_{band}_total"]).all() and (stars[f"flux_{band}_total"] > 0).all()


def test_bright_stars_are_rarely_red():
    stars = draw_stars(5000, 0.2, np.random.default_rng(1), MANY_BRIGHT)
    g_r = stars["mag_g_total"] - stars["mag_r_total"]
    bright, faint = stars["mag_r_total"] < 14, stars["mag_r_total"] > 20
    assert (g_r[bright] > 1.1).mean() < 0.1 < (g_r[faint] > 1.1).mean()


def test_spike_angles_share_light_per_band():
    angles = spike_angles(np.random.default_rng(2), "gri")
    for lines in angles.values():
        assert len(lines) % 2 == 0 and np.isclose(sum(w for _, w in lines), 1.0)


def test_bright_star_gets_striped_saturated_core():
    star = draw_stars(2000, 0.2, np.random.default_rng(3), MANY_BRIGHT).nsmallest(1, "mag_r_total")
    star = star.assign(x_pix=100.0, y_pix=100.0)
    assert star["mag_r_total"].iloc[0] < 12
    image = render_stars(star, (6, 201, 201), (0, 0), {b: IMAGE_CONFIG["base_fwhm"] for b in "ugrizy"}, IMAGE_CONFIG,
                         np.random.default_rng(4))
    assert image[2].sum() > 0.9 * star["flux_r_total"].iloc[0]  # most of the light lands in the stamp
    spikes = np.zeros_like(image)
    again = render_stars(star, (6, 201, 201), (0, 0), {b: IMAGE_CONFIG["base_fwhm"] for b in "ugrizy"}, IMAGE_CONFIG,
                         np.random.default_rng(4), spikes_out=spikes)
    assert np.array_equal(again, image)  # collecting the spikes does not change the render
    assert 0 < spikes[2].sum() < 0.05 * image[2].sum() and (spikes >= 0).all()
    coadd = single_coadd(0, IMAGE_CONFIG, epoch="10y")["10y_nominal"]
    signal, _ = synthesise_coadd(image, coadd, IMAGE_CONFIG, star, (0, 0))
    core = signal[2, 95:106, 95:106]
    assert all(np.ptp(row) < 1e-3 * row.mean() for row in core[3:8, 3:8])  # rows inside the core are flat ...
    assert np.ptp(core[3:8, 5]) > 0  # ... but differ from row to row (the stripes)

    masked_signal, _, mask = synthesise_coadd(image, coadd, IMAGE_CONFIG, star, (0, 0), return_mask=True)
    assert np.array_equal(masked_signal, signal)  # asking for the mask does not change the coadd
    saturated = (mask & (1 << MASK_PLANES["SAT"])) > 0
    assert saturated[2, 98:103, 98:103].all()  # the core is SAT ...
    assert not saturated[:, :, :60].any()  # ... and nothing far from it
    replaced = ~np.isclose(signal, synthesise_coadd(image, coadd, IMAGE_CONFIG)[0])  # cores changed by saturation
    assert not (replaced & ~saturated).any()  # every pixel the cores changed is masked
